import json
import logging
import os
import sqlite3
import struct
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite_vec

from mnemomatic import model_config
from mnemomatic.models import Document, Knowledge, Note, SearchResult

logger = logging.getLogger("mnemomatic")

# Defaults to the bundled model's dimension (model_config.json, written by the
# Docker build for the EMBED_MODEL chosen), falling back to 384 (MiniLM-class)
# when no config exists. Changing dimension requires MNEMOMATIC_REINDEX=1 once
# to rebuild the index — the server fails fast on the mismatch otherwise.
EMBEDDING_DIM = int(os.environ.get("MNEMOMATIC_EMBED_DIM", model_config.CONFIG.get("dim", 384)))
BUSY_TIMEOUT_MS = 5000

# Bumped whenever the on-disk schema changes shape. Stored in PRAGMA
# user_version; Database._init_schema migrates older databases forward.
# Version 1: vec0 tables gained a `namespace` partition key so namespace-
# filtered KNN happens inside the index instead of post-filtering in Python.
# Version 2: usage-tracking columns (retrieval_count, last_accessed) on the
# content tables, plus the `revisions` table capturing prior state on
# update/delete so content can be inspected and restored.
# Version 3: temporal facts — knowledge rows gain valid_until/superseded_by;
# changing a fact closes the old row and inserts a successor instead of
# overwriting, and the (namespace, subject) unique index becomes partial so
# it only constrains current rows.
# Version 4: the append-only audit_log table — one row per write operation
# (event trail for accountability, complementing revisions' content capture).
SCHEMA_VERSION = 4
CHUNK_THRESHOLD = int(os.environ.get("MNEMOMATIC_CHUNK_THRESHOLD", "2000"))
CHUNK_SIZE = int(os.environ.get("MNEMOMATIC_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("MNEMOMATIC_CHUNK_OVERLAP", "200"))

# Revisions retained per item; the oldest beyond this are pruned as new ones
# are captured. 0 disables revision capture entirely.
REVISIONS_KEEP = int(os.environ.get("MNEMOMATIC_REVISIONS_KEEP", "10"))

# Audit events older than this are pruned as new ones are appended.
# 0 keeps the trail forever.
AUDIT_KEEP_DAYS = int(os.environ.get("MNEMOMATIC_AUDIT_KEEP_DAYS", "730"))


@dataclass(frozen=True)
class _TableSpec:
    """Everything that varies between the three content tables.

    One record per table, so adding a content type means adding one entry here
    rather than editing a dozen parallel dicts and hoping none was missed.
    """

    table: str                          # SQL table name, e.g. "documents"
    item_type: str                      # singular name used by the tools, e.g. "document"
    model: type                         # the pydantic model rows deserialize into
    alias: str                          # short alias used in multi-table SQL
    columns: tuple[str, ...]            # full column list, mirroring the model's fields
    update_fields: frozenset[str]       # what update_* is allowed to change
    title_field: str                    # the human-facing label ("title"/"subject")
    snippet_field: str                  # the body field search snippets come from
    snippet_len: int | None             # snippet truncation; None means "whole field"
    summary_columns: tuple[str, ...]    # list_page's columns — never the body, so pages stay small
    resource_uri: str                   # MCP resource template, formatted with the row id
    current_only: str = ""              # extra WHERE fragment hiding non-current rows

    @property
    def resource_fields(self) -> tuple[str, ...]:
        """What the per-namespace list resources project.

        The same shape as `summary_columns` minus the usage counters, which
        those resources have never exposed — derived rather than repeated so
        the two cannot drift apart.
        """
        return tuple(c for c in self.summary_columns if c not in _USAGE_COLUMNS)


# Bookkeeping columns present on every content table, tracked separately from
# the content itself.
_USAGE_COLUMNS = ("retrieval_count", "last_accessed")


# The three content tables, in display order. Each has a parallel vec_<table>.
# Only knowledge sets current_only: superseded facts stay in the table (that is
# the point of temporal knowledge) but are excluded from search, listings,
# counts, and upsert lookups.
_SPECS: dict[str, _TableSpec] = {
    "documents": _TableSpec(
        table="documents", item_type="document", model=Document, alias="d",
        columns=("id", "namespace", "title", "content", "mime_type",
                 "tags", "metadata", "created_at", "updated_at"),
        update_fields=frozenset({"title", "content", "mime_type", "tags", "metadata"}),
        title_field="title", snippet_field="content", snippet_len=200,
        summary_columns=("id", "title", "mime_type", "tags", "updated_at",
                         "retrieval_count", "last_accessed"),
        resource_uri="mnemomatic://document/{id}",
    ),
    "knowledge": _TableSpec(
        table="knowledge", item_type="knowledge", model=Knowledge, alias="k",
        columns=("id", "namespace", "subject", "fact", "confidence", "source",
                 "tags", "metadata", "created_at", "updated_at",
                 "valid_until", "superseded_by"),
        update_fields=frozenset({"subject", "fact", "confidence", "source", "tags", "metadata"}),
        title_field="subject", snippet_field="fact", snippet_len=None,
        summary_columns=("id", "subject", "fact", "confidence", "tags", "updated_at",
                         "retrieval_count", "last_accessed"),
        resource_uri="mnemomatic://knowledge-entry/{id}",
        current_only=" AND {alias}valid_until IS NULL",
    ),
    "notes": _TableSpec(
        table="notes", item_type="note", model=Note, alias="n",
        columns=("id", "namespace", "title", "content", "source",
                 "tags", "metadata", "created_at", "updated_at"),
        update_fields=frozenset({"title", "content", "source", "tags", "metadata"}),
        title_field="title", snippet_field="content", snippet_len=200,
        summary_columns=("id", "title", "source", "tags", "updated_at",
                         "retrieval_count", "last_accessed"),
        resource_uri="mnemomatic://note/{id}",
    ),
}

_TABLES = tuple(_SPECS)
# Same specs keyed by the singular item_type the tools speak.
_SPEC_BY_ITEM_TYPE: dict[str, _TableSpec] = {s.item_type: s for s in _SPECS.values()}


def _current_filter(table: str, alias: str = "") -> str:
    return _SPECS[table].current_only.format(alias=f"{alias}." if alias else "")


def _item_filters(tags: list[str] | None, updated_after: str | None,
                  alias: str = "") -> tuple[str, list]:
    """WHERE fragments (and their params) for the optional search filters.

    Tags AND together — an item must carry every requested tag (json_each
    over the stored JSON array). updated_after compares ISO-8601 strings,
    which orders correctly against the stored isoformat timestamps.
    """
    prefix = f"{alias}." if alias else ""
    sql, params = "", []
    for tag in tags or ():
        sql += f" AND EXISTS (SELECT 1 FROM json_each({prefix}tags) WHERE json_each.value = ?)"
        params.append(tag)
    if updated_after:
        sql += f" AND {prefix}updated_at >= ?"
        params.append(updated_after)
    return sql, params


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking at paragraph/sentence boundaries when possible.

    Break points are only accepted past ``start + overlap`` so that the next
    window start (``end - overlap``) always moves forward. Accepting an earlier
    break used to let ``start`` stall or move backward, looping forever and
    appending chunks until memory ran out.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            para_break = text.rfind("\n\n", start + overlap, end)
            if para_break > start:
                end = para_break + 2
            else:
                for sep in (". ", "! ", "? ", "\n"):
                    sent_break = text.rfind(sep, start + overlap, end)
                    if sent_break > start:
                        end = sent_break + len(sep)
                        break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # max() guards forward progress even if overlap >= chunk_size.
        start = max(end - overlap, start + 1)
    return chunks


def _row_to_search_result(table: str, row, score: float) -> SearchResult:
    title_field = _SPECS[table].title_field
    snippet_field = _SPECS[table].snippet_field
    snippet = row[snippet_field]
    max_len = _SPECS[table].snippet_len
    if max_len:
        snippet = snippet[:max_len]
    return SearchResult(
        id=row["id"],
        type=_SPECS[table].item_type,
        namespace=row["namespace"],
        title=row[title_field],
        snippet=snippet,
        resource_uri=_SPECS[table].resource_uri.format(id=row["id"]),
        score=score,
        tags=_safe_json_loads(row["tags"], [], f"tags row {row['id']}"),
    )


def _serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _l2_to_cosine(distance: float) -> float:
    """Convert an L2 distance to cosine similarity for normalized embeddings.

    cosine_sim = 1 - (L2^2 / 2), which ranges -1..1; clamp to 0..1.
    """
    return max(0.0, 1.0 - (distance * distance / 2.0))


def _safe_json_loads(s: str, default, context: str = ""):
    """Parse JSON, logging a warning and returning default on corruption.

    Args:
        s: JSON string to parse
        default: Value to return if parsing fails
        context: Optional context for the warning message

    Returns:
        Parsed JSON or default value if parsing fails
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Corrupted JSON field%s — returning default. Error: %s",
                       f" ({context})" if context else "", e)
        return default


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def _row_to_model(model_cls, row: dict):
    """Hydrate a content-table row into its pydantic model.

    Ignores extra row keys (e.g. rowid from RETURNING clauses); parses the
    JSON and timestamp columns that SQLite stores as text.
    """
    data = {k: v for k, v in row.items() if k in model_cls.model_fields}
    data["tags"] = _safe_json_loads(row["tags"], [], f"tags row {row['id']}")
    data["metadata"] = _safe_json_loads(row["metadata"], {}, f"metadata row {row['id']}")
    data["created_at"] = datetime.fromisoformat(row["created_at"])
    data["updated_at"] = datetime.fromisoformat(row["updated_at"])
    return model_cls(**data)


def _item_column_values(item, columns) -> list:
    """The model's values for `columns`, serialized the way the tables store them."""
    values = []
    for col in columns:
        v = getattr(item, col)
        if col in ("tags", "metadata"):
            v = json.dumps(v)
        elif isinstance(v, datetime):
            v = v.isoformat()
        values.append(v)
    return values


class Database:
    # schema_meta keys recording which embedder built the vector index. The
    # dimension lives in `embed_dim` instead, because it is structural — the
    # vec0 tables are declared with it — while these only describe identity.
    _IDENTITY_KEYS = ("embed_model", "embed_query_prefix", "embed_doc_prefix")

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        allow_reindex: bool = False,
        embed_identity: dict[str, str] | None = None,
    ):
        """Args:
            db_path: SQLite file path, or ":memory:".
            allow_reindex: when True, a change that invalidates the stored
                vectors — EMBEDDING_DIM, or the embedding identity below —
                does not fail startup; instead `reindex_pending` is set and the
                caller is expected to run rebuild_vec_tables() + re-embed (the
                MNEMOMATIC_REINDEX flow).
            embed_identity: the current embedder's identity, keyed by
                _IDENTITY_KEYS. Injected by the caller because the embedding
                configuration lives in the server layer. Omitted (or without an
                `embed_model`) means "unknown", which disables both the identity
                check and its recording — an FTS-only run must neither fail
                against a fingerprinted database nor overwrite its fingerprint.
        """
        self.db_path = str(db_path)
        self.allow_reindex = allow_reindex
        self.reindex_pending = False
        self.embed_identity = embed_identity or {}
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = _dict_factory
            conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._local.conn = conn
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT 'text/markdown',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                subject TEXT NOT NULL,
                fact TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'unknown',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                valid_until TEXT,
                superseded_by TEXT
            );

            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'text',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace);
            CREATE INDEX IF NOT EXISTS idx_knowledge_namespace ON knowledge(namespace);
            CREATE INDEX IF NOT EXISTS idx_notes_namespace ON notes(namespace);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_ns_title ON documents(namespace, title);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_ns_title ON notes(namespace, title);
        """)
        # The knowledge uniqueness index is created by _migrate_to_v3 (it is
        # partial — current rows only — and pre-v3 databases carry a full one
        # that must be dropped first, which can't be expressed as IF NOT EXISTS).

        # FTS5 tables
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, content, content=documents, content_rowid=rowid
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                subject, fact, content=knowledge, content_rowid=rowid
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                title, content, content=notes, content_rowid=rowid
            );
        """)

        # FTS sync triggers
        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
                INSERT INTO documents_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, subject, fact)
                VALUES (new.rowid, new.subject, new.fact);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, subject, fact)
                VALUES ('delete', old.rowid, old.subject, old.fact);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, subject, fact)
                VALUES ('delete', old.rowid, old.subject, old.fact);
                INSERT INTO knowledge_fts(rowid, subject, fact)
                VALUES (new.rowid, new.subject, new.fact);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
                INSERT INTO notes_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;
        """)

        # Chunks table for large documents
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()

        # Vector tables: created fresh with a namespace partition key, or
        # migrated in place when the database predates SCHEMA_VERSION 1.
        self._ensure_vec_schema(conn)

        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS document_chunks_ad AFTER DELETE ON document_chunks BEGIN
                DELETE FROM vec_document_chunks WHERE rowid = old.id;
            END;
        """)

        conn.commit()

    # ── Vec schema + migration ──

    # (vec table, parent table whose namespace partitions it, SQL that yields
    # rowid/namespace/embedding for every valid legacy row — orphans dropped)
    _VEC_MIGRATION_SOURCES = {
        "vec_documents": "SELECT v.rowid AS rowid, t.namespace AS namespace, v.embedding AS embedding "
                         "FROM vec_documents v JOIN documents t ON t.rowid = v.rowid",
        "vec_knowledge": "SELECT v.rowid AS rowid, t.namespace AS namespace, v.embedding AS embedding "
                         "FROM vec_knowledge v JOIN knowledge t ON t.rowid = v.rowid",
        "vec_notes": "SELECT v.rowid AS rowid, t.namespace AS namespace, v.embedding AS embedding "
                     "FROM vec_notes v JOIN notes t ON t.rowid = v.rowid",
        "vec_document_chunks": "SELECT v.rowid AS rowid, t.namespace AS namespace, v.embedding AS embedding "
                               "FROM vec_document_chunks v "
                               "JOIN document_chunks dc ON dc.id = v.rowid "
                               "JOIN documents t ON t.id = dc.document_id",
    }

    @staticmethod
    def _create_vec_table(conn: sqlite3.Connection, name: str) -> None:
        conn.execute(f"""
            CREATE VIRTUAL TABLE {name}
            USING vec0(namespace TEXT partition key, embedding float[{EMBEDDING_DIM}])
        """)

    def _identity_known(self) -> bool:
        """Whether the caller told us which embedder is configured."""
        return bool(self.embed_identity.get("embed_model"))

    def _stamp_identity(self, conn: sqlite3.Connection) -> None:
        """Record the configured embedding identity. No-op when unknown, so an
        FTS-only run never erases the fingerprint of a real embedder."""
        if not self._identity_known():
            return
        for key in self._IDENTITY_KEYS:
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                (key, self.embed_identity.get(key) or ""),
            )

    def _check_identity(self, conn: sqlite3.Connection) -> None:
        """Fail fast when a different embedder is configured than the one that
        built the index.

        The dimension check alone cannot catch this: models of equal dimension
        swap silently, and the result is not a broken index so much as a subtly
        wrong one — queries embedded by one model, searched against another
        model's vectors, return plausible but degraded results with no error.

        A database written before identities were recorded has nothing to
        compare against, so it adopts whatever is configured now. That cannot
        detect a swap that already happened, hence the warning.
        """
        if not self._identity_known():
            return
        placeholders = ",".join("?" * len(self._IDENTITY_KEYS))
        stored = {
            row["key"]: row["value"]
            for row in conn.execute(
                f"SELECT key, value FROM schema_meta WHERE key IN ({placeholders})",
                self._IDENTITY_KEYS,
            ).fetchall()
        }

        if "embed_model" not in stored:
            logger.warning(
                "Recording embedding identity (model %r) for a database that predates this "
                "check — model and prefix changes will be caught from now on. If the model "
                "was already changed without a reindex, run MNEMOMATIC_REINDEX=1 once.",
                self.embed_identity.get("embed_model"),
            )
            self._stamp_identity(conn)
            conn.commit()
            return

        changed = [
            (key, stored.get(key, ""), self.embed_identity.get(key) or "")
            for key in self._IDENTITY_KEYS
            if stored.get(key, "") != (self.embed_identity.get(key) or "")
        ]
        if not changed:
            return

        detail = "; ".join(
            f"{key.removeprefix('embed_')} was {old!r}, now {new!r}" for key, old, new in changed
        )
        if self.allow_reindex:
            logger.warning(
                "Embedding identity changing (%s) — vec tables will be rebuilt and all "
                "content re-embedded", detail,
            )
            self.reindex_pending = True
            return
        raise RuntimeError(
            f"Embedding identity mismatch: {detail}. The stored vectors were produced with a "
            f"different embedding configuration, so searching against them returns wrong "
            f"results with no error to notice. Restore the previous settings to keep the "
            f"existing index, or set MNEMOMATIC_REINDEX=1 to rebuild the index and re-embed "
            f"all content on startup."
        )

    def _ensure_vec_schema(self, conn: sqlite3.Connection) -> None:
        """Create or migrate the vec0 tables, then record schema version, dim,
        and the identity of the embedder that built the index.

        Fails fast when MNEMOMATIC_EMBED_DIM disagrees with the dimension the
        database was created with — a mismatch would otherwise surface later as
        confusing insert/search errors against half-usable vector tables.
        """
        version = conn.execute("PRAGMA user_version").fetchone()["user_version"]

        if version >= 1:
            stored = conn.execute("SELECT value FROM schema_meta WHERE key = 'embed_dim'").fetchone()
            if stored and int(stored["value"]) != EMBEDDING_DIM:
                if self.allow_reindex:
                    logger.warning(
                        "Embedding dimension changing from %s to %d — vec tables will be "
                        "rebuilt and all content re-embedded", stored["value"], EMBEDDING_DIM,
                    )
                    self.reindex_pending = True
                else:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: database was created with dim {stored['value']} "
                        f"but MNEMOMATIC_EMBED_DIM={EMBEDDING_DIM}. Set MNEMOMATIC_EMBED_DIM={stored['value']} "
                        f"to keep the existing index, or set MNEMOMATIC_REINDEX=1 to rebuild the index and "
                        f"re-embed all content at the new dimension on startup."
                    )
            self._check_identity(conn)
            if version < SCHEMA_VERSION:
                self._migrate_content_schema(conn)
                # A pending rebuild leaves the version to rebuild_vec_tables,
                # which stamps it once the vec tables actually match.
                if not self.reindex_pending:
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    conn.commit()
            return

        # Version 0: either a fresh database or one from before schema
        # versioning whose vec tables lack the partition key.
        legacy = [
            name for name in self._VEC_MIGRATION_SOURCES
            if (row := conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
            ).fetchone()) and "partition key" not in row["sql"].lower()
        ]

        conn.execute("BEGIN")
        try:
            for name, source_sql in self._VEC_MIGRATION_SOURCES.items():
                rows = []
                if name in legacy:
                    rows = conn.execute(source_sql).fetchall()
                    if rows:
                        found_dim = len(rows[0]["embedding"]) // 4
                        if found_dim != EMBEDDING_DIM:
                            if self.allow_reindex:
                                # No point copying wrong-dim embeddings; the
                                # reindex rebuild will replace these tables.
                                conn.rollback()
                                self.reindex_pending = True
                                logger.warning(
                                    "Legacy embeddings have dim %d, configured %d — deferring "
                                    "to reindex rebuild", found_dim, EMBEDDING_DIM,
                                )
                                self._migrate_content_schema(conn)
                                conn.commit()
                                return
                            raise RuntimeError(
                                f"Cannot migrate {name}: stored embeddings have dim {found_dim} "
                                f"but MNEMOMATIC_EMBED_DIM={EMBEDDING_DIM}. "
                                f"Set MNEMOMATIC_EMBED_DIM={found_dim} and retry, or set "
                                f"MNEMOMATIC_REINDEX=1 to re-embed everything at the new dimension."
                            )
                    conn.execute(f"DROP TABLE {name}")
                    logger.info("Migrating %s to partitioned schema (%d embeddings)", name, len(rows))
                self._create_vec_table(conn, name)
                for row in rows:
                    conn.execute(
                        f"INSERT INTO {name} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                        (row["rowid"], row["namespace"], row["embedding"]),
                    )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('embed_dim', ?)",
                (str(EMBEDDING_DIM),),
            )
            self._stamp_identity(conn)
            self._migrate_content_schema(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @classmethod
    def _migrate_content_schema(cls, conn: sqlite3.Connection) -> None:
        """Apply the content-table migrations (v2, v3, v4) in order.

        Every step is idempotent (column-existence checks, IF NOT EXISTS) so
        this is safe to run on any database regardless of which path reached it.
        """
        cls._migrate_to_v2(conn)
        cls._migrate_to_v3(conn)
        cls._migrate_to_v4(conn)

    @staticmethod
    def _migrate_to_v2(conn: sqlite3.Connection) -> None:
        """Version 2: usage-tracking columns and the revisions table."""
        for table in _TABLES:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if "retrieval_count" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN retrieval_count INTEGER NOT NULL DEFAULT 0")
            if "last_accessed" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN last_accessed TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                title TEXT NOT NULL,
                op TEXT NOT NULL,
                payload TEXT NOT NULL,
                revised_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_revisions_item ON revisions(item_type, item_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_revisions_ns ON revisions(namespace, revised_at)")

    @staticmethod
    def _migrate_to_v3(conn: sqlite3.Connection) -> None:
        """Version 3: temporal facts — validity columns on knowledge, and the
        (namespace, subject) uniqueness constraint narrowed to current rows so
        superseded history can share a subject."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(knowledge)")}
        if "valid_until" not in cols:
            conn.execute("ALTER TABLE knowledge ADD COLUMN valid_until TEXT")
        if "superseded_by" not in cols:
            conn.execute("ALTER TABLE knowledge ADD COLUMN superseded_by TEXT")
        conn.execute("DROP INDEX IF EXISTS idx_knowledge_ns_subject")
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_ns_subject_current
            ON knowledge(namespace, subject) WHERE valid_until IS NULL
        """)

    @staticmethod
    def _migrate_to_v4(conn: sqlite3.Connection) -> None:
        """Version 4: the append-only audit log.

        Event trail, not content: revisions hold what an item *was* (for
        restore, pruned per item); the audit log holds what *happened* and
        who did it (for accountability, never pruned).
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                op TEXT NOT NULL,
                item_type TEXT,
                item_id TEXT,
                namespace TEXT,
                title TEXT,
                actor TEXT,
                client TEXT,
                ip TEXT,
                detail TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_item ON audit_log(item_type, item_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ns ON audit_log(namespace, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)")

    def rebuild_vec_tables(self) -> None:
        """Drop and recreate all vec0 tables empty, at the configured dimension.

        The reindex flow: content tables are untouched; the caller re-embeds
        every item afterwards. Also records the (possibly new) dimension,
        embedding identity, and schema version, and clears any pending rebuild.
        """
        logger.info("Rebuilding vector tables at dim %d...", EMBEDDING_DIM)
        conn = self._get_conn()
        conn.execute("BEGIN")
        try:
            for name in self._VEC_MIGRATION_SOURCES:
                conn.execute(f"DROP TABLE IF EXISTS {name}")
                self._create_vec_table(conn, name)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('embed_dim', ?)",
                (str(EMBEDDING_DIM),),
            )
            self._stamp_identity(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        self.reindex_pending = False
        logger.info("Vector tables rebuilt, re-embedding content for %d dims", EMBEDDING_DIM)

    def stored_embed_dim(self) -> int | None:
        """The dimension the vector index was built with, or None for a
        database created before schema_meta existed (pre-versioning)."""
        row = self._get_conn().execute(
            "SELECT value FROM schema_meta WHERE key = 'embed_dim'"
        ).fetchone()
        return int(row["value"]) if row else None

    def stored_embed_identity(self) -> dict[str, str]:
        """The embedding identity the vector index was built with, keyed by
        _IDENTITY_KEYS. Empty for a database written before it was recorded."""
        placeholders = ",".join("?" * len(self._IDENTITY_KEYS))
        return {
            row["key"]: row["value"]
            for row in self._get_conn().execute(
                f"SELECT key, value FROM schema_meta WHERE key IN ({placeholders})",
                self._IDENTITY_KEYS,
            ).fetchall()
        }

    def set_embedding(self, item_type: str, item_id: str, embedding: list[float]) -> bool:
        """Write an item's embedding without touching its content or timestamps.

        Used by the reindex flow. Returns False when the item doesn't exist.
        """
        spec = _SPEC_BY_ITEM_TYPE.get(item_type)
        if spec is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        table = spec.table
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT rowid, namespace FROM {table} WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return False
        self._upsert_vec(conn, f"vec_{table}", row["rowid"], embedding, row["namespace"])
        conn.commit()
        return True

    # ── Usage tracking & revisions ──

    def record_access(self, refs: list[tuple[str, str]]) -> None:
        """Bump retrieval_count/last_accessed for the given (item_type, id) pairs.

        Called explicitly by the read/search surfaces only — never from
        internal reads — so exports, backups, list pages, and the web viewer
        don't inflate the counters. updated_at is deliberately untouched.
        """
        if not refs:
            return
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        by_table: dict[str, set[str]] = {}
        for item_type, item_id in refs:
            spec = _SPEC_BY_ITEM_TYPE.get(item_type)
            if spec:
                by_table.setdefault(spec.table, set()).add(item_id)
        for table, ids in by_table.items():
            conn.execute(
                f"UPDATE {table} SET retrieval_count = retrieval_count + 1, last_accessed = ? "
                f"WHERE id IN ({','.join('?' * len(ids))})",
                (now, *ids),
            )
        conn.commit()

    def _capture_revision(self, conn: sqlite3.Connection, table: str, row: dict, op: str) -> None:
        """Save a row's prior state into revisions and prune per-item history. Does not commit.

        `row` is the full table row about to be overwritten or deleted; the
        payload keeps only the model columns (tags/metadata stay in their
        stored JSON-string form, hydrated again on restore). op is 'update'
        or 'delete'.
        """
        if REVISIONS_KEEP <= 0:
            return
        payload = {col: row[col] for col in _SPECS[table].columns}
        conn.execute(
            "INSERT INTO revisions (item_type, item_id, namespace, title, op, payload, revised_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_SPECS[table].item_type, row["id"], row["namespace"],
             row[_SPECS[table].title_field], op, json.dumps(payload),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "DELETE FROM revisions WHERE item_type = ? AND item_id = ? AND id NOT IN "
            "(SELECT id FROM revisions WHERE item_type = ? AND item_id = ? ORDER BY id DESC LIMIT ?)",
            (_SPECS[table].item_type, row["id"], _SPECS[table].item_type, row["id"], REVISIONS_KEEP),
        )

    def list_revisions(self, item_type: str | None = None, item_id: str | None = None,
                       namespace: str | None = None, limit: int = 20) -> list[dict]:
        """Revision summaries, newest first — no payloads, so listings stay small."""
        if item_type is not None and item_type not in _SPEC_BY_ITEM_TYPE:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        sql = "SELECT id, item_type, item_id, namespace, title, op, revised_at FROM revisions"
        clauses, params = [], []
        for column, value in (("item_type", item_type), ("item_id", item_id), ("namespace", namespace)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._get_conn().execute(sql, params).fetchall()

    def get_revision(self, revision_id: int) -> dict | None:
        """One revision with its payload hydrated into the item's model (as 'item')."""
        row = self._get_conn().execute(
            "SELECT * FROM revisions WHERE id = ?", (revision_id,)
        ).fetchone()
        if row is None:
            return None
        table = _SPEC_BY_ITEM_TYPE[row["item_type"]].table
        payload = _safe_json_loads(row["payload"], None, f"revision {revision_id}")
        if payload is None:
            return None
        row["item"] = _row_to_model(_SPECS[table].model, payload)
        return row

    def item_vectors(self, table: str, namespace: str) -> list[tuple[str, str, list[float]]]:
        """(id, title/subject, embedding) for every current item in the namespace
        that has a whole-item vector.

        Feeds the consolidation report's duplicate clustering. Chunked
        documents (no whole-document vector) and superseded facts (vector
        dropped at supersession) are naturally absent.
        """
        if table not in _TABLES:
            raise ValueError(f"Invalid table {table!r}")
        rows = self._get_conn().execute(
            f"SELECT t.id AS id, t.{_SPECS[table].title_field} AS title, v.embedding AS embedding "
            f"FROM {table} t JOIN vec_{table} v ON v.rowid = t.rowid "
            f"WHERE t.namespace = ?{_current_filter(table, 't')}",
            (namespace,),
        ).fetchall()
        return [
            (r["id"], r["title"],
             list(struct.unpack(f"{len(r['embedding']) // 4}f", r["embedding"])))
            for r in rows
        ]

    def stale_items(self, namespace: str, cutoff: str, limit: int = 50) -> list[dict]:
        """Current items never retrieved since usage tracking began and not
        updated since `cutoff` (ISO timestamp), oldest first."""
        out: list[dict] = []
        conn = self._get_conn()
        for table in _TABLES:
            rows = conn.execute(
                f"SELECT id, {_SPECS[table].title_field} AS title, updated_at, retrieval_count "
                f"FROM {table} WHERE namespace = ? AND retrieval_count = 0 "
                f"AND updated_at < ?{_current_filter(table)} "
                f"ORDER BY updated_at LIMIT ?",
                (namespace, cutoff, limit),
            ).fetchall()
            for row in rows:
                row["type"] = _SPECS[table].item_type
            out.extend(rows)
        out.sort(key=lambda r: r["updated_at"])
        return out[:limit]

    def append_audit(self, op: str, *, item_type: str | None = None,
                     item_id: str | None = None, namespace: str | None = None,
                     title: str | None = None, actor: str | None = None,
                     client: str | None = None, ip: str | None = None,
                     detail: dict | None = None) -> None:
        """Append one event to the audit log, pruning events past retention.

        Retention is time-based (AUDIT_KEEP_DAYS, default two years; 0 keeps
        forever) — accountability wants age, not a per-item count like
        revisions. The prune is an indexed range delete, cheap on every append.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc)
        conn.execute(
            "INSERT INTO audit_log (ts, op, item_type, item_id, namespace, title, "
            "actor, client, ip, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now.isoformat(), op, item_type, item_id, namespace,
             title, actor, client, ip, json.dumps(detail) if detail else None),
        )
        if AUDIT_KEEP_DAYS > 0:
            cutoff = (now - timedelta(days=AUDIT_KEEP_DAYS)).isoformat()
            conn.execute("DELETE FROM audit_log WHERE ts < ?", (cutoff,))
        conn.commit()

    def list_audit(self, item_type: str | None = None, item_id: str | None = None,
                   namespace: str | None = None, op: str | None = None,
                   limit: int = 50) -> list[dict]:
        """Audit events, newest first, with optional filters. detail is parsed JSON."""
        if item_type is not None and item_type not in _SPEC_BY_ITEM_TYPE:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        sql = "SELECT * FROM audit_log"
        clauses, params = [], []
        for column, value in (("item_type", item_type), ("item_id", item_id),
                              ("namespace", namespace), ("op", op)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._get_conn().execute(sql, params).fetchall()
        for row in rows:
            if row["detail"]:
                row["detail"] = _safe_json_loads(row["detail"], None, f"audit {row['id']}")
        return rows

    def item_embedding(self, item_type: str, item_id: str) -> list[float] | None:
        """The stored embedding for an item, or None when it has no vector.

        Chunked documents carry no whole-document vector; the renormalized
        mean of their chunk vectors stands in as a centroid — good enough
        for "more like this" neighbor queries.
        """
        spec = _SPEC_BY_ITEM_TYPE.get(item_type)
        if spec is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        table = spec.table
        conn = self._get_conn()
        row = conn.execute(f"SELECT rowid FROM {table} WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        vec = conn.execute(
            f"SELECT embedding FROM vec_{table} WHERE rowid = ?", (row["rowid"],)
        ).fetchone()
        if vec is not None:
            dim = len(vec["embedding"]) // 4
            return list(struct.unpack(f"{dim}f", vec["embedding"]))
        if table != "documents":
            return None
        chunk_vecs = conn.execute(
            "SELECT v.embedding AS embedding FROM document_chunks dc "
            "JOIN vec_document_chunks v ON v.rowid = dc.id WHERE dc.document_id = ?",
            (item_id,),
        ).fetchall()
        if not chunk_vecs:
            return None
        dim = len(chunk_vecs[0]["embedding"]) // 4
        centroid = [0.0] * dim
        for r in chunk_vecs:
            for i, v in enumerate(struct.unpack(f"{dim}f", r["embedding"])):
                centroid[i] += v
        norm = sum(v * v for v in centroid) ** 0.5
        return [v / norm for v in centroid] if norm else None

    def find_by_key(self, item_type: str, namespace: str, key_value: str) -> str | None:
        """The id currently occupying (namespace, title/subject), or None. Used by
        restore to refuse recreating an item under a key another item now owns."""
        spec = _SPEC_BY_ITEM_TYPE.get(item_type)
        if spec is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        table = spec.table
        row = self._get_conn().execute(
            f"SELECT id FROM {table} WHERE namespace = ? AND {_SPECS[table].title_field} = ?"
            f"{_current_filter(table)}",
            (namespace, key_value),
        ).fetchone()
        return row["id"] if row else None

    # ── Generic CRUD helpers ──

    def _store_item(self, table: str, item, embedding: list[float] | None,
                    chunks: list[tuple[str, list[float]]] | None = None):
        """Upsert an item keyed by (namespace, title/subject).

        Returns (stored item, created). Documents additionally get their chunk
        rows replaced — even when chunks is None, which clears stale chunks on
        an update.
        """
        conn = self._get_conn()
        key = _SPECS[table].title_field
        columns = _SPECS[table].columns
        existing = conn.execute(
            f"SELECT rowid, * FROM {table} WHERE namespace = ? AND {key} = ?{_current_filter(table)}",
            (item.namespace, getattr(item, key)),
        ).fetchone()

        if existing:
            self._capture_revision(conn, table, existing, "update")
            stored = item.model_copy(update={
                "id": existing["id"],
                "created_at": datetime.fromisoformat(existing["created_at"]),
                "updated_at": datetime.now(timezone.utc),
            })
            update_cols = [c for c in columns if c not in ("id", "namespace", key, "created_at")]
            conn.execute(
                f"UPDATE {table} SET {', '.join(f'{c} = ?' for c in update_cols)} WHERE id = ?",
                (*_item_column_values(stored, update_cols), existing["id"]),
            )
            if embedding is not None:
                self._upsert_vec(conn, f"vec_{table}", existing["rowid"], embedding, item.namespace)
            if table == "documents":
                self._replace_document_chunks(conn, existing["id"], chunks, namespace=item.namespace)
            conn.commit()
            return stored, False

        rowid = conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))}) RETURNING rowid",
            _item_column_values(item, columns),
        ).fetchone()["rowid"]
        if embedding is not None:
            conn.execute(
                f"INSERT INTO vec_{table} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                (rowid, item.namespace, _serialize_embedding(embedding)),
            )
        if table == "documents":
            self._replace_document_chunks(conn, item.id, chunks, namespace=item.namespace)
        conn.commit()
        return item, True

    def _get_item(self, table: str, item_id: str):
        row = self._get_conn().execute(
            f"SELECT * FROM {table} WHERE id = ?", (item_id,)
        ).fetchone()
        return _row_to_model(_SPECS[table].model, row) if row else None

    def _delete_item(self, table: str, item_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            f"DELETE FROM {table} WHERE id = ? RETURNING rowid, *", (item_id,)
        ).fetchone()
        if not row:
            return False
        self._capture_revision(conn, table, row, "delete")
        conn.execute(f"DELETE FROM vec_{table} WHERE rowid = ?", (row["rowid"],))
        conn.commit()
        return True

    def _upsert_vec(self, conn: sqlite3.Connection, vec_table: str, rowid: int, embedding: list[float], namespace: str) -> None:
        """Write an embedding for rowid, inserting the vec row if it doesn't exist yet.

        The insert fallback matters when a row was first stored without an embedding
        (e.g. FTS-only mode) and later re-stored once an embedder is available.
        The item's namespace never changes on update, so only the embedding column
        is written (vec0 forbids UPDATE on partition key columns anyway).
        """
        updated = conn.execute(
            f"UPDATE {vec_table} SET embedding = ? WHERE rowid = ?",
            (_serialize_embedding(embedding), rowid),
        ).rowcount
        if updated == 0:
            conn.execute(
                f"INSERT INTO {vec_table} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                (rowid, namespace, _serialize_embedding(embedding)),
            )

    def _list_items(self, table: str, namespace: str) -> list:
        rows = self._get_conn().execute(
            f"SELECT * FROM {table} WHERE namespace = ?{_current_filter(table)} "
            f"ORDER BY updated_at DESC", (namespace,)
        ).fetchall()
        return [_row_to_model(_SPECS[table].model, r) for r in rows]

    def _update_item(self, table: str, item_id: str, embedding: list[float] | None, **fields):
        invalid = set(fields) - _SPECS[table].update_fields
        if invalid:
            raise ValueError(f"Invalid {table} fields: {invalid}")
        conn = self._get_conn()
        prior = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (item_id,)).fetchone()
        if prior is None:
            return None
        self._capture_revision(conn, table, prior, "update")
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clauses = []
        values = []
        for key, value in fields.items():
            if key in ("tags", "metadata"):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            values.append(value)
        values.append(item_id)
        row = conn.execute(
            f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = ? RETURNING rowid, *", values
        ).fetchone()
        if not row:
            return None
        if embedding is not None:
            self._upsert_vec(conn, f"vec_{table}", row["rowid"], embedding, row["namespace"])
        conn.commit()
        return _row_to_model(_SPECS[table].model, row)

    # ── Documents CRUD ──

    def store_document(self, doc: Document, embedding: list[float] | None, chunks: list[tuple[str, list[float]]] | None = None) -> tuple[Document, bool]:
        return self._store_item("documents", doc, embedding, chunks)

    def get_document(self, doc_id: str) -> Document | None:
        return self._get_item("documents", doc_id)

    def update_document(self, doc_id: str, embedding: list[float] | None = None, **fields) -> Document | None:
        return self._update_item("documents", doc_id, embedding, **fields)

    def delete_document(self, doc_id: str) -> bool:
        return self._delete_item("documents", doc_id)

    def list_documents(self, namespace: str) -> list[Document]:
        return self._list_items("documents", namespace)

    def replace_document_chunks(self, doc_id: str, chunks: list[tuple[str, list[float]]] | None) -> None:
        """Replace all chunks for a document. Deletes existing chunks, then inserts new ones if provided."""
        conn = self._get_conn()
        self._replace_document_chunks(conn, doc_id, chunks)
        conn.commit()

    # ── Knowledge CRUD ──

    def store_knowledge(self, k: Knowledge, embedding: list[float] | None) -> tuple[Knowledge, bool, str | None]:
        """Store a fact with temporal upsert semantics.

        Returns (stored, created, superseded_id). No current row with this
        (namespace, subject): plain insert. Current row with the *same* fact:
        in-place refresh of the other fields (revision captured), no history
        entry. Current row with a *different* fact: the old row is closed
        (valid_until, superseded_by) and kept as history, and the new fact is
        inserted as its successor.
        """
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT rowid, * FROM knowledge WHERE namespace = ? AND subject = ? "
            "AND valid_until IS NULL",
            (k.namespace, k.subject),
        ).fetchone()
        if existing is None or existing["fact"] == k.fact:
            stored, created = self._store_item("knowledge", k, embedding)
            return stored, created, None
        successor = self._supersede(conn, existing, k, embedding)
        return successor, True, existing["id"]

    def supersede_knowledge(self, k_id: str, successor: Knowledge,
                            embedding: list[float] | None) -> Knowledge | None:
        """Close the fact `k_id` and insert `successor` as its replacement.

        The update path for fact changes. Returns None when k_id doesn't
        exist; raises ValueError when it is already superseded (history is
        immutable) and sqlite3.IntegrityError when the successor's subject
        collides with a different current fact.
        """
        conn = self._get_conn()
        old = conn.execute("SELECT rowid, * FROM knowledge WHERE id = ?", (k_id,)).fetchone()
        if old is None:
            return None
        if old["valid_until"] is not None:
            raise ValueError(f"knowledge {k_id} is already superseded — history is immutable")
        return self._supersede(conn, old, successor, embedding)

    def _supersede(self, conn: sqlite3.Connection, old_row: dict, new_item: Knowledge,
                   embedding: list[float] | None) -> Knowledge:
        """Close old_row and insert new_item as the current fact. Commits.

        The old row keeps its content and usage counters — supersession *is*
        the history, so no revision is captured. Its vector is dropped: only
        current facts participate in semantic search.
        """
        now = datetime.now(timezone.utc)
        stored = new_item.model_copy(update={
            "created_at": now, "updated_at": now,
            "valid_until": None, "superseded_by": None,
        })
        try:
            conn.execute(
                "UPDATE knowledge SET valid_until = ?, superseded_by = ? WHERE id = ?",
                (now.isoformat(), stored.id, old_row["id"]),
            )
            conn.execute("DELETE FROM vec_knowledge WHERE rowid = ?", (old_row["rowid"],))
            columns = _SPECS["knowledge"].columns
            rowid = conn.execute(
                f"INSERT INTO knowledge ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))}) RETURNING rowid",
                _item_column_values(stored, columns),
            ).fetchone()["rowid"]
            if embedding is not None:
                conn.execute(
                    "INSERT INTO vec_knowledge (rowid, namespace, embedding) VALUES (?, ?, ?)",
                    (rowid, stored.namespace, _serialize_embedding(embedding)),
                )
            conn.commit()
        except Exception:
            # A subject conflict on the successor insert must not leave the
            # old row closed in the open transaction.
            conn.rollback()
            raise
        return stored

    def knowledge_history(self, namespace: str, subject: str) -> list[Knowledge]:
        """All facts ever held for (namespace, subject): current first, then
        superseded versions newest first."""
        rows = self._get_conn().execute(
            "SELECT * FROM knowledge WHERE namespace = ? AND subject = ? "
            "ORDER BY (valid_until IS NULL) DESC, created_at DESC",
            (namespace, subject),
        ).fetchall()
        return [_row_to_model(Knowledge, r) for r in rows]

    def get_knowledge(self, k_id: str) -> Knowledge | None:
        return self._get_item("knowledge", k_id)

    def update_knowledge(self, k_id: str, embedding: list[float] | None = None, **fields) -> Knowledge | None:
        return self._update_item("knowledge", k_id, embedding, **fields)

    def delete_knowledge(self, k_id: str) -> bool:
        return self._delete_item("knowledge", k_id)

    def list_knowledge(self, namespace: str) -> list[Knowledge]:
        return self._list_items("knowledge", namespace)

    # ── Notes CRUD ──

    def store_note(self, note: Note, embedding: list[float] | None) -> tuple[Note, bool]:
        return self._store_item("notes", note, embedding)

    def get_note(self, note_id: str) -> Note | None:
        return self._get_item("notes", note_id)

    def update_note(self, note_id: str, embedding: list[float] | None = None, **fields) -> Note | None:
        return self._update_item("notes", note_id, embedding, **fields)

    def delete_note(self, note_id: str) -> bool:
        return self._delete_item("notes", note_id)

    def list_notes(self, namespace: str) -> list[Note]:
        return self._list_items("notes", namespace)

    # ── Tags ──

    def update_tags(self, item_id: str, item_type: str, add_tags: list[str] | None = None, remove_tags: list[str] | None = None) -> list[str]:
        conn = self._get_conn()
        spec = _SPEC_BY_ITEM_TYPE.get(item_type)
        if spec is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        table = spec.table
        row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"{item_type} {item_id} not found")
        self._capture_revision(conn, table, row, "update")
        tags = set(_safe_json_loads(row["tags"], [], f"tags row {row.get('id','?')}"))
        if add_tags:
            tags.update(add_tags)
        if remove_tags:
            tags -= set(remove_tags)
        tag_list = sorted(tags)
        conn.execute(
            f"UPDATE {table} SET tags = ?, updated_at = ? WHERE id = ?",
            (json.dumps(tag_list), datetime.now(timezone.utc).isoformat(), item_id),
        )
        conn.commit()
        return tag_list

    # ── Search ──

    def search_fts(self, query: str, table: str = "all", namespace: str | None = None, limit: int = 20,
                   tags: list[str] | None = None, updated_after: str | None = None) -> list[SearchResult]:
        results = []
        for t in _TABLES:
            if table in ("all", t):
                results.extend(self._fts_search_table(t, query, namespace, limit, tags, updated_after))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_vec(self, embedding: list[float], table: str = "all", namespace: str | None = None, limit: int = 20,
                   tags: list[str] | None = None, updated_after: str | None = None) -> list[SearchResult]:
        results = []
        if table in ("all", "documents"):
            chunk_results = self._vec_search_document_chunks(embedding, namespace, limit, tags, updated_after)
            chunked_ids = {r.id for r in chunk_results}
            results.extend(chunk_results)
            # Also search whole-doc vectors (small docs and pre-chunk legacy data)
            for r in self._vec_search_table("documents", embedding, namespace, limit, tags, updated_after):
                if r.id not in chunked_ids:
                    results.append(r)
        for t in ("knowledge", "notes"):
            if table in ("all", t):
                results.extend(self._vec_search_table(t, embedding, namespace, limit, tags, updated_after))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_hybrid(self, query: str, embedding: list[float], table: str = "all", namespace: str | None = None, limit: int = 20,
                      tags: list[str] | None = None, updated_after: str | None = None) -> list[SearchResult]:
        fts_results = self.search_fts(query, table, namespace, limit * 2, tags, updated_after)
        vec_results = self.search_vec(embedding, table, namespace, limit * 2, tags, updated_after)

        # Reciprocal Rank Fusion — rank-based merging that's immune to score scale differences
        k = 60  # standard RRF constant
        rrf_scores: dict[str, dict] = {}

        for rank, r in enumerate(fts_results):
            rrf_scores[r.id] = {"result": r, "score": 1.0 / (k + rank + 1)}
        for rank, r in enumerate(vec_results):
            if r.id in rrf_scores:
                rrf_scores[r.id]["score"] += 1.0 / (k + rank + 1)
                # Prefer the semantic result: it may carry a precise chunk snippet
                rrf_scores[r.id]["result"] = r
            else:
                rrf_scores[r.id] = {"result": r, "score": 1.0 / (k + rank + 1)}

        merged = []
        for entry in rrf_scores.values():
            entry["result"].score = round(entry["score"], 6)
            merged.append(entry["result"])

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:limit]

    # ── Namespaces ──

    def rename_namespace(self, old: str, new: str) -> tuple[dict[str, int], dict[str, int]]:
        """Move every item in `old` to `new`, merging into an existing target.

        On a title/subject collision the moved item replaces the target's item,
        mirroring the upsert semantics of the store_* operations. Returns
        (moved counts, replaced counts) per table.
        """
        if old == new:
            raise ValueError("old and new namespace are identical — nothing to rename")
        conn = self._get_conn()
        counts = {}
        replaced = {}
        # vec0 forbids UPDATE on partition key columns, so the moved rows'
        # vectors are captured up front and rewritten under the new namespace.
        vec_rows = {
            name: conn.execute(f"{source_sql} WHERE t.namespace = ?", (old,)).fetchall()
            for name, source_sql in self._VEC_MIGRATION_SOURCES.items()
        }
        try:
            for table in _TABLES:
                key = _SPECS[table].title_field
                # The moved item wins a collision: drop the target's row (and
                # its vector; document chunks cascade via FK + trigger) first.
                # For knowledge only current rows collide — superseded history
                # on either side moves/stays untouched (subjects may repeat).
                losers = conn.execute(
                    f"""DELETE FROM {table} WHERE namespace = ?{_current_filter(table)} AND {key} IN
                        (SELECT {key} FROM {table} WHERE namespace = ?{_current_filter(table)})
                        RETURNING rowid, *""",
                    (new, old),
                ).fetchall()
                replaced[table] = len(losers)
                for loser in losers:
                    self._capture_revision(conn, table, loser, "delete")
                    conn.execute(f"DELETE FROM vec_{table} WHERE rowid = ?", (loser["rowid"],))
                cur = conn.execute(
                    f"UPDATE {table} SET namespace = ? WHERE namespace = ?", (new, old)
                )
                counts[table] = cur.rowcount
            for name, rows in vec_rows.items():
                for row in rows:
                    conn.execute(f"DELETE FROM {name} WHERE rowid = ?", (row["rowid"],))
                    conn.execute(
                        f"INSERT INTO {name} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                        (row["rowid"], new, row["embedding"]),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return counts, replaced

    def delete_namespace(self, namespace: str) -> dict[str, int]:
        conn = self._get_conn()
        counts = {}
        for table in _TABLES:
            rows = conn.execute(
                f"DELETE FROM {table} WHERE namespace = ? RETURNING rowid, *", (namespace,)
            ).fetchall()
            for row in rows:
                self._capture_revision(conn, table, row, "delete")
            counts[table] = len(rows)
            if rows:
                rowids = [r["rowid"] for r in rows]
                conn.execute(
                    f"DELETE FROM vec_{table} WHERE rowid IN ({','.join('?' * len(rowids))})",
                    rowids,
                )
        conn.commit()
        return counts

    def list_namespaces(self) -> list[str]:
        rows = self._get_conn().execute("""
            SELECT DISTINCT namespace FROM documents
            UNION
            SELECT DISTINCT namespace FROM knowledge WHERE valid_until IS NULL
            UNION
            SELECT DISTINCT namespace FROM notes
            ORDER BY namespace
        """).fetchall()
        return [r["namespace"] for r in rows]

    def list_page(self, item_type: str, namespace: str, limit: int, offset: int) -> tuple[list[dict], int]:
        """One page of item summaries in a namespace, newest first.

        Returns (items, total). Summaries carry the same fields as the list
        resources — no document/note content, so a page stays small no matter
        how large the underlying items are.
        """
        spec = _SPEC_BY_ITEM_TYPE.get(item_type)
        if spec is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        table = spec.table
        conn = self._get_conn()
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE namespace = ?{_current_filter(table)}",
            (namespace,),
        ).fetchone()["n"]
        columns = ", ".join(_SPECS[table].summary_columns)
        rows = conn.execute(
            f"SELECT {columns} FROM {table} WHERE namespace = ?{_current_filter(table)} "
            f"ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (namespace, limit, offset),
        ).fetchall()
        for row in rows:
            row["tags"] = _safe_json_loads(row["tags"], [], f"tags row {row['id']}")
        return rows, total

    def namespace_counts(self) -> dict[str, dict[str, int]]:
        """Per-namespace item counts for each content table, keyed by namespace.

        One COUNT query per table — no row content is loaded. Namespaces come
        back in sorted order (dicts preserve insertion order).
        """
        conn = self._get_conn()
        counts: dict[str, dict[str, int]] = {}
        for table in _TABLES:
            where = f"WHERE 1=1{_current_filter(table)} " if _current_filter(table) else ""
            for row in conn.execute(f"SELECT namespace, COUNT(*) AS n FROM {table} {where}GROUP BY namespace"):
                counts.setdefault(row["namespace"], dict.fromkeys(_TABLES, 0))[table] = row["n"]
        return {ns: counts[ns] for ns in sorted(counts)}

    # ── Private helpers ──

    def _fts_search_table(self, table: str, query: str, namespace: str | None, limit: int,
                          tags: list[str] | None = None, updated_after: str | None = None) -> list[SearchResult]:
        conn = self._get_conn()
        fts_table = f"{table}_fts"
        alias = _SPECS[table].alias
        filter_sql, filter_params = _item_filters(tags, updated_after, alias)
        sql = f"""
            SELECT {alias}.*, {fts_table}.rank
            FROM {fts_table}
            JOIN {table} {alias} ON {alias}.rowid = {fts_table}.rowid
            WHERE {fts_table} MATCH ?{_current_filter(table, alias)}{filter_sql}
        """
        params: list = [query, *filter_params]
        if namespace:
            sql += f" AND {alias}.namespace = ?"
            params.append(namespace)
        sql += f" ORDER BY {fts_table}.rank LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            # FTS5 rank is negative BM25 (more negative = better match)
            # Negate so higher = better, then normalize with 1/(1+x) to get 0-1 range
            score = 1.0 / (1.0 + abs(row["rank"]))
            results.append(_row_to_search_result(table, row, score))
        return results

    def _vec_search_table(self, table: str, embedding: list[float], namespace: str | None, limit: int,
                          tags: list[str] | None = None, updated_after: str | None = None) -> list[SearchResult]:
        conn = self._get_conn()
        vec_table = f"vec_{table}"

        # sqlite-vec requires LIMIT to be directly on a simple vec0 query — JOINs and
        # CTEs hide the LIMIT from its query planner. So we do two queries:
        # 1. KNN scan on vec0 (satisfies LIMIT requirement) → rowids + distances.
        #    The namespace partition key filters inside the index, so a small
        #    namespace still yields its own `limit` nearest neighbors.
        # 2. Single IN lookup on the main table → all detail rows at once (not N+1)
        # Tag/recency filters can only apply at step 2, so the KNN over-fetches
        # to compensate for neighbors the filters will drop.
        filter_sql, filter_params = _item_filters(tags, updated_after)
        knn_limit = limit * 3 if filter_sql else limit
        knn_sql = f"SELECT rowid, distance FROM {vec_table} WHERE embedding MATCH ? AND k = ?"
        knn_params: list = [_serialize_embedding(embedding), knn_limit]
        if namespace:
            knn_sql += " AND namespace = ?"
            knn_params.append(namespace)
        vec_rows = conn.execute(knn_sql, knn_params).fetchall()

        if not vec_rows:
            return []

        rowid_distance = {row["rowid"]: row["distance"] for row in vec_rows}
        placeholders = ",".join("?" * len(vec_rows))
        params: list = [row["rowid"] for row in vec_rows]

        detail_rows = conn.execute(
            f"SELECT *, rowid FROM {table} WHERE rowid IN ({placeholders}){filter_sql}",
            params + filter_params
        ).fetchall()

        results = []
        for row in sorted(detail_rows, key=lambda r: rowid_distance[r["rowid"]])[:limit]:
            distance = rowid_distance[row["rowid"]]
            score = _l2_to_cosine(distance)
            results.append(_row_to_search_result(table, row, score))
        return results

    def _replace_document_chunks(self, conn: sqlite3.Connection, doc_id: str, chunks: list[tuple[str, list[float]]] | None, namespace: str | None = None) -> None:
        """Delete existing chunks for a document and optionally insert new ones. Does not commit.

        namespace partitions the chunk vectors; when the caller doesn't already
        have it (the public replace path), it is looked up from the document.
        """
        conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
        if not chunks:
            return
        if namespace is None:
            row = conn.execute("SELECT namespace FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row is None:
                return
            namespace = row["namespace"]
        for i, (content, chunk_embedding) in enumerate(chunks):
            cursor = conn.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (?, ?, ?)",
                (doc_id, i, content),
            )
            conn.execute(
                "INSERT INTO vec_document_chunks (rowid, namespace, embedding) VALUES (?, ?, ?)",
                (cursor.lastrowid, namespace, _serialize_embedding(chunk_embedding)),
            )

    def _vec_search_document_chunks(self, embedding: list[float], namespace: str | None, limit: int,
                                    tags: list[str] | None = None, updated_after: str | None = None) -> list[SearchResult]:
        """Search chunk-level embeddings; returns the best matching chunk per document."""
        conn = self._get_conn()
        fetch_limit = limit * 3  # over-fetch to account for per-document dedup (and any filters)

        # Namespace filtering happens inside the KNN via the partition key, so
        # the over-fetch only compensates for multiple chunks per document.
        knn_sql = "SELECT rowid, distance FROM vec_document_chunks WHERE embedding MATCH ? AND k = ?"
        knn_params: list = [_serialize_embedding(embedding), fetch_limit]
        if namespace:
            knn_sql += " AND namespace = ?"
            knn_params.append(namespace)
        vec_rows = conn.execute(knn_sql, knn_params).fetchall()
        if not vec_rows:
            return []

        rowid_distance = {row["rowid"]: row["distance"] for row in vec_rows}
        placeholders = ",".join("?" * len(vec_rows))
        params: list = [row["rowid"] for row in vec_rows]

        filter_sql, filter_params = _item_filters(tags, updated_after, "d")
        rows = conn.execute(f"""
            SELECT dc.id AS chunk_rowid, dc.document_id, dc.content AS chunk_content,
                   d.namespace, d.title, d.tags
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.id IN ({placeholders}){filter_sql}
        """, params + filter_params).fetchall()

        # Keep only the best-scoring chunk per document
        best: dict[str, tuple] = {}
        for row in rows:
            dist = rowid_distance[row["chunk_rowid"]]
            if row["document_id"] not in best or dist < best[row["document_id"]][1]:
                best[row["document_id"]] = (row, dist)

        results = []
        for row, distance in sorted(best.values(), key=lambda x: x[1])[:limit]:
            score = _l2_to_cosine(distance)
            results.append(SearchResult(
                id=row["document_id"],
                type="document",
                namespace=row["namespace"],
                title=row["title"],
                snippet=row["chunk_content"],
                resource_uri=f"mnemomatic://document/{row['document_id']}",
                score=score,
                tags=_safe_json_loads(row["tags"], [], f"tags doc {row['document_id']}"),
                partial=True,
            ))
        return results

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
