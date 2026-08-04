import json
import logging
import os
import sqlite3
import struct
import threading
from datetime import datetime, timezone
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
SCHEMA_VERSION = 2
CHUNK_THRESHOLD = int(os.environ.get("MNEMOMATIC_CHUNK_THRESHOLD", "2000"))
CHUNK_SIZE = int(os.environ.get("MNEMOMATIC_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("MNEMOMATIC_CHUNK_OVERLAP", "200"))

# Revisions retained per item; the oldest beyond this are pruned as new ones
# are captured. 0 disables revision capture entirely.
REVISIONS_KEEP = int(os.environ.get("MNEMOMATIC_REVISIONS_KEEP", "10"))

# The three content tables, in display order. Each has a parallel vec_<table>.
_TABLES = ("documents", "knowledge", "notes")

# Per-table shape: the pydantic model, the full column list (mirroring the
# model's fields), and the fields update_* may change.
_TABLE_MODEL = {"documents": Document, "knowledge": Knowledge, "notes": Note}
_TABLE_COLUMNS = {
    "documents": ("id", "namespace", "title", "content", "mime_type",
                  "tags", "metadata", "created_at", "updated_at"),
    "knowledge": ("id", "namespace", "subject", "fact", "confidence", "source",
                  "tags", "metadata", "created_at", "updated_at"),
    "notes": ("id", "namespace", "title", "content", "source",
              "tags", "metadata", "created_at", "updated_at"),
}
_TABLE_UPDATE_FIELDS = {
    "documents": frozenset({"title", "content", "mime_type", "tags", "metadata"}),
    "knowledge": frozenset({"subject", "fact", "confidence", "source", "tags", "metadata"}),
    "notes": frozenset({"title", "content", "source", "tags", "metadata"}),
}

# Maps singular item_type strings (used by update_tags) to table names
_ITEM_TYPE_TO_TABLE = {"document": "documents", "knowledge": "knowledge", "note": "notes"}

# Per-table field mappings for search result construction
_TABLE_TO_TYPE = {"documents": "document", "knowledge": "knowledge", "notes": "note"}
_TABLE_TITLE_FIELD = {"documents": "title", "knowledge": "subject", "notes": "title"}
_TABLE_SNIPPET_FIELD = {"documents": "content", "knowledge": "fact", "notes": "content"}
_TABLE_SNIPPET_LEN = {"documents": 200, "knowledge": None, "notes": 200}
_TABLE_RESOURCE_URI = {
    "documents": "mnemomatic://document/{id}",
    "knowledge": "mnemomatic://knowledge-entry/{id}",
    "notes": "mnemomatic://note/{id}",
}

# Summary columns returned by list_page — mirrors the list resources; never
# includes document/note content so pages stay small.
_LIST_SUMMARY_COLUMNS = {
    "documents": ("id", "title", "mime_type", "tags", "updated_at", "retrieval_count", "last_accessed"),
    "knowledge": ("id", "subject", "fact", "confidence", "tags", "updated_at", "retrieval_count", "last_accessed"),
    "notes": ("id", "title", "source", "tags", "updated_at", "retrieval_count", "last_accessed"),
}


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
    title_field = _TABLE_TITLE_FIELD[table]
    snippet_field = _TABLE_SNIPPET_FIELD[table]
    snippet = row[snippet_field]
    max_len = _TABLE_SNIPPET_LEN[table]
    if max_len:
        snippet = snippet[:max_len]
    return SearchResult(
        id=row["id"],
        type=_TABLE_TO_TYPE[table],
        namespace=row["namespace"],
        title=row[title_field],
        snippet=snippet,
        resource_uri=_TABLE_RESOURCE_URI[table].format(id=row["id"]),
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
        elif col in ("created_at", "updated_at"):
            v = v.isoformat()
        values.append(v)
    return values


class Database:
    def __init__(self, db_path: str | Path = ":memory:", allow_dim_change: bool = False):
        """Args:
            db_path: SQLite file path, or ":memory:".
            allow_dim_change: when True, an EMBEDDING_DIM mismatch with the
                stored dimension does not fail startup; instead
                `dim_change_pending` is set and the caller is expected to run
                rebuild_vec_tables() + re-embed (the MNEMOMATIC_REINDEX flow).
        """
        self.db_path = str(db_path)
        self.allow_dim_change = allow_dim_change
        self.dim_change_pending = False
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
                updated_at TEXT NOT NULL
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
            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_ns_subject ON knowledge(namespace, subject);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_ns_title ON notes(namespace, title);
        """)

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

    def _ensure_vec_schema(self, conn: sqlite3.Connection) -> None:
        """Create or migrate the vec0 tables, then record schema version and dim.

        Fails fast when MNEMOMATIC_EMBED_DIM disagrees with the dimension the
        database was created with — a mismatch would otherwise surface later as
        confusing insert/search errors against half-usable vector tables.
        """
        version = conn.execute("PRAGMA user_version").fetchone()["user_version"]

        if version >= 1:
            stored = conn.execute("SELECT value FROM schema_meta WHERE key = 'embed_dim'").fetchone()
            if stored and int(stored["value"]) != EMBEDDING_DIM:
                if self.allow_dim_change:
                    logger.warning(
                        "Embedding dimension changing from %s to %d — vec tables will be "
                        "rebuilt and all content re-embedded", stored["value"], EMBEDDING_DIM,
                    )
                    self.dim_change_pending = True
                else:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: database was created with dim {stored['value']} "
                        f"but MNEMOMATIC_EMBED_DIM={EMBEDDING_DIM}. Set MNEMOMATIC_EMBED_DIM={stored['value']} "
                        f"to keep the existing index, or set MNEMOMATIC_REINDEX=1 to rebuild the index and "
                        f"re-embed all content at the new dimension on startup."
                    )
            if version < SCHEMA_VERSION:
                self._migrate_to_v2(conn)
                # A pending dim change leaves the version to rebuild_vec_tables,
                # which stamps it once the vec tables actually match.
                if not self.dim_change_pending:
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
                            if self.allow_dim_change:
                                # No point copying wrong-dim embeddings; the
                                # reindex rebuild will replace these tables.
                                conn.rollback()
                                self.dim_change_pending = True
                                logger.warning(
                                    "Legacy embeddings have dim %d, configured %d — deferring "
                                    "to reindex rebuild", found_dim, EMBEDDING_DIM,
                                )
                                self._migrate_to_v2(conn)
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
            self._migrate_to_v2(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _migrate_to_v2(conn: sqlite3.Connection) -> None:
        """Version 2: usage-tracking columns and the revisions table.

        Idempotent (column-existence checks, IF NOT EXISTS) so it is safe to
        run on any database regardless of which path reached it.
        """
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

    def rebuild_vec_tables(self) -> None:
        """Drop and recreate all vec0 tables empty, at the configured dimension.

        The reindex flow: content tables are untouched; the caller re-embeds
        every item afterwards. Also records the (possibly new) dimension and
        schema version, and clears any pending dim change.
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
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        self.dim_change_pending = False
        logger.info("Vector tables rebuilt, re-embedding content for %d dims", EMBEDDING_DIM)

    def stored_embed_dim(self) -> int | None:
        """The dimension the vector index was built with, or None for a
        database created before schema_meta existed (pre-versioning)."""
        row = self._get_conn().execute(
            "SELECT value FROM schema_meta WHERE key = 'embed_dim'"
        ).fetchone()
        return int(row["value"]) if row else None

    def set_embedding(self, item_type: str, item_id: str, embedding: list[float]) -> bool:
        """Write an item's embedding without touching its content or timestamps.

        Used by the reindex flow. Returns False when the item doesn't exist.
        """
        table = _ITEM_TYPE_TO_TABLE.get(item_type)
        if table is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
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
            table = _ITEM_TYPE_TO_TABLE.get(item_type)
            if table:
                by_table.setdefault(table, set()).add(item_id)
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
        payload = {col: row[col] for col in _TABLE_COLUMNS[table]}
        conn.execute(
            "INSERT INTO revisions (item_type, item_id, namespace, title, op, payload, revised_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_TABLE_TO_TYPE[table], row["id"], row["namespace"],
             row[_TABLE_TITLE_FIELD[table]], op, json.dumps(payload),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "DELETE FROM revisions WHERE item_type = ? AND item_id = ? AND id NOT IN "
            "(SELECT id FROM revisions WHERE item_type = ? AND item_id = ? ORDER BY id DESC LIMIT ?)",
            (_TABLE_TO_TYPE[table], row["id"], _TABLE_TO_TYPE[table], row["id"], REVISIONS_KEEP),
        )

    def list_revisions(self, item_type: str | None = None, item_id: str | None = None,
                       namespace: str | None = None, limit: int = 20) -> list[dict]:
        """Revision summaries, newest first — no payloads, so listings stay small."""
        if item_type is not None and item_type not in _ITEM_TYPE_TO_TABLE:
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
        table = _ITEM_TYPE_TO_TABLE[row["item_type"]]
        payload = _safe_json_loads(row["payload"], None, f"revision {revision_id}")
        if payload is None:
            return None
        row["item"] = _row_to_model(_TABLE_MODEL[table], payload)
        return row

    def find_by_key(self, item_type: str, namespace: str, key_value: str) -> str | None:
        """The id occupying (namespace, title/subject), or None. Used by restore
        to refuse recreating an item under a key another item now owns."""
        table = _ITEM_TYPE_TO_TABLE.get(item_type)
        if table is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        row = self._get_conn().execute(
            f"SELECT id FROM {table} WHERE namespace = ? AND {_TABLE_TITLE_FIELD[table]} = ?",
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
        key = _TABLE_TITLE_FIELD[table]
        columns = _TABLE_COLUMNS[table]
        existing = conn.execute(
            f"SELECT rowid, * FROM {table} WHERE namespace = ? AND {key} = ?",
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
        return _row_to_model(_TABLE_MODEL[table], row) if row else None

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
            f"SELECT * FROM {table} WHERE namespace = ? ORDER BY updated_at DESC", (namespace,)
        ).fetchall()
        return [_row_to_model(_TABLE_MODEL[table], r) for r in rows]

    def _update_item(self, table: str, item_id: str, embedding: list[float] | None, **fields):
        invalid = set(fields) - _TABLE_UPDATE_FIELDS[table]
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
        return _row_to_model(_TABLE_MODEL[table], row)

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

    def store_knowledge(self, k: Knowledge, embedding: list[float] | None) -> tuple[Knowledge, bool]:
        return self._store_item("knowledge", k, embedding)

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
        table = _ITEM_TYPE_TO_TABLE.get(item_type)
        if table is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
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

    def search_fts(self, query: str, table: str = "all", namespace: str | None = None, limit: int = 20) -> list[SearchResult]:
        results = []
        for t in _TABLES:
            if table in ("all", t):
                results.extend(self._fts_search_table(t, query, namespace, limit))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_vec(self, embedding: list[float], table: str = "all", namespace: str | None = None, limit: int = 20) -> list[SearchResult]:
        results = []
        if table in ("all", "documents"):
            chunk_results = self._vec_search_document_chunks(embedding, namespace, limit)
            chunked_ids = {r.id for r in chunk_results}
            results.extend(chunk_results)
            # Also search whole-doc vectors (small docs and pre-chunk legacy data)
            for r in self._vec_search_table("documents", embedding, namespace, limit):
                if r.id not in chunked_ids:
                    results.append(r)
        for t in ("knowledge", "notes"):
            if table in ("all", t):
                results.extend(self._vec_search_table(t, embedding, namespace, limit))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_hybrid(self, query: str, embedding: list[float], table: str = "all", namespace: str | None = None, limit: int = 20) -> list[SearchResult]:
        fts_results = self.search_fts(query, table, namespace, limit * 2)
        vec_results = self.search_vec(embedding, table, namespace, limit * 2)

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
                key = _TABLE_TITLE_FIELD[table]
                # The moved item wins a collision: drop the target's row (and
                # its vector; document chunks cascade via FK + trigger) first.
                losers = conn.execute(
                    f"""DELETE FROM {table} WHERE namespace = ? AND {key} IN
                        (SELECT {key} FROM {table} WHERE namespace = ?)
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
            SELECT DISTINCT namespace FROM knowledge
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
        table = _ITEM_TYPE_TO_TABLE.get(item_type)
        if table is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        conn = self._get_conn()
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE namespace = ?", (namespace,)
        ).fetchone()["n"]
        columns = ", ".join(_LIST_SUMMARY_COLUMNS[table])
        rows = conn.execute(
            f"SELECT {columns} FROM {table} WHERE namespace = ? "
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
            for row in conn.execute(f"SELECT namespace, COUNT(*) AS n FROM {table} GROUP BY namespace"):
                counts.setdefault(row["namespace"], dict.fromkeys(_TABLES, 0))[table] = row["n"]
        return {ns: counts[ns] for ns in sorted(counts)}

    # ── Private helpers ──

    def _fts_search_table(self, table: str, query: str, namespace: str | None, limit: int) -> list[SearchResult]:
        conn = self._get_conn()
        fts_table = f"{table}_fts"
        alias = {"documents": "d", "knowledge": "k", "notes": "n"}[table]
        sql = f"""
            SELECT {alias}.*, {fts_table}.rank
            FROM {fts_table}
            JOIN {table} {alias} ON {alias}.rowid = {fts_table}.rowid
            WHERE {fts_table} MATCH ?
        """
        params: list = [query]
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

    def _vec_search_table(self, table: str, embedding: list[float], namespace: str | None, limit: int) -> list[SearchResult]:
        conn = self._get_conn()
        vec_table = f"vec_{table}"

        # sqlite-vec requires LIMIT to be directly on a simple vec0 query — JOINs and
        # CTEs hide the LIMIT from its query planner. So we do two queries:
        # 1. KNN scan on vec0 (satisfies LIMIT requirement) → rowids + distances.
        #    The namespace partition key filters inside the index, so a small
        #    namespace still yields its own `limit` nearest neighbors.
        # 2. Single IN lookup on the main table → all detail rows at once (not N+1)
        knn_sql = f"SELECT rowid, distance FROM {vec_table} WHERE embedding MATCH ? AND k = ?"
        knn_params: list = [_serialize_embedding(embedding), limit]
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
            f"SELECT *, rowid FROM {table} WHERE rowid IN ({placeholders})", params
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

    def _vec_search_document_chunks(self, embedding: list[float], namespace: str | None, limit: int) -> list[SearchResult]:
        """Search chunk-level embeddings; returns the best matching chunk per document."""
        conn = self._get_conn()
        fetch_limit = limit * 3  # over-fetch to account for per-document dedup

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

        rows = conn.execute(f"""
            SELECT dc.id AS chunk_rowid, dc.document_id, dc.content AS chunk_content,
                   d.namespace, d.title, d.tags
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.id IN ({placeholders})
        """, params).fetchall()

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
