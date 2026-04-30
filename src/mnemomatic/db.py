import json
import logging
import os
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from mnemomatic.models import Document, Knowledge, Note, SearchResult

logger = logging.getLogger("mnemomatic")

EMBEDDING_DIM = int(os.environ.get("MNEMOMATIC_EMBED_DIM", "384"))
BUSY_TIMEOUT_MS = 5000

_DOCUMENT_FIELDS = frozenset({"title", "content", "mime_type", "tags", "metadata"})
_KNOWLEDGE_FIELDS = frozenset({"subject", "fact", "confidence", "source", "tags", "metadata"})
_NOTE_FIELDS = frozenset({"title", "content", "source", "tags", "metadata"})

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


class Database:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
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

        # Vector tables
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents
            USING vec0(embedding float[{EMBEDDING_DIM}])
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_knowledge
            USING vec0(embedding float[{EMBEDDING_DIM}])
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_notes
            USING vec0(embedding float[{EMBEDDING_DIM}])
        """)

        conn.commit()

    # ── Generic CRUD helpers ──

    def _get_item(self, table: str, converter, item_id: str):
        row = self._get_conn().execute(
            f"SELECT * FROM {table} WHERE id = ?", (item_id,)
        ).fetchone()
        return converter(row) if row else None

    def _delete_item(self, table: str, vec_table: str, item_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            f"DELETE FROM {table} WHERE id = ? RETURNING rowid", (item_id,)
        ).fetchone()
        if not row:
            return False
        conn.execute(f"DELETE FROM {vec_table} WHERE rowid = ?", (row["rowid"],))
        conn.commit()
        return True

    def _list_items(self, table: str, converter, namespace: str) -> list:
        rows = self._get_conn().execute(
            f"SELECT * FROM {table} WHERE namespace = ? ORDER BY updated_at DESC", (namespace,)
        ).fetchall()
        return [converter(r) for r in rows]

    def _update_item(self, table: str, vec_table: str, allowed_fields: frozenset, converter, item_id: str, embedding: list[float] | None, **fields):
        invalid = set(fields) - allowed_fields
        if invalid:
            raise ValueError(f"Invalid {table} fields: {invalid}")
        conn = self._get_conn()
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
            conn.execute(
                f"UPDATE {vec_table} SET embedding = ? WHERE rowid = ?",
                (_serialize_embedding(embedding), row["rowid"]),
            )
        conn.commit()
        return converter(row)

    # ── Documents CRUD ──

    def store_document(self, doc: Document, embedding: list[float] | None) -> tuple[Document, bool]:
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id, rowid, created_at FROM documents WHERE namespace = ? AND title = ?",
            (doc.namespace, doc.title),
        ).fetchone()
        if existing:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """UPDATE documents SET content = ?, mime_type = ?, tags = ?, metadata = ?, updated_at = ?
                   WHERE id = ?""",
                (doc.content, doc.mime_type, json.dumps(doc.tags), json.dumps(doc.metadata), now, existing["id"]),
            )
            if embedding is not None:
                conn.execute(
                    "UPDATE vec_documents SET embedding = ? WHERE rowid = ?",
                    (_serialize_embedding(embedding), existing["rowid"]),
                )
            conn.commit()
            return Document(
                id=existing["id"],
                namespace=doc.namespace,
                title=doc.title,
                content=doc.content,
                mime_type=doc.mime_type,
                tags=doc.tags,
                metadata=doc.metadata,
                created_at=datetime.fromisoformat(existing["created_at"]),
                updated_at=datetime.fromisoformat(now),
            ), False

        rowid = conn.execute(
            """INSERT INTO documents (id, namespace, title, content, mime_type, tags, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING rowid""",
            (doc.id, doc.namespace, doc.title, doc.content, doc.mime_type,
             json.dumps(doc.tags), json.dumps(doc.metadata),
             doc.created_at.isoformat(), doc.updated_at.isoformat()),
        ).fetchone()["rowid"]
        if embedding is not None:
            conn.execute(
                "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_embedding(embedding)),
            )
        conn.commit()
        return doc, True

    def get_document(self, doc_id: str) -> Document | None:
        return self._get_item("documents", self._row_to_document, doc_id)

    def update_document(self, doc_id: str, embedding: list[float] | None = None, **fields) -> Document | None:
        return self._update_item("documents", "vec_documents", _DOCUMENT_FIELDS, self._row_to_document, doc_id, embedding, **fields)

    def delete_document(self, doc_id: str) -> bool:
        return self._delete_item("documents", "vec_documents", doc_id)

    def list_documents(self, namespace: str) -> list[Document]:
        return self._list_items("documents", self._row_to_document, namespace)

    # ── Knowledge CRUD ──

    def store_knowledge(self, k: Knowledge, embedding: list[float] | None) -> tuple[Knowledge, bool]:
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id, rowid, created_at FROM knowledge WHERE namespace = ? AND subject = ?",
            (k.namespace, k.subject),
        ).fetchone()
        if existing:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """UPDATE knowledge SET fact = ?, confidence = ?, source = ?, tags = ?, metadata = ?, updated_at = ?
                   WHERE id = ?""",
                (k.fact, k.confidence, k.source, json.dumps(k.tags), json.dumps(k.metadata), now, existing["id"]),
            )
            if embedding is not None:
                conn.execute(
                    "UPDATE vec_knowledge SET embedding = ? WHERE rowid = ?",
                    (_serialize_embedding(embedding), existing["rowid"]),
                )
            conn.commit()
            return Knowledge(
                id=existing["id"],
                namespace=k.namespace,
                subject=k.subject,
                fact=k.fact,
                confidence=k.confidence,
                source=k.source,
                tags=k.tags,
                metadata=k.metadata,
                created_at=datetime.fromisoformat(existing["created_at"]),
                updated_at=datetime.fromisoformat(now),
            ), False

        rowid = conn.execute(
            """INSERT INTO knowledge (id, namespace, subject, fact, confidence, source, tags, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING rowid""",
            (k.id, k.namespace, k.subject, k.fact, k.confidence, k.source,
             json.dumps(k.tags), json.dumps(k.metadata),
             k.created_at.isoformat(), k.updated_at.isoformat()),
        ).fetchone()["rowid"]
        if embedding is not None:
            conn.execute(
                "INSERT INTO vec_knowledge (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_embedding(embedding)),
            )
        conn.commit()
        return k, True

    def get_knowledge(self, k_id: str) -> Knowledge | None:
        return self._get_item("knowledge", self._row_to_knowledge, k_id)

    def update_knowledge(self, k_id: str, embedding: list[float] | None = None, **fields) -> Knowledge | None:
        return self._update_item("knowledge", "vec_knowledge", _KNOWLEDGE_FIELDS, self._row_to_knowledge, k_id, embedding, **fields)

    def delete_knowledge(self, k_id: str) -> bool:
        return self._delete_item("knowledge", "vec_knowledge", k_id)

    def list_knowledge(self, namespace: str) -> list[Knowledge]:
        return self._list_items("knowledge", self._row_to_knowledge, namespace)

    # ── Notes CRUD ──

    def store_note(self, note: Note, embedding: list[float] | None) -> tuple[Note, bool]:
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id, rowid, created_at FROM notes WHERE namespace = ? AND title = ?",
            (note.namespace, note.title),
        ).fetchone()
        if existing:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """UPDATE notes SET content = ?, source = ?, tags = ?, metadata = ?, updated_at = ?
                   WHERE id = ?""",
                (note.content, note.source, json.dumps(note.tags), json.dumps(note.metadata), now, existing["id"]),
            )
            if embedding is not None:
                conn.execute(
                    "UPDATE vec_notes SET embedding = ? WHERE rowid = ?",
                    (_serialize_embedding(embedding), existing["rowid"]),
                )
            conn.commit()
            return Note(
                id=existing["id"],
                namespace=note.namespace,
                title=note.title,
                content=note.content,
                source=note.source,
                tags=note.tags,
                metadata=note.metadata,
                created_at=datetime.fromisoformat(existing["created_at"]),
                updated_at=datetime.fromisoformat(now),
            ), False

        rowid = conn.execute(
            """INSERT INTO notes (id, namespace, title, content, source, tags, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING rowid""",
            (note.id, note.namespace, note.title, note.content, note.source,
             json.dumps(note.tags), json.dumps(note.metadata),
             note.created_at.isoformat(), note.updated_at.isoformat()),
        ).fetchone()["rowid"]
        if embedding is not None:
            conn.execute(
                "INSERT INTO vec_notes (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_embedding(embedding)),
            )
        conn.commit()
        return note, True

    def get_note(self, note_id: str) -> Note | None:
        return self._get_item("notes", self._row_to_note, note_id)

    def update_note(self, note_id: str, embedding: list[float] | None = None, **fields) -> Note | None:
        return self._update_item("notes", "vec_notes", _NOTE_FIELDS, self._row_to_note, note_id, embedding, **fields)

    def delete_note(self, note_id: str) -> bool:
        return self._delete_item("notes", "vec_notes", note_id)

    def list_notes(self, namespace: str) -> list[Note]:
        return self._list_items("notes", self._row_to_note, namespace)

    # ── Tags ──

    def update_tags(self, item_id: str, item_type: str, add_tags: list[str] | None = None, remove_tags: list[str] | None = None) -> list[str]:
        conn = self._get_conn()
        table = _ITEM_TYPE_TO_TABLE.get(item_type)
        if table is None:
            raise ValueError(f"Invalid type {item_type!r}: must be 'document', 'knowledge', or 'note'")
        row = conn.execute(f"SELECT tags FROM {table} WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"{item_type} {item_id} not found")
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
        if table in ("all","documents"):
            results.extend(self._fts_search_table("documents", query, namespace, limit))
        if table in ("all","knowledge"):
            results.extend(self._fts_search_table("knowledge", query, namespace, limit))
        if table in ("all","notes"):
            results.extend(self._fts_search_table("notes", query, namespace, limit))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_vec(self, embedding: list[float], table: str = "all", namespace: str | None = None, limit: int = 20) -> list[SearchResult]:
        results = []
        if table in ("all","documents"):
            results.extend(self._vec_search_table("documents", embedding, namespace, limit))
        if table in ("all","knowledge"):
            results.extend(self._vec_search_table("knowledge", embedding, namespace, limit))
        if table in ("all","notes"):
            results.extend(self._vec_search_table("notes", embedding, namespace, limit))
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
            else:
                rrf_scores[r.id] = {"result": r, "score": 1.0 / (k + rank + 1)}

        merged = []
        for entry in rrf_scores.values():
            entry["result"].score = round(entry["score"], 6)
            merged.append(entry["result"])

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:limit]

    # ── Namespaces ──

    def rename_namespace(self, old: str, new: str) -> dict[str, int]:
        conn = self._get_conn()
        counts = {}
        try:
            for table in ("documents", "knowledge", "notes"):
                cur = conn.execute(
                    f"UPDATE {table} SET namespace = ? WHERE namespace = ?", (new, old)
                )
                counts[table] = cur.rowcount
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
            raise ValueError(
                f"Cannot rename '{old}' to '{new}': title/subject conflict with existing items in '{new}'"
            )
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
        # Fetch extra rows when namespace filtering so we still have `limit` after filtering.
        fetch_limit = limit * 3 if namespace else limit

        # sqlite-vec requires LIMIT to be directly on a simple vec0 query — JOINs and
        # CTEs hide the LIMIT from its query planner. So we do two queries:
        # 1. KNN scan on vec0 (satisfies LIMIT requirement) → rowids + distances
        # 2. Single IN lookup on the main table → all detail rows at once (not N+1)
        vec_rows = conn.execute(
            f"SELECT rowid, distance FROM {vec_table} WHERE embedding MATCH ? AND k = ?",
            (_serialize_embedding(embedding), fetch_limit),
        ).fetchall()

        if not vec_rows:
            return []

        rowid_distance = {row["rowid"]: row["distance"] for row in vec_rows}
        placeholders = ",".join("?" * len(vec_rows))
        params: list = [row["rowid"] for row in vec_rows]

        sql = f"SELECT *, rowid FROM {table} WHERE rowid IN ({placeholders})"
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)

        detail_rows = conn.execute(sql, params).fetchall()

        results = []
        for row in sorted(detail_rows, key=lambda r: rowid_distance[r["rowid"]])[:limit]:
            distance = rowid_distance[row["rowid"]]
            # L2 distance to cosine similarity for normalized embeddings:
            # cosine_sim = 1 - (L2^2 / 2), range: -1 to 1, clamp to 0-1
            score = max(0.0, 1.0 - (distance * distance / 2.0))
            results.append(_row_to_search_result(table, row, score))
        return results

    def _row_to_document(self, row: dict) -> Document:
        return Document(
            id=row["id"], namespace=row["namespace"], title=row["title"],
            content=row["content"], mime_type=row["mime_type"],
            tags=_safe_json_loads(row["tags"], [], f"tags row {row['id']}"),
            metadata=_safe_json_loads(row["metadata"], {}, f"metadata row {row['id']}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_note(self, row: dict) -> Note:
        return Note(
            id=row["id"], namespace=row["namespace"], title=row["title"],
            content=row["content"], source=row["source"],
            tags=_safe_json_loads(row["tags"], [], f"tags row {row['id']}"),
            metadata=_safe_json_loads(row["metadata"], {}, f"metadata row {row['id']}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_knowledge(self, row: dict) -> Knowledge:
        return Knowledge(
            id=row["id"], namespace=row["namespace"], subject=row["subject"],
            fact=row["fact"], confidence=row["confidence"], source=row["source"],
            tags=_safe_json_loads(row["tags"], [], f"tags row {row['id']}"),
            metadata=_safe_json_loads(row["metadata"], {}, f"metadata row {row['id']}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
