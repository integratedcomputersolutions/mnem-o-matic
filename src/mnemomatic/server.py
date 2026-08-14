import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from mnemomatic import model_config
from mnemomatic.audit import RequestMetaMiddleware, request_meta
from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.compact import CompactToolsMiddleware
from mnemomatic.db import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_THRESHOLD,
    EMBEDDING_DIM,
    Database,
    _chunk_text,
    _ITEM_TYPE_TO_TABLE,
    _TABLE_UPDATE_FIELDS,
)
from mnemomatic.models import Document, Knowledge, Note

logger = logging.getLogger("mnemomatic")

DB_PATH = os.environ.get("MNEMOMATIC_DB_PATH", "/data/mnemomatic.db")
HOST = os.environ.get("MNEMOMATIC_HOST", "0.0.0.0")
PORT = int(os.environ.get("MNEMOMATIC_PORT", "8000"))
API_KEY = os.environ.get("MNEMOMATIC_API_KEY", "")
CORS_ORIGINS = os.environ.get("MNEMOMATIC_CORS_ORIGINS", "")
EMBED_URL = os.environ.get("MNEMOMATIC_EMBED_URL", "")
EMBED_MODEL = os.environ.get("MNEMOMATIC_EMBED_MODEL", "")
UI_TOKEN = os.environ.get("MNEMOMATIC_UI_TOKEN", "").strip()
MAX_SEARCH_LIMIT = 100
MAX_LIST_LIMIT = 200

# Scheduled backups of the export archive — disabled unless a directory is set.
BACKUP_DIR = os.environ.get("MNEMOMATIC_BACKUP_DIR", "").strip()
BACKUP_INTERVAL_HOURS = float(os.environ.get("MNEMOMATIC_BACKUP_INTERVAL", "24"))
BACKUP_KEEP = int(os.environ.get("MNEMOMATIC_BACKUP_KEEP", "7"))

# Cosine similarity at or above which two items count as near-duplicates —
# used by the `similar` field on store responses and by consolidation_report's
# clustering. Correct-but-distinct search hits typically score 0.3–0.6 across
# the bundled models, near-duplicates well above 0.8. 0 disables the store-time
# check entirely.
SIMILAR_THRESHOLD = float(os.environ.get("MNEMOMATIC_SIMILAR_THRESHOLD", "0.8"))
_SIMILAR_LIMIT = 3

# Task prefixes for asymmetric embedding models. When the bundled model is
# trained with task prompts (e.g. EmbeddingGemma, multilingual-e5), the Docker
# build records them in model_config.json and they apply by default when
# embedding locally. External endpoints (MNEMOMATIC_EMBED_URL) default to no
# prefix since their model is unknown. Explicit env vars always win. Prefixes
# are baked into stored vectors, so changing them (like changing models)
# requires re-embedding existing content.
_DEFAULT_QUERY_PREFIX = "" if EMBED_URL else model_config.CONFIG.get("query_prefix", "")
_DEFAULT_DOC_PREFIX = "" if EMBED_URL else model_config.CONFIG.get("doc_prefix", "")
EMBED_QUERY_PREFIX = os.environ.get("MNEMOMATIC_EMBED_QUERY_PREFIX", _DEFAULT_QUERY_PREFIX)
EMBED_DOC_PREFIX = os.environ.get("MNEMOMATIC_EMBED_DOC_PREFIX", _DEFAULT_DOC_PREFIX)

# When set, startup rebuilds the vector index and re-embeds every stored item
# with the current embedder/dim/prefixes, then serves normally. Remove the
# flag after the run — it re-embeds on every boot while set.
REINDEX = os.environ.get("MNEMOMATIC_REINDEX", "").strip().lower() in ("1", "true", "yes")


def _embed_identity() -> dict[str, str]:
    """What the database records about which embedder built its vector index.

    The model name comes from the bundled model's config, or from
    MNEMOMATIC_EMBED_MODEL for an external endpoint. An empty name means the
    identity is unknown (FTS-only, or an external endpoint that never named its
    model), which disables the startup identity check — see Database.
    """
    return {
        "embed_model": model_config.CONFIG.get("model") or EMBED_MODEL or "",
        "embed_query_prefix": EMBED_QUERY_PREFIX,
        "embed_doc_prefix": EMBED_DOC_PREFIX,
    }


# Tool annotation presets
_ANN_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_ANN_STORE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_ANN_UPDATE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
_ANN_DELETE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)
_ANN_TAG = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

mcp = FastMCP(
    "Mnem-O-matic",
    json_response=True,
    host=HOST,
    port=PORT,
)

db: Database | None = None
_db_lock = threading.Lock()

_embedder_instance = None   # None means "no embedder available"
_embedder_initialized = False
_embedder_lock = threading.Lock()


def _db() -> Database:
    global db
    if db is None:
        with _db_lock:
            # Double-check pattern: verify again inside lock
            if db is None:
                os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
                db = Database(
                    DB_PATH, allow_reindex=REINDEX, embed_identity=_embed_identity()
                )
    return db


def _resolve_embedder():
    """Initialize and return the appropriate embedder, or None for FTS-only mode."""
    if EMBED_URL:
        try:
            from mnemomatic.embeddings import HttpEmbedder
            embedder = HttpEmbedder(EMBED_URL, EMBED_MODEL)
            logger.info("Embedder: %s endpoint %s (model=%r)", embedder.mode, EMBED_URL, EMBED_MODEL)
            _validate_embedding_dimension(embedder)
            return embedder
        except ValueError as e:
            logger.error("Invalid embedder configuration: %s", e)
        except Exception as e:
            logger.error("Failed to initialize HTTP embedder: %s: %s", type(e).__name__, e)
        return None

    model_path = os.environ.get("MNEMOMATIC_MODEL_PATH", "/app/model/model.onnx")
    if os.path.exists(model_path):
        try:
            from mnemomatic.embeddings import OnnxEmbedder
            embedder = OnnxEmbedder()
            logger.info("Embedder: %s (%s)", embedder.mode, model_path)
            _validate_embedding_dimension(embedder)
            return embedder
        except ImportError:
            logger.warning("onnxruntime not installed — starting in FTS-only mode")
        except (FileNotFoundError, RuntimeError) as e:
            logger.error("Failed to load embedding model: %s", e)
        except Exception as e:
            logger.error("Unexpected error initializing embedder: %s: %s", type(e).__name__, e)
    else:
        logger.warning("No embedding model found at %s — starting in FTS-only mode", model_path)
    return None


def _embedder():
    global _embedder_instance, _embedder_initialized
    if _embedder_initialized:
        return _embedder_instance
    with _embedder_lock:
        if _embedder_initialized:
            return _embedder_instance
        _embedder_instance = _resolve_embedder()
        _embedder_initialized = True
    return _embedder_instance


def _format_validation_error(e: ValidationError) -> str:
    """Format Pydantic ValidationError into a user-friendly message."""
    errors = []
    for error in e.errors():
        field = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")
    return "; ".join(errors)


def _validate_embedding_dimension(embedder) -> None:
    """Validate that configured embedding dimension matches actual embeddings.

    Computes a test embedding and checks its length matches MNEMOMATIC_EMBED_DIM.
    Logs a warning if there's a mismatch (could cause silent data corruption).
    """
    from mnemomatic.db import EMBEDDING_DIM
    try:
        test_embedding = embedder.embed("test")
        actual_dim = len(test_embedding)
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                "Embedding dimension mismatch: configured=%d, actual=%d. "
                "Set MNEMOMATIC_EMBED_DIM=%d to match your embedder.",
                EMBEDDING_DIM, actual_dim, actual_dim
            )
    except Exception as e:
        logger.debug("Could not validate embedding dimension: %s", e)


def _escape_fts_query(query: str) -> str:
    """Escape special characters in FTS5 queries.

    FTS5 gives punctuation syntactic meaning — AND/OR/NOT/NEAR operators,
    ``()`` grouping, ``*`` prefix match, ``-`` and ``:`` column filters,
    ``^`` initial-token match, ``"`` phrases — and anything else it can't
    tokenize is a syntax error (e.g. a trailing ``?``). So instead of
    blacklisting known operators, allow a query through bare only when it is
    entirely plain words; everything else is quoted into a literal phrase.

    Examples:
        "import AND" → '"import AND"'
        "std::vector" → '"std::vector"'
        "remains open?" → '"remains open?"'
    """
    is_bare_words = bool(re.fullmatch(r"[\w\s]+", query))
    has_operators = bool(re.search(r"\b(AND|OR|NOT|NEAR)\b", query, re.IGNORECASE))

    if is_bare_words and not has_operators:
        return query
    # Quote the entire query to make it a phrase search. This treats the whole
    # query as a literal phrase, preventing operator interpretation.
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


def _safe_embed(text: str) -> list[float] | None:
    """Safely compute embedding for text, returning None if embedding fails.

    Falls back to FTS-only search if embedder is unavailable or fails.
    Logs errors for debugging.
    """
    emb = _embedder()
    if emb is None:
        return None

    try:
        return emb.embed(text)
    except RuntimeError as e:
        logger.error("Embedding failed (will use FTS-only search): %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error during embedding: %s: %s", type(e).__name__, e)
        return None


def _safe_embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed many texts at once, one None per failed item.

    Uses the embedder's embed_batch (single padded inference for ONNX,
    concurrent requests for HTTP) and falls back to sequential _safe_embed
    for embedders that don't provide one.
    """
    emb = _embedder()
    if emb is None or not texts:
        return [None] * len(texts)
    batch = getattr(emb, "embed_batch", None)
    if batch is None:
        return [_safe_embed(t) for t in texts]
    try:
        return batch(texts)
    except Exception as e:
        logger.error("Batch embedding failed: %s: %s", type(e).__name__, e)
        return [None] * len(texts)


def _embed_query(text: str) -> list[float] | None:
    """Embedding for a search query, with the configured query prefix applied."""
    return _safe_embed(EMBED_QUERY_PREFIX + text)


def _embed_content(text: str) -> list[float] | None:
    """Embedding for stored content, with the configured document prefix applied."""
    return _safe_embed(EMBED_DOC_PREFIX + text)


def _similar_items(table: str, item_id: str, namespace: str,
                   embedding: list[float] | None) -> list[dict]:
    """Near-duplicates of a just-stored item, for the agent mid-write to judge.

    The server only flags — merging, superseding, or ignoring is the caller's
    decision. Empty when there is nothing above SIMILAR_THRESHOLD, no
    embedding (FTS-only mode, chunked documents), or the check is disabled.
    Never breaks the store that triggered it.
    """
    if embedding is None or SIMILAR_THRESHOLD <= 0:
        return []
    try:
        results = _db().search_vec(embedding, table=table, namespace=namespace,
                                   limit=_SIMILAR_LIMIT + 1)
    except Exception as e:
        logger.warning("Similar-item check failed: %s: %s", type(e).__name__, e)
        return []
    return [
        {"id": r.id, "title": r.title, "score": round(r.score, 3)}
        for r in results if r.id != item_id and r.score >= SIMILAR_THRESHOLD
    ][:_SIMILAR_LIMIT]


def _knowledge_embed_text(subject: str, fact: str) -> str:
    """Text embedded for a knowledge entry. Shared by store and update so they never drift."""
    return f"{subject}: {fact}"


def _note_embed_text(title: str, content: str) -> str:
    """Text embedded for a note. Shared by store and update so they never drift."""
    return f"{title}\n{content}"


def _embed_document_body(title: str, content: str) -> tuple[list[float] | None, list[tuple[str, list[float]]] | None]:
    """Compute the search representation for a document body.

    Documents at or above CHUNK_THRESHOLD are split into overlapping chunks, each
    embedded independently, so search can surface the most relevant passage. Smaller
    documents get a single whole-document embedding of "{title}\n{content}".

    Returns (embedding, chunks): exactly one is non-None. Chunks with failed
    embeddings are dropped; if none survive, chunks is None.
    """
    if len(content) >= CHUNK_THRESHOLD:
        texts = _chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        # The document prefix is applied for embedding only; stored chunk
        # content stays raw so snippets never leak the prefix.
        embeddings = _safe_embed_batch([EMBED_DOC_PREFIX + t for t in texts])
        chunks = [(c, e) for c, e in zip(texts, embeddings) if e is not None]
        return None, (chunks or None)
    return _embed_content(f"{title}\n{content}"), None


def _run_reindex() -> None:
    """Rebuild the vector index and re-embed every stored item.

    Runs at startup when MNEMOMATIC_REINDEX is set — after a change of
    embedding model, dimension, or task prefixes. Content tables are never
    modified (timestamps included); only vectors and document chunks are
    recomputed. Items whose embedding fails are logged and left FTS-only.
    """
    database = _db()
    if _embedder() is None:
        if database.reindex_pending:
            raise RuntimeError(
                "MNEMOMATIC_REINDEX is set and the embedding dimension changed, but no "
                "embedder is available — cannot rebuild the index. Configure an embedder "
                "or restore the previous MNEMOMATIC_EMBED_DIM."
            )
        logger.error("MNEMOMATIC_REINDEX is set but no embedder is available — skipping reindex")
        return

    logger.warning(
        "Reindex starting: rebuilding vector index at dim %d and re-embedding all content. "
        "Remove MNEMOMATIC_REINDEX after this run — it re-embeds on every startup while set.",
        EMBEDDING_DIM,
    )
    database.rebuild_vec_tables()

    counts = {"documents": 0, "knowledge": 0, "notes": 0, "failed": 0}
    for namespace in database.list_namespaces():
        for doc in database.list_documents(namespace):
            embedding, chunks = _embed_document_body(doc.title, doc.content)
            database.replace_document_chunks(doc.id, chunks)
            if embedding is not None:
                database.set_embedding("document", doc.id, embedding)
            if embedding is None and chunks is None:
                counts["failed"] += 1
                logger.error("Reindex: embedding failed for document %s (%r)", doc.id, doc.title)
            else:
                counts["documents"] += 1
        for k in database.list_knowledge(namespace):
            embedding = _embed_content(_knowledge_embed_text(k.subject, k.fact))
            if embedding is not None and database.set_embedding("knowledge", k.id, embedding):
                counts["knowledge"] += 1
            else:
                counts["failed"] += 1
                logger.error("Reindex: embedding failed for knowledge %s (%r)", k.id, k.subject)
        for note in database.list_notes(namespace):
            embedding = _embed_content(_note_embed_text(note.title, note.content))
            if embedding is not None and database.set_embedding("note", note.id, embedding):
                counts["notes"] += 1
            else:
                counts["failed"] += 1
                logger.error("Reindex: embedding failed for note %s (%r)", note.id, note.title)

    logger.info(
        "Reindex complete: %d documents, %d knowledge, %d notes re-embedded, %d failed",
        counts["documents"], counts["knowledge"], counts["notes"], counts["failed"],
    )


# ── Tools ──


@mcp.tool(annotations=_ANN_STORE)
def store_document(
    namespace: str,
    title: str,
    content: str,
    mime_type: str = "text/markdown",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store a document in Mnem-O-matic's shared memory.

    Use for structured, long-form reference material: code files, specs, configs,
    architecture docs, runbooks, README content, API schemas, or any content with
    a clear title that other sessions should be able to retrieve and read in full.

    Prefer documents over knowledge when the content is multi-line or prose-form.
    Prefer documents over notes when the content is structured and reusable rather
    than a passing thought.

    Uses upsert semantics: if a document with the same namespace + title already
    exists, it is updated in place. Check `created` in the response to distinguish
    a new entry (true) from an update (false).

    When the new content is nearly identical to items already in the namespace,
    the response includes `similar` (id, title, score) — review those before
    creating another near-duplicate.

    Args:
        namespace: Logical grouping for the document (e.g. "webapp", "infra", "global").
                   Use a project name to scope content, or "global" for cross-project material.
        title: Short, descriptive title. Acts as the deduplication key within a namespace.
        content: Full document body. Markdown is recommended for prose; raw text or code is fine too.
        mime_type: MIME type hint for the content (default "text/markdown"). Use "text/plain" for
                   plain text or "application/json" for JSON blobs.
        tags: Optional list of tags for filtering (e.g. ["auth", "backend", "draft"]).
        metadata: Optional free-form dict for structured annotations (e.g. {"author": "alice", "version": "2"}).
    """
    try:
        doc = Document(
            namespace=namespace,
            title=title,
            content=content,
            mime_type=mime_type,
            tags=tags or [],
            metadata=metadata or {},
        )
    except ValidationError as e:
        return {"error": "Invalid document", "details": _format_validation_error(e)}

    embedding, chunks = _embed_document_body(title, content)

    stored, created = _db().store_document(doc, embedding, chunks)
    response = {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}
    similar = _similar_items("documents", stored.id, namespace, embedding)
    if similar:
        response["similar"] = similar
    _audit("store", item_type="document", item_id=stored.id, namespace=stored.namespace,
           title=stored.title, created=created)
    return response


@mcp.tool(annotations=_ANN_STORE)
def store_knowledge(
    namespace: str,
    subject: str,
    fact: str,
    confidence: float = 1.0,
    source: str = "unknown",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store a knowledge entry (a discrete fact or decision) in Mnem-O-matic.

    Use for atomic, self-contained facts: architectural decisions, technology choices,
    conventions, constraints, or observations that can be expressed in one or two sentences.

    Prefer knowledge over documents when the content is a single fact rather than
    long-form material. Prefer knowledge over notes when the content is a confirmed fact
    rather than a tentative idea or rough thought.

    Good subjects: "auth mechanism", "database choice", "deploy pipeline", "rate limit policy"
    Good facts: "Uses JWT with RS256 signing", "Postgres, not MySQL — chosen for JSONB support"

    Uses temporal upsert semantics on namespace + subject: storing the same
    fact again refreshes the entry in place, while storing a *different* fact
    for an existing subject supersedes it — the old fact is kept as queryable
    history (see the fact_history tool) and the response carries its id in
    `superseded`.

    When the new content is nearly identical to items already in the namespace,
    the response includes `similar` (id, title, score) — review those before
    creating another near-duplicate.

    Args:
        namespace: Logical grouping (e.g. "webapp", "infra", "global").
        subject: Short label for what this fact is about. Acts as the deduplication key.
        fact: The fact itself, stated plainly and completely in one or two sentences.
        confidence: How certain this fact is, from 0.0 to 1.0 (default 1.0).
                    Use lower values for inferred or tentative knowledge.
        source: Where this fact came from (default "unknown"). E.g. "user", "code-review", "docs".
        tags: Optional list of tags for filtering.
        metadata: Optional free-form dict for structured annotations.
    """
    try:
        k = Knowledge(
            namespace=namespace,
            subject=subject,
            fact=fact,
            confidence=confidence,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )
    except ValidationError as e:
        return {"error": "Invalid knowledge entry", "details": _format_validation_error(e)}

    embedding = _embed_content(_knowledge_embed_text(subject, fact))
    stored, created, superseded = _db().store_knowledge(k, embedding)
    response = {"id": stored.id, "namespace": stored.namespace, "subject": stored.subject, "created": created}
    if superseded:
        response["superseded"] = superseded
    similar = _similar_items("knowledge", stored.id, namespace, embedding)
    if similar:
        response["similar"] = similar
    _audit("store", item_type="knowledge", item_id=stored.id, namespace=stored.namespace,
           title=stored.subject, created=created,
           **({"superseded": superseded} if superseded else {}))
    return response


# ── Shared update machinery ──


def _collect_fields(**kwargs) -> dict:
    """Drop None-valued kwargs, leaving only the fields the caller actually set."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _document_update_embedding(id: str, existing: Document, fields: dict) -> list[float] | None:
    """Recompute a document's embedding when its body changed, rewriting chunks as a side effect."""
    if "content" in fields:
        new_title = fields.get("title", existing.title)
        embedding, chunks = _embed_document_body(new_title, fields["content"])
        _db().replace_document_chunks(id, chunks)
        return embedding
    if "title" in fields and len(existing.content) < CHUNK_THRESHOLD:
        embedding, _ = _embed_document_body(fields["title"], existing.content)
        return embedding
    return None


def _knowledge_update_embedding(id: str, existing: Knowledge, fields: dict) -> list[float] | None:
    if "subject" in fields or "fact" in fields:
        return _embed_content(_knowledge_embed_text(
            fields.get("subject", existing.subject), fields.get("fact", existing.fact)))
    return None


def _note_update_embedding(id: str, existing: Note, fields: dict) -> list[float] | None:
    if "title" in fields or "content" in fields:
        return _embed_content(_note_embed_text(
            fields.get("title", existing.title), fields.get("content", existing.content)))
    return None


_UPDATE_CONFIG = {
    "document": {
        "model": Document,
        "getter": Database.get_document,
        "updater": lambda db, id, emb, fields: db.update_document(id, embedding=emb, **fields),
        "embed": _document_update_embedding,
        "key": "title",
    },
    "knowledge": {
        "model": Knowledge,
        "getter": Database.get_knowledge,
        "updater": lambda db, id, emb, fields: db.update_knowledge(id, embedding=emb, **fields),
        "embed": _knowledge_update_embedding,
        "key": "subject",
    },
    "note": {
        "model": Note,
        "getter": Database.get_note,
        "updater": lambda db, id, emb, fields: db.update_note(id, embedding=emb, **fields),
        "embed": _note_update_embedding,
        "key": "title",
    },
}


def _handle_update(item_type: str, id: str, fields: dict) -> dict:
    """Shared body for the update_* tools: validate the merged item, recompute its embedding,
    persist, and return {id, <key>, updated}.

    Knowledge is temporal: a superseded entry is immutable history, and a
    change to the fact itself supersedes (closes the current entry and inserts
    a successor) instead of overwriting.
    """
    cfg = _UPDATE_CONFIG[item_type]
    db = _db()
    existing = cfg["getter"](db, id)
    if existing is None:
        return {"error": f"{item_type.capitalize()} {id} not found"}
    if item_type == "knowledge" and existing.valid_until is not None:
        return {"error": "Cannot update a superseded fact",
                "details": f"Knowledge {id} is history (superseded by {existing.superseded_by}). "
                           f"Update the current fact for this subject instead, or use fact_history to inspect it."}

    try:
        cfg["model"](**{**existing.model_dump(), **fields})
    except ValidationError as e:
        return {"error": "Invalid update", "details": _format_validation_error(e)}

    if item_type == "knowledge" and "fact" in fields and fields["fact"] != existing.fact:
        return _supersede_update(existing, fields)

    embedding = cfg["embed"](id, existing, fields)
    updated = cfg["updater"](db, id, embedding, fields)
    if updated is None:
        return {"error": f"{item_type.capitalize()} {id} not found"}
    _audit("update", item_type=item_type, item_id=updated.id, namespace=updated.namespace,
           title=getattr(updated, cfg["key"]), fields=sorted(fields))
    return {"id": updated.id, cfg["key"]: getattr(updated, cfg["key"]), "updated": True}


def _supersede_update(existing: Knowledge, fields: dict) -> dict:
    """A fact change: build the successor entry and close the current one."""
    data = {**existing.model_dump(), **fields}
    for reset in ("id", "created_at", "updated_at", "valid_until", "superseded_by",
                  "retrieval_count", "last_accessed"):
        data.pop(reset, None)
    successor = Knowledge(**data)  # fresh id and timestamps; merged fields pre-validated
    embedding = _embed_content(_knowledge_embed_text(successor.subject, successor.fact))
    try:
        stored = _db().supersede_knowledge(existing.id, successor, embedding)
    except sqlite3.IntegrityError:
        return {"error": "Subject conflict",
                "details": f"Another current fact already holds subject {successor.subject!r} "
                           f"in namespace {successor.namespace!r}"}
    if stored is None:
        return {"error": f"Knowledge {existing.id} not found"}
    _audit("supersede", item_type="knowledge", item_id=stored.id, namespace=stored.namespace,
           title=stored.subject, superseded=existing.id)
    return {"id": stored.id, "subject": stored.subject, "updated": True, "superseded": existing.id}


@mcp.tool(annotations=_ANN_UPDATE)
def update_document(
    id: str,
    title: str | None = None,
    content: str | None = None,
    mime_type: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update an existing document in Mnem-O-matic. Only provided fields are changed; omitted fields are left as-is.

    Use this when you have the document's ID and want to modify specific fields without
    replacing the whole entry. To replace content entirely, provide the full new content.
    To update just tags or metadata without touching content, omit title and content.

    If title or content changes, the search embedding is automatically recomputed.

    Args:
        id: The document ID returned by store_document or search.
        title: New title (optional). Changes the deduplication key — avoid conflicts with existing titles.
        content: New content body (optional).
        mime_type: New MIME type (optional).
        tags: Replacement tag list (optional). This replaces all existing tags. Use the `tag` tool
              to add/remove individual tags without replacing the full list.
        metadata: Replacement metadata dict (optional). Replaces all existing metadata.
    """
    fields = _collect_fields(title=title, content=content, mime_type=mime_type, tags=tags, metadata=metadata)
    return _handle_update("document", id, fields)


@mcp.tool(annotations=_ANN_UPDATE)
def update_knowledge(
    id: str,
    subject: str | None = None,
    fact: str | None = None,
    confidence: float | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update an existing knowledge entry in Mnem-O-matic. Only provided fields are changed; omitted fields are left as-is.

    Use this when you have the entry's ID and want to correct or refine specific fields.
    For example, update `fact` when something has changed, adjust `confidence` as certainty
    increases, or update `source` when the origin becomes known.

    If subject or fact changes, the search embedding is automatically recomputed.

    Args:
        id: The knowledge entry ID returned by store_knowledge or search.
        subject: New subject label (optional). Changes the deduplication key.
        fact: New fact text (optional).
        confidence: New confidence score 0.0–1.0 (optional).
        source: New source string (optional).
        tags: Replacement tag list (optional). Replaces all existing tags.
        metadata: Replacement metadata dict (optional). Replaces all existing metadata.
    """
    fields = _collect_fields(subject=subject, fact=fact, confidence=confidence, source=source, tags=tags, metadata=metadata)
    return _handle_update("knowledge", id, fields)


@mcp.tool(annotations=_ANN_DELETE)
def delete_document(id: str) -> dict:
    """Delete a document from Mnem-O-matic.

    Use when a document is no longer relevant or was stored by mistake. The prior
    state is kept as a revision, so a mistaken delete can be undone via
    list_revisions + restore (until the item's revisions are pruned). If the
    document might still be useful later, consider updating it or adding a
    "deprecated" tag instead.

    Args:
        id: The document ID to delete.
    """
    existing = _db().get_document(id)
    deleted = _db().delete_document(id)
    if deleted:
        _audit("delete", item_type="document", item_id=id,
               namespace=existing.namespace if existing else None,
               title=getattr(existing, "title", None) if existing else None)
    return {"id": id, "deleted": deleted}


@mcp.tool(annotations=_ANN_DELETE)
def delete_knowledge(id: str) -> dict:
    """Delete a knowledge entry from Mnem-O-matic.

    Use when a fact was stored by mistake or should never have existed. If the
    fact simply changed, do NOT delete — store or update the corrected fact and
    the old one is kept as queryable history (see fact_history). A mistaken
    delete can be undone via list_revisions + restore.

    Args:
        id: The knowledge entry ID to delete.
    """
    existing = _db().get_knowledge(id)
    deleted = _db().delete_knowledge(id)
    if deleted:
        _audit("delete", item_type="knowledge", item_id=id,
               namespace=existing.namespace if existing else None,
               title=getattr(existing, "subject", None) if existing else None)
    return {"id": id, "deleted": deleted}


@mcp.tool(annotations=_ANN_STORE)
def store_note(
    namespace: str,
    title: str,
    content: str,
    source: str = "text",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store a note in Mnem-O-matic's shared memory.

    Use for informal, unstructured content: quick thoughts, ideas, observations,
    voice transcripts, meeting notes, brainstorms, or anything that doesn't yet have
    the structure of a document or the certainty of a knowledge entry.

    Prefer notes over documents when the content is rough or exploratory rather than
    finalized reference material. Prefer notes over knowledge when the content is more
    than one sentence or not yet a confirmed fact.

    Uses upsert semantics: if a note with the same namespace + title already exists,
    it is updated in place. Check `created` in the response to distinguish new vs updated.

    When the new content is nearly identical to items already in the namespace,
    the response includes `similar` (id, title, score) — review those before
    creating another near-duplicate.

    Args:
        namespace: Logical grouping (e.g. "personal", "webapp", "global").
        title: Short label for the note. Acts as the deduplication key within a namespace.
        content: The note body. No structure required — raw prose, bullet points, or transcribed speech.
        source: Origin of the content (default "text"). Use "voice" for transcribed audio,
                "clipboard" for pasted content, or any other label that helps identify provenance.
        tags: Optional list of tags for filtering.
        metadata: Optional free-form dict for structured annotations.
    """
    try:
        note = Note(
            namespace=namespace,
            title=title,
            content=content,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )
    except ValidationError as e:
        return {"error": "Invalid note", "details": _format_validation_error(e)}

    embedding = _embed_content(_note_embed_text(title, content))
    stored, created = _db().store_note(note, embedding)
    response = {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}
    similar = _similar_items("notes", stored.id, namespace, embedding)
    if similar:
        response["similar"] = similar
    _audit("store", item_type="note", item_id=stored.id, namespace=stored.namespace,
           title=stored.title, created=created)
    return response


@mcp.tool(annotations=_ANN_UPDATE)
def update_note(
    id: str,
    title: str | None = None,
    content: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update an existing note in Mnem-O-matic. Only provided fields are changed; omitted fields are left as-is.

    Use this to expand, correct, or refine a note after it was stored. For example,
    append to a transcript, correct a misheard word, or update the source label.

    If title or content changes, the search embedding is automatically recomputed.

    Args:
        id: The note ID returned by store_note or search.
        title: New title (optional). Changes the deduplication key.
        content: New content body (optional).
        source: New source label (optional).
        tags: Replacement tag list (optional). Replaces all existing tags.
        metadata: Replacement metadata dict (optional). Replaces all existing metadata.
    """
    fields = _collect_fields(title=title, content=content, source=source, tags=tags, metadata=metadata)
    return _handle_update("note", id, fields)


@mcp.tool(annotations=_ANN_DELETE)
def delete_note(id: str) -> dict:
    """Delete a note from Mnem-O-matic.

    Use when a note is no longer relevant or was stored by mistake. The prior
    state is kept as a revision, so a mistaken delete can be undone via
    list_revisions + restore (until the item's revisions are pruned). If the
    content might still be useful, consider updating it or adding an
    "archived" tag instead.

    Args:
        id: The note ID to delete.
    """
    existing = _db().get_note(id)
    deleted = _db().delete_note(id)
    if deleted:
        _audit("delete", item_type="note", item_id=id,
               namespace=existing.namespace if existing else None,
               title=getattr(existing, "title", None) if existing else None)
    return {"id": id, "deleted": deleted}


@mcp.tool(annotations=_ANN_TAG)
def tag(
    item_id: str,
    item_type: str,
    add_tags: list[str] | None = None,
    remove_tags: list[str] | None = None,
) -> dict:
    """Add or remove tags on a document, knowledge entry, or note.

    Prefer this over update_document/update_knowledge/update_note when you only want
    to change tags, as it merges changes rather than replacing the entire tag list.
    You can add and remove tags in a single call.

    Args:
        item_id: The ID of the item to tag.
        item_type: The item type — must be "document", "knowledge", or "note".
        add_tags: Tags to add. Tags already present are ignored (no duplicates).
        remove_tags: Tags to remove. Tags not present are ignored (no error).
    """
    try:
        tags = _db().update_tags(item_id, item_type, add_tags=add_tags, remove_tags=remove_tags)
        _audit("tag", item_type=item_type, item_id=item_id,
               added=add_tags or [], removed=remove_tags or [])
        return {"id": item_id, "tags": tags}
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool(annotations=_ANN_READ_ONLY)
def search(
    query: str,
    content_type: str = "all",
    namespace: str | None = None,
    limit: int = 10,
    mode: str = "hybrid",
    tags: list[str] | None = None,
    updated_after: str | None = None,
) -> list[dict]:
    """Search across documents, knowledge, and notes in Mnem-O-matic.

    Always search before storing — Mnem-O-matic may already contain what you're looking for,
    and searching first avoids creating duplicates. Also search at the start of a session
    to load relevant context before answering questions or starting work.

    Search modes:
    - "hybrid" (default): Combines keyword and semantic search using Reciprocal Rank Fusion.
      Best general-purpose choice — catches both exact matches and conceptually related content.
      Falls back to fulltext if embedder is unavailable (will include "_metadata" in response).
    - "fulltext": Keyword and phrase matching. Use when searching for a specific term, name,
      or exact phrase. Faster but misses synonyms and paraphrased content.
    - "semantic": Embedding-based similarity search. Use when the query is a concept or question
      and the stored content may use different words. E.g. "authentication" finds "JWT login tokens".
      Returns error if embedder is unavailable.

    Args:
        query: The search query. Can be a keyword, phrase, question, or concept. Cannot be empty.
        content_type: Filter by content type. "documents", "knowledge", "notes", or "all" (default, searches all types).
        namespace: Restrict results to a specific namespace (optional). Omit to search globally.
        limit: Maximum number of results to return (default 10, max 100). Increase for broader recall.
        mode: Search algorithm — "hybrid" (default), "fulltext", or "semantic".
        tags: Only return items carrying ALL of these tags (optional).
        updated_after: Only return items updated at or after this ISO date or
                       datetime, e.g. "2026-08-01" (optional). Useful for
                       "recent" queries; ordering is still by relevance.
    """
    valid_types = {"documents", "knowledge", "notes", "all"}
    if content_type not in valid_types:
        return [{"error": "Invalid content_type", "details": f"Must be one of: {', '.join(sorted(valid_types))}"}]

    valid_modes = {"hybrid", "fulltext", "semantic"}
    if mode not in valid_modes:
        return [{"error": "Invalid search mode", "details": f"Must be one of: {', '.join(sorted(valid_modes))}"}]

    # Validate query is not empty
    if not query or not query.strip():
        return [{"error": "Query cannot be empty", "details": "Provide a non-empty search query"}]

    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))

    if updated_after is not None:
        try:
            datetime.fromisoformat(updated_after)
        except ValueError:
            return [{"error": "Invalid updated_after",
                     "details": "Must be an ISO date or datetime, e.g. 2026-08-01 or 2026-08-01T12:00:00"}]
    filters = {"tags": tags or None, "updated_after": updated_after}

    # FTS5 needs special characters escaped; semantic embedding uses the original query
    fts_query = _escape_fts_query(query)

    table = content_type
    emb = _embedder()
    degraded = False

    if mode == "semantic" and emb is None:
        return [{"error": "Semantic search not available",
                 "details": "No embedder configured. Set MNEMOMATIC_EMBED_URL or use the full image with the built-in model."}]

    try:
        # hybrid silently degrades to fulltext when no embedder is available
        if mode == "fulltext" or (mode == "hybrid" and emb is None):
            results = _db().search_fts(fts_query, table=table, namespace=namespace, limit=limit, **filters)
            if mode == "hybrid" and emb is None:
                degraded = True
        elif mode == "semantic":
            embedding = _embed_query(query)
            if embedding is None:
                return [{"error": "Semantic search failed", "details": "Embedding service is unavailable. Try fulltext mode."}]
            results = _db().search_vec(embedding, table=table, namespace=namespace, limit=limit, **filters)
        else:  # hybrid with embedder
            embedding = _embed_query(query)
            # If embedding fails, degrade to fulltext search
            if embedding is None:
                logger.info("Hybrid search degrading to fulltext due to embedding failure")
                results = _db().search_fts(fts_query, table=table, namespace=namespace, limit=limit, **filters)
                degraded = True
            else:
                results = _db().search_hybrid(fts_query, embedding, table=table, namespace=namespace, limit=limit, **filters)
    except sqlite3.Error as e:
        # Escaping should keep FTS5 syntax errors out, but any residual DB
        # error must come back as a tool-level error, not a protocol failure.
        logger.warning("Search failed for query %r: %s", query, e)
        return [{"error": "Search failed", "details": str(e)}]

    _record_access([(r.type, r.id) for r in results])

    # Convert results to dicts and add degradation metadata if applicable
    response = [r.model_dump() for r in results]
    if degraded:
        # Add a metadata entry indicating degradation
        response.append({
            "_metadata": {
                "degraded": True,
                "reason": "Semantic search unavailable; results from fulltext search only"
            }
        })

    return response


_READ_GETTERS = {
    "document": Database.get_document,
    "knowledge": Database.get_knowledge,
    "note": Database.get_note,
}


def _record_access(refs: list[tuple[str, str]]) -> None:
    """Usage bookkeeping for items surfaced to a client — never breaks the request."""
    try:
        _db().record_access(refs)
    except Exception as e:
        logger.warning("Recording item access failed: %s: %s", type(e).__name__, e)


def _audit(op: str, *, item_type: str | None = None, item_id: str | None = None,
           namespace: str | None = None, title: str | None = None, **detail) -> None:
    """Append an audit event, enriched with the request's identity fields.

    Called from the write tools' success paths only; a failing audit write is
    logged and never breaks the operation it describes.
    """
    try:
        meta = request_meta()
        _db().append_audit(op, item_type=item_type, item_id=item_id,
                           namespace=namespace, title=title,
                           actor=meta["actor"], client=meta["client"], ip=meta["ip"],
                           detail=detail or None)
    except Exception as e:
        logger.warning("Audit write failed: %s: %s", type(e).__name__, e)


def _get_resource(item_type: str, id: str) -> str:
    """Shared body for the get_* MCP resources: fetch by id, return JSON or a not-found error."""
    obj = _READ_GETTERS[item_type](_db(), id)
    if obj is None:
        return json.dumps({"error": f"{item_type.capitalize()} {id} not found"})
    _record_access([(item_type, id)])
    return obj.model_dump_json()


@mcp.tool(annotations=_ANN_READ_ONLY)
def list_items(item_type: str, namespace: str, limit: int = 50, offset: int = 0) -> dict:
    """List items of one type in a namespace, newest first, with pagination.

    Use this to browse or inventory a namespace — e.g. reviewing what's stored,
    finding stale entries, or walking a large namespace page by page. For
    finding content by topic, prefer the search tool.

    Results are summaries (id, title/subject, tags, updated_at, ...) without
    document/note bodies; use the read tool to fetch an item's full content.
    The response's `total` is the overall item count, so `offset + len(items)
    < total` means there are more pages.

    Args:
        item_type: The item type — "document", "knowledge", or "note".
        namespace: The namespace to list.
        limit: Maximum items per page (default 50, max 200).
        offset: Number of items to skip — pass the previous offset + limit to
                fetch the next page (default 0).
    """
    try:
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        offset = max(0, int(offset))
        items, total = _db().list_page(item_type, namespace, limit, offset)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "namespace": namespace,
        "item_type": item_type,
    }


@mcp.tool(annotations=_ANN_READ_ONLY)
def read(item_type: str, id: str) -> dict:
    """Read the full content of a document, knowledge entry, or note by ID.

    Use this after searching to retrieve complete content — search results only
    contain a snippet. The item's resource_uri field tells you the type and ID.

    Args:
        item_type: The item type — "document", "knowledge", or "note".
        id: The unique item ID (UUID returned by store/search).
    """
    getter = _READ_GETTERS.get(item_type)
    if getter is None:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_READ_GETTERS))}"}
    item = getter(_db(), id)
    if item is None:
        return {"error": f"{item_type} not found", "id": id}
    _record_access([(item_type, id)])
    return json.loads(item.model_dump_json())


@mcp.tool(annotations=_ANN_READ_ONLY)
def related(item_type: str, id: str, namespace: str | None = None, limit: int = 5) -> dict:
    """Find items most similar to an existing item — "more like this".

    Use after reading an item to discover connected context without crafting
    a search query: related decisions around a document, notes touching the
    same topic as a fact. Results span all content types, ranked by embedding
    similarity to the given item.

    Requires semantic search (an embedder); works for chunked documents via
    their chunk-vector centroid. The item itself is never returned.

    Args:
        item_type: The item's type — "document", "knowledge", or "note".
        id: The item to find neighbors for.
        namespace: Restrict results to one namespace (optional). Omit to
                   search across all namespaces.
        limit: Maximum related items to return (default 5, max 100).
    """
    if item_type not in _READ_GETTERS:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_READ_GETTERS))}"}
    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))

    if _READ_GETTERS[item_type](_db(), id) is None:
        return {"error": f"{item_type} not found", "id": id}
    embedding = _db().item_embedding(item_type, id)
    if embedding is None:
        return {"error": "No embedding for this item",
                "details": "The item has no stored vector (FTS-only mode, or stored before "
                           "an embedder was configured — a MNEMOMATIC_REINDEX=1 restart embeds it)."}

    try:
        results = _db().search_vec(embedding, table="all", namespace=namespace, limit=limit + 1)
    except sqlite3.Error as e:
        logger.warning("Related-items search failed for %s %s: %s", item_type, id, e)
        return {"error": "Related search failed", "details": str(e)}
    neighbors = [r for r in results if r.id != id][:limit]
    _record_access([(r.type, r.id) for r in neighbors])
    return {"item_type": item_type, "item_id": id,
            "related": [r.model_dump() for r in neighbors]}


@mcp.tool(annotations=_ANN_READ_ONLY)
def fact_history(namespace: str, subject: str) -> dict:
    """The full timeline of a fact: the current entry first, then every
    superseded version, newest first.

    Knowledge is temporal — when a fact changes (via store_knowledge or
    update_knowledge), the old entry is closed rather than overwritten. Use
    this to answer "what did we believe before?" or to audit when an answer
    changed: each superseded entry carries valid_until (when it stopped being
    current) and superseded_by (the id of its replacement).

    History entries are read-only; only the current entry can be updated or
    superseded.

    Args:
        namespace: The fact's namespace.
        subject: The fact's subject (the deduplication key).
    """
    history = _db().knowledge_history(namespace, subject)
    _record_access([("knowledge", k.id) for k in history])
    return {
        "namespace": namespace,
        "subject": subject,
        "count": len(history),
        "history": [json.loads(k.model_dump_json()) for k in history],
    }


@mcp.tool(annotations=_ANN_READ_ONLY)
def list_revisions(
    item_type: str | None = None,
    item_id: str | None = None,
    namespace: str | None = None,
    limit: int = 20,
) -> dict:
    """List saved revisions — prior versions of items captured on every update and delete.

    Use this to find a version to roll back to (then call the restore tool with
    the revision's id), to recover something deleted by mistake, or to review
    what recently changed in a namespace. Filters combine; with no filters the
    newest revisions across the whole store are returned.

    Each revision is a summary (revision id, item_type, item_id, namespace,
    title/subject, op, revised_at) — op is "update" (the item changed after
    this state was saved) or "delete" (the item was deleted). The server keeps
    a limited number of revisions per item; older ones are pruned.

    Args:
        item_type: Filter by type — "document", "knowledge", or "note" (optional).
        item_id: Filter to one item's history (optional).
        namespace: Filter by namespace (optional).
        limit: Maximum revisions to return, newest first (default 20, max 200).
    """
    if item_type is not None and item_type not in _READ_GETTERS:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_READ_GETTERS))}"}
    limit = max(1, min(int(limit), MAX_LIST_LIMIT))
    revisions = _db().list_revisions(item_type=item_type, item_id=item_id,
                                     namespace=namespace, limit=limit)
    return {"revisions": revisions, "limit": limit}


@mcp.tool(annotations=_ANN_READ_ONLY)
def list_audit(
    item_type: str | None = None,
    item_id: str | None = None,
    namespace: str | None = None,
    op: str | None = None,
    limit: int = 50,
) -> dict:
    """List the audit trail — one event per write operation, newest first.

    Use this to review recent activity ("what changed in this namespace and
    when?"), trace what happened to a specific item, or see where a change
    came from. Complements revisions: revisions hold the content an item had
    (for restore), the audit log holds the events (who did what, when).

    Each event carries: ts, op (store/update/supersede/delete/tag/restore/
    rename_namespace/delete_namespace), item_type/item_id/namespace/title,
    actor (the client's self-declared X-Mnemomatic-Actor header, if any),
    client (user-agent), ip, and op-specific detail. With a shared API key
    the actor is self-reported, not authenticated.

    Args:
        item_type: Filter by type — "document", "knowledge", or "note" (optional).
        item_id: Filter to one item's events (optional).
        namespace: Filter by namespace (optional).
        op: Filter by operation name (optional).
        limit: Maximum events to return, newest first (default 50, max 200).
    """
    if item_type is not None and item_type not in _READ_GETTERS:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_READ_GETTERS))}"}
    limit = max(1, min(int(limit), MAX_LIST_LIMIT))
    events = _db().list_audit(item_type=item_type, item_id=item_id,
                              namespace=namespace, op=op, limit=limit)
    return {"events": events, "limit": limit}


@mcp.tool(annotations=_ANN_UPDATE)
def restore(revision_id: int) -> dict:
    """Restore an item to a saved revision — undo an update or recover a deleted item.

    Find the revision id with the list_revisions tool first. If the item still
    exists, its content is rolled back to the revision's state (the current
    state is saved as a new revision first, so a restore can itself be undone).
    If the item was deleted, it is recreated with its original id.

    Restoring re-embeds the content, so search reflects the restored state
    immediately.

    Args:
        revision_id: The revision to restore, from list_revisions.
    """
    try:
        rev = _db().get_revision(revision_id)
    except ValidationError as e:
        return {"error": "Revision payload no longer validates", "details": _format_validation_error(e)}
    if rev is None:
        return {"error": f"Revision {revision_id} not found"}

    item_type, item = rev["item_type"], rev["item"]
    key = _UPDATE_CONFIG[item_type]["key"]

    if _READ_GETTERS[item_type](_db(), rev["item_id"]) is not None:
        # Roll the live item back through the normal update path — it captures
        # the current state as a revision and re-embeds what changed.
        fields = {f: getattr(item, f) for f in _TABLE_UPDATE_FIELDS[_ITEM_TYPE_TO_TABLE[item_type]]}
        result = _handle_update(item_type, rev["item_id"], fields)
        if "error" in result:
            return result
        _audit("restore", item_type=item_type, item_id=rev["item_id"],
               namespace=rev["namespace"], title=rev["title"],
               revision_id=revision_id, recreated=False)
        return {**result, "restored_revision": revision_id, "recreated": False}

    # The item is gone — recreate it, unless its key now belongs to another item.
    if item_type == "knowledge" and item.valid_until is not None:
        return {"error": "Cannot restore a superseded fact",
                "details": "This revision is of a history entry; restore or re-store "
                           "the current fact for the subject instead."}
    occupant = _db().find_by_key(item_type, item.namespace, getattr(item, key))
    if occupant is not None:
        return {"error": "Cannot restore: key is taken",
                "details": f"{item_type} {occupant} now occupies "
                           f"{item.namespace!r}/{getattr(item, key)!r} — delete or rename it first"}

    item = item.model_copy(update={"updated_at": datetime.now(timezone.utc)})
    if item_type == "document":
        embedding, chunks = _embed_document_body(item.title, item.content)
        stored, _ = _db().store_document(item, embedding, chunks)
    elif item_type == "knowledge":
        stored, _, _ = _db().store_knowledge(item, _embed_content(_knowledge_embed_text(item.subject, item.fact)))
    else:
        stored, _ = _db().store_note(item, _embed_content(_note_embed_text(item.title, item.content)))
    _audit("restore", item_type=item_type, item_id=stored.id, namespace=stored.namespace,
           title=getattr(stored, key), revision_id=revision_id, recreated=True)
    return {"id": stored.id, key: getattr(stored, key), "namespace": stored.namespace,
            "restored_revision": revision_id, "recreated": True}


_ITEM_TYPE_TO_TABLE_INV = {table: item_type for item_type, table in _ITEM_TYPE_TO_TABLE.items()}


def _duplicate_clusters(item_type: str, vectors: list[tuple[str, str, list[float]]],
                        threshold: float) -> list[dict]:
    """Group items whose pairwise cosine similarity reaches the threshold.

    Vectors are stored L2-normalized, so the dot product is the cosine.
    Union-find over qualifying pairs; clusters report their strongest pair.
    """
    parent = list(range(len(vectors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = sum(a * b for a, b in zip(vectors[i][2], vectors[j][2]))
            if score >= threshold:
                pairs.append((i, j, score))
                parent[find(i)] = find(j)

    members: dict[int, list[int]] = {}
    for i in range(len(vectors)):
        members.setdefault(find(i), []).append(i)
    best: dict[int, float] = {}
    for i, j, score in pairs:
        root = find(i)
        best[root] = max(best.get(root, 0.0), score)

    return [
        {"type": item_type,
         "similarity": round(best[root], 3),
         "items": [{"id": vectors[i][0], "title": vectors[i][1]} for i in group]}
        for root, group in members.items() if len(group) > 1
    ]


@mcp.tool(annotations=_ANN_READ_ONLY)
def consolidation_report(namespace: str, similarity_threshold: float | None = None,
                         stale_days: int = 90) -> dict:
    """Mechanical consolidation candidates for a namespace: near-duplicate
    clusters and stale items. The report only flags — reviewing each candidate
    and deciding to merge, supersede, tag, delete, or keep is your job (the
    `consolidate` prompt walks through it).

    - duplicate_clusters: groups of same-type items whose embeddings are
      nearly identical (cosine >= similarity_threshold). Chunked documents
      have no whole-document vector and can't be clustered.
    - stale: current items never retrieved since usage tracking began and not
      updated in `stale_days` days, oldest first. On a server where tracking
      was enabled recently, "never retrieved" spans only that period — don't
      treat a low count as meaning unused forever.

    Args:
        namespace: The namespace to analyze.
        similarity_threshold: Cosine similarity for clustering (default: the
            server's MNEMOMATIC_SIMILAR_THRESHOLD, normally 0.8).
        stale_days: Only items untouched for this many days count as stale
            (default 90).
    """
    threshold = SIMILAR_THRESHOLD if similarity_threshold is None else float(similarity_threshold)
    if threshold <= 0:
        return {"error": "Invalid similarity_threshold", "details": "Must be positive (cosine similarity)"}

    clusters = []
    for table in ("documents", "knowledge", "notes"):
        vectors = _db().item_vectors(table, namespace)
        clusters.extend(_duplicate_clusters(_ITEM_TYPE_TO_TABLE_INV[table], vectors, threshold))
    clusters.sort(key=lambda c: c["similarity"], reverse=True)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(0, int(stale_days)))).isoformat()
    stale = _db().stale_items(namespace, cutoff)

    return {
        "namespace": namespace,
        "similarity_threshold": threshold,
        "stale_days": stale_days,
        "duplicate_clusters": clusters,
        "stale": stale,
        "counts": _db().namespace_counts().get(namespace, {}),
    }


# ── Prompts ──


@mcp.prompt()
def consolidate(namespace: str) -> str:
    """Review and tidy a namespace: merge duplicates, refresh or retire stale items."""
    return f"""You are consolidating the Mnem-O-matic namespace {namespace!r} — merging \
near-duplicates and reviewing stale content so the memory stays trustworthy and searchable.

1. Call consolidation_report(namespace={namespace!r}).

2. For each duplicate cluster, read() every member, then decide:
   - Same information twice → merge: keep the better-written item, fold any unique details \
into it with update_*, delete the other. Prefer merging content over discarding it.
   - Knowledge entries that disagree → the newer/correct fact should supersede: \
update_knowledge(id, fact=...) on the current entry closes the old one as history. \
Never edit superseded entries (they are immutable history).
   - Genuinely distinct items that merely look alike → leave them; consider sharper \
titles/subjects so they stay distinguishable.

3. For each stale item, read() it and decide: still true and useful → leave it (or tag \
"evergreen"); outdated but historically relevant → tag "deprecated"; wrong or worthless → \
delete it (deletes are recoverable via list_revisions/restore).

4. Be conservative: when unsure, keep the item and say so. Never delete or modify anything \
you have not read in full.

5. Finish with a short summary: actions taken (with ids), items flagged but deliberately \
kept, and anything a human should look at."""


@mcp.prompt()
def briefing(task: str, namespace: str = "") -> str:
    """Assemble relevant memory context for a task before starting work."""
    scope = f"namespace={namespace!r}" if namespace else "the whole store (omit the namespace argument)"
    return f"""Build a briefing from Mnem-O-matic for the following task, searching {scope}:

<task>
{task}
</task>

1. Derive 3–5 different search queries from the task: key terms, but also paraphrases \
and related concepts the stored content might use instead. Run search() for each — \
hybrid mode by default, semantic mode for the conceptual ones.

2. read() the items whose snippets look relevant; snippets are truncated and chunked \
documents return only the matching passage (partial: true).

3. Where a knowledge entry is central to the task, check fact_history(namespace, subject) \
— knowing an answer changed recently (and from what) is often as important as the answer.

4. Reply with a briefing, not a search log:
   - Established facts and decisions that constrain the task (cite item ids, note \
confidence and freshness).
   - Relevant reference material (documents/notes) with one-line summaries.
   - Gaps and open questions the memory does not answer.
   Keep it tight — only what changes how the task should be done."""


# ── Resources ──


@mcp.resource("mnemomatic://health")
def health() -> str:
    """Health check endpoint. Returns server status and configuration."""
    embedder = _embedder()
    embedding_mode = embedder.mode if embedder is not None else "FTS-only (no embedder)"

    return json.dumps({
        "status": "ok",
        "version": _server_version(),
        "embedding_mode": embedding_mode,
        "auth_enabled": bool(API_KEY),
    })


def _server_version() -> str:
    # Distribution name is "mnemomatic-server"; fall back gracefully when
    # running from a source tree with no installed metadata.
    try:
        return version("mnemomatic-server")
    except PackageNotFoundError:
        return "unknown"


def _make_export(namespace: str | None) -> tuple[bytes, str]:
    """Build the export archive; shared by /export and the web viewer."""
    from mnemomatic.export import build_export_zip

    return build_export_zip(_db(), namespace, server_version=_server_version())


async def _export_route(request):
    """GET /export[?namespace=...] — zip download, behind the Bearer middleware."""
    from starlette.responses import JSONResponse, Response

    namespace = request.query_params.get("namespace") or None
    if namespace and namespace not in _db().list_namespaces():
        return JSONResponse(
            {"error": "Namespace not found", "details": f"No items in namespace {namespace!r}"},
            status_code=404,
        )
    data, filename = _make_export(namespace)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Hugging Face model cards for the models the Docker build can bake in, keyed
# by the exact names it writes to model_config.json. Links point at the source
# model card (weights provenance, benchmarks, license), not the ONNX mirror
# the image downloads from. Unknown/external models simply get no link.
_HF_MODEL_PAGES = {
    "all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "gte-multilingual-base": "https://huggingface.co/Alibaba-NLP/gte-multilingual-base",
    "embeddinggemma-300m": "https://huggingface.co/google/embeddinggemma-300m",
    "amaretto-embed-148m": "https://huggingface.co/AmarettoLabs/amaretto-embed-148m",
}


def _settings_info() -> dict:
    """Configuration snapshot for the web viewer's settings page.

    Reads EMBEDDING_DIM through the db module so it reflects the value the
    running Database actually used (tests patch it there).
    """
    from mnemomatic import db as db_module
    from mnemomatic.embeddings import EMBED_API, MODEL_MAX_TOKENS

    embedder = _embedder()
    model_name = _embed_identity()["embed_model"] or None
    info = {
        "version": _server_version(),
        "mode": embedder.mode if embedder is not None else "FTS-only (no embedder)",
        "model": model_name,
        "model_url": _HF_MODEL_PAGES.get(model_name),
        "dim_configured": db_module.EMBEDDING_DIM,
        "dim_database": _db().stored_embed_dim(),
        "model_database": _db().stored_embed_identity().get("embed_model") or None,
        "query_prefix": EMBED_QUERY_PREFIX,
        "doc_prefix": EMBED_DOC_PREFIX,
        "chunk_threshold": CHUNK_THRESHOLD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    if EMBED_URL:
        info["endpoint_url"] = EMBED_URL
        info["wire_api"] = EMBED_API
    else:
        info["max_tokens"] = MODEL_MAX_TOKENS
    return info


@mcp.tool(annotations=_ANN_DELETE)
def delete_namespace(namespace: str) -> dict:
    """Delete all items in a namespace.

    Removes every document, knowledge entry, and note in the given namespace in
    a single atomic operation. Each item's final state is kept as a revision, so
    individual items can be recovered via list_revisions + restore — but there
    is no one-call undo for the whole namespace, so treat this as destructive.
    If you only want to reorganize content, use rename_namespace instead.

    Args:
        namespace: The namespace to delete.
    """
    counts = _db().delete_namespace(namespace)
    _audit("delete_namespace", namespace=namespace, deleted=sum(counts.values()))
    return {
        "namespace": namespace,
        "deleted": counts,
        "total": sum(counts.values()),
    }


@mcp.tool(annotations=_ANN_UPDATE)
def rename_namespace(old_namespace: str, new_namespace: str) -> dict:
    """Rename a namespace across all documents, knowledge entries, and notes.

    Moves every item in old_namespace to new_namespace atomically. If
    new_namespace already exists this acts as a merge: on a title/subject
    collision the moved item replaces the target's item (the same upsert
    semantics as the store tools). Check `replaced` in the response to see
    how many target items were overwritten.

    Args:
        old_namespace: The namespace to rename.
        new_namespace: The new name for the namespace. Must differ from old_namespace.
    """
    try:
        counts, replaced = _db().rename_namespace(old_namespace, new_namespace)
    except ValueError as e:
        return {"error": str(e)}
    _audit("rename_namespace", namespace=old_namespace, new_namespace=new_namespace,
           moved=sum(counts.values()), replaced=sum(replaced.values()))
    return {
        "old_namespace": old_namespace,
        "new_namespace": new_namespace,
        "renamed": counts,
        "replaced": replaced,
        "total": sum(counts.values()),
    }


@mcp.resource("mnemomatic://namespaces")
def list_namespaces() -> str:
    """List all namespaces in Mnem-O-matic."""
    namespaces = _db().list_namespaces()
    return json.dumps(namespaces)


@mcp.resource("mnemomatic://documents/{namespace}")
def list_documents(namespace: str) -> str:
    """List all documents in a namespace."""
    docs = _db().list_documents(namespace)
    return json.dumps([
        {"id": d.id, "title": d.title, "mime_type": d.mime_type,
         "tags": d.tags, "updated_at": d.updated_at.isoformat()}
        for d in docs
    ])


@mcp.resource("mnemomatic://knowledge/{namespace}")
def list_knowledge(namespace: str) -> str:
    """List all knowledge entries in a namespace."""
    entries = _db().list_knowledge(namespace)
    return json.dumps([
        {"id": k.id, "subject": k.subject, "fact": k.fact,
         "confidence": k.confidence, "tags": k.tags, "updated_at": k.updated_at.isoformat()}
        for k in entries
    ])


@mcp.resource("mnemomatic://notes/{namespace}")
def list_notes(namespace: str) -> str:
    """List all notes in a namespace."""
    notes = _db().list_notes(namespace)
    return json.dumps([
        {"id": n.id, "title": n.title, "source": n.source,
         "tags": n.tags, "updated_at": n.updated_at.isoformat()}
        for n in notes
    ])


@mcp.resource("mnemomatic://note/{id}")
def get_note(id: str) -> str:
    """Get a specific note by ID."""
    return _get_resource("note", id)


@mcp.resource("mnemomatic://document/{id}")
def get_document(id: str) -> str:
    """Get a specific document by ID."""
    return _get_resource("document", id)


@mcp.resource("mnemomatic://knowledge-entry/{id}")
def get_knowledge_entry(id: str) -> str:
    """Get a specific knowledge entry by ID."""
    return _get_resource("knowledge", id)


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Mnem-O-matic MCP server")
    logger.info("Configuration: db_path=%s, host=%s, port=%s", DB_PATH, HOST, PORT)

    # Pre-warm db and resolve embedder so the first request doesn't pay setup costs
    logger.info("Initializing database...")
    _db()
    logger.info("Initializing embedder...")
    _embedder()

    # Opt-in full re-embed (model/dim/prefix changes) before serving traffic.
    if REINDEX:
        _run_reindex()

    # Scheduled backups on a daemon thread (the Database hands each thread its
    # own connection, so the loop reads safely alongside request handling).
    if BACKUP_DIR:
        from pathlib import Path

        from mnemomatic.backup import start_backup_thread
        start_backup_thread(_db, Path(BACKUP_DIR), interval_hours=BACKUP_INTERVAL_HOURS,
                            keep=BACKUP_KEEP, server_version=_server_version())
        logger.info("Scheduled backups: every %gh to %s (keeping %d)",
                    BACKUP_INTERVAL_HOURS, BACKUP_DIR, BACKUP_KEEP)

    # Always use unified ASGI app + Uvicorn code path
    # Authentication is optional based on API_KEY environment variable
    logger.info("Building ASGI application...")
    app = mcp.streamable_http_app()

    # Zip export download. Inserted ahead of the MCP catch-all; NOT exempt
    # from Bearer auth — it returns the entire store.
    from starlette.routing import Route
    app.router.routes.insert(0, Route("/export", _export_route, methods=["GET"]))

    # Optional read-only web viewer at /ui, gated by a single shared secret.
    # Disabled unless MNEMOMATIC_UI_TOKEN is set, so it never exposes data by default.
    if UI_TOKEN:
        from mnemomatic.webui import register_webui
        register_webui(app, _db, UI_TOKEN, settings_info=_settings_info, make_export=_make_export)
        logger.info("Web viewer enabled at /ui")
    else:
        logger.info("Web viewer disabled (set MNEMOMATIC_UI_TOKEN to enable)")

    app = CompactToolsMiddleware(app)

    # Capture actor/client/ip per request for the audit log.
    app = RequestMetaMiddleware(app)

    # Middleware handles both authenticated and non-authenticated modes
    # If API_KEY is empty, auth is disabled but logging still tracks requests.
    # /ui is exempt from Bearer auth only when the viewer is actually registered.
    app = BearerAuthMiddleware(app, api_key=API_KEY, exempt_ui=bool(UI_TOKEN))

    if CORS_ORIGINS:
        from starlette.middleware.cors import CORSMiddleware
        origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
        if "*" in origins and not API_KEY:
            logger.warning(
                "SECURITY: CORS is open to all origins (*) and authentication is disabled — "
                "any website can read from and write to this server."
            )
        app = CORSMiddleware(
            app,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )
        logger.info("CORS enabled for origins: %s", origins)

    logger.info("Starting server on %s:%d", HOST, PORT)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
