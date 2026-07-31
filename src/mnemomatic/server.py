import json
import logging
import os
import re
import sqlite3
import threading
from importlib.metadata import PackageNotFoundError, version

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from mnemomatic import model_config
from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.compact import CompactToolsMiddleware
from mnemomatic.db import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_THRESHOLD, EMBEDDING_DIM, Database, _chunk_text
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
                db = Database(DB_PATH, allow_dim_change=REINDEX)
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
        if database.dim_change_pending:
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
    return {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}


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

    Uses upsert semantics: if an entry with the same namespace + subject already exists,
    it is updated in place. Check `created` in the response to distinguish new vs updated.

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
    stored, created = _db().store_knowledge(k, embedding)
    return {"id": stored.id, "namespace": stored.namespace, "subject": stored.subject, "created": created}


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
    persist, and return {id, <key>, updated}."""
    cfg = _UPDATE_CONFIG[item_type]
    db = _db()
    existing = cfg["getter"](db, id)
    if existing is None:
        return {"error": f"{item_type.capitalize()} {id} not found"}

    try:
        cfg["model"](**{**existing.model_dump(), **fields})
    except ValidationError as e:
        return {"error": "Invalid update", "details": _format_validation_error(e)}

    embedding = cfg["embed"](id, existing, fields)
    updated = cfg["updater"](db, id, embedding, fields)
    if updated is None:
        return {"error": f"{item_type.capitalize()} {id} not found"}
    return {"id": updated.id, cfg["key"]: getattr(updated, cfg["key"]), "updated": True}


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
    """Permanently delete a document from Mnem-O-matic.

    Use when a document is no longer relevant or was stored by mistake. This action
    is irreversible. If the document might still be useful later, consider updating
    it or adding a "deprecated" tag instead.

    Args:
        id: The document ID to delete.
    """
    return {"id": id, "deleted": _db().delete_document(id)}


@mcp.tool(annotations=_ANN_DELETE)
def delete_knowledge(id: str) -> dict:
    """Permanently delete a knowledge entry from Mnem-O-matic.

    Use when a fact is no longer true or was stored incorrectly. This action is
    irreversible. If the fact is still true but outdated, prefer using update_knowledge
    to correct it rather than deleting and re-creating it.

    Args:
        id: The knowledge entry ID to delete.
    """
    return {"id": id, "deleted": _db().delete_knowledge(id)}


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
    return {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}


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
    """Permanently delete a note from Mnem-O-matic.

    Use when a note is no longer relevant or was stored by mistake. This action is
    irreversible. If the content might still be useful, consider updating it or
    adding a "archived" tag instead of deleting.

    Args:
        id: The note ID to delete.
    """
    return {"id": id, "deleted": _db().delete_note(id)}


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
            results = _db().search_fts(fts_query, table=table, namespace=namespace, limit=limit)
            if mode == "hybrid" and emb is None:
                degraded = True
        elif mode == "semantic":
            embedding = _embed_query(query)
            if embedding is None:
                return [{"error": "Semantic search failed", "details": "Embedding service is unavailable. Try fulltext mode."}]
            results = _db().search_vec(embedding, table=table, namespace=namespace, limit=limit)
        else:  # hybrid with embedder
            embedding = _embed_query(query)
            # If embedding fails, degrade to fulltext search
            if embedding is None:
                logger.info("Hybrid search degrading to fulltext due to embedding failure")
                results = _db().search_fts(fts_query, table=table, namespace=namespace, limit=limit)
                degraded = True
            else:
                results = _db().search_hybrid(fts_query, embedding, table=table, namespace=namespace, limit=limit)
    except sqlite3.Error as e:
        # Escaping should keep FTS5 syntax errors out, but any residual DB
        # error must come back as a tool-level error, not a protocol failure.
        logger.warning("Search failed for query %r: %s", query, e)
        return [{"error": "Search failed", "details": str(e)}]

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


def _get_resource(item_type: str, id: str) -> str:
    """Shared body for the get_* MCP resources: fetch by id, return JSON or a not-found error."""
    obj = _READ_GETTERS[item_type](_db(), id)
    if obj is None:
        return json.dumps({"error": f"{item_type.capitalize()} {id} not found"})
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
    return json.loads(item.model_dump_json())


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
    model_name = model_config.CONFIG.get("model") or EMBED_MODEL or None
    info = {
        "version": _server_version(),
        "mode": embedder.mode if embedder is not None else "FTS-only (no embedder)",
        "model": model_name,
        "model_url": _HF_MODEL_PAGES.get(model_name),
        "dim_configured": db_module.EMBEDDING_DIM,
        "dim_database": _db().stored_embed_dim(),
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
    """Permanently delete all items in a namespace.

    Removes every document, knowledge entry, and note in the given namespace in
    a single atomic operation. This is irreversible — deleted items cannot be
    recovered. If you only want to reorganize content, use rename_namespace instead.

    Args:
        namespace: The namespace to delete.
    """
    counts = _db().delete_namespace(namespace)
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
