import json
import logging
import os
import re
import threading
from importlib.metadata import version

import uvicorn
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.db import Database
from mnemomatic.models import Document, Knowledge, Note

logger = logging.getLogger("mnemomatic")

DB_PATH = os.environ.get("MNEMOMATIC_DB_PATH", "/data/mnemomatic.db")
HOST = os.environ.get("MNEMOMATIC_HOST", "0.0.0.0")
PORT = int(os.environ.get("MNEMOMATIC_PORT", "8000"))
API_KEY = os.environ.get("MNEMOMATIC_API_KEY", "")
EMBED_URL = os.environ.get("MNEMOMATIC_EMBED_URL", "")
EMBED_MODEL = os.environ.get("MNEMOMATIC_EMBED_MODEL", "")

mcp = FastMCP(
    "Mnem-O-matic",
    json_response=True,
    host=HOST,
    port=PORT,
)

db: Database | None = None
_db_lock = threading.Lock()

# Sentinel distinguishing "not yet initialised" from "no embedder available"
_UNSET = object()
_embedder_instance = _UNSET
_embedder_lock = threading.Lock()


def _db() -> Database:
    global db
    if db is None:
        with _db_lock:
            # Double-check pattern: verify again inside lock
            if db is None:
                os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
                db = Database(DB_PATH)
    return db


def _embedder():
    global _embedder_instance
    if _embedder_instance is not _UNSET:
        return _embedder_instance

    with _embedder_lock:
        # Double-check pattern: verify again inside lock
        if _embedder_instance is not _UNSET:
            return _embedder_instance

        if EMBED_URL:
            try:
                from mnemomatic.embeddings import HttpEmbedder
                _embedder_instance = HttpEmbedder(EMBED_URL, EMBED_MODEL)
                logger.info("Embedder: external HTTP endpoint %s (model=%r)", EMBED_URL, EMBED_MODEL)
                _validate_embedding_dimension(_embedder_instance)
            except ValueError as e:
                logger.error("Invalid embedder configuration: %s", e)
                _embedder_instance = None
            except Exception as e:
                logger.error("Failed to initialize HTTP embedder: %s: %s", type(e).__name__, e)
                _embedder_instance = None
        else:
            model_path = os.environ.get("MNEMOMATIC_MODEL_PATH", "/app/model/model.onnx")
            if os.path.exists(model_path):
                try:
                    from mnemomatic.embeddings import OnnxEmbedder
                    _embedder_instance = OnnxEmbedder()
                    logger.info("Embedder: built-in ONNX model (%s)", model_path)
                    _validate_embedding_dimension(_embedder_instance)
                except ImportError:
                    logger.warning("onnxruntime not installed — starting in FTS-only mode")
                    _embedder_instance = None
                except (FileNotFoundError, RuntimeError) as e:
                    logger.error("Failed to load embedding model: %s", e)
                    _embedder_instance = None
                except Exception as e:
                    logger.error("Unexpected error initializing embedder: %s: %s", type(e).__name__, e)
                    _embedder_instance = None
            else:
                logger.warning("No embedding model found at %s — starting in FTS-only mode", model_path)
                _embedder_instance = None

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

    FTS5 treats certain characters as operators (AND, OR, NOT, *, etc.).
    This function escapes them so they're treated as literal search terms.

    Examples:
        "import AND" → '"import AND"'
        "std::vector" → '"std::vector"'
    """
    # FTS5 operators and special characters: AND, OR, NOT, parentheses, quotes, etc.
    # Check for FTS5 operators (case-insensitive, word boundaries)
    has_operators = bool(re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE))
    has_special_chars = any(char in query for char in ["(", ")", "*", "-", '"'])

    if has_operators or has_special_chars:
        # Quote the entire query to make it a phrase search
        # This treats the whole query as a literal phrase, preventing operator interpretation
        escaped = query.replace('"', '""')
        return f'"{escaped}"'
    return query


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


# ── Tools ──


@mcp.tool()
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

    embedding = _safe_embed(f"{title}\n{content}")
    stored, created = _db().store_document(doc, embedding)
    return {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}


@mcp.tool()
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

    embedding = _safe_embed(f"{subject}: {fact}")
    stored, created = _db().store_knowledge(k, embedding)
    return {"id": stored.id, "namespace": stored.namespace, "subject": stored.subject, "created": created}


@mcp.tool()
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
    fields = {}
    if title is not None:
        fields["title"] = title
    if content is not None:
        fields["content"] = content
    if mime_type is not None:
        fields["mime_type"] = mime_type
    if tags is not None:
        fields["tags"] = tags
    if metadata is not None:
        fields["metadata"] = metadata

    doc = _db().get_document(id)
    if not doc:
        return {"error": f"Document {id} not found"}

    try:
        Document(
            namespace=doc.namespace,
            title=fields.get("title", doc.title),
            content=fields.get("content", doc.content),
            mime_type=fields.get("mime_type", doc.mime_type),
            tags=fields.get("tags", doc.tags),
            metadata=fields.get("metadata", doc.metadata),
        )
    except ValidationError as e:
        return {"error": "Invalid update", "details": _format_validation_error(e)}

    embedding = None
    if "title" in fields or "content" in fields:
        new_title = fields.get("title", doc.title)
        new_content = fields.get("content", doc.content)
        embedding = _safe_embed(f"{new_title}\n{new_content}")

    updated = _db().update_document(id, embedding=embedding, **fields)
    if not updated:
        return {"error": f"Document {id} not found"}
    return {"id": updated.id, "title": updated.title, "updated": True}


@mcp.tool()
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
    fields = {}
    if subject is not None:
        fields["subject"] = subject
    if fact is not None:
        fields["fact"] = fact
    if confidence is not None:
        fields["confidence"] = confidence
    if source is not None:
        fields["source"] = source
    if tags is not None:
        fields["tags"] = tags
    if metadata is not None:
        fields["metadata"] = metadata

    k = _db().get_knowledge(id)
    if not k:
        return {"error": f"Knowledge {id} not found"}

    try:
        Knowledge(
            namespace=k.namespace,
            subject=fields.get("subject", k.subject),
            fact=fields.get("fact", k.fact),
            confidence=fields.get("confidence", k.confidence),
            source=fields.get("source", k.source),
            tags=fields.get("tags", k.tags),
            metadata=fields.get("metadata", k.metadata),
        )
    except ValidationError as e:
        return {"error": "Invalid update", "details": _format_validation_error(e)}

    embedding = None
    if "subject" in fields or "fact" in fields:
        new_subject = fields.get("subject", k.subject)
        new_fact = fields.get("fact", k.fact)
        embedding = _safe_embed(f"{new_subject}: {new_fact}")

    updated = _db().update_knowledge(id, embedding=embedding, **fields)
    if not updated:
        return {"error": f"Knowledge {id} not found"}
    return {"id": updated.id, "subject": updated.subject, "updated": True}


@mcp.tool()
def delete_document(id: str) -> dict:
    """Permanently delete a document from Mnem-O-matic.

    Use when a document is no longer relevant or was stored by mistake. This action
    is irreversible. If the document might still be useful later, consider updating
    it or adding a "deprecated" tag instead.

    Args:
        id: The document ID to delete.
    """
    deleted = _db().delete_document(id)
    return {"id": id, "deleted": deleted}


@mcp.tool()
def delete_knowledge(id: str) -> dict:
    """Permanently delete a knowledge entry from Mnem-O-matic.

    Use when a fact is no longer true or was stored incorrectly. This action is
    irreversible. If the fact is still true but outdated, prefer using update_knowledge
    to correct it rather than deleting and re-creating it.

    Args:
        id: The knowledge entry ID to delete.
    """
    deleted = _db().delete_knowledge(id)
    return {"id": id, "deleted": deleted}


@mcp.tool()
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

    embedding = _safe_embed(f"{title}\n{content}")
    stored, created = _db().store_note(note, embedding)
    return {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}


@mcp.tool()
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
    fields = {}
    if title is not None:
        fields["title"] = title
    if content is not None:
        fields["content"] = content
    if source is not None:
        fields["source"] = source
    if tags is not None:
        fields["tags"] = tags
    if metadata is not None:
        fields["metadata"] = metadata

    note = _db().get_note(id)
    if not note:
        return {"error": f"Note {id} not found"}

    try:
        Note(
            namespace=note.namespace,
            title=fields.get("title", note.title),
            content=fields.get("content", note.content),
            source=fields.get("source", note.source),
            tags=fields.get("tags", note.tags),
            metadata=fields.get("metadata", note.metadata),
        )
    except ValidationError as e:
        return {"error": "Invalid update", "details": _format_validation_error(e)}

    embedding = None
    if "title" in fields or "content" in fields:
        new_title = fields.get("title", note.title)
        new_content = fields.get("content", note.content)
        embedding = _safe_embed(f"{new_title}\n{new_content}")

    updated = _db().update_note(id, embedding=embedding, **fields)
    if not updated:
        return {"error": f"Note {id} not found"}
    return {"id": updated.id, "title": updated.title, "updated": True}


@mcp.tool()
def delete_note(id: str) -> dict:
    """Permanently delete a note from Mnem-O-matic.

    Use when a note is no longer relevant or was stored by mistake. This action is
    irreversible. If the content might still be useful, consider updating it or
    adding a "archived" tag instead of deleting.

    Args:
        id: The note ID to delete.
    """
    deleted = _db().delete_note(id)
    return {"id": id, "deleted": deleted}


@mcp.tool()
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


@mcp.tool()
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

    # Clamp limit to reasonable range for personal use
    MAX_SEARCH_LIMIT = 100
    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))

    # FTS5 needs special characters escaped; semantic embedding uses the original query
    fts_query = _escape_fts_query(query)

    table = content_type
    emb = _embedder()
    degraded = False

    if mode == "semantic" and emb is None:
        return [{"error": "Semantic search not available",
                 "details": "No embedder configured. Set MNEMOMATIC_EMBED_URL or use the full image with the built-in model."}]

    # hybrid silently degrades to fulltext when no embedder is available
    if mode == "fulltext" or (mode == "hybrid" and emb is None):
        results = _db().search_fts(fts_query, table=table, namespace=namespace, limit=limit)
        if mode == "hybrid" and emb is None:
            degraded = True
    elif mode == "semantic":
        embedding = _safe_embed(query)
        if embedding is None:
            return [{"error": "Semantic search failed", "details": "Embedding service is unavailable. Try fulltext mode."}]
        results = _db().search_vec(embedding, table=table, namespace=namespace, limit=limit)
    else:  # hybrid with embedder
        embedding = _safe_embed(query)
        # If embedding fails, degrade to fulltext search
        if embedding is None:
            logger.info("Hybrid search degrading to fulltext due to embedding failure")
            results = _db().search_fts(fts_query, table=table, namespace=namespace, limit=limit)
            degraded = True
        else:
            results = _db().search_hybrid(fts_query, embedding, table=table, namespace=namespace, limit=limit)

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


# ── Resources ──


@mcp.resource("mnemomatic://health")
def health() -> str:
    """Health check endpoint. Returns server status and configuration."""
    embedder = _embedder()
    embedding_mode = "built-in ONNX" if hasattr(embedder, 'session') else \
                     "external HTTP" if embedder is not None else \
                     "FTS-only (no embedder)"

    return json.dumps({
        "status": "ok",
        "version": version("mnemomatic"),
        "embedding_mode": embedding_mode,
        "auth_enabled": bool(API_KEY),
    })


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
    note = _db().get_note(id)
    if not note:
        return json.dumps({"error": f"Note {id} not found"})
    return note.model_dump_json()


@mcp.resource("mnemomatic://document/{id}")
def get_document(id: str) -> str:
    """Get a specific document by ID."""
    doc = _db().get_document(id)
    if not doc:
        return json.dumps({"error": f"Document {id} not found"})
    return doc.model_dump_json()


@mcp.resource("mnemomatic://knowledge-entry/{id}")
def get_knowledge_entry(id: str) -> str:
    """Get a specific knowledge entry by ID."""
    k = _db().get_knowledge(id)
    if not k:
        return json.dumps({"error": f"Knowledge {id} not found"})
    return k.model_dump_json()


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Mnem-O-matic MCP server")
    logger.info("Configuration: db_path=%s, host=%s, port=%s", DB_PATH, HOST, PORT)

    # Pre-warm db and resolve embedder so the first request doesn't pay setup costs
    logger.info("Initializing database...")
    _db()
    logger.info("Initializing embedder...")
    _embedder()

    # Always use unified ASGI app + Uvicorn code path
    # Authentication is optional based on API_KEY environment variable
    logger.info("Building ASGI application...")
    app = mcp.streamable_http_app()

    # Middleware handles both authenticated and non-authenticated modes
    # If API_KEY is empty, auth is disabled but logging still tracks requests
    app = BearerAuthMiddleware(app, api_key=API_KEY)

    logger.info("Starting server on %s:%d", HOST, PORT)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
