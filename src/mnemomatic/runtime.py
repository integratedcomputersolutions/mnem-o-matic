"""The server's shared runtime: the MCP app, its singletons, and embedding.

Everything the tool modules depend on but none of them owns — the FastMCP
instance they register against, the lazily created Database and embedder, the
embedding helpers, and the audit/usage bookkeeping.

Tests replace `_db`, `_embedder`, `_safe_embed`, and `_safe_embed_batch`, so
callers must reach those through this module (`runtime._db()`) rather than
importing them by name. A name imported at module load is a second binding
that patching cannot reach, which would let a test pass while exercising the
real database. Everything else here is a pure helper and is safe to import
directly.
"""

import logging
import os
import re
import threading

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from mnemomatic import config
from mnemomatic.audit import request_meta
from mnemomatic.db import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_THRESHOLD,
    Database,
    _chunk_text,
)

logger = logging.getLogger("mnemomatic")

mcp = FastMCP(
    "Mnem-O-matic",
    json_response=True,
    host=config.HOST,
    port=config.PORT,
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
                os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
                db = Database(
                    config.DB_PATH, allow_reindex=config.REINDEX, embed_identity=config.embed_identity()
                )
    return db


def _resolve_embedder():
    """Initialize and return the appropriate embedder, or None for FTS-only mode."""
    if config.EMBED_URL:
        try:
            from mnemomatic.embeddings import HttpEmbedder
            embedder = HttpEmbedder(config.EMBED_URL, config.EMBED_MODEL)
            logger.info("Embedder: %s endpoint %s (model=%r)", embedder.mode, config.EMBED_URL, config.EMBED_MODEL)
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
    return _safe_embed(config.EMBED_QUERY_PREFIX + text)


def _embed_content(text: str) -> list[float] | None:
    """Embedding for stored content, with the configured document prefix applied."""
    return _safe_embed(config.EMBED_DOC_PREFIX + text)


def _similar_items(table: str, item_id: str, namespace: str,
                   embedding: list[float] | None) -> list[dict]:
    """Near-duplicates of a just-stored item, for the agent mid-write to judge.

    The server only flags — merging, superseding, or ignoring is the caller's
    decision. Empty when there is nothing above config.SIMILAR_THRESHOLD, no
    embedding (FTS-only mode, chunked documents), or the check is disabled.
    Never breaks the store that triggered it.
    """
    if embedding is None or config.SIMILAR_THRESHOLD <= 0:
        return []
    try:
        results = _db().search_vec(embedding, table=table, namespace=namespace,
                                   limit=config.SIMILAR_LIMIT + 1)
    except Exception as e:
        logger.warning("Similar-item check failed: %s: %s", type(e).__name__, e)
        return []
    return [
        {"id": r.id, "title": r.title, "score": round(r.score, 3)}
        for r in results if r.id != item_id and r.score >= config.SIMILAR_THRESHOLD
    ][:config.SIMILAR_LIMIT]


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
        embeddings = _safe_embed_batch([config.EMBED_DOC_PREFIX + t for t in texts])
        chunks = [(c, e) for c, e in zip(texts, embeddings) if e is not None]
        return None, (chunks or None)
    return _embed_content(f"{title}\n{content}"), None


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
