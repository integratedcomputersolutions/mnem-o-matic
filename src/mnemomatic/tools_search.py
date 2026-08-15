"""Recall tools: search, read, related, and the browsing resources."""
import json
import logging
import sqlite3
from datetime import datetime

from mnemomatic import config, runtime
from mnemomatic.db import _SPEC_BY_ITEM_TYPE
from mnemomatic.runtime import (
    _embed_query,
    _escape_fts_query,
    _record_access,
    mcp,
)
from mnemomatic.tools_content import _OPS

logger = logging.getLogger("mnemomatic")


@mcp.tool(annotations=config.ANN_READ_ONLY)
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

    limit = max(1, min(int(limit), config.MAX_SEARCH_LIMIT))

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
    emb = runtime._embedder()
    degraded = False

    if mode == "semantic" and emb is None:
        return [{"error": "Semantic search not available",
                 "details": "No embedder configured. Set MNEMOMATIC_EMBED_URL or use the full image with the built-in model."}]

    try:
        # hybrid silently degrades to fulltext when no embedder is available
        if mode == "fulltext" or (mode == "hybrid" and emb is None):
            results = runtime._db().search_fts(fts_query, table=table, namespace=namespace, limit=limit, **filters)
            if mode == "hybrid" and emb is None:
                degraded = True
        elif mode == "semantic":
            embedding = _embed_query(query)
            if embedding is None:
                return [{"error": "Semantic search failed", "details": "Embedding service is unavailable. Try fulltext mode."}]
            results = runtime._db().search_vec(embedding, table=table, namespace=namespace, limit=limit, **filters)
        else:  # hybrid with embedder
            embedding = _embed_query(query)
            # If embedding fails, degrade to fulltext search
            if embedding is None:
                logger.info("Hybrid search degrading to fulltext due to embedding failure")
                results = runtime._db().search_fts(fts_query, table=table, namespace=namespace, limit=limit, **filters)
                degraded = True
            else:
                results = runtime._db().search_hybrid(fts_query, embedding, table=table, namespace=namespace, limit=limit, **filters)
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


def _get_resource(item_type: str, id: str) -> str:
    """Shared body for the get_* MCP resources: fetch by id, return JSON or a not-found error."""
    obj = _OPS[item_type].get(runtime._db(), id)
    if obj is None:
        return json.dumps({"error": f"{item_type.capitalize()} {id} not found"})
    _record_access([(item_type, id)])
    return obj.model_dump_json()


@mcp.tool(annotations=config.ANN_READ_ONLY)
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
        limit = max(1, min(int(limit), config.MAX_LIST_LIMIT))
        offset = max(0, int(offset))
        items, total = runtime._db().list_page(item_type, namespace, limit, offset)
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


@mcp.tool(annotations=config.ANN_READ_ONLY)
def read(item_type: str, id: str) -> dict:
    """Read the full content of a document, knowledge entry, or note by ID.

    Use this after searching to retrieve complete content — search results only
    contain a snippet. The item's resource_uri field tells you the type and ID.

    Args:
        item_type: The item type — "document", "knowledge", or "note".
        id: The unique item ID (UUID returned by store/search).
    """
    ops = _OPS.get(item_type)
    if ops is None:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_OPS))}"}
    item = ops.get(runtime._db(), id)
    if item is None:
        return {"error": f"{item_type} not found", "id": id}
    _record_access([(item_type, id)])
    return json.loads(item.model_dump_json())


@mcp.tool(annotations=config.ANN_READ_ONLY)
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
    if item_type not in _OPS:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_OPS))}"}
    limit = max(1, min(int(limit), config.MAX_SEARCH_LIMIT))

    if _OPS[item_type].get(runtime._db(), id) is None:
        return {"error": f"{item_type} not found", "id": id}
    embedding = runtime._db().item_embedding(item_type, id)
    if embedding is None:
        return {"error": "No embedding for this item",
                "details": "The item has no stored vector (FTS-only mode, or stored before "
                           "an embedder was configured — a MNEMOMATIC_REINDEX=1 restart embeds it)."}

    try:
        results = runtime._db().search_vec(embedding, table="all", namespace=namespace, limit=limit + 1)
    except sqlite3.Error as e:
        logger.warning("Related-items search failed for %s %s: %s", item_type, id, e)
        return {"error": "Related search failed", "details": str(e)}
    neighbors = [r for r in results if r.id != id][:limit]
    _record_access([(r.type, r.id) for r in neighbors])
    return {"item_type": item_type, "item_id": id,
            "related": [r.model_dump() for r in neighbors]}


@mcp.resource("mnemomatic://namespaces")
def list_namespaces() -> str:
    """List all namespaces in Mnem-O-matic."""
    namespaces = runtime._db().list_namespaces()
    return json.dumps(namespaces)


def _json_scalar(value):
    """Datetimes serialize as ISO-8601; everything else is already JSON-safe."""
    return value.isoformat() if isinstance(value, datetime) else value


def _list_resource(item_type: str, namespace: str) -> str:
    """Shared body for the per-namespace list resources: summaries only, with
    the projection taken from the table spec so it tracks the schema."""
    fields = _SPEC_BY_ITEM_TYPE[item_type].resource_fields
    items = _OPS[item_type].list(runtime._db(), namespace)
    return json.dumps([
        {f: _json_scalar(getattr(item, f)) for f in fields} for item in items
    ])


@mcp.resource("mnemomatic://documents/{namespace}")
def list_documents(namespace: str) -> str:
    """List all documents in a namespace."""
    return _list_resource("document", namespace)


@mcp.resource("mnemomatic://knowledge/{namespace}")
def list_knowledge(namespace: str) -> str:
    """List all knowledge entries in a namespace."""
    return _list_resource("knowledge", namespace)


@mcp.resource("mnemomatic://notes/{namespace}")
def list_notes(namespace: str) -> str:
    """List all notes in a namespace."""
    return _list_resource("note", namespace)


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
