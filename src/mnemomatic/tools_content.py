"""Content tools: storing, updating, deleting, and tagging items.

The three item types share their update and delete bodies through
_handle_update/_handle_delete; only genuinely per-type behaviour lives in
_OPS. Each MCP tool stays its own function because its name, signature,
and docstring are the agent-facing API.
"""
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from mnemomatic import config, runtime
from mnemomatic.db import _SPEC_BY_ITEM_TYPE, CHUNK_THRESHOLD, Database
from mnemomatic.models import Document, Knowledge, Note
from mnemomatic.runtime import (
    _audit,
    _embed_content,
    _embed_document_body,
    _format_validation_error,
    _knowledge_embed_text,
    _note_embed_text,
    _similar_items,
    mcp,
)


@mcp.tool(annotations=config.ANN_STORE)
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

    stored, created = runtime._db().store_document(doc, embedding, chunks)
    response = {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}
    similar = _similar_items("documents", stored.id, namespace, embedding)
    if similar:
        response["similar"] = similar
    _audit("store", item_type="document", item_id=stored.id, namespace=stored.namespace,
           title=stored.title, created=created)
    return response


@mcp.tool(annotations=config.ANN_STORE)
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
    stored, created, superseded = runtime._db().store_knowledge(k, embedding)
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
        runtime._db().replace_document_chunks(id, chunks)
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


@dataclass(frozen=True)
class _ItemOps:
    """The per-type database calls the tools dispatch through.

    Only genuinely per-type behaviour lives here. Everything descriptive —
    model, title field, resource shape — comes from the table spec in db.py,
    so the two never disagree about what a "note" is.
    """

    get: Callable                # (db, id) -> item | None
    delete: Callable             # (db, id) -> bool
    list: Callable               # (db, namespace) -> [item]
    update: Callable             # (db, id, embedding, fields) -> item | None
    embed_update: Callable       # (id, existing, fields) -> embedding | None


_OPS: dict[str, _ItemOps] = {
    "document": _ItemOps(
        get=Database.get_document,
        delete=Database.delete_document,
        list=Database.list_documents,
        update=lambda db, id, emb, fields: db.update_document(id, embedding=emb, **fields),
        embed_update=_document_update_embedding,
    ),
    "knowledge": _ItemOps(
        get=Database.get_knowledge,
        delete=Database.delete_knowledge,
        list=Database.list_knowledge,
        update=lambda db, id, emb, fields: db.update_knowledge(id, embedding=emb, **fields),
        embed_update=_knowledge_update_embedding,
    ),
    "note": _ItemOps(
        get=Database.get_note,
        delete=Database.delete_note,
        list=Database.list_notes,
        update=lambda db, id, emb, fields: db.update_note(id, embedding=emb, **fields),
        embed_update=_note_update_embedding,
    ),
}


def _handle_update(item_type: str, id: str, fields: dict) -> dict:
    """Shared body for the update_* tools: validate the merged item, recompute its embedding,
    persist, and return {id, <key>, updated}.

    Knowledge is temporal: a superseded entry is immutable history, and a
    change to the fact itself supersedes (closes the current entry and inserts
    a successor) instead of overwriting.
    """
    ops, spec = _OPS[item_type], _SPEC_BY_ITEM_TYPE[item_type]
    db = runtime._db()
    existing = ops.get(db, id)
    if existing is None:
        return {"error": f"{item_type.capitalize()} {id} not found"}
    if item_type == "knowledge" and existing.valid_until is not None:
        return {"error": "Cannot update a superseded fact",
                "details": f"Knowledge {id} is history (superseded by {existing.superseded_by}). "
                           f"Update the current fact for this subject instead, or use fact_history to inspect it."}

    try:
        spec.model(**{**existing.model_dump(), **fields})
    except ValidationError as e:
        return {"error": "Invalid update", "details": _format_validation_error(e)}

    if item_type == "knowledge" and "fact" in fields and fields["fact"] != existing.fact:
        return _supersede_update(existing, fields)

    embedding = ops.embed_update(id, existing, fields)
    updated = ops.update(db, id, embedding, fields)
    if updated is None:
        return {"error": f"{item_type.capitalize()} {id} not found"}
    key = spec.title_field
    _audit("update", item_type=item_type, item_id=updated.id, namespace=updated.namespace,
           title=getattr(updated, key), fields=sorted(fields))
    return {"id": updated.id, key: getattr(updated, key), "updated": True}


def _handle_delete(item_type: str, id: str) -> dict:
    """Shared body for the delete_* tools.

    The item is read before it goes, so the audit event can still name the
    namespace and title it had — the row is gone by the time the event is
    written. A delete that removed nothing is not an event.
    """
    db = runtime._db()
    existing = _OPS[item_type].get(db, id)
    deleted = _OPS[item_type].delete(db, id)
    if deleted:
        title_field = _SPEC_BY_ITEM_TYPE[item_type].title_field
        _audit("delete", item_type=item_type, item_id=id,
               namespace=existing.namespace if existing else None,
               title=getattr(existing, title_field, None) if existing else None)
    return {"id": id, "deleted": deleted}


def _supersede_update(existing: Knowledge, fields: dict) -> dict:
    """A fact change: build the successor entry and close the current one."""
    data = {**existing.model_dump(), **fields}
    for reset in ("id", "created_at", "updated_at", "valid_until", "superseded_by",
                  "retrieval_count", "last_accessed"):
        data.pop(reset, None)
    successor = Knowledge(**data)  # fresh id and timestamps; merged fields pre-validated
    embedding = _embed_content(_knowledge_embed_text(successor.subject, successor.fact))
    try:
        stored = runtime._db().supersede_knowledge(existing.id, successor, embedding)
    except sqlite3.IntegrityError:
        return {"error": "Subject conflict",
                "details": f"Another current fact already holds subject {successor.subject!r} "
                           f"in namespace {successor.namespace!r}"}
    if stored is None:
        return {"error": f"Knowledge {existing.id} not found"}
    _audit("supersede", item_type="knowledge", item_id=stored.id, namespace=stored.namespace,
           title=stored.subject, superseded=existing.id)
    return {"id": stored.id, "subject": stored.subject, "updated": True, "superseded": existing.id}


@mcp.tool(annotations=config.ANN_UPDATE)
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


@mcp.tool(annotations=config.ANN_UPDATE)
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


@mcp.tool(annotations=config.ANN_DELETE)
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
    return _handle_delete("document", id)


@mcp.tool(annotations=config.ANN_DELETE)
def delete_knowledge(id: str) -> dict:
    """Delete a knowledge entry from Mnem-O-matic.

    Use when a fact was stored by mistake or should never have existed. If the
    fact simply changed, do NOT delete — store or update the corrected fact and
    the old one is kept as queryable history (see fact_history). A mistaken
    delete can be undone via list_revisions + restore.

    Args:
        id: The knowledge entry ID to delete.
    """
    return _handle_delete("knowledge", id)


@mcp.tool(annotations=config.ANN_STORE)
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
    stored, created = runtime._db().store_note(note, embedding)
    response = {"id": stored.id, "namespace": stored.namespace, "title": stored.title, "created": created}
    similar = _similar_items("notes", stored.id, namespace, embedding)
    if similar:
        response["similar"] = similar
    _audit("store", item_type="note", item_id=stored.id, namespace=stored.namespace,
           title=stored.title, created=created)
    return response


@mcp.tool(annotations=config.ANN_UPDATE)
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


@mcp.tool(annotations=config.ANN_DELETE)
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
    return _handle_delete("note", id)


@mcp.tool(annotations=config.ANN_TAG)
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
        tags = runtime._db().update_tags(item_id, item_type, add_tags=add_tags, remove_tags=remove_tags)
        _audit("tag", item_type=item_type, item_id=item_id,
               added=add_tags or [], removed=remove_tags or [])
        return {"id": item_id, "tags": tags}
    except ValueError as e:
        return {"error": str(e)}
