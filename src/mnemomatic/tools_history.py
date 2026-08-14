"""History and hygiene: revisions, restore, fact history, the audit trail,
the consolidation report, and the maintenance prompts."""
import json
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from mnemomatic import config, runtime
from mnemomatic.db import _SPEC_BY_ITEM_TYPE, _SPECS
from mnemomatic.runtime import (
    _audit,
    _embed_content,
    _embed_document_body,
    _format_validation_error,
    _knowledge_embed_text,
    _note_embed_text,
    _record_access,
    mcp,
)
from mnemomatic.tools_content import _OPS, _handle_update


@mcp.tool(annotations=config.ANN_READ_ONLY)
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
    history = runtime._db().knowledge_history(namespace, subject)
    _record_access([("knowledge", k.id) for k in history])
    return {
        "namespace": namespace,
        "subject": subject,
        "count": len(history),
        "history": [json.loads(k.model_dump_json()) for k in history],
    }


@mcp.tool(annotations=config.ANN_READ_ONLY)
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
    if item_type is not None and item_type not in _OPS:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_OPS))}"}
    limit = max(1, min(int(limit), config.MAX_LIST_LIMIT))
    revisions = runtime._db().list_revisions(item_type=item_type, item_id=item_id,
                                     namespace=namespace, limit=limit)
    return {"revisions": revisions, "limit": limit}


@mcp.tool(annotations=config.ANN_READ_ONLY)
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
    rename_namespace/delete_namespace, plus reindex when the whole store was
    re-embedded), item_type/item_id/namespace/title,
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
    if item_type is not None and item_type not in _OPS:
        return {"error": "Invalid item_type", "details": f"Must be one of: {', '.join(sorted(_OPS))}"}
    limit = max(1, min(int(limit), config.MAX_LIST_LIMIT))
    events = runtime._db().list_audit(item_type=item_type, item_id=item_id,
                              namespace=namespace, op=op, limit=limit)
    return {"events": events, "limit": limit}


@mcp.tool(annotations=config.ANN_UPDATE)
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
        rev = runtime._db().get_revision(revision_id)
    except ValidationError as e:
        return {"error": "Revision payload no longer validates", "details": _format_validation_error(e)}
    if rev is None:
        return {"error": f"Revision {revision_id} not found"}

    item_type, item = rev["item_type"], rev["item"]
    key = _SPEC_BY_ITEM_TYPE[item_type].title_field

    if _OPS[item_type].get(runtime._db(), rev["item_id"]) is not None:
        # Roll the live item back through the normal update path — it captures
        # the current state as a revision and re-embeds what changed.
        fields = {f: getattr(item, f) for f in _SPEC_BY_ITEM_TYPE[item_type].update_fields}
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
    occupant = runtime._db().find_by_key(item_type, item.namespace, getattr(item, key))
    if occupant is not None:
        return {"error": "Cannot restore: key is taken",
                "details": f"{item_type} {occupant} now occupies "
                           f"{item.namespace!r}/{getattr(item, key)!r} — delete or rename it first"}

    item = item.model_copy(update={"updated_at": datetime.now(timezone.utc)})
    if item_type == "document":
        embedding, chunks = _embed_document_body(item.title, item.content)
        stored, _ = runtime._db().store_document(item, embedding, chunks)
    elif item_type == "knowledge":
        stored, _, _ = runtime._db().store_knowledge(item, _embed_content(_knowledge_embed_text(item.subject, item.fact)))
    else:
        stored, _ = runtime._db().store_note(item, _embed_content(_note_embed_text(item.title, item.content)))
    _audit("restore", item_type=item_type, item_id=stored.id, namespace=stored.namespace,
           title=getattr(stored, key), revision_id=revision_id, recreated=True)
    return {"id": stored.id, key: getattr(stored, key), "namespace": stored.namespace,
            "restored_revision": revision_id, "recreated": True}


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


@mcp.tool(annotations=config.ANN_READ_ONLY)
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
    threshold = config.SIMILAR_THRESHOLD if similarity_threshold is None else float(similarity_threshold)
    if threshold <= 0:
        return {"error": "Invalid similarity_threshold", "details": "Must be positive (cosine similarity)"}

    clusters = []
    for table, spec in _SPECS.items():
        vectors = runtime._db().item_vectors(table, namespace)
        clusters.extend(_duplicate_clusters(spec.item_type, vectors, threshold))
    clusters.sort(key=lambda c: c["similarity"], reverse=True)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(0, int(stale_days)))).isoformat()
    stale = runtime._db().stale_items(namespace, cutoff)

    return {
        "namespace": namespace,
        "similarity_threshold": threshold,
        "stale_days": stale_days,
        "duplicate_clusters": clusters,
        "stale": stale,
        "counts": runtime._db().namespace_counts().get(namespace, {}),
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
