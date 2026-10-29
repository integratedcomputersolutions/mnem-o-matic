"""Re-embedding the whole store: rebuild the vector index at the configured dimension and embed
every document, knowledge entry and note again.

Two ways in. At startup, under MNEMOMATIC_REINDEX (``auto`` when the embedder changed, ``1``
always), before the server answers. And by an administrator from the Settings page
(``POST /api/admin/reindex``), in a thread while the server keeps serving: semantic search
answers from whatever is re-embedded so far, fulltext search is never affected.

Before an administrator's run empties the index, the embedder is asked for one vector. An
embedder that doesn't answer (a wrong address, a refused token) gets the request refused and
leaves the index as it is: emptying it first and stamping the new identity on it would make an
empty index look up to date, with nothing left to notice that semantic search returns nothing.
The startup run is unchanged by this and behaves as MNEMOMATIC_REINDEX always has.

Content tables are never modified (timestamps included); only vectors and document chunks are
recomputed. Items whose embedding fails are logged and left fulltext-only.
"""

import logging
import threading
import time
from datetime import datetime, timezone

from mnemomatic import config, db as db_module, runtime
from mnemomatic.audit import write_event
from mnemomatic.db import _SPECS

logger = logging.getLogger("mnemomatic")

PROBE_TEXT = "Mnem-O-matic re-index probe"


class ReindexBusy(RuntimeError):
    """A re-index is already running."""


class EmbedderUnavailable(RuntimeError):
    """No embedder, or one that doesn't answer: the index is left as it is."""


_lock = threading.Lock()
_state: dict = {
    "running": False,
    "trigger": None,       # "startup" or "admin"
    "actor": None,         # the administrator who asked (admin runs)
    "started_at": None,
    "finished_at": None,
    "done": 0,             # items re-embedded or failed so far
    "total": 0,            # items to re-embed
    "counts": None,        # the last run's {"documents", "knowledge", "notes", "failed"}
    "error": None,         # why the last run stopped before the end
}


def status() -> dict:
    """What the Settings page shows: whether a run is on, its progress, the last run's outcome."""
    with _lock:
        return dict(_state)


def probe() -> None:
    """Ask the embedder for one vector; EmbedderUnavailable when there is none or it fails."""
    embedder = runtime._embedder()
    if embedder is None:
        raise EmbedderUnavailable("No embedder is configured: nothing can re-embed the store.")
    try:
        embedder.embed(config.EMBED_DOC_PREFIX + PROBE_TEXT)
    except Exception as e:  # noqa: BLE001 — any failure means the same: don't empty the index
        raise EmbedderUnavailable(f"The embedder didn't answer ({type(e).__name__}: {e}); "
                                  "the index is left as it is.") from e


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(*, trigger: str, actor: str | None = None) -> dict:
    """Re-embed everything, synchronously; the counts at the end.

    ReindexBusy when a run is on. Any error stops the run and is recorded in the status.
    """
    with _lock:
        if _state["running"]:
            raise ReindexBusy("A re-index is already running.")
        _state.update(running=True, trigger=trigger, actor=actor, started_at=_now(), finished_at=None,
                      done=0, total=0, counts=None, error=None)
    try:
        counts = _run()
    except Exception as e:
        with _lock:
            _state.update(running=False, finished_at=_now(), error=f"{type(e).__name__}: {e}")
        raise
    with _lock:
        _state.update(running=False, finished_at=_now(), counts=counts)
    return counts


def _run() -> dict:
    database = runtime._db()
    dim = db_module.EMBEDDING_DIM
    items = [(spec, item) for namespace in database.list_namespaces() for spec in _SPECS.values()
             for item in database.list_items(spec.item_type, namespace)]
    with _lock:
        _state["total"] = len(items)
    if config.REINDEX_MODE == "force" and _state["trigger"] == "startup":
        logger.warning(
            "Reindex starting (MNEMOMATIC_REINDEX=1): rebuilding vector index at dim %d and "
            "re-embedding all content. Remove the flag after this run — it re-embeds on every "
            "startup while set. MNEMOMATIC_REINDEX=auto re-embeds only when the embedder "
            "actually changes, and is safe to leave set.", dim)
    elif _state["trigger"] == "startup":
        logger.warning("Embedder changed — reindexing automatically (MNEMOMATIC_REINDEX=auto): rebuilding "
                       "vector index at dim %d and re-embedding all content before serving.", dim)
    else:
        logger.warning("Reindex started by %s: rebuilding vector index at dim %d and re-embedding "
                       "%d items while serving.", _state["actor"] or "an administrator", dim, len(items))
    database.rebuild_vec_tables()

    counts = {**dict.fromkeys(_SPECS, 0), "failed": 0}
    for spec, item in items:
        embedding, chunks = runtime._embed_item(spec.item_type, item)
        if spec.item_type == "document":
            database.replace_document_chunks(item.id, chunks)
        # A chunked document is embedded through its chunks alone.
        if embedding is not None:
            ok = database.set_embedding(spec.item_type, item.id, embedding)
        else:
            ok = chunks is not None
        if ok:
            counts[spec.table] += 1
        else:
            counts["failed"] += 1
            logger.error("Reindex: embedding failed for %s %s (%r)", spec.item_type, item.id,
                         getattr(item, spec.title_field))
        with _lock:
            _state["done"] += 1

    logger.info("Reindex complete: %d documents, %d knowledge, %d notes re-embedded, %d failed",
                counts["documents"], counts["knowledge"], counts["notes"], counts["failed"])
    # A whole-store re-embed is worth a trail entry: it explains why every item's vector changed
    # at once; at startup nobody typed a command, from the Settings page the actor is named.
    write_event(database, "reindex", actor=_state["actor"],
                mode=config.REINDEX_MODE if _state["trigger"] == "startup" else "admin",
                dim=dim, model=config.embed_identity()["embed_model"] or None, **counts)
    return counts


def start(*, actor: str) -> None:
    """Run a re-index in a thread for an administrator; the status reports its progress.
    ReindexBusy or EmbedderUnavailable at once, before the thread starts."""
    with _lock:
        if _state["running"]:
            raise ReindexBusy("A re-index is already running.")
    probe()

    def worker() -> None:
        try:
            run(trigger="admin", actor=actor)
        except Exception as e:  # noqa: BLE001 — recorded in the status, nothing else to tell
            logger.error("Reindex stopped: %s: %s", type(e).__name__, e)

    threading.Thread(target=worker, name="mnemomatic-reindex", daemon=True).start()
    # Let the worker take the lock before the caller reads the status it answers with.
    for _ in range(50):
        if _state["running"]:
            break
        time.sleep(0.01)
