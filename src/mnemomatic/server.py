"""Server entry point: startup, the reindex pass, and ASGI assembly.

The MCP surface itself lives in the tools_* modules; importing them here is
what registers their tools, resources, and prompts on the shared app. Import
order is the registration order clients see, so it is kept stable.
"""

import logging

import uvicorn

from mnemomatic import config, runtime
from mnemomatic.audit import RequestMetaMiddleware
from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.compact import CompactToolsMiddleware
from mnemomatic.db import EMBEDDING_DIM
from mnemomatic.runtime import (
    _audit,
    _embed_content,
    _embed_document_body,
    _knowledge_embed_text,
    _note_embed_text,
    mcp,
)

# Imported for their registration side effects; the order below is the order
# tools appear to a client.
from mnemomatic import tools_content, tools_search, tools_history, tools_admin  # noqa: F401,E402
from mnemomatic.tools_admin import _export_route, _make_export, _server_version, _settings_info

logger = logging.getLogger("mnemomatic")


def _run_reindex() -> None:
    """Rebuild the vector index and re-embed every stored item.

    Runs at startup after a change of embedding model, dimension, or task
    prefixes — triggered by the recorded identity under MNEMOMATIC_REINDEX=auto,
    or unconditionally under =1. Content tables are never modified (timestamps
    included); only vectors and document chunks are recomputed. Items whose
    embedding fails are logged and left FTS-only.
    """
    database = runtime._db()
    if runtime._embedder() is None:
        if database.reindex_pending:
            # Never drop vectors that cannot be rebuilt: the rebuild empties the
            # index first, so without an embedder this would leave the store with
            # no semantic search at all and no way back.
            raise RuntimeError(
                "The configured embedder differs from the one that built this index, but no "
                "embedder is available to rebuild it. Configure an embedder, or restore the "
                "previous embedding settings to keep the existing index."
            )
        logger.error("MNEMOMATIC_REINDEX is set but no embedder is available — skipping reindex")
        return

    if config.REINDEX_MODE == "force":
        logger.warning(
            "Reindex starting (MNEMOMATIC_REINDEX=1): rebuilding vector index at dim %d and "
            "re-embedding all content. Remove the flag after this run — it re-embeds on every "
            "startup while set. MNEMOMATIC_REINDEX=auto re-embeds only when the embedder "
            "actually changes, and is safe to leave set.",
            EMBEDDING_DIM,
        )
    else:
        logger.warning(
            "Embedder changed — reindexing automatically (MNEMOMATIC_REINDEX=auto): rebuilding "
            "vector index at dim %d and re-embedding all content before serving.",
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
    # A whole-store re-embed is worth a trail entry: it explains why every
    # item's vector changed at once, and under =auto nobody typed a command.
    _audit("reindex", mode=config.REINDEX_MODE, dim=EMBEDDING_DIM,
           model=config.embed_identity()["embed_model"] or None, **counts)


# ── Tools ──


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Mnem-O-matic MCP server")
    logger.info("Configuration: db_path=%s, host=%s, port=%s", config.DB_PATH, config.HOST, config.PORT)

    # Pre-warm db and resolve embedder so the first request doesn't pay setup costs
    logger.info("Initializing database...")
    runtime._db()
    logger.info("Initializing embedder...")
    runtime._embedder()

    # Opt-in full re-embed (model/dim/prefix changes) before serving traffic.
    # Under "auto" the database has already compared the recorded embedding
    # identity against the configured one, so reindex_pending is the whole
    # question: nothing changed, nothing to do.
    if config.REINDEX_MODE == "force" or (config.REINDEX_MODE == "auto" and runtime._db().reindex_pending):
        _run_reindex()
    elif config.REINDEX_MODE == "auto":
        logger.info("Embedder matches the stored index — no reindex needed")

    # Scheduled backups on a daemon thread (the Database hands each thread its
    # own connection, so the loop reads safely alongside request handling).
    if config.BACKUP_DIR:
        from pathlib import Path

        from mnemomatic.backup import start_backup_thread
        start_backup_thread(runtime._db, Path(config.BACKUP_DIR), interval_hours=config.BACKUP_INTERVAL_HOURS,
                            keep=config.BACKUP_KEEP, server_version=_server_version())
        logger.info("Scheduled backups: every %gh to %s (keeping %d)",
                    config.BACKUP_INTERVAL_HOURS, config.BACKUP_DIR, config.BACKUP_KEEP)

    # Always use unified ASGI app + Uvicorn code path
    # Authentication is optional based on config.API_KEY environment variable
    logger.info("Building ASGI application...")
    app = mcp.streamable_http_app()

    # Zip export download. Inserted ahead of the MCP catch-all; NOT exempt
    # from Bearer auth — it returns the entire store.
    from starlette.routing import Route
    app.router.routes.insert(0, Route("/export", _export_route, methods=["GET"]))

    # Optional read-only web viewer at /ui, gated by a single shared secret.
    # Disabled unless MNEMOMATIC_UI_TOKEN is set, so it never exposes data by default.
    if config.UI_TOKEN:
        from mnemomatic.webui import register_webui
        register_webui(app, runtime._db, config.UI_TOKEN, settings_info=_settings_info, make_export=_make_export)
        logger.info("Web viewer enabled at /ui")
    else:
        logger.info("Web viewer disabled (set MNEMOMATIC_UI_TOKEN to enable)")

    app = CompactToolsMiddleware(app)

    # Capture actor/client/ip per request for the audit log.
    app = RequestMetaMiddleware(app)

    # Middleware handles both authenticated and non-authenticated modes
    # If config.API_KEY is empty, auth is disabled but logging still tracks requests.
    # /ui is exempt from Bearer auth only when the viewer is actually registered.
    app = BearerAuthMiddleware(app, api_key=config.API_KEY, exempt_ui=bool(config.UI_TOKEN))

    if config.CORS_ORIGINS:
        from starlette.middleware.cors import CORSMiddleware
        origins = [o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()]
        if "*" in origins and not config.API_KEY:
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

    logger.info("Starting server on %s:%d", config.HOST, config.PORT)
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
