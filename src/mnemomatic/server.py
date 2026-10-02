"""Server entry point: startup, the reindex pass, and ASGI assembly.

The MCP surface itself lives in the tools_* modules; importing them here is
what registers their tools, resources, and prompts on the shared app. Import
order is the registration order clients see, so it is kept stable.
"""

# ruff: noqa: I001 — the tools_* imports below are in tool-registration order,
# not alphabetical, and sorting this file's imports would change the published
# tool list. See the comment above them.

import asyncio
import contextlib
import logging
import signal
from pathlib import Path

import uvicorn

from mnemomatic import config, runtime
from mnemomatic.api import SecurityHeadersMiddleware, build_api_routes
from mnemomatic.audit import RequestMetaMiddleware
from mnemomatic.auth import AuthMiddleware
from mnemomatic.bodylimit import BodyLimitMiddleware
from mnemomatic.compact import CompactToolsMiddleware
from mnemomatic.db import _SPECS, EMBEDDING_DIM
from mnemomatic.spa import APP_DIR, ListenerTag, PlainPortGate, build_spa_routes, build_tls_routes
from mnemomatic.tlsca import TlsState, start_renewal_thread
from mnemomatic.identity import IdentityError, ensure_bootstrap
from mnemomatic.runtime import (
    _audit,
    _embed_item,
    mcp,
)

# Imported for their registration side effects. The order below is the order
# tools appear to a client, so it must not be alphabetised — an import sorter
# would silently rearrange the published tool list. tests/test_tool_registration.py
# pins the result so that change cannot pass unnoticed.
from mnemomatic import tools_content  # noqa: F401
from mnemomatic import tools_search   # noqa: F401
from mnemomatic import tools_history  # noqa: F401
from mnemomatic import tools_admin    # noqa: F401
from mnemomatic.tools_admin import (
    _export_route,
    _health_route,
    _server_version,
    _settings_info,
)

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

    counts = {**dict.fromkeys(_SPECS, 0), "failed": 0}
    for namespace in database.list_namespaces():
        for spec in _SPECS.values():
            for item in database.list_items(spec.item_type, namespace):
                embedding, chunks = _embed_item(spec.item_type, item)
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

    logger.info(
        "Reindex complete: %d documents, %d knowledge, %d notes re-embedded, %d failed",
        counts["documents"], counts["knowledge"], counts["notes"], counts["failed"],
    )
    # A whole-store re-embed is worth a trail entry: it explains why every
    # item's vector changed at once, and under =auto nobody typed a command.
    _audit("reindex", mode=config.REINDEX_MODE, dim=EMBEDDING_DIM,
           model=config.embed_identity()["embed_model"] or None, **counts)


# ── Tools ──


class _Server(uvicorn.Server):
    """A uvicorn Server that leaves signal handling to the caller, so two of
    them can share one event loop (plain and TLS) without the second one
    stealing the first's SIGTERM/SIGINT handlers."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def _serve_all(app, tls: TlsState | None) -> None:
    """Run the plain listener, and the TLS listener whenever a certificate
    exists — at boot, or the moment an admin names the host.

    The TLS server runs with lifespan="off": the application's lifespan (the
    MCP session manager) may only be entered once, and the plain server owns
    it. Both listeners serve the same app; a per-listener tag lets the plain
    port gate know which side a request came in on.
    """
    loop = asyncio.get_running_loop()
    servers: list[_Server] = []
    tasks: list[asyncio.Task] = []
    common = dict(host=config.HOST, log_level="info",
                  proxy_headers=bool(config.TRUSTED_PROXIES), forwarded_allow_ips=config.TRUSTED_PROXIES)

    plain = _Server(uvicorn.Config(ListenerTag(app, "plain"), port=config.PORT, lifespan="on", **common))
    servers.append(plain)
    tasks.append(loop.create_task(plain.serve()))

    tls_started = False

    async def run_tls(server: _Server) -> None:
        try:
            await server.serve()
        except SystemExit:
            logger.error("HTTPS listener on port %d could not start (port in use?)", config.HTTPS_PORT)
        except Exception as e:
            logger.error("HTTPS listener failed: %s: %s", type(e).__name__, e)

    def start_tls() -> None:
        nonlocal tls_started
        if tls is None or tls_started or not tls.holder.ready:
            return
        tls_started = True
        server = _Server(uvicorn.Config(ListenerTag(app, "tls"), port=config.HTTPS_PORT, lifespan="off",
                                        ssl_context_factory=tls.holder.factory, **common))
        servers.append(server)
        tasks.append(loop.create_task(run_tls(server)))
        logger.info("HTTPS listening on %s:%d", config.HOST, config.HTTPS_PORT)

    if tls is not None:
        tls.on_ready = lambda: loop.call_soon_threadsafe(start_tls)
        start_tls()

    def stop() -> None:
        for s in servers:
            s.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:  # not the main thread, or Windows
            pass

    await tasks[0]
    stop()
    for task in tasks[1:]:
        await task


def build_app(tls: TlsState | None, app_dir: Path = APP_DIR):
    """Assemble the ASGI application: the MCP endpoint, the plain HTTP routes,
    the browser API, the setup and CA routes, the single-page app, and the
    middleware around them — innermost first.

    Pure assembly, no startup work: the database, embedder, bootstrap and TLS
    state are prepared by main() (or a test) and handed in, which is what lets
    a test build the real stack in-process.
    """
    app = mcp.streamable_http_app()

    # Both inserted ahead of the MCP catch-all. /export returns the entire
    # store and takes a session or a token; /health takes nothing (see
    # AuthMiddleware) so probes that cannot present credentials still work.
    from starlette.routing import Route
    app.router.routes.insert(0, Route("/export", _export_route, methods=["GET"]))
    app.router.routes.insert(0, Route("/health", _health_route, methods=["GET"]))

    # The browser's JSON API. Session-authenticated by AuthMiddleware; content
    # stays read-only here — only MCP writes.
    app.router.routes.insert(0, build_api_routes(
        identity=runtime._identity, db_getter=runtime._db, settings_info=_settings_info,
        first_run=runtime.first_run, https=tls,
    ))
    app.router.routes[0:0] = build_tls_routes(tls)

    # The single-page app last: its final route is a catch-all.
    app.router.routes.extend(build_spa_routes(app_dir))

    app = CompactToolsMiddleware(app)

    # CSP, nosniff, referrer policy on everything but /mcp. HSTS only once
    # HTTPS is confirmed and only on 443: it binds to the hostname, not the
    # port, and would otherwise send browsers to https://host:<plain port>.
    app = SecurityHeadersMiddleware(
        app,
        pending_origin=tls.pending_origin if tls else None,
        hsts=(lambda: tls.active() and config.HTTPS_PORT == 443) if tls else None,
    )

    # Capture the principal, client and ip per request for the audit log.
    # Inside AuthMiddleware, which is what puts the principal in the scope.
    app = RequestMetaMiddleware(app)

    # Every /mcp call carries a per-user token; every /api call a session.
    app = AuthMiddleware(app, identity=runtime._identity)

    # Outside auth so an oversized body is refused before anything buffers it,
    # inside CORS so a 413 still carries the CORS headers a browser needs.
    app = BodyLimitMiddleware(app, max_bytes=config.MAX_BODY_BYTES)

    if config.CORS_ORIGINS:
        from starlette.middleware.cors import CORSMiddleware
        origins = [o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()]
        if "*" in origins:
            logger.warning(
                "CORS is open to all origins (*): any website may call /mcp with a token it holds."
            )
        app = CORSMiddleware(
            app,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )
        logger.info("CORS enabled for origins: %s", origins)

    # Outermost: once HTTPS is confirmed, the plain port only hands clients
    # over to it. Sits outside CORS on purpose — a 403 here is not for browsers
    # that already reached the right origin.
    app = PlainPortGate(app, tls)

    return app


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Mnem-O-matic MCP server")
    logger.info("Configuration: db_path=%s, host=%s, port=%s", config.DB_PATH, config.HOST, config.PORT)

    # Pre-warm db and resolve embedder so the first request doesn't pay setup costs
    logger.info("Initializing database...")
    runtime._db()
    # Someone must be able to log in: create `admin` from the environment or
    # print a one-time setup code for the browser's first-run screen.
    try:
        ensure_bootstrap(runtime._identity(), runtime.first_run, config.ADMIN_PASSWORD)
    except IdentityError as e:
        logger.error("MNEMOMATIC_ADMIN_PASSWORD rejected: %s", e.details)
        raise SystemExit(1)
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
        from mnemomatic.backup import start_backup_thread
        start_backup_thread(runtime._db, Path(config.BACKUP_DIR), interval_hours=config.BACKUP_INTERVAL_HOURS,
                            keep=config.BACKUP_KEEP, server_version=_server_version())
        logger.info("Scheduled backups: every %gh to %s (keeping %d)",
                    config.BACKUP_INTERVAL_HOURS, config.BACKUP_DIR, config.BACKUP_KEEP)

    # Built-in TLS: a private CA plus an HTTPS listener, enabled from the UI.
    # Off means TLS terminates elsewhere (a reverse proxy) and the HTTPS
    # endpoints answer 409.
    tls = None
    if config.TLS_MODE == "auto":
        tls = TlsState(runtime._db, Path(config.TLS_DIR), config.HTTPS_PORT, public_host=config.PUBLIC_HOST)
        tls.bootstrap()
        start_renewal_thread(tls)
    else:
        logger.info("Built-in TLS is off (MNEMOMATIC_TLS=off)")

    logger.info("Building ASGI application...")
    app = build_app(tls)

    logger.info("Trusted proxies: %s", ", ".join(config.TRUSTED_PROXIES) or "none")
    if "*" in config.TRUSTED_PROXIES:
        # The server cannot see whether its port is published, so say it once:
        # with "*" any client that reaches the port directly can claim any
        # address, which defeats the sign-in throttles and the audit log's ip.
        logger.warning("MNEMOMATIC_TRUSTED_PROXIES=* believes X-Forwarded-For from every peer; "
                       "this is safe only if the server port is reachable solely through your proxy. "
                       "Otherwise name the proxy's address instead.")
    logger.info("Starting server on %s:%d", config.HOST, config.PORT)
    # With trusted proxies configured, uvicorn rewrites the client address and
    # scheme from X-Forwarded-For / X-Forwarded-Proto — but only for requests
    # whose socket peer is on the list. Everything downstream (the throttles,
    # the audit log, the session cookie's Secure flag) then reads the real
    # client without doing its own header parsing. Unset means no forwarded
    # header is believed from anyone.
    asyncio.run(_serve_all(app, tls))


if __name__ == "__main__":
    main()
