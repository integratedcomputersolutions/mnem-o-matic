"""Read-only web viewer for Mnem-O-matic contents.

A minimal, server-rendered HTML browser for the stored documents, knowledge,
and notes. It is strictly read-only — no create, edit, or delete — and has no
user accounts. Access is gated by a single pre-shared secret
(``MNEMOMATIC_UI_TOKEN``): the visitor enters it once on a login page. The
cookie carries a random session id issued at login and held in memory — never
the token itself — so a captured cookie reveals nothing about the secret,
logging out revokes it server-side, and a restart ends every session. Repeated
wrong tokens from the same client trigger a temporary lockout. When the token
is unset the viewer is not registered at all, so it stays off by default.

The pages carry a strict Content-Security-Policy: the viewer ships no
JavaScript at all, so scripts are refused outright. Escaping every rendered
value is still what prevents injection; the policy is the second line.

The routes live under ``/ui`` on the same ASGI app as the MCP endpoint;
``BearerAuthMiddleware`` exempts that prefix because the viewer carries its own
gate rather than the MCP Bearer token.

Styling is stock Bootstrap 5 (vendored under ``static/`` and served at
``/ui/static/bootstrap.min.css``) — defaults only, no custom CSS.
"""

import hmac
import html
import secrets
import time
from datetime import timezone
from pathlib import Path
from urllib.parse import parse_qs, quote

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from mnemomatic.throttle import FailureThrottle

COOKIE_NAME = "mnemomatic_ui"
STATIC_DIR = Path(__file__).parent / "static"

# Live sessions are held in a dict, so cap it: 256 concurrent viewers is far
# more than a shared-secret viewer sees, and past that the oldest is evicted
# rather than letting repeated logins grow the table without bound.
_MAX_SESSIONS = 256
_SESSION_MAX_AGE = 30 * 86400

# Per-type detail rendering: which getter to call to fetch a single item.
_ITEM_GETTERS = {
    "document": "get_document",
    "knowledge": "get_knowledge",
    "note": "get_note",
}


# Sent with every HTML page. The viewer has no JavaScript whatsoever, so
# scripts are denied outright rather than allow-listed; the only styling is the
# vendored Bootstrap file plus two inline `style=` attributes on table headers,
# hence 'self' and 'unsafe-inline' for styles alone.
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"


def _html(body: str, status_code: int = 200, headers: dict | None = None) -> HTMLResponse:
    """An HTMLResponse carrying the viewer's fixed security headers.

    Every HTML route goes through here so the headers cannot be forgotten on a
    new page. Per-response headers (Retry-After) are merged on top.
    """
    return HTMLResponse(body, status_code=status_code,
                        headers={**_SECURITY_HEADERS, **(headers or {})})


async def _form_token(request: Request) -> str | None:
    """The submitted token, or None when the body is not a plain urlencoded form.

    Parsed with the standard library on purpose. The login page is reachable
    without credentials, so Starlette's form parsing put a general multipart
    parser — considerably more code than a one-field form needs — in front of
    the gate. Body size is already bounded by BodyLimitMiddleware.
    """
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != _FORM_CONTENT_TYPE:
        return None
    body = await request.body()
    fields = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True)
    return fields.get("token", [""])[0].strip()


def _is_https(request: Request) -> bool:
    """True when the client connection is HTTPS.

    Behind a reverse proxy this depends on MNEMOMATIC_TRUSTED_PROXIES, which
    lets uvicorn resolve the scheme from X-Forwarded-Proto — but only for
    proxies on the trust list. Reading that header here instead would believe
    it from anyone.
    """
    return request.url.scheme == "https"


def _ns_href(namespace: str) -> str:
    """Namespace link with the path segment percent-encoded (a namespace may
    contain any character, including quotes and slashes)."""
    return f"/ui/ns/{quote(namespace, safe='')}"


def _esc(value) -> str:
    """HTML-escape any value as text (everything stored is user-supplied)."""
    return html.escape("" if value is None else str(value))


def _page(title: str, body: str, show_logout: bool = True) -> str:
    """Wrap body fragments in the shared page chrome (stock Bootstrap).

    show_logout doubles as "the visitor is authenticated": it also controls
    the nav links, so the login page shows neither.
    """
    nav = (
        # Explicit text-light: the link sits in a plain flex div, not a
        # navbar-nav, so the navbar's dark theme doesn't color it.
        '<a class="nav-link px-0 text-light" href="/ui/settings">Settings</a>'
        '<form method="post" action="/ui/logout" class="m-0">'
        '<button type="submit" class="btn btn-sm btn-outline-light">Log out</button></form>'
        if show_logout else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} · Mnem-O-matic</title>
<link rel="stylesheet" href="/ui/static/bootstrap.min.css">
</head>
<body>
<nav class="navbar navbar-expand bg-dark" data-bs-theme="dark">
  <div class="container">
    <a class="navbar-brand" href="/ui">Mnem-O-matic</a>
    <div class="d-flex align-items-center gap-3">
      <span class="navbar-text">read-only viewer</span>
      {nav}
    </div>
  </div>
</nav>
<main class="container my-4">
{body}
</main>
</body>
</html>
"""


def _fmt_dt(dt) -> str:
    """Human-readable UTC timestamp (e.g. 'May 29, 2026, 22:41 UTC').

    The exact ISO-8601 value is kept in a title attribute for hover/precision.
    Stored timestamps are UTC; normalize defensively in case of an offset.
    """
    utc = dt.astimezone(timezone.utc)
    text = utc.strftime("%b %d, %Y, %H:%M UTC")
    return f'<span title="{_esc(dt.isoformat())}">{_esc(text)}</span>'


def _tags_html(tags: list[str]) -> str:
    if not tags:
        return '<span class="text-muted">—</span>'
    return "".join(
        f'<span class="badge text-bg-secondary me-1">{_esc(t)}</span>' for t in tags
    )


def _breadcrumb(*crumbs: tuple) -> str:
    """Build a Bootstrap breadcrumb from (label, href|None) pairs; last is active."""
    items = []
    for i, (label, href) in enumerate(crumbs):
        if i == len(crumbs) - 1 or href is None:
            items.append(f'<li class="breadcrumb-item active" aria-current="page">{_esc(label)}</li>')
        else:
            items.append(f'<li class="breadcrumb-item"><a href="{_esc(href)}">{_esc(label)}</a></li>')
    return f'<nav aria-label="breadcrumb"><ol class="breadcrumb">{"".join(items)}</ol></nav>'


def _login_page(error: str = "") -> str:
    alert = f'<div class="alert alert-danger py-2">{_esc(error)}</div>' if error else ""
    body = f"""
<div class="row justify-content-center">
  <div class="col-sm-9 col-md-6 col-lg-4">
    <div class="card mt-5">
      <div class="card-body">
        <h1 class="h5 card-title">Enter access token</h1>
        <p class="text-muted small">This viewer is protected by a shared access token.</p>
        {alert}
        <form method="post" action="/ui/login">
          <div class="mb-3">
            <input type="password" name="token" class="form-control" placeholder="Access token" autofocus required>
          </div>
          <button type="submit" class="btn btn-primary w-100">View</button>
        </form>
      </div>
    </div>
  </div>
</div>
"""
    return _page("Login", body, show_logout=False)


def build_routes(db_getter, token: str, settings_info=None, make_export=None) -> list[Route]:
    """Build the viewer's Starlette routes, closing over the db accessor and token.

    Args:
        db_getter: zero-arg callable returning the shared Database instance.
        token: the shared secret required to view; never empty when registered.
        settings_info: optional zero-arg callable returning the configuration
            dict rendered on /ui/settings (see server._settings_info). When
            None the page renders with placeholders.
        make_export: optional callable (namespace | None) -> (zip bytes,
            filename) powering the settings page's export download (see
            server._make_export). When None the section and route are absent.
    """
    # Session ids issued at login, mapped to their expiry. In memory only: a
    # restart ends every session, which is also how a rotated MNEMOMATIC_UI_TOKEN
    # takes effect. Bounded so a flood of logins cannot grow it without limit.
    sessions: dict[str, float] = {}
    throttle = FailureThrottle()

    def _authed(request: Request) -> bool:
        """True when the request carries a live session cookie."""
        # A 256-bit random id looked up in a dict is not a timing oracle worth
        # defending: the comparison is against a hash of the presented value,
        # not against a secret-derived string, so compare_digest buys nothing
        # here. (It is still what checks the shared token below.)
        sid = request.cookies.get(COOKIE_NAME, "")
        expiry = sessions.get(sid)
        if expiry is None:
            return False
        if expiry <= time.monotonic():
            sessions.pop(sid, None)
            return False
        return True

    def _new_session() -> str:
        if len(sessions) >= _MAX_SESSIONS:
            sessions.pop(min(sessions, key=sessions.get), None)
        sid = secrets.token_urlsafe(32)
        sessions[sid] = time.monotonic() + _SESSION_MAX_AGE
        return sid

    def _client(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def static_css(request: Request) -> Response:
        # Public asset (no cookie) so the login page is styled too. Only
        # nosniff applies — it is a stylesheet, not a document with a policy.
        return FileResponse(STATIC_DIR / "bootstrap.min.css", media_type="text/css",
                            headers={"X-Content-Type-Options": "nosniff"})

    def login_form(request: Request) -> Response:
        if _authed(request):
            return RedirectResponse("/ui", status_code=303)
        return _html(_login_page())

    async def login_submit(request: Request) -> Response:
        client = _client(request)
        wait = throttle.retry_after(client)
        if wait:
            return _html(
                _login_page(f"Too many failed attempts. Try again in {wait} seconds."),
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        supplied = await _form_token(request)
        if supplied is None:
            return _html(_login_page("Unsupported form encoding."), status_code=415)
        if not hmac.compare_digest(supplied, token):
            throttle.record_failure(client)
            return _html(_login_page("Incorrect token."), status_code=401)
        throttle.record_success(client)
        resp = RedirectResponse("/ui", status_code=303)
        # HttpOnly so client JS can't read it; SameSite=Lax is fine for a viewer.
        # Secure whenever the client connected over HTTPS (directly or via proxy).
        resp.set_cookie(
            COOKIE_NAME, _new_session(), httponly=True, samesite="lax",
            secure=_is_https(request), max_age=_SESSION_MAX_AGE,
        )
        return resp

    def logout(request: Request) -> Response:
        # Drop the session server-side, not just the browser's copy of it —
        # otherwise a captured cookie outlives the logout that was meant to
        # end it.
        sessions.pop(request.cookies.get(COOKIE_NAME, ""), None)
        resp = RedirectResponse("/ui/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    def index(request: Request) -> Response:
        if not _authed(request):
            return RedirectResponse("/ui/login", status_code=303)
        # COUNT queries only — the old per-namespace list_*() calls loaded
        # every row including full document content just to len() them.
        counts = db_getter().namespace_counts()
        if not counts:
            body = '<h1 class="h3 mb-3">Namespaces</h1><div class="alert alert-secondary">No data yet.</div>'
            return _html(_page("Namespaces", body))
        rows = []
        for ns, c in counts.items():
            rows.append(
                f'<tr><td><a href="{_esc(_ns_href(ns))}">{_esc(ns)}</a></td>'
                f'<td class="text-end">{c["documents"]}</td>'
                f'<td class="text-end">{c["knowledge"]}</td>'
                f'<td class="text-end">{c["notes"]}</td></tr>'
            )
        body = (
            '<h1 class="h3 mb-3">Namespaces</h1>'
            '<table class="table table-hover align-middle">'
            '<thead><tr><th>Namespace</th><th class="text-end">Documents</th>'
            '<th class="text-end">Knowledge</th><th class="text-end">Notes</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
        return _html(_page("Namespaces", body))

    def namespace_view(request: Request) -> Response:
        if not _authed(request):
            return RedirectResponse("/ui/login", status_code=303)
        ns = request.path_params["namespace"]
        db = db_getter()

        def section(heading, items, type_key, title_attr, extra_label, extra_attr):
            if not items:
                return f'<h2 class="h5 mt-4">{_esc(heading)}</h2><p class="text-muted">None.</p>'
            body_rows = []
            for it in items:
                body_rows.append(
                    f'<tr><td><a href="/ui/item/{type_key}/{_esc(it.id)}">{_esc(getattr(it, title_attr))}</a></td>'
                    f"<td>{_esc(getattr(it, extra_attr))}</td><td>{_tags_html(it.tags)}</td>"
                    f'<td class="text-muted text-nowrap">{_fmt_dt(it.updated_at)}</td></tr>'
                )
            return (
                f'<h2 class="h5 mt-4">{_esc(heading)}</h2>'
                '<table class="table table-sm table-hover align-middle">'
                f'<thead><tr><th>{_esc(title_attr.capitalize())}</th><th>{_esc(extra_label)}</th>'
                '<th>Tags</th><th>Updated</th></tr></thead>'
                f'<tbody>{"".join(body_rows)}</tbody></table>'
            )

        body = (
            _breadcrumb(("Namespaces", "/ui"), (ns, None))
            + f'<h1 class="h3 mb-3">{_esc(ns)}</h1>'
            + section("Documents", db.list_documents(ns), "document", "title", "MIME type", "mime_type")
            + section("Knowledge", db.list_knowledge(ns), "knowledge", "subject", "Source", "source")
            + section("Notes", db.list_notes(ns), "note", "title", "Source", "source")
        )
        return _html(_page(ns, body))

    def settings_view(request: Request) -> Response:
        if not _authed(request):
            return RedirectResponse("/ui/login", status_code=303)
        info = settings_info() if settings_info is not None else {}

        def val(key, suffix=""):
            v = info.get(key)
            return f"{_esc(v)}{suffix}" if v is not None else '<span class="text-muted">—</span>'

        def prefix(key):
            # Prefixes are shown quoted in <code> so a trailing space — which
            # is load-bearing for asymmetric models — is visible.
            v = info.get(key)
            return f"<code>&quot;{_esc(v)}&quot;</code>" if v else '<span class="text-muted">(none)</span>'

        dim_cfg, dim_db = info.get("dim_configured"), info.get("dim_database")
        alert = ""
        if dim_cfg is not None and dim_db is not None and dim_cfg != dim_db:
            alert = (
                '<div class="alert alert-warning">The vector index was built at dimension '
                f"<strong>{_esc(dim_db)}</strong> but the server is configured for "
                f"<strong>{_esc(dim_cfg)}</strong>. Semantic search cannot work like this — "
                "set <code>MNEMOMATIC_REINDEX=auto</code> and restart to rebuild the index "
                "and re-embed all content.</div>"
            )

        model_cfg, model_db = info.get("model"), info.get("model_database")
        if model_cfg and model_db and model_cfg != model_db:
            alert += (
                '<div class="alert alert-warning">The vector index was built by '
                f"<strong>{_esc(model_db)}</strong> but the server is configured for "
                f"<strong>{_esc(model_cfg)}</strong>. Queries embedded by one model and "
                "searched against another model's vectors return wrong results with no "
                "error — set <code>MNEMOMATIC_REINDEX=auto</code> and restart to rebuild "
                "the index and re-embed all content.</div>"
            )

        model_html = val("model")
        if info.get("model") and info.get("model_url"):
            model_html = (
                f'<a href="{_esc(info["model_url"])}" target="_blank" rel="noopener">'
                f'{_esc(info["model"])}</a>'
            )
        embed_rows = [
            ("Embedding mode", val("mode")),
            ("Model", model_html),
            ("Embedding dimension", val("dim_configured")),
            ("Index built at dimension", val("dim_database")),
            # Absent on databases written before the server recorded which model
            # built the index — say so rather than showing a bare dash.
            ("Index built by model", val("model_database")
             if info.get("model_database")
             else '<span class="text-muted">not recorded (pre-dates this check)</span>'),
        ]
        if info.get("endpoint_url"):
            embed_rows += [
                ("Endpoint URL", f"<code>{_esc(info['endpoint_url'])}</code>"),
                ("Wire format", val("wire_api")),
            ]
        else:
            embed_rows.append(("Token truncation limit", val("max_tokens", " tokens")))
        embed_rows += [
            ("Query prefix", prefix("query_prefix")),
            ("Document prefix", prefix("doc_prefix")),
        ]
        chunk_rows = [
            ("Chunk threshold", val("chunk_threshold", " chars")),
            ("Chunk size", val("chunk_size", " chars")),
            ("Chunk overlap", val("chunk_overlap", " chars")),
        ]

        def table(rows):
            body_rows = "".join(
                f'<tr><th scope="row" class="text-nowrap" style="width:14rem">{_esc(label)}</th>'
                f"<td>{value}</td></tr>"
                for label, value in rows
            )
            return f'<table class="table table-sm"><tbody>{body_rows}</tbody></table>'

        export_html = ""
        if make_export is not None:
            export_html = (
                '<h2 class="h5 mt-4">Export</h2>'
                '<p class="text-muted mb-2">Download all namespaces as a zip of markdown '
                "files with metadata sidecars — human-readable, and independent of the "
                "embedding model.</p>"
                '<a class="btn btn-sm btn-primary" href="/ui/export">Download export</a>'
            )

        # Sectioned so future settings (retention, ...) can be appended as
        # further h2 blocks.
        body = (
            _breadcrumb(("Namespaces", "/ui"), ("Settings", None))
            + '<h1 class="h3 mb-1">Settings</h1>'
            + f'<p class="text-muted">mnemomatic-server {val("version")}</p>'
            + alert
            + '<h2 class="h5 mt-4">Embedding model</h2>'
            + table(embed_rows)
            + '<h2 class="h5 mt-4">Document chunking</h2>'
            + table(chunk_rows)
            + export_html
        )
        return _html(_page("Settings", body))

    def export_download(request: Request) -> Response:
        if not _authed(request):
            return RedirectResponse("/ui/login", status_code=303)
        if make_export is None:
            return _html(
                _page("Not found", '<div class="alert alert-warning">Export is not available.</div>'),
                status_code=404,
            )
        data, filename = make_export(None)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def item_view(request: Request) -> Response:
        if not _authed(request):
            return RedirectResponse("/ui/login", status_code=303)
        item_type = request.path_params["item_type"]
        item_id = request.path_params["id"]
        getter_name = _ITEM_GETTERS.get(item_type)
        if getter_name is None:
            return _html(
                _page("Not found", '<div class="alert alert-warning">Unknown item type.</div>'),
                status_code=404,
            )
        item = getattr(db_getter(), getter_name)(item_id)
        if item is None:
            return _html(
                _page("Not found", '<div class="alert alert-warning">Item not found.</div>'),
                status_code=404,
            )

        # Field order per type: the main long-form field renders as <pre>.
        if item_type == "document":
            heading, long_label, long_value = item.title, "Content", item.content
            meta_rows = [("MIME type", item.mime_type)]
        elif item_type == "knowledge":
            heading, long_label, long_value = item.subject, "Fact", item.fact
            meta_rows = [("Confidence", item.confidence), ("Source", item.source)]
        else:  # note
            heading, long_label, long_value = item.title, "Content", item.content
            meta_rows = [("Source", item.source)]

        info = [
            ("Namespace", f'<a href="{_esc(_ns_href(item.namespace))}">{_esc(item.namespace)}</a>'),
            *[(label, _esc(val)) for label, val in meta_rows],
            ("Tags", _tags_html(item.tags)),
            ("Created", _fmt_dt(item.created_at)),
            ("Updated", _fmt_dt(item.updated_at)),
            ("ID", f"<code>{_esc(item.id)}</code>"),
        ]
        info_rows = "".join(
            f'<tr><th scope="row" class="text-nowrap" style="width:10rem">{_esc(label)}</th><td>{val}</td></tr>'
            for label, val in info
        )

        metadata_html = ""
        if item.metadata:
            meta_items = "".join(
                f"<tr><th scope=\"row\">{_esc(k)}</th><td>{_esc(v)}</td></tr>"
                for k, v in item.metadata.items()
            )
            metadata_html = (
                '<h2 class="h5 mt-4">Metadata</h2>'
                f'<table class="table table-sm"><tbody>{meta_items}</tbody></table>'
            )

        body = (
            _breadcrumb(("Namespaces", "/ui"), (item.namespace, _ns_href(item.namespace)), (heading, None))
            + '<div class="card">'
            + f'<div class="card-header"><h1 class="h5 mb-0">{_esc(heading)}</h1></div>'
            + '<div class="card-body">'
            + f'<table class="table table-sm mb-0"><tbody>{info_rows}</tbody></table>'
            + "</div></div>"
            + f'<h2 class="h5 mt-4">{_esc(long_label)}</h2>'
            + f'<pre class="border rounded bg-light p-3"><code>{_esc(long_value)}</code></pre>'
            + metadata_html
        )
        return _html(_page(heading, body))

    return [
        Route("/ui", index, methods=["GET"]),
        Route("/ui/login", login_form, methods=["GET"]),
        Route("/ui/login", login_submit, methods=["POST"]),
        Route("/ui/logout", logout, methods=["POST"]),
        Route("/ui/settings", settings_view, methods=["GET"]),
        Route("/ui/export", export_download, methods=["GET"]),
        Route("/ui/static/bootstrap.min.css", static_css, methods=["GET"]),
        # :path so namespaces containing '/' (decoded from %2F before routing)
        # still resolve to the namespace view.
        Route("/ui/ns/{namespace:path}", namespace_view, methods=["GET"]),
        Route("/ui/item/{item_type}/{id}", item_view, methods=["GET"]),
    ]


def register_webui(app, db_getter, token: str, settings_info=None, make_export=None) -> None:
    """Attach the viewer routes to an existing Starlette app under ``/ui``.

    Routes are inserted ahead of the app's own routes so the viewer prefix is
    matched before any MCP catch-all. No-op semantics are the caller's job:
    only call this when ``token`` is set.
    """
    app.router.routes[:0] = build_routes(db_getter, token, settings_info, make_export)
