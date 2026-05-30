"""Read-only web viewer for Mnem-O-matic contents.

A minimal, server-rendered HTML browser for the stored documents, knowledge,
and notes. It is strictly read-only — no create, edit, or delete — and has no
user accounts. Access is gated by a single pre-shared secret
(``MNEMOMATIC_UI_TOKEN``): the visitor enters it once on a login page and it is
stored in an HttpOnly cookie thereafter. When the token is unset the viewer is
not registered at all, so it stays off by default.

The routes live under ``/ui`` on the same ASGI app as the MCP endpoint;
``BearerAuthMiddleware`` exempts that prefix because the viewer carries its own
gate rather than the MCP Bearer token.

Styling is stock Bootstrap 5 (vendored under ``static/`` and served at
``/ui/static/bootstrap.min.css``) — defaults only, no custom CSS.
"""

import hmac
import html
from datetime import timezone
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

COOKIE_NAME = "mnemomatic_ui"
STATIC_DIR = Path(__file__).parent / "static"

# Per-type detail rendering: which getter to call to fetch a single item.
_ITEM_GETTERS = {
    "document": "get_document",
    "knowledge": "get_knowledge",
    "note": "get_note",
}


def _authed(request: Request, token: str) -> bool:
    """True when the request carries a cookie matching the shared secret."""
    cookie = request.cookies.get(COOKIE_NAME, "")
    return bool(token) and hmac.compare_digest(cookie, token)


def _esc(value) -> str:
    """HTML-escape any value as text (everything stored is user-supplied)."""
    return html.escape("" if value is None else str(value))


def _page(title: str, body: str, show_logout: bool = True) -> str:
    """Wrap body fragments in the shared page chrome (stock Bootstrap)."""
    logout = (
        '<a class="btn btn-sm btn-outline-light" href="/ui/logout">Log out</a>'
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
      {logout}
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
            items.append(f'<li class="breadcrumb-item"><a href="{href}">{_esc(label)}</a></li>')
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


def build_routes(db_getter, token: str) -> list[Route]:
    """Build the viewer's Starlette routes, closing over the db accessor and token.

    Args:
        db_getter: zero-arg callable returning the shared Database instance.
        token: the shared secret required to view; never empty when registered.
    """

    def static_css(request: Request) -> Response:
        # Public asset (no cookie) so the login page is styled too.
        return FileResponse(STATIC_DIR / "bootstrap.min.css", media_type="text/css")

    def login_form(request: Request) -> Response:
        if _authed(request, token):
            return RedirectResponse("/ui", status_code=303)
        return HTMLResponse(_login_page())

    async def login_submit(request: Request) -> Response:
        form = await request.form()
        supplied = (form.get("token") or "").strip()
        if not hmac.compare_digest(supplied, token):
            return HTMLResponse(_login_page("Incorrect token."), status_code=401)
        resp = RedirectResponse("/ui", status_code=303)
        # HttpOnly so client JS can't read it; SameSite=Lax is fine for a viewer.
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=30 * 86400)
        return resp

    def logout(request: Request) -> Response:
        resp = RedirectResponse("/ui/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    def index(request: Request) -> Response:
        if not _authed(request, token):
            return RedirectResponse("/ui/login", status_code=303)
        db = db_getter()
        namespaces = db.list_namespaces()
        if not namespaces:
            body = '<h1 class="h3 mb-3">Namespaces</h1><div class="alert alert-secondary">No data yet.</div>'
            return HTMLResponse(_page("Namespaces", body))
        rows = []
        for ns in namespaces:
            rows.append(
                f'<tr><td><a href="/ui/ns/{_esc(ns)}">{_esc(ns)}</a></td>'
                f'<td class="text-end">{len(db.list_documents(ns))}</td>'
                f'<td class="text-end">{len(db.list_knowledge(ns))}</td>'
                f'<td class="text-end">{len(db.list_notes(ns))}</td></tr>'
            )
        body = (
            '<h1 class="h3 mb-3">Namespaces</h1>'
            '<table class="table table-hover align-middle">'
            '<thead><tr><th>Namespace</th><th class="text-end">Documents</th>'
            '<th class="text-end">Knowledge</th><th class="text-end">Notes</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
        return HTMLResponse(_page("Namespaces", body))

    def namespace_view(request: Request) -> Response:
        if not _authed(request, token):
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
        return HTMLResponse(_page(ns, body))

    def item_view(request: Request) -> Response:
        if not _authed(request, token):
            return RedirectResponse("/ui/login", status_code=303)
        item_type = request.path_params["item_type"]
        item_id = request.path_params["id"]
        getter_name = _ITEM_GETTERS.get(item_type)
        if getter_name is None:
            return HTMLResponse(
                _page("Not found", '<div class="alert alert-warning">Unknown item type.</div>'),
                status_code=404,
            )
        item = getattr(db_getter(), getter_name)(item_id)
        if item is None:
            return HTMLResponse(
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
            ("Namespace", f'<a href="/ui/ns/{_esc(item.namespace)}">{_esc(item.namespace)}</a>'),
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
            _breadcrumb(("Namespaces", "/ui"), (item.namespace, f"/ui/ns/{item.namespace}"), (heading, None))
            + '<div class="card">'
            + f'<div class="card-header"><h1 class="h5 mb-0">{_esc(heading)}</h1></div>'
            + '<div class="card-body">'
            + f'<table class="table table-sm mb-0"><tbody>{info_rows}</tbody></table>'
            + "</div></div>"
            + f'<h2 class="h5 mt-4">{_esc(long_label)}</h2>'
            + f'<pre class="border rounded bg-light p-3"><code>{_esc(long_value)}</code></pre>'
            + metadata_html
        )
        return HTMLResponse(_page(heading, body))

    return [
        Route("/ui", index, methods=["GET"]),
        Route("/ui/login", login_form, methods=["GET"]),
        Route("/ui/login", login_submit, methods=["POST"]),
        Route("/ui/logout", logout, methods=["GET"]),
        Route("/ui/static/bootstrap.min.css", static_css, methods=["GET"]),
        Route("/ui/ns/{namespace}", namespace_view, methods=["GET"]),
        Route("/ui/item/{item_type}/{id}", item_view, methods=["GET"]),
    ]


def register_webui(app, db_getter, token: str) -> None:
    """Attach the viewer routes to an existing Starlette app under ``/ui``.

    Routes are inserted ahead of the app's own routes so the viewer prefix is
    matched before any MCP catch-all. No-op semantics are the caller's job:
    only call this when ``token`` is set.
    """
    app.router.routes[:0] = build_routes(db_getter, token)
