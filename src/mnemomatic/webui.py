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
"""

import hmac
import html

from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

COOKIE_NAME = "mnemomatic_ui"

# Per-type detail rendering: which getter to call and which fields to show.
# Kept here (not in db.py) so the viewer owns its own presentation concerns.
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


def _page(title: str, body: str) -> str:
    """Wrap body fragments in the shared page chrome."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} · Mnem-O-matic</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto;
          padding: 0 1rem; line-height: 1.5; }}
  header {{ display: flex; align-items: baseline; gap: 1rem; border-bottom: 1px solid #8884;
           padding-bottom: .5rem; margin-bottom: 1.5rem; }}
  header h1 {{ font-size: 1.2rem; margin: 0; }}
  nav a {{ margin-right: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #8883;
           vertical-align: top; }}
  th {{ font-weight: 600; }}
  code, pre {{ background: #8881; border-radius: 4px; }}
  code {{ padding: .1rem .3rem; }}
  pre {{ padding: 1rem; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
  .tag {{ display: inline-block; background: #6cf3; border-radius: 4px;
          padding: .05rem .4rem; margin: 0 .2rem .2rem 0; font-size: .85em; }}
  .muted {{ color: #8888; }}
  .count {{ font-variant-numeric: tabular-nums; }}
  form.login {{ display: flex; gap: .5rem; margin-top: 2rem; }}
</style>
</head>
<body>
<header>
  <h1><a href="/ui">Mnem-O-matic</a></h1>
  <span class="muted">read-only viewer</span>
</header>
{body}
</body>
</html>
"""


def _tags_html(tags: list[str]) -> str:
    if not tags:
        return '<span class="muted">—</span>'
    return "".join(f'<span class="tag">{_esc(t)}</span>' for t in tags)


def _login_page(error: str = "") -> str:
    note = f'<p class="muted">{_esc(error)}</p>' if error else ""
    body = f"""
<h2>Enter access token</h2>
<p class="muted">This viewer is protected by a shared secret.</p>
{note}
<form class="login" method="post" action="/ui/login">
  <input type="password" name="token" placeholder="access token" autofocus required>
  <button type="submit">View</button>
</form>
"""
    return _page("Login", body)


def build_routes(db_getter, token: str) -> list[Route]:
    """Build the viewer's Starlette routes, closing over the db accessor and token.

    Args:
        db_getter: zero-arg callable returning the shared Database instance.
        token: the shared secret required to view; never empty when registered.
    """

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
        rows = []
        for ns in namespaces:
            n_docs = len(db.list_documents(ns))
            n_know = len(db.list_knowledge(ns))
            n_notes = len(db.list_notes(ns))
            rows.append(
                f"<tr><td><a href=\"/ui/ns/{_esc(ns)}\">{_esc(ns)}</a></td>"
                f"<td class=\"count\">{n_docs}</td>"
                f"<td class=\"count\">{n_know}</td>"
                f"<td class=\"count\">{n_notes}</td></tr>"
            )
        table = (
            "<table><thead><tr><th>Namespace</th><th>Documents</th>"
            "<th>Knowledge</th><th>Notes</th></tr></thead><tbody>"
            + ("".join(rows) or '<tr><td colspan="4" class="muted">No data yet.</td></tr>')
            + "</tbody></table>"
        )
        body = f'<nav><a href="/ui/logout">Log out</a></nav><h2>Namespaces</h2>{table}'
        return HTMLResponse(_page("Namespaces", body))

    def namespace_view(request: Request) -> Response:
        if not _authed(request, token):
            return RedirectResponse("/ui/login", status_code=303)
        ns = request.path_params["namespace"]
        db = db_getter()

        def item_table(items, type_key, title_attr, extra_label, extra_attr):
            if not items:
                return '<p class="muted">None.</p>'
            head = (
                f"<table><thead><tr><th>{_esc(title_attr.capitalize())}</th>"
                f"<th>{_esc(extra_label)}</th><th>Tags</th><th>Updated</th></tr></thead><tbody>"
            )
            body_rows = []
            for it in items:
                title = getattr(it, title_attr)
                extra = getattr(it, extra_attr)
                body_rows.append(
                    f'<tr><td><a href="/ui/item/{type_key}/{_esc(it.id)}">{_esc(title)}</a></td>'
                    f"<td>{_esc(extra)}</td><td>{_tags_html(it.tags)}</td>"
                    f'<td class="muted">{_esc(it.updated_at.isoformat())}</td></tr>'
                )
            return head + "".join(body_rows) + "</tbody></table>"

        docs = item_table(db.list_documents(ns), "document", "title", "MIME type", "mime_type")
        know = item_table(db.list_knowledge(ns), "knowledge", "subject", "Source", "source")
        notes = item_table(db.list_notes(ns), "note", "title", "Source", "source")
        body = (
            f'<nav><a href="/ui">← Namespaces</a></nav>'
            f"<h2>{_esc(ns)}</h2>"
            f"<h3>Documents</h3>{docs}"
            f"<h3>Knowledge</h3>{know}"
            f"<h3>Notes</h3>{notes}"
        )
        return HTMLResponse(_page(ns, body))

    def item_view(request: Request) -> Response:
        if not _authed(request, token):
            return RedirectResponse("/ui/login", status_code=303)
        item_type = request.path_params["item_type"]
        item_id = request.path_params["id"]
        getter_name = _ITEM_GETTERS.get(item_type)
        if getter_name is None:
            return HTMLResponse(_page("Not found", '<p class="muted">Unknown item type.</p>'), status_code=404)
        item = getattr(db_getter(), getter_name)(item_id)
        if item is None:
            return HTMLResponse(
                _page("Not found", '<p class="muted">Item not found.</p>'), status_code=404
            )

        # Field order per type: (label, value) — the main long-form field renders as <pre>.
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
            ("Created", _esc(item.created_at.isoformat())),
            ("Updated", _esc(item.updated_at.isoformat())),
            ("ID", f"<code>{_esc(item.id)}</code>"),
        ]
        info_table = "<table><tbody>" + "".join(
            f"<tr><th>{_esc(label)}</th><td>{val}</td></tr>" for label, val in info
        ) + "</tbody></table>"

        metadata_html = ""
        if item.metadata:
            meta_items = "".join(
                f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in item.metadata.items()
            )
            metadata_html = f"<h3>Metadata</h3><table><tbody>{meta_items}</tbody></table>"

        body = (
            f'<nav><a href="/ui/ns/{_esc(item.namespace)}">← {_esc(item.namespace)}</a></nav>'
            f"<h2>{_esc(heading)}</h2>"
            f"{info_table}"
            f"<h3>{_esc(long_label)}</h3><pre>{_esc(long_value)}</pre>"
            f"{metadata_html}"
        )
        return HTMLResponse(_page(heading, body))

    return [
        Route("/ui", index, methods=["GET"]),
        Route("/ui/login", login_form, methods=["GET"]),
        Route("/ui/login", login_submit, methods=["POST"]),
        Route("/ui/logout", logout, methods=["GET"]),
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
