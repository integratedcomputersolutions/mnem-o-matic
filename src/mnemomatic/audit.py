"""Request identity for the audit log.

The audit trail wants to answer *who* changed *what*. The layers, from most
to least trustworthy:

- user / token: the authenticated principal AuthMiddleware resolved — the
  username behind the session cookie or API token, plus the token's id and
  hint so a revoked token's history stays traceable. This is the audit
  log's ``actor``.
- ip / user-agent: what the connection itself reveals. Behind a reverse
  proxy the ip is the proxy's own unless MNEMOMATIC_TRUSTED_PROXIES names it,
  which lets uvicorn resolve the real client from X-Forwarded-For.
- actor header: an optional self-declared label from ``X-Mnemomatic-Actor``
  (e.g. "laptop" vs "ci" under one person's token). Recorded in the event's
  detail, never as the actor.

Tool handlers run deep inside the MCP app with no view of the HTTP request,
so RequestMetaMiddleware captures these fields into a contextvar that the
audit writer reads. Pure ASGI (no BaseHTTPMiddleware) so streamed MCP
responses pass through untouched. It sits *inside* AuthMiddleware, which is
where the principal in ``scope["state"]`` comes from.
"""

import logging
from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("mnemomatic")

_EMPTY = {
    "actor": None, "client": None, "ip": None,
    "user": None, "user_id": None, "is_admin": False, "via": None,
    "token_id": None, "token_hint": None, "token_name": None,
}
_request_meta: ContextVar[dict] = ContextVar("mnemomatic_request_meta", default=_EMPTY)


def request_meta() -> dict:
    """The current request's identity fields: actor (header label), client
    (user-agent), ip, and — when authenticated — user, user_id, is_admin,
    via ("session" or "token"), token_id, token_hint, token_name."""
    return _request_meta.get()


def write_event(db, op: str, *, actor: str | None = None, item_type: str | None = None,
                item_id: str | None = None, namespace: str | None = None,
                title: str | None = None, **detail) -> None:
    """Append one audit event, enriched with the current request's identity.

    The one place every audit row is built: MCP tool writes, identity events
    from the API, bootstrap and the recovery CLI all come through here, so
    they agree on what an event carries. The actor defaults to the
    authenticated user; callers name it explicitly when there is no request
    (bootstrap, the CLI) or when the user is not signed in yet (a login). A
    token's id, hint and name go in the detail so the event stays readable
    after the token is revoked, and the optional X-Mnemomatic-Actor header
    is kept as `label` — a sub-identity someone chose for one client.

    A failing audit write is logged and never breaks the operation it
    describes.
    """
    meta = request_meta()
    if meta.get("token_id") is not None:
        detail["token"] = {"id": meta["token_id"], "hint": meta["token_hint"], "name": meta.get("token_name")}
    if meta.get("actor"):
        detail["label"] = meta["actor"]
    try:
        db.append_audit(op, item_type=item_type, item_id=item_id, namespace=namespace, title=title,
                        actor=actor or meta.get("user"), client=meta.get("client"), ip=meta.get("ip"),
                        detail=detail or None)
    except Exception as e:
        logger.warning("Audit write failed: %s: %s", type(e).__name__, e)


class RequestMetaMiddleware:
    """Capture per-request identity fields for the audit log."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        meta = {
            **_EMPTY,
            "actor": headers.get("x-mnemomatic-actor"),
            "client": headers.get("user-agent"),
            "ip": scope["client"][0] if scope.get("client") else None,
        }
        principal = scope.get("state", {}).get("principal")
        if principal is not None:
            meta.update(user=principal.user.username, user_id=principal.user.id,
                        is_admin=principal.user.is_admin, via=principal.via,
                        token_id=principal.token_id, token_hint=principal.token_hint,
                        token_name=principal.token_name)
        token = _request_meta.set(meta)
        try:
            await self.app(scope, receive, send)
        finally:
            _request_meta.reset(token)
