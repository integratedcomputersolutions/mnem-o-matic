"""Request identity for the audit log.

The audit trail wants to answer *who* changed *what*. With a single shared
API key the server has no authenticated user identity, so the honest answer
is layered:

- ip / user-agent: what the connection itself reveals. Behind a reverse
  proxy the ip is the proxy's own unless MNEMOMATIC_TRUSTED_PROXIES names it,
  which lets uvicorn resolve the real client from X-Forwarded-For.
- actor: an optional self-declared label from the ``X-Mnemomatic-Actor``
  request header (e.g. set per client in its MCP config). Trustworthy among
  cooperating clients, not authenticated.

Tool handlers run deep inside the MCP app with no view of the HTTP request,
so RequestMetaMiddleware captures these fields into a contextvar that the
audit writer reads. Pure ASGI (no BaseHTTPMiddleware) so streamed MCP
responses pass through untouched.
"""

from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

_EMPTY = {"actor": None, "client": None, "ip": None}
_request_meta: ContextVar[dict] = ContextVar("mnemomatic_request_meta", default=_EMPTY)


def request_meta() -> dict:
    """The current request's identity fields: actor, client (user-agent), ip."""
    return _request_meta.get()


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
        token = _request_meta.set({
            "actor": headers.get("x-mnemomatic-actor"),
            "client": headers.get("user-agent"),
            "ip": scope["client"][0] if scope.get("client") else None,
        })
        try:
            await self.app(scope, receive, send)
        finally:
            _request_meta.reset(token)
