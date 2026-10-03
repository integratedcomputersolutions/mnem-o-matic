"""Who is calling, decided once per request.

Two credentials, two audiences:

- **Bearer tokens** (``Authorization: Bearer mnm_…``) are what agents hold.
  They are the only thing accepted on ``/mcp``.
- **Session cookies** are what a browser holds after logging in. They are
  the only thing accepted under ``/api``. Over HTTPS the cookie is
  ``__Host-mnm_session`` (Secure, host-only); over plain HTTP it is
  ``mnm_session``. Each scheme reads only its own name, so a cookie set or
  sniffed over plain HTTP never stands in for an HTTPS session.

``/export`` takes either, since both the CLI and the web UI download it.
``/health``, the login endpoints, the CA download, the setup page, and the
SPA itself need nothing — they either carry no data or exist so a client can
*get* a credential. Whatever is resolved lands in ``scope["state"]["principal"]``
for route handlers (``request.state.principal``) and for the audit log.

Pure ASGI rather than BaseHTTPMiddleware so streamed MCP responses pass
through untouched, and so a rejection is one small send rather than a
Response object built around a request that never reached the app.
"""

import json
import logging
from http.cookies import CookieError, SimpleCookie

from starlette.types import ASGIApp, Receive, Scope, Send

from mnemomatic.identity import Principal
from mnemomatic.throttle import FailureThrottle, client_key

logger = logging.getLogger("mnemomatic")

COOKIE_NAME = "mnm_session"
SECURE_COOKIE_NAME = "__Host-mnm_session"


def session_cookie_name(scheme: str) -> str:
    """The session cookie a request on `scheme` sets and reads."""
    return SECURE_COOKIE_NAME if scheme == "https" else COOKIE_NAME

# Reachable with no credential at all.
PUBLIC_PATHS = frozenset({
    "/health", "/api/instance-id", "/api/login", "/api/first-run", "/ca.crt", "/setup",
})
# What a user who must still choose a password may call.
PASSWORD_GATE_EXEMPT = frozenset({"/api/password", "/api/logout", "/api/session"})

_BEARER_FORMAT = "Required format: 'Authorization: Bearer <token>'"


def classify(method: str, path: str) -> str:
    """Which credential a path takes: public, bearer, session, or any."""
    if path in PUBLIC_PATHS:
        return "public"
    if path == "/api/session" and method in ("GET", "HEAD"):
        return "public"
    if path == "/mcp" or path.startswith("/mcp/"):
        return "bearer"
    if path == "/export":
        return "any"
    if path == "/api" or path.startswith("/api/"):
        return "session"
    return "public"          # the SPA and its assets


async def _send_json(send: Send, status: int, body: dict, headers: dict[str, str] | None = None) -> None:
    payload = json.dumps(body).encode()
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"cache-control", b"no-store"),
    ]
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode(), v.encode()))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": payload})


def _cookie_value(headers: dict[str, str], name: str) -> str:
    raw = headers.get("cookie")
    if not raw:
        return ""
    jar = SimpleCookie()
    try:
        jar.load(raw)
    except CookieError:
        return ""
    morsel = jar.get(name)
    return morsel.value if morsel else ""


class AuthMiddleware:
    """Resolve the caller's principal, or refuse the request.

    `identity` is a callable returning the Identity store (so tests can hand
    in their own, and the server can hand in the lazily built singleton).
    Repeated invalid tokens from one address trip the same lockout the old
    shared key had: five failures in a minute, five minutes out.
    """

    def __init__(self, app: ASGIApp, identity, *, throttle: FailureThrottle | None = None):
        self.app = app
        self._identity = identity
        self._throttle = throttle or FailureThrottle()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method, path = scope["method"], scope["path"]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        ip = scope["client"][0] if scope.get("client") else "unknown"
        kind = classify(method, path)
        principal: Principal | None = None

        if kind == "public":
            # A public /api route (the session probe) still likes to know who
            # is asking; a bad or absent cookie simply means "nobody".
            if path.startswith("/api"):
                principal = self._from_cookie(scope, headers)
        elif kind == "bearer":
            principal, refusal = self._from_bearer(headers, ip, method, path)
            if refusal:
                await _send_json(send, *refusal)
                return
        elif kind == "any":
            if "authorization" in headers:
                principal, refusal = self._from_bearer(headers, ip, method, path)
                if refusal:
                    await _send_json(send, *refusal)
                    return
            else:
                principal = self._from_cookie(scope, headers)
                if principal is None:
                    await _send_json(send, 401, {
                        "error": "unauthenticated",
                        "details": "Sign in, or send 'Authorization: Bearer <token>'.",
                    })
                    return
        else:  # session
            principal = self._from_cookie(scope, headers)
            if principal is None:
                logger.debug("Unauthenticated %s %s from %s", method, path, ip)
                await _send_json(send, 401, {"error": "unauthenticated", "details": "Sign in first."})
                return
            if principal.user.must_change_password and path not in PASSWORD_GATE_EXEMPT:
                await _send_json(send, 403, {
                    "error": "password_change_required",
                    "details": "Choose a new password before doing anything else.",
                })
                return

        scope.setdefault("state", {})["principal"] = principal
        await self.app(scope, receive, send)

    # ── resolvers ──

    def _from_cookie(self, scope: Scope, headers: dict[str, str]) -> Principal | None:
        raw = _cookie_value(headers, session_cookie_name(scope.get("scheme", "http")))
        return self._identity().resolve_session(raw) if raw else None

    def _from_bearer(self, headers: dict[str, str], ip: str, method: str, path: str):
        """(principal, None) on success, (None, (status, body, headers)) to refuse."""
        wait = self._throttle.retry_after(client_key(ip))
        if wait:
            logger.warning("Throttled %s %s from %s (too many failed tokens)", method, path, ip)
            return None, (429, {"error": "throttled",
                                "details": f"Too many failed authentication attempts; retry after {wait} seconds"},
                          {"Retry-After": str(wait)})

        header = headers.get("authorization", "").strip()
        if not header:
            logger.warning("Missing Authorization header (%s %s from %s)", method, path, ip)
            return None, (401, {"error": "missing_authorization", "details": _BEARER_FORMAT}, None)
        if not header.lower().startswith("bearer "):
            logger.warning("Malformed Authorization header (%s %s from %s)", method, path, ip)
            return None, (401, {"error": "invalid_authorization", "details": _BEARER_FORMAT}, None)
        token = header[7:].strip()
        if not token:
            return None, (401, {"error": "invalid_authorization", "details": "Token is empty"}, None)

        principal = self._identity().resolve_token(token)
        if principal is None:
            self._throttle.record_failure(client_key(ip))
            logger.warning("Invalid API token (%s %s from %s)", method, path, ip)
            return None, (403, {"error": "invalid_token",
                                "details": "The token is unknown, revoked, expired, or its owner is disabled"},
                          None)
        self._throttle.record_success(client_key(ip))
        logger.debug("Authenticated %s via token %s (%s %s from %s)",
                     principal.user.username, principal.token_hint, method, path, ip)
        return principal, None
