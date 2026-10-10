"""Who is calling, decided once per request.

Two credentials, two audiences:

- **Bearer tokens** (``Authorization: Bearer mnm_…``) are what agents hold.
  They are the only thing accepted on ``/mcp``.
- **Session cookies** are what a browser holds after logging in. They are
  the only thing accepted under ``/api``. Over HTTPS the cookie is
  ``__Host-mnm_session`` (Secure, host-only); over plain HTTP it is
  ``mnm_session``. Each scheme reads only its own name, so a cookie set or
  sniffed over plain HTTP never stands in for an HTTPS session.

A **trusted proxy** is the opt-in third way in (``MNEMOMATIC_PROXY_SECRET``):
a gateway that signs people in itself sends its secret as the bearer token
on ``/mcp`` or ``/export`` and names the user in a header, and the user is
created on first sight. The headers mean nothing without the secret, and the
secret means nothing without a user — there is no shared or anonymous
principal behind it. Personal tokens keep working alongside.

``/export`` takes either, since both the CLI and the web UI download it.
``/health``, the login endpoints, the CA download, the setup page, and the
SPA itself need nothing — they either carry no data or exist so a client can
*get* a credential. Whatever is resolved lands in ``scope["state"]["principal"]``
for route handlers (``request.state.principal``) and for the audit log.

Pure ASGI rather than BaseHTTPMiddleware so streamed MCP responses pass
through untouched, and so a rejection is one small send rather than a
Response object built around a request that never reached the app.
"""

import hmac
import json
import logging
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

from mnemomatic import config
from mnemomatic.identity import IdentityError, Principal
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

_PASSWORD_CHANGE_REQUIRED = (403, {
    "error": "password_change_required",
    "details": "Choose a new password before doing anything else.",
})


@dataclass(frozen=True)
class ProxyAuth:
    """How a trusted proxy identifies itself and the user it speaks for.

    The secret arrives as the bearer token; the headers carry the user's
    identity (required), display name and role (both optional, only read
    when configured). Header names are matched case-insensitively.
    """
    secret: str
    user_header: str
    name_header: str | None = None
    role_header: str | None = None

    @classmethod
    def from_config(cls) -> "ProxyAuth | None":
        """The proxy the environment describes, or None when the mode is off."""
        if not config.PROXY_SECRET:
            return None
        return cls(secret=config.PROXY_SECRET, user_header=config.PROXY_USER_HEADER,
                   name_header=config.PROXY_NAME_HEADER, role_header=config.PROXY_ROLE_HEADER)

    def matches(self, token: str) -> bool:
        """Constant-time: a guess learns nothing from how long the refusal took."""
        return hmac.compare_digest(token.encode("utf-8"), self.secret.encode("utf-8"))


def _header_text(headers: dict[str, str], name: str | None) -> str | None:
    """A header's value as text, or None. Header values were decoded as
    latin-1 to get them into a dict; a proxy sends names and addresses as
    UTF-8, so undo that here for the ones that carry a person's name."""
    if not name:
        return None
    raw = headers.get(name.lower())
    if raw is None:
        return None
    return raw.encode("latin-1").decode("utf-8", "replace").strip()


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


def _cookie_value(scope: Scope, name: str) -> str:
    """The value of cookie `name`, or "" when it is absent or sent twice.

    Parsed leniently, one `name=value` pair at a time, so a malformed cookie
    from some other app on the host (a space, a stray quote, JSON) cannot
    hide ours — the stdlib's SimpleCookie gives up on the whole header.
    Every Cookie header counts (HTTP/2 clients may split them). Two values
    for the name mean one was planted, from a sibling subdomain or another
    path, and there is no telling which: treat it as no session at all.
    """
    values = []
    for key, raw in scope.get("headers", []):
        if key.lower() != b"cookie":
            continue
        for pair in raw.decode("latin-1").split(";"):
            k, sep, v = pair.partition("=")
            if sep and k.strip() == name:
                values.append(v.strip())
    return values[0] if len(values) == 1 else ""


class AuthMiddleware:
    """Resolve the caller's principal, or refuse the request.

    `identity` is a callable returning the Identity store (so tests can hand
    in their own, and the server can hand in the lazily built singleton).
    `proxy` turns on trusted-proxy sign-in for the bearer paths. Repeated
    invalid tokens from one address trip the same lockout the old shared key
    had: five failures in a minute, five minutes out.
    """

    def __init__(self, app: ASGIApp, identity, *, throttle: FailureThrottle | None = None,
                 proxy: ProxyAuth | None = None):
        self.app = app
        self._identity = identity
        self._throttle = throttle or FailureThrottle()
        self._proxy = proxy

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
                principal = self._from_cookie(scope)
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
                principal = self._from_cookie(scope)
                if principal is None:
                    await _send_json(send, 401, {
                        "error": "unauthenticated",
                        "details": "Sign in, or send 'Authorization: Bearer <token>'.",
                    })
                    return
                # A browser still on an admin-issued temporary password gets
                # nothing until it picks its own, the whole-store download
                # included. Tokens are the owner's and outlive a reset.
                if principal.user.must_change_password:
                    await _send_json(send, *_PASSWORD_CHANGE_REQUIRED)
                    return
        else:  # session
            principal = self._from_cookie(scope)
            if principal is None:
                logger.debug("Unauthenticated %s %s from %s", method, path, ip)
                await _send_json(send, 401, {"error": "unauthenticated", "details": "Sign in first."})
                return
            if principal.user.must_change_password and path not in PASSWORD_GATE_EXEMPT:
                await _send_json(send, *_PASSWORD_CHANGE_REQUIRED)
                return

        scope.setdefault("state", {})["principal"] = principal
        await self.app(scope, receive, send)

    # ── resolvers ──

    def _from_cookie(self, scope: Scope) -> Principal | None:
        raw = _cookie_value(scope, session_cookie_name(scope.get("scheme", "http")))
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

        if self._proxy is not None and self._proxy.matches(token):
            return self._from_proxy(headers, ip, method, path)

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

    def _from_proxy(self, headers: dict[str, str], ip: str, method: str, path: str):
        """The secret checked out, so the proxy is vouching for whoever the
        user header names. No header, no user: that is a misconfigured proxy
        rather than a guess, so it is refused without counting against the
        throttle, and never falls back to a shared or anonymous principal."""
        proxy = self._proxy
        external_id = _header_text(headers, proxy.user_header)
        if not external_id:
            logger.warning("Proxy secret without a %s header (%s %s from %s)", proxy.user_header, method, path, ip)
            return None, (401, {"error": "proxy_user_missing",
                                "details": f"The trusted proxy must name the user in the {proxy.user_header} header."},
                          None)
        role = (_header_text(headers, proxy.role_header) or "").lower() or None
        try:
            principal = self._identity().resolve_proxy_user(
                external_id, display_name=_header_text(headers, proxy.name_header), role=role)
        except IdentityError as e:
            logger.warning("Proxy user refused: %s (%s %s from %s)", e.code, method, path, ip)
            return None, (e.status, {"error": e.code, "details": e.details}, None)
        self._throttle.record_success(client_key(ip))
        logger.debug("Authenticated %s via proxy (%s %s from %s)", principal.actor, method, path, ip)
        return principal, None
