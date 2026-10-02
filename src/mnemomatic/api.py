"""The browser's JSON API under /api.

Everything the web UI needs and nothing an agent needs: session and
first-run, password changes, personal API tokens, user administration, the
HTTPS setup flow, and read-only views of the store (namespaces, pages of
items, one item, its revisions, search, the audit trail). Content writes stay
MCP-only — this API never creates, edits, or deletes a document, fact, or
note.

Conventions, applied by the `_endpoint` wrapper so a handler cannot forget
them:

- Every response is JSON with ``Cache-Control: no-store``; errors are
  ``{"error": "<code>", "details": "<text>"}``.
- A state-changing request must carry an ``Origin`` header matching the
  ``Host`` it arrived on. That is the CSRF defence: the session cookie is
  ``SameSite=Strict``, and a cross-site form post cannot forge a matching
  Origin. No per-form tokens.
- Bodies are JSON objects of at most ``config.API_MAX_BODY`` bytes.
- ``admin=True`` endpoints refuse non-admins with 403 before the handler runs.

AuthMiddleware has already decided who is calling (``request.state.principal``)
and refused anyone it should; handlers here only ever see a principal on
session-class paths. Audit events for identity operations are written here,
with the acting user as actor and the affected user or token as the item.
"""

import json
import logging
import secrets
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from mnemomatic import config
from mnemomatic import db as db_module
from mnemomatic.audit import request_meta, write_event
from mnemomatic.auth import COOKIE_NAME
from mnemomatic.db import _SPEC_BY_ITEM_TYPE
from mnemomatic.tlsca import TlsError
from mnemomatic.identity import (
    SESSION_TTL,
    FirstRun,
    Identity,
    IdentityError,
    LoginThrottle,
    Principal,
    normalize_username,
    validate_password,
)

logger = logging.getLogger("mnemomatic")

# Random per process. The HTTPS confirm step fetches it from the new origin
# to prove the name the admin typed reaches this very server, not a neighbour.
INSTANCE_ID = secrets.token_hex(16)

_ITEM_TYPES = tuple(_SPEC_BY_ITEM_TYPE)
_SEARCH_TYPES = ("all", "documents", "knowledge", "notes")
_SEARCH_MODES = ("hybrid", "fulltext", "semantic")


class ApiError(Exception):
    def __init__(self, code: str, status: int, details: str):
        super().__init__(details)
        self.code, self.status, self.details = code, status, details


# ── Response helpers ────────────────────────────────────────────────────────

def _json(data, status: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(data, status_code=status,
                        headers={"Cache-Control": "no-store", **(headers or {})})


def _error(code: str, status: int, details: str, headers: dict | None = None) -> JSONResponse:
    return _json({"error": code, "details": details}, status, headers)


def _no_content() -> Response:
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


async def _body(request: Request) -> dict:
    """The JSON object in the request body, bounded and validated."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > config.API_MAX_BODY:
        raise ApiError("payload_too_large", 413, f"Request bodies are limited to {config.API_MAX_BODY} bytes.")
    raw = await request.body()
    if len(raw) > config.API_MAX_BODY:
        raise ApiError("payload_too_large", 413, f"Request bodies are limited to {config.API_MAX_BODY} bytes.")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise ApiError("invalid_json", 400, "The request body is not valid JSON.")
    if not isinstance(data, dict):
        raise ApiError("invalid_json", 400, "The request body must be a JSON object.")
    return data


def _str(data: dict, key: str, *, required: bool = True, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ApiError("invalid_field", 400, f"'{key}' must be a string.")
    if required and not value.strip():
        raise ApiError("missing_field", 400, f"'{key}' is required.")
    return value


def _int_param(request: Request, key: str, default: int, lo: int, hi: int) -> int:
    raw = request.query_params.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError("invalid_parameter", 400, f"'{key}' must be an integer.")
    return max(lo, min(value, hi))


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:
        # AuthMiddleware would have refused already; this guards a misuse of
        # the route table rather than a real request path.
        raise ApiError("unauthenticated", 401, "Sign in first.")
    return principal


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _origin_matches(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return False
    host = request.headers.get("host", "")
    candidates = {host.lower()}
    if config.TRUSTED_PROXIES and request.headers.get("x-forwarded-host"):
        candidates.add(request.headers["x-forwarded-host"].split(",")[0].strip().lower())
    return urlsplit(origin).netloc.lower() in candidates


def _is_https(request: Request) -> bool:
    return request.url.scheme == "https"


def _set_session_cookie(resp: Response, raw: str, request: Request) -> None:
    resp.set_cookie(COOKIE_NAME, raw, max_age=int(SESSION_TTL.total_seconds()), path="/",
                    httponly=True, samesite="strict", secure=_is_https(request))


def _clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(COOKIE_NAME, path="/")


# ── Security headers ────────────────────────────────────────────────────────

class SecurityHeadersMiddleware:
    """Browser-facing defence-in-depth on everything but the MCP endpoint.

    The CSP allows scripts only from this origin (the Vite build emits no
    inline script), styles from this origin plus inline attributes (Svelte's
    scoped styles are extracted; `style=` attributes remain), and network
    calls to this origin plus — only while an HTTPS name is pending
    confirmation — the new HTTPS origin, so the browser can fetch the
    instance id from it.

    HSTS is a hostname-wide instruction — browsers ignore the port — so it
    is sent only when `hsts()` says so: once HTTPS is confirmed, and only on
    port 443, where the upgrade a browser performs lands on the TLS listener.
    Sent any earlier, or on another port, it would make the browser upgrade
    http://host:8000 to https://host:8000 — the plain listener — and lock
    the person out of the very page they were confirming from.
    """

    def __init__(self, app: ASGIApp, pending_origin=None, hsts=None):
        self.app = app
        self._pending_origin = pending_origin or (lambda: None)
        self._hsts = hsts or (lambda: False)

    def _csp(self) -> str:
        connect = "'self'"
        extra = self._pending_origin()
        if extra:
            connect += f" {extra}"
        return ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; font-src 'self'; "
                f"connect-src {connect}; manifest-src 'self'; base-uri 'none'; "
                "form-action 'self'; frame-ancestors 'none'")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/mcp" or scope["path"].startswith("/mcp/"):
            await self.app(scope, receive, send)
            return
        https = scope.get("scheme") == "https" and self._hsts()

        async def guarded_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"content-security-policy", self._csp().encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ]
                if https:
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, guarded_send)


# ── Route construction ──────────────────────────────────────────────────────

def build_api_routes(*, identity, db_getter, settings_info, first_run: FirstRun,
                     https=None, throttle: LoginThrottle | None = None) -> Mount:
    """The /api mount.

    identity: callable returning the Identity store.
    db_getter: callable returning the Database.
    settings_info: callable returning the embedding/configuration snapshot.
    first_run: the setup-code holder issued at startup.
    https: the TLS state object (tlsca.TlsState) or None when built-in TLS is off.
    """
    throttle = throttle or LoginThrottle()

    def ident() -> Identity:
        return identity()

    def record(op: str, *, actor: str | None = None, item_type: str | None = None,
               item_id: str | None = None, title: str | None = None, **detail) -> None:
        """An audit row for an identity operation (see audit.write_event)."""
        write_event(db_getter(), op, actor=actor, item_type=item_type, item_id=item_id, title=title, **detail)

    def https_status() -> dict:
        if https is None or config.TLS_MODE == "off":
            # Nothing to manage here; what matters in this mode is whether the
            # proxy in front is trusted, so the page can say so instead of nagging.
            return {"state": "off", "trusted_proxies": list(config.TRUSTED_PROXIES)}
        return https.status()

    def endpoint(fn, *, admin: bool = False):
        async def handler(request: Request):
            try:
                if request.method not in ("GET", "HEAD", "OPTIONS") and not _origin_matches(request):
                    return _error("origin_mismatch", 403,
                                  "State-changing requests must come from this site (Origin must match Host).")
                if admin and not _principal(request).user.is_admin:
                    return _error("forbidden", 403, "Administrators only.")
                return await fn(request)
            except (IdentityError, TlsError, ApiError) as e:
                return _error(e.code, e.status, e.details)
        handler.__name__ = fn.__name__
        return handler

    def route(path: str, fn, methods: list[str], *, admin: bool = False) -> Route:
        return Route(path, endpoint(fn, admin=admin), methods=methods)

    # ── session ──

    async def session(request: Request):
        principal = getattr(request.state, "principal", None)
        base = {"version": settings_info().get("version"), "https": https_status()}
        if principal is None:
            return _json({**base, "authenticated": False, "first_run": ident().count_users() == 0})
        return _json({**base, "authenticated": True, "user": principal.user.public()})

    async def first_run_setup(request: Request):
        data = await _body(request)
        code = _str(data, "setup_code")
        username = _str(data, "username")
        password = _str(data, "password")
        if ident().count_users() > 0:
            return _error("already_set_up", 409, "A user already exists; sign in instead.")
        if not first_run.check(code):
            record("auth.first_run_failed", item_type="user", item_id=normalize_username(username),
                   reason="bad_setup_code")
            return _error("bad_setup_code", 403, "That setup code is not the one in the server log.")
        validate_password(password)
        user, _ = ident().create_user(username, role="admin", display_name=_str(data, "display_name", required=False)
                                      or "Administrator", password=password)
        first_run.clear()
        record("admin.created", actor=user.username, item_type="user", item_id=user.username,
               source="setup_code")
        raw = ident().create_session(user.id)
        record("auth.login", actor=user.username, item_type="user", item_id=user.username)
        resp = _json({"user": user.public()}, 201)
        _set_session_cookie(resp, raw, request)
        return resp

    async def login(request: Request):
        data = await _body(request)
        username = normalize_username(_str(data, "username"))
        password = _str(data, "password")
        ip = _client_ip(request)
        wait = throttle.retry_after(username, ip)
        if wait:
            record("auth.login_failed", item_type="user", item_id=username, reason="throttled")
            return _error("throttled", 429, f"Too many attempts; retry after {wait} seconds.",
                          {"Retry-After": str(wait)})
        try:
            user = ident().authenticate(username, password)
        except IdentityError as e:
            if e.code == "invalid_credentials":
                throttle.record_failure(username, ip)
            record("auth.login_failed", item_type="user", item_id=username, reason=e.code)
            raise
        throttle.record_success(username, ip)
        raw = ident().create_session(user.id)
        record("auth.login", actor=user.username, item_type="user", item_id=user.username)
        resp = _json({"user": user.public()})
        _set_session_cookie(resp, raw, request)
        return resp

    async def logout(request: Request):
        principal = _principal(request)
        ident().delete_session(request.cookies.get(COOKIE_NAME, ""))
        record("auth.logout", item_type="user", item_id=principal.user.username)
        resp = _no_content()
        _clear_session_cookie(resp)
        return resp

    async def password(request: Request):
        principal = _principal(request)
        data = await _body(request)
        ident().change_password(principal.user.id, _str(data, "current_password"),
                                _str(data, "new_password"),
                                keep_session=request.cookies.get(COOKIE_NAME))
        record("password.changed", item_type="user", item_id=principal.user.username)
        return _no_content()

    # ── my tokens ──

    async def tokens_list(request: Request):
        principal = _principal(request)
        return _json({"tokens": ident().list_tokens(principal.user.id)})

    async def tokens_create(request: Request):
        principal = _principal(request)
        data = await _body(request)
        rec, raw = ident().create_token(principal.user.id, _str(data, "name"),
                                        data.get("expires_in_days", 0))
        record("token.created", item_type="token", item_id=str(rec["id"]), title=rec["name"],
               hint=rec["hint"], expires_at=rec["expires_at"])
        return _json({**rec, "token": raw}, 201)

    async def tokens_revoke(request: Request):
        principal = _principal(request)
        rec = ident().revoke_token(principal.user.id, request.path_params["token_id"])
        if rec is None:
            return _error("not_found", 404, "No such token.")
        record("token.revoked", item_type="token", item_id=str(rec["id"]), title=rec["name"], hint=rec["hint"])
        return _no_content()

    # ── user administration ──

    async def users_list(request: Request):
        return _json({"users": ident().list_users()})

    async def users_create(request: Request):
        data = await _body(request)
        role = _str(data, "role", required=False, default="user") or "user"
        user, temp = ident().create_user(_str(data, "username"), role=role,
                                         display_name=_str(data, "display_name", required=False))
        record("user.created", item_type="user", item_id=user.username, role=user.role)
        return _json({"user": user.public(), "temporary_password": temp,
                      "expires_at": user.temp_password_expires_at}, 201)

    def target(request: Request):
        user = ident().get_user(request.path_params["user_id"])
        if user is None:
            raise ApiError("not_found", 404, "No such user.")
        return user

    async def users_reset_password(request: Request):
        me = _principal(request).user
        user = target(request)
        temp, expires = ident().reset_password(user.id, acting_user_id=me.id)
        record("password.reset", item_type="user", item_id=user.username, expires_at=expires)
        return _json({"temporary_password": temp, "expires_at": expires})

    async def users_set_active(request: Request):
        me = _principal(request).user
        user = target(request)
        data = await _body(request)
        active = data.get("active")
        if not isinstance(active, bool):
            raise ApiError("invalid_field", 400, "'active' must be true or false.")
        updated = ident().set_active(user.id, active, acting_user_id=me.id)
        record("user.reactivated" if active else "user.deactivated", item_type="user", item_id=user.username)
        return _json({"user": updated.public()})

    async def users_set_role(request: Request):
        me = _principal(request).user
        user = target(request)
        data = await _body(request)
        role = _str(data, "role")
        updated = ident().set_role(user.id, role, acting_user_id=me.id)
        if updated.role != user.role:
            record("user.role_changed", item_type="user", item_id=user.username,
                   **{"from": user.role, "to": updated.role})
        return _json({"user": updated.public()})

    async def users_delete(request: Request):
        me = _principal(request).user
        user = target(request)
        ident().delete_user(user.id, acting_user_id=me.id)
        record("user.deleted", item_type="user", item_id=user.username, role=user.role)
        return _no_content()

    # ── audit ──

    async def audit(request: Request):
        q = request.query_params
        limit = _int_param(request, "limit", 50, 1, config.MAX_LIST_LIMIT)
        offset = _int_param(request, "offset", 0, 0, 10_000_000)
        filters = {k: (q.get(k) or None) for k in ("item_type", "item_id", "namespace", "op", "actor")}
        db = db_getter()
        try:
            events = db.list_audit(**filters, limit=limit, offset=offset)
            total = db.count_audit(**filters)
        except ValueError as e:
            raise ApiError("invalid_filter", 400, str(e))
        return _json({"events": events, "total": total, "limit": limit, "offset": offset})

    # ── HTTPS ──

    def require_https():
        if https is None or config.TLS_MODE == "off":
            raise ApiError("tls_disabled", 409,
                           "Built-in TLS is off (MNEMOMATIC_TLS=off); terminate TLS in your proxy instead.")
        return https

    async def https_get(request: Request):
        return _json(https_status())

    async def https_name(request: Request):
        data = await _body(request)
        status = require_https().set_name(_str(data, "name"))
        record("https.changed", item_type="https", item_id=status.get("name"), state=status.get("state"),
               fingerprint=status.get("ca_fingerprint"))
        return _json(status)

    async def https_confirm(request: Request):
        data = await _body(request)
        status = require_https().confirm(_str(data, "name"), _str(data, "instance_id"), INSTANCE_ID)
        record("https.changed", item_type="https", item_id=status.get("name"), state=status.get("state"),
               fingerprint=status.get("ca_fingerprint"))
        return _json(status)

    async def https_disable(request: Request):
        status = require_https().disable()
        record("https.changed", item_type="https", item_id=status.get("name"), state=status.get("state"))
        return _json(status)

    async def instance_id(request: Request):
        return _json({"instance_id": INSTANCE_ID}, headers={"Access-Control-Allow-Origin": "*"})

    # ── connect and settings ──

    async def connect(request: Request):
        origin = f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"
        status = https_status()
        info = {
            "origin": origin,
            "mcp_url": f"{origin}/mcp",
            "compact_url": f"{origin}/mcp?compact=true",
            "export_url": f"{origin}/export",
            "https": status,
            "builtin_ca": bool(status.get("ca_fingerprint")),
            "ca_url": f"{origin}/ca.crt" if status.get("ca_fingerprint") else None,
            "ca_fingerprint": status.get("ca_fingerprint"),
            "token_prefix": "mnm_",
        }
        if status.get("https_url"):
            info["https_mcp_url"] = f"{status['https_url']}/mcp"
        return _json(info)

    async def settings(request: Request):
        info = dict(settings_info())
        info["tls"] = https_status()
        info["audit_keep_days"] = db_module.AUDIT_KEEP_DAYS
        info["backup"] = {"dir": config.BACKUP_DIR or None, "interval_hours": config.BACKUP_INTERVAL_HOURS,
                          "keep": config.BACKUP_KEEP} if config.BACKUP_DIR else None
        info["trusted_proxies"] = config.TRUSTED_PROXIES
        return _json(info)

    # ── read-only store views ──

    async def namespaces(request: Request):
        counts = db_getter().namespace_counts()
        return _json({"namespaces": [
            {"name": ns, "documents": c["documents"], "knowledge": c["knowledge"], "notes": c["notes"]}
            for ns, c in counts.items()
        ]})

    def item_type_param(value: str | None) -> str:
        if value not in _ITEM_TYPES:
            raise ApiError("invalid_type", 400, f"'type' must be one of: {', '.join(_ITEM_TYPES)}.")
        return value

    async def items(request: Request):
        namespace = request.query_params.get("namespace", "")
        if not namespace:
            raise ApiError("missing_parameter", 400, "'namespace' is required.")
        item_type = item_type_param(request.query_params.get("type"))
        limit = _int_param(request, "limit", 50, 1, config.MAX_LIST_LIMIT)
        offset = _int_param(request, "offset", 0, 0, 10_000_000)
        rows, total = db_getter().list_page(item_type, namespace, limit, offset)
        return _json({"namespace": namespace, "type": item_type, "items": rows,
                      "total": total, "limit": limit, "offset": offset})

    def fetch_item(request: Request):
        from mnemomatic.tools_content import _OPS
        item_type = item_type_param(request.path_params["item_type"])
        obj = _OPS[item_type].get(db_getter(), request.path_params["item_id"])
        if obj is None:
            raise ApiError("not_found", 404, f"No {item_type} with that id.")
        return item_type, obj

    async def item(request: Request):
        item_type, obj = fetch_item(request)
        return _json({"type": item_type, "item": obj.model_dump(mode="json")})

    async def item_revisions(request: Request):
        item_type, obj = fetch_item(request)
        rows = db_getter().list_revisions(item_type=item_type, item_id=obj.id, limit=50)
        return _json({"revisions": rows})

    async def item_related(request: Request):
        from mnemomatic import tools_search
        item_type, obj = fetch_item(request)
        result = tools_search.related(item_type, obj.id, limit=_int_param(request, "limit", 5, 1, 20))
        if "error" in result:
            return _json({"related": [], "unavailable": result["error"]})
        return _json({"related": result["related"]})

    async def search(request: Request):
        from mnemomatic import tools_search
        q = request.query_params
        query = q.get("q", "").strip()
        if not query:
            raise ApiError("missing_parameter", 400, "'q' is required.")
        content_type = q.get("type") or "all"
        if content_type not in _SEARCH_TYPES:
            raise ApiError("invalid_type", 400, f"'type' must be one of: {', '.join(_SEARCH_TYPES)}.")
        mode = q.get("mode") or "hybrid"
        if mode not in _SEARCH_MODES:
            raise ApiError("invalid_mode", 400, f"'mode' must be one of: {', '.join(_SEARCH_MODES)}.")
        results = tools_search.search(query=query, content_type=content_type,
                                      namespace=q.get("namespace") or None,
                                      limit=_int_param(request, "limit", 20, 1, config.MAX_SEARCH_LIMIT),
                                      mode=mode)
        if results and "error" in results[0]:
            raise ApiError("search_failed", 400, f"{results[0]['error']}: {results[0].get('details', '')}")
        degraded = any("_metadata" in r for r in results)
        return _json({"results": [r for r in results if "_metadata" not in r], "degraded": degraded})

    return Mount("/api", routes=[
        route("/session", session, ["GET"]),
        route("/first-run", first_run_setup, ["POST"]),
        route("/login", login, ["POST"]),
        route("/logout", logout, ["POST"]),
        route("/password", password, ["POST"]),
        route("/me/tokens", tokens_list, ["GET"]),
        route("/me/tokens", tokens_create, ["POST"]),
        route("/me/tokens/{token_id:int}", tokens_revoke, ["DELETE"]),
        route("/admin/users", users_list, ["GET"], admin=True),
        route("/admin/users", users_create, ["POST"], admin=True),
        route("/admin/users/{user_id:int}/reset-password", users_reset_password, ["POST"], admin=True),
        route("/admin/users/{user_id:int}/active", users_set_active, ["POST"], admin=True),
        route("/admin/users/{user_id:int}/role", users_set_role, ["POST"], admin=True),
        route("/admin/users/{user_id:int}", users_delete, ["DELETE"], admin=True),
        route("/audit", audit, ["GET"]),
        route("/admin/https", https_get, ["GET"], admin=True),
        route("/admin/https/name", https_name, ["POST"], admin=True),
        route("/admin/https/confirm", https_confirm, ["POST"], admin=True),
        route("/admin/https/disable", https_disable, ["POST"], admin=True),
        route("/instance-id", instance_id, ["GET"]),
        route("/connect", connect, ["GET"]),
        route("/settings", settings, ["GET"]),
        route("/namespaces", namespaces, ["GET"]),
        route("/items", items, ["GET"]),
        route("/items/{item_type}/{item_id}", item, ["GET"]),
        route("/items/{item_type}/{item_id}/revisions", item_revisions, ["GET"]),
        route("/items/{item_type}/{item_id}/related", item_related, ["GET"]),
        route("/search", search, ["GET"]),
    ])
