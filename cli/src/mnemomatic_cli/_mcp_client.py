"""Minimal MCP client over Streamable HTTP JSON-RPC."""

import json
import urllib.error
import urllib.request
import urllib.parse
from importlib.metadata import version, PackageNotFoundError

_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB

try:
    _VERSION = version("mnemomatic")
except PackageNotFoundError:
    _VERSION = "0.0.0"

_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _check_scheme(url: str) -> None:
    """Only http and https: urllib would otherwise happily read file:// or
    fetch ftp:// with the token in hand."""
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {scheme!r} — use http or https")


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or {"http": 80, "https": 443}.get(scheme)


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only within the origin the request went to.

    urllib's own handler copies every header but Content-* to wherever the
    server points, whatever the host or scheme, so a proxy or a mistyped URL
    answering 30x would receive the bearer token. Returning None makes
    urllib raise the HTTPError instead; _describe_http_error says where the
    server tried to send us."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(newurl) != _origin(req.full_url):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(req: urllib.request.Request, *, timeout: float, ssl_context=None):
    """urlopen for requests that carry the token: same-origin redirects only."""
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context),
                                         _SameOriginRedirects())
    return opener.open(req, timeout=timeout)


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """One line a person can act on. The server's JSON error carries a code
    and a sentence of details (and, when plain HTTP is closed, the HTTPS URL
    to use instead); fall back to the status line when it does not."""
    body = {}
    try:
        body = json.loads(exc.read().decode() or "{}")
    except Exception:
        pass
    code, details = body.get("error"), body.get("details")
    if 300 <= exc.code < 400 and exc.headers.get("Location"):
        target = urllib.parse.urljoin(exc.url, exc.headers["Location"])
        return (f"Server redirected to {target} — not following it with your token. "
                f"If that address is right, use it as the server URL")
    if code == "https_required":
        return f"Plain HTTP is closed on this server — use {body.get('https_url') or 'the HTTPS URL'} instead"
    if exc.code in (401, 403):
        hint = f": {details}" if details else ""
        return f"Authentication failed — check --token (or MNEMOMATIC_TOKEN){hint}"
    return f"HTTP {exc.code} from server: {details or exc.reason}"


class MCPClient:
    """Minimal MCP client over Streamable HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000/mcp", api_key: str = "",
                 ssl_context=None):
        _check_scheme(base_url)
        self.base_url = base_url
        self.api_key = api_key
        self._ssl_context = ssl_context
        self.session_id = None
        self._next_id = 1
        self._initialize()

    def _send(self, payload: dict) -> dict | None:
        headers = dict(_HEADERS)
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url, data=data, headers=headers)
        try:
            resp = _open(req, timeout=30, ssl_context=self._ssl_context)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_describe_http_error(exc)) from exc
        except OSError as exc:
            raise RuntimeError(f"Cannot connect to server at {self.base_url}") from exc
        if not self.session_id:
            self.session_id = resp.headers.get("mcp-session-id")
        raw = resp.read(_MAX_RESPONSE_BYTES).decode()
        return json.loads(raw) if raw else None

    def _initialize(self):
        self._send({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "mnemomatic-cli", "version": _VERSION},
            },
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _rpc(self, method: str, params: dict) -> dict:
        self._next_id += 1
        body = self._send({
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        })
        if "error" in body:
            raise RuntimeError(body["error"].get("message", str(body["error"])))
        return body["result"]

    def call_tool(self, name: str, arguments: dict) -> dict | list:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return _one_or_all([json.loads(item["text"]) for item in result["content"]])

    def read_resource(self, uri: str) -> str | list[str]:
        result = self._rpc("resources/read", {"uri": uri})
        return _one_or_all([item["text"] for item in result["contents"]])


def _one_or_all(values: list):
    """A single result unwrapped; several as a list."""
    return values[0] if len(values) == 1 else values
