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


class MCPClient:
    """Minimal MCP client over Streamable HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000/mcp", api_key: str = ""):
        scheme = urllib.parse.urlparse(base_url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme {scheme!r} — use http or https")
        self.base_url = base_url
        self.api_key = api_key
        self.session_id = None
        self._next_id = 1
        self._initialize()

    def _send(self, payload: dict) -> dict | None:
        if self.session_id or self.api_key:
            headers = dict(_HEADERS)
            if self.session_id:
                headers["mcp-session-id"] = self.session_id
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers = _HEADERS
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url, data=data, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError("Authentication failed — check --api-key") from exc
            raise RuntimeError(f"HTTP {exc.code} from server: {exc.reason}") from exc
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

    @staticmethod
    def _check_error(body: dict) -> None:
        if "error" in body:
            raise RuntimeError(body["error"].get("message", str(body["error"])))

    def _rpc(self, method: str, params: dict) -> dict:
        self._next_id += 1
        body = self._send({
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        })
        self._check_error(body)
        return body["result"]

    def call_tool(self, name: str, arguments: dict) -> dict | list:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        items = result["content"]
        if len(items) == 1:
            return json.loads(items[0]["text"])
        return [json.loads(item["text"]) for item in items]

    def read_resource(self, uri: str) -> str | list[str]:
        result = self._rpc("resources/read", {"uri": uri})
        items = result["contents"]
        if len(items) == 1:
            return items[0]["text"]
        return [item["text"] for item in items]
