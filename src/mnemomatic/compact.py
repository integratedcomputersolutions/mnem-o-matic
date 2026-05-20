"""Compact tool descriptions for small-context LLMs.

When a client appends ?compact=true to the MCP endpoint URL, CompactToolsMiddleware
intercepts tools/list responses and replaces verbose docstrings with the short
descriptions defined in _COMPACT_DESCRIPTIONS, strips parameter descriptions, and
restores concise hints only for parameters with constrained valid values.

All other requests (tool calls, SSE streams) are forwarded unchanged.
"""

import json
from urllib.parse import parse_qs, urlencode

from starlette.types import ASGIApp, Receive, Scope, Send

_COMPACT_DESCRIPTIONS: dict[str, str] = {
    "search":           "Search memory by keyword or concept. Modes: hybrid (default), fulltext, semantic.",
    "read":             "Fetch full content of an item by item_type (document/knowledge/note) and id.",
    "store_document":   "Save long-form content (specs, docs, code). Upsert by namespace+title.",
    "store_knowledge":  "Save a single atomic fact or decision. Upsert by namespace+subject.",
    "store_note":       "Save informal or rough content. Upsert by namespace+title.",
    "update_document":  "Update specific fields of a document by id. Only supplied fields change.",
    "update_knowledge": "Update specific fields of a knowledge entry by id. Only supplied fields change.",
    "update_note":      "Update specific fields of a note by id. Only supplied fields change.",
    "delete_document":  "Permanently delete a document by id.",
    "delete_knowledge": "Permanently delete a knowledge entry by id.",
    "delete_note":      "Permanently delete a note by id.",
    "tag":              "Add or remove tags on any item without changing other fields.",
    "rename_namespace": "Rename a namespace across all items. Works as merge if target exists.",
    "delete_namespace": "Permanently delete all items in a namespace.",
}

# Param descriptions only for constrained/non-obvious values; all others are stripped.
_COMPACT_PARAMS: dict[str, dict[str, str]] = {
    "search":          {"mode": "hybrid|fulltext|semantic", "content_type": "all|documents|knowledge|notes", "namespace": "filter by namespace; omit for global search"},
    "read":            {"item_type": "document|knowledge|note"},
    "tag":             {"item_type": "document|knowledge|note"},
    "store_knowledge": {"confidence": "0.0-1.0"},
}

# Optional tool subset: remove tool names to hide them in compact mode.
# None exposes all tools.
_COMPACT_TOOL_SUBSET: set[str] | None = None


def _simplify_prop(prop: dict) -> dict:
    """Strip schema noise from a single parameter property."""
    # Unwrap anyOf: [{type: X}, {type: null}] → {type: X}
    if "anyOf" in prop:
        non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
        prop = non_null[0] if len(non_null) == 1 else {"anyOf": non_null}

    result: dict = {}
    if "type" in prop:
        result["type"] = prop["type"]
    if prop.get("type") == "array":
        result["type"] = "array"  # drop items detail
    if "anyOf" in prop:
        result["anyOf"] = prop["anyOf"]
    return result


def _compact_tools_body(body: bytes) -> bytes:
    """Replace verbose tool descriptions in a tools/list JSON-RPC response."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body

    tools = data.get("result", {}).get("tools")
    if not isinstance(tools, list):
        return body

    if _COMPACT_TOOL_SUBSET is not None:
        tools = [t for t in tools if t.get("name") in _COMPACT_TOOL_SUBSET]
        data["result"]["tools"] = tools

    for tool in tools:
        name = tool.get("name", "")
        if name in _COMPACT_DESCRIPTIONS:
            tool["description"] = _COMPACT_DESCRIPTIONS[name]

        schema = tool.get("inputSchema", {})
        param_hints = _COMPACT_PARAMS.get(name, {})
        simplified: dict = {}
        for param, prop in schema.get("properties", {}).items():
            simplified[param] = _simplify_prop(prop)
            if param in param_hints:
                simplified[param]["description"] = param_hints[param]

        tool["inputSchema"] = {
            "type": "object",
            "properties": simplified,
            **({"required": schema["required"]} if "required" in schema else {}),
        }

    return json.dumps(data, separators=(",", ":")).encode()


class CompactToolsMiddleware:
    """Serve compact tool descriptions when ?compact=true is in the request URL.

    Strips the query parameter before forwarding to FastMCP and post-processes
    tools/list responses. SSE streams are passed through unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        params = parse_qs(scope.get("query_string", b"").decode())
        if "compact" not in params:
            await self.app(scope, receive, send)
            return

        # Strip ?compact from query string before FastMCP sees it
        remaining = {k: v for k, v in params.items() if k != "compact"}
        scope = {**scope, "query_string": urlencode(remaining, doseq=True).encode()}

        is_sse = False
        start_msg: dict = {}
        body_chunks: list[bytes] = []

        async def send_wrapper(message: dict) -> None:
            nonlocal is_sse, start_msg

            if message["type"] == "http.response.start":
                start_msg = message
                if b"text/event-stream" in dict(message["headers"]).get(b"content-type", b""):
                    is_sse = True
                    await send(message)
                return

            if message["type"] == "http.response.body":
                if is_sse:
                    await send(message)
                    return
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    body = _compact_tools_body(b"".join(body_chunks))
                    headers = [
                        (b"content-length", str(len(body)).encode()) if k == b"content-length" else (k, v)
                        for k, v in start_msg["headers"]
                    ]
                    await send({**start_msg, "headers": headers})
                    await send({"type": "http.response.body", "body": body, "more_body": False})
                return

            await send(message)

        await self.app(scope, receive, send_wrapper)
