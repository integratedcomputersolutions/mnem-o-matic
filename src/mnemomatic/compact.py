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
    "list_items":       "List item summaries in a namespace, newest first. Paginate with limit/offset; response includes total.",
    "store_document":   "Save long-form content (specs, docs, code). Upsert by namespace+title.",
    "store_knowledge":  "Save a single atomic fact or decision. Upsert by namespace+subject.",
    "store_note":       "Save informal or rough content. Upsert by namespace+title.",
    "update_document":  "Update specific fields of a document by id. Only supplied fields change.",
    "update_knowledge": "Update specific fields of a knowledge entry by id. Only supplied fields change.",
    "update_note":      "Update specific fields of a note by id. Only supplied fields change.",
    "delete_document":  "Delete a document by id. Undoable via list_revisions + restore.",
    "delete_knowledge": "Delete a knowledge entry by id. For changed facts, store the new fact instead (keeps history). Undoable via restore.",
    "delete_note":      "Delete a note by id. Undoable via list_revisions + restore.",
    "tag":              "Add or remove tags on any item without changing other fields.",
    "rename_namespace": "Rename a namespace across all items. Merges into existing target; moved items win title/subject conflicts.",
    "delete_namespace": "Delete all items in a namespace. Items individually restorable via revisions; no bulk undo.",
    "list_revisions":   "List saved prior versions of items (captured on every update/delete), newest first. Filter by item_type/item_id/namespace.",
    "restore":          "Restore an item to a revision from list_revisions: rolls back an update or recreates a deleted item.",
    "fact_history":     "Timeline of a knowledge fact by namespace+subject: current entry first, then superseded versions newest first.",
    "consolidation_report": "Consolidation candidates for a namespace: near-duplicate clusters (vector similarity) and stale never-retrieved items.",
    "list_audit":       "Audit trail of write operations, newest first. Filter by item_type/item_id/namespace/op.",
}

# Param descriptions only for constrained/non-obvious values; all others are stripped.
_COMPACT_PARAMS: dict[str, dict[str, str]] = {
    "search":          {"mode": "hybrid|fulltext|semantic", "content_type": "all|documents|knowledge|notes", "namespace": "filter by namespace; omit for global search"},
    "read":            {"item_type": "document|knowledge|note"},
    "list_items":      {"item_type": "document|knowledge|note"},
    "tag":             {"item_type": "document|knowledge|note"},
    "store_knowledge": {"confidence": "0.0-1.0"},
    "list_revisions":  {"item_type": "document|knowledge|note"},
    "list_audit":      {"item_type": "document|knowledge|note"},
}

def _simplify_prop(prop: dict) -> dict:
    """Strip schema noise from a single parameter property."""
    # Unwrap anyOf: [{type: X}, {type: null}] → {type: X}
    if "anyOf" in prop:
        non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
        prop = non_null[0] if len(non_null) == 1 else {"anyOf": non_null}

    result: dict = {}
    if "type" in prop:
        result["type"] = prop["type"]  # array/object detail (items, properties) is dropped
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
