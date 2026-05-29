"""Tests for compact tool descriptions (?compact=true) middleware.

The drift guard is the important part: _COMPACT_DESCRIPTIONS is a hand-maintained
shrunk copy of the tool docstrings, so it must stay in sync with the registered
tools or compact mode silently falls back to verbose text for the missing ones.
"""

import asyncio
import json
import unittest

from mnemomatic import server
from mnemomatic.compact import (
    _COMPACT_DESCRIPTIONS,
    _COMPACT_PARAMS,
    _compact_tools_body,
    _simplify_prop,
)


def _registered_tools():
    return asyncio.run(server.mcp.list_tools())


def _registered_tool_names():
    return {t.name for t in _registered_tools()}


class TestCompactDescriptionsDrift(unittest.TestCase):
    def test_descriptions_cover_exactly_registered_tools(self):
        """Every registered tool has a compact description and vice versa."""
        self.assertEqual(set(_COMPACT_DESCRIPTIONS), _registered_tool_names())

    def test_param_hints_reference_known_tools(self):
        """_COMPACT_PARAMS only carries hints for tools that exist."""
        self.assertTrue(set(_COMPACT_PARAMS) <= set(_COMPACT_DESCRIPTIONS))


class TestCompactToolsBody(unittest.TestCase):
    def _tools_list_body(self) -> bytes:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                    for t in _registered_tools()
                ]
            },
        }
        return json.dumps(response).encode()

    def test_descriptions_replaced_and_params_stripped(self):
        out = json.loads(_compact_tools_body(self._tools_list_body()))
        by_name = {t["name"]: t for t in out["result"]["tools"]}

        # Verbose description swapped for the compact one
        self.assertEqual(by_name["search"]["description"], _COMPACT_DESCRIPTIONS["search"])

        props = by_name["search"]["inputSchema"]["properties"]
        # Constrained param keeps a concise hint
        self.assertEqual(props["mode"]["description"], _COMPACT_PARAMS["search"]["mode"])
        # Free-form param loses its verbose description entirely
        self.assertNotIn("description", props["query"])

    def test_non_tools_body_passes_through_unchanged(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        self.assertEqual(_compact_tools_body(body), body)

    def test_invalid_json_passes_through_unchanged(self):
        body = b"not valid json"
        self.assertEqual(_compact_tools_body(body), body)


class TestSimplifyProp(unittest.TestCase):
    def test_unwraps_nullable_anyof(self):
        self.assertEqual(
            _simplify_prop({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            {"type": "string"},
        )

    def test_drops_array_items_detail(self):
        self.assertEqual(
            _simplify_prop({"type": "array", "items": {"type": "string"}}),
            {"type": "array"},
        )

    def test_strips_verbose_description(self):
        self.assertEqual(
            _simplify_prop({"type": "string", "description": "a long verbose hint"}),
            {"type": "string"},
        )


if __name__ == "__main__":
    unittest.main()
