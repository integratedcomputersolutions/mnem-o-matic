"""Pins the published MCP surface: which tools exist, and in what order.

The tools live in four modules now, and importing those modules is what
registers them — so the import order in server.py is the order clients see.
An import sorter alphabetising that block would rearrange the published tool
list without touching a single tool, and nothing else in the suite would
notice. This test is the guard against that.

It also pins the resource templates and prompts, so a tool quietly moving
between modules, losing its decorator, or gaining one shows up here.
"""

import asyncio
import unittest

import mnemomatic.server as server

# Registration order, not alphabetical: content, then search, then history,
# then admin — matching the import order in server.py.
EXPECTED_TOOLS = [
    "store_document",
    "store_knowledge",
    "update_document",
    "update_knowledge",
    "delete_document",
    "delete_knowledge",
    "store_note",
    "update_note",
    "delete_note",
    "tag",
    "search",
    "list_items",
    "read",
    "related",
    "fact_history",
    "list_revisions",
    "list_audit",
    "restore",
    "consolidation_report",
    "embedding_info",
    "delete_namespace",
    "rename_namespace",
]

EXPECTED_RESOURCES = {
    "mnemomatic://document/{id}",
    "mnemomatic://documents/{namespace}",
    "mnemomatic://knowledge-entry/{id}",
    "mnemomatic://knowledge/{namespace}",
    "mnemomatic://note/{id}",
    "mnemomatic://notes/{namespace}",
}

EXPECTED_PROMPTS = {"briefing", "consolidate"}


class TestToolRegistration(unittest.TestCase):
    def test_tools_and_their_order(self):
        tools = [t.name for t in asyncio.run(server.mcp.list_tools())]
        self.assertEqual(tools, EXPECTED_TOOLS)

    def test_every_tool_has_a_description(self):
        # The docstring is the agent-facing API; a tool without one is a bug.
        for tool in asyncio.run(server.mcp.list_tools()):
            with self.subTest(tool=tool.name):
                self.assertTrue((tool.description or "").strip(), f"{tool.name} has no description")

    def test_resource_templates(self):
        templates = {r.uriTemplate for r in asyncio.run(server.mcp.list_resource_templates())}
        self.assertEqual(templates, EXPECTED_RESOURCES)

    def test_prompts(self):
        self.assertEqual({p.name for p in asyncio.run(server.mcp.list_prompts())}, EXPECTED_PROMPTS)

    def test_health_resource_is_registered(self):
        resources = {str(r.uri) for r in asyncio.run(server.mcp.list_resources())}
        self.assertIn("mnemomatic://health", resources)


if __name__ == "__main__":
    unittest.main()
