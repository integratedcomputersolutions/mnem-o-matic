"""Integration tests for Mnem-O-matic MCP API.

Requires the server to be running at http://localhost:8686/mcp.
Start with: docker compose up --build -d

Run with: python -m unittest tests/test_mcp_api.py -v
"""

import unittest

from mnemomatic_cli._mcp_client import MCPClient

BASE_URL = "http://localhost:8686/mcp"
NS = "test-integration"


def _search_ids(client: MCPClient, query: str) -> list[str]:
    """Search and return a flat list of result IDs."""
    results = client.call_tool("search", {
        "query": query,
        "namespace": NS,
        "mode": "fulltext",
    })
    if isinstance(results, dict):
        results = [results]
    return [r["id"] for r in results]


class TestDocuments(unittest.TestCase):

    def setUp(self):
        self.client = MCPClient(BASE_URL)
        self._cleanup_ids = []

    def tearDown(self):
        for doc_id in self._cleanup_ids:
            self.client.call_tool("delete_document", {"id": doc_id})

    def test_store_and_read(self):
        """Store a document and find it via search."""
        result = self.client.call_tool("store_document", {
            "namespace": NS,
            "title": "Store Read Test Doc",
            "content": "Created by integration test.",
            "tags": ["test"],
        })
        self._cleanup_ids.append(result["id"])

        self.assertTrue(result["created"])
        self.assertEqual(result["namespace"], NS)
        self.assertEqual(result["title"], "Store Read Test Doc")
        self.assertIn(result["id"], _search_ids(self.client, "Store Read Test Doc"))

    def test_duplicate_upserts(self):
        """Same namespace+title should update in place, not create a duplicate."""
        r1 = self.client.call_tool("store_document", {
            "namespace": NS,
            "title": "Upsert Doc Target",
            "content": "Version 1",
        })
        self._cleanup_ids.append(r1["id"])
        self.assertTrue(r1["created"])

        r2 = self.client.call_tool("store_document", {
            "namespace": NS,
            "title": "Upsert Doc Target",
            "content": "Version 2",
        })
        self.assertFalse(r2["created"])
        self.assertEqual(r2["id"], r1["id"])

        # Only one result with this ID
        ids = _search_ids(self.client, "Upsert Doc Target")
        self.assertEqual(ids.count(r1["id"]), 1)

    def test_delete(self):
        """Deleting a document removes it from search; deleting again returns false."""
        r = self.client.call_tool("store_document", {
            "namespace": NS,
            "title": "Doc To Delete",
            "content": "Will be removed.",
        })
        doc_id = r["id"]

        delete_result = self.client.call_tool("delete_document", {"id": doc_id})
        self.assertTrue(delete_result["deleted"])
        self.assertNotIn(doc_id, _search_ids(self.client, "Doc To Delete"))

        delete_again = self.client.call_tool("delete_document", {"id": doc_id})
        self.assertFalse(delete_again["deleted"])


class TestKnowledge(unittest.TestCase):

    def setUp(self):
        self.client = MCPClient(BASE_URL)
        self._cleanup_ids = []

    def tearDown(self):
        for k_id in self._cleanup_ids:
            self.client.call_tool("delete_knowledge", {"id": k_id})

    def test_store_and_read(self):
        """Store a knowledge entry and find it via search."""
        result = self.client.call_tool("store_knowledge", {
            "namespace": NS,
            "subject": "Store Read Test Fact",
            "fact": "Mnem-O-matic tests run against the live MCP API.",
            "tags": ["test"],
        })
        self._cleanup_ids.append(result["id"])

        self.assertTrue(result["created"])
        self.assertIn(result["id"], _search_ids(self.client, "Store Read Test Fact"))

    def test_duplicate_upserts(self):
        """Same namespace+subject should update in place, not create a duplicate."""
        r1 = self.client.call_tool("store_knowledge", {
            "namespace": NS,
            "subject": "Upsert Knowledge Target",
            "fact": "Original fact",
        })
        self._cleanup_ids.append(r1["id"])
        self.assertTrue(r1["created"])

        r2 = self.client.call_tool("store_knowledge", {
            "namespace": NS,
            "subject": "Upsert Knowledge Target",
            "fact": "Updated fact",
        })
        self.assertFalse(r2["created"])
        self.assertEqual(r2["id"], r1["id"])

        ids = _search_ids(self.client, "Upsert Knowledge Target")
        self.assertEqual(ids.count(r1["id"]), 1)

    def test_delete(self):
        """Deleting a knowledge entry removes it from search; deleting again returns false."""
        r = self.client.call_tool("store_knowledge", {
            "namespace": NS,
            "subject": "Ephemeral Fact",
            "fact": "This fact will be deleted.",
        })
        k_id = r["id"]

        delete_result = self.client.call_tool("delete_knowledge", {"id": k_id})
        self.assertTrue(delete_result["deleted"])
        self.assertNotIn(k_id, _search_ids(self.client, "Ephemeral Fact"))

        delete_again = self.client.call_tool("delete_knowledge", {"id": k_id})
        self.assertFalse(delete_again["deleted"])


class TestNotes(unittest.TestCase):

    def setUp(self):
        self.client = MCPClient(BASE_URL)
        self._cleanup_ids = []

    def tearDown(self):
        for note_id in self._cleanup_ids:
            self.client.call_tool("delete_note", {"id": note_id})

    def test_store_and_read(self):
        """Store a note and find it via search."""
        result = self.client.call_tool("store_note", {
            "namespace": NS,
            "title": "Store Read Test Note",
            "content": "A quick thought from the integration test.",
            "tags": ["test"],
        })
        self._cleanup_ids.append(result["id"])

        self.assertTrue(result["created"])
        self.assertEqual(result["namespace"], NS)
        self.assertEqual(result["title"], "Store Read Test Note")
        self.assertIn(result["id"], _search_ids(self.client, "Store Read Test Note"))

    def test_duplicate_upserts(self):
        """Same namespace+title should update in place, not create a duplicate."""
        r1 = self.client.call_tool("store_note", {
            "namespace": NS,
            "title": "Upsert Note Target",
            "content": "Version 1",
        })
        self._cleanup_ids.append(r1["id"])
        self.assertTrue(r1["created"])

        r2 = self.client.call_tool("store_note", {
            "namespace": NS,
            "title": "Upsert Note Target",
            "content": "Version 2",
        })
        self.assertFalse(r2["created"])
        self.assertEqual(r2["id"], r1["id"])

        ids = _search_ids(self.client, "Upsert Note Target")
        self.assertEqual(ids.count(r1["id"]), 1)

    def test_delete(self):
        """Deleting a note removes it from search; deleting again returns false."""
        r = self.client.call_tool("store_note", {
            "namespace": NS,
            "title": "Note To Delete",
            "content": "Will be removed.",
        })
        note_id = r["id"]

        delete_result = self.client.call_tool("delete_note", {"id": note_id})
        self.assertTrue(delete_result["deleted"])
        self.assertNotIn(note_id, _search_ids(self.client, "Note To Delete"))

        delete_again = self.client.call_tool("delete_note", {"id": note_id})
        self.assertFalse(delete_again["deleted"])


if __name__ == "__main__":
    unittest.main()
