"""Tests for the list_items MCP tool (pagination over namespace contents).

Drives server.list_items against a real in-memory Database so the tool's
validation, clamping, and response shape are exercised end to end.
"""

import unittest
from unittest.mock import patch

import mnemomatic.server as server
from mnemomatic import config
from mnemomatic.db import Database
from mnemomatic.models import Document
from mnemomatic import runtime


class TestListItemsTool(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        for i in range(5):
            self.db.store_document(
                Document(namespace="proj", title=f"doc {i}", content="body"), None
            )
        self._patch = patch.object(runtime, "_db", return_value=self.db)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.db.close()

    def test_response_shape_and_paging(self):
        resp = server.list_items(item_type="document", namespace="proj", limit=2, offset=2)
        self.assertEqual(resp["total"], 5)
        self.assertEqual(resp["limit"], 2)
        self.assertEqual(resp["offset"], 2)
        self.assertEqual(resp["namespace"], "proj")
        self.assertEqual(resp["item_type"], "document")
        self.assertEqual([i["title"] for i in resp["items"]], ["doc 2", "doc 1"])

    def test_defaults_list_first_page_newest_first(self):
        resp = server.list_items(item_type="document", namespace="proj")
        self.assertEqual(len(resp["items"]), 5)
        self.assertEqual(resp["items"][0]["title"], "doc 4")

    def test_limit_clamped_to_maximum(self):
        resp = server.list_items(item_type="document", namespace="proj", limit=10_000)
        self.assertEqual(resp["limit"], config.MAX_LIST_LIMIT)

    def test_nonpositive_limit_and_negative_offset_clamped(self):
        resp = server.list_items(item_type="document", namespace="proj", limit=0, offset=-3)
        self.assertEqual(resp["limit"], 1)
        self.assertEqual(resp["offset"], 0)
        self.assertEqual(len(resp["items"]), 1)

    def test_invalid_item_type_returns_error(self):
        resp = server.list_items(item_type="widget", namespace="proj")
        self.assertIn("error", resp)
        self.assertIn("widget", resp["error"])

    def test_empty_namespace_returns_zero_total(self):
        resp = server.list_items(item_type="document", namespace="ghost")
        self.assertEqual(resp["items"], [])
        self.assertEqual(resp["total"], 0)


if __name__ == "__main__":
    unittest.main()
