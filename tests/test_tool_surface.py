"""Tests for the delete tools and the MCP resources.

These paths were previously exercised only by tests/test_mcp_api.py, which is
CI-only (it needs a composed server), so a local run could not catch a break in
them. Covered here at unit level: the three delete_* tools, the per-namespace
list resources, and the by-id get resources.
"""

import json
import unittest
from unittest.mock import patch

import mnemomatic.server as server
from mnemomatic.db import Database
from mnemomatic.models import Document, Knowledge, Note
from mnemomatic import runtime
from tests._support import axis


class _ToolTest(unittest.TestCase):
    """A real in-memory database behind the tools, with embedding stubbed out
    so the tests stay independent of any embedder being available."""

    def setUp(self):
        self.db = Database(":memory:")
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_safe_embed", return_value=axis(0)),
            patch.object(runtime, "_safe_embed_batch", side_effect=lambda ts: [axis(0)] * len(ts)),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.db.close()

    def add_document(self, title="doc", namespace="ns") -> str:
        doc, _ = self.db.store_document(
            Document(namespace=namespace, title=title, content="body"), axis(0))
        return doc.id

    def add_knowledge(self, subject="subj", namespace="ns") -> str:
        entry, _, _ = self.db.store_knowledge(
            Knowledge(namespace=namespace, subject=subject, fact="a fact"), axis(1))
        return entry.id

    def add_note(self, title="note", namespace="ns") -> str:
        note, _ = self.db.store_note(
            Note(namespace=namespace, title=title, content="jotting"), axis(2))
        return note.id


class TestDeleteTools(_ToolTest):
    def test_delete_document_removes_it(self):
        doc_id = self.add_document()
        self.assertEqual(server.delete_document(doc_id), {"id": doc_id, "deleted": True})
        self.assertIsNone(self.db.get_document(doc_id))

    def test_delete_knowledge_removes_it(self):
        k_id = self.add_knowledge()
        self.assertEqual(server.delete_knowledge(k_id), {"id": k_id, "deleted": True})
        self.assertIsNone(self.db.get_knowledge(k_id))

    def test_delete_note_removes_it(self):
        note_id = self.add_note()
        self.assertEqual(server.delete_note(note_id), {"id": note_id, "deleted": True})
        self.assertIsNone(self.db.get_note(note_id))

    def test_deleting_a_missing_item_reports_false(self):
        for tool in (server.delete_document, server.delete_knowledge, server.delete_note):
            with self.subTest(tool=tool.__name__):
                self.assertEqual(tool("no-such-id"), {"id": "no-such-id", "deleted": False})

    def test_delete_writes_an_audit_event_naming_the_item(self):
        doc_id = self.add_document(title="minutes", namespace="meetings")
        server.delete_document(doc_id)
        event = self.db.list_audit(op="delete")[0]
        self.assertEqual(event["item_type"], "document")
        self.assertEqual(event["item_id"], doc_id)
        # Captured before the row went away, so the trail stays readable.
        self.assertEqual(event["namespace"], "meetings")
        self.assertEqual(event["title"], "minutes")

    def test_knowledge_audit_records_the_subject_as_title(self):
        k_id = self.add_knowledge(subject="deploy target")
        server.delete_knowledge(k_id)
        self.assertEqual(self.db.list_audit(op="delete")[0]["title"], "deploy target")

    def test_failed_delete_writes_no_audit_event(self):
        server.delete_document("no-such-id")
        self.assertEqual(self.db.list_audit(op="delete"), [])

    def test_deleted_item_is_recoverable_from_its_revision(self):
        # The docstrings promise delete is undoable; hold them to it.
        doc_id = self.add_document(title="keeper")
        server.delete_document(doc_id)
        revision = self.db.list_revisions(item_id=doc_id)[0]
        self.assertEqual(revision["op"], "delete")


class TestListResources(_ToolTest):
    def test_list_documents_returns_summaries(self):
        self.add_document(title="alpha")
        entries = json.loads(server.list_documents("ns"))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "alpha")
        self.assertEqual(set(entries[0]), {"id", "title", "mime_type", "tags", "updated_at"})

    def test_list_knowledge_returns_summaries(self):
        self.add_knowledge(subject="beta")
        entries = json.loads(server.list_knowledge("ns"))
        self.assertEqual(entries[0]["subject"], "beta")
        self.assertEqual(set(entries[0]),
                         {"id", "subject", "fact", "confidence", "tags", "updated_at"})

    def test_list_notes_returns_summaries(self):
        self.add_note(title="gamma")
        entries = json.loads(server.list_notes("ns"))
        self.assertEqual(entries[0]["title"], "gamma")
        self.assertEqual(set(entries[0]), {"id", "title", "source", "tags", "updated_at"})

    def test_list_resources_scope_to_their_namespace(self):
        self.add_document(title="here", namespace="ns")
        self.add_document(title="elsewhere", namespace="other")
        titles = [d["title"] for d in json.loads(server.list_documents("ns"))]
        self.assertEqual(titles, ["here"])

    def test_empty_namespace_returns_an_empty_list(self):
        for resource in (server.list_documents, server.list_knowledge, server.list_notes):
            with self.subTest(resource=resource.__name__):
                self.assertEqual(json.loads(resource("nothing-here")), [])

    def test_list_namespaces_resource(self):
        self.add_document(namespace="one")
        self.add_note(namespace="two")
        self.assertEqual(sorted(json.loads(server.list_namespaces())), ["one", "two"])


class TestGetResources(_ToolTest):
    def test_get_document_returns_the_full_item(self):
        doc_id = self.add_document(title="readme")
        payload = json.loads(server.get_document(doc_id))
        self.assertEqual(payload["title"], "readme")
        self.assertEqual(payload["content"], "body")

    def test_get_note_returns_the_full_item(self):
        note_id = self.add_note(title="thought")
        self.assertEqual(json.loads(server.get_note(note_id))["title"], "thought")

    def test_get_knowledge_entry_returns_the_full_item(self):
        k_id = self.add_knowledge(subject="choice")
        self.assertEqual(json.loads(server.get_knowledge_entry(k_id))["subject"], "choice")

    def test_missing_id_returns_an_error_not_an_exception(self):
        cases = [(server.get_document, "Document"), (server.get_note, "Note"),
                 (server.get_knowledge_entry, "Knowledge")]
        for resource, label in cases:
            with self.subTest(resource=resource.__name__):
                self.assertEqual(json.loads(resource("no-such-id")),
                                 {"error": f"{label} no-such-id not found"})

    def test_reading_a_resource_counts_as_retrieval(self):
        doc_id = self.add_document()
        self.assertEqual(self.db.get_document(doc_id).retrieval_count, 0)
        server.get_document(doc_id)
        self.assertEqual(self.db.get_document(doc_id).retrieval_count, 1)

    def test_a_missing_item_does_not_count_as_retrieval(self):
        server.get_document("no-such-id")
        self.assertEqual(self.db.list_audit(), [])


if __name__ == "__main__":
    unittest.main()
