"""Tests for Phase A of the memory-foundations work: usage tracking and revisions.

Covers the v1→v2 schema migration, record_access semantics (what counts and
what doesn't), revision capture across every mutation path (update, delete,
upsert-overwrite, tag edits, namespace delete/rename-replace), per-item
pruning, and the list_revisions/restore tools end to end.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import mnemomatic.db as db_module
from mnemomatic.db import Database
from mnemomatic.models import Document, Knowledge, Note
from mnemomatic import runtime
from mnemomatic import tools_history
from mnemomatic import tools_search


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def _note(self, title="n", content="body", namespace="proj") -> Note:
        stored, _ = self.db.store_note(
            Note(namespace=namespace, title=title, content=content), embedding=None)
        return stored


class TestMigration(unittest.TestCase):
    def test_v1_database_gains_columns_and_revisions_table(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            # Build a current database, then strip it back to version-1 shape.
            db = Database(tmp.name)
            db.store_note(Note(namespace="p", title="t", content="c"), embedding=None)
            conn = db._get_conn()
            conn.execute("DROP TABLE revisions")
            for table in ("documents", "knowledge", "notes"):
                conn.execute(f"ALTER TABLE {table} DROP COLUMN retrieval_count")
                conn.execute(f"ALTER TABLE {table} DROP COLUMN last_accessed")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
            db.close()

            migrated = Database(tmp.name)
            conn = migrated._get_conn()
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"],
                             db_module.SCHEMA_VERSION)
            row = conn.execute("SELECT retrieval_count, last_accessed FROM notes").fetchone()
            self.assertEqual(row["retrieval_count"], 0)
            self.assertIsNone(row["last_accessed"])
            conn.execute("SELECT * FROM revisions")  # table exists
            migrated.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class TestRecordAccess(DbTestCase):
    def test_bumps_count_and_timestamp_not_updated_at(self):
        note = self._note()
        before = self.db.get_note(note.id)
        self.db.record_access([("note", note.id)])
        self.db.record_access([("note", note.id)])
        after = self.db.get_note(note.id)
        self.assertEqual(after.retrieval_count, 2)
        self.assertIsNotNone(after.last_accessed)
        self.assertEqual(after.updated_at, before.updated_at)

    def test_mixed_types_and_unknown_ids_are_safe(self):
        note = self._note()
        doc, _ = self.db.store_document(
            Document(namespace="proj", title="d", content="x"), embedding=None)
        self.db.record_access([("note", note.id), ("document", doc.id),
                               ("document", "no-such-id"), ("bogus-type", "x")])
        self.assertEqual(self.db.get_note(note.id).retrieval_count, 1)
        self.assertEqual(self.db.get_document(doc.id).retrieval_count, 1)

    def test_stores_and_updates_leave_counters_alone(self):
        note = self._note()
        self.db.record_access([("note", note.id)])
        # Upsert-overwrite and field update must both preserve the counters.
        self.db.store_note(Note(namespace="proj", title="n", content="v2"), embedding=None)
        self.db.update_note(note.id, content="v3")
        self.assertEqual(self.db.get_note(note.id).retrieval_count, 1)


class TestRevisionCapture(DbTestCase):
    def _revisions(self, item_id):
        return self.db.list_revisions(item_id=item_id)

    def test_update_captures_prior_state(self):
        note = self._note(content="v1")
        self.db.update_note(note.id, content="v2")
        revs = self._revisions(note.id)
        self.assertEqual(len(revs), 1)
        self.assertEqual(revs[0]["op"], "update")
        self.assertEqual(self.db.get_revision(revs[0]["id"])["item"].content, "v1")

    def test_upsert_overwrite_captures_prior_state(self):
        note = self._note(content="v1")
        self.db.store_note(Note(namespace="proj", title="n", content="v2"), embedding=None)
        revs = self._revisions(note.id)
        self.assertEqual([r["op"] for r in revs], ["update"])
        self.assertEqual(self.db.get_revision(revs[0]["id"])["item"].content, "v1")

    def test_delete_captures_final_state(self):
        note = self._note(content="last words")
        self.db.delete_note(note.id)
        revs = self._revisions(note.id)
        self.assertEqual([r["op"] for r in revs], ["delete"])
        self.assertEqual(self.db.get_revision(revs[0]["id"])["item"].content, "last words")

    def test_tag_edit_captures_prior_state(self):
        note = self._note()
        self.db.update_tags(note.id, "note", add_tags=["a"])
        revs = self._revisions(note.id)
        self.assertEqual(len(revs), 1)
        self.assertEqual(self.db.get_revision(revs[0]["id"])["item"].tags, [])

    def test_delete_namespace_captures_every_item(self):
        note = self._note()
        k, _, _ = self.db.store_knowledge(
            Knowledge(namespace="proj", subject="s", fact="f"), embedding=None)
        self.db.delete_namespace("proj")
        revs = self.db.list_revisions(namespace="proj")
        self.assertEqual({r["item_id"] for r in revs}, {note.id, k.id})
        self.assertEqual({r["op"] for r in revs}, {"delete"})

    def test_rename_merge_captures_replaced_items(self):
        loser = self._note(namespace="target", title="clash", content="replaced")
        self._note(namespace="source", title="clash", content="winner")
        self.db.rename_namespace("source", "target")
        revs = self._revisions(loser.id)
        self.assertEqual([r["op"] for r in revs], ["delete"])
        self.assertEqual(self.db.get_revision(revs[0]["id"])["item"].content, "replaced")

    def test_prunes_to_keep_limit_per_item(self):
        note = self._note(content="v0")
        with patch.object(db_module, "REVISIONS_KEEP", 3):
            for i in range(1, 6):
                self.db.update_note(note.id, content=f"v{i}")
        revs = self._revisions(note.id)
        self.assertEqual(len(revs), 3)
        # Newest three priors survive: v4, v3, v2.
        contents = [self.db.get_revision(r["id"])["item"].content for r in revs]
        self.assertEqual(contents, ["v4", "v3", "v2"])

    def test_keep_zero_disables_capture(self):
        note = self._note()
        with patch.object(db_module, "REVISIONS_KEEP", 0):
            self.db.update_note(note.id, content="v2")
            self.db.delete_note(note.id)
        self.assertEqual(self._revisions(note.id), [])

    def test_payload_excludes_usage_counters(self):
        note = self._note()
        self.db.record_access([("note", note.id)])
        self.db.update_note(note.id, content="v2")
        item = self.db.get_revision(self._revisions(note.id)[0]["id"])["item"]
        self.assertEqual(item.retrieval_count, 0)  # model default, not the live counter


class TestListRevisionsFilters(DbTestCase):
    def test_filters_and_order(self):
        note = self._note(namespace="a")
        doc, _ = self.db.store_document(
            Document(namespace="b", title="d", content="x"), embedding=None)
        self.db.update_note(note.id, content="v2")
        self.db.update_document(doc.id, content="y")

        self.assertEqual([r["item_id"] for r in self.db.list_revisions(item_type="document")],
                         [doc.id])
        self.assertEqual([r["item_id"] for r in self.db.list_revisions(namespace="a")],
                         [note.id])
        both = self.db.list_revisions()
        self.assertEqual([r["item_id"] for r in both], [doc.id, note.id])  # newest first
        with self.assertRaises(ValueError):
            self.db.list_revisions(item_type="bogus")


class ToolTestCase(DbTestCase):
    """Server tools against a real in-memory Database, FTS-only (no embedder)."""

    def setUp(self):
        super().setUp()
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_embedder", return_value=None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()


class TestAccessRecordingTools(ToolTestCase):
    def test_read_tool_records(self):
        note = self._note()
        tools_search.read("note", note.id)
        self.assertEqual(self.db.get_note(note.id).retrieval_count, 1)

    def test_search_records_surfaced_items(self):
        note = self._note(title="alpha topic", content="about alpha")
        self._note(title="unrelated", content="beta")
        tools_search.search("alpha", mode="fulltext")
        self.assertEqual(self.db.get_note(note.id).retrieval_count, 1)

    def test_list_items_does_not_record(self):
        note = self._note()
        tools_search.list_items("note", "proj")
        self.assertEqual(self.db.get_note(note.id).retrieval_count, 0)


class TestRestoreTool(ToolTestCase):
    def test_restore_update_rolls_back_and_is_undoable(self):
        note = self._note(content="v1")
        self.db.update_note(note.id, content="v2")
        rev_id = self.db.list_revisions(item_id=note.id)[0]["id"]

        result = tools_history.restore(rev_id)
        self.assertEqual(result["restored_revision"], rev_id)
        self.assertFalse(result["recreated"])
        self.assertEqual(self.db.get_note(note.id).content, "v1")
        # The rollback captured v2, so the restore itself can be undone.
        newest = self.db.list_revisions(item_id=note.id)[0]
        self.assertEqual(self.db.get_revision(newest["id"])["item"].content, "v2")

    def test_restore_delete_recreates_with_original_id(self):
        note = self._note(content="precious")
        self.db.delete_note(note.id)
        rev_id = self.db.list_revisions(item_id=note.id)[0]["id"]

        result = tools_history.restore(rev_id)
        self.assertTrue(result["recreated"])
        restored = self.db.get_note(note.id)
        self.assertEqual(restored.content, "precious")
        self.assertEqual(restored.created_at, note.created_at)

    def test_restore_refuses_taken_key(self):
        note = self._note(title="the-title", content="old")
        self.db.delete_note(note.id)
        usurper = self._note(title="the-title", content="new item, new id")
        rev_id = self.db.list_revisions(item_id=note.id)[0]["id"]

        result = tools_history.restore(rev_id)
        self.assertIn("error", result)
        self.assertIn(usurper.id, result["details"])
        self.assertEqual(self.db.get_note(usurper.id).content, "new item, new id")

    def test_restore_unknown_revision(self):
        self.assertIn("error", tools_history.restore(99999))

    def test_restored_document_is_searchable(self):
        doc, _ = self.db.store_document(
            Document(namespace="proj", title="spec", content="the flux capacitor design"),
            embedding=None)
        self.db.delete_document(doc.id)
        rev_id = self.db.list_revisions(item_id=doc.id)[0]["id"]
        tools_history.restore(rev_id)
        results = tools_search.search("flux capacitor", mode="fulltext")
        self.assertIn(doc.id, [r["id"] for r in results if "id" in r])


class TestListRevisionsTool(ToolTestCase):
    def test_response_shape_and_validation(self):
        note = self._note()
        self.db.update_note(note.id, content="v2")
        resp = tools_history.list_revisions(item_id=note.id)
        self.assertEqual(len(resp["revisions"]), 1)
        rev = resp["revisions"][0]
        self.assertEqual(rev["item_id"], note.id)
        self.assertEqual(rev["op"], "update")
        self.assertNotIn("payload", rev)  # summaries stay small
        self.assertIn("error", tools_history.list_revisions(item_type="bogus"))


if __name__ == "__main__":
    unittest.main()
