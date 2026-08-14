"""Tests for MNEMOMATIC_REINDEX=auto — reindex only when the embedder changed.

The database records which embedder built its index, so startup can tell a
deliberate model change from an unchanged one. "auto" acts on that: rebuild
when it differs, do nothing when it does not, and never destroy an index that
cannot be rebuilt.
"""

import unittest
from unittest.mock import patch

import mnemomatic.server as server
from mnemomatic.db import Database
from mnemomatic.models import Document
from tests._support import FakeEmbedder, axis

GEMMA = {
    "embed_model": "embeddinggemma-300m",
    "embed_query_prefix": "task: search result | query: ",
    "embed_doc_prefix": "title: none | text: ",
}
AMARETTO = {**GEMMA, "embed_model": "amaretto-embed-148m"}


class TestModeParsing(unittest.TestCase):
    def _mode(self, value):
        env = {} if value is None else {"MNEMOMATIC_REINDEX": value}
        with patch.dict(server.os.environ, env, clear=False):
            if value is None:
                server.os.environ.pop("MNEMOMATIC_REINDEX", None)
            return server._reindex_mode()

    def test_unset_is_off(self):
        self.assertEqual(self._mode(None), "off")

    def test_auto(self):
        for raw in ("auto", "AUTO", " Auto "):
            with self.subTest(raw=raw):
                self.assertEqual(self._mode(raw), "auto")

    def test_truthy_values_force(self):
        for raw in ("1", "true", "YES"):
            with self.subTest(raw=raw):
                self.assertEqual(self._mode(raw), "force")

    def test_falsey_values_are_off(self):
        for raw in ("", "0", "false", "no"):
            with self.subTest(raw=raw):
                self.assertEqual(self._mode(raw), "off")

    def test_unrecognised_value_warns_and_is_off(self):
        # A typo must not silently become "auto" — it falls back to the safe
        # default, where a changed embedder refuses startup.
        with self.assertLogs("mnemomatic", level="WARNING") as logs:
            self.assertEqual(self._mode("atuo"), "off")
        self.assertIn("atuo", "".join(logs.output))


class TestAutoTrigger(unittest.TestCase):
    """Does the identity comparison decide correctly whether work is needed?"""

    def setUp(self):
        self.tmp = __import__("tempfile").TemporaryDirectory()
        self.path = f"{self.tmp.name}/auto.db"
        db = Database(self.path, embed_identity=GEMMA)
        db.store_document(Document(namespace="ns", title="d", content="body"), axis(0))
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def open_as(self, identity, mode="auto"):
        return Database(self.path, allow_reindex=mode in ("auto", "force"),
                        embed_identity=identity)

    def test_unchanged_embedder_needs_no_reindex(self):
        db = self.open_as(GEMMA)
        self.assertFalse(db.reindex_pending)
        db.close()

    def test_changed_model_marks_reindex_pending(self):
        db = self.open_as(AMARETTO)
        self.assertTrue(db.reindex_pending)
        db.close()

    def test_changed_prefix_marks_reindex_pending(self):
        db = self.open_as({**GEMMA, "embed_query_prefix": "query: "})
        self.assertTrue(db.reindex_pending)
        db.close()

    def test_off_mode_still_refuses_a_changed_embedder(self):
        with self.assertRaises(RuntimeError):
            self.open_as(AMARETTO, mode="off")

    def test_pending_clears_and_identity_is_restamped_after_rebuild(self):
        db = self.open_as(AMARETTO)
        db.rebuild_vec_tables()
        self.assertFalse(db.reindex_pending)
        self.assertEqual(db.stored_embed_identity()["embed_model"], "amaretto-embed-148m")
        db.close()
        # And the next start is quiet.
        db = self.open_as(AMARETTO)
        self.assertFalse(db.reindex_pending)
        db.close()


class TestRunReindexUnderAuto(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.doc, _ = self.db.store_document(
            Document(namespace="ns", title="d", content="body"), None)
        self.embedder = FakeEmbedder()
        self._patches = [
            patch.object(server, "_db", return_value=self.db),
            patch.object(server, "_embedder", return_value=self.embedder),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.db.close()

    def _vec_count(self):
        return self.db._get_conn().execute(
            "SELECT COUNT(*) AS n FROM vec_documents").fetchone()["n"]

    def test_auto_reindex_embeds_content(self):
        with patch.object(server, "REINDEX_MODE", "auto"):
            server._run_reindex()
        self.assertEqual(self._vec_count(), 1)

    def test_auto_logs_that_the_embedder_changed(self):
        with patch.object(server, "REINDEX_MODE", "auto"):
            with self.assertLogs("mnemomatic", level="WARNING") as logs:
                server._run_reindex()
        message = "".join(logs.output)
        self.assertIn("auto", message)
        # The "remove the flag afterwards" advice belongs to =1 only.
        self.assertNotIn("Remove the flag", message)

    def test_force_still_tells_you_to_remove_the_flag(self):
        with patch.object(server, "REINDEX_MODE", "force"):
            with self.assertLogs("mnemomatic", level="WARNING") as logs:
                server._run_reindex()
        self.assertIn("Remove the flag", "".join(logs.output))

    def test_reindex_is_recorded_in_the_audit_trail(self):
        with patch.object(server, "REINDEX_MODE", "auto"):
            server._run_reindex()
        events = self.db.list_audit(op="reindex")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["detail"]["mode"], "auto")
        self.assertEqual(events[0]["detail"]["documents"], 1)


class TestNeverDestroyWhatCannotBeRebuilt(unittest.TestCase):
    """The rebuild empties the index before re-embedding, so a missing embedder
    must stop the run rather than leave the store with no vectors."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.store_document(Document(namespace="ns", title="d", content="body"), axis(0))

    def tearDown(self):
        self.db.close()

    def test_pending_reindex_without_an_embedder_raises(self):
        self.db.reindex_pending = True
        with patch.object(server, "_db", return_value=self.db), \
             patch.object(server, "_embedder", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                server._run_reindex()
        self.assertIn("no embedder is available", str(ctx.exception))

    def test_existing_vectors_survive_the_refusal(self):
        self.db.reindex_pending = True
        with patch.object(server, "_db", return_value=self.db), \
             patch.object(server, "_embedder", return_value=None):
            with self.assertRaises(RuntimeError):
                server._run_reindex()
        count = self.db._get_conn().execute(
            "SELECT COUNT(*) AS n FROM vec_documents").fetchone()["n"]
        self.assertEqual(count, 1)

    def test_no_pending_change_without_an_embedder_just_skips(self):
        with patch.object(server, "_db", return_value=self.db), \
             patch.object(server, "_embedder", return_value=None):
            server._run_reindex()  # logs and returns; nothing dropped
        count = self.db._get_conn().execute(
            "SELECT COUNT(*) AS n FROM vec_documents").fetchone()["n"]
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
