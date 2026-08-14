"""Tests for the MNEMOMATIC_REINDEX flow (rebuild vector index + re-embed).

Covers the db primitives (rebuild_vec_tables, set_embedding, deferred dim
change) and server._run_reindex end to end against a real database with a
fake embedder — including a dimension change.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mnemomatic.db
import mnemomatic.server as server
from mnemomatic.db import SCHEMA_VERSION, Database
from mnemomatic.models import Document, Knowledge, Note
from tests._support import EMBEDDING_DIM, FakeEmbedder, axis


class TestRebuildVecTables(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.doc, _ = self.db.store_document(
            Document(namespace="ns", title="T", content="c"), axis(0)
        )

    def tearDown(self):
        self.db.close()

    def test_rebuild_clears_vectors_but_keeps_content(self):
        self.assertEqual(len(self.db.search_vec(axis(0), table="documents")), 1)
        self.db.rebuild_vec_tables()
        self.assertEqual(self.db.search_vec(axis(0), table="documents"), [])
        self.assertEqual(self.db.get_document(self.doc.id).title, "T")

    def test_rebuild_records_dim_and_version(self):
        self.db.rebuild_vec_tables()
        conn = self.db._get_conn()
        meta = conn.execute("SELECT value FROM schema_meta WHERE key='embed_dim'").fetchone()
        self.assertEqual(int(meta["value"]), EMBEDDING_DIM)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"], SCHEMA_VERSION)


class TestSetEmbedding(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_set_embedding_makes_item_searchable(self):
        doc, _ = self.db.store_document(Document(namespace="ns", title="T", content="c"), None)
        self.assertEqual(self.db.search_vec(axis(3), table="documents"), [])
        self.assertTrue(self.db.set_embedding("document", doc.id, axis(3)))
        results = self.db.search_vec(axis(3), table="documents", namespace="ns")
        self.assertEqual([r.id for r in results], [doc.id])

    def test_set_embedding_does_not_touch_timestamps(self):
        doc, _ = self.db.store_document(Document(namespace="ns", title="T", content="c"), None)
        before = self.db.get_document(doc.id).updated_at
        self.db.set_embedding("document", doc.id, axis(1))
        self.assertEqual(self.db.get_document(doc.id).updated_at, before)

    def test_missing_item_returns_false(self):
        self.assertFalse(self.db.set_embedding("document", "no-such-id", axis(0)))

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            self.db.set_embedding("widget", "id", axis(0))


class TestDimChangeDeferral(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.path = self._tmp.name
        Database(self.path).close()  # created at the real EMBEDDING_DIM

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def test_mismatch_without_flag_still_raises(self):
        with patch.object(mnemomatic.db, "EMBEDDING_DIM", 8):
            with self.assertRaises(RuntimeError) as cm:
                Database(self.path)
        self.assertIn("MNEMOMATIC_REINDEX", str(cm.exception))

    def test_mismatch_with_flag_defers_to_reindex(self):
        with patch.object(mnemomatic.db, "EMBEDDING_DIM", 8):
            db = Database(self.path, allow_reindex=True)
            try:
                self.assertTrue(db.reindex_pending)
                db.rebuild_vec_tables()
                self.assertFalse(db.reindex_pending)
                conn = db._get_conn()
                meta = conn.execute("SELECT value FROM schema_meta WHERE key='embed_dim'").fetchone()
                self.assertEqual(int(meta["value"]), 8)
            finally:
                db.close()


class TestRunReindex(unittest.TestCase):
    """server._run_reindex against a real database and fake embedder."""

    def setUp(self):
        self.db = Database(":memory:")
        # Content stored with NO embeddings (as if from an FTS-only era).
        self.doc, _ = self.db.store_document(
            Document(namespace="ns", title="small", content="short body"), None)
        self.big, _ = self.db.store_document(
            Document(namespace="ns", title="big", content="paragraph. " * 300), None)
        self.k, _, _ = self.db.store_knowledge(
            Knowledge(namespace="other", subject="s", fact="f"), None)
        self.note, _ = self.db.store_note(
            Note(namespace="ns", title="n", content="c"), None)
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

    def _vec_count(self, table):
        return self.db._get_conn().execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    def test_reindex_embeds_everything(self):
        server._run_reindex()
        self.assertEqual(self._vec_count("vec_documents"), 1)   # small doc only
        self.assertGreater(self._vec_count("vec_document_chunks"), 1)  # big doc chunked
        self.assertEqual(self._vec_count("vec_knowledge"), 1)
        self.assertEqual(self._vec_count("vec_notes"), 1)
        # Everything is now semantically searchable.
        emb = self.embedder.embed("small\nshort body")
        results = self.db.search_vec(emb, table="documents", namespace="ns")
        self.assertIn(self.doc.id, [r.id for r in results])

    def test_reindex_applies_current_doc_prefix(self):
        with patch.object(server, "EMBED_DOC_PREFIX", "D>> "):
            server._run_reindex()
        # Every content embedding request carried the prefix.
        self.assertTrue(all(c.startswith("D>> ") for c in self.embedder.calls))

    def test_reindex_replaces_stale_vectors(self):
        # Pre-existing vector from an "old model" points at axis 0; after
        # reindex, searching with the old vector must not return the doc.
        self.db.set_embedding("document", self.doc.id, axis(0))
        server._run_reindex()
        new_emb = self.embedder.embed("small\nshort body")
        if new_emb[0] != 1.0:  # only meaningful when the fake axis differs
            results = self.db.search_vec(axis(0), table="documents", namespace="ns", limit=1)
            self.assertTrue(not results or results[0].score < 0.999)

    def test_reindex_with_dim_change_end_to_end(self):
        # Simulate MNEMOMATIC_EMBED_DIM=8 with a dim-8 embedder: after
        # reindex, dim-8 semantic search works against the rebuilt index.
        small_embedder = FakeEmbedder(dim=8)
        with patch.object(mnemomatic.db, "EMBEDDING_DIM", 8), \
             patch.object(server, "_embedder", return_value=small_embedder):
            self.db.reindex_pending = True
            server._run_reindex()
            emb = small_embedder.embed("s: f")
            results = self.db.search_vec(emb, table="knowledge", namespace="other")
            self.assertEqual([r.id for r in results], [self.k.id])

    def test_reindex_no_embedder_skips_without_dim_change(self):
        with patch.object(server, "_embedder", return_value=None):
            server._run_reindex()  # must not raise
        # Index untouched — nothing was rebuilt or embedded.
        self.assertEqual(self._vec_count("vec_documents"), 0)

    def test_reindex_no_embedder_with_dim_change_is_fatal(self):
        self.db.reindex_pending = True
        with patch.object(server, "_embedder", return_value=None):
            with self.assertRaises(RuntimeError):
                server._run_reindex()

    def test_reindex_counts_failures_and_continues(self):
        flaky = FakeEmbedder()
        original = flaky.embed

        def sometimes_fail(text):
            if "s: f" in text:
                raise RuntimeError("embedder hiccup")
            return original(text)

        flaky.embed = sometimes_fail
        with patch.object(server, "_embedder", return_value=flaky):
            server._run_reindex()  # must not raise
        self.assertEqual(self._vec_count("vec_knowledge"), 0)  # the failed one
        self.assertEqual(self._vec_count("vec_notes"), 1)      # others proceeded


if __name__ == "__main__":
    unittest.main()
