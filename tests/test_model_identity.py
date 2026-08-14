"""Tests for the embedding-identity fingerprint recorded in schema_meta.

The dimension check catches a model swap only when the dimension changes. Most
built-in models share 768 dimensions, so a swap between them passes that check
and leaves the index quietly wrong — queries embedded by one model searched
against another model's vectors. These tests cover the identity check that
closes that gap: what gets recorded, when a mismatch refuses startup, and the
two cases that must never fail (a database predating the check, and a run with
no embedder configured).
"""

import tempfile
import unittest
from pathlib import Path

from mnemomatic.db import Database

GEMMA = {
    "embed_model": "embeddinggemma-300m",
    "embed_query_prefix": "task: search result | query: ",
    "embed_doc_prefix": "title: none | text: ",
}
AMARETTO = {**GEMMA, "embed_model": "amaretto-embed-148m"}


class _TempDatabaseTest(unittest.TestCase):
    """A file-backed database, so it can be closed and reopened with a
    different identity the way a restart would."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.dir.name) / "test.db")

    def tearDown(self):
        self.dir.cleanup()

    def open(self, identity=None, allow_reindex=False) -> Database:
        return Database(self.path, allow_reindex=allow_reindex, embed_identity=identity)


class TestRecording(_TempDatabaseTest):
    def test_fresh_database_records_identity(self):
        db = self.open(GEMMA)
        self.assertEqual(db.stored_embed_identity(), GEMMA)
        db.close()

    def test_reopening_with_same_identity_is_clean(self):
        self.open(GEMMA).close()
        db = self.open(GEMMA)
        self.assertFalse(db.reindex_pending)
        self.assertEqual(db.stored_embed_identity(), GEMMA)
        db.close()

    def test_no_identity_records_nothing(self):
        db = self.open()
        self.assertEqual(db.stored_embed_identity(), {})
        db.close()


class TestMismatch(_TempDatabaseTest):
    def test_model_change_refuses_startup(self):
        self.open(GEMMA).close()
        with self.assertRaises(RuntimeError) as ctx:
            self.open(AMARETTO)
        message = str(ctx.exception)
        self.assertIn("embeddinggemma-300m", message)
        self.assertIn("amaretto-embed-148m", message)
        self.assertIn("MNEMOMATIC_REINDEX=1", message)

    def test_query_prefix_change_refuses_startup(self):
        self.open(GEMMA).close()
        with self.assertRaises(RuntimeError) as ctx:
            self.open({**GEMMA, "embed_query_prefix": "query: "})
        self.assertIn("query_prefix", str(ctx.exception))

    def test_doc_prefix_change_refuses_startup(self):
        self.open(GEMMA).close()
        with self.assertRaises(RuntimeError) as ctx:
            self.open({**GEMMA, "embed_doc_prefix": ""})
        self.assertIn("doc_prefix", str(ctx.exception))

    def test_message_names_every_changed_field(self):
        self.open(GEMMA).close()
        with self.assertRaises(RuntimeError) as ctx:
            self.open({"embed_model": "gte-multilingual-base",
                       "embed_query_prefix": "", "embed_doc_prefix": ""})
        message = str(ctx.exception)
        for field in ("model", "query_prefix", "doc_prefix"):
            self.assertIn(field, message)

    def test_same_dimension_swap_is_caught(self):
        # The case the dimension check cannot see: both models are 768-dim, so
        # nothing else would notice the index no longer matches the embedder.
        self.open(GEMMA).close()
        with self.assertRaises(RuntimeError):
            self.open(AMARETTO)


class TestReindexFlow(_TempDatabaseTest):
    def test_mismatch_under_allow_reindex_defers_instead_of_raising(self):
        self.open(GEMMA).close()
        db = self.open(AMARETTO, allow_reindex=True)
        self.assertTrue(db.reindex_pending)
        # Still the old identity until the rebuild actually happens.
        self.assertEqual(db.stored_embed_identity()["embed_model"], "embeddinggemma-300m")
        db.close()

    def test_rebuild_records_the_new_identity_and_clears_pending(self):
        self.open(GEMMA).close()
        db = self.open(AMARETTO, allow_reindex=True)
        db.rebuild_vec_tables()
        self.assertFalse(db.reindex_pending)
        self.assertEqual(db.stored_embed_identity(), AMARETTO)
        db.close()

    def test_reopening_after_rebuild_is_clean(self):
        self.open(GEMMA).close()
        db = self.open(AMARETTO, allow_reindex=True)
        db.rebuild_vec_tables()
        db.close()
        db = self.open(AMARETTO)
        self.assertFalse(db.reindex_pending)
        db.close()


class TestLegacyDatabase(_TempDatabaseTest):
    """A database written before identities were recorded has nothing to
    compare against, so it must adopt rather than refuse to start."""

    def test_adopts_current_identity(self):
        self.open().close()  # no identity recorded
        with self.assertLogs("mnemomatic", level="WARNING") as logs:
            db = self.open(GEMMA)
        self.assertEqual(db.stored_embed_identity(), GEMMA)
        self.assertIn("MNEMOMATIC_REINDEX=1", "".join(logs.output))
        db.close()

    def test_adoption_does_not_set_pending(self):
        self.open().close()
        db = self.open(GEMMA)
        self.assertFalse(db.reindex_pending)
        db.close()

    def test_enforces_on_the_next_open(self):
        self.open().close()
        self.open(GEMMA).close()  # adopts
        with self.assertRaises(RuntimeError):
            self.open(AMARETTO)


class TestUnknownIdentity(_TempDatabaseTest):
    """FTS-only runs, and external endpoints that never name their model, must
    neither trip the check nor damage a fingerprint that is already there."""

    def test_opens_against_a_fingerprinted_database(self):
        self.open(GEMMA).close()
        db = self.open()  # no embedder configured
        self.assertFalse(db.reindex_pending)
        db.close()

    def test_preserves_the_existing_fingerprint(self):
        self.open(GEMMA).close()
        self.open().close()
        db = self.open(GEMMA)
        self.assertEqual(db.stored_embed_identity(), GEMMA)
        db.close()

    def test_empty_model_name_counts_as_unknown(self):
        self.open(GEMMA).close()
        db = self.open({"embed_model": "", "embed_query_prefix": "x", "embed_doc_prefix": "y"})
        self.assertEqual(db.stored_embed_identity(), GEMMA)
        db.close()


if __name__ == "__main__":
    unittest.main()
