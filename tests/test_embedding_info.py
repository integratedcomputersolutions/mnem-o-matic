"""Tests for the embedding_info tool.

It answers three questions an agent cannot otherwise ask: is semantic search
available, which model is embedding queries, and does that model match the one
that built the stored vectors. The last is the interesting one — a mismatch
returns plausible but wrong results with no error, so the tool has to report
it plainly.
"""

import unittest
from unittest.mock import patch

from mnemomatic import config, runtime, tools_admin
from mnemomatic.db import Database
from tests._support import EMBEDDING_DIM, FakeEmbedder

GEMMA = {
    "embed_model": "embeddinggemma-300m",
    "embed_query_prefix": "task: search result | query: ",
    "embed_doc_prefix": "title: none | text: ",
}


class _InfoTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:", embed_identity=GEMMA)
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_embedder", return_value=FakeEmbedder()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.db.close()

    def info(self):
        return tools_admin.embedding_info()


class TestReporting(_InfoTest):
    def test_reports_semantic_search_available(self):
        self.assertTrue(self.info()["semantic_search"])

    def test_reports_fts_only_when_no_embedder(self):
        with patch.object(runtime, "_embedder", return_value=None):
            info = self.info()
        self.assertFalse(info["semantic_search"])
        self.assertIn("FTS-only", info["mode"])

    def test_reports_the_model_that_built_the_index(self):
        self.assertEqual(self.info()["index_model"], "embeddinggemma-300m")

    def test_reports_dimensions(self):
        info = self.info()
        self.assertEqual(info["dimensions"], EMBEDDING_DIM)
        self.assertEqual(info["index_dimensions"], EMBEDDING_DIM)

    def test_reports_the_task_prefixes(self):
        with patch.object(config, "EMBED_QUERY_PREFIX", "q: "), \
             patch.object(config, "EMBED_DOC_PREFIX", "d: "):
            info = self.info()
        self.assertEqual(info["query_prefix"], "q: ")
        self.assertEqual(info["doc_prefix"], "d: ")


class TestIndexAgreement(_InfoTest):
    def test_matches_when_configured_model_built_the_index(self):
        with patch.object(config, "embed_identity", return_value=GEMMA):
            info = self.info()
        self.assertEqual(info["model"], "embeddinggemma-300m")
        self.assertTrue(info["matches_index"])

    def test_reports_a_mismatch_rather_than_hiding_it(self):
        # The case the tool exists for: querying with one model against
        # another model's vectors. Same dimension, so nothing else notices.
        swapped = {**GEMMA, "embed_model": "amaretto-embed-148m"}
        with patch.object(config, "embed_identity", return_value=swapped):
            info = self.info()
        self.assertEqual(info["model"], "amaretto-embed-148m")
        self.assertEqual(info["index_model"], "embeddinggemma-300m")
        self.assertFalse(info["matches_index"])

    def test_unknown_when_the_index_predates_identity_recording(self):
        legacy = Database(":memory:")   # no identity recorded
        with patch.object(runtime, "_db", return_value=legacy):
            info = self.info()
        self.assertIsNone(info["index_model"])
        self.assertIsNone(info["matches_index"], "unknowable is not the same as mismatched")
        legacy.close()


class TestBackendDetail(_InfoTest):
    def test_built_in_model_reports_its_token_limit(self):
        self.assertIn("max_tokens", self.info())

    def test_external_endpoint_reports_url_and_wire_format(self):
        with patch.object(config, "EMBED_URL", "http://embed:8181/v1/embeddings"):
            info = self.info()
        self.assertEqual(info["endpoint"], "http://embed:8181/v1/embeddings")
        self.assertIn("wire_api", info)
        self.assertNotIn("max_tokens", info)

    def test_known_model_carries_a_card_link(self):
        with patch.object(config, "embed_identity", return_value=GEMMA):
            self.assertIn("huggingface.co", self.info()["model_url"])

    def test_unknown_model_omits_the_link(self):
        unknown = {**GEMMA, "embed_model": "some-private-model"}
        with patch.object(config, "embed_identity", return_value=unknown):
            self.assertNotIn("model_url", self.info())


if __name__ == "__main__":
    unittest.main()
