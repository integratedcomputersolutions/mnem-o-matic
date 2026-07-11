"""Tests for asymmetric embedding task prefixes.

MNEMOMATIC_EMBED_QUERY_PREFIX / MNEMOMATIC_EMBED_DOC_PREFIX are prepended to
text at embedding time only: queries get the query prefix, stored content gets
the document prefix, and stored text/snippets never contain either. Defaults
come from the bundled model's model_config.json (asymmetric models like
EmbeddingGemma record their task prompts there); without a config file — as in
this test environment — and for external endpoints, defaults are empty.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mnemomatic.server as server

EMBEDDING = [0.1, 0.2]


class PrefixTestBase(unittest.TestCase):
    """Patches prefixes to distinctive markers and records what gets embedded."""

    def setUp(self):
        self.embedded: list[str] = []

        def record(text):
            self.embedded.append(text)
            return EMBEDDING

        self._patches = [
            patch.object(server, "EMBED_QUERY_PREFIX", "Q>> "),
            patch.object(server, "EMBED_DOC_PREFIX", "D>> "),
            patch.object(server, "_safe_embed", side_effect=record),
            patch.object(server, "_embedder", return_value=object()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()


class TestQueryPrefix(PrefixTestBase):
    def test_semantic_search_prefixes_query(self):
        fake_db = MagicMock()
        fake_db.search_vec.return_value = []
        with patch.object(server, "_db", return_value=fake_db):
            server.search(query="hello world", mode="semantic")
        self.assertEqual(self.embedded, ["Q>> hello world"])

    def test_hybrid_search_prefixes_query_but_not_fts(self):
        fake_db = MagicMock()
        fake_db.search_hybrid.return_value = []
        with patch.object(server, "_db", return_value=fake_db):
            server.search(query="hello world", mode="hybrid")
        self.assertEqual(self.embedded, ["Q>> hello world"])
        # FTS receives the raw (escaped) query, never the embedding prefix.
        fts_query = fake_db.search_hybrid.call_args[0][0]
        self.assertEqual(fts_query, "hello world")


class TestDocumentPrefix(PrefixTestBase):
    def _fake_db(self):
        fake_db = MagicMock()
        stored = MagicMock()
        stored.id, stored.namespace, stored.title, stored.subject = "id", "ns", "t", "s"
        fake_db.store_document.return_value = (stored, True)
        fake_db.store_knowledge.return_value = (stored, True)
        fake_db.store_note.return_value = (stored, True)
        return fake_db

    def test_store_knowledge_prefixes_content(self):
        with patch.object(server, "_db", return_value=self._fake_db()):
            server.store_knowledge(namespace="ns", subject="auth", fact="JWT")
        self.assertEqual(self.embedded, ["D>> auth: JWT"])

    def test_store_note_prefixes_content(self):
        with patch.object(server, "_db", return_value=self._fake_db()):
            server.store_note(namespace="ns", title="n", content="c")
        self.assertEqual(self.embedded, ["D>> n\nc"])

    def test_store_small_document_prefixes_content(self):
        with patch.object(server, "_db", return_value=self._fake_db()):
            server.store_document(namespace="ns", title="T", content="small body")
        self.assertEqual(self.embedded, ["D>> T\nsmall body"])

    def test_chunked_document_prefixes_embeds_but_stores_raw_chunks(self):
        batch_calls: list[list[str]] = []

        def fake_batch(texts):
            batch_calls.append(texts)
            return [EMBEDDING] * len(texts)

        content = ("chunk sentence. " * 40 + "\n\n") * 5  # over CHUNK_THRESHOLD
        with patch.object(server, "_safe_embed_batch", side_effect=fake_batch):
            embedding, chunks = server._embed_document_body("T", content)
        self.assertIsNone(embedding)
        self.assertTrue(all(t.startswith("D>> ") for t in batch_calls[0]))
        # Stored chunk text is the raw content — the prefix never persists.
        self.assertTrue(all(not c.startswith("D>> ") for c, _ in chunks))
        self.assertEqual(len(chunks), len(batch_calls[0]))


class TestPrefixDefaults(unittest.TestCase):
    def test_no_model_config_means_empty_prefixes(self):
        # No model_config.json in the test environment → symmetric defaults,
        # byte-identical embedding input.
        self.assertEqual(server.EMBED_QUERY_PREFIX, "")
        self.assertEqual(server.EMBED_DOC_PREFIX, "")
        recorded = []
        with patch.object(server, "_safe_embed", side_effect=lambda t: recorded.append(t) or EMBEDDING):
            server._embed_query("hello")
            server._embed_content("world")
        self.assertEqual(recorded, ["hello", "world"])


if __name__ == "__main__":
    unittest.main()
