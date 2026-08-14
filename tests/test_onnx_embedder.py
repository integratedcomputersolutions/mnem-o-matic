"""Tests for OnnxEmbedder output handling.

The bundled EmbeddingGemma export declares a `sentence_embedding` output with
pooling, dense projection layers, and normalization baked into the graph — the
embedder must use it as-is. Plain transformer exports (a custom
MNEMOMATIC_MODEL_PATH) only produce token embeddings, which are mean-pooled
and normalized in Python.

These tests exercise `_embed` directly on a hand-built instance; they need
numpy (part of the `onnx` extra) and are skipped where it isn't installed.
"""

import unittest
from unittest.mock import MagicMock

try:
    import numpy as np
except ImportError:
    np = None

from mnemomatic.embeddings import OnnxEmbedder


def _make_embedder(sentence_output, run_result):
    """Build an OnnxEmbedder without loading a real model or tokenizer."""
    emb = OnnxEmbedder.__new__(OnnxEmbedder)
    emb._np = np
    emb._input_names = {"input_ids", "attention_mask"}
    emb._sentence_output = sentence_output

    encoded = MagicMock()
    encoded.ids = [2, 10, 11, 1]
    encoded.attention_mask = [1, 1, 1, 1]
    emb.tokenizer = MagicMock()
    emb.tokenizer.encode.return_value = encoded

    emb.session = MagicMock()
    emb.session.run.return_value = run_result
    return emb


@unittest.skipUnless(np is not None, "requires numpy (onnx extra)")
class TestSentenceEmbeddingOutput(unittest.TestCase):
    def test_uses_graph_output_and_normalizes(self):
        # A quantized graph can return a slightly off-unit vector; _embed must
        # request exactly the sentence_embedding output and re-normalize it.
        emb = _make_embedder("sentence_embedding", [np.array([[3.0, 4.0]])])
        result = emb._embed("hello")
        emb.session.run.assert_called_once()
        self.assertEqual(emb.session.run.call_args[0][0], ["sentence_embedding"])
        for got, want in zip(result, [0.6, 0.8]):
            self.assertAlmostEqual(got, want, places=6)

    def test_no_manual_pooling_of_graph_output(self):
        # The (1, dim) graph output must pass through untouched apart from
        # normalization — pooling it again would collapse the vector.
        vec = np.array([[1.0, 0.0, 0.0]])
        emb = _make_embedder("sentence_embedding", [vec])
        self.assertEqual(emb._embed("hello"), [1.0, 0.0, 0.0])


@unittest.skipUnless(np is not None, "requires numpy (onnx extra)")
class TestTokenOutputFallback(unittest.TestCase):
    def test_mean_pools_and_normalizes_token_embeddings(self):
        # Without a sentence_embedding output, token embeddings (1, seq, dim)
        # are mean-pooled over the attention mask and L2-normalized.
        tokens = np.array([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]])
        emb = _make_embedder(None, [tokens])
        result = emb._embed("hello")
        self.assertEqual(emb.session.run.call_args[0][0], None)
        expected = 1.0 / np.sqrt(2.0)
        for got in result:
            self.assertAlmostEqual(got, expected, places=6)


if __name__ == "__main__":
    unittest.main()
