"""Tests for OnnxEmbedder output handling.

The bundled EmbeddingGemma export declares a `sentence_embedding` output with
pooling, dense projection layers, and normalization baked into the graph — the
embedder must use it as-is. Plain transformer exports (a custom
MNEMOMATIC_MODEL_PATH) only produce token embeddings, which are pooled in
Python — mean by default, or CLS for models trained that way (arctic-embed) —
and normalized.

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


def _make_embedder(sentence_output, run_result, pooling="mean"):
    """Build an OnnxEmbedder without loading a real model or tokenizer."""
    emb = OnnxEmbedder.__new__(OnnxEmbedder)
    emb._pooling = pooling
    emb._np = np
    emb._input_names = {"input_ids", "attention_mask"}
    emb._sentence_output = sentence_output

    # Token-embedding results are (1, seq, dim); keep the fake encoding the
    # same length as the tensor so the mask lines up. Tests that care about
    # padding override attention_mask afterwards.
    seq = run_result[0].shape[1] if run_result[0].ndim == 3 else 4
    encoded = MagicMock()
    encoded.ids = list(range(seq))
    encoded.attention_mask = [1] * seq
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

    def test_padding_is_excluded_from_the_mean(self):
        # Masked positions must not drag the mean toward zero.
        tokens = np.array([[[2.0, 0.0], [2.0, 0.0], [99.0, 99.0], [99.0, 99.0]]])
        emb = _make_embedder(None, [tokens])
        emb.tokenizer.encode.return_value.attention_mask = [1, 1, 0, 0]
        self.assertEqual(emb._embed("hello"), [1.0, 0.0])


@unittest.skipUnless(np is not None, "requires numpy (onnx extra)")
class TestClsPooling(unittest.TestCase):
    """Models trained on their CLS token (arctic-embed) must not be mean-pooled.

    This is the quiet failure: mean-pooling a CLS-trained model returns a
    perfectly well-formed unit vector that simply retrieves badly, so nothing
    errors and only search quality suffers.
    """

    def test_cls_pooling_takes_the_first_token(self):
        tokens = np.array([[[3.0, 4.0], [100.0, 0.0], [0.0, 100.0]]])
        emb = _make_embedder(None, [tokens], pooling="cls")
        # First token only, normalized — the rest are ignored entirely.
        for got, want in zip(emb._embed("hello"), [0.6, 0.8]):
            self.assertAlmostEqual(got, want, places=6)

    def test_cls_and_mean_disagree_on_the_same_tokens(self):
        # Guards against the setting being silently ignored: if these two ever
        # matched, POOLING would not be doing anything.
        tokens = np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
        cls = _make_embedder(None, [tokens], pooling="cls")._embed("x")
        mean = _make_embedder(None, [tokens], pooling="mean")._embed("x")
        self.assertNotEqual([round(v, 6) for v in cls], [round(v, 6) for v in mean])

    def test_unknown_pooling_falls_back_to_mean(self):
        tokens = np.array([[[2.0, 0.0], [0.0, 2.0]]])
        emb = _make_embedder(None, [tokens], pooling="something-else")
        expected = 1.0 / np.sqrt(2.0)
        for got in emb._embed("hello"):
            self.assertAlmostEqual(got, expected, places=6)

    def test_graph_pooling_still_wins_over_the_setting(self):
        # A sentence_embedding output means pooling is already in the graph;
        # POOLING must not cause it to be re-pooled.
        emb = _make_embedder("sentence_embedding", [np.array([[3.0, 4.0]])], pooling="cls")
        for got, want in zip(emb._embed("hello"), [0.6, 0.8]):
            self.assertAlmostEqual(got, want, places=6)


if __name__ == "__main__":
    unittest.main()
