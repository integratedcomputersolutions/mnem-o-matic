"""Tests for embedder error handling and graceful degradation.

This tests CRITICAL #1: Error Handling Gaps
- HttpEmbedder network failures
- OnnxEmbedder initialization failures
- Graceful fallback to FTS search
- Proper error logging and messages
"""

import json
import logging
import time
import unittest
from unittest.mock import MagicMock, Mock, patch
import urllib.error
import socket

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnemomatic.embeddings import HttpEmbedder


class TestHttpEmbedderErrors(unittest.TestCase):
    """Test HttpEmbedder error handling."""

    def setUp(self):
        """Create embedder instance."""
        self.embedder = HttpEmbedder("http://localhost:11434/api/embeddings", model="test-model")

    def test_http_embedder_network_unreachable(self):
        """Network error (URLError) should be caught and re-raised as RuntimeError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("test text")

            self.assertIn("Cannot reach embedding service", str(cm.exception))
            self.assertIn("Connection refused", str(cm.exception))

    def test_http_embedder_timeout(self):
        """Socket timeout should be caught and re-raised as RuntimeError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = socket.timeout()

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("test text")

            self.assertIn("did not respond within", str(cm.exception))

    def test_http_embedder_http_error(self):
        """HTTP error (e.g. 500) should be caught and re-raised as RuntimeError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            http_error = urllib.error.HTTPError(
                url="http://localhost:11434/api/embeddings",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None,
            )
            mock_urlopen.side_effect = http_error

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("test text")

            self.assertIn("HTTP 500", str(cm.exception))

    def test_http_embedder_invalid_json(self):
        """Invalid JSON in response should be caught and re-raised as RuntimeError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"not valid json"
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("test text")

            self.assertIn("invalid JSON", str(cm.exception))

    def test_http_embedder_missing_embedding_field(self):
        """Missing 'embedding' field in response should raise RuntimeError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"result": "something"}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("test text")

            self.assertIn("missing 'embedding' field", str(cm.exception))

    def test_http_embedder_embedding_not_list(self):
        """Embedding field that is not a list should raise RuntimeError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": "not a list"}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("test text")

            self.assertIn("invalid embedding", str(cm.exception))

    def test_http_embedder_success(self):
        """Successful embedding request should return the (normalized) vector."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            embedding = [0.6, 0.8]  # already unit length → passes through unchanged
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": embedding}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            result = self.embedder.embed("test text")
            self.assertEqual(result, embedding)

    def test_http_embedder_normalizes_unnormalized_vectors(self):
        """External models may return non-unit vectors; scoring assumes unit."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": [3.0, 4.0]}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            result = self.embedder.embed("unnormalized")
            self.assertAlmostEqual(result[0], 0.6, places=9)
            self.assertAlmostEqual(result[1], 0.8, places=9)

    def test_http_embedder_zero_vector_rejected(self):
        """A zero vector is a broken embedding and must not be stored."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": [0.0, 0.0]}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("zero")
            self.assertIn("invalid embedding", str(cm.exception))

    def test_http_embedder_non_numeric_vector_rejected(self):
        """Non-numeric elements previously slipped through to the DB layer."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": ["a", "b"]}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as cm:
                self.embedder.embed("strings")
            self.assertIn("invalid embedding", str(cm.exception))

    def test_http_embedder_validation_url_required(self):
        """HttpEmbedder should require non-empty URL."""
        with self.assertRaises(ValueError) as cm:
            HttpEmbedder("", model="test")

        self.assertIn("MNEMOMATIC_EMBED_URL must be set", str(cm.exception))

    def test_http_embedder_caching(self):
        """Same text should be cached (second call shouldn't hit network)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            embedding = [0.6, 0.8]
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": embedding}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            embedder = HttpEmbedder("http://localhost:11434/api/embeddings", model="test")

            result1 = embedder.embed("test text")
            result2 = embedder.embed("test text")

            # Both should return the same result
            self.assertEqual(result1, embedding)
            self.assertEqual(result2, embedding)

            # But urlopen should only be called once (cached)
            self.assertEqual(mock_urlopen.call_count, 1)


class TestEmbedderFallback(unittest.TestCase):
    """Test server-level embedder fallback to FTS-only mode.

    This tests CRITICAL #5: Embedding Init Failures
    - Model file missing → FTS-only mode
    - OnnxEmbedder init fails → FTS-only mode
    - onnxruntime not installed → FTS-only mode
    - _safe_embed with None embedder → returns None
    """

    def test_missing_model_file_gives_none(self):
        """_embedder() returns None if model file is missing."""
        from mnemomatic import server

        # Reset the cached embedder
        server._embedder_initialized = False

        # Patch os.path.exists to simulate missing model file
        with patch("os.path.exists", return_value=False):
            result = server._embedder()

        self.assertIsNone(result)
        self.assertEqual(server._embedder_instance, None)

    def test_onnx_init_failure_gives_none(self):
        """_embedder() returns None if OnnxEmbedder.__init__ fails."""
        from mnemomatic import server

        # Reset the cached embedder
        server._embedder_initialized = False

        # Patch os.path.exists to simulate model file exists
        # But patch OnnxEmbedder (in embeddings module) to fail during init
        with patch("os.path.exists", return_value=True), \
             patch("mnemomatic.embeddings.OnnxEmbedder", side_effect=RuntimeError("Model load failed")):
            result = server._embedder()

        self.assertIsNone(result)
        self.assertEqual(server._embedder_instance, None)

    def test_onnx_import_error_gives_none(self):
        """_embedder() returns None if onnxruntime is not installed."""
        from mnemomatic import server

        # Reset the cached embedder
        server._embedder_initialized = False

        # Patch os.path.exists to simulate model file exists
        # But patch the OnnxEmbedder import to fail
        with patch("os.path.exists", return_value=True):
            with patch("mnemomatic.embeddings.OnnxEmbedder", side_effect=ImportError("No module named 'onnxruntime'")):
                result = server._embedder()

        self.assertIsNone(result)
        self.assertEqual(server._embedder_instance, None)

    def test_safe_embed_returns_none_when_embedder_is_none(self):
        """_safe_embed(text) returns None when embedder is unavailable."""
        from mnemomatic import server

        # Reset the cached embedder to None (simulating FTS-only mode)
        server._embedder_initialized = True
        server._embedder_instance = None

        # Call _safe_embed - it should return None since embedder is None
        result = server._safe_embed("test text")
        self.assertIsNone(result)


def _mock_embedding_response(req, timeout=None):
    """urlopen stand-in: returns an embedding derived from the request's prompt."""
    prompt = json.loads(req.data)["prompt"]
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(
        {"embedding": [float(len(prompt)), 0.5]}
    ).encode()
    return cm


class TestHttpEmbedderBatch(unittest.TestCase):
    """embed_batch runs requests concurrently while preserving order and
    per-item failure semantics."""

    def setUp(self):
        self.embedder = HttpEmbedder("http://localhost:11434/api/embeddings", model="m")

    def test_batch_preserves_order(self):
        texts = ["a", "bb", "ccc", "dddd"]
        with patch("urllib.request.urlopen", side_effect=_mock_embedding_response):
            results = self.embedder.embed_batch(texts)
        # Each embedding encodes its prompt's length as [len, 0.5]; the vectors
        # are normalized on return, but the component ratio (2·len) survives.
        self.assertEqual([round(r[0] / r[1]) for r in results], [2, 4, 6, 8])

    def test_batch_partial_failure_returns_none_for_failed_items(self):
        def flaky(req, timeout=None):
            if json.loads(req.data)["prompt"] == "bad":
                raise urllib.error.URLError("boom")
            return _mock_embedding_response(req)

        with patch("urllib.request.urlopen", side_effect=flaky):
            results = self.embedder.embed_batch(["ok", "bad", "fine"])
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])

    def test_batch_empty_input(self):
        self.assertEqual(self.embedder.embed_batch([]), [])

    def test_batch_uses_embed_cache(self):
        # Duplicate texts hit the lru_cache; the network sees each text once.
        with patch("urllib.request.urlopen", side_effect=_mock_embedding_response) as mock_urlopen:
            self.embedder.embed_batch(["same", "same", "same"])
        self.assertEqual(mock_urlopen.call_count, 1)

    def test_batch_requests_run_concurrently(self):
        # 8 requests at 100ms each: sequential would take ≥800ms; concurrent
        # execution must finish in roughly one round trip.
        def slow(req, timeout=None):
            time.sleep(0.1)
            return _mock_embedding_response(req)

        texts = [f"text {i}" for i in range(8)]
        with patch("urllib.request.urlopen", side_effect=slow):
            start = time.perf_counter()
            results = self.embedder.embed_batch(texts)
            elapsed = time.perf_counter() - start
        self.assertTrue(all(r is not None for r in results))
        self.assertLess(elapsed, 0.5, f"batch took {elapsed:.2f}s — requests ran sequentially?")


class TestSafeEmbedBatch(unittest.TestCase):
    """server._safe_embed_batch dispatch: batch when available, safe fallbacks."""

    def _with_embedder(self, embedder):
        from mnemomatic import server
        return patch.object(server, "_embedder", return_value=embedder)

    def test_no_embedder_returns_all_none(self):
        from mnemomatic import server
        with self._with_embedder(None):
            self.assertEqual(server._safe_embed_batch(["a", "b"]), [None, None])

    def test_embedder_with_batch_called_once(self):
        from mnemomatic import server
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[0.1], [0.2]]
        with self._with_embedder(embedder):
            results = server._safe_embed_batch(["a", "b"])
        embedder.embed_batch.assert_called_once_with(["a", "b"])
        self.assertEqual(results, [[0.1], [0.2]])

    def test_embedder_without_batch_falls_back_to_sequential(self):
        from mnemomatic import server
        embedder = Mock(spec=["embed"])  # no embed_batch attribute
        embedder.embed.side_effect = lambda t: [float(len(t))]
        with self._with_embedder(embedder):
            results = server._safe_embed_batch(["a", "bb"])
        self.assertEqual(results, [[1.0], [2.0]])

    def test_batch_exception_degrades_to_all_none(self):
        from mnemomatic import server
        embedder = MagicMock()
        embedder.embed_batch.side_effect = RuntimeError("embedder down")
        with self._with_embedder(embedder):
            self.assertEqual(server._safe_embed_batch(["a", "b"]), [None, None])

    def test_document_body_chunks_embed_as_one_batch(self):
        # A chunked document must produce exactly one embed_batch call, with
        # failed chunks dropped from the result.
        from mnemomatic import server
        embedder = MagicMock()
        embedder.embed_batch.side_effect = lambda texts: [
            [0.1] if i % 2 == 0 else None for i in range(len(texts))
        ]
        content = ("paragraph text. " * 40 + "\n\n") * 5  # well over CHUNK_THRESHOLD
        with self._with_embedder(embedder):
            embedding, chunks = server._embed_document_body("T", content)
        self.assertIsNone(embedding)
        embedder.embed_batch.assert_called_once()
        texts = embedder.embed_batch.call_args[0][0]
        self.assertGreater(len(texts), 1)
        self.assertEqual(len(chunks), (len(texts) + 1) // 2)  # odd indices dropped
        self.assertTrue(all(e == [0.1] for _, e in chunks))


if __name__ == "__main__":
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
