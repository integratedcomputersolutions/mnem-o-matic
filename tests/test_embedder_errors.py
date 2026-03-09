"""Tests for embedder error handling and graceful degradation.

This tests CRITICAL #1: Error Handling Gaps
- HttpEmbedder network failures
- OnnxEmbedder initialization failures
- Graceful fallback to FTS search
- Proper error logging and messages
"""

import json
import logging
import unittest
from unittest.mock import MagicMock, Mock, patch
import urllib.error
import socket

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnemomatic.embeddings import HttpEmbedder, OnnxEmbedder


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
        """Successful embedding request should return list of floats."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            embedding = [0.1, 0.2, 0.3, 0.4]
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": embedding}).encode()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            result = self.embedder.embed("test text")
            self.assertEqual(result, embedding)

    def test_http_embedder_validation_url_required(self):
        """HttpEmbedder should require non-empty URL."""
        with self.assertRaises(ValueError) as cm:
            HttpEmbedder("", model="test")

        self.assertIn("MNEMOMATIC_EMBED_URL must be set", str(cm.exception))

    def test_http_embedder_caching(self):
        """Same text should be cached (second call shouldn't hit network)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            embedding = [0.1, 0.2, 0.3, 0.4]
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


class TestOnnxEmbedderErrors(unittest.TestCase):
    """Test OnnxEmbedder error handling.

    Note: Full OnnxEmbedder tests require onnxruntime, tokenizers, and numpy
    to be installed. These tests verify the error handling logic structure.
    """

    def test_onnx_embedder_init_signature(self):
        """OnnxEmbedder.__init__ should accept no arguments."""
        from mnemomatic.embeddings import OnnxEmbedder
        import inspect

        sig = inspect.signature(OnnxEmbedder.__init__)
        # Should have only 'self' parameter
        params = [p for p in sig.parameters.keys() if p != 'self']
        self.assertEqual(len(params), 0)


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
        server._embedder_instance = server._UNSET

        # Patch os.path.exists to simulate missing model file
        with patch("os.path.exists", return_value=False):
            result = server._embedder()

        self.assertIsNone(result)
        self.assertEqual(server._embedder_instance, None)

    def test_onnx_init_failure_gives_none(self):
        """_embedder() returns None if OnnxEmbedder.__init__ fails."""
        from mnemomatic import server

        # Reset the cached embedder
        server._embedder_instance = server._UNSET

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
        server._embedder_instance = server._UNSET

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
        server._embedder_instance = None

        # Call _safe_embed - it should return None since embedder is None
        result = server._safe_embed("test text")
        self.assertIsNone(result)


if __name__ == "__main__":
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
