"""Tests for the bundled-model metadata loader (model_config.py).

The Docker build writes /app/model/model_config.json describing the model the
image was built with; server defaults (dimension, prefixes, token limit) are
driven by it. The loader must return {} for anything unreadable — a broken
config file must degrade to the MiniLM-compatible fallbacks, never crash the
server at import time.
"""

import json
import tempfile
import unittest
from unittest.mock import patch

from mnemomatic import model_config


class TestLoad(unittest.TestCase):
    def _load_from(self, content: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(content)
        with patch.object(model_config, "MODEL_CONFIG_PATH", f.name):
            return model_config.load()

    def test_valid_config(self):
        config = self._load_from(json.dumps({
            "model": "embeddinggemma-300m", "dim": 768, "max_tokens": 2048,
            "query_prefix": "task: search result | query: ",
            "doc_prefix": "title: none | text: ",
        }))
        self.assertEqual(config["dim"], 768)
        self.assertEqual(config["query_prefix"], "task: search result | query: ")

    def test_missing_file_returns_empty(self):
        with patch.object(model_config, "MODEL_CONFIG_PATH", "/nonexistent/model_config.json"):
            self.assertEqual(model_config.load(), {})

    def test_malformed_json_returns_empty(self):
        self.assertEqual(self._load_from("{not json"), {})

    def test_non_object_json_returns_empty(self):
        self.assertEqual(self._load_from('["a", "list"]'), {})

    def test_test_environment_has_no_config(self):
        # The fallbacks the rest of the suite relies on: without a config
        # file, dim defaults to 384 and prefixes to "".
        self.assertEqual(model_config.CONFIG, {})


if __name__ == "__main__":
    unittest.main()
