import functools
import json
import logging
import os
import socket
import urllib.error
import urllib.request

logger = logging.getLogger("mnemomatic")

MODEL_PATH = os.environ.get("MNEMOMATIC_MODEL_PATH", "/app/model/model.onnx")
TOKENIZER_PATH = os.environ.get("MNEMOMATIC_TOKENIZER_PATH", "/app/model/tokenizer.json")
EMBED_TIMEOUT = int(os.environ.get("MNEMOMATIC_EMBED_TIMEOUT", "30"))


class OnnxEmbedder:
    """Local ONNX embedding model (requires onnxruntime, tokenizers, numpy)."""

    def __init__(self):
        # Lazy imports so this module can be imported without the ML stack installed
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np

        # Load ONNX model with detailed error messages
        try:
            self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Embedding model not found at {MODEL_PATH}. "
                f"Set MNEMOMATIC_MODEL_PATH or ensure the model file exists."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from {MODEL_PATH}: {type(e).__name__}: {e}"
            )

        # Load tokenizer with detailed error messages
        try:
            self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Tokenizer not found at {TOKENIZER_PATH}. "
                f"Set MNEMOMATIC_TOKENIZER_PATH or ensure the tokenizer file exists."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer from {TOKENIZER_PATH}: {type(e).__name__}: {e}"
            )

        self.tokenizer.enable_truncation(max_length=512)
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        self.embed = functools.lru_cache(maxsize=256)(self._embed)

    @property
    def mode(self) -> str:
        return "built-in ONNX"

    def _embed(self, text: str) -> list[float]:
        np = self._np
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        token_embeddings = self.session.run(None, feed)[0].astype(np.float32)  # (1, seq, dim)
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        mean_pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (mean_pooled / norm)[0].tolist()


class HttpEmbedder:
    """Ollama-compatible HTTP embedding endpoint.

    Expects a POST endpoint that accepts {"model": "...", "prompt": "..."}
    and returns {"embedding": [...]}.

    Compatible with Ollama's /api/embeddings endpoint.
    """

    def __init__(self, url: str, model: str = ""):
        if not url:
            raise ValueError("MNEMOMATIC_EMBED_URL must be set and non-empty")
        self.url = url
        self.model = model
        self.embed = functools.lru_cache(maxsize=256)(self._embed)

    @property
    def mode(self) -> str:
        return "external HTTP"

    def _embed(self, text: str) -> list[float]:
        """Fetch embedding from remote HTTP endpoint.

        Raises:
            RuntimeError: If the embedding service is unreachable, returns invalid data,
                         or responds with an error.
        """
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
                response_data = resp.read()
        except urllib.error.HTTPError as e:
            logger.error(
                "Embedding service HTTP error: status=%d, url=%s",
                e.code, self.url,
            )
            raise RuntimeError(
                f"Embedding service returned HTTP {e.code} at {self.url}"
            )
        except urllib.error.URLError as e:
            logger.error(
                "Embedding service unreachable: %s (url=%s)",
                e.reason, self.url,
            )
            raise RuntimeError(
                f"Cannot reach embedding service at {self.url}: {e.reason}"
            )
        except socket.timeout:
            logger.error("Embedding service timeout after %ds: %s", EMBED_TIMEOUT, self.url)
            raise RuntimeError(
                f"Embedding service at {self.url} did not respond within {EMBED_TIMEOUT}s"
            )
        except Exception as e:
            logger.error(
                "Unexpected error contacting embedding service: %s: %s",
                type(e).__name__, e,
            )
            raise RuntimeError(f"Failed to contact embedding service: {type(e).__name__}: {e}")

        # Parse response
        try:
            data = json.loads(response_data)
        except json.JSONDecodeError as e:
            logger.error(
                "Embedding service returned invalid JSON: %s (first 200 chars: %s)",
                e, response_data[:200],
            )
            raise RuntimeError(
                f"Embedding service at {self.url} returned invalid JSON: {e}"
            )

        # Extract embedding
        try:
            embedding = data["embedding"]
            if not isinstance(embedding, list):
                raise TypeError(f"embedding field is {type(embedding).__name__}, expected list")
            return embedding
        except KeyError:
            logger.error(
                "Embedding service response missing 'embedding' field. Got: %s",
                list(data.keys()),
            )
            raise RuntimeError(
                f"Embedding service response missing 'embedding' field. Got: {list(data.keys())}"
            )
        except (TypeError, ValueError) as e:
            logger.error("Embedding value is invalid: %s", e)
            raise RuntimeError(f"Embedding service returned invalid embedding: {e}")
