import functools
import json
import logging
import math
import os
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from mnemomatic import model_config

logger = logging.getLogger("mnemomatic")

MODEL_PATH = os.environ.get("MNEMOMATIC_MODEL_PATH", "/app/model/model.onnx")
TOKENIZER_PATH = os.environ.get("MNEMOMATIC_TOKENIZER_PATH", "/app/model/tokenizer.json")
# Token truncation limit for the built-in model. Defaults to the bundled
# model's context from model_config.json (2048 for EmbeddingGemma, 512 for
# arctic-embed/gte), falling back to 512 when no config exists.
MODEL_MAX_TOKENS = int(os.environ.get(
    "MNEMOMATIC_MODEL_MAX_TOKENS", model_config.CONFIG.get("max_tokens", 512)
))
# How token embeddings become one vector, when the ONNX graph does not already
# do it. Must match what the model was trained with: mean-pooling a model
# trained on its CLS token produces vectors that are not wrong enough to error,
# only wrong enough to retrieve badly. Defaults to mean — what gte and
# EmbeddingGemma use, and what every pre-existing build assumed.
#
# Deliberately not overridable by environment: pooling changes the vectors just
# as the model does, but unlike the model name it is not part of the recorded
# embedding identity, so a mismatch would invalidate the index with nothing to
# catch it. Tying it to the bundled model's config keeps the two inseparable.
POOLING = str(model_config.CONFIG.get("pooling", "mean")).lower()
EMBED_TIMEOUT = int(os.environ.get("MNEMOMATIC_EMBED_TIMEOUT", "30"))
# Concurrent requests used by HttpEmbedder.embed_batch (chunked documents).
EMBED_CONCURRENCY = int(os.environ.get("MNEMOMATIC_EMBED_CONCURRENCY", "8"))
# Wire format of the embedding endpoint: "openai" (llama.cpp, vLLM, LM Studio,
# Ollama's /v1/embeddings, hosted APIs) or "ollama" (native /api/embeddings).
EMBED_API = os.environ.get("MNEMOMATIC_EMBED_API", "openai").strip().lower()


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale a vector to unit length.

    The score math in db.py (L2 distance → cosine similarity) is only correct
    for unit vectors. The built-in ONNX embedder normalizes as part of pooling;
    external endpoints return whatever their model produces, so their output is
    normalized here. Already-unit vectors pass through unchanged.

    Raises:
        ValueError: for a zero vector or non-numeric elements — a broken
                    embedding must not be stored.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ValueError("embedding has zero norm")
    if abs(norm - 1.0) < 1e-6:
        return vec
    return [x / norm for x in vec]


class OnnxEmbedder:
    """Local ONNX embedding model (requires onnxruntime, tokenizers, numpy).

    Supports two graph shapes:

    - Sentence-transformers exports (the EmbeddingGemma build option) declare
      a ``sentence_embedding`` output with pooling, projection layers, and
      normalization baked into the graph — it is used as-is.
    - Plain transformer exports (arctic-embed, and older MiniLM/e5 builds) only
      produce token embeddings; those are pooled here according to POOLING —
      mean by default, CLS for models trained that way — then normalized.
    """

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

        self.tokenizer.enable_truncation(max_length=MODEL_MAX_TOKENS)
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        output_names = [out.name for out in self.session.get_outputs()]
        # A sentence_embedding output means pooling + any projection layers are
        # part of the graph; pooling token embeddings ourselves would silently
        # skip those layers and produce garbage vectors.
        self._sentence_output = "sentence_embedding" if "sentence_embedding" in output_names else None
        self._pooling = POOLING
        self.embed = functools.lru_cache(maxsize=256)(self._embed)

    @property
    def mode(self) -> str:
        name = model_config.CONFIG.get("model")
        return f"built-in ONNX ({name})" if name else "built-in ONNX"

    # No embed_batch: benchmarked against real chunk workloads, a padded batch
    # inference is neutral-to-slower than sequential embed() on CPU — ORT
    # already parallelizes single runs across cores, and padding to the longest
    # chunk wastes compute. The sequential fallback also keeps the lru_cache.

    def _embed(self, text: str) -> list[float]:
        np = self._np
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        if self._sentence_output is not None:
            pooled = self.session.run([self._sentence_output], feed)[0].astype(np.float32)  # (1, dim)
        else:
            token_embeddings = self.session.run(None, feed)[0].astype(np.float32)  # (1, seq, dim)
            if self._pooling == "cls":
                # The first token carries the sentence representation for models
                # trained that way (BERT-style [CLS]); the rest are ignored.
                pooled = token_embeddings[:, 0, :]
            else:
                mask = attention_mask[..., np.newaxis].astype(np.float32)
                pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)
        # Normalize both paths: quantized sentence_embedding outputs can drift
        # slightly off unit length, and the score math needs unit vectors.
        norm = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norm)[0].tolist()


class HttpEmbedder:
    """HTTP embedding endpoint in one of two wire formats (MNEMOMATIC_EMBED_API):

    - "openai" (default): POST {"model", "input"} → {"data": [{"embedding": [...]}]}.
      Served by llama.cpp's llama-server, vLLM, LM Studio, Ollama's
      /v1/embeddings, and hosted APIs.
    - "ollama": POST {"model", "prompt"} → {"embedding": [...]}.
      Ollama's native /api/embeddings endpoint.
    """

    def __init__(self, url: str, model: str = "", api: str | None = None):
        if not url:
            raise ValueError("MNEMOMATIC_EMBED_URL must be set and non-empty")
        self.api = (api or EMBED_API)
        if self.api not in ("openai", "ollama"):
            raise ValueError(
                f"MNEMOMATIC_EMBED_API must be 'openai' or 'ollama', got {self.api!r}"
            )
        self.url = url
        self.model = model
        self.embed = functools.lru_cache(maxsize=256)(self._embed)

        # A URL that clearly belongs to the other flavor is almost certainly a
        # misconfiguration — say so up front instead of failing per request.
        if self.api == "openai" and "/api/embeddings" in url:
            logger.warning(
                "MNEMOMATIC_EMBED_API=openai but the URL looks like Ollama's native "
                "endpoint (%s). Set MNEMOMATIC_EMBED_API=ollama, or point the URL at "
                "the OpenAI-compatible /v1/embeddings.", url,
            )
        elif self.api == "ollama" and "/v1/embeddings" in url:
            logger.warning(
                "MNEMOMATIC_EMBED_API=ollama but the URL looks OpenAI-compatible (%s). "
                "Set MNEMOMATIC_EMBED_API=openai, or point the URL at /api/embeddings.", url,
            )

    @property
    def mode(self) -> str:
        return f"external HTTP ({self.api})"

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed many texts with concurrent requests.

        One text per request (uniform across both wire formats), with up to
        EMBED_CONCURRENCY requests in flight instead of one 30s-timeout round
        trip per chunk. Failed items come back as None (order preserved) so
        the caller can drop just those chunks, matching the per-chunk
        semantics of sequential embeds.
        """
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)

        def _one(i: int, text: str) -> None:
            try:
                results[i] = self.embed(text)
            except Exception as e:
                logger.error("Batch embedding failed for item %d/%d: %s", i + 1, len(texts), e)

        with ThreadPoolExecutor(max_workers=min(EMBED_CONCURRENCY, len(texts))) as pool:
            for i, text in enumerate(texts):
                pool.submit(_one, i, text)
        return results

    def _embed(self, text: str) -> list[float]:
        """Fetch embedding from remote HTTP endpoint.

        Raises:
            RuntimeError: If the embedding service is unreachable, returns invalid data,
                         or responds with an error.
        """
        if self.api == "openai":
            body = {"model": self.model, "input": text}
        else:
            body = {"model": self.model, "prompt": text}
        payload = json.dumps(body).encode()
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

        # Extract embedding; normalize so downstream cosine scoring holds for
        # any external model (raises for zero/non-numeric vectors).
        try:
            if self.api == "openai":
                embedding = data["data"][0]["embedding"]
            else:
                embedding = data["embedding"]
            if not isinstance(embedding, list):
                raise TypeError(f"embedding field is {type(embedding).__name__}, expected list")
            return _l2_normalize(embedding)
        except (KeyError, IndexError):
            expected = "data[0].embedding" if self.api == "openai" else "embedding"
            got = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            logger.error(
                "Embedding service response missing '%s' field. Got: %s", expected, got,
            )
            raise RuntimeError(
                f"Embedding service response missing '{expected}' field. Got: {got}"
            )
        except (TypeError, ValueError) as e:
            logger.error("Embedding value is invalid: %s", e)
            raise RuntimeError(f"Embedding service returned invalid embedding: {e}")
