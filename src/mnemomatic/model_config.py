"""Metadata for the bundled embedding model.

The Docker model stage writes /app/model/model_config.json next to the model
weights, describing the model the image was built with: name, embedding
dimension, token truncation limit, and task prefixes. Server defaults are
driven by this file, so selecting a model at build time (EMBED_MODEL build
arg) needs no matching runtime configuration. Explicit MNEMOMATIC_* env vars
always override it.

Native (non-Docker) runs usually have no config file; every consumer falls
back to conservative defaults (384 dims, no prefixes, 512 tokens, mean
pooling) — the shape an unconfigured external embedder is most likely to have.
"""

import json
import logging
import os

logger = logging.getLogger("mnemomatic")

MODEL_CONFIG_PATH = os.environ.get(
    "MNEMOMATIC_MODEL_CONFIG_PATH", "/app/model/model_config.json"
)


def load() -> dict:
    """Read the bundled model's metadata, or {} when absent or unreadable."""
    try:
        with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Ignoring unreadable model config %s: %s", MODEL_CONFIG_PATH, e)
        return {}
    if not isinstance(config, dict):
        logger.warning("Ignoring model config %s: expected a JSON object", MODEL_CONFIG_PATH)
        return {}
    return config


CONFIG = load()
