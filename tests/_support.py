"""Shared helpers for the test suite.

Import with the package prefix so both documented ways of running the tests
work — `python -m pytest` and `python -m unittest tests/test_db.py`:

    from tests._support import EMBEDDING_DIM, FakeEmbedder, axis
"""

import math

from mnemomatic import server as runtime

# The module owning the runtime singletons tests patch (`_db`, `_embedder`,
# `_safe_embed`). Patch through this alias rather than importing
# mnemomatic.server directly: when those singletons move to their own module,
# the patch target changes here once instead of at every call site.
__all__ = [
    "EMBEDDING_DIM", "FakeEmbedder", "axis", "mix", "tilted_axis", "runtime",
]

# The dimension the suite embeds at. Matches the default the server falls back
# to with no bundled model config, which is the state tests run in.
EMBEDDING_DIM = 384


def axis(i: int, dim: int = EMBEDDING_DIM, scale: float = 1.0) -> list[float]:
    """A vector pointing along axis `i` — unit length unless `scale` says otherwise."""
    vec = [0.0] * dim
    vec[i] = scale
    return vec


def mix(i: int, j: int, wi: float, wj: float, dim: int = EMBEDDING_DIM) -> list[float]:
    """A normalized blend of two axes — cosine to axis `i` is `wi`."""
    norm = (wi * wi + wj * wj) ** 0.5
    vec = [0.0] * dim
    vec[i] = wi / norm
    vec[j] = wj / norm
    return vec


def tilted_axis(i: int, wobble: float = 0.0, dim: int = EMBEDDING_DIM) -> list[float]:
    """A unit vector on axis `i`, optionally tilted slightly toward axis 1."""
    vec = [0.0] * dim
    vec[i] = 1.0
    if wobble:
        vec[1] += wobble
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class FakeEmbedder:
    """Deterministic embedder: axis chosen by text hash, configurable dim."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vec = [0.0] * self.dim
        vec[hash(text) % self.dim] = 1.0
        return vec
