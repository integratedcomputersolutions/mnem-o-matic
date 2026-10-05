"""Shared helpers for the test suite.

Import with the package prefix so both documented ways of running the tests
work — `python -m pytest` and `python -m unittest tests/test_db.py`:

    from tests._support import EMBEDDING_DIM, FakeEmbedder, axis
"""

import math
import random
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from starlette.testclient import TestClient

from mnemomatic import identity as identity_module
from mnemomatic import runtime
from mnemomatic.db import Database

__all__ = [
    "AMARETTO", "CookieClient", "EMBEDDING_DIM", "FakeEmbedder", "GEMMA", "IdentityFixture", "MemDbCase", "SPA_HTML",
    "ToolCase", "axis", "fast_scrypt", "mix", "random_unit_vector", "serve", "temp_db_path", "temp_dir",
    "tilted_axis",
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


def random_unit_vector(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """A deterministic dense unit vector seeded by the text — unlike axis(),
    every component is non-zero, as with a real model."""
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# Two embedder identities of equal dimension: the swap the dimension check
# cannot see, which the recorded model name exists to catch.
GEMMA = {
    "embed_model": "embeddinggemma-300m",
    "embed_query_prefix": "task: search result | query: ",
    "embed_doc_prefix": "title: none | text: ",
}
AMARETTO = {**GEMMA, "embed_model": "amaretto-embed-148m"}

# A stand-in for the built web UI's index.html.
SPA_HTML = "<!doctype html><html><head><title>Mnem-O-matic</title></head><body><div id=app></div></body></html>"


# What sign-in verifies for an unknown username, made at the cheap work factor
# like every other hash under fast_scrypt — the real one costs ~100 ms a check.
_CHEAP_DUMMY_HASH = identity_module.hash_password("dummy", log_n=10, p=1)


def fast_scrypt():
    """Patch the scrypt work factor down so the suite does not spend seconds hashing."""
    return patch.multiple(identity_module, SCRYPT_LOG_N=10, SCRYPT_P=1, DUMMY_HASH=_CHEAP_DUMMY_HASH)


def temp_dir(test: unittest.TestCase) -> Path:
    """A fresh directory removed, with everything in it, when `test` ends."""
    holder = tempfile.TemporaryDirectory(prefix="mnemomatic-test-")
    test.addCleanup(holder.cleanup)
    return Path(holder.name)


def temp_db_path(test: unittest.TestCase) -> Path:
    """Where a test's file-backed Database goes: not created yet, and removed
    with its -wal/-shm companions when `test` ends, whatever the test did."""
    return temp_dir(test) / "test.db"


def serve(test: unittest.TestCase, handler_cls) -> ThreadingHTTPServer:
    """A local HTTP server on a free port, on a thread, stopped when `test` ends."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    # serve_forever checks for shutdown every poll_interval (0.5 s by
    # default), so each stop would otherwise cost up to half a second.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return server


class CookieClient(TestClient):
    """A TestClient whose per-request `cookies=` go out as an explicit Cookie
    header. Starlette deprecates per-request cookies; and with an explicit
    header httpx leaves the client's cookie jar out, so the request carries
    exactly the cookies under test — not those of an earlier sign-in."""

    def request(self, method, url, *args, cookies=None, headers=None, **kwargs):
        if cookies:
            headers = dict(headers or {})
            headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in dict(cookies).items())
        return super().request(method, url, *args, headers=headers, **kwargs)


class MemDbCase(unittest.TestCase):
    """A fresh in-memory Database per test, as self.db."""

    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.close)


class ToolCase(MemDbCase):
    """Tool functions run against self.db: runtime._db and runtime._embedder
    are patched for the test. FTS-only unless a subclass sets `embedder`."""

    embedder = None

    def setUp(self):
        super().setUp()
        for target, value in (("_db", self.db), ("_embedder", self.embedder)):
            patcher = patch.object(runtime, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)


class FakeEmbedder:
    """Deterministic embedder: axis chosen by text hash, configurable dim."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.calls: list[str] = []

    @property
    def mode(self) -> str:
        """Real embedders describe themselves for the settings page; so does this."""
        return "fake (test)"

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vec = [0.0] * self.dim
        vec[hash(text) % self.dim] = 1.0
        return vec


class IdentityFixture:
    """A file-backed Database with an Identity store, one admin, one plain
    user, and a token for each — for tests that drive AuthMiddleware.

    File-backed because Starlette's TestClient serves on a worker thread and
    each thread gets its own connection. scrypt is patched down to a cheap
    work factor for the fixture's lifetime.

        fx = IdentityFixture(); self.addCleanup(fx.close)
        fx.admin_token, fx.user_token, fx.session_for(fx.admin)
    """

    ADMIN_PASSWORD = "admin-password-1"
    USER_PASSWORD = "user-password-12"

    def __init__(self):
        import tempfile
        from pathlib import Path

        from mnemomatic.identity import Identity

        self._patch = fast_scrypt()
        self._patch.start()
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.path = Path(tmp.name)
        self.db = Database(str(self.path))
        self.identity = Identity(self.db)
        self.admin, _ = self.identity.create_user("admin", role="admin", password=self.ADMIN_PASSWORD)
        self.user, _ = self.identity.create_user("alice", role="user", password=self.USER_PASSWORD)
        _, self.admin_token = self.identity.create_token(self.admin.id, "admin-agent")
        _, self.user_token = self.identity.create_token(self.user.id, "alice-agent")

    def session_for(self, user) -> str:
        return self.identity.create_session(user.id)

    def close(self):
        from pathlib import Path
        self.db.close()
        self._patch.stop()
        for p in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            p.unlink(missing_ok=True)
