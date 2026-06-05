"""Characterization tests for the search() tool's mode-dispatch.

search() is the most-used tool but its dispatch matrix — the interaction of
mode (hybrid/fulltext/semantic), embedder presence, and embedding success — had
no unit coverage. These tests pin the current behavior exactly so the dispatch
can be refactored safely.

The matrix (7 cells):
  fulltext                       -> search_fts,    not degraded
  semantic + no embedder         -> error "Semantic search not available"
  semantic + embedder + ok       -> search_vec
  semantic + embedder + fail     -> error "Semantic search failed"
  hybrid   + no embedder         -> search_fts,    degraded (+_metadata), no log
  hybrid   + embedder + ok       -> search_hybrid
  hybrid   + embedder + fail     -> search_fts,    degraded (+_metadata), info-log

Nuances pinned: the two distinct semantic error messages; the degrade info-log
fires ONLY on embed-failure, not the no-embedder case; semantic/hybrid embed the
original query while FTS uses the escaped query; validation and limit clamping.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mnemomatic.server as server

# A non-None stand-in for "an embedder is configured". search() only checks the
# embedder for None-ness; the actual embedding is produced by _safe_embed, which
# we patch independently.
SENTINEL_EMBEDDER = object()
EMBEDDING = [0.1, 0.2, 0.3]


class FakeResult:
    """A search hit whose model_dump() identifies which backend produced it."""

    def __init__(self, tag):
        self.tag = tag

    def model_dump(self):
        return {"source": self.tag}


class FakeDB:
    """Records which search_* method ran and with what arguments."""

    def __init__(self):
        self.calls = []

    def search_fts(self, query, table, namespace, limit):
        self.calls.append(("fts", query, table, namespace, limit))
        return [FakeResult("fts")]

    def search_vec(self, embedding, table, namespace, limit):
        self.calls.append(("vec", embedding, table, namespace, limit))
        return [FakeResult("vec")]

    def search_hybrid(self, query, embedding, table, namespace, limit):
        self.calls.append(("hybrid", query, embedding, table, namespace, limit))
        return [FakeResult("hybrid")]


class SearchDispatchTest(unittest.TestCase):
    def setUp(self):
        self.fake_db = FakeDB()

    def _search(self, embedder, embed_result, **kwargs):
        """Run search() with the DB, embedder presence, and embed outcome stubbed.

        embedder:     object returned by _embedder() — SENTINEL_EMBEDDER or None.
        embed_result: value returned by _safe_embed() — an embedding list or None.
        """
        with patch.object(server, "_db", return_value=self.fake_db), \
             patch.object(server, "_embedder", return_value=embedder), \
             patch.object(server, "_safe_embed", return_value=embed_result):
            return server.search(**kwargs)

    def _backends(self):
        return [c[0] for c in self.fake_db.calls]

    # --- the 7-cell dispatch matrix ------------------------------------------

    def test_fulltext_uses_fts_not_degraded(self):
        # fulltext ignores the embedder even when one is present.
        res = self._search(SENTINEL_EMBEDDER, EMBEDDING, query="hello", mode="fulltext")
        self.assertEqual(self._backends(), ["fts"])
        self.assertEqual(res, [{"source": "fts"}])  # no _metadata appended

    def test_semantic_no_embedder_returns_not_available(self):
        res = self._search(None, None, query="hello", mode="semantic")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["error"], "Semantic search not available")
        self.assertEqual(self.fake_db.calls, [])  # no DB query attempted

    def test_semantic_embed_ok_uses_vec(self):
        res = self._search(SENTINEL_EMBEDDER, EMBEDDING, query="hello", mode="semantic")
        self.assertEqual(self._backends(), ["vec"])
        self.assertEqual(self.fake_db.calls[0][1], EMBEDDING)  # embedding handed to vec
        self.assertEqual(res, [{"source": "vec"}])

    def test_semantic_embed_fail_returns_failed(self):
        # Distinct message from the no-embedder case.
        res = self._search(SENTINEL_EMBEDDER, None, query="hello", mode="semantic")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["error"], "Semantic search failed")
        self.assertEqual(self.fake_db.calls, [])

    def test_hybrid_no_embedder_degrades_silently(self):
        # No embedder: degrade to FTS and append the degrade entry, but DO NOT
        # log — the info-log is reserved for the embed-failure case below. The
        # appended _metadata content itself is pinned by test_degraded_metadata.
        with self.assertNoLogs("mnemomatic", level="INFO"):
            res = self._search(None, None, query="hello", mode="hybrid")
        self.assertEqual(self._backends(), ["fts"])
        self.assertEqual(res[0], {"source": "fts"})
        self.assertEqual(len(res), 2)  # fts hit + degrade entry

    def test_hybrid_embed_ok_uses_hybrid(self):
        res = self._search(SENTINEL_EMBEDDER, EMBEDDING, query="hello", mode="hybrid")
        self.assertEqual(self._backends(), ["hybrid"])
        self.assertEqual(res, [{"source": "hybrid"}])  # no degradation metadata

    def test_hybrid_embed_fail_degrades_with_log(self):
        # Embedder present but embedding fails: degrade to FTS AND log the reason.
        with self.assertLogs("mnemomatic", level="INFO") as cm:
            res = self._search(SENTINEL_EMBEDDER, None, query="hello", mode="hybrid")
        self.assertEqual(self._backends(), ["fts"])
        self.assertEqual(len(res), 2)  # fts hit + degrade entry
        self.assertTrue(any("degrading to fulltext" in m for m in cm.output))

    # --- degradation metadata shape ------------------------------------------

    def test_degraded_metadata(self):
        # Both degrade branches append this exact entry from one code path, so
        # pinning it once (here) covers the metadata content for #5 and #7 too.
        res = self._search(None, None, query="hello", mode="hybrid")
        self.assertEqual(
            res[-1],
            {"_metadata": {
                "degraded": True,
                "reason": "Semantic search unavailable; results from fulltext search only",
            }},
        )

    # --- input validation (single-element error list, no DB call) ------------

    def test_invalid_content_type_rejected(self):
        res = self._search(SENTINEL_EMBEDDER, EMBEDDING, query="hi", content_type="bogus")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["error"], "Invalid content_type")
        self.assertEqual(self.fake_db.calls, [])

    def test_invalid_mode_rejected(self):
        res = self._search(SENTINEL_EMBEDDER, EMBEDDING, query="hi", mode="bogus")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["error"], "Invalid search mode")
        self.assertEqual(self.fake_db.calls, [])

    def test_empty_query_rejected(self):
        for q in ("", "   "):
            with self.subTest(query=repr(q)):
                self.fake_db.calls.clear()
                res = self._search(SENTINEL_EMBEDDER, EMBEDDING, query=q)
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0]["error"], "Query cannot be empty")
                self.assertEqual(self.fake_db.calls, [])

    # --- limit clamping to [1, MAX_SEARCH_LIMIT] -----------------------------

    def test_limit_clamped_to_range(self):
        # (requested, expected) — below range, above range, and in range.
        cases = [(0, 1), (10_000, server.MAX_SEARCH_LIMIT), (25, 25)]
        for requested, expected in cases:
            with self.subTest(limit=requested):
                self.fake_db.calls.clear()
                self._search(SENTINEL_EMBEDDER, EMBEDDING, query="hi",
                             mode="fulltext", limit=requested)
                self.assertEqual(self.fake_db.calls[0][4], expected)

    # --- query routing & passthrough -----------------------------------------

    def test_content_type_and_namespace_passthrough(self):
        self._search(
            SENTINEL_EMBEDDER, EMBEDDING, query="hi", mode="fulltext",
            content_type="documents", namespace="proj",
        )
        _, _query, table, namespace, _limit = self.fake_db.calls[0]
        self.assertEqual(table, "documents")
        self.assertEqual(namespace, "proj")

    def test_fts_escapes_query_while_embedding_uses_raw(self):
        # A query containing an FTS operator must reach FTS quoted, but the
        # embedder must receive the original, unescaped query.
        safe = MagicMock(return_value=EMBEDDING)
        with patch.object(server, "_db", return_value=self.fake_db), \
             patch.object(server, "_embedder", return_value=SENTINEL_EMBEDDER), \
             patch.object(server, "_safe_embed", safe):
            server.search(query="a AND b", mode="hybrid")
        safe.assert_called_once_with("a AND b")          # raw query embedded
        backend, fts_arg, embedding, _table, _ns, _limit = self.fake_db.calls[0]
        self.assertEqual(backend, "hybrid")
        self.assertEqual(fts_arg, '"a AND b"')           # FTS arg escaped
        self.assertEqual(embedding, EMBEDDING)           # embedding handed to hybrid


if __name__ == "__main__":
    unittest.main()
