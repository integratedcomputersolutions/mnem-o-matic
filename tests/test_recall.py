"""Tests for the recall improvements: search filters and related items.

Covers tag/recency filtering across all three search modes and both search
legs (FTS and vector, including chunked documents), the interaction with
namespaces and superseded facts, plus the `related` tool: neighbor ranking,
self-exclusion, chunk-centroid fallback, and its error paths.
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from mnemomatic.db import CHUNK_THRESHOLD, Database
from mnemomatic.models import Document, Knowledge, Note
from mnemomatic import runtime
from mnemomatic import tools_search
from tests._support import axis, mix


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def _age(self, table, item_id, days):
        """Backdate an item's updated_at so recency filters have something to bite on."""
        ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self.db._get_conn()
        conn.execute(f"UPDATE {table} SET updated_at = ? WHERE id = ?", (ts, item_id))
        conn.commit()


class TestFtsFilters(DbTestCase):
    def setUp(self):
        super().setUp()
        self.old, _ = self.db.store_note(
            Note(namespace="proj", title="old note", content="shared topic",
                 tags=["archive", "docs"]), axis(0))
        self.new, _ = self.db.store_note(
            Note(namespace="proj", title="new note", content="shared topic",
                 tags=["docs"]), axis(0))
        self._age("notes", self.old.id, 100)

    def _ids(self, **kw):
        return {r.id for r in self.db.search_fts("topic", table="all", **kw)}

    def test_unfiltered_returns_both(self):
        self.assertEqual(self._ids(), {self.old.id, self.new.id})

    def test_single_tag(self):
        self.assertEqual(self._ids(tags=["archive"]), {self.old.id})

    def test_tags_and_together(self):
        self.assertEqual(self._ids(tags=["archive", "docs"]), {self.old.id})
        self.assertEqual(self._ids(tags=["archive", "missing"]), set())

    def test_updated_after(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        self.assertEqual(self._ids(updated_after=cutoff), {self.new.id})

    def test_filters_combine(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        self.assertEqual(self._ids(tags=["docs"], updated_after=cutoff), {self.new.id})

    def test_tag_match_is_exact_not_substring(self):
        self.assertEqual(self._ids(tags=["arch"]), set())  # not a prefix of "archive"


class TestVecFilters(DbTestCase):
    def setUp(self):
        super().setUp()
        self.tagged, _ = self.db.store_note(
            Note(namespace="proj", title="tagged", content="x", tags=["keep"]), axis(0))
        self.plain, _ = self.db.store_note(
            Note(namespace="proj", title="plain", content="y"), mix(0, 1, 0.9, 0.44))

    def _ids(self, **kw):
        return {r.id for r in self.db.search_vec(axis(0), table="all", **kw)}

    def test_tag_filter_applies_to_vector_leg(self):
        self.assertEqual(self._ids(), {self.tagged.id, self.plain.id})
        self.assertEqual(self._ids(tags=["keep"]), {self.tagged.id})

    def test_recency_filter_applies_to_vector_leg(self):
        self._age("notes", self.plain.id, 60)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        self.assertEqual(self._ids(updated_after=cutoff), {self.tagged.id})

    def test_chunked_document_filtered_by_tag(self):
        body = "chunk body paragraph.\n\n" * 200
        self.assertGreaterEqual(len(body), CHUNK_THRESHOLD)
        doc, _ = self.db.store_document(
            Document(namespace="proj", title="big", content=body, tags=["report"]),
            embedding=None, chunks=[(body[:500], axis(0)), (body[500:1000], axis(0))])
        hits = self.db.search_vec(axis(0), table="documents", tags=["report"])
        self.assertEqual([r.id for r in hits], [doc.id])
        self.assertTrue(hits[0].partial)
        self.assertEqual(self.db.search_vec(axis(0), table="documents", tags=["other"]), [])

    def test_hybrid_applies_filters_to_both_legs(self):
        results = self.db.search_hybrid("plain", axis(0), table="all", tags=["keep"])
        self.assertEqual([r.id for r in results], [self.tagged.id])


class TestFilterInteractions(DbTestCase):
    def test_namespace_and_tags_compose(self):
        a, _ = self.db.store_note(Note(namespace="a", title="n", content="topic", tags=["t"]), axis(0))
        self.db.store_note(Note(namespace="b", title="n", content="topic", tags=["t"]), axis(0))
        hits = self.db.search_fts("topic", namespace="a", tags=["t"])
        self.assertEqual([r.id for r in hits], [a.id])

    def test_superseded_facts_stay_excluded_when_filtering(self):
        self.db.store_knowledge(
            Knowledge(namespace="proj", subject="s", fact="old answer", tags=["t"]), axis(0))
        current, _, superseded = self.db.store_knowledge(
            Knowledge(namespace="proj", subject="s", fact="new answer", tags=["t"]), axis(0))
        self.assertIsNotNone(superseded)
        for hits in (self.db.search_fts("answer", tags=["t"]),
                     self.db.search_vec(axis(0), table="knowledge", tags=["t"])):
            self.assertEqual([r.id for r in hits], [current.id])


class TestItemEmbedding(DbTestCase):
    def test_returns_stored_vector(self):
        note, _ = self.db.store_note(Note(namespace="p", title="n", content="x"), axis(3))
        emb = self.db.item_embedding("note", note.id)
        self.assertEqual(emb[3], 1.0)

    def test_none_without_vector_or_item(self):
        note, _ = self.db.store_note(Note(namespace="p", title="n", content="x"), embedding=None)
        self.assertIsNone(self.db.item_embedding("note", note.id))
        self.assertIsNone(self.db.item_embedding("note", "no-such-id"))
        with self.assertRaises(ValueError):
            self.db.item_embedding("bogus", "x")

    def test_chunked_document_uses_chunk_centroid(self):
        doc, _ = self.db.store_document(
            Document(namespace="p", title="big", content="body"),
            embedding=None, chunks=[("a", axis(0)), ("b", axis(1))])
        emb = self.db.item_embedding("document", doc.id)
        # Mean of two orthogonal unit axes, renormalized: both components 1/sqrt(2).
        self.assertAlmostEqual(emb[0], 0.7071, places=3)
        self.assertAlmostEqual(emb[1], 0.7071, places=3)
        self.assertAlmostEqual(sum(v * v for v in emb) ** 0.5, 1.0, places=5)


class ToolTestCase(DbTestCase):
    def setUp(self):
        super().setUp()
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_embedder", return_value=object()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()


class TestRelatedTool(ToolTestCase):
    def setUp(self):
        super().setUp()
        self.anchor, _ = self.db.store_note(
            Note(namespace="proj", title="anchor", content="x"), axis(0))
        self.near, _, _ = self.db.store_knowledge(
            Knowledge(namespace="proj", subject="near", fact="y"), mix(0, 1, 0.95, 0.31))
        self.far, _ = self.db.store_note(
            Note(namespace="proj", title="far", content="z"), axis(1))
        self.elsewhere, _ = self.db.store_note(
            Note(namespace="other", title="elsewhere", content="w"), axis(0))

    def test_ranks_neighbors_and_excludes_self(self):
        resp = tools_search.related(item_type="note", id=self.anchor.id)
        ids = [r["id"] for r in resp["related"]]
        self.assertNotIn(self.anchor.id, ids)
        self.assertEqual(ids[0], self.elsewhere.id)  # identical vector, other namespace
        self.assertIn(self.near.id, ids)
        self.assertLess(ids.index(self.near.id), ids.index(self.far.id))

    def test_crosses_content_types(self):
        resp = tools_search.related(item_type="note", id=self.anchor.id)
        self.assertIn("knowledge", {r["type"] for r in resp["related"]})

    def test_namespace_scope(self):
        resp = tools_search.related(item_type="note", id=self.anchor.id, namespace="proj")
        self.assertNotIn(self.elsewhere.id, [r["id"] for r in resp["related"]])

    def test_limit_respected(self):
        resp = tools_search.related(item_type="note", id=self.anchor.id, limit=1)
        self.assertEqual(len(resp["related"]), 1)

    def test_records_access_for_neighbors_not_anchor(self):
        tools_search.related(item_type="note", id=self.anchor.id)
        self.assertEqual(self.db.get_note(self.anchor.id).retrieval_count, 0)
        self.assertEqual(self.db.get_note(self.far.id).retrieval_count, 1)

    def test_error_paths(self):
        self.assertIn("error", tools_search.related(item_type="bogus", id="x"))
        self.assertIn("error", tools_search.related(item_type="note", id="no-such-id"))
        unembedded, _ = self.db.store_note(
            Note(namespace="proj", title="no vector", content="x"), embedding=None)
        resp = tools_search.related(item_type="note", id=unembedded.id)
        self.assertIn("error", resp)
        self.assertIn("REINDEX", resp["details"])


class TestSearchToolFilters(ToolTestCase):
    def test_filters_reach_the_db(self):
        recent, _ = self.db.store_note(
            Note(namespace="proj", title="recent", content="topic", tags=["keep"]), axis(0))
        old, _ = self.db.store_note(
            Note(namespace="proj", title="old", content="topic", tags=["keep"]), axis(0))
        self._age("notes", old.id, 90)

        with patch.object(runtime, "_safe_embed", return_value=axis(0)):
            tagged = tools_search.search("topic", tags=["keep"], mode="fulltext")
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
            fresh = tools_search.search("topic", updated_after=cutoff, mode="hybrid")
        self.assertEqual({r["id"] for r in tagged}, {recent.id, old.id})
        self.assertEqual([r["id"] for r in fresh], [recent.id])

    def test_invalid_updated_after_is_rejected(self):
        result = tools_search.search("x", updated_after="last tuesday")
        self.assertIn("error", result[0])
        self.assertIn("ISO", result[0]["details"])

    def test_empty_tag_list_is_not_a_filter(self):
        note, _ = self.db.store_note(Note(namespace="proj", title="n", content="topic"), axis(0))
        results = tools_search.search("topic", tags=[], mode="fulltext")
        self.assertEqual([r["id"] for r in results], [note.id])


if __name__ == "__main__":
    unittest.main()
