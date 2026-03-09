"""Unit tests for the Database layer.

Uses in-memory SQLite — no Docker or live server required.

Run with: python -m unittest tests/test_db.py -v
"""

import math
import random
import unittest

from mnemomatic.db import Database, _DOCUMENT_FIELDS, _KNOWLEDGE_FIELDS, _NOTE_FIELDS
from mnemomatic.models import Document, Knowledge, Note

EMBEDDING_DIM = 384


def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding — seeded by text hash, L2-normalised."""
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    vec = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# ── Documents ──────────────────────────────────────────────────────────────────

class TestDocumentCRUD(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_store_and_get(self):
        doc = Document(namespace="ns", title="Title", content="Body")
        stored, created = self.db.store_document(doc, _fake_embedding("Title\nBody"))
        self.assertTrue(created)
        self.assertEqual(stored.title, "Title")
        fetched = self.db.get_document(stored.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.content, "Body")
        self.assertEqual(fetched.namespace, "ns")

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.db.get_document("no-such-id"))

    def test_upsert_updates_in_place(self):
        doc = Document(namespace="ns", title="T", content="v1")
        stored, created = self.db.store_document(doc, _fake_embedding("T\nv1"))
        self.assertTrue(created)

        doc2 = Document(namespace="ns", title="T", content="v2")
        stored2, created2 = self.db.store_document(doc2, _fake_embedding("T\nv2"))
        self.assertFalse(created2)
        self.assertEqual(stored2.id, stored.id)
        self.assertEqual(stored2.content, "v2")
        self.assertEqual(stored2.created_at, stored.created_at)

    def test_upsert_different_namespace_creates_new(self):
        doc_a = Document(namespace="a", title="T", content="C")
        doc_b = Document(namespace="b", title="T", content="C")
        _, created_a = self.db.store_document(doc_a, _fake_embedding("T\nC"))
        _, created_b = self.db.store_document(doc_b, _fake_embedding("T\nC"))
        self.assertTrue(created_a)
        self.assertTrue(created_b)

    def test_update_content(self):
        doc = Document(namespace="ns", title="T", content="old")
        stored, _ = self.db.store_document(doc, _fake_embedding("T\nold"))
        updated = self.db.update_document(stored.id, content="new")
        self.assertEqual(updated.content, "new")
        self.assertEqual(self.db.get_document(stored.id).content, "new")

    def test_update_with_embedding(self):
        doc = Document(namespace="ns", title="T", content="old")
        stored, _ = self.db.store_document(doc, _fake_embedding("T\nold"))
        new_emb = _fake_embedding("T\nnew")
        updated = self.db.update_document(stored.id, content="new", embedding=new_emb)
        self.assertEqual(updated.content, "new")

    def test_update_nonexistent_returns_none(self):
        self.assertIsNone(self.db.update_document("no-such-id", content="x"))

    def test_update_invalid_field_raises(self):
        doc = Document(namespace="ns", title="T", content="C")
        stored, _ = self.db.store_document(doc, _fake_embedding("T\nC"))
        with self.assertRaises(ValueError):
            self.db.update_document(stored.id, bad_field="x")

    def test_delete(self):
        doc = Document(namespace="ns", title="T", content="C")
        stored, _ = self.db.store_document(doc, _fake_embedding("T\nC"))
        self.assertTrue(self.db.delete_document(stored.id))
        self.assertIsNone(self.db.get_document(stored.id))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.db.delete_document("no-such-id"))

    def test_list(self):
        for i in range(3):
            doc = Document(namespace="ns", title=f"T{i}", content="C")
            self.db.store_document(doc, _fake_embedding(f"T{i}\nC"))
        self.assertEqual(len(self.db.list_documents("ns")), 3)
        self.assertEqual(len(self.db.list_documents("other")), 0)

    def test_allowlist_constants(self):
        self.assertIn("title", _DOCUMENT_FIELDS)
        self.assertIn("content", _DOCUMENT_FIELDS)
        self.assertNotIn("namespace", _DOCUMENT_FIELDS)
        self.assertNotIn("id", _DOCUMENT_FIELDS)


# ── Knowledge ──────────────────────────────────────────────────────────────────

class TestKnowledgeCRUD(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_store_and_get(self):
        k = Knowledge(namespace="ns", subject="auth", fact="Uses JWT")
        stored, created = self.db.store_knowledge(k, _fake_embedding("auth: Uses JWT"))
        self.assertTrue(created)
        fetched = self.db.get_knowledge(stored.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.fact, "Uses JWT")

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.db.get_knowledge("no-such-id"))

    def test_upsert_updates_in_place(self):
        k = Knowledge(namespace="ns", subject="db", fact="Postgres")
        stored, _ = self.db.store_knowledge(k, _fake_embedding("db: Postgres"))
        k2 = Knowledge(namespace="ns", subject="db", fact="SQLite")
        stored2, created2 = self.db.store_knowledge(k2, _fake_embedding("db: SQLite"))
        self.assertFalse(created2)
        self.assertEqual(stored2.id, stored.id)
        self.assertEqual(stored2.fact, "SQLite")

    def test_update_fact(self):
        k = Knowledge(namespace="ns", subject="auth", fact="old")
        stored, _ = self.db.store_knowledge(k, _fake_embedding("auth: old"))
        updated = self.db.update_knowledge(stored.id, fact="new")
        self.assertEqual(updated.fact, "new")

    def test_update_nonexistent_returns_none(self):
        self.assertIsNone(self.db.update_knowledge("no-such-id", fact="x"))

    def test_update_invalid_field_raises(self):
        k = Knowledge(namespace="ns", subject="s", fact="f")
        stored, _ = self.db.store_knowledge(k, _fake_embedding("s: f"))
        with self.assertRaises(ValueError):
            self.db.update_knowledge(stored.id, bad_field="x")

    def test_delete(self):
        k = Knowledge(namespace="ns", subject="s", fact="f")
        stored, _ = self.db.store_knowledge(k, _fake_embedding("s: f"))
        self.assertTrue(self.db.delete_knowledge(stored.id))
        self.assertIsNone(self.db.get_knowledge(stored.id))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.db.delete_knowledge("no-such-id"))

    def test_list(self):
        for i in range(3):
            k = Knowledge(namespace="ns", subject=f"s{i}", fact="f")
            self.db.store_knowledge(k, _fake_embedding(f"s{i}: f"))
        self.assertEqual(len(self.db.list_knowledge("ns")), 3)
        self.assertEqual(len(self.db.list_knowledge("other")), 0)

    def test_allowlist_constants(self):
        self.assertIn("fact", _KNOWLEDGE_FIELDS)
        self.assertIn("confidence", _KNOWLEDGE_FIELDS)
        self.assertNotIn("namespace", _KNOWLEDGE_FIELDS)
        self.assertNotIn("id", _KNOWLEDGE_FIELDS)


# ── Notes ──────────────────────────────────────────────────────────────────────

class TestNoteCRUD(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_store_and_get(self):
        note = Note(namespace="ns", title="Idea", content="Quick thought")
        stored, created = self.db.store_note(note, _fake_embedding("Idea\nQuick thought"))
        self.assertTrue(created)
        fetched = self.db.get_note(stored.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.content, "Quick thought")

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.db.get_note("no-such-id"))

    def test_upsert_updates_in_place(self):
        note = Note(namespace="ns", title="T", content="v1")
        stored, _ = self.db.store_note(note, _fake_embedding("T\nv1"))
        note2 = Note(namespace="ns", title="T", content="v2")
        stored2, created2 = self.db.store_note(note2, _fake_embedding("T\nv2"))
        self.assertFalse(created2)
        self.assertEqual(stored2.id, stored.id)
        self.assertEqual(stored2.content, "v2")

    def test_update_content(self):
        note = Note(namespace="ns", title="T", content="old")
        stored, _ = self.db.store_note(note, _fake_embedding("T\nold"))
        updated = self.db.update_note(stored.id, content="new")
        self.assertEqual(updated.content, "new")

    def test_update_nonexistent_returns_none(self):
        self.assertIsNone(self.db.update_note("no-such-id", content="x"))

    def test_update_invalid_field_raises(self):
        note = Note(namespace="ns", title="T", content="C")
        stored, _ = self.db.store_note(note, _fake_embedding("T\nC"))
        with self.assertRaises(ValueError):
            self.db.update_note(stored.id, bad_field="x")

    def test_delete(self):
        note = Note(namespace="ns", title="T", content="C")
        stored, _ = self.db.store_note(note, _fake_embedding("T\nC"))
        self.assertTrue(self.db.delete_note(stored.id))
        self.assertIsNone(self.db.get_note(stored.id))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.db.delete_note("no-such-id"))

    def test_list(self):
        for i in range(3):
            note = Note(namespace="ns", title=f"T{i}", content="C")
            self.db.store_note(note, _fake_embedding(f"T{i}\nC"))
        self.assertEqual(len(self.db.list_notes("ns")), 3)
        self.assertEqual(len(self.db.list_notes("other")), 0)

    def test_allowlist_constants(self):
        self.assertIn("content", _NOTE_FIELDS)
        self.assertIn("source", _NOTE_FIELDS)
        self.assertNotIn("namespace", _NOTE_FIELDS)
        self.assertNotIn("id", _NOTE_FIELDS)


# ── Tags ───────────────────────────────────────────────────────────────────────

class TestTags(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")
        doc = Document(namespace="ns", title="T", content="C", tags=["a", "b"])
        self.doc_id = self.db.store_document(doc, _fake_embedding("T\nC"))[0].id

    def tearDown(self):
        self.db.close()

    def test_add_tags(self):
        tags = self.db.update_tags(self.doc_id, "document", add_tags=["c"])
        self.assertIn("c", tags)
        self.assertIn("a", tags)

    def test_remove_tags(self):
        tags = self.db.update_tags(self.doc_id, "document", remove_tags=["a"])
        self.assertNotIn("a", tags)
        self.assertIn("b", tags)

    def test_add_and_remove_in_one_call(self):
        tags = self.db.update_tags(self.doc_id, "document", add_tags=["c"], remove_tags=["a"])
        self.assertIn("c", tags)
        self.assertNotIn("a", tags)

    def test_add_duplicate_tag_is_idempotent(self):
        tags = self.db.update_tags(self.doc_id, "document", add_tags=["a"])
        self.assertEqual(tags.count("a"), 1)

    def test_remove_missing_tag_is_idempotent(self):
        tags = self.db.update_tags(self.doc_id, "document", remove_tags=["nonexistent"])
        self.assertIn("a", tags)

    def test_invalid_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.db.update_tags(self.doc_id, "invalid_type", add_tags=["x"])

    def test_tags_work_on_knowledge(self):
        k = Knowledge(namespace="ns", subject="s", fact="f", tags=["x"])
        k_id = self.db.store_knowledge(k, _fake_embedding("s: f"))[0].id
        tags = self.db.update_tags(k_id, "knowledge", add_tags=["y"])
        self.assertIn("x", tags)
        self.assertIn("y", tags)

    def test_tags_work_on_notes(self):
        note = Note(namespace="ns", title="T", content="C", tags=["x"])
        note_id = self.db.store_note(note, _fake_embedding("T\nC"))[0].id
        tags = self.db.update_tags(note_id, "note", add_tags=["y"])
        self.assertIn("y", tags)


# ── Namespaces ─────────────────────────────────────────────────────────────────

class TestNamespaces(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_lists_all_namespaces(self):
        doc = Document(namespace="alpha", title="T", content="C")
        self.db.store_document(doc, _fake_embedding("T\nC"))
        k = Knowledge(namespace="beta", subject="s", fact="f")
        self.db.store_knowledge(k, _fake_embedding("s: f"))
        note = Note(namespace="gamma", title="T", content="C")
        self.db.store_note(note, _fake_embedding("T\nC"))

        namespaces = self.db.list_namespaces()
        self.assertIn("alpha", namespaces)
        self.assertIn("beta", namespaces)
        self.assertIn("gamma", namespaces)

    def test_empty_db_returns_empty_list(self):
        self.assertEqual(self.db.list_namespaces(), [])

    def test_deduplicates_namespaces(self):
        for i in range(3):
            doc = Document(namespace="shared", title=f"T{i}", content="C")
            self.db.store_document(doc, _fake_embedding(f"T{i}\nC"))
        namespaces = self.db.list_namespaces()
        self.assertEqual(namespaces.count("shared"), 1)


# ── Search ─────────────────────────────────────────────────────────────────────

class TestSearch(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")
        self.doc = Document(namespace="ns", title="authentication guide", content="JWT tokens for login")
        self.doc_emb = _fake_embedding("authentication guide\nJWT tokens for login")
        self.doc_id = self.db.store_document(self.doc, self.doc_emb)[0].id

        self.k = Knowledge(namespace="ns", subject="database choice", fact="SQLite for portability")
        self.k_emb = _fake_embedding("database choice: SQLite for portability")
        self.k_id = self.db.store_knowledge(self.k, self.k_emb)[0].id

        self.note = Note(namespace="ns", title="meeting notes", content="discussed deploy pipeline")
        self.note_emb = _fake_embedding("meeting notes\ndiscussed deploy pipeline")
        self.note_id = self.db.store_note(self.note, self.note_emb)[0].id

    def tearDown(self):
        self.db.close()

    def _ids(self, results):
        return [r.id for r in results]

    def test_fts_finds_document(self):
        results = self.db.search_fts("authentication", namespace="ns")
        self.assertIn(self.doc_id, self._ids(results))

    def test_fts_finds_knowledge(self):
        results = self.db.search_fts("SQLite", namespace="ns")
        self.assertIn(self.k_id, self._ids(results))

    def test_fts_finds_note(self):
        results = self.db.search_fts("deploy pipeline", namespace="ns")
        self.assertIn(self.note_id, self._ids(results))

    def test_fts_type_filter_documents_only(self):
        results = self.db.search_fts("SQLite", table="documents", namespace="ns")
        self.assertEqual(results, [])

    def test_fts_type_filter_knowledge_only(self):
        results = self.db.search_fts("SQLite", table="knowledge", namespace="ns")
        self.assertIn(self.k_id, self._ids(results))

    def test_fts_namespace_filter(self):
        other = Document(namespace="other", title="authentication", content="other content")
        self.db.store_document(other, _fake_embedding("authentication\nother content"))
        results = self.db.search_fts("authentication", namespace="ns")
        ids = self._ids(results)
        self.assertIn(self.doc_id, ids)
        for r in results:
            self.assertEqual(r.namespace, "ns")

    def test_fts_no_match_returns_empty(self):
        results = self.db.search_fts("xyznonexistent")
        self.assertEqual(results, [])

    def test_vec_returns_results(self):
        results = self.db.search_vec(self.doc_emb, namespace="ns")
        self.assertTrue(len(results) > 0)

    def test_vec_exact_embedding_is_top_result(self):
        results = self.db.search_vec(self.doc_emb, table="documents", namespace="ns")
        self.assertEqual(results[0].id, self.doc_id)
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

    def test_vec_type_filter(self):
        results = self.db.search_vec(self.doc_emb, table="knowledge", namespace="ns")
        for r in results:
            self.assertEqual(r.type, "knowledge")

    def test_hybrid_returns_results(self):
        results = self.db.search_hybrid("authentication", self.doc_emb, namespace="ns")
        self.assertTrue(len(results) > 0)
        self.assertIn(self.doc_id, self._ids(results))

    def test_hybrid_scores_are_positive(self):
        results = self.db.search_hybrid("authentication", self.doc_emb, namespace="ns")
        for r in results:
            self.assertGreater(r.score, 0)

    def test_search_deleted_item_not_returned(self):
        self.db.delete_document(self.doc_id)
        results = self.db.search_fts("authentication", namespace="ns")
        self.assertNotIn(self.doc_id, self._ids(results))

    def test_result_type_field(self):
        doc_results = self.db.search_fts("authentication", table="documents")
        self.assertTrue(all(r.type == "document" for r in doc_results))

        k_results = self.db.search_fts("SQLite", table="knowledge")
        self.assertTrue(all(r.type == "knowledge" for r in k_results))

        note_results = self.db.search_fts("deploy", table="notes")
        self.assertTrue(all(r.type == "note" for r in note_results))

    def test_limit_respected(self):
        for i in range(10):
            doc = Document(namespace="ns", title=f"auth doc {i}", content="authentication content")
            self.db.store_document(doc, _fake_embedding(f"auth doc {i}\nauthentication content"))
        results = self.db.search_fts("authentication", limit=3)
        self.assertLessEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
