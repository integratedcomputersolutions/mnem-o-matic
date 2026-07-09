"""Unit tests for the Database layer.

Uses in-memory SQLite — no Docker or live server required.

Run with: python -m unittest tests/test_db.py -v
"""

import math
import random
import signal
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlite_vec

import mnemomatic.db
from mnemomatic.db import (
    Database, SCHEMA_VERSION, _chunk_text, _serialize_embedding,
    _DOCUMENT_FIELDS, _KNOWLEDGE_FIELDS, _NOTE_FIELDS,
)
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
        # The new embedding must actually be written: querying with it returns
        # this doc as the exact (score ~1.0) top hit.
        results = self.db.search_vec(new_emb, table="documents", namespace="ns")
        self.assertEqual(results[0].id, stored.id)
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

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

    def test_restore_with_embedding_after_none_is_searchable(self):
        """Re-storing with an embedding must insert the vec row, not silently no-op.

        Regression: the upsert previously did a bare UPDATE on vec_knowledge, so an
        entry first stored without an embedding (FTS-only mode) never became
        semantically searchable once an embedder was added.
        """
        emb = _fake_embedding("auth: Uses JWT")
        self.db.store_knowledge(Knowledge(namespace="ns", subject="auth", fact="Uses JWT"), None)
        self.assertEqual(self.db.search_vec(emb, table="knowledge"), [])
        self.db.store_knowledge(Knowledge(namespace="ns", subject="auth", fact="Uses JWT"), emb)
        results = self.db.search_vec(emb, table="knowledge")
        self.assertTrue(any(r.title == "auth" for r in results))

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

    def test_restore_with_embedding_after_none_is_searchable(self):
        """Re-storing with an embedding must insert the vec row, not silently no-op.

        Regression: see the matching knowledge test — store_note had the same bare-UPDATE
        bug, so notes first stored FTS-only never became semantically searchable.
        """
        emb = _fake_embedding("Idea\nQuick thought")
        self.db.store_note(Note(namespace="ns", title="Idea", content="Quick thought"), None)
        self.assertEqual(self.db.search_vec(emb, table="notes"), [])
        self.db.store_note(Note(namespace="ns", title="Idea", content="Quick thought"), emb)
        results = self.db.search_vec(emb, table="notes")
        self.assertTrue(any(r.title == "Idea" for r in results))

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


# ── Rename Namespace ───────────────────────────────────────────────────────────

class TestRenameNamespace(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def _store_all(self):
        doc = Document(namespace="old", title="T", content="C")
        self.doc_id = self.db.store_document(doc, _fake_embedding("T\nC"))[0].id
        k = Knowledge(namespace="old", subject="s", fact="f")
        self.k_id = self.db.store_knowledge(k, _fake_embedding("s: f"))[0].id
        note = Note(namespace="old", title="N", content="C")
        self.note_id = self.db.store_note(note, _fake_embedding("N\nC"))[0].id

    def test_rename_moves_all_content_types(self):
        self._store_all()
        counts, replaced = self.db.rename_namespace("old", "new")
        self.assertEqual(counts, {"documents": 1, "knowledge": 1, "notes": 1})
        self.assertEqual(sum(replaced.values()), 0)
        self.assertNotIn("old", self.db.list_namespaces())
        self.assertIn("new", self.db.list_namespaces())

    def test_renamed_items_are_retrievable(self):
        self._store_all()
        self.db.rename_namespace("old", "new")
        self.assertEqual(self.db.get_document(self.doc_id).namespace, "new")
        self.assertEqual(self.db.get_knowledge(self.k_id).namespace, "new")
        self.assertEqual(self.db.get_note(self.note_id).namespace, "new")

    def test_rename_into_existing_namespace_merges(self):
        doc_old = Document(namespace="old", title="T-old", content="C")
        self.db.store_document(doc_old, _fake_embedding("T-old\nC"))
        doc_new = Document(namespace="new", title="T-new", content="C")
        self.db.store_document(doc_new, _fake_embedding("T-new\nC"))

        counts, replaced = self.db.rename_namespace("old", "new")
        self.assertEqual(counts["documents"], 1)
        self.assertEqual(sum(replaced.values()), 0)
        docs = self.db.list_documents("new")
        titles = {d.title for d in docs}
        self.assertIn("T-old", titles)
        self.assertIn("T-new", titles)

    def test_rename_conflict_moved_item_wins(self):
        # Merge semantics mirror the store_* upsert: on a title collision the
        # moved item replaces the target's, and the replacement is reported.
        doc_a, _ = self.db.store_document(
            Document(namespace="old", title="Same", content="from-old"), _fake_embedding("a"))
        doc_b, _ = self.db.store_document(
            Document(namespace="new", title="Same", content="from-new"), _fake_embedding("b"))

        counts, replaced = self.db.rename_namespace("old", "new")
        self.assertEqual(counts["documents"], 1)
        self.assertEqual(replaced["documents"], 1)
        docs = self.db.list_documents("new")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, doc_a.id)
        self.assertEqual(docs[0].content, "from-old")
        self.assertIsNone(self.db.get_document(doc_b.id))

    def test_rename_conflict_replaces_vectors_and_chunks(self):
        # The overwritten target's vector and chunk rows must not linger.
        loser_emb = _fake_embedding("loser")
        self.db.store_document(
            Document(namespace="new", title="Same", content="x" * 3000), None,
            chunks=[("loser chunk", loser_emb)],
        )
        winner, _ = self.db.store_document(
            Document(namespace="old", title="Same", content="winner"), _fake_embedding("winner"))

        _, replaced = self.db.rename_namespace("old", "new")
        self.assertEqual(replaced["documents"], 1)
        conn = self.db._get_conn()
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM document_chunks").fetchone()["n"], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM vec_document_chunks").fetchone()["n"], 0)
        # The loser's vector is gone: searching with it returns only the winner.
        results = self.db.search_vec(loser_emb, table="documents", namespace="new", limit=5)
        self.assertEqual([r.id for r in results], [winner.id])

    def test_rename_to_same_namespace_raises(self):
        self._store_all()
        with self.assertRaises(ValueError):
            self.db.rename_namespace("old", "old")
        # Nothing was deleted by the guard.
        self.assertEqual(len(self.db.list_documents("old")), 1)

    def test_rename_nonexistent_namespace_returns_zero_counts(self):
        counts, replaced = self.db.rename_namespace("ghost", "new")
        self.assertEqual(sum(counts.values()), 0)
        self.assertEqual(sum(replaced.values()), 0)

    def test_renamed_items_searchable_in_new_namespace(self):
        doc = Document(namespace="old", title="auth guide", content="JWT tokens")
        emb = _fake_embedding("auth guide\nJWT tokens")
        self.db.store_document(doc, emb)
        self.db.rename_namespace("old", "new")

        results = self.db.search_fts("JWT", namespace="new")
        self.assertTrue(len(results) > 0)
        results_old = self.db.search_fts("JWT", namespace="old")
        self.assertEqual(results_old, [])

    def test_namespace_counts(self):
        self._store_all()  # one of each type in "old"
        self.db.store_document(Document(namespace="zeta", title="d", content="c"), None)
        counts = self.db.namespace_counts()
        self.assertEqual(list(counts), ["old", "zeta"])  # sorted
        self.assertEqual(counts["old"], {"documents": 1, "knowledge": 1, "notes": 1})
        self.assertEqual(counts["zeta"], {"documents": 1, "knowledge": 0, "notes": 0})

    def test_namespace_counts_empty_db(self):
        self.assertEqual(self.db.namespace_counts(), {})


# ── Delete Namespace ────────────────────────────────────────────────────────────

class TestDeleteNamespace(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def _store_all(self, namespace="target"):
        doc = Document(namespace=namespace, title="T", content="C")
        self.doc_id = self.db.store_document(doc, _fake_embedding("T\nC"))[0].id
        k = Knowledge(namespace=namespace, subject="s", fact="f")
        self.k_id = self.db.store_knowledge(k, _fake_embedding("s: f"))[0].id
        note = Note(namespace=namespace, title="N", content="C")
        self.note_id = self.db.store_note(note, _fake_embedding("N\nC"))[0].id

    def test_delete_removes_all_content_types(self):
        self._store_all()
        counts = self.db.delete_namespace("target")
        self.assertEqual(counts, {"documents": 1, "knowledge": 1, "notes": 1})

    def test_deleted_namespace_gone_from_list(self):
        self._store_all()
        self.db.delete_namespace("target")
        self.assertNotIn("target", self.db.list_namespaces())

    def test_deleted_items_not_retrievable(self):
        self._store_all()
        self.db.delete_namespace("target")
        self.assertIsNone(self.db.get_document(self.doc_id))
        self.assertIsNone(self.db.get_knowledge(self.k_id))
        self.assertIsNone(self.db.get_note(self.note_id))

    def test_delete_only_affects_target_namespace(self):
        self._store_all("target")
        doc2 = Document(namespace="other", title="T", content="C")
        other_id = self.db.store_document(doc2, _fake_embedding("T\nC"))[0].id
        self.db.delete_namespace("target")
        self.assertIsNotNone(self.db.get_document(other_id))
        self.assertIn("other", self.db.list_namespaces())

    def test_delete_nonexistent_namespace_returns_zero_counts(self):
        counts = self.db.delete_namespace("ghost")
        self.assertEqual(sum(counts.values()), 0)

    def test_delete_multiple_items_per_type(self):
        for i in range(3):
            doc = Document(namespace="target", title=f"T{i}", content="C")
            self.db.store_document(doc, _fake_embedding(f"T{i}\nC"))
        for i in range(2):
            k = Knowledge(namespace="target", subject=f"s{i}", fact="f")
            self.db.store_knowledge(k, _fake_embedding(f"s{i}: f"))

        counts = self.db.delete_namespace("target")
        self.assertEqual(counts["documents"], 3)
        self.assertEqual(counts["knowledge"], 2)
        self.assertEqual(counts["notes"], 0)

    def test_deleted_items_not_returned_by_search(self):
        doc = Document(namespace="target", title="auth guide", content="JWT tokens")
        emb = _fake_embedding("auth guide\nJWT tokens")
        self.db.store_document(doc, emb)
        self.db.delete_namespace("target")

        results = self.db.search_fts("JWT", namespace="target")
        self.assertEqual(results, [])

    def test_deleted_items_not_returned_by_vec_search(self):
        doc = Document(namespace="target", title="auth guide", content="JWT tokens")
        emb = _fake_embedding("auth guide\nJWT tokens")
        self.db.store_document(doc, emb)
        self.db.delete_namespace("target")

        results = self.db.search_vec(emb, namespace="target")
        self.assertEqual(results, [])


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

    def test_vec_all_tables_finds_document(self):
        # Default table="all" unions every type; the stored doc must be present.
        results = self.db.search_vec(self.doc_emb, namespace="ns")
        self.assertIn(self.doc_id, [r.id for r in results])

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


# ── Namespace-partitioned vector search ────────────────────────────────────────

def _axis_embedding(axis: int, wobble: float = 0.0) -> list[float]:
    """A unit vector on `axis`, optionally tilted slightly toward axis 1."""
    vec = [0.0] * EMBEDDING_DIM
    vec[axis] = 1.0
    if wobble:
        vec[1] += wobble
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestVecNamespacePartition(unittest.TestCase):
    """Namespace filtering must happen inside the KNN, not by post-filtering.

    Regression: the old code fetched limit*3 global nearest neighbors and then
    dropped other-namespace rows in Python, so a small namespace drowned out by
    a large one returned zero results despite having perfectly good matches.
    """

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_small_namespace_not_drowned_out_by_large_one(self):
        # 30 "big" docs sit right next to the query vector; the 2 "small" docs
        # are orthogonal to it, so none of them are in the global top limit*3.
        for i in range(30):
            self.db.store_document(
                Document(namespace="big", title=f"big {i}", content="x"),
                _axis_embedding(0, wobble=0.001 * (i + 1)),
            )
        for i in range(2):
            self.db.store_document(
                Document(namespace="small", title=f"small {i}", content="x"),
                _axis_embedding(100 + i),
            )
        results = self.db.search_vec(_axis_embedding(0), table="documents", namespace="small", limit=5)
        self.assertEqual(sorted(r.title for r in results), ["small 0", "small 1"])

    def test_namespace_filter_applies_to_knowledge_and_notes(self):
        self.db.store_knowledge(Knowledge(namespace="a", subject="s", fact="f"), _axis_embedding(0))
        self.db.store_note(Note(namespace="b", title="n", content="c"), _axis_embedding(0, wobble=0.01))
        results = self.db.search_vec(_axis_embedding(0), table="all", namespace="a", limit=10)
        self.assertEqual([r.type for r in results], ["knowledge"])

    def test_chunk_search_respects_namespace(self):
        chunks_a = [("alpha chunk", _axis_embedding(0))]
        chunks_b = [("beta chunk", _axis_embedding(0, wobble=0.01))]
        self.db.store_document(Document(namespace="a", title="doc a", content="x" * 3000), None, chunks_a)
        self.db.store_document(Document(namespace="b", title="doc b", content="y" * 3000), None, chunks_b)
        results = self.db.search_vec(_axis_embedding(0), table="documents", namespace="a", limit=5)
        self.assertEqual([r.title for r in results], ["doc a"])
        self.assertEqual(results[0].snippet, "alpha chunk")

    def test_rename_namespace_moves_vectors(self):
        self.db.store_document(Document(namespace="old-ns", title="d", content="x"), _axis_embedding(0))
        self.db.store_document(
            Document(namespace="old-ns", title="big", content="x" * 3000), None,
            [("chunky", _axis_embedding(2))],
        )
        self.db.rename_namespace("old-ns", "new-ns")
        hits = self.db.search_vec(_axis_embedding(0), table="documents", namespace="new-ns", limit=5)
        self.assertIn("d", [r.title for r in hits])
        chunk_hits = self.db.search_vec(_axis_embedding(2), table="documents", namespace="new-ns", limit=5)
        self.assertIn("big", [r.title for r in chunk_hits])
        self.assertEqual(self.db.search_vec(_axis_embedding(0), table="documents", namespace="old-ns", limit=5), [])

    def test_upsert_after_fts_only_store_lands_in_namespace(self):
        # Stored without an embedding first (FTS-only mode), then re-stored
        # with one: the vec row must be inserted with the right partition.
        self.db.store_document(Document(namespace="ns", title="t", content="x"), None)
        self.db.store_document(Document(namespace="ns", title="t", content="x"), _axis_embedding(0))
        results = self.db.search_vec(_axis_embedding(0), table="documents", namespace="ns", limit=5)
        self.assertEqual([r.title for r in results], ["t"])


# ── Schema versioning + migration ──────────────────────────────────────────────

_LEGACY_DOCUMENTS_DDL = """
    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        mime_type TEXT NOT NULL DEFAULT 'text/markdown',
        tags TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE UNIQUE INDEX idx_documents_ns_title ON documents(namespace, title);
    CREATE TABLE document_chunks (
        id INTEGER PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL
    );
"""


def _build_legacy_db(path: str, dim: int = EMBEDDING_DIM) -> None:
    """Create a pre-versioning database: vec0 tables WITHOUT a partition key."""
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(_LEGACY_DOCUMENTS_DDL)
    conn.execute(f"CREATE VIRTUAL TABLE vec_documents USING vec0(embedding float[{dim}])")
    conn.execute(f"CREATE VIRTUAL TABLE vec_document_chunks USING vec0(embedding float[{dim}])")
    for i, ns in enumerate(["proj-a", "proj-a", "proj-b"]):
        rowid = conn.execute(
            "INSERT INTO documents (id, namespace, title, content, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00') RETURNING rowid",
            (f"id-{i}", ns, f"doc {i}", "content"),
        ).fetchone()[0]
        emb = [0.0] * dim
        emb[i] = 1.0
        conn.execute(
            "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_embedding(emb)),
        )
    chunk_rowid = conn.execute(
        "INSERT INTO document_chunks (document_id, chunk_index, content) VALUES ('id-0', 0, 'chunk text') RETURNING id"
    ).fetchone()[0]
    emb = [0.0] * dim
    emb[5] = 1.0
    conn.execute(
        "INSERT INTO vec_document_chunks (rowid, embedding) VALUES (?, ?)",
        (chunk_rowid, _serialize_embedding(emb)),
    )
    conn.commit()
    conn.close()


class TestSchemaMigration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.path = self._tmp.name
        Path(self.path).unlink()  # legacy builder wants to create it fresh

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def _user_version(self, db: Database) -> int:
        return db._get_conn().execute("PRAGMA user_version").fetchone()["user_version"]

    def test_fresh_database_is_current_version(self):
        db = Database(self.path)
        try:
            self.assertEqual(self._user_version(db), SCHEMA_VERSION)
            meta = db._get_conn().execute("SELECT value FROM schema_meta WHERE key='embed_dim'").fetchone()
            self.assertEqual(int(meta["value"]), EMBEDDING_DIM)
        finally:
            db.close()

    def test_legacy_database_is_migrated_with_data_preserved(self):
        _build_legacy_db(self.path)
        db = Database(self.path)
        try:
            self.assertEqual(self._user_version(db), SCHEMA_VERSION)
            sql = db._get_conn().execute(
                "SELECT sql FROM sqlite_master WHERE name='vec_documents'"
            ).fetchone()["sql"]
            self.assertIn("partition key", sql.lower())
            # Embeddings survived and are namespace-partitioned now.
            results = db.search_vec(_axis_embedding(2), table="documents", namespace="proj-b", limit=5)
            self.assertEqual([r.title for r in results], ["doc 2"])
            # Chunk vector survived too (chunk hit shadows the whole-doc one).
            chunk_hits = db.search_vec(_axis_embedding(5), table="documents", namespace="proj-a", limit=5)
            self.assertIn("chunk text", [r.snippet for r in chunk_hits])
            # Old data remains readable.
            self.assertEqual(db.get_document("id-0").title, "doc 0")
        finally:
            db.close()

    def test_migration_is_idempotent_across_reopens(self):
        _build_legacy_db(self.path)
        Database(self.path).close()
        db = Database(self.path)  # must not attempt to migrate again
        try:
            self.assertEqual(self._user_version(db), SCHEMA_VERSION)
            results = db.search_vec(_axis_embedding(2), table="documents", namespace="proj-b", limit=5)
            self.assertEqual(len(results), 1)
        finally:
            db.close()

    def test_dim_mismatch_on_versioned_db_fails_fast(self):
        Database(self.path).close()  # created with the real EMBEDDING_DIM
        with patch.object(mnemomatic.db, "EMBEDDING_DIM", EMBEDDING_DIM * 2):
            with self.assertRaises(RuntimeError) as cm:
                Database(self.path)
        self.assertIn("MNEMOMATIC_EMBED_DIM", str(cm.exception))
        self.assertIn(str(EMBEDDING_DIM), str(cm.exception))

    def test_dim_mismatch_on_legacy_db_fails_before_migrating(self):
        _build_legacy_db(self.path, dim=EMBEDDING_DIM)
        with patch.object(mnemomatic.db, "EMBEDDING_DIM", 8):
            with self.assertRaises(RuntimeError) as cm:
                Database(self.path)
        self.assertIn("MNEMOMATIC_EMBED_DIM", str(cm.exception))
        # The failed migration must not have bumped the version or dropped data.
        conn = sqlite3.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 3)
        conn.close()


# ── Chunking ───────────────────────────────────────────────────────────────────

class TestChunkText(unittest.TestCase):
    """_chunk_text must terminate and make forward progress on any input.

    Regression: a paragraph break landing within `overlap` chars of a window's
    start made `start = end - overlap` stall forever, appending identical
    chunks until memory ran out (server killed mid tool-call).
    """

    def setUp(self):
        # Abort loudly instead of hanging CI if the loop ever regresses.
        signal.alarm(20)

    def tearDown(self):
        signal.alarm(0)

    def _assert_covers(self, text, chunks, chunk_size):
        self.assertGreater(len(chunks), 0)
        self.assertTrue(text.startswith(chunks[0][:10]))
        self.assertTrue(text.endswith(chunks[-1][-10:]))
        self.assertGreaterEqual(sum(len(c) for c in chunks), len(text))
        for c in chunks:
            self.assertLessEqual(len(c), chunk_size)

    def test_paragraph_break_near_window_start_terminates(self):
        # A "\n\n" exactly `overlap` chars past a window start reproduced the
        # stall: end = break + 2 = 365, next start = 365 - 200 = 165 = start.
        text = "x" * 363 + "\n\n" + "y" * 2000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        self.assertLess(len(chunks), 20)
        self._assert_covers(text, chunks, 1000)

    def test_all_paragraph_breaks_early_in_window(self):
        # Repeated short paragraphs followed by long unbreakable runs keep the
        # candidate break early in every window.
        text = ("ab\n\n" + "z" * 1500) * 5
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        self.assertLess(len(chunks), 50)
        self._assert_covers(text, chunks, 1000)

    def test_no_break_points_at_all(self):
        text = "a" * 5000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        self.assertLess(len(chunks), 10)
        self._assert_covers(text, chunks, 1000)

    def test_short_text_single_chunk(self):
        self.assertEqual(_chunk_text("short", chunk_size=1000, overlap=200), ["short"])

    def test_overlap_not_smaller_than_chunk_size_still_terminates(self):
        # Nonsensical config must degrade gracefully, not loop forever.
        text = "word. " * 500
        chunks = _chunk_text(text, chunk_size=100, overlap=100)
        self._assert_covers(text, chunks, 100)

    def test_consecutive_chunks_overlap(self):
        # The start of each chunk re-covers the tail of the previous one.
        text = ("Sentence one is here. " * 20 + "\n\n") * 10
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        self.assertGreater(len(chunks), 1)
        for a, b in zip(chunks, chunks[1:]):
            self.assertIn(b[:20], a, "consecutive chunks should share overlapping text")


if __name__ == "__main__":
    unittest.main()
