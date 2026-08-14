"""Tests for Phase B: temporal facts (supersede instead of overwrite).

Covers the v3 migration (validity columns, partial unique index), the three
store outcomes (insert / same-fact refresh / supersede), the server update
path (fact change supersedes, other fields edit in place, history is
immutable), exclusion of superseded rows from search/listings/counts, the
fact_history tool, namespace rename/delete interplay, and restore guards.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import mnemomatic.db as db_module
import mnemomatic.server as server
from mnemomatic.db import EMBEDDING_DIM, Database
from mnemomatic.models import Knowledge
from mnemomatic import runtime


def _emb(seed: float) -> list[float]:
    return [seed] + [0.0] * (EMBEDDING_DIM - 1)


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def _store(self, fact, subject="topic", namespace="proj", embedding=None, **kw):
        return self.db.store_knowledge(
            Knowledge(namespace=namespace, subject=subject, fact=fact, **kw), embedding)

    def _vec_count(self):
        return self.db._get_conn().execute(
            "SELECT COUNT(*) AS n FROM vec_knowledge").fetchone()["n"]


class TestMigrationV3(unittest.TestCase):
    def test_v2_database_gains_validity_and_partial_index(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            # Build a current database, then strip it back to version-2 shape.
            db = Database(tmp.name)
            db.store_knowledge(Knowledge(namespace="p", subject="s", fact="f"), None)
            conn = db._get_conn()
            conn.execute("DROP INDEX idx_knowledge_ns_subject_current")
            conn.execute("ALTER TABLE knowledge DROP COLUMN valid_until")
            conn.execute("ALTER TABLE knowledge DROP COLUMN superseded_by")
            conn.execute("CREATE UNIQUE INDEX idx_knowledge_ns_subject ON knowledge(namespace, subject)")
            conn.execute("PRAGMA user_version = 2")
            conn.commit()
            db.close()

            migrated = Database(tmp.name)
            conn = migrated._get_conn()
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"],
                             db_module.SCHEMA_VERSION)
            row = conn.execute("SELECT valid_until, superseded_by FROM knowledge").fetchone()
            self.assertIsNone(row["valid_until"])  # existing facts stay current
            indexes = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='knowledge'")}
            self.assertIn("idx_knowledge_ns_subject_current", indexes)
            self.assertNotIn("idx_knowledge_ns_subject", indexes)
            migrated.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class TestStoreSemantics(DbTestCase):
    def test_different_fact_supersedes(self):
        old, _, _ = self._store("Postgres", embedding=_emb(0.1))
        new, created, superseded = self._store("SQLite", embedding=_emb(0.2))
        self.assertTrue(created)
        self.assertEqual(superseded, old.id)
        self.assertNotEqual(new.id, old.id)

        closed = self.db.get_knowledge(old.id)  # history stays readable by id
        self.assertIsNotNone(closed.valid_until)
        self.assertEqual(closed.superseded_by, new.id)
        self.assertEqual(closed.fact, "Postgres")
        current = self.db.get_knowledge(new.id)
        self.assertIsNone(current.valid_until)

    def test_same_fact_refreshes_in_place_without_history(self):
        old, _, _ = self._store("Postgres", confidence=1.0)
        new, created, superseded = self._store("Postgres", confidence=0.6)
        self.assertFalse(created)
        self.assertIsNone(superseded)
        self.assertEqual(new.id, old.id)
        self.assertEqual(self.db.knowledge_history("proj", "topic")[0].confidence, 0.6)
        self.assertEqual(len(self.db.knowledge_history("proj", "topic")), 1)

    def test_supersession_leaves_no_revision(self):
        old, _, _ = self._store("v1")
        self._store("v2")
        self.assertEqual(self.db.list_revisions(item_id=old.id), [])  # the row IS the history

    def test_superseded_vector_is_dropped(self):
        self._store("v1", embedding=_emb(0.1))
        self.assertEqual(self._vec_count(), 1)
        self._store("v2", embedding=_emb(0.2))
        self.assertEqual(self._vec_count(), 1)  # successor only

    def test_new_chain_after_delete(self):
        old, _, _ = self._store("v1")
        self._store("v2")
        current = self.db.knowledge_history("proj", "topic")[0]
        self.db.delete_knowledge(current.id)
        fresh, created, superseded = self._store("v3")
        self.assertTrue(created)
        self.assertIsNone(superseded)  # no current row existed; fresh chain
        # History query still sees all rows ever held for the subject.
        facts = [k.fact for k in self.db.knowledge_history("proj", "topic")]
        self.assertEqual(facts, ["v3", "v1"])


class TestHistoryOrder(DbTestCase):
    def test_current_first_then_newest(self):
        self._store("v1")
        self._store("v2")
        self._store("v3")
        history = self.db.knowledge_history("proj", "topic")
        self.assertEqual([k.fact for k in history], ["v3", "v2", "v1"])
        self.assertIsNone(history[0].valid_until)
        self.assertTrue(all(k.valid_until is not None for k in history[1:]))
        # superseded_by pointers chain oldest -> newest.
        self.assertEqual(history[2].superseded_by, history[1].id)
        self.assertEqual(history[1].superseded_by, history[0].id)


class TestExclusionFromReads(DbTestCase):
    def setUp(self):
        super().setUp()
        self.old, _, _ = self._store("the old answer", embedding=_emb(0.1))
        self.new, _, _ = self._store("the new answer", embedding=_emb(0.2))

    def test_fts_search_excludes_superseded(self):
        results = self.db.search_fts("answer", table="knowledge")
        self.assertEqual([r.id for r in results], [self.new.id])

    def test_listings_and_counts_exclude_superseded(self):
        self.assertEqual([k.id for k in self.db.list_knowledge("proj")], [self.new.id])
        items, total = self.db.list_page("knowledge", "proj", 10, 0)
        self.assertEqual(total, 1)
        self.assertEqual(self.db.namespace_counts()["proj"]["knowledge"], 1)

    def test_history_only_namespace_hidden(self):
        current = self.db.knowledge_history("proj", "topic")[0]
        self.db.delete_knowledge(current.id)  # only the superseded row remains
        self.assertNotIn("proj", self.db.list_namespaces())

    def test_find_by_key_sees_current_only(self):
        self.assertEqual(self.db.find_by_key("knowledge", "proj", "topic"), self.new.id)


class TestRenameNamespace(DbTestCase):
    def test_history_moves_and_current_collision_resolves(self):
        self._store("old-src", namespace="src")
        self._store("new-src", namespace="src")  # src now has history + current
        target_current, _, _ = self._store("tgt", namespace="tgt")

        counts, replaced = self.db.rename_namespace("src", "tgt")
        self.assertEqual(replaced["knowledge"], 1)  # tgt's current lost to src's
        history = self.db.knowledge_history("tgt", "topic")
        self.assertEqual(history[0].fact, "new-src")
        self.assertIn("old-src", [k.fact for k in history])  # history moved along

    def test_history_only_collision_spares_target_current(self):
        self._store("src-old", namespace="src")
        self._store("src-new", namespace="src")
        chain_current = self.db.knowledge_history("src", "topic")[0]
        self.db.delete_knowledge(chain_current.id)  # src keeps history only
        target_current, _, _ = self._store("tgt-current", namespace="tgt")

        _, replaced = self.db.rename_namespace("src", "tgt")
        self.assertEqual(replaced["knowledge"], 0)
        self.assertEqual(self.db.get_knowledge(target_current.id).fact, "tgt-current")


class ToolTestCase(DbTestCase):
    def setUp(self):
        super().setUp()
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_embedder", return_value=None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()


class TestUpdateTool(ToolTestCase):
    def test_fact_change_supersedes(self):
        old, _, _ = self._store("v1")
        result = server.update_knowledge(id=old.id, fact="v2")
        self.assertEqual(result["superseded"], old.id)
        self.assertNotEqual(result["id"], old.id)
        self.assertIsNotNone(self.db.get_knowledge(old.id).valid_until)

    def test_non_fact_fields_update_in_place(self):
        old, _, _ = self._store("v1")
        result = server.update_knowledge(id=old.id, confidence=0.4, tags=["a"])
        self.assertEqual(result["id"], old.id)
        self.assertNotIn("superseded", result)
        self.assertIsNone(self.db.get_knowledge(old.id).valid_until)

    def test_unchanged_fact_does_not_supersede(self):
        old, _, _ = self._store("v1")
        result = server.update_knowledge(id=old.id, fact="v1", confidence=0.4)
        self.assertEqual(result["id"], old.id)
        self.assertNotIn("superseded", result)

    def test_superseded_entry_is_immutable(self):
        old, _, _ = self._store("v1")
        self._store("v2")
        result = server.update_knowledge(id=old.id, confidence=0.1)
        self.assertIn("error", result)
        self.assertIn("superseded", result["error"])

    def test_subject_conflict_on_supersede(self):
        self._store("other fact", subject="taken")
        old, _, _ = self._store("v1", subject="mine")
        result = server.update_knowledge(id=old.id, subject="taken", fact="v2")
        self.assertIn("error", result)
        # The original chain is untouched by the failed supersede.
        self.assertIsNone(self.db.get_knowledge(old.id).valid_until)


class TestFactHistoryTool(ToolTestCase):
    def test_returns_timeline_and_records_access(self):
        self._store("v1")
        self._store("v2")
        resp = server.fact_history(namespace="proj", subject="topic")
        self.assertEqual(resp["count"], 2)
        self.assertEqual([e["fact"] for e in resp["history"]], ["v2", "v1"])
        self.assertIsNone(resp["history"][0]["valid_until"])
        self.assertIsNotNone(resp["history"][1]["valid_until"])
        for entry in resp["history"]:
            self.assertEqual(self.db.get_knowledge(entry["id"]).retrieval_count, 1)

    def test_unknown_subject_is_empty(self):
        resp = server.fact_history(namespace="proj", subject="nope")
        self.assertEqual(resp["count"], 0)


class TestRestoreInterplay(ToolTestCase):
    def test_restore_deleted_current_fact(self):
        k, _, _ = self._store("precious")
        self.db.delete_knowledge(k.id)
        rev_id = self.db.list_revisions(item_id=k.id)[0]["id"]
        result = server.restore(rev_id)
        self.assertTrue(result["recreated"])
        self.assertEqual(self.db.get_knowledge(k.id).fact, "precious")

    def test_restore_of_superseded_history_row_is_refused(self):
        old, _, _ = self._store("v1")
        self._store("v2")
        current = self.db.knowledge_history("proj", "topic")[0]
        self.db.delete_knowledge(old.id)  # prune the history row itself
        rev_id = self.db.list_revisions(item_id=old.id)[0]["id"]
        result = server.restore(rev_id)
        self.assertIn("error", result)
        # Current fact untouched.
        self.assertEqual(self.db.get_knowledge(current.id).fact, "v2")


if __name__ == "__main__":
    unittest.main()
