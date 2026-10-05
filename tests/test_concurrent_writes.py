"""Writes racing on worker threads must serialize, not fail or wedge SQLite.

MCP tools and the identity calls behind the web API run on pooled worker
threads, each with its own SQLite connection. Every write goes through
Database.write(): BEGIN IMMEDIATE takes the write lock before a
read-then-write decides anything, and a failure rolls back instead of
leaving the transaction (and the lock) open on the thread.
"""

import logging
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

from mnemomatic import runtime, tools_content
from mnemomatic.db import Database
from mnemomatic.identity import MAX_ACTIVE_TOKENS, Identity, IdentityError
from mnemomatic.models import Knowledge, Note
from tests._support import fast_scrypt, temp_db_path


def _race(n, fn):
    """Run fn(i) on n threads released together; returns (results, errors)."""
    barrier = threading.Barrier(n)
    results, errors = [None] * n, []

    def go(i):
        barrier.wait()
        try:
            results[i] = fn(i)
        except Exception as e:          # collected, asserted on by the caller
            errors.append(e)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


class _FileDb(unittest.TestCase):
    def setUp(self):
        self.path = temp_db_path(self)
        self.db = Database(str(self.path))
        self.addCleanup(self.db.close)

    def count(self, sql, *params):
        return self.db.connection().execute(sql, params).fetchone()["n"]


class TestContentWrites(_FileDb):
    def setUp(self):
        super().setUp()
        for p in (patch.object(runtime, "_db", return_value=self.db),
                  patch.object(runtime, "_embedder", return_value=None)):
            p.start()
            self.addCleanup(p.stop)

    def test_concurrent_stores_of_one_title_upsert(self):
        # Before: one insert won, one raised UNIQUE and kept its transaction
        # open, and every other writer hit "database is locked" after 5 s.
        results, errors = _race(8, lambda i: tools_content.store_note(namespace="ns", title="same",
                                                                       content=f"v{i}"))
        self.assertEqual(errors, [])
        self.assertEqual(sum(1 for r in results if r["created"]), 1)
        self.assertEqual(self.count("SELECT COUNT(*) AS n FROM notes"), 1)
        self.assertEqual(self.count("SELECT COUNT(*) AS n FROM revisions"), 7)   # one per update

    def test_concurrent_fact_changes_form_one_history(self):
        results, errors = _race(6, lambda i: self.db.store_knowledge(
            Knowledge(namespace="ns", subject="s", fact=f"fact {i}"), embedding=None))
        self.assertEqual(errors, [])
        self.assertEqual(self.count("SELECT COUNT(*) AS n FROM knowledge WHERE valid_until IS NULL"), 1)
        self.assertEqual(self.count("SELECT COUNT(*) AS n FROM knowledge"), 6)

    def test_failed_write_releases_the_lock(self):
        a, _, _ = self.db.store_knowledge(Knowledge(namespace="ns", subject="a", fact="1"), embedding=None)
        self.db.store_knowledge(Knowledge(namespace="ns", subject="b", fact="2"), embedding=None)
        failed = []

        def collide():
            # The successor's subject collides with current fact "b".
            try:
                self.db.supersede_knowledge(a.id, Knowledge(namespace="ns", subject="b", fact="3"), None)
            except sqlite3.IntegrityError:
                failed.append(self.db.connection().in_transaction)

        t = threading.Thread(target=collide)
        t.start()
        t.join()
        self.assertEqual(failed, [False])                      # rolled back, not left open
        start = time.monotonic()
        self.db.store_note(Note(namespace="ns", title="after", content="x"), embedding=None)
        self.assertLess(time.monotonic() - start, 1.0)
        # And the rollback was whole: fact "a" was not left closed.
        self.assertIsNone(self.db.get_knowledge(a.id).valid_until)

    def test_nested_write_commits_once_with_the_outer(self):
        with self.db.write() as conn:
            conn.execute("INSERT INTO settings (key, value) VALUES ('outer', '1')")
            self.db.set_setting("inner", "1")                   # joins, does not commit early
            self.assertTrue(conn.in_transaction)
        self.assertEqual(self.db.get_setting("inner"), "1")
        with self.assertRaises(RuntimeError):
            with self.db.write():
                self.db.set_setting("doomed", "1")
                raise RuntimeError("boom")
        self.assertIsNone(self.db.get_setting("doomed"))

    def test_failed_nested_write_is_undone_on_its_own(self):
        # A caller that catches a nested failure and carries on must not
        # commit half of what the nested write did.
        with self.db.write():
            self.db.set_setting("outer", "1")
            try:
                with self.db.write() as conn:
                    conn.execute("INSERT INTO settings (key, value) VALUES ('half', '1')")
                    raise RuntimeError("nested failure")
            except RuntimeError:
                pass
        self.assertEqual(self.db.get_setting("outer"), "1")
        self.assertIsNone(self.db.get_setting("half"))

    def test_export_reads_one_snapshot(self):
        # Before: a rename committing between two namespaces' reads listed
        # the moved note under both.
        from mnemomatic.export import build_export_zip
        import io
        import zipfile
        for ns in ("a", "b"):
            self.db.store_note(Note(namespace=ns, title=f"in {ns}", content="x"), embedding=None)
        real, fired = self.db.list_items, []

        def list_items(item_type, ns):
            out = real(item_type, ns)
            if ns == "a" and item_type == "note" and not fired:
                fired.append(True)
                t = threading.Thread(target=self.db.rename_namespace, args=("a", "b"))
                t.start()
                t.join()
            return out

        with patch.object(self.db, "list_items", list_items):
            data, _ = build_export_zip(self.db, None, server_version="t")
        notes = sorted(n for n in zipfile.ZipFile(io.BytesIO(data)).namelist() if n.endswith(".md"))
        self.assertEqual(fired, [True])                                  # the rename did land mid-export
        self.assertEqual(notes, ["a/notes/in_a.md", "b/notes/in_b.md"])  # the export is as of its start
        self.assertEqual(sorted(n.title for n in real("note", "b")), ["in a", "in b"])  # and the store moved on

    def test_snapshot_is_read_only(self):
        with self.db.snapshot():
            with self.assertRaises(RuntimeError):
                self.db.set_setting("k", "v")
        self.db.set_setting("k", "v")                                      # fine afterwards
        self.assertFalse(self.db.connection().in_transaction)

    def test_offload_wrapper_discards_a_leaked_transaction(self):
        import asyncio

        def leaky() -> dict:
            self.db.connection().execute("INSERT INTO settings (key, value) VALUES ('leak', '1')")
            return {}

        with patch.object(runtime, "db", self.db), self.assertLogs("mnemomatic", logging.ERROR) as logs:
            asyncio.run(runtime._offloaded(leaky)())
        self.assertIn("left a database transaction open", logs.output[0])
        self.assertIsNone(self.db.get_setting("leak"))


class TestIdentityWrites(_FileDb):
    def setUp(self):
        super().setUp()
        p = fast_scrypt()
        p.start()
        self.addCleanup(p.stop)
        self.ident = Identity(self.db)

    def test_token_limit_holds_under_a_burst(self):
        user, _ = self.ident.create_user("alice", password="alicepassword1")
        results, errors = _race(MAX_ACTIVE_TOKENS + 5, lambda i: self.ident.create_token(user.id, f"t{i}"))
        self.assertEqual(sum(1 for r in results if r), MAX_ACTIVE_TOKENS)
        self.assertTrue(errors and all(isinstance(e, IdentityError) and e.code == "token_limit" for e in errors))

    def test_two_admins_cannot_demote_each_other_to_none(self):
        a, _ = self.ident.create_user("anna", role="admin", password="annapassword12")
        b, _ = self.ident.create_user("bert", role="admin", password="bertpassword12")
        _race(2, lambda i: self.ident.set_role((b, a)[i].id, "user", acting_user_id=(a, b)[i].id))
        admins = self.count("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1")
        self.assertEqual(admins, 1)

    def test_only_if_no_users_lets_exactly_one_through(self):
        results, errors = _race(6, lambda i: self.ident.create_user(
            f"first{i}", role="admin", password="firstpassword1", only_if_no_users=True))
        self.assertEqual(sum(1 for r in results if r), 1)
        self.assertEqual({e.code for e in errors}, {"already_set_up"})
        self.assertEqual(self.count("SELECT COUNT(*) AS n FROM users"), 1)

    def test_duplicate_username_race_leaves_no_open_transaction(self):
        results, errors = _race(4, lambda i: self.ident.create_user("same", password="samepassword12"))
        self.assertEqual(sum(1 for r in results if r), 1)
        self.assertEqual({e.code for e in errors}, {"user_exists"})
        start = time.monotonic()
        self.ident.create_user("other", password="otherpassword1")
        self.assertLess(time.monotonic() - start, 1.0)


if __name__ == "__main__":
    unittest.main()
