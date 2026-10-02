"""Opening a v4 (2.x) database under 3.0 must add the identity tables, leave
every existing row alone, stamp version 5, and say so in the audit log."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mnemomatic.db import SCHEMA_VERSION, Database


def _build_v4_database(path: Path) -> None:
    """A v4 file made the way 2.x made it: open with the real Database, then
    strip the v5 tables and wind the version back. Hand-writing the whole 2.x
    DDL would drift from the one `_init_schema` actually produces."""
    db = Database(str(path))
    conn = db.connection()
    conn.executescript("""
        DROP TABLE sessions; DROP TABLE api_tokens; DROP TABLE users; DROP TABLE settings;
        DELETE FROM audit_log;
    """)
    conn.execute(
        "INSERT INTO audit_log (ts, op, item_type, item_id, namespace, title, actor, client, ip, detail) "
        "VALUES ('2026-08-01T00:00:00+00:00', 'store', 'note', 'n1', 'proj', 'first', "
        "'self-declared', 'claude-code', '10.0.0.5', ?)", (json.dumps({"created": True}),),
    )
    conn.execute(
        "INSERT INTO notes (id, namespace, title, content, source, tags, metadata, created_at, updated_at) "
        "VALUES ('n1', 'proj', 'first', 'body', 'text', '[]', '{}', "
        "'2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00')"
    )
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    db.close()


class TestMigrationV5(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.path = Path(tmp.name)
        _build_v4_database(self.path)

    def tearDown(self):
        for p in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            p.unlink(missing_ok=True)

    def _tables(self, conn) -> set[str]:
        return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    def test_fixture_really_is_v4(self):
        conn = sqlite3.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 4)
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertNotIn("users", names)
        conn.close()

    def test_upgrade(self):
        db = Database(str(self.path))
        conn = db.connection()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"], SCHEMA_VERSION)
        self.assertTrue({"users", "sessions", "api_tokens", "settings"} <= self._tables(conn))

        # Content and the pre-existing audit row are untouched.
        note = conn.execute("SELECT * FROM notes WHERE id = 'n1'").fetchone()
        self.assertEqual(note["title"], "first")
        old = db.list_audit(op="store")
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0]["actor"], "self-declared")
        self.assertEqual(old[0]["detail"], {"created": True})

        # The migration announced itself once.
        events = db.list_audit(op="schema.migrated")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["actor"], "system")
        self.assertEqual(events[0]["item_type"], "schema")
        self.assertEqual(events[0]["detail"], {"from": 4, "to": SCHEMA_VERSION})

        # No users yet: that is the first-run state the server bootstraps from.
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"], 0)
        db.close()

    def test_upgrade_adds_credential_version(self):
        db = Database(str(self.path))
        cols = {r["name"] for r in db.connection().execute("PRAGMA table_info(users)")}
        self.assertIn("credential_version", cols)
        db.close()

    def test_v5_database_gains_credential_version(self):
        # A 3.0 database: users exist, the column does not.
        db = Database(str(self.path))
        conn = db.connection()
        conn.execute("ALTER TABLE users DROP COLUMN credential_version")
        conn.execute("INSERT INTO users (username, role, password_hash, created_at) "
                     "VALUES ('alice', 'user', 'x', '2026-10-01T00:00:00+00:00')")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
        db.close()
        db = Database(str(self.path))
        row = db.connection().execute("SELECT credential_version FROM users WHERE username = 'alice'").fetchone()
        self.assertEqual(row["credential_version"], 0)
        self.assertEqual(db.list_audit(op="schema.migrated")[0]["detail"], {"from": 5, "to": SCHEMA_VERSION})
        db.close()

    def test_reopen_is_a_no_op(self):
        Database(str(self.path)).close()
        db = Database(str(self.path))
        self.assertEqual(len(db.list_audit(op="schema.migrated")), 1)
        self.assertEqual(db.count_audit(), 2)
        db.close()

    def test_fresh_database_records_no_migration(self):
        fresh = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        fresh.close()
        try:
            db = Database(fresh.name)
            self.assertEqual(db.list_audit(op="schema.migrated"), [])
            self.assertTrue({"users", "sessions", "api_tokens", "settings"} <= self._tables(db.connection()))
            db.close()
        finally:
            for p in (Path(fresh.name), Path(fresh.name + "-wal"), Path(fresh.name + "-shm")):
                p.unlink(missing_ok=True)

    def test_settings_helpers(self):
        db = Database(str(self.path))
        self.assertIsNone(db.get_setting("https"))
        db.set_setting("https", '{"name": "memory.example"}')
        self.assertEqual(db.get_setting("https"), '{"name": "memory.example"}')
        db.set_setting("https", None)
        self.assertIsNone(db.get_setting("https"))
        db.close()


if __name__ == "__main__":
    unittest.main()
