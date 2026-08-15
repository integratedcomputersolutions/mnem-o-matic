"""Tests for the write audit log.

Covers the v4 migration, append/list with filters, event capture across
every write tool (store/update/supersede/delete/tag/restore/namespace ops),
request-identity fields via RequestMetaMiddleware, the failure guarantee
(a broken audit write never breaks the operation), and read-only tools
staying silent.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from mnemomatic.audit import RequestMetaMiddleware, request_meta
from mnemomatic.db import Database
from mnemomatic import runtime
from mnemomatic import tools_admin
from mnemomatic import tools_content
from mnemomatic import tools_history
from mnemomatic import tools_search


class TestMigrationV4(unittest.TestCase):
    def test_v3_database_gains_audit_table(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            db = Database(tmp.name)
            conn = db._get_conn()
            conn.execute("DROP TABLE audit_log")
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
            db.close()

            migrated = Database(tmp.name)
            conn = migrated._get_conn()
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"], 4)
            conn.execute("SELECT * FROM audit_log")  # table exists
            migrated.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class TestDbAudit(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_retention_prunes_old_events_on_append(self):
        import mnemomatic.db as db_module
        conn = self.db._get_conn()
        conn.execute(
            "INSERT INTO audit_log (ts, op) VALUES ('2020-01-01T00:00:00+00:00', 'store')")
        conn.commit()
        with patch.object(db_module, "AUDIT_KEEP_DAYS", 365):
            self.db.append_audit("delete", item_id="x")
        ops = [e["op"] for e in self.db.list_audit()]
        self.assertEqual(ops, ["delete"])  # the 2020 event aged out

    def test_retention_zero_keeps_forever(self):
        import mnemomatic.db as db_module
        conn = self.db._get_conn()
        conn.execute(
            "INSERT INTO audit_log (ts, op) VALUES ('2020-01-01T00:00:00+00:00', 'store')")
        conn.commit()
        with patch.object(db_module, "AUDIT_KEEP_DAYS", 0):
            self.db.append_audit("delete", item_id="x")
        self.assertEqual(len(self.db.list_audit()), 2)

    def test_append_and_list_with_filters(self):
        self.db.append_audit("store", item_type="note", item_id="n1", namespace="a",
                             title="t", actor="matt", client="ua", ip="1.2.3.4",
                             detail={"created": True})
        self.db.append_audit("delete", item_type="note", item_id="n1", namespace="a")
        self.db.append_audit("store", item_type="document", item_id="d1", namespace="b")

        all_events = self.db.list_audit()
        self.assertEqual([e["op"] for e in all_events], ["store", "delete", "store"])  # newest first
        self.assertEqual(all_events[2]["actor"], "matt")
        self.assertEqual(all_events[2]["detail"], {"created": True})
        self.assertIsNone(all_events[1]["detail"])

        self.assertEqual(len(self.db.list_audit(item_id="n1")), 2)
        self.assertEqual(len(self.db.list_audit(namespace="b")), 1)
        self.assertEqual(len(self.db.list_audit(op="delete")), 1)
        self.assertEqual(len(self.db.list_audit(item_type="document")), 1)
        with self.assertRaises(ValueError):
            self.db.list_audit(item_type="bogus")


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_embedder", return_value=None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.db.close()

    def _ops(self, **filters):
        return [e["op"] for e in self.db.list_audit(**filters)]


class TestToolCoverage(ToolTestCase):
    def test_full_write_lifecycle_is_audited(self):
        note = tools_content.store_note(namespace="proj", title="n", content="v1")
        tools_content.update_note(id=note["id"], content="v2")
        tools_content.tag(item_id=note["id"], item_type="note", add_tags=["x"])
        tools_content.delete_note(id=note["id"])
        rev = self.db.list_revisions(item_id=note["id"])[0]
        tools_history.restore(revision_id=rev["id"])

        events = self.db.list_audit(item_id=note["id"])
        self.assertEqual([e["op"] for e in events],
                         ["restore", "delete", "tag", "update", "store"])
        delete_event = events[1]
        self.assertEqual(delete_event["namespace"], "proj")  # captured pre-delete
        self.assertEqual(delete_event["title"], "n")
        self.assertEqual(events[0]["detail"]["recreated"], True)
        self.assertEqual(events[3]["detail"]["fields"], ["content"])

    def test_knowledge_supersede_paths(self):
        r1 = tools_content.store_knowledge(namespace="proj", subject="s", fact="v1")
        r2 = tools_content.store_knowledge(namespace="proj", subject="s", fact="v2")
        tools_content.update_knowledge(id=r2["id"], fact="v3")

        ops = self.db.list_audit(namespace="proj")
        self.assertEqual([e["op"] for e in ops], ["supersede", "store", "store"])
        self.assertEqual(ops[1]["detail"]["superseded"], r1["id"])
        self.assertEqual(ops[0]["detail"]["superseded"], r2["id"])

    def test_namespace_ops_audited(self):
        tools_content.store_note(namespace="src", title="n", content="x")
        tools_admin.rename_namespace(old_namespace="src", new_namespace="dst")
        tools_admin.delete_namespace(namespace="dst")
        self.assertEqual(self._ops(op="rename_namespace"), ["rename_namespace"])
        event = self.db.list_audit(op="delete_namespace")[0]
        self.assertEqual(event["namespace"], "dst")
        self.assertEqual(event["detail"]["deleted"], 1)

    def test_failed_writes_are_not_audited(self):
        tools_content.delete_note(id="no-such-id")
        tools_content.update_note(id="no-such-id", content="x")
        self.assertEqual(self.db.list_audit(), [])

    def test_reads_are_not_audited(self):
        note = tools_content.store_note(namespace="proj", title="n", content="x")
        tools_search.read("note", note["id"])
        tools_search.search("x", mode="fulltext")
        tools_search.list_items("note", "proj")
        self.assertEqual(self._ops(), ["store"])

    def test_broken_audit_never_breaks_the_operation(self):
        with patch.object(self.db, "append_audit", side_effect=RuntimeError("disk full")):
            result = tools_content.store_note(namespace="proj", title="n", content="x")
        self.assertIn("id", result)
        self.assertIsNotNone(self.db.get_note(result["id"]))

    def test_list_audit_tool_shape(self):
        tools_content.store_note(namespace="proj", title="n", content="x")
        resp = tools_history.list_audit(namespace="proj")
        self.assertEqual(len(resp["events"]), 1)
        self.assertEqual(resp["events"][0]["op"], "store")
        self.assertIn("error", tools_history.list_audit(item_type="bogus"))


class TestRequestMeta(unittest.TestCase):
    def test_middleware_captures_and_resets(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        async def echo(request):
            return JSONResponse(request_meta())

        app = RequestMetaMiddleware(Starlette(routes=[Route("/", echo)]))
        client = TestClient(app)

        meta = client.get("/", headers={"X-Mnemomatic-Actor": "matt-laptop",
                                        "User-Agent": "test-agent/1.0"}).json()
        self.assertEqual(meta["actor"], "matt-laptop")
        self.assertEqual(meta["client"], "test-agent/1.0")
        self.assertTrue(meta["ip"])

        # Without the header the actor is absent, and nothing leaks between requests.
        meta = client.get("/", headers={"User-Agent": "other/2.0"}).json()
        self.assertIsNone(meta["actor"])
        self.assertEqual(meta["client"], "other/2.0")

    def test_defaults_outside_a_request(self):
        self.assertEqual(request_meta(), {"actor": None, "client": None, "ip": None})


if __name__ == "__main__":
    unittest.main()
