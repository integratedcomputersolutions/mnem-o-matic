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
from mnemomatic.db import SCHEMA_VERSION, Database
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
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"], SCHEMA_VERSION)
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

    def test_oversized_fields_are_clipped(self):
        self.db.append_audit("auth.login_failed", item_type="user", item_id="u" * 100_000,
                             client="ua" * 100_000, detail={"label": "x" * 100_000})
        event = self.db.list_audit()[0]
        self.assertLessEqual(len(event["item_id"]), 513)
        self.assertLessEqual(len(event["client"]), 513)
        self.assertTrue(event["detail"]["truncated"])

    def test_mcp_list_audit_shows_identity_events_to_admins_only(self):
        self.db.append_audit("auth.login", item_type="user", item_id="alice")
        self.db.append_audit("store", item_type="note", item_id="n1", namespace="ns")
        with patch.object(tools_history, "request_meta", return_value={"user": "bob", "is_admin": False}):
            self.assertEqual([e["op"] for e in tools_history.list_audit()["events"]], ["store"])
        with patch.object(tools_history, "request_meta", return_value={"user": "root", "is_admin": True}):
            self.assertEqual([e["op"] for e in tools_history.list_audit()["events"]], ["store", "auth.login"])

    def test_viewer_sees_ip_and_client_only_on_own_rows(self):
        self.db.append_audit("store", item_type="note", item_id="a", actor="alice", ip="10.0.0.1", client="curl/8")
        self.db.append_audit("store", item_type="note", item_id="b", actor="bob", ip="10.0.0.2", client="claude-code")
        rows = {e["actor"]: e for e in self.db.list_audit(viewer="bob")}
        self.assertEqual((rows["alice"]["ip"], rows["alice"]["client"]), (None, None))
        self.assertEqual((rows["bob"]["ip"], rows["bob"]["client"]), ("10.0.0.2", "claude-code"))
        self.assertEqual(rows["alice"]["item_id"], "a")                  # who did what stays visible
        full = {e["actor"]: e for e in self.db.list_audit()}
        self.assertEqual(full["alice"]["ip"], "10.0.0.1")

    def test_mcp_list_audit_hides_other_peoples_ip_from_non_admins(self):
        self.db.append_audit("store", item_type="note", item_id="a", actor="alice", ip="10.0.0.1", client="curl/8")
        self.db.append_audit("export", actor="carol", ip="10.0.0.3", client="firefox")
        with patch.object(tools_history, "request_meta", return_value={"user": "bob", "is_admin": False}):
            events = tools_history.list_audit()["events"]
        self.assertEqual({(e["ip"], e["client"]) for e in events}, {(None, None)})
        with patch.object(tools_history, "request_meta", return_value={"user": "root", "is_admin": True}):
            events = tools_history.list_audit()["events"]
        self.assertEqual({e["ip"] for e in events}, {"10.0.0.1", "10.0.0.3"})


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
        self.assertEqual(request_meta(), {
            "actor": None, "client": None, "ip": None,
            "user": None, "user_id": None, "is_admin": False, "via": None,
            "token_id": None, "token_hint": None, "token_name": None,
        })

    def test_principal_in_scope_is_copied(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from mnemomatic.identity import Principal, User

        user = User(id=7, username="matt", display_name="", role="admin", active=True,
                    must_change_password=False, temp_password_expires_at=None, created_at="t")

        async def echo(request):
            return JSONResponse(request_meta())

        async def inject(scope, receive, send):
            scope.setdefault("state", {})["principal"] = Principal(user=user, via="token",
                                                                   token_id=3, token_hint="mnm_abcdef",
                                                                   token_name="laptop")
            await app(scope, receive, send)

        app = RequestMetaMiddleware(Starlette(routes=[Route("/", echo)]))
        meta = TestClient(inject).get("/", headers={"X-Mnemomatic-Actor": "laptop"}).json()
        self.assertEqual(meta["user"], "matt")
        self.assertEqual(meta["user_id"], 7)
        self.assertEqual(meta["via"], "token")
        self.assertEqual(meta["token_id"], 3)
        self.assertEqual(meta["token_hint"], "mnm_abcdef")
        self.assertEqual(meta["token_name"], "laptop")
        self.assertEqual(meta["actor"], "laptop")


class TestAuditEnrichment(ToolTestCase):
    """runtime._audit turns the request meta into actor, token detail, label."""

    def _with_meta(self, **fields):
        from mnemomatic.audit import _EMPTY, _request_meta
        return _request_meta.set({**_EMPTY, **fields})

    def test_token_principal(self):
        from mnemomatic.audit import _request_meta
        tok = self._with_meta(user="matt", user_id=1, via="token", token_id=9, token_hint="mnm_xyz123",
                              token_name="laptop", client="ua/1", ip="10.1.1.1")
        try:
            tools_content.store_note(namespace="proj", title="n", content="c")
        finally:
            _request_meta.reset(tok)
        event = self.db.list_audit(op="store")[0]
        self.assertEqual(event["actor"], "matt")
        self.assertEqual(event["client"], "ua/1")
        self.assertEqual(event["detail"]["token"], {"id": 9, "hint": "mnm_xyz123", "name": "laptop"})
        self.assertNotIn("label", event["detail"])

    def test_header_label_rides_in_detail(self):
        from mnemomatic.audit import _request_meta
        tok = self._with_meta(user="matt", user_id=1, via="session", actor="laptop")
        try:
            tools_content.store_note(namespace="proj", title="n", content="c")
        finally:
            _request_meta.reset(tok)
        event = self.db.list_audit(op="store")[0]
        self.assertEqual(event["actor"], "matt")
        self.assertEqual(event["detail"]["label"], "laptop")
        self.assertNotIn("token", event["detail"])

    def test_no_principal_means_no_actor(self):
        tools_content.store_note(namespace="proj", title="n", content="c")
        event = self.db.list_audit(op="store")[0]
        self.assertIsNone(event["actor"])


if __name__ == "__main__":
    unittest.main()
