"""Read-only tokens: a token's scope, not its owner's role, decides whether it
may change anything over MCP.

The gate keys off the tools' own annotations, so those are pinned here first:
a tool mislabelled read-only would be callable with a read token, and the
same hints tell clients which calls can run without asking.
"""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mnemomatic import runtime, server  # noqa: F401  (server registers every tool)
from mnemomatic.db import SCHEMA_VERSION, Database
from mnemomatic.identity import IdentityError, Principal
from mnemomatic_cli._mcp_client import MCPClient
from tests._support import SPA_HTML, CookieClient, IdentityFixture, temp_db_path

# name -> (readOnlyHint, destructiveHint, idempotentHint); None where the hint
# does not apply (read-only tools carry only readOnlyHint).
EXPECTED_ANNOTATIONS = {
    "search": (True, None, None),
    "read": (True, None, None),
    "list_items": (True, None, None),
    "related": (True, None, None),
    "embedding_info": (True, None, None),
    "fact_history": (True, None, None),
    "list_revisions": (True, None, None),
    "list_audit": (True, None, None),
    "consolidation_report": (True, None, None),
    "store_document": (False, False, True),
    "store_knowledge": (False, False, True),
    "store_note": (False, False, True),
    "update_document": (False, False, False),
    "update_knowledge": (False, False, False),
    "update_note": (False, False, False),
    "tag": (False, False, False),
    "rename_namespace": (False, False, False),
    "restore": (False, False, False),
    "delete_document": (False, True, True),
    "delete_knowledge": (False, True, True),
    "delete_note": (False, True, True),
    "delete_namespace": (False, True, True),
}
READ_TOOLS = {name for name, (ro, _, _) in EXPECTED_ANNOTATIONS.items() if ro}


class TestToolAnnotations(unittest.TestCase):
    def test_every_tool_carries_its_expected_hints(self):
        tools = {t.name: t for t in asyncio.run(runtime.mcp.list_tools())}
        self.assertEqual(set(tools), set(EXPECTED_ANNOTATIONS), "a tool was added or removed: pin its hints here")
        for name, (read_only, destructive, idempotent) in EXPECTED_ANNOTATIONS.items():
            with self.subTest(tool=name):
                ann = tools[name].annotations
                self.assertIsNotNone(ann)
                self.assertIs(ann.readOnlyHint, read_only)
                self.assertIs(ann.destructiveHint, destructive)
                self.assertIs(ann.idempotentHint, idempotent)
                self.assertIs(ann.openWorldHint, False)


class TestTokenScopeInIdentity(unittest.TestCase):
    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)

    def test_scope_is_stored_listed_and_resolved(self):
        rec, raw = self.fx.identity.create_token(self.fx.user.id, "reader", scope="read")
        self.assertEqual(rec["scope"], "read")
        listed = {t["name"]: t["scope"] for t in self.fx.identity.list_tokens(self.fx.user.id)}
        self.assertEqual(listed, {"reader": "read", "alice-agent": "write"})
        principal = self.fx.identity.resolve_token(raw)
        self.assertEqual(principal.token_scope, "read")
        self.assertFalse(principal.can_write)

    def test_default_scope_is_write(self):
        principal = self.fx.identity.resolve_token(self.fx.user_token)
        self.assertEqual(principal.token_scope, "write")
        self.assertTrue(principal.can_write)

    def test_unknown_scope_is_refused(self):
        for scope in ("admin", "", "READ", None):
            with self.subTest(scope=scope), self.assertRaises(IdentityError) as cm:
                self.fx.identity.create_token(self.fx.user.id, "x", scope=scope)
            self.assertEqual(cm.exception.code, "invalid_scope")

    def test_a_session_writes_whatever_the_role(self):
        self.assertTrue(Principal(user=self.fx.user, via="session").can_write)
        # A token with no recorded scope is not trusted to write.
        self.assertFalse(Principal(user=self.fx.admin, via="token", token_scope=None).can_write)


class TestMigrationV7(unittest.TestCase):
    def test_existing_tokens_become_write(self):
        path = temp_db_path(self)
        fx_db = Database(str(path))
        conn = fx_db.connection()
        conn.execute("INSERT INTO users (username, role, password_hash, created_at) "
                     "VALUES ('alice', 'user', 'x', '2026-10-01T00:00:00+00:00')")
        conn.execute("INSERT INTO api_tokens (user_id, name, token_hash, hint, created_at) "
                     "VALUES (1, 'old', 'h', 'mnm_abcdef', '2026-10-01T00:00:00+00:00')")
        conn.execute("ALTER TABLE api_tokens DROP COLUMN scope")
        conn.execute("PRAGMA user_version = 6")
        conn.commit()
        fx_db.close()

        db = Database(str(path))
        self.addCleanup(db.close)
        conn = db.connection()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"], SCHEMA_VERSION)
        self.assertEqual(conn.execute("SELECT scope FROM api_tokens").fetchone()["scope"], "write")
        self.assertEqual(db.list_audit(op="schema.migrated")[0]["detail"], {"from": 6, "to": SCHEMA_VERSION})
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("UPDATE api_tokens SET scope = 'admin'")


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
ORIGIN = {"Origin": "http://testserver"}


class TestScopeOverMcp(unittest.TestCase):
    """The assembled server: tokens minted through the API, used on /mcp."""

    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app_dir = Path(tmp.name)
        (app_dir / "assets").mkdir()
        (app_dir / "index.html").write_text(SPA_HTML)
        for target, value in (("_db", self.fx.db), ("_embedder", None)):
            p = patch.object(runtime, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        runtime.mcp._session_manager = None
        self.client = CookieClient(server.build_app(None, app_dir=app_dir), base_url="http://testserver")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def mint(self, user, password, **body) -> dict:
        self.client.cookies.clear()
        resp = self.client.post("/api/login", json={"username": user, "password": password}, headers=ORIGIN)
        self.assertEqual(resp.status_code, 200)
        resp = self.client.post("/api/me/tokens", json={"name": "agent", **body}, headers=ORIGIN)
        self.client.cookies.clear()
        return resp

    def connect(self, token: str) -> dict:
        headers = {**MCP_HEADERS, "Authorization": f"Bearer {token}"}
        resp = self.client.post("/mcp", json=INIT, headers=headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        headers["mcp-session-id"] = resp.headers["mcp-session-id"]
        self.client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers)
        return headers

    def rpc(self, headers, method, params=None) -> dict:
        resp = self.client.post("/mcp", headers=headers,
                                json={"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["result"]

    def call(self, headers, name, **arguments) -> dict:
        return self.rpc(headers, "tools/call", {"name": name, "arguments": arguments})

    def test_read_token_sees_and_calls_only_read_tools(self):
        resp = self.mint("alice", IdentityFixture.USER_PASSWORD, scope="read")
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["scope"], "read")
        reader = self.connect(resp.json()["token"])

        listed = {t["name"] for t in self.rpc(reader, "tools/list")["tools"]}
        self.assertEqual(listed, READ_TOOLS)

        result = self.call(reader, "search", query="anything")
        self.assertFalse(result.get("isError"), result)

        for name, arguments in (("store_note", {"namespace": "p", "title": "t", "content": "c"}),
                                ("delete_namespace", {"namespace": "p"}),
                                ("restore", {"revision_id": 1})):
            with self.subTest(tool=name):
                result = self.call(reader, name, **arguments)
                self.assertTrue(result["isError"])
                self.assertIn("read_only_token", result["content"][0]["text"])
        self.assertEqual(self.fx.db.list_namespaces(), [])

    def test_admin_read_token_is_read_only_too(self):
        resp = self.mint("admin", IdentityFixture.ADMIN_PASSWORD, scope="read")
        reader = self.connect(resp.json()["token"])
        self.assertEqual({t["name"] for t in self.rpc(reader, "tools/list")["tools"]}, READ_TOOLS)
        result = self.call(reader, "store_note", namespace="p", title="t", content="c")
        self.assertTrue(result["isError"])
        self.assertEqual(self.fx.db.list_namespaces(), [])

    def test_write_token_reaches_every_tool(self):
        resp = self.mint("alice", IdentityFixture.USER_PASSWORD, scope="write")
        writer = self.connect(resp.json()["token"])
        self.assertEqual({t["name"] for t in self.rpc(writer, "tools/list")["tools"]}, set(EXPECTED_ANNOTATIONS))
        result = self.call(writer, "store_note", namespace="p", title="t", content="c")
        self.assertTrue(json.loads(result["content"][0]["text"])["created"])

    def test_unknown_tool_is_reported_as_unknown(self):
        resp = self.mint("alice", IdentityFixture.USER_PASSWORD, scope="read")
        result = self.call(self.connect(resp.json()["token"]), "no_such_tool")
        self.assertTrue(result["isError"])
        self.assertNotIn("read_only_token", result["content"][0]["text"])

    def test_read_token_can_export(self):
        token = self.mint("alice", IdentityFixture.USER_PASSWORD, scope="read").json()["token"]
        resp = self.client.get("/export", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)

    def test_scope_defaults_to_write_and_is_audited(self):
        resp = self.mint("alice", IdentityFixture.USER_PASSWORD)
        self.assertEqual(resp.json()["scope"], "write")
        event = self.fx.db.list_audit(op="token.created")[0]
        self.assertEqual(event["detail"]["scope"], "write")

    def test_bad_scope_is_a_400(self):
        resp = self.mint("alice", IdentityFixture.USER_PASSWORD, scope="admin")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_scope")


class TestCliReportsRefusals(unittest.TestCase):
    def test_tool_error_becomes_the_message(self):
        client = MCPClient.__new__(MCPClient)
        refused = {"isError": True, "content": [{"type": "text", "text": "read_only_token: this token is read-only"}]}
        with patch.object(MCPClient, "_rpc", return_value=refused):
            with self.assertRaises(RuntimeError) as cm:
                client.call_tool("store_note", {})
        self.assertIn("read_only_token", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
