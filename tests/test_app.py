"""The whole server stack, built in-process by server.build_app: the MCP
endpoint behind token auth, the browser API behind a session, the plain
routes, and the SPA — wired the way main() wires them.

Until this test the only thing exercising the assembled stack was the
integration job against a Docker image. This covers the same path without
Docker: sign in, mint a token, call an MCP tool with it, and read the audit
row it left — with the token's name on it."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from mnemomatic import runtime, server
from mnemomatic.auth import COOKIE_NAME
from tests._support import IdentityFixture

HTML = "<!doctype html><html><head><title>Mnem-O-matic</title></head><body><div id=app></div></body></html>"
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


class TestAssembledStack(unittest.TestCase):
    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app_dir = Path(tmp.name)
        (app_dir / "assets").mkdir()
        (app_dir / "index.html").write_text(HTML)
        for target, value in (("_db", self.fx.db), ("_embedder", None)):
            p = patch.object(runtime, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        # The MCP session manager runs as the app's lifespan and can start only
        # once per instance. FastMCP builds it lazily, so dropping the cached
        # one gives each test a fresh manager (and a fresh lifespan).
        runtime.mcp._session_manager = None
        self.client = TestClient(server.build_app(None, app_dir=app_dir), base_url="http://testserver")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    ORIGIN = {"Origin": "http://testserver"}

    def login(self):
        resp = self.client.post("/api/login", json={"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD},
                                headers=self.ORIGIN)
        self.assertEqual(resp.status_code, 200)

    def test_public_surface(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        page = self.client.get("/browse/anything", headers={"Accept": "text/html"})
        self.assertEqual(page.status_code, 200)
        self.assertIn("<div id=app>", page.text)
        self.assertIn("script-src 'self'", page.headers["content-security-policy"])
        self.assertEqual(self.client.get("/api/session").json()["authenticated"], False)
        self.assertEqual(self.client.get("/ca.crt").status_code, 404)        # TLS off in this test
        self.assertIn("not set up", self.client.get("/setup").text)
        self.assertEqual(self.client.get("/api/namespaces").status_code, 401)
        self.assertEqual(self.client.post("/mcp", json=INIT, headers=MCP_HEADERS).status_code, 401)

    def test_person_signs_in_mints_token_agent_writes_audit_names_both(self):
        self.login()
        self.assertTrue(self.client.get("/api/session").json()["authenticated"])
        self.assertEqual(self.client.get("/api/namespaces").json()["namespaces"], [])

        minted = self.client.post("/api/me/tokens", json={"name": "laptop"}, headers=self.ORIGIN).json()
        bearer = {**MCP_HEADERS, "Authorization": f"Bearer {minted['token']}", "X-Mnemomatic-Actor": "claude-code"}

        resp = self.client.post("/mcp", json=INIT, headers=bearer)
        self.assertEqual(resp.status_code, 200, resp.text)
        session_id = resp.headers["mcp-session-id"]
        bearer["mcp-session-id"] = session_id
        self.client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=bearer)
        resp = self.client.post("/mcp", headers=bearer, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "store_note", "arguments": {"namespace": "proj", "title": "n", "content": "hello"}},
        })
        self.assertEqual(resp.status_code, 200, resp.text)
        body = json.loads(resp.json()["result"]["content"][0]["text"])
        self.assertTrue(body["created"])

        # The browser sees what the agent stored, and who stored it.
        self.assertEqual(self.client.get("/api/namespaces").json()["namespaces"],
                         [{"name": "proj", "documents": 0, "knowledge": 0, "notes": 1}])
        event = self.client.get("/api/audit?op=store").json()["events"][0]
        self.assertEqual(event["actor"], "admin")
        self.assertEqual(event["detail"]["token"]["name"], "laptop")
        self.assertEqual(event["detail"]["token"]["id"], minted["id"])
        self.assertEqual(event["detail"]["label"], "claude-code")
        self.assertEqual(event["namespace"], "proj")

        # Revoke the token: the agent is out, the person is still in.
        self.assertEqual(self.client.delete(f"/api/me/tokens/{minted['id']}", headers=self.ORIGIN).status_code, 204)
        self.assertEqual(self.client.post("/mcp", json=INIT, headers=bearer).status_code, 403)
        self.assertTrue(self.client.get("/api/session").json()["authenticated"])

    def test_export_takes_either_credential_and_is_audited(self):
        self.assertEqual(self.client.get("/export").status_code, 401)
        self.login()
        resp = self.client.get("/export")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/zip")
        resp = self.client.get("/export", headers={"Authorization": f"Bearer {self.fx.user_token}"},
                               cookies={COOKIE_NAME: "ignored"})
        self.assertEqual(resp.status_code, 200)
        actors = [e["actor"] for e in self.client.get("/api/audit?op=export").json()["events"]]
        self.assertEqual(actors, ["alice", "admin"])

    def test_https_endpoints_report_off(self):
        self.login()
        self.assertEqual(self.client.get("/api/admin/https").json(), {"state": "off", "trusted_proxies": []})
        self.assertEqual(self.client.get("/api/connect").json()["builtin_ca"], False)


if __name__ == "__main__":
    unittest.main()
