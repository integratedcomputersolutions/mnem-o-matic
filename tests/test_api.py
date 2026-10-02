"""Tests for the browser JSON API (mnemomatic.api), driven through the same
middleware stack the server uses: SecurityHeaders → RequestMeta → Auth → routes."""

import asyncio
import unittest
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.testclient import TestClient

from mnemomatic import config, runtime
from mnemomatic.api import INSTANCE_ID, SecurityHeadersMiddleware, build_api_routes
from mnemomatic.audit import RequestMetaMiddleware
from mnemomatic.auth import COOKIE_NAME, AuthMiddleware
from mnemomatic.identity import FirstRun
from mnemomatic.models import Document, Note
from tests._support import IdentityFixture

SETTINGS = {"version": "3.0.0-test", "mode": "FTS-only (no embedder)", "model": None}


class ApiCase(unittest.TestCase):
    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)
        self.first_run = FirstRun()
        self.pending = {"origin": None, "hsts": False}
        mount = build_api_routes(identity=lambda: self.fx.identity, db_getter=lambda: self.fx.db,
                                 settings_info=lambda: dict(SETTINGS), first_run=self.first_run,
                                 https=None)
        app = Starlette(routes=[mount])
        app = RequestMetaMiddleware(app)
        app = AuthMiddleware(app, identity=lambda: self.fx.identity)
        app = SecurityHeadersMiddleware(app, pending_origin=lambda: self.pending["origin"],
                                        hsts=lambda: self.pending["hsts"])
        self.client = TestClient(app, base_url="http://testserver")
        # The tool modules reach the database through runtime._db.
        self._patches = [patch.object(runtime, "_db", return_value=self.fx.db),
                         patch.object(runtime, "_embedder", return_value=None)]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    ORIGIN = {"Origin": "http://testserver"}

    def admin(self):
        return {COOKIE_NAME: self.fx.session_for(self.fx.admin)}

    def user(self):
        return {COOKIE_NAME: self.fx.session_for(self.fx.user)}

    def post(self, path, json=None, cookies=None, origin=True, **kw):
        headers = dict(self.ORIGIN) if origin else {}
        headers.update(kw.pop("headers", {}))
        return self.client.post(path, json=json, cookies=cookies, headers=headers, **kw)

    def delete(self, path, cookies=None, origin=True):
        return self.client.delete(path, cookies=cookies, headers=dict(self.ORIGIN) if origin else {})

    def events(self, op):
        return self.fx.db.list_audit(op=op)


class TestConventions(ApiCase):
    def test_no_store_and_security_headers(self):
        resp = self.client.get("/api/session")
        self.assertEqual(resp.headers["cache-control"], "no-store")
        csp = resp.headers["content-security-policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("script-src 'self' 'unsafe-inline'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("connect-src 'self';", csp)
        self.assertEqual(resp.headers["x-content-type-options"], "nosniff")
        self.assertEqual(resp.headers["referrer-policy"], "no-referrer")
        self.assertNotIn("strict-transport-security", resp.headers)

    def test_pending_https_origin_widens_connect_src(self):
        self.pending["origin"] = "https://memory.example:8443"
        csp = self.client.get("/api/session").headers["content-security-policy"]
        self.assertIn("connect-src 'self' https://memory.example:8443;", csp)

    def test_hsts_only_when_enabled_and_only_over_https(self):
        https = TestClient(self.client.app, base_url="https://testserver")
        # Pending or non-443: never, even over HTTPS — HSTS binds to the host,
        # not the port, and would redirect browsers at the plain listener.
        self.assertNotIn("strict-transport-security", https.get("/api/session").headers)
        self.pending["hsts"] = True
        self.assertEqual(https.get("/api/session").headers["strict-transport-security"], "max-age=31536000")
        self.assertNotIn("strict-transport-security", self.client.get("/api/session").headers)

    def test_origin_guard(self):
        body = {"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD}
        resp = self.post("/api/login", body, origin=False)
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "origin_mismatch")
        resp = self.client.post("/api/login", json=body, headers={"Origin": "http://evil.example"})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(self.post("/api/login", body).status_code, 200)

    def test_forwarded_host_counts_only_behind_a_trusted_proxy(self):
        body = {"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD}
        headers = {"Origin": "https://memory.example", "X-Forwarded-Host": "memory.example"}
        self.assertEqual(self.client.post("/api/login", json=body, headers=headers).status_code, 403)
        with patch.object(config, "TRUSTED_PROXIES", ["*"]):
            self.assertEqual(self.client.post("/api/login", json=body, headers=headers).status_code, 200)

    def test_body_validation(self):
        cookies = self.admin()
        resp = self.client.post("/api/me/tokens", content=b"not json", cookies=cookies,
                                headers={**self.ORIGIN, "Content-Type": "application/json"})
        self.assertEqual(resp.json()["error"], "invalid_json")
        resp = self.post("/api/me/tokens", json=[1, 2], cookies=cookies)
        self.assertEqual(resp.json()["error"], "invalid_json")
        resp = self.post("/api/me/tokens", json={}, cookies=cookies)
        self.assertEqual(resp.json()["error"], "missing_field")

    def test_body_cap(self):
        with patch.object(config, "API_MAX_BODY", 64):
            resp = self.post("/api/me/tokens", json={"name": "x" * 100}, cookies=self.admin())
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.json()["error"], "payload_too_large")

    def test_admin_only(self):
        self.assertEqual(self.client.get("/api/admin/users", cookies=self.user()).status_code, 403)
        self.assertEqual(self.client.get("/api/admin/users", cookies=self.user()).json()["error"], "forbidden")
        self.assertEqual(self.client.get("/api/admin/users", cookies=self.admin()).status_code, 200)

    def test_unauthenticated(self):
        self.assertEqual(self.client.get("/api/me/tokens").status_code, 401)


class TestSessionAndLogin(ApiCase):
    def test_session_anonymous(self):
        body = self.client.get("/api/session").json()
        self.assertEqual(body["authenticated"], False)
        self.assertEqual(body["first_run"], False)
        self.assertEqual(body["version"], "3.0.0-test")
        self.assertEqual(body["https"]["state"], "off")

    def test_login_logout_round_trip(self):
        resp = self.post("/api/login", {"username": "Admin", "password": IdentityFixture.ADMIN_PASSWORD})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["user"]["username"], "admin")
        cookie = resp.headers["set-cookie"]
        self.assertIn(f"{COOKIE_NAME}=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertNotIn("Secure", cookie)                  # plain http in this test
        self.assertIn("Path=/", cookie)
        body = self.client.get("/api/session").json()
        self.assertTrue(body["authenticated"])
        self.assertEqual(body["user"]["role"], "admin")
        self.assertEqual(self.events("auth.login")[0]["actor"], "admin")

        resp = self.post("/api/logout")
        self.assertEqual(resp.status_code, 204)
        self.assertIn(f'{COOKIE_NAME}=""', resp.headers["set-cookie"])
        self.assertFalse(self.client.get("/api/session").json()["authenticated"])
        self.assertEqual(self.events("auth.logout")[0]["actor"], "admin")

    def test_secure_cookie_over_https(self):
        https = TestClient(self.client.app, base_url="https://testserver")
        resp = https.post("/api/login", json={"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD},
                          headers={"Origin": "https://testserver"})
        self.assertIn("Secure", resp.headers["set-cookie"])

    def test_bad_credentials(self):
        resp = self.post("/api/login", {"username": "admin", "password": "nope"})
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "invalid_credentials")
        self.assertNotIn("set-cookie", resp.headers)
        resp = self.post("/api/login", {"username": "ghost", "password": "nope"})
        self.assertEqual(resp.status_code, 401)
        failed = self.events("auth.login_failed")
        self.assertEqual(len(failed), 2)
        self.assertIsNone(failed[0]["actor"])
        self.assertEqual(failed[0]["item_id"], "ghost")
        self.assertEqual(failed[0]["detail"]["reason"], "invalid_credentials")

    def test_login_throttle(self):
        for _ in range(5):
            self.post("/api/login", {"username": "admin", "password": "nope"})
        resp = self.post("/api/login", {"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD})
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Retry-After", resp.headers)
        # The refused retry is not audited; only the five real attempts are.
        failed = self.events("auth.login_failed")
        self.assertEqual(len(failed), 5)
        self.assertEqual({e["detail"]["reason"] for e in failed}, {"invalid_credentials"})
        # Another account from the same address is still fine (per-ip limit is higher).
        resp = self.post("/api/login", {"username": "alice", "password": IdentityFixture.USER_PASSWORD})
        self.assertEqual(resp.status_code, 200)

    def burst(self, path, body, n, cookies=None):
        """Send `n` identical POSTs concurrently; returns the status codes."""
        async def go():
            transport = httpx.ASGITransport(app=self.client.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver",
                                         headers=self.ORIGIN, cookies=cookies) as c:
                resps = await asyncio.gather(*(c.post(path, json=body) for _ in range(n)))
            return [r.status_code for r in resps]
        return asyncio.run(go())

    def test_login_throttle_holds_under_concurrency(self):
        codes = self.burst("/api/login", {"username": "admin", "password": "nope"}, 60)
        self.assertEqual(codes.count(401), 5)
        self.assertEqual(codes.count(429), 55)
        # The burst's failures locked the address out, right password or not.
        codes = self.burst("/api/login", {"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD}, 1)
        self.assertEqual(codes, [429])

    def test_login_sets_device_cookie_for_login_only(self):
        resp = self.post("/api/login", {"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD})
        self.assertEqual(resp.status_code, 200)
        device = [c for c in resp.headers.get_list("set-cookie") if c.startswith("mnm_device=")]
        self.assertEqual(len(device), 1)
        self.assertIn("Path=/api/login", device[0])
        self.assertIn("HttpOnly", device[0])
        proof = resp.cookies["mnm_device"]
        self.assertTrue(self.fx.identity.is_known_device("admin", proof))

    def test_impossible_username_is_not_audited(self):
        junk = "Correct Horse Battery Staple " * 100
        resp = self.post("/api/login", {"username": junk, "password": "nope"})
        self.assertEqual(resp.status_code, 401)
        failed = self.events("auth.login_failed")
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0]["item_id"])

    def test_password_change_throttle(self):
        cookies = self.user()
        for _ in range(5):
            resp = self.post("/api/password", {"current_password": "wrong-password",
                                               "new_password": "a-brand-new-password"}, cookies=cookies)
            self.assertEqual(resp.status_code, 401)
        resp = self.post("/api/password", {"current_password": IdentityFixture.USER_PASSWORD,
                                           "new_password": "a-brand-new-password"}, cookies=cookies)
        self.assertEqual(resp.status_code, 429)

    def test_password_change_throttle_holds_under_concurrency(self):
        codes = self.burst("/api/password", {"current_password": "wrong-password",
                                             "new_password": "a-brand-new-password"}, 30, cookies=self.user())
        self.assertEqual(codes.count(401), 5)
        self.assertEqual(codes.count(429), 25)

    def test_disabled_account(self):
        self.fx.identity.set_active(self.fx.user.id, False, acting_user_id=self.fx.admin.id)
        resp = self.post("/api/login", {"username": "alice", "password": IdentityFixture.USER_PASSWORD})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "account_disabled")

    def test_missing_fields(self):
        resp = self.post("/api/login", {"username": "admin"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "missing_field")

    def test_password_change(self):
        cookies = self.user()
        resp = self.post("/api/password", {"current_password": "wrong", "new_password": "a new long password"},
                         cookies=cookies)
        self.assertEqual(resp.status_code, 401)
        resp = self.post("/api/password", {"current_password": IdentityFixture.USER_PASSWORD,
                                           "new_password": "a new long password"}, cookies=cookies)
        self.assertEqual(resp.status_code, 204)
        # The session doing the change survives.
        self.assertTrue(self.client.get("/api/session", cookies=cookies).json()["authenticated"])
        self.assertEqual(self.events("password.changed")[0]["item_id"], "alice")


class TestFirstRun(ApiCase):
    def _empty(self):
        conn = self.fx.db.connection()
        conn.execute("DELETE FROM users")
        conn.commit()

    def test_flow(self):
        self._empty()
        code = self.first_run.issue()
        self.assertTrue(self.client.get("/api/session").json()["first_run"])
        resp = self.post("/api/first-run", {"setup_code": "XXXX-XXXX-XXXX", "username": "matt",
                                            "password": "a long enough password"})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "bad_setup_code")
        resp = self.post("/api/first-run", {"setup_code": code.lower(), "username": "matt",
                                            "password": "a long enough password"})
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["user"]["role"], "admin")
        self.assertIn(f"{COOKIE_NAME}=", resp.headers["set-cookie"])
        self.assertIsNone(self.first_run.code)
        created = self.events("admin.created")[0]
        self.assertEqual((created["actor"], created["item_id"], created["detail"]["source"]),
                         ("matt", "matt", "setup_code"))
        self.assertTrue(self.client.get("/api/session").json()["authenticated"])

    def test_refused_once_users_exist(self):
        code = self.first_run.issue()
        resp = self.post("/api/first-run", {"setup_code": code, "username": "x", "password": "a long password 1"})
        self.assertEqual(resp.status_code, 409)

    def test_weak_password(self):
        self._empty()
        code = self.first_run.issue()
        resp = self.post("/api/first-run", {"setup_code": code, "username": "matt", "password": "short"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(self.fx.identity.count_users(), 0)
        self.assertIsNotNone(self.first_run.code)


class TestTokens(ApiCase):
    def test_lifecycle(self):
        cookies = self.user()
        resp = self.post("/api/me/tokens", {"name": "laptop", "expires_in_days": 30}, cookies=cookies)
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertTrue(body["token"].startswith("mnm_"))
        self.assertEqual(body["hint"], body["token"][:10])
        self.assertIsNotNone(body["expires_at"])
        listing = self.client.get("/api/me/tokens", cookies=cookies).json()["tokens"]
        self.assertEqual(listing[0]["name"], "laptop")
        self.assertNotIn("token", listing[0])
        self.assertEqual(self.events("token.created")[0]["detail"]["hint"], body["hint"])

        resp = self.delete(f"/api/me/tokens/{body['id']}", cookies=cookies)
        self.assertEqual(resp.status_code, 204)
        self.assertIsNotNone(self.client.get("/api/me/tokens", cookies=cookies).json()["tokens"][0]["revoked_at"])
        self.assertEqual(self.events("token.revoked")[0]["item_id"], str(body["id"]))

    def test_cannot_revoke_someone_elses(self):
        rec, _ = self.fx.identity.create_token(self.fx.admin.id, "admins")
        self.assertEqual(self.delete(f"/api/me/tokens/{rec['id']}", cookies=self.user()).status_code, 404)

    def test_validation_passes_through(self):
        resp = self.post("/api/me/tokens", {"name": "x", "expires_in_days": 99999}, cookies=self.user())
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_expiry")


class TestUsersAdmin(ApiCase):
    def test_create_and_manage(self):
        cookies = self.admin()
        resp = self.post("/api/admin/users", {"username": "Bob", "display_name": "Bob B"}, cookies=cookies)
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["user"]["username"], "bob")
        self.assertEqual(body["user"]["role"], "user")
        self.assertEqual(len(body["temporary_password"]), 16)
        uid = body["user"]["id"]
        self.assertEqual(self.events("user.created")[0]["detail"]["role"], "user")

        users = self.client.get("/api/admin/users", cookies=cookies).json()["users"]
        self.assertEqual([u["username"] for u in users], ["admin", "alice", "bob"])
        self.assertIn("token_count", users[0])

        resp = self.post(f"/api/admin/users/{uid}/role", {"role": "admin"}, cookies=cookies)
        self.assertEqual(resp.json()["user"]["role"], "admin")
        self.assertEqual(self.events("user.role_changed")[0]["detail"], {"from": "user", "to": "admin"})

        resp = self.post(f"/api/admin/users/{uid}/active", {"active": False}, cookies=cookies)
        self.assertFalse(resp.json()["user"]["active"])
        self.assertEqual(self.events("user.deactivated")[0]["item_id"], "bob")
        resp = self.post(f"/api/admin/users/{uid}/active", {"active": "yes"}, cookies=cookies)
        self.assertEqual(resp.status_code, 400)

        resp = self.post(f"/api/admin/users/{uid}/reset-password", cookies=cookies)
        self.assertEqual(len(resp.json()["temporary_password"]), 16)
        self.assertEqual(self.events("password.reset")[0]["item_id"], "bob")

        self.assertEqual(self.delete(f"/api/admin/users/{uid}", cookies=cookies).status_code, 204)
        self.assertEqual(self.events("user.deleted")[0]["item_id"], "bob")
        self.assertEqual(self.delete(f"/api/admin/users/{uid}", cookies=cookies).status_code, 404)

    def test_guards_surface_as_codes(self):
        cookies = self.admin()
        me = self.fx.admin.id
        self.assertEqual(self.delete(f"/api/admin/users/{me}", cookies=cookies).json()["error"], "self_action")
        self.assertEqual(self.post(f"/api/admin/users/{me}/role", {"role": "user"}, cookies=cookies).status_code, 403)
        # Promote alice, then demote admin → alice is the only admin → demoting her must fail.
        self.post(f"/api/admin/users/{self.fx.user.id}/role", {"role": "admin"}, cookies=cookies)
        alice = {COOKIE_NAME: self.fx.session_for(self.fx.user)}
        self.post(f"/api/admin/users/{me}/role", {"role": "user"}, cookies=alice)
        resp = self.post(f"/api/admin/users/{self.fx.user.id}/role", {"role": "user"}, cookies=cookies)
        self.assertEqual(resp.status_code, 403)        # admin is now a plain user
        resp = self.post(f"/api/admin/users/{self.fx.user.id}/active", {"active": False}, cookies=alice)
        self.assertEqual(resp.json()["error"], "self_action")
        resp = self.post("/api/admin/users", {"username": "alice"}, cookies=alice)
        self.assertEqual(resp.status_code, 409)


class TestStoreViews(ApiCase):
    def setUp(self):
        super().setUp()
        self.doc, _ = self.fx.db.store_document(
            Document(namespace="proj", title="Design notes", content="The cache uses an LRU policy.",
                     tags=["design"], metadata={"owner": "team"}), embedding=None)
        self.fx.db.store_note(Note(namespace="proj", title="todo", content="write tests"), embedding=None)
        self.fx.db.store_note(Note(namespace="other", title="misc", content="unrelated"), embedding=None)
        self.fx.db.update_document(self.doc.id, content="The cache uses an LRU policy, 256 entries.")

    def test_namespaces(self):
        body = self.client.get("/api/namespaces", cookies=self.user()).json()
        self.assertEqual(body["namespaces"], [
            {"name": "other", "documents": 0, "knowledge": 0, "notes": 1},
            {"name": "proj", "documents": 1, "knowledge": 0, "notes": 1},
        ])

    def test_items_page(self):
        body = self.client.get("/api/items?namespace=proj&type=document", cookies=self.user()).json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["items"][0]["title"], "Design notes")
        self.assertNotIn("content", body["items"][0])
        self.assertEqual(body["items"][0]["tags"], ["design"])
        body = self.client.get("/api/items?namespace=proj&type=note&limit=1&offset=5", cookies=self.user()).json()
        self.assertEqual((body["total"], body["items"], body["limit"], body["offset"]), (1, [], 1, 5))
        self.assertEqual(self.client.get("/api/items?type=note", cookies=self.user()).status_code, 400)
        self.assertEqual(self.client.get("/api/items?namespace=proj&type=bogus", cookies=self.user()).json()["error"],
                         "invalid_type")
        self.assertEqual(self.client.get("/api/items?namespace=proj&type=note&limit=abc",
                                         cookies=self.user()).json()["error"], "invalid_parameter")

    def test_item_detail_and_revisions(self):
        body = self.client.get(f"/api/items/document/{self.doc.id}", cookies=self.user()).json()
        self.assertEqual(body["type"], "document")
        self.assertEqual(body["item"]["metadata"], {"owner": "team"})
        self.assertIn("256 entries", body["item"]["content"])
        self.assertIsInstance(body["item"]["created_at"], str)
        revs = self.client.get(f"/api/items/document/{self.doc.id}/revisions", cookies=self.user()).json()
        self.assertEqual(len(revs["revisions"]), 1)
        self.assertEqual(revs["revisions"][0]["op"], "update")
        self.assertEqual(self.client.get("/api/items/document/nope", cookies=self.user()).status_code, 404)
        self.assertEqual(self.client.get("/api/items/widget/x", cookies=self.user()).status_code, 400)

    def test_related_degrades_without_embedder(self):
        body = self.client.get(f"/api/items/document/{self.doc.id}/related", cookies=self.user()).json()
        self.assertEqual(body["related"], [])
        self.assertIn("unavailable", body)

    def test_search(self):
        body = self.client.get("/api/search?q=cache&mode=fulltext", cookies=self.user()).json()
        self.assertEqual([r["id"] for r in body["results"]], [self.doc.id])
        self.assertFalse(body["degraded"])
        body = self.client.get("/api/search?q=cache", cookies=self.user()).json()   # hybrid, no embedder
        self.assertTrue(body["degraded"])
        self.assertEqual([r["id"] for r in body["results"]], [self.doc.id])
        body = self.client.get("/api/search?q=unrelated&namespace=proj&mode=fulltext", cookies=self.user()).json()
        self.assertEqual(body["results"], [])
        self.assertEqual(self.client.get("/api/search?q=", cookies=self.user()).status_code, 400)
        self.assertEqual(self.client.get("/api/search?q=x&mode=magic", cookies=self.user()).json()["error"],
                         "invalid_mode")
        resp = self.client.get("/api/search?q=x&mode=semantic", cookies=self.user())
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "search_failed")

    def test_search_does_not_record_access(self):
        # The viewer browsing is not an agent retrieving: retrieval_count is
        # bumped by the MCP surfaces only (Database.record_access).
        body = self.client.get("/api/search?q=cache&mode=fulltext", cookies=self.user()).json()
        self.assertEqual([r["id"] for r in body["results"]], [self.doc.id])
        self.assertEqual(self.fx.db.get_document(self.doc.id).retrieval_count, 0)

    def test_audit_listing(self):
        self.post("/api/login", {"username": "admin", "password": IdentityFixture.ADMIN_PASSWORD})
        body = self.client.get("/api/audit?op=auth.login", cookies=self.admin()).json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["events"][0]["actor"], "admin")
        self.post("/api/logout", cookies=self.admin())
        body = self.client.get("/api/audit?limit=1", cookies=self.admin()).json()
        self.assertEqual(len(body["events"]), 1)
        self.assertEqual(body["events"][0]["op"], "auth.logout")
        self.assertGreaterEqual(body["total"], 2)
        body = self.client.get("/api/audit?limit=1&offset=1", cookies=self.admin()).json()
        self.assertEqual(body["events"][0]["op"], "auth.login")
        body = self.client.get("/api/audit?actor=nobody", cookies=self.admin()).json()
        self.assertEqual(body["total"], 0)
        self.assertEqual(self.client.get("/api/audit?item_type=bogus", cookies=self.user()).json()["error"],
                         "invalid_filter")

    def test_audit_hides_identity_events_from_non_admins(self):
        self.post("/api/login", {"username": "ghost", "password": "nope"})
        self.fx.db.append_audit("store", item_type="note", item_id="n1", namespace="ns", actor="alice")
        body = self.client.get("/api/audit", cookies=self.user()).json()
        self.assertEqual([e["op"] for e in body["events"]], ["store"])
        self.assertEqual(body["total"], 1)
        body = self.client.get("/api/audit?op=auth.login_failed", cookies=self.user()).json()
        self.assertEqual(body["total"], 0)
        body = self.client.get("/api/audit?op=auth.login_failed", cookies=self.admin()).json()
        self.assertEqual(body["total"], 1)


class TestConnectSettingsHttps(ApiCase):
    def test_connect(self):
        body = self.client.get("/api/connect", cookies=self.user()).json()
        self.assertEqual(body["origin"], "http://testserver")
        self.assertEqual(body["mcp_url"], "http://testserver/mcp")
        self.assertEqual(body["compact_url"], "http://testserver/mcp?compact=true")
        self.assertFalse(body["builtin_ca"])
        self.assertIsNone(body["ca_url"])
        self.assertEqual(body["token_prefix"], "mnm_")

    def test_settings(self):
        body = self.client.get("/api/settings", cookies=self.user()).json()
        self.assertEqual(body["version"], "3.0.0-test")
        self.assertEqual(body["tls"]["state"], "off")
        self.assertIn("audit_keep_days", body)
        self.assertIsNone(body["backup"])

    def test_instance_id_is_public_and_cors_open(self):
        resp = self.client.get("/api/instance-id")
        self.assertEqual(resp.json(), {"instance_id": INSTANCE_ID})
        self.assertEqual(resp.headers["access-control-allow-origin"], "*")
        self.assertEqual(len(INSTANCE_ID), 32)

    def test_https_endpoints_without_builtin_tls(self):
        cookies = self.admin()
        self.assertEqual(self.client.get("/api/admin/https", cookies=cookies).json(), {"state": "off", "trusted_proxies": []})
        with patch.object(config, "TRUSTED_PROXIES", ["*"]):
            self.assertEqual(self.client.get("/api/admin/https", cookies=cookies).json()["trusted_proxies"], ["*"])
        resp = self.post("/api/admin/https/name", {"name": "memory.example"}, cookies=cookies)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "tls_disabled")
        self.assertEqual(self.client.get("/api/admin/https", cookies=self.user()).status_code, 403)


if __name__ == "__main__":
    unittest.main()
