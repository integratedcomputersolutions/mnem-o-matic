"""Trusted-proxy sign-in (MNEMOMATIC_PROXY_SECRET).

A gateway that signs people in itself sends its secret as the bearer token
and names the user in a header; the user is created on first sight and the
audit log records the proxy's identity for them. Driven through the real
AuthMiddleware and RequestMetaMiddleware stack, plus the identity-store
rules on their own and the schema migration that adds the column.
"""

import importlib
import os
import sqlite3
import unittest
from unittest.mock import patch

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic import config, server
from mnemomatic.audit import RequestMetaMiddleware, request_meta
from mnemomatic.auth import AuthMiddleware, ProxyAuth
from mnemomatic.db import SCHEMA_VERSION, Database
from mnemomatic.identity import (
    NO_PASSWORD_HASH,
    Identity,
    IdentityError,
    normalize_external_id,
    username_from_external_id,
)
from tests._support import IdentityFixture, fast_scrypt, temp_db_path

SECRET = "s" * 40
USER_HEADER = "X-Mnemomatic-User"


async def _whoami(request):
    p = request.state.principal
    meta = request_meta()
    return JSONResponse({"user": p.user.username, "role": p.user.role, "via": p.via,
                         "external_id": p.user.external_id, "display_name": p.user.display_name,
                         "user_id": p.user.id, "actor": meta["user"], "is_admin": meta["is_admin"]})


def _app(proxy: ProxyAuth | None, identity):
    inner = Starlette(routes=[
        Route("/mcp", _whoami, methods=["GET", "POST"]),
        Route("/export", _whoami),
    ])
    return AuthMiddleware(RequestMetaMiddleware(inner), identity=lambda: identity, proxy=proxy)


class ProxyCase(unittest.TestCase):
    name_header: str | None = "X-Mnemomatic-Name"
    role_header: str | None = "X-Mnemomatic-Role"

    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)
        self.proxy = ProxyAuth(secret=SECRET, user_header=USER_HEADER,
                               name_header=self.name_header, role_header=self.role_header)
        self.client = TestClient(_app(self.proxy, self.fx.identity))

    def call(self, user=None, secret=SECRET, path="/mcp", **extra):
        headers = {}
        if secret is not None:
            headers["Authorization"] = f"Bearer {secret}"
        if user is not None:
            headers[USER_HEADER] = user
        headers.update(extra)
        return self.client.get(path, headers=headers)

    def users(self):
        return {u["username"]: u for u in self.fx.identity.list_users()}


class TestSignIn(ProxyCase):
    def test_secret_and_user_header_create_the_user_on_first_sight(self):
        before = set(self.users())
        r = self.call("Bob.Smith@Example.com")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["via"], "proxy")
        self.assertEqual(body["user"], "bob.smith")
        self.assertEqual(body["role"], "user")
        self.assertEqual(body["external_id"], "bob.smith@example.com")
        self.assertEqual(body["display_name"], "bob.smith@example.com")
        # The audit actor is the proxy's identity, so history lines up with 2.x logs.
        self.assertEqual(body["actor"], "bob.smith@example.com")
        self.assertFalse(body["is_admin"])
        self.assertEqual(set(self.users()) - before, {"bob.smith"})
        created = [e for e in self.fx.db.list_audit() if e["op"] == "user.created"]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["actor"], "bob.smith@example.com")
        self.assertEqual(created[0]["detail"]["source"], "proxy")

    def test_returning_user_matches_on_identity_whatever_the_case(self):
        first = self.call("bob@example.com").json()
        again = self.call("BOB@example.com").json()
        self.assertEqual(first["user_id"], again["user_id"])
        self.assertEqual(len([u for u in self.users() if u.startswith("bob")]), 1)

    def test_username_collision_gets_a_suffix(self):
        # "alice" exists already (the fixture's local user).
        r = self.call("alice@example.com")
        self.assertEqual(r.json()["user"], "alice-2")
        self.assertEqual(self.call("alice@other.example").json()["user"], "alice-3")
        # The local alice is untouched and still her own person.
        self.assertIsNone(self.users()["alice"]["external_id"])

    def test_export_takes_the_proxy_too(self):
        r = self.call("bob@example.com", path="/export")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["via"], "proxy")

    def test_personal_tokens_still_work_and_ignore_the_user_header(self):
        r = self.call("mallory@example.com", secret=self.fx.user_token)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["user"], "alice")
        self.assertEqual(r.json()["via"], "token")
        self.assertNotIn("mallory", "".join(self.users()))

    def test_name_and_role_headers(self):
        r = self.call("bob@example.com", **{"X-Mnemomatic-Name": "Bob Smith", "X-Mnemomatic-Role": "admin"})
        self.assertEqual(r.json()["display_name"], "Bob Smith")
        self.assertEqual(r.json()["role"], "admin")
        self.assertTrue(r.json()["is_admin"])
        # Absent on a later request: the row keeps what it has.
        self.assertEqual(self.call("bob@example.com").json()["role"], "admin")
        # Present and different: the proxy's word is the current one.
        r = self.call("bob@example.com", **{"X-Mnemomatic-Name": "Robert", "X-Mnemomatic-Role": "user"})
        self.assertEqual(r.json()["display_name"], "Robert")
        self.assertEqual(r.json()["role"], "user")

    def test_utf8_display_name(self):
        r = self.call("jose@example.com", **{"X-Mnemomatic-Name": "José Núñez".encode("utf-8")})
        self.assertEqual(r.json()["display_name"], "José Núñez")

    def test_bad_role_is_refused(self):
        r = self.call("bob@example.com", **{"X-Mnemomatic-Role": "root"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"], "invalid_role")
        self.assertNotIn("bob", self.users())

    def test_proxy_cannot_demote_the_only_admin(self):
        r = self.call("boss@example.com", **{"X-Mnemomatic-Role": "admin"})
        self.assertEqual(r.json()["role"], "admin")
        boss_id = r.json()["user_id"]
        self.fx.identity.set_active(self.fx.admin.id, False, acting_user_id=boss_id)
        with self.assertLogs("mnemomatic", "WARNING"):
            r = self.call("boss@example.com", **{"X-Mnemomatic-Role": "user"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["role"], "admin")

    def test_deactivated_proxy_user_is_refused(self):
        uid = self.call("bob@example.com").json()["user_id"]
        self.fx.identity.set_active(uid, False, acting_user_id=self.fx.admin.id)
        r = self.call("bob@example.com")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error"], "account_disabled")


class TestRefusals(ProxyCase):
    def test_secret_without_user_header_is_401_and_leaves_nothing_behind(self):
        before = set(self.users())
        for value in (None, "", "   "):
            r = self.call(value)
            self.assertEqual(r.status_code, 401, value)
            self.assertEqual(r.json()["error"], "proxy_user_missing")
        self.assertEqual(set(self.users()), before)

    def test_secret_without_user_header_does_not_count_toward_the_lockout(self):
        for _ in range(6):
            self.assertEqual(self.call(None).status_code, 401)
        self.assertEqual(self.call("bob@example.com").status_code, 200)

    def test_user_header_without_secret_is_refused(self):
        before = set(self.users())
        r = self.call("bob@example.com", secret=None)
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"], "missing_authorization")
        r = self.call("bob@example.com", secret="mnm_" + "x" * 43)
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error"], "invalid_token")
        r = self.call("bob@example.com", secret=SECRET[:-1] + "t")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(set(self.users()), before)

    def test_wrong_secret_counts_toward_the_lockout_and_a_right_one_clears_it(self):
        for _ in range(4):
            self.assertEqual(self.call("bob@example.com", secret="wrong" * 8).status_code, 403)
        self.assertEqual(self.call("bob@example.com").status_code, 200)
        for _ in range(5):
            self.assertEqual(self.call("bob@example.com", secret="wrong" * 8).status_code, 403)
        r = self.call("bob@example.com")
        self.assertEqual(r.status_code, 429)
        self.assertIn("Retry-After", r.headers)

    def test_bad_identity_is_refused(self):
        r = self.call("bob smith")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"], "invalid_external_id")
        self.assertEqual(self.call("x" * 255).status_code, 400)

    def test_secret_never_appears_in_a_refusal_or_log(self):
        with self.assertLogs("mnemomatic", "DEBUG") as logs:
            r = self.call(None)
            self.call("bob@example.com", secret=SECRET[:-1] + "t")
        self.assertNotIn(SECRET, r.text)
        self.assertNotIn(SECRET, "\n".join(logs.output))


class TestModeOff(unittest.TestCase):
    def test_without_a_secret_configured_the_headers_mean_nothing(self):
        fx = IdentityFixture()
        self.addCleanup(fx.close)
        client = TestClient(_app(None, fx.identity))
        r = client.get("/mcp", headers={"Authorization": f"Bearer {SECRET}", USER_HEADER: "bob@example.com"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error"], "invalid_token")
        self.assertEqual({u["username"] for u in fx.identity.list_users()}, {"admin", "alice"})

    def test_from_config(self):
        with patch.multiple(config, PROXY_SECRET=None):
            self.assertIsNone(ProxyAuth.from_config())
        with patch.multiple(config, PROXY_SECRET=SECRET, PROXY_USER_HEADER="X-User",
                            PROXY_NAME_HEADER=None, PROXY_ROLE_HEADER="X-Role"):
            proxy = ProxyAuth.from_config()
        self.assertEqual(proxy, ProxyAuth(secret=SECRET, user_header="X-User", role_header="X-Role"))
        self.assertTrue(proxy.matches(SECRET))
        self.assertFalse(proxy.matches(SECRET + " "))
        self.assertFalse(proxy.matches(""))


class TestStartupChecks(unittest.TestCase):
    def test_short_secret_refuses_to_start(self):
        with patch.multiple(config, PROXY_SECRET="short", TRUSTED_PROXIES=["*"]):
            with self.assertRaises(SystemExit), self.assertLogs("mnemomatic", "ERROR"):
                server._check_proxy_config()

    def test_missing_trusted_proxies_is_a_warning(self):
        with patch.multiple(config, PROXY_SECRET=SECRET, TRUSTED_PROXIES=[]):
            with self.assertLogs("mnemomatic", "WARNING") as logs:
                server._check_proxy_config()
        self.assertIn("MNEMOMATIC_TRUSTED_PROXIES", "\n".join(logs.output))
        self.assertNotIn(SECRET, "\n".join(logs.output))

    def test_off_is_silent(self):
        with patch.multiple(config, PROXY_SECRET=None):
            with self.assertNoLogs("mnemomatic"):
                server._check_proxy_config()

    def test_stale_2x_credentials_are_named_once(self):
        env = {"MNEMOMATIC_API_KEY": "old-shared-key", "MNEMOMATIC_UI_TOKEN": "old-ui-token"}
        try:
            with patch.dict(os.environ, env), self.assertLogs("mnemomatic", "WARNING") as logs:
                importlib.reload(config)
        finally:
            importlib.reload(config)
        text = "\n".join(logs.output)
        self.assertIn("MNEMOMATIC_API_KEY", text)
        self.assertIn("MNEMOMATIC_PROXY_SECRET", text)
        self.assertIn("MNEMOMATIC_UI_TOKEN", text)
        self.assertNotIn("old-shared-key", text)
        self.assertNotIn("old-ui-token", text)


class TestIdentityRules(unittest.TestCase):
    def test_username_from_external_id(self):
        cases = {
            "bob.smith@example.com": "bob.smith",
            "Bob_Smith+tag@example.com": "bob_smith-tag",
            "...@example.com": "user",
            "a@example.com": "user",
            "élodie@example.com": "lodie",
            "x" * 40 + "@example.com": "x" * 32,
            "plain-handle": "plain-handle",
        }
        for ext, expected in cases.items():
            with self.subTest(ext=ext):
                self.assertEqual(username_from_external_id(ext), expected)

    def test_normalize_external_id(self):
        self.assertEqual(normalize_external_id("  Bob@Example.COM "), "bob@example.com")
        for bad in ("", "   ", "a b", "a\tb", "a\x00b", "x" * 255, None):
            with self.subTest(bad=bad), self.assertRaises(IdentityError):
                normalize_external_id(bad)


class TestProxyUsersInTheStore(unittest.TestCase):
    def setUp(self):
        self._patch = fast_scrypt()
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.db = Database(str(temp_db_path(self)))
        self.addCleanup(self.db.close)
        self.identity = Identity(self.db)
        self.admin, _ = self.identity.create_user("admin", role="admin", password="admin-password-1")

    def test_no_password_until_an_admin_issues_one(self):
        p = self.identity.resolve_proxy_user("bob@example.com")
        row = self.db.connection().execute("SELECT password_hash FROM users WHERE id = ?", (p.user.id,)).fetchone()
        self.assertEqual(row["password_hash"], NO_PASSWORD_HASH)
        for guess in ("", "!", NO_PASSWORD_HASH, "bob@example.com"):
            with self.assertRaises(IdentityError) as cm:
                self.identity.authenticate("bob", guess)
            self.assertEqual(cm.exception.code, "invalid_credentials")
        self.assertFalse(p.user.must_change_password)
        # An admin can hand a proxy user a real password later; it works as any other.
        temp, _ = self.identity.reset_password(p.user.id, acting_user_id=self.admin.id)
        self.assertEqual(self.identity.authenticate("bob", temp).id, p.user.id)

    def test_public_view_carries_the_identity_and_never_the_hash(self):
        p = self.identity.resolve_proxy_user("bob@example.com")
        shown = p.user.public()
        self.assertEqual(shown["external_id"], "bob@example.com")
        self.assertNotIn("password_hash", shown)
        self.assertIsNone(self.admin.public()["external_id"])

    def test_actor_is_the_identity_however_they_arrive(self):
        p = self.identity.resolve_proxy_user("bob@example.com")
        self.assertEqual(p.actor, "bob@example.com")
        _, raw = self.identity.create_token(p.user.id, "agent")
        self.assertEqual(self.identity.resolve_token(raw).actor, "bob@example.com")
        self.assertEqual(self.identity.resolve_token(raw).via, "token")
        local = self.identity.resolve_session(self.identity.create_session(self.admin.id))
        self.assertEqual(local.actor, "admin")

    def test_concurrent_first_sight_resolves_to_one_user(self):
        import threading
        results, errors = [], []

        def hit():
            try:
                results.append(self.identity.resolve_proxy_user("bob@example.com").user.id)
            except Exception as e:  # pragma: no cover - a failure is the finding
                errors.append(e)

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(len([u for u in self.identity.list_users() if u["external_id"]]), 1)


class TestMigrationV7(unittest.TestCase):
    """A v6 (3.0.0) database gains the external_id column and its index."""

    def test_v6_database_migrates_forward(self):
        path = temp_db_path(self)
        db = Database(str(path))
        conn = db.connection()
        conn.executescript("""
            DROP INDEX idx_users_external_id;
            ALTER TABLE users DROP COLUMN external_id;
            DELETE FROM audit_log;
            PRAGMA user_version = 6;
        """)
        conn.commit()
        db.close()

        db = Database(str(path))
        self.addCleanup(db.close)
        conn = db.connection()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()["user_version"], SCHEMA_VERSION)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        self.assertIn("external_id", cols)
        indexes = {r["name"] for r in conn.execute("PRAGMA index_list(users)")}
        self.assertIn("idx_users_external_id", indexes)
        migrated = [e for e in db.list_audit() if e["op"] == "schema.migrated"]
        self.assertEqual(migrated[0]["detail"], {"from": 6, "to": SCHEMA_VERSION})

        # Two local users without an identity do not collide on the partial index.
        with fast_scrypt():
            identity = Identity(db)
            identity.create_user("one", password="password-one-1")
            identity.create_user("two", password="password-two-2")
            identity.resolve_proxy_user("bob@example.com")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO users (username, display_name, role, password_hash, created_at, external_id) "
                             "VALUES ('bob2', '', 'user', '!', 'now', 'bob@example.com')")


if __name__ == "__main__":
    unittest.main()
