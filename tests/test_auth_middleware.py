"""Tests for AuthMiddleware (mnemomatic.auth).

Driven end to end through a Starlette TestClient: which paths take which
credential, the exact refusals, the brute-force lockout on tokens, the
forced-password gate on sessions, and what the app sees in request.state.
"""

import unittest

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic.auth import COOKIE_NAME, AuthMiddleware, classify
from tests._support import IdentityFixture


async def _ok(request):
    return PlainTextResponse("ok")


async def _whoami(request):
    p = request.state.principal
    if p is None:
        return JSONResponse({"user": None})
    return JSONResponse({"user": p.user.username, "role": p.user.role, "via": p.via,
                         "token_id": p.token_id, "token_hint": p.token_hint})


def _app():
    return Starlette(routes=[
        Route("/health", _ok),
        Route("/mcp", _whoami, methods=["GET", "POST"]),
        Route("/mcp/sub", _ok),
        Route("/export", _whoami),
        Route("/api/session", _whoami),
        Route("/api/login", _ok, methods=["POST"]),
        Route("/api/things", _whoami, methods=["GET", "POST"]),
        Route("/api/password", _ok, methods=["POST"]),
        Route("/api/logout", _ok, methods=["POST"]),
        Route("/", _whoami),
        Route("/assets/app.js", _ok),
        Route("/browse/anything", _whoami),
    ])


class TestClassify(unittest.TestCase):
    def test_classes(self):
        cases = {
            ("GET", "/health"): "public", ("GET", "/api/instance-id"): "public",
            ("POST", "/api/login"): "public", ("POST", "/api/first-run"): "public",
            ("GET", "/ca.crt"): "public", ("GET", "/setup"): "public",
            ("GET", "/api/session"): "public", ("POST", "/api/session"): "session",
            ("POST", "/mcp"): "bearer", ("GET", "/mcp/"): "bearer", ("GET", "/mcp/x"): "bearer",
            ("GET", "/mcpx"): "public",
            ("GET", "/export"): "any",
            ("GET", "/api"): "session", ("GET", "/api/things"): "session", ("DELETE", "/api/me/tokens/1"): "session",
            ("GET", "/"): "public", ("GET", "/assets/x.js"): "public", ("GET", "/browse/ns"): "public",
        }
        for (method, path), expected in cases.items():
            with self.subTest(path=path, method=method):
                self.assertEqual(classify(method, path), expected)


class AuthCase(unittest.TestCase):
    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)
        self.client = TestClient(AuthMiddleware(_app(), identity=lambda: self.fx.identity))

    def bearer(self, token):
        return {"Authorization": f"Bearer {token}"}

    def cookie(self, user):
        return {COOKIE_NAME: self.fx.session_for(user)}


class TestPublic(AuthCase):
    def test_health_and_spa_need_nothing(self):
        for path in ("/health", "/", "/assets/app.js", "/browse/anything"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_public_paths_ignore_bad_credentials(self):
        self.assertEqual(self.client.get("/health", headers=self.bearer("mnm_garbage")).status_code, 200)
        self.assertEqual(self.client.get("/", cookies={COOKIE_NAME: "stale"}).status_code, 200)

    def test_spa_sees_no_principal(self):
        self.assertEqual(self.client.get("/").json(), {"user": None})

    def test_session_probe_is_public_but_knows_the_user(self):
        self.assertEqual(self.client.get("/api/session").json(), {"user": None})
        body = self.client.get("/api/session", cookies=self.cookie(self.fx.user)).json()
        self.assertEqual((body["user"], body["via"]), ("alice", "session"))


class TestBearer(AuthCase):
    def test_valid_token(self):
        body = self.client.post("/mcp", headers=self.bearer(self.fx.admin_token)).json()
        self.assertEqual((body["user"], body["role"], body["via"]), ("admin", "admin", "token"))
        self.assertEqual(body["token_hint"], self.fx.admin_token[:10])
        self.assertIsInstance(body["token_id"], int)

    def test_scheme_case_and_whitespace(self):
        self.assertEqual(self.client.get("/mcp", headers={"Authorization": f"bearer {self.fx.user_token}"}).status_code, 200)
        self.assertEqual(self.client.get("/mcp", headers={"Authorization": f"Bearer  {self.fx.user_token} "}).status_code, 200)

    def test_missing_header_401(self):
        resp = self.client.post("/mcp")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "missing_authorization")
        self.assertEqual(resp.headers["cache-control"], "no-store")

    def test_wrong_scheme_401(self):
        resp = self.client.get("/mcp", headers={"Authorization": f"Basic {self.fx.user_token}"})
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "invalid_authorization")

    def test_unknown_revoked_expired_403(self):
        self.assertEqual(self.client.get("/mcp", headers=self.bearer("mnm_nope")).status_code, 403)
        self.assertEqual(self.client.get("/mcp", headers=self.bearer("sk-other-format")).status_code, 403)
        rec, raw = self.fx.identity.create_token(self.fx.user.id, "temp")
        self.fx.identity.revoke_token(self.fx.user.id, rec["id"])
        resp = self.client.get("/mcp", headers=self.bearer(raw))
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "invalid_token")

    def test_cookie_is_not_accepted_on_mcp(self):
        resp = self.client.get("/mcp", cookies=self.cookie(self.fx.admin))
        self.assertEqual(resp.status_code, 401)

    def test_subpaths_covered(self):
        self.assertEqual(self.client.get("/mcp/sub").status_code, 401)
        self.assertEqual(self.client.get("/mcp/sub", headers=self.bearer(self.fx.user_token)).status_code, 200)


class TestThrottling(AuthCase):
    def test_lockout_after_repeated_bad_tokens(self):
        for _ in range(5):
            self.assertEqual(self.client.get("/mcp", headers=self.bearer("mnm_wrong")).status_code, 403)
        resp = self.client.get("/mcp", headers=self.bearer(self.fx.admin_token))
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.json()["error"], "throttled")
        self.assertIn("Retry-After", resp.headers)

    def test_success_clears_failures(self):
        for _ in range(4):
            self.client.get("/mcp", headers=self.bearer("mnm_wrong"))
        self.assertEqual(self.client.get("/mcp", headers=self.bearer(self.fx.admin_token)).status_code, 200)
        self.assertEqual(self.client.get("/mcp", headers=self.bearer("mnm_wrong")).status_code, 403)
        self.assertEqual(self.client.get("/mcp", headers=self.bearer(self.fx.admin_token)).status_code, 200)

    def test_missing_header_does_not_count(self):
        for _ in range(10):
            self.assertEqual(self.client.get("/mcp").status_code, 401)
        self.assertEqual(self.client.get("/mcp", headers=self.bearer(self.fx.admin_token)).status_code, 200)


class TestSession(AuthCase):
    def test_cookie_admits_api(self):
        body = self.client.get("/api/things", cookies=self.cookie(self.fx.user)).json()
        self.assertEqual((body["user"], body["via"], body["token_id"]), ("alice", "session", None))

    def test_no_or_bad_cookie_401(self):
        resp = self.client.get("/api/things")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "unauthenticated")
        self.assertEqual(self.client.get("/api/things", cookies={COOKIE_NAME: "nope"}).status_code, 401)
        self.assertEqual(self.client.get("/api/things", cookies={COOKIE_NAME: "a=b; c"}).status_code, 401)

    def test_bearer_is_not_accepted_on_api(self):
        self.assertEqual(self.client.get("/api/things", headers=self.bearer(self.fx.admin_token)).status_code, 401)

    def test_logged_out_session_is_refused(self):
        raw = self.fx.session_for(self.fx.user)
        self.assertEqual(self.client.get("/api/things", cookies={COOKIE_NAME: raw}).status_code, 200)
        self.fx.identity.delete_session(raw)
        self.assertEqual(self.client.get("/api/things", cookies={COOKIE_NAME: raw}).status_code, 401)

    def test_forced_password_change_gate(self):
        user, temp = self.fx.identity.create_user("newbie")
        cookies = {COOKIE_NAME: self.fx.session_for(user)}
        resp = self.client.get("/api/things", cookies=cookies)
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "password_change_required")
        self.assertEqual(self.client.post("/api/password", cookies=cookies).status_code, 200)
        self.assertEqual(self.client.post("/api/logout", cookies=cookies).status_code, 200)
        self.assertEqual(self.client.get("/api/session", cookies=cookies).status_code, 200)
        self.fx.identity.change_password(user.id, temp, "a proper password now",
                                         keep_session=cookies[COOKIE_NAME])
        self.assertEqual(self.client.get("/api/things", cookies=cookies).status_code, 200)


class TestExportTakesEither(AuthCase):
    def test_token(self):
        body = self.client.get("/export", headers=self.bearer(self.fx.user_token)).json()
        self.assertEqual((body["user"], body["via"]), ("alice", "token"))

    def test_cookie(self):
        body = self.client.get("/export", cookies=self.cookie(self.fx.admin)).json()
        self.assertEqual((body["user"], body["via"]), ("admin", "session"))

    def test_neither(self):
        resp = self.client.get("/export")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "unauthenticated")

    def test_bad_token_beats_good_cookie(self):
        # A presented Authorization header is judged on its own merits.
        resp = self.client.get("/export", headers=self.bearer("mnm_bad"), cookies=self.cookie(self.fx.admin))
        self.assertEqual(resp.status_code, 403)


class TestNonHttp(unittest.TestCase):
    def test_lifespan_passes_through(self):
        import asyncio
        seen = {}

        async def inner(scope, receive, send):
            seen["type"] = scope["type"]

        mw = AuthMiddleware(inner, identity=lambda: None)
        asyncio.run(mw({"type": "lifespan"}, None, None))
        self.assertEqual(seen["type"], "lifespan")


if __name__ == "__main__":
    unittest.main()
