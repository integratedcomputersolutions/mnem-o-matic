"""Tests for the HTTP /health endpoint.

Liveness has to work for callers that cannot authenticate — a container
HEALTHCHECK, a load balancer, an uptime monitor — so /health needs no
credential. That exemption is the security-sensitive part: the response must
stay minimal, and the data-bearing routes beside it must stay guarded.
"""

import json
import unittest

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from mnemomatic.auth import AuthMiddleware
from mnemomatic.tools_admin import _health_route
from tests._support import IdentityFixture

_FX: IdentityFixture | None = None


def setUpModule():
    global _FX
    _FX = IdentityFixture()


def tearDownModule():
    _FX.close()


def _app():
    """The real /health route behind the real auth middleware, plus guarded
    routes to prove the exemption is scoped to /health alone."""
    async def guarded(request):
        return JSONResponse({"secret": "data"})

    app = Starlette(routes=[
        Route("/health", _health_route, methods=["GET"]),
        Route("/export", guarded, methods=["GET"]),
        Route("/api/healthy-looking", guarded, methods=["GET"]),
        Route("/mcp/healthy-looking", guarded, methods=["GET"]),
    ])
    return TestClient(AuthMiddleware(app, identity=lambda: _FX.identity))


class TestReachability(unittest.TestCase):
    def test_health_needs_no_credentials(self):
        resp = _app().get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_health_works_with_credentials_too(self):
        resp = _app().get("/health", headers={"Authorization": f"Bearer {_FX.admin_token}"})
        self.assertEqual(resp.status_code, 200)

    def test_a_bad_token_does_not_break_health(self):
        # A probe misconfigured with a stale key must still report liveness.
        resp = _app().get("/health", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(resp.status_code, 200)



class TestExemptionIsNarrow(unittest.TestCase):
    """The exemption is exactly /health; the guarded prefixes stay guarded."""

    def test_other_routes_stay_protected(self):
        self.assertEqual(_app().get("/export").status_code, 401)

    def test_a_path_merely_containing_health_is_protected(self):
        self.assertEqual(_app().get("/api/healthy-looking").status_code, 401)
        self.assertEqual(_app().get("/mcp/healthy-looking").status_code, 401)


class TestResponseLeaksNothing(unittest.TestCase):
    """An unauthenticated caller learns only that something is listening."""

    def test_body_carries_status_and_nothing_else(self):
        body = _app().get("/health").json()
        self.assertEqual(list(body), ["status"])

    def test_no_version_or_configuration_disclosed(self):
        raw = json.dumps(_app().get("/health").json()).lower()
        for leak in ("version", "model", "embed", "auth", "path", "namespace"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, raw)


class TestDoesNotTouchTheDatabase(unittest.TestCase):
    """Probing must not depend on the database: polling it adds load, and a
    momentary lock would turn into a flapping health state."""

    def test_health_answers_with_no_database_configured(self):
        # No _db patching anywhere — if the route touched the database it would
        # try the real DB_PATH and fail.
        resp = _app().get("/health")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
