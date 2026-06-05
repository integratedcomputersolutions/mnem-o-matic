"""Tests for the Bearer token authentication middleware (mnemomatic.auth).

These drive BearerAuthMiddleware end-to-end through a Starlette TestClient so the
real dispatch() path is exercised: 401/403 status codes, the exact error bodies,
the /ui bypass, and the constant-time token comparison — not merely constructor
state. A few constructor tests cover the api_key normalization that gates it all.
"""

import hmac
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic.auth import BearerAuthMiddleware

API_KEY = "test-secret-key-12345"


async def _ok(request):
    return PlainTextResponse("ok")


def _client(api_key):
    """A TestClient over BearerAuthMiddleware wrapping a trivial app.

    Routes mirror the real shape: a protected MCP path and the exempt /ui paths.
    """
    app = Starlette(routes=[
        Route("/mcp", _ok),
        Route("/ui", _ok),
        Route("/ui/login", _ok),
    ])
    return TestClient(BearerAuthMiddleware(app, api_key=api_key))


class TestConstructor(unittest.TestCase):
    """api_key normalization and the auth_enabled flag derived from it."""

    def test_empty_key_disables_auth(self):
        mw = BearerAuthMiddleware(AsyncMock(), api_key="")
        self.assertFalse(mw.auth_enabled)
        self.assertEqual(mw.api_key, "")

    def test_whitespace_only_key_disables_auth(self):
        # A key of only whitespace trims to "" → auth disabled.
        mw = BearerAuthMiddleware(AsyncMock(), api_key="   ")
        self.assertFalse(mw.auth_enabled)
        self.assertEqual(mw.api_key, "")

    def test_nonempty_key_enables_auth_and_is_trimmed(self):
        mw = BearerAuthMiddleware(AsyncMock(), api_key="  secret  ")
        self.assertTrue(mw.auth_enabled)
        self.assertEqual(mw.api_key, "secret")


class TestAuthDisabled(unittest.TestCase):
    """With no key, every request passes through regardless of headers."""

    def test_request_without_header_allowed(self):
        resp = _client("").get("/mcp")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "ok")

    def test_request_with_header_allowed(self):
        resp = _client("").get("/mcp", headers={"Authorization": "Bearer anything"})
        self.assertEqual(resp.status_code, 200)


class TestAuthEnabled(unittest.TestCase):
    """With a key set, dispatch() enforces the Bearer scheme and token."""

    def test_valid_token_accepted(self):
        resp = _client(API_KEY).get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "ok")

    def test_scheme_is_case_insensitive(self):
        resp = _client(API_KEY).get("/mcp", headers={"Authorization": f"bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 200)

    def test_surrounding_whitespace_in_token_tolerated(self):
        # token = auth_header[7:].strip(), so extra spaces still match.
        resp = _client(API_KEY).get("/mcp", headers={"Authorization": f"Bearer  {API_KEY} "})
        self.assertEqual(resp.status_code, 200)

    def test_missing_header_401(self):
        resp = _client(API_KEY).get("/mcp")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json(), {
            "error": "Missing Authorization header",
            "details": "Required format: 'Authorization: Bearer <token>'",
        })

    def test_wrong_scheme_401(self):
        resp = _client(API_KEY).get("/mcp", headers={"Authorization": f"Basic {API_KEY}"})
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "Invalid Authorization header format")

    def test_bearer_with_no_token_401(self):
        # "Bearer " is trimmed to "Bearer" by the leading strip(), so it fails
        # the "bearer " prefix check and is reported as a format error. (The
        # separate "Token is empty" branch is therefore unreachable via a real
        # header — the strip() removes the trailing space that would expose it.)
        resp = _client(API_KEY).get("/mcp", headers={"Authorization": "Bearer "})
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "Invalid Authorization header format")

    def test_wrong_token_403(self):
        resp = _client(API_KEY).get("/mcp", headers={"Authorization": "Bearer wrong-key"})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "Invalid API key")

    def test_token_compared_with_constant_time_digest(self):
        # The middleware must delegate to hmac.compare_digest (timing-safe),
        # not a plain ==. Wrap the real function so behavior is unchanged while
        # we assert it was called with (presented token, configured key).
        with patch("mnemomatic.auth.hmac.compare_digest", wraps=hmac.compare_digest) as cd:
            resp = _client(API_KEY).get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 200)
        cd.assert_called_once_with(API_KEY, API_KEY)


class TestUIExemption(unittest.TestCase):
    """/ui carries its own shared-secret gate, so the MCP Bearer token is not
    enforced there even when auth is enabled."""

    def test_ui_root_exempt_without_token(self):
        resp = _client(API_KEY).get("/ui")
        self.assertEqual(resp.status_code, 200)

    def test_ui_subpath_exempt_without_token(self):
        resp = _client(API_KEY).get("/ui/login")
        self.assertEqual(resp.status_code, 200)


class TestLogging(unittest.TestCase):
    """Initialization and per-request logging side effects."""

    def test_enabled_logs_info_on_init(self):
        with patch("mnemomatic.auth.logger") as log:
            BearerAuthMiddleware(AsyncMock(), api_key="secret")
        log.info.assert_called_once_with("Authentication enabled (Bearer token required)")

    def test_disabled_logs_warning_on_init(self):
        with patch("mnemomatic.auth.logger") as log:
            BearerAuthMiddleware(AsyncMock(), api_key="")
        log.warning.assert_called_once()

    def test_unauthorized_request_logged_as_warning(self):
        client = _client(API_KEY)
        with patch("mnemomatic.auth.logger") as log:
            client.get("/mcp")  # missing header → 401
        log.warning.assert_called()

    def test_authenticated_request_logged_at_debug(self):
        client = _client(API_KEY)
        with patch("mnemomatic.auth.logger") as log:
            client.get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        log.debug.assert_called()


if __name__ == "__main__":
    unittest.main()
