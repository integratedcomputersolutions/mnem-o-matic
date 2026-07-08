"""Tests for the Bearer token authentication middleware (mnemomatic.auth).

These drive BearerAuthMiddleware end-to-end through a Starlette TestClient so the
real dispatch() path is exercised: 401/403 status codes, the exact error bodies,
the /ui bypass, the constant-time token comparison, and the brute-force lockout —
not merely constructor state. A few constructor tests cover the api_key
normalization that gates it all.
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
from mnemomatic.throttle import FailureThrottle

API_KEY = "test-secret-key-12345"


async def _ok(request):
    return PlainTextResponse("ok")


def _client(api_key, exempt_ui=False):
    """A TestClient over BearerAuthMiddleware wrapping a trivial app.

    Routes mirror the real shape: a protected MCP path and the /ui paths that
    are exempt only when the web viewer is registered (exempt_ui=True).
    """
    app = Starlette(routes=[
        Route("/mcp", _ok),
        Route("/ui", _ok),
        Route("/ui/login", _ok),
    ])
    return TestClient(BearerAuthMiddleware(app, api_key=api_key, exempt_ui=exempt_ui))


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
    enforced there — but only when the viewer is registered (exempt_ui=True)."""

    def test_ui_root_exempt_without_token(self):
        resp = _client(API_KEY, exempt_ui=True).get("/ui")
        self.assertEqual(resp.status_code, 200)

    def test_ui_subpath_exempt_without_token(self):
        resp = _client(API_KEY, exempt_ui=True).get("/ui/login")
        self.assertEqual(resp.status_code, 200)

    def test_mcp_path_still_protected_when_ui_exempt(self):
        resp = _client(API_KEY, exempt_ui=True).get("/mcp")
        self.assertEqual(resp.status_code, 401)

    def test_ui_requires_bearer_when_viewer_disabled(self):
        # Default (viewer not registered): /ui paths get no free pass.
        client = _client(API_KEY)
        self.assertEqual(client.get("/ui").status_code, 401)
        self.assertEqual(client.get("/ui/login").status_code, 401)


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


class TestThrottling(unittest.TestCase):
    """Repeated invalid keys must lock the client out with 429 responses."""

    def test_lockout_after_repeated_invalid_keys(self):
        client = _client(API_KEY)
        for _ in range(5):
            resp = client.get("/mcp", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(resp.status_code, 403)
        # Locked out: even the correct key is refused until the lockout expires.
        resp = client.get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Retry-After", resp.headers)

    def test_success_clears_failure_count(self):
        client = _client(API_KEY)
        for _ in range(4):
            client.get("/mcp", headers={"Authorization": "Bearer wrong"})
        # A success before the threshold resets the counter...
        resp = client.get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 200)
        # ...so the next failure doesn't trip the lockout.
        resp = client.get("/mcp", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(resp.status_code, 403)
        resp = client.get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 200)

    def test_missing_header_does_not_count_toward_lockout(self):
        # Only credential guesses are throttled; misconfigured clients that
        # send no header at all keep getting 401, never 429.
        client = _client(API_KEY)
        for _ in range(10):
            self.assertEqual(client.get("/mcp").status_code, 401)
        resp = client.get("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(resp.status_code, 200)

    def test_auth_disabled_never_throttles(self):
        client = _client("")
        for _ in range(10):
            resp = client.get("/mcp", headers={"Authorization": "Bearer whatever"})
            self.assertEqual(resp.status_code, 200)


class TestFailureThrottle(unittest.TestCase):
    """Unit tests for the throttle primitive itself."""

    def test_locks_after_max_failures(self):
        throttle = FailureThrottle(max_failures=3, window=60, lockout=300)
        for _ in range(2):
            throttle.record_failure("1.2.3.4")
        self.assertEqual(throttle.retry_after("1.2.3.4"), 0)
        throttle.record_failure("1.2.3.4")
        self.assertGreater(throttle.retry_after("1.2.3.4"), 0)

    def test_clients_are_independent(self):
        throttle = FailureThrottle(max_failures=2, window=60, lockout=300)
        throttle.record_failure("attacker")
        throttle.record_failure("attacker")
        self.assertGreater(throttle.retry_after("attacker"), 0)
        self.assertEqual(throttle.retry_after("innocent"), 0)

    def test_success_resets_failures(self):
        throttle = FailureThrottle(max_failures=2, window=60, lockout=300)
        throttle.record_failure("ip")
        throttle.record_success("ip")
        throttle.record_failure("ip")
        self.assertEqual(throttle.retry_after("ip"), 0)

    def test_lockout_expires(self):
        throttle = FailureThrottle(max_failures=1, window=60, lockout=0.0)
        throttle.record_failure("ip")
        self.assertEqual(throttle.retry_after("ip"), 0)


if __name__ == "__main__":
    unittest.main()
