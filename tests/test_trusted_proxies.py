"""Tests for MNEMOMATIC_TRUSTED_PROXIES.

The server does not parse forwarded headers itself — it hands the trust list
to uvicorn, whose ProxyHeadersMiddleware rewrites scope["client"] and
scope["scheme"] only when the socket peer is on that list. What matters here
is the consequence: the brute-force throttle, the audit log, and the viewer's
Secure cookie all read the request's client, so they follow the real client
when a proxy is trusted and the proxy itself when it is not.

These wrap the app the way uvicorn does, so the middleware under test is the
same object the server runs behind.
"""

import os
import unittest
from unittest.mock import patch

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from mnemomatic import config
from mnemomatic.auth import BearerAuthMiddleware

API_KEY = "test-secret-key-12345"
PROXY = "10.0.0.1"


async def _ok(request):
    return PlainTextResponse("ok")


def _client(trusted, peer=PROXY):
    """A TestClient whose stack matches the server's: uvicorn's proxy-header
    middleware outermost, then Bearer auth with its throttle."""
    app = Starlette(routes=[Route("/mcp", _ok, methods=["GET", "POST"])])
    guarded = BearerAuthMiddleware(app, api_key=API_KEY)
    return TestClient(ProxyHeadersMiddleware(guarded, trusted_hosts=trusted),
                      client=(peer, 40000))


def _lock_out(client, forwarded_for):
    """Spend the throttle's allowance from one forwarded address."""
    headers = {"Authorization": "Bearer wrong", "X-Forwarded-For": forwarded_for}
    for _ in range(5):
        client.post("/mcp", headers=headers)


class TestConfigParsing(unittest.TestCase):
    def _parsed(self, value):
        env = {} if value is None else {"MNEMOMATIC_TRUSTED_PROXIES": value}
        with patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("MNEMOMATIC_TRUSTED_PROXIES", None)
            return config._trusted_proxies()

    def test_unset_is_empty(self):
        self.assertEqual(self._parsed(None), [])
        self.assertEqual(self._parsed(""), [])

    def test_wildcard(self):
        self.assertEqual(self._parsed("*"), ["*"])

    def test_list_of_addresses_and_cidrs(self):
        self.assertEqual(
            self._parsed("10.0.0.1, 172.16.0.0/12"), ["10.0.0.1", "172.16.0.0/12"]
        )

    def test_blank_entries_dropped(self):
        self.assertEqual(self._parsed(" , 10.0.0.1 ,, "), ["10.0.0.1"])


class TestThrottleKeying(unittest.TestCase):
    def test_trusted_proxy_locks_only_the_forwarded_client(self):
        client = _client([PROXY])
        _lock_out(client, "203.0.113.7")

        # The offender is locked out even with the right key...
        locked = client.post("/mcp", headers={"Authorization": f"Bearer {API_KEY}",
                                              "X-Forwarded-For": "203.0.113.7"})
        self.assertEqual(locked.status_code, 429)

        # ...while everyone else behind the same proxy is unaffected. Without a
        # trusted proxy this is the denial of service: one client's failures
        # would lock the shared address for all of them.
        other = client.post("/mcp", headers={"Authorization": f"Bearer {API_KEY}",
                                             "X-Forwarded-For": "203.0.113.8"})
        self.assertEqual(other.status_code, 200)

    def test_untrusted_peer_keys_on_the_socket_address(self):
        # The peer is not on the trust list, so the header is ignored and every
        # request keys on the peer: the lockout catches a different X-Forwarded-For.
        client = _client(["192.0.2.1"])
        _lock_out(client, "203.0.113.7")
        resp = client.post("/mcp", headers={"Authorization": f"Bearer {API_KEY}",
                                            "X-Forwarded-For": "203.0.113.8"})
        self.assertEqual(resp.status_code, 429)

    def test_forwarded_address_is_not_believed_from_an_untrusted_peer(self):
        # Spoofing X-Forwarded-For to dodge a lockout only works from a peer
        # the operator has declared trustworthy.
        client = _client([], peer="198.51.100.5")
        _lock_out(client, "203.0.113.7")
        resp = client.post("/mcp", headers={"Authorization": f"Bearer {API_KEY}",
                                            "X-Forwarded-For": "203.0.113.9"})
        self.assertEqual(resp.status_code, 429)


class TestScheme(unittest.TestCase):
    """X-Forwarded-Proto reaches request.url.scheme on the same terms."""

    def _scheme(self, trusted, header):
        seen = {}

        async def report(request):
            seen["scheme"] = request.url.scheme
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/s", report)])
        client = TestClient(ProxyHeadersMiddleware(app, trusted_hosts=trusted),
                            client=(PROXY, 40000))
        client.get("http://testserver/s", headers={"X-Forwarded-Proto": header})
        return seen["scheme"]

    def test_trusted_proxy_sets_the_scheme(self):
        self.assertEqual(self._scheme([PROXY], "https"), "https")

    def test_untrusted_peer_leaves_the_scheme_alone(self):
        self.assertEqual(self._scheme(["192.0.2.1"], "https"), "http")


if __name__ == "__main__":
    unittest.main()
