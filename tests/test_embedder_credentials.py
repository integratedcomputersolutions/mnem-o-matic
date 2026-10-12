"""Credentials for an external embedding endpoint stay with the operator.

MNEMOMATIC_EMBED_API_KEY travels as a header rather than in the URL; the URL
is shown only with userinfo and query stripped (logs, errors, settings); and
the endpoint is never followed through a redirect, so the key cannot be
carried to another host.
"""

import base64
import hashlib
import hmac
import json
import time
import unittest
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

from mnemomatic import audit, config, server
from mnemomatic.embeddings import EMBED_USER_TOKEN_TTL, HttpEmbedder, mint_user_token, redact_url
from tests._support import serve

USER_HEADER = "X-Gateway-User-Jwt"
USER_SECRET = "u" * 40


class _Endpoint(BaseHTTPRequestHandler):
    seen: list = []
    identities: list = []
    redirect_to: str | None = None

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        type(self).identities.append(self.headers.get(USER_HEADER))
        if self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"data": [{"embedding": [3.0, 4.0]}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # urllib follows a 302 to a POST as a GET; record those too, or "the
        # other host saw nothing" would hold even if the redirect were followed.
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


class TestRedactUrl(unittest.TestCase):
    def test_strips_userinfo_query_and_fragment(self):
        self.assertEqual(redact_url("https://u:p@api.example.com/v1/embeddings?key=s#f"),
                         "https://api.example.com/v1/embeddings?…")
        self.assertEqual(redact_url("http://embed:8181/v1/embeddings"), "http://embed:8181/v1/embeddings")
        self.assertEqual(redact_url("http://[::1]:11434/api/embeddings"), "http://[::1]:11434/api/embeddings")
        self.assertEqual(redact_url("http://u@host/x"), "http://host/x")


class TestHttpEmbedderCredentials(unittest.TestCase):
    def setUp(self):
        class Endpoint(_Endpoint):
            seen = []
            identities = []
            redirect_to = None

        class Elsewhere(_Endpoint):
            seen = []
            identities = []
            redirect_to = None

        self.Endpoint, self.Elsewhere = Endpoint, Elsewhere
        for cls in (Endpoint, Elsewhere):
            server = serve(self, cls)
            cls.url = f"http://127.0.0.1:{server.server_port}"

    def test_api_key_sent_as_bearer_header(self):
        e = HttpEmbedder(self.Endpoint.url + "/v1/embeddings", api="openai", api_key="sk-secret")
        self.assertEqual(e.embed("hello"), [0.6, 0.8])
        self.assertEqual(self.Endpoint.seen, [("/v1/embeddings", "Bearer sk-secret")])

    def test_no_key_no_header(self):
        HttpEmbedder(self.Endpoint.url + "/v1/embeddings", api="openai", api_key="").embed("hello")
        self.assertEqual(self.Endpoint.seen, [("/v1/embeddings", None)])

    def test_redirect_is_not_followed(self):
        self.Endpoint.redirect_to = self.Elsewhere.url + "/stolen"
        e = HttpEmbedder(self.Endpoint.url + "/v1/embeddings", api="openai", api_key="sk-secret")
        with self.assertRaises(RuntimeError) as ctx:
            e.embed("hello")
        self.assertEqual(self.Elsewhere.seen, [])          # the key never reached the other host
        self.assertIn("HTTP 302", str(ctx.exception))

    def test_errors_show_the_redacted_url(self):
        e = HttpEmbedder("http://user:hunter2@127.0.0.1:9/v1/embeddings?key=sk-secret", api="openai")
        with self.assertRaises(RuntimeError) as ctx:
            e.embed("hello")
        message = str(ctx.exception)
        self.assertIn("127.0.0.1:9/v1/embeddings", message)
        for secret in ("hunter2", "sk-secret", "user"):
            self.assertNotIn(secret, message)


def _decode(token: str, secret: str) -> dict:
    """Verify an HS256 JWT the way the gateway would and return its claims."""
    header, claims, signature = token.split(".")
    expected = hmac.new(secret.encode(), f"{header}.{claims}".encode(), hashlib.sha256).digest()

    def pad(part):
        return part + "=" * (-len(part) % 4)

    assert hmac.compare_digest(base64.urlsafe_b64decode(pad(signature)), expected), "bad signature"
    assert json.loads(base64.urlsafe_b64decode(pad(header))) == {"alg": "HS256", "typ": "JWT"}
    return json.loads(base64.urlsafe_b64decode(pad(claims)))


class TestUserToken(unittest.TestCase):
    def test_mint_and_verify(self):
        claims = _decode(mint_user_token("bob@example.com", USER_SECRET, now=1_700_000_000.9), USER_SECRET)
        self.assertEqual(claims, {"email": "bob@example.com", "iat": 1_700_000_000,
                                  "exp": 1_700_000_000 + EMBED_USER_TOKEN_TTL})
        with self.assertRaises(AssertionError):
            _decode(mint_user_token("bob@example.com", USER_SECRET), "w" * 40)


class _RequestContext:
    """Enter a request's identity as RequestMetaMiddleware would."""

    def __init__(self, **fields):
        self.meta = {**audit._EMPTY, **fields}

    def __enter__(self):
        self._token = audit._request_meta.set(self.meta)
        return self.meta

    def __exit__(self, *exc):
        audit._request_meta.reset(self._token)


class TestEmbeddingCallsNameThePerson(TestHttpEmbedderCredentials):
    def embedder(self, header=USER_HEADER, secret=USER_SECRET):
        return HttpEmbedder(self.Endpoint.url + "/v1/embeddings", api="openai", api_key="",
                            user_header=header, user_secret=secret)

    def test_a_proxy_identified_person_is_named(self):
        before = time.time()
        with _RequestContext(user="bob@example.com", external_id="bob@example.com", via="proxy"):
            self.embedder().embed("hello")
        [token] = self.Endpoint.identities
        claims = _decode(token, USER_SECRET)
        self.assertEqual(claims["email"], "bob@example.com")
        self.assertGreaterEqual(claims["iat"], int(before))
        self.assertEqual(claims["exp"], claims["iat"] + EMBED_USER_TOKEN_TTL)

    def test_nobody_identifiable_means_no_header(self):
        # A local account has no proxy identity; the startup reindex has no request at all.
        with _RequestContext(user="alice", via="session"):
            self.embedder().embed("one")
        self.embedder().embed("two")
        self.assertEqual(self.Endpoint.identities, [None, None])
        self.assertNotIn("embed_user_jwt", audit._EMPTY)

    def test_half_a_setting_sends_nothing(self):
        with _RequestContext(external_id="bob@example.com"):
            self.embedder(secret="").embed("one")
            self.embedder(header="").embed("two")
        self.assertEqual(self.Endpoint.identities, [None, None])

    def test_every_chunk_of_one_request_carries_the_same_token(self):
        with _RequestContext(external_id="bob@example.com") as meta:
            vectors = self.embedder().embed_batch(["a", "b", "c"])
            again = self.embedder().embed("d")
        self.assertEqual(len([v for v in vectors if v]), 3)
        self.assertEqual(again, [0.6, 0.8])
        tokens = set(self.Endpoint.identities)
        self.assertEqual(len(self.Endpoint.identities), 4)
        self.assertEqual(len(tokens), 1, "one token per request, workers included")
        self.assertEqual(tokens, {meta["embed_user_jwt"]})

    def test_two_requests_two_tokens(self):
        with _RequestContext(external_id="bob@example.com"):
            self.embedder().embed("x")
        with _RequestContext(external_id="carol@example.com"):
            self.embedder().embed("y")
        a, b = self.Endpoint.identities
        self.assertEqual(_decode(a, USER_SECRET)["email"], "bob@example.com")
        self.assertEqual(_decode(b, USER_SECRET)["email"], "carol@example.com")

    def test_token_and_secret_stay_out_of_error_text(self):
        e = HttpEmbedder("http://127.0.0.1:9/v1/embeddings", api="openai", api_key="",
                         user_header=USER_HEADER, user_secret=USER_SECRET)
        with _RequestContext(external_id="bob@example.com") as meta, self.assertRaises(RuntimeError) as ctx:
            e.embed("hello")
        self.assertNotIn(USER_SECRET, str(ctx.exception))
        self.assertNotIn(meta["embed_user_jwt"], str(ctx.exception))


class TestEmbedUserStartupCheck(unittest.TestCase):
    def _run(self, header, secret, url="http://embed:8181/v1/embeddings"):
        with patch.multiple("mnemomatic.embeddings", EMBED_USER_HEADER=header, EMBED_USER_SECRET=secret), \
             patch.object(config, "EMBED_URL", url):
            server._check_embed_user_config()

    def test_short_secret_refuses_to_start(self):
        with self.assertRaises(SystemExit), self.assertLogs("mnemomatic", "ERROR"):
            self._run(USER_HEADER, "short")

    def test_half_a_setting_warns(self):
        for header, secret in ((USER_HEADER, ""), ("", USER_SECRET)):
            with self.subTest(header=header, secret=secret), self.assertLogs("mnemomatic", "WARNING") as logs:
                self._run(header, secret)
            self.assertIn("only together", "\n".join(logs.output))
            self.assertNotIn(USER_SECRET, "\n".join(logs.output))

    def test_without_an_external_embedder_warns(self):
        with self.assertLogs("mnemomatic", "WARNING") as logs:
            self._run(USER_HEADER, USER_SECRET, url="")
        self.assertIn("MNEMOMATIC_EMBED_URL", "\n".join(logs.output))

    def test_both_set_is_announced_without_the_secret(self):
        with self.assertLogs("mnemomatic", "INFO") as logs:
            self._run(USER_HEADER, USER_SECRET)
        text = "\n".join(logs.output)
        self.assertIn(USER_HEADER, text)
        self.assertNotIn(USER_SECRET, text)

    def test_unset_is_silent(self):
        with self.assertNoLogs("mnemomatic"):
            self._run("", "")


if __name__ == "__main__":
    unittest.main()
