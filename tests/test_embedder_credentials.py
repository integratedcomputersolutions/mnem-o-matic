"""Credentials for an external embedding endpoint stay with the operator.

MNEMOMATIC_EMBED_API_KEY travels as a header rather than in the URL; the URL
is shown only with userinfo and query stripped (logs, errors, settings); and
the endpoint is never followed through a redirect, so the key cannot be
carried to another host.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mnemomatic.embeddings import HttpEmbedder, redact_url


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _Endpoint(BaseHTTPRequestHandler):
    seen: list = []
    redirect_to: str | None = None

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).seen.append((self.path, self.headers.get("Authorization")))
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
            redirect_to = None

        class Elsewhere(_Endpoint):
            seen = []
            redirect_to = None

        self.Endpoint, self.Elsewhere = Endpoint, Elsewhere
        for cls in (Endpoint, Elsewhere):
            server = _serve(cls)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
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


if __name__ == "__main__":
    unittest.main()
