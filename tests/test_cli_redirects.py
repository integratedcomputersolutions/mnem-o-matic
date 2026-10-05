"""The CLI must not hand its bearer token to whatever a 30x points at.

urllib's default redirect handler copies Authorization to any host or
scheme. Both CLI call sites (MCP and export) go through _open, which follows
redirects only within the origin the request was sent to. Two real local
servers stand in for "the server" and "somewhere else".
"""

import types
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

from mnemomatic_cli import cli
from mnemomatic_cli._mcp_client import MCPClient, _describe_http_error, _open, _origin
from tests._support import serve


class _Recorder(BaseHTTPRequestHandler):
    seen: list = []

    def _handle(self):
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = do_POST = _handle

    def log_message(self, *a):
        pass


class TestRedirects(unittest.TestCase):
    def setUp(self):
        _Recorder.seen = []
        self.other = serve(self, _Recorder)
        other_url = f"http://127.0.0.1:{self.other.server_port}"

        class Redirector(_Recorder):
            seen = []

            def _handle(self):
                if self.path.startswith("/away"):
                    target = other_url + "/stolen"
                elif self.path.startswith("/here"):
                    target = "/landed"
                else:
                    return super()._handle()
                type(self).seen.append((self.path, self.headers.get("Authorization")))
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _handle

        self.Redirector = Redirector
        self.server = serve(self, Redirector)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def _get(self, path):
        req = urllib.request.Request(self.base + path, headers={"Authorization": "Bearer mnm_secret"})
        return _open(req, timeout=5)

    def test_cross_origin_redirect_is_refused_and_token_never_leaves(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/away")
        self.assertEqual(ctx.exception.code, 302)
        self.assertEqual(_Recorder.seen, [])
        message = _describe_http_error(ctx.exception)
        self.assertIn(f"redirected to http://127.0.0.1:{self.other.server_port}/stolen", message)
        self.assertIn("not following it", message)

    def test_same_origin_redirect_is_followed_with_the_token(self):
        with self._get("/here") as resp:
            self.assertEqual(resp.status, 200)
        self.assertEqual(self.Redirector.seen[-1], ("/landed", "Bearer mnm_secret"))

    def test_export_refuses_cross_origin_redirect(self):
        args = types.SimpleNamespace(namespace=None, output="-")
        with patch.object(cli, "_err", side_effect=SystemExit) as err:
            with self.assertRaises(SystemExit):
                cli._cmd_export(args, self.base + "/away", "mnm_secret")
        self.assertIn("not following it", err.call_args.args[0])
        self.assertEqual(_Recorder.seen, [])

    def test_mcp_client_refuses_cross_origin_redirect(self):
        with self.assertRaises(RuntimeError) as ctx:
            MCPClient(base_url=self.base + "/away", api_key="mnm_secret")
        self.assertIn("not following it", str(ctx.exception))
        self.assertEqual(_Recorder.seen, [])


class TestOrigin(unittest.TestCase):
    def test_default_ports_and_case(self):
        self.assertEqual(_origin("https://Memory.Example/mcp"), _origin("https://memory.example:443/x"))
        self.assertNotEqual(_origin("https://memory.example/mcp"), _origin("http://memory.example/mcp"))
        self.assertNotEqual(_origin("https://memory.example/mcp"), _origin("https://memory.example:8443/mcp"))
        self.assertNotEqual(_origin("https://memory.example/mcp"), _origin("https://evil.example/mcp"))


if __name__ == "__main__":
    unittest.main()
