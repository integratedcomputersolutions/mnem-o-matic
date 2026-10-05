"""Tests for serving the single-page app (mnemomatic.spa.build_spa_routes)."""

import tempfile
import unittest
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic.spa import build_spa_routes
from tests._support import SPA_HTML



def _stub(request):
    return JSONResponse({"stub": request.url.path})


class SpaCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app_dir = Path(self.tmp.name)
        (self.app_dir / "assets").mkdir()
        (self.app_dir / "assets" / "index-abc123.js").write_text("console.log('hi')")
        (self.app_dir / "favicon.svg").write_text("<svg/>")
        (self.app_dir / "index.html").write_text(SPA_HTML)
        self.client = self._client()

    def _client(self):
        routes = [Route("/mcp", _stub, methods=["GET", "POST"]), Route("/api/ping", _stub)]
        routes += build_spa_routes(self.app_dir)
        return TestClient(Starlette(routes=routes), follow_redirects=False)

    HTML_ACCEPT = {"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}


class TestIndex(SpaCase):
    def test_root_and_deep_paths_serve_index(self):
        for path in ("/", "/browse", "/browse/ns/document/abc", "/admin/users"):
            resp = self.client.get(path, headers=self.HTML_ACCEPT)
            self.assertEqual(resp.status_code, 200, path)
            self.assertTrue(resp.headers["content-type"].startswith("text/html"))
            self.assertEqual(resp.headers["cache-control"], "no-cache")
            self.assertIn("<div id=app>", resp.text)

    def test_head_works(self):
        self.assertEqual(self.client.head("/", headers=self.HTML_ACCEPT).status_code, 200)

    def test_post_is_not_allowed(self):
        self.assertEqual(self.client.post("/", headers=self.HTML_ACCEPT).status_code, 405)

    def test_reserved_prefixes_404_as_json(self):
        for path in ("/api/nope", "/.well-known/oauth-authorization-server", "/mcp/extra", "/export/x",
                     "/health/x", "/setup/x", "/assets/missing.js"):
            resp = self.client.get(path, headers=self.HTML_ACCEPT)
            self.assertEqual(resp.status_code, 404, path)
            if path != "/assets/missing.js":
                self.assertEqual(resp.json()["error"], "not_found")

    def test_non_html_clients_get_404_not_a_page(self):
        resp = self.client.get("/foo", headers={"Accept": "application/json"})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"], "not_found")
        resp = self.client.get("/foo", headers={"Accept": "application/json, text/event-stream"})
        self.assertEqual(resp.status_code, 404)
        # A bare */* (curl) still gets the page.
        self.assertEqual(self.client.get("/foo", headers={"Accept": "*/*"}).status_code, 200)

    def test_other_routes_still_win(self):
        self.assertEqual(self.client.get("/mcp").json(), {"stub": "/mcp"})
        self.assertEqual(self.client.get("/api/ping").json(), {"stub": "/api/ping"})


class TestAssets(SpaCase):
    def test_hashed_assets_are_immutable(self):
        resp = self.client.get("/assets/index-abc123.js")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["cache-control"], "public, max-age=31536000, immutable")
        self.assertIn("javascript", resp.headers["content-type"])

    def test_root_files_served_uncached(self):
        resp = self.client.get("/favicon.svg")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["cache-control"], "no-cache")
        self.assertIn("svg", resp.headers["content-type"])

    def test_traversal_is_not_a_file(self):
        for path in ("/assets/../index.html", "/..%2Findex.html", "/.env"):
            resp = self.client.get(path, headers=self.HTML_ACCEPT)
            self.assertIn(resp.status_code, (200, 404), path)
            if resp.status_code == 200:
                self.assertIn("<div id=app>", resp.text)        # fell through to index, not a file


class TestMissingBuild(unittest.TestCase):
    def test_503_with_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("mnemomatic", level="WARNING"):
                routes = build_spa_routes(Path(tmp))
            client = TestClient(Starlette(routes=routes))
            resp = client.get("/", headers={"Accept": "text/html"})
            self.assertEqual(resp.status_code, 503)
            self.assertIn("npm --prefix web run build", resp.text)
            self.assertEqual(client.get("/assets/x.js").status_code, 404)


if __name__ == "__main__":
    unittest.main()
