"""Tests for the read-only web viewer (mnemomatic.webui).

Covers the shared-secret gate, navigation, content rendering with HTML
escaping, and the BearerAuthMiddleware /ui exemption.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from starlette.applications import Starlette
from starlette.testclient import TestClient

from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.db import Database
from mnemomatic.models import Document, Knowledge, Note
from mnemomatic.webui import COOKIE_NAME, register_webui

TOKEN = "s3cret-token"


def _seed(db: Database) -> dict:
    """Populate one of each item type; return their ids."""
    doc, _ = db.store_document(
        Document(namespace="proj", title="Design Doc", content="# Heading\nbody text"),
        embedding=None,
    )
    # Knowledge content carries an XSS probe to verify escaping.
    know, _ = db.store_knowledge(
        Knowledge(namespace="proj", subject="auth", fact="<script>alert(1)</script>"),
        embedding=None,
    )
    note, _ = db.store_note(
        Note(namespace="proj", title="Scratch", content="rough idea"),
        embedding=None,
    )
    return {"document": doc.id, "knowledge": know.id, "note": note.id}


class WebUITestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = Database(self._tmp.name)
        self.ids = _seed(self.db)
        app = Starlette()
        register_webui(app, lambda: self.db, TOKEN)
        self.client = TestClient(app, follow_redirects=False)

    def tearDown(self):
        self.db.close()
        Path(self._tmp.name).unlink(missing_ok=True)

    def _auth(self):
        """Log in, leaving the auth cookie on the client."""
        resp = self.client.post("/ui/login", data={"token": TOKEN})
        self.assertEqual(resp.status_code, 303)
        self.assertIn(COOKIE_NAME, self.client.cookies)


class TestGate(WebUITestBase):
    def test_index_requires_auth(self):
        resp = self.client.get("/ui")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_login_page_renders(self):
        resp = self.client.get("/ui/login")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access token", resp.text)

    def test_static_css_public(self):
        # The stylesheet must load without the cookie so the login page is styled.
        resp = self.client.get("/ui/static/bootstrap.min.css")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/css", resp.headers["content-type"])
        self.assertIn("Bootstrap", resp.text[:200])

    def test_wrong_token_rejected(self):
        resp = self.client.post("/ui/login", data={"token": "nope"})
        self.assertEqual(resp.status_code, 401)
        self.assertNotIn(COOKIE_NAME, self.client.cookies)

    def test_correct_token_sets_cookie(self):
        self._auth()

    def test_logout_clears_cookie(self):
        self._auth()
        resp = self.client.post("/ui/logout")
        self.assertEqual(resp.status_code, 303)
        # cookie removed → index redirects back to login
        self.assertEqual(self.client.get("/ui").status_code, 303)

    def test_logout_rejects_get(self):
        # Logout mutates state, so a plain link/GET (CSRF-able) must not work.
        self._auth()
        self.assertEqual(self.client.get("/ui/logout").status_code, 405)

    def test_cookie_does_not_contain_token(self):
        # The cookie must hold a derived session value, never the shared secret.
        self._auth()
        self.assertNotEqual(self.client.cookies[COOKIE_NAME], TOKEN)
        self.assertNotIn(TOKEN, self.client.cookies[COOKIE_NAME])

    def test_raw_token_as_cookie_rejected(self):
        # Forging the cookie from a known/leaked token must not authenticate.
        self.client.cookies.set(COOKIE_NAME, TOKEN)
        resp = self.client.get("/ui")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_login_throttled_after_repeated_failures(self):
        for _ in range(5):
            self.assertEqual(
                self.client.post("/ui/login", data={"token": "nope"}).status_code, 401
            )
        # Locked out now — even the correct token is refused until the lockout expires.
        resp = self.client.post("/ui/login", data={"token": TOKEN})
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Retry-After", resp.headers)
        self.assertNotIn(COOKIE_NAME, self.client.cookies)

    def test_namespace_view_requires_auth(self):
        resp = self.client.get("/ui/ns/proj")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_item_view_requires_auth(self):
        resp = self.client.get(f"/ui/item/document/{self.ids['document']}")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")


class TestViews(WebUITestBase):
    def setUp(self):
        super().setUp()
        self._auth()

    def test_index_lists_namespace(self):
        resp = self.client.get("/ui")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("proj", resp.text)
        self.assertIn("/ui/ns/proj", resp.text)

    def test_index_shows_per_type_counts(self):
        # Seeded: 1 document, 1 knowledge, 1 note. Add a second document so the
        # counts are distinguishable per column.
        self.db.store_document(
            Document(namespace="proj", title="Second Doc", content="more"), embedding=None
        )
        resp = self.client.get("/ui")
        row = next(line for line in resp.text.splitlines() if "/ui/ns/proj" in line)
        self.assertIn('<td class="text-end">2</td>', row)  # documents
        self.assertEqual(row.count('<td class="text-end">1</td>'), 2)  # knowledge, notes

    def test_namespace_lists_items(self):
        resp = self.client.get("/ui/ns/proj")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Design Doc", resp.text)
        self.assertIn("auth", resp.text)
        self.assertIn("Scratch", resp.text)

    def test_document_detail(self):
        resp = self.client.get(f"/ui/item/document/{self.ids['document']}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Design Doc", resp.text)
        self.assertIn("body text", resp.text)

    def test_knowledge_detail_escapes_html(self):
        resp = self.client.get(f"/ui/item/knowledge/{self.ids['knowledge']}")
        self.assertEqual(resp.status_code, 200)
        # The stored script tag must be escaped, never emitted raw.
        self.assertNotIn("<script>alert(1)</script>", resp.text)
        self.assertIn("&lt;script&gt;", resp.text)

    def test_unknown_item_type_404(self):
        resp = self.client.get(f"/ui/item/widget/{self.ids['document']}")
        self.assertEqual(resp.status_code, 404)

    def test_missing_item_404(self):
        resp = self.client.get("/ui/item/document/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_malicious_namespace_escaped_everywhere(self):
        # A namespace is attacker-controlled via the MCP API. It must be
        # HTML-escaped in every rendering context, including link hrefs
        # (the item-view breadcrumb used to interpolate it raw).
        evil_ns = '"><script>alert(1)</script>'
        doc, _ = self.db.store_document(
            Document(namespace=evil_ns, title="T", content="c"), embedding=None
        )
        for path in ("/ui", f"/ui/item/document/{doc.id}"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200)
            self.assertNotIn("<script>alert(1)</script>", resp.text, f"unescaped namespace in {path}")

    def test_namespace_links_url_encoded(self):
        # Namespaces may contain URL-special characters; links must encode them
        # so the round trip back to the namespace view works.
        ns = 'a/b?c d"e'
        self.db.store_note(Note(namespace=ns, title="n", content="x"), embedding=None)
        index = self.client.get("/ui")
        encoded = f"/ui/ns/{quote(ns, safe='')}"
        self.assertIn(encoded, index.text)
        resp = self.client.get(encoded)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Notes", resp.text)


class TestBearerExemption(unittest.TestCase):
    """BearerAuthMiddleware must let /ui through without an MCP Bearer token,
    but only when the viewer is registered (exempt_ui=True)."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = Database(self._tmp.name)
        _seed(self.db)
        app = Starlette()
        register_webui(app, lambda: self.db, TOKEN)
        wrapped = BearerAuthMiddleware(app, api_key="mcp-key", exempt_ui=True)
        self.client = TestClient(wrapped, follow_redirects=False)

    def tearDown(self):
        self.db.close()
        Path(self._tmp.name).unlink(missing_ok=True)

    def test_ui_reachable_without_bearer(self):
        # No Authorization header — would be 401 for MCP paths, but /ui is exempt
        # so it reaches its own gate and redirects to login.
        resp = self.client.get("/ui")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_non_ui_still_requires_bearer(self):
        resp = self.client.get("/mcp")
        self.assertEqual(resp.status_code, 401)

    def test_ui_requires_bearer_when_viewer_disabled(self):
        # Without exempt_ui (viewer not registered), /ui paths get no free pass.
        wrapped = BearerAuthMiddleware(Starlette(), api_key="mcp-key")
        client = TestClient(wrapped, follow_redirects=False)
        self.assertEqual(client.get("/ui").status_code, 401)
        self.assertEqual(client.get("/ui/anything").status_code, 401)


if __name__ == "__main__":
    unittest.main()
