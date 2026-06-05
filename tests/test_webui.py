"""Tests for the read-only web viewer (mnemomatic.webui).

Covers the shared-secret gate, navigation, content rendering with HTML
escaping, and the BearerAuthMiddleware /ui exemption.
"""

import sys
import tempfile
import unittest
from pathlib import Path

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
        resp = self.client.get("/ui/logout")
        self.assertEqual(resp.status_code, 303)
        # cookie removed → index redirects back to login
        self.assertEqual(self.client.get("/ui").status_code, 303)

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


class TestBearerExemption(unittest.TestCase):
    """BearerAuthMiddleware must let /ui through without an MCP Bearer token."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = Database(self._tmp.name)
        _seed(self.db)
        app = Starlette()
        register_webui(app, lambda: self.db, TOKEN)
        wrapped = BearerAuthMiddleware(app, api_key="mcp-key")
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


if __name__ == "__main__":
    unittest.main()
