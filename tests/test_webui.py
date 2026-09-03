"""Tests for the read-only web viewer (mnemomatic.webui).

Covers the shared-secret gate, per-login sessions, the security headers on
every page, navigation, content rendering with HTML escaping, and the
BearerAuthMiddleware /ui exemption.
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.testclient import TestClient

from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.db import Database
from mnemomatic.models import Document, Knowledge, Note
from mnemomatic import webui
from mnemomatic.webui import _MAX_SESSIONS, _SESSION_MAX_AGE, COOKIE_NAME, register_webui

TOKEN = "s3cret-token"

# What server._settings_info() produces for a built-in model; tests mutate a
# copy of this to exercise the mismatch alert and placeholder paths.
MODEL_INFO = {
    "version": "0.0.0-test",
    "mode": "built-in ONNX (test-model)",
    "model": "test-model",
    "model_url": "https://huggingface.co/test-org/test-model",
    "dim_configured": 384,
    "dim_database": 384,
    "max_tokens": 512,
    "query_prefix": "q: ",
    "doc_prefix": "",
    "chunk_threshold": 2000,
    "chunk_size": 1000,
    "chunk_overlap": 200,
}


def _seed(db: Database) -> dict:
    """Populate one of each item type; return their ids."""
    doc, _ = db.store_document(
        Document(namespace="proj", title="Design Doc", content="# Heading\nbody text"),
        embedding=None,
    )
    # Knowledge content carries an XSS probe to verify escaping.
    know, _, _ = db.store_knowledge(
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
        self.settings_info = dict(MODEL_INFO)
        self.app = Starlette()
        register_webui(self.app, lambda: self.db, TOKEN,
                       settings_info=lambda: self.settings_info,
                       make_export=lambda ns: (b"PK\x05\x06" + b"\x00" * 18, "test-export.zip"))
        self.client = TestClient(self.app, follow_redirects=False)

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
        # The cookie must hold a random session id, never the shared secret.
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

    def test_cookie_not_secure_over_plain_http(self):
        resp = self.client.post("/ui/login", data={"token": TOKEN})
        self.assertNotIn("Secure", resp.headers["set-cookie"])

    def test_forwarded_proto_header_alone_does_not_set_secure(self):
        # The Secure flag follows the connection scheme, which uvicorn resolves
        # from X-Forwarded-Proto only for proxies named in
        # MNEMOMATIC_TRUSTED_PROXIES. Read straight off the request, the header
        # would be believed from any client.
        resp = self.client.post("/ui/login", data={"token": TOKEN},
                                headers={"X-Forwarded-Proto": "https"})
        self.assertNotIn("Secure", resp.headers["set-cookie"])

    def test_cookie_secure_over_https(self):
        client = TestClient(self.app, base_url="https://testserver", follow_redirects=False)
        resp = client.post("/ui/login", data={"token": TOKEN})
        self.assertEqual(resp.status_code, 303)
        self.assertIn("Secure", resp.headers["set-cookie"])

    def test_multipart_login_rejected(self):
        # The login route is reachable without credentials, so it accepts only
        # the one encoding its form actually uses — no multipart parser runs
        # ahead of the gate.
        resp = self.client.post("/ui/login", files={"token": (None, TOKEN)})
        self.assertEqual(resp.status_code, 415)
        self.assertNotIn(COOKIE_NAME, self.client.cookies)

    def test_login_without_content_type_rejected(self):
        resp = self.client.request("POST", "/ui/login", content=b"token=" + TOKEN.encode())
        self.assertEqual(resp.status_code, 415)

    def test_urlencoded_login_still_works(self):
        resp = self.client.post(
            "/ui/login",
            content=b"token=" + TOKEN.encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self.assertEqual(resp.status_code, 303)

    def test_login_form_encoding_tolerates_a_charset_parameter(self):
        resp = self.client.post(
            "/ui/login",
            content=b"token=" + TOKEN.encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
        )
        self.assertEqual(resp.status_code, 303)

    def test_missing_token_field_is_a_wrong_token_not_a_crash(self):
        resp = self.client.post("/ui/login", data={"other": "x"})
        self.assertEqual(resp.status_code, 401)

    def test_namespace_view_requires_auth(self):
        resp = self.client.get("/ui/ns/proj")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_item_view_requires_auth(self):
        resp = self.client.get(f"/ui/item/document/{self.ids['document']}")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_settings_view_requires_auth(self):
        resp = self.client.get("/ui/settings")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_export_requires_auth(self):
        resp = self.client.get("/ui/export")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")


class TestSessions(WebUITestBase):
    """Session ids are per login, revocable, bounded, and expiring."""

    def _login(self, client=None):
        client = client or TestClient(self.app, follow_redirects=False)
        resp = client.post("/ui/login", data={"token": TOKEN})
        self.assertEqual(resp.status_code, 303)
        return client.cookies[COOKIE_NAME]

    def test_two_logins_get_different_cookies(self):
        self.assertNotEqual(self._login(), self._login())

    def test_logout_revokes_the_session_server_side(self):
        # Deleting the browser's copy is not enough: a captured cookie must
        # stop working, which means the server has to forget it.
        stale = self._login(self.client)
        self.client.post("/ui/logout")
        self.client.cookies.set(COOKIE_NAME, stale)
        resp = self.client.get("/ui")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/ui/login")

    def test_logout_leaves_other_sessions_alone(self):
        other = TestClient(self.app, follow_redirects=False)
        self._login(other)
        self._login(self.client)
        self.client.post("/ui/logout")
        self.assertEqual(other.get("/ui").status_code, 200)

    def test_oldest_session_evicted_when_the_table_is_full(self):
        first = self._login()
        for _ in range(_MAX_SESSIONS - 1):
            self._login()
        self.assertEqual(self._with_cookie(first).get("/ui").status_code, 200)
        self._login()  # one past the cap — the oldest goes
        self.assertEqual(self._with_cookie(first).get("/ui").status_code, 303)

    def test_expired_session_rejected(self):
        cookie = self._login()
        client = self._with_cookie(cookie)
        self.assertEqual(client.get("/ui").status_code, 200)
        later = time.monotonic() + _SESSION_MAX_AGE + 1
        with patch.object(webui, "time") as clock:
            clock.monotonic.return_value = later
            self.assertEqual(client.get("/ui").status_code, 303)
        # The expired lookup drops the id, so it stays rejected afterwards.
        self.assertEqual(client.get("/ui").status_code, 303)

    def _with_cookie(self, value):
        client = TestClient(self.app, follow_redirects=False)
        client.cookies.set(COOKIE_NAME, value)
        return client


class TestSecurityHeaders(WebUITestBase):
    """Every HTML page carries the same policy; the other responses do not
    pretend to be documents."""

    CSP = "default-src 'none'"

    def setUp(self):
        super().setUp()
        self._auth()

    def _assert_html_headers(self, resp):
        self.assertIn(self.CSP, resp.headers["content-security-policy"])
        self.assertEqual(resp.headers["x-content-type-options"], "nosniff")
        self.assertEqual(resp.headers["referrer-policy"], "no-referrer")

    def test_every_html_page_carries_the_headers(self):
        paths = [
            "/ui",
            "/ui/ns/proj",
            f"/ui/item/document/{self.ids['document']}",
            "/ui/settings",
            "/ui/item/widget/x",          # 404, unknown type
            "/ui/item/document/nope",     # 404, missing item
        ]
        for path in paths:
            with self.subTest(path=path):
                self._assert_html_headers(self.client.get(path))

    def test_login_pages_carry_the_headers(self):
        fresh = TestClient(self.app, follow_redirects=False)
        self._assert_html_headers(fresh.get("/ui/login"))
        self._assert_html_headers(fresh.post("/ui/login", data={"token": "nope"}))

    def test_throttled_login_keeps_both_the_headers_and_retry_after(self):
        fresh = TestClient(self.app, follow_redirects=False)
        for _ in range(5):
            fresh.post("/ui/login", data={"token": "nope"})
        resp = fresh.post("/ui/login", data={"token": TOKEN})
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Retry-After", resp.headers)
        self._assert_html_headers(resp)

    def test_policy_forbids_scripts_and_framing(self):
        csp = self.client.get("/ui").headers["content-security-policy"]
        # No script-src directive is needed to block scripts: default-src 'none'
        # covers them, and nothing relaxes it.
        self.assertNotIn("script-src", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("form-action 'self'", csp)

    def test_stylesheet_gets_nosniff_but_no_policy(self):
        resp = self.client.get("/ui/static/bootstrap.min.css")
        self.assertEqual(resp.headers["x-content-type-options"], "nosniff")
        self.assertNotIn("content-security-policy", resp.headers)

    def test_export_download_is_not_an_html_page(self):
        resp = self.client.get("/ui/export")
        self.assertEqual(resp.headers["content-type"], "application/zip")
        self.assertNotIn("content-security-policy", resp.headers)


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

    def test_settings_page_renders_configuration(self):
        resp = self.client.get("/ui/settings")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("built-in ONNX (test-model)", resp.text)
        self.assertIn("test-model", resp.text)
        self.assertIn("512 tokens", resp.text)
        # Query prefix is shown quoted so the trailing space is visible; an
        # empty doc prefix renders as a placeholder, not an empty string.
        self.assertIn("&quot;q: &quot;", resp.text)
        self.assertIn("(none)", resp.text)
        self.assertNotIn("alert-warning", resp.text)

    def test_settings_page_links_model_to_hf(self):
        resp = self.client.get("/ui/settings")
        self.assertIn(
            '<a href="https://huggingface.co/test-org/test-model" target="_blank" '
            'rel="noopener">test-model</a>',
            resp.text,
        )

    def test_settings_page_unknown_model_has_no_link(self):
        self.settings_info["model_url"] = None
        resp = self.client.get("/ui/settings")
        self.assertIn("test-model", resp.text)
        self.assertNotIn("huggingface.co", resp.text)

    def test_settings_page_flags_dimension_mismatch(self):
        self.settings_info["dim_database"] = 768
        resp = self.client.get("/ui/settings")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("alert-warning", resp.text)
        self.assertIn("MNEMOMATIC_REINDEX=auto", resp.text)

    def test_settings_page_flags_model_mismatch(self):
        self.settings_info["model_database"] = "a-different-model"
        resp = self.client.get("/ui/settings")
        self.assertIn("alert-warning", resp.text)
        self.assertIn("a-different-model", resp.text)
        # The reason a mismatch matters: it does not announce itself.
        self.assertIn("wrong results with no", resp.text)

    def test_settings_page_names_the_model_that_built_the_index(self):
        self.settings_info["model_database"] = "test-model"
        resp = self.client.get("/ui/settings")
        self.assertIn("Index built by model", resp.text)
        self.assertNotIn("alert-warning", resp.text)

    def test_settings_page_says_when_the_index_model_was_never_recorded(self):
        self.settings_info["model_database"] = None
        resp = self.client.get("/ui/settings")
        self.assertIn("not recorded", resp.text)
        self.assertNotIn("alert-warning", resp.text)

    def test_settings_page_shows_external_endpoint(self):
        self.settings_info.pop("max_tokens")
        self.settings_info["endpoint_url"] = "http://embed-host:8181/v1/embeddings"
        self.settings_info["wire_api"] = "openai"
        resp = self.client.get("/ui/settings")
        self.assertIn("http://embed-host:8181/v1/embeddings", resp.text)
        self.assertIn("openai", resp.text)
        self.assertNotIn("Token truncation limit", resp.text)

    def test_settings_page_without_provider_renders_placeholders(self):
        # A viewer registered without settings_info (old call signature) must not error.
        app = Starlette()
        register_webui(app, lambda: self.db, TOKEN)
        client = TestClient(app, follow_redirects=False)
        client.post("/ui/login", data={"token": TOKEN})
        resp = client.get("/ui/settings")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Embedding mode", resp.text)
        # Without make_export there is no Export section and no download.
        self.assertNotIn("/ui/export", resp.text)
        self.assertEqual(client.get("/ui/export").status_code, 404)

    def test_settings_page_offers_export_download(self):
        resp = self.client.get("/ui/settings")
        self.assertIn('href="/ui/export"', resp.text)

    def test_export_download(self):
        resp = self.client.get("/ui/export")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/zip")
        self.assertIn('filename="test-export.zip"', resp.headers["content-disposition"])
        self.assertEqual(resp.content[:2], b"PK")

    def test_navbar_links_settings_page(self):
        resp = self.client.get("/ui")
        self.assertIn('href="/ui/settings"', resp.text)
        # The login page (unauthenticated chrome) must not advertise it.
        self.client.cookies.clear()
        self.assertNotIn('href="/ui/settings"', self.client.get("/ui/login").text)

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
