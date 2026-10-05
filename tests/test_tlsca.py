"""Tests for the built-in CA (mnemomatic.tlsca) and the plain-port gate."""

import json
import ssl
import stat
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic import tlsca
from mnemomatic.db import Database
from mnemomatic.spa import ListenerTag, PlainPortGate
from mnemomatic.tlsca import CertHolder, TlsError, TlsState, fingerprint, validate_name


class TestValidateName(unittest.TestCase):
    def test_accepts_dns_names(self):
        self.assertEqual(validate_name(" Memory.Example. "), "memory.example")
        self.assertEqual(validate_name("ariosto"), "ariosto")
        self.assertEqual(validate_name("a-b.c0"), "a-b.c0")

    def test_rejects_ips_and_junk(self):
        for bad in ("", "192.168.0.114", "::1", "[::1]", "-bad.example", "bad-.example", "a b", "x" * 254, "under_score"):
            with self.subTest(bad=bad):
                with self.assertRaises(TlsError) as cm:
                    validate_name(bad)
                self.assertEqual(cm.exception.code, "invalid_name")


class TlsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db = Database(str(root / "m.db"))
        self.addCleanup(self.db.close)
        self.dir = root / "tls"
        self.ready = MagicMock()
        self.state = TlsState(lambda: self.db, self.dir, 8443, on_ready=self.ready)


class TestIssuance(TlsCase):
    def test_set_name_issues_constrained_ca_and_leaf(self):
        status = self.state.set_name("Memory.Example")
        self.assertEqual(status["state"], "pending")
        self.assertEqual(status["name"], "memory.example")
        self.assertEqual(status["https_url"], "https://memory.example:8443")
        self.assertRegex(status["ca_fingerprint"], r"^([0-9A-F]{2}:){31}[0-9A-F]{2}$")
        self.ready.assert_called_once()
        self.assertTrue(self.state.holder.ready)

        # Files and permissions.
        self.assertEqual(stat.S_IMODE(self.dir.stat().st_mode), 0o700)
        for name in ("ca.key", "leaf.key"):
            self.assertEqual(stat.S_IMODE((self.dir / name).stat().st_mode), 0o600)
        for name in ("ca.crt", "leaf.crt"):
            self.assertTrue((self.dir / name).exists())

        ca = x509.load_pem_x509_certificate((self.dir / "ca.crt").read_bytes())
        bc = ca.extensions.get_extension_for_class(x509.BasicConstraints)
        self.assertTrue(bc.value.ca)
        self.assertEqual(bc.value.path_length, 0)
        nc = ca.extensions.get_extension_for_class(x509.NameConstraints)
        self.assertTrue(nc.critical)
        self.assertEqual([s.value for s in nc.value.permitted_subtrees], ["memory.example"])
        self.assertEqual({str(s.value) for s in nc.value.excluded_subtrees}, {"0.0.0.0/0", "::/0"})
        self.assertAlmostEqual((ca.not_valid_after_utc - datetime.now(timezone.utc)).days, 3650, delta=2)
        self.assertEqual(fingerprint(ca), status["ca_fingerprint"])

        leaf_pem = (self.dir / "leaf.crt").read_bytes()
        chain = x509.load_pem_x509_certificates(leaf_pem)
        self.assertEqual(len(chain), 2)                     # leaf then CA
        leaf = chain[0]
        san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertEqual(san.get_values_for_type(x509.DNSName), ["memory.example"])
        eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        self.assertIn(ExtendedKeyUsageOID.SERVER_AUTH, eku)
        self.assertEqual(leaf.issuer, ca.subject)
        self.assertAlmostEqual((leaf.not_valid_after_utc - datetime.now(timezone.utc)).days, 397, delta=2)
        self.assertFalse(leaf.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)

    def test_same_name_keeps_ca_new_name_replaces_it(self):
        fp1 = self.state.set_name("memory.example")["ca_fingerprint"]
        fp2 = self.state.set_name("memory.example")["ca_fingerprint"]
        self.assertEqual(fp1, fp2)
        fp3 = self.state.set_name("other.example")["ca_fingerprint"]
        self.assertNotEqual(fp1, fp3)
        self.assertEqual(self.state.state(), "pending")

    def test_rejects_ip_and_leaves_nothing_behind(self):
        with self.assertRaises(TlsError):
            self.state.set_name("10.0.0.5")
        self.assertFalse(self.dir.exists())
        self.assertEqual(self.state.state(), "unconfigured")
        self.assertIsNone(self.state.https_url())

    def test_default_port_omitted_from_url(self):
        state = TlsState(lambda: self.db, self.dir, 443)
        self.assertEqual(state.set_name("memory.example")["https_url"], "https://memory.example")


class TestStateMachine(TlsCase):
    def test_unconfigured(self):
        status = self.state.status()
        self.assertEqual(status["state"], "unconfigured")
        self.assertIsNone(status["name"])
        self.assertIsNone(status["ca_fingerprint"])
        self.assertFalse(self.state.active())
        self.assertIsNone(self.state.pending_origin())
        self.assertIsNone(self.state.ca_pem())

    def test_confirm_flow(self):
        self.state.set_name("memory.example")
        self.assertEqual(self.state.pending_origin(), "https://memory.example:8443")
        with self.assertRaises(TlsError) as cm:
            self.state.confirm("memory.example", "wrong", "right")
        self.assertEqual(cm.exception.code, "instance_mismatch")
        with self.assertRaises(TlsError) as cm:
            self.state.confirm("other.example", "right", "right")
        self.assertEqual(cm.exception.code, "name_mismatch")
        self.assertFalse(self.state.active())

        status = self.state.confirm("MEMORY.example", "right", "right")
        self.assertEqual(status["state"], "active")
        self.assertTrue(self.state.active())
        self.assertIsNone(self.state.pending_origin())
        self.assertEqual(json.loads(self.db.get_setting("https")), {"name": "memory.example", "confirmed": True})

        with self.assertRaises(TlsError) as cm:
            self.state.confirm("memory.example", "right", "right")
        self.assertEqual(cm.exception.code, "not_pending")

    def test_disable_and_reconfirm(self):
        self.state.set_name("memory.example")
        self.state.confirm("memory.example", "id", "id")
        status = self.state.disable()
        self.assertEqual(status["state"], "pending")
        self.assertFalse(self.state.active())
        self.assertTrue((self.dir / "ca.crt").exists())
        self.state.confirm("memory.example", "id", "id")
        self.assertTrue(self.state.active())

    def test_renaming_after_confirm_goes_back_to_pending(self):
        self.state.set_name("memory.example")
        self.state.confirm("memory.example", "id", "id")
        self.state.set_name("other.example")
        self.assertEqual(self.state.state(), "pending")

    def test_bootstrap_picks_up_existing_and_seeds_public_host(self):
        self.state.set_name("memory.example")
        self.state.confirm("memory.example", "id", "id")
        again = TlsState(lambda: self.db, self.dir, 8443)
        again.bootstrap()
        self.assertEqual(again.state(), "active")
        self.assertTrue(again.holder.ready)

        fresh_dir = Path(self.tmp.name) / "tls2"
        db2 = Database(str(Path(self.tmp.name) / "m2.db"))
        self.addCleanup(db2.close)
        seeded = TlsState(lambda: db2, fresh_dir, 8443, public_host="seeded.example", on_ready=MagicMock())
        seeded.bootstrap()
        self.assertEqual(seeded.state(), "pending")
        self.assertEqual(seeded.name, "seeded.example")
        seeded.on_ready.assert_not_called()       # the listener starts from boot, not from a callback

        bad = TlsState(lambda: db2, fresh_dir, 8443, public_host="10.0.0.1")
        bad.bootstrap()                            # logs, does not raise

    def test_bootstrap_ignores_public_host_once_configured(self):
        self.state.set_name("memory.example")
        again = TlsState(lambda: self.db, self.dir, 8443, public_host="other.example")
        again.bootstrap()
        self.assertEqual(again.name, "memory.example")


class TestCustomCertificate(TlsCase):
    def setUp(self):
        super().setUp()
        self.state.set_name("memory.example")
        # Repurpose the built-in pair as a "custom" one for a different name.
        ca, ca_key = tlsca.make_ca("custom.example")
        leaf, key = tlsca.make_leaf("custom.example", ca, ca_key)
        from cryptography.hazmat.primitives import serialization
        (self.dir / "custom.crt").write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        (self.dir / "custom.key").write_bytes(tlsca._pem_key(key))

    def test_custom_wins(self):
        status = self.state.status()
        self.assertEqual(status["state"], "external")
        self.assertEqual(status["name"], "custom.example")
        self.assertTrue(status["custom"])
        self.assertIsNone(status["ca_fingerprint"])
        self.assertTrue(self.state.active())
        self.assertIsNone(self.state.ca_pem())
        self.assertEqual(self.state.https_url(), "https://custom.example:8443")
        with self.assertRaises(TlsError):
            self.state.set_name("whatever.example")
        with self.assertRaises(TlsError):
            self.state.disable()
        self.assertFalse(self.state.renew_if_needed())
        self.state.bootstrap()
        self.assertTrue(self.state.holder.ready)


class TestRenewal(TlsCase):
    def test_renews_only_when_close_to_expiry(self):
        self.state.set_name("memory.example")
        before = x509.load_pem_x509_certificates((self.dir / "leaf.crt").read_bytes())[0]
        self.assertFalse(self.state.renew_if_needed())
        soon = before.not_valid_after_utc - timedelta(days=10)
        self.assertTrue(self.state.renew_if_needed(now=soon))
        after = x509.load_pem_x509_certificates((self.dir / "leaf.crt").read_bytes())[0]
        self.assertNotEqual(before.serial_number, after.serial_number)
        self.assertEqual(after.issuer, before.issuer)
        self.assertFalse(self.state.renew_if_needed())


class TestCertHolder(TlsCase):
    def test_reload_swaps_the_served_context(self):
        self.state.set_name("memory.example")
        holder = self.state.holder
        front = holder.factory(None, None)
        self.assertIsInstance(front, ssl.SSLContext)
        self.assertEqual(front.minimum_version, ssl.TLSVersion.TLSv1_2)
        first = holder._current
        self.state.renew_if_needed(now=datetime.now(timezone.utc) + timedelta(days=380))
        self.assertIs(holder.factory(None, None), front)          # uvicorn keeps the same object
        self.assertIsNot(holder._current, first)                  # but handshakes get the new one
        sock = MagicMock()
        holder._sni(sock, "memory.example", front)
        self.assertIs(sock.context, holder._current)

    def test_factory_before_load_raises(self):
        with self.assertRaises(RuntimeError):
            CertHolder().factory()


class TestPlainPortGate(unittest.TestCase):
    """Behaviour per listener and TLS state, with a stand-in state object."""

    def _client(self, listener, active):
        state = MagicMock()
        state.active.return_value = active
        state.https_url.return_value = "https://memory.example:8443"

        async def ok(request):
            return PlainTextResponse("ok")

        routes = [Route(p, ok, methods=["GET", "POST"]) for p in
                  ("/", "/health", "/setup", "/ca.crt", "/api/instance-id", "/api/session", "/mcp", "/export",
                   "/browse")]
        # Same nesting as the server: the listener tag is outermost, the gate inside it.
        app = ListenerTag(PlainPortGate(Starlette(routes=routes), state), listener)
        return TestClient(app, follow_redirects=False)

    def test_inactive_passes_everything(self):
        c = self._client("plain", active=False)
        for path in ("/", "/api/session", "/mcp", "/export", "/browse"):
            self.assertEqual(c.get(path).status_code, 200, path)

    def test_tls_listener_unaffected(self):
        c = self._client("tls", active=True)
        for path in ("/", "/api/session", "/mcp", "/export"):
            self.assertEqual(c.get(path).status_code, 200, path)

    def test_active_plain_port(self):
        c = self._client("plain", active=True)
        for path in ("/health", "/setup", "/ca.crt", "/api/instance-id"):
            self.assertEqual(c.get(path).status_code, 200, path)
        for path in ("/api/session", "/mcp", "/export"):
            resp = c.post(path)
            self.assertEqual(resp.status_code, 403, path)
            self.assertEqual(resp.json()["error"], "https_required")
            self.assertEqual(resp.json()["https_url"], "https://memory.example:8443")
        for path in ("/", "/browse"):
            resp = c.get(path)
            self.assertEqual(resp.status_code, 302, path)
            self.assertEqual(resp.headers["location"], "/setup")

    def test_no_tls_state_passes_everything(self):
        app = ListenerTag(PlainPortGate(Starlette(routes=[Route("/mcp", lambda r: PlainTextResponse("ok"))]), None),
                          "plain")
        self.assertEqual(TestClient(app).get("/mcp").status_code, 200)


class TestSetupAndCaRoutes(TlsCase):
    def _client(self):
        from mnemomatic.spa import build_tls_routes
        return TestClient(Starlette(routes=build_tls_routes(self.state)))

    def test_unconfigured(self):
        c = self._client()
        resp = c.get("/setup")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("not set up yet", resp.text)
        self.assertEqual(c.get("/ca.crt").status_code, 404)

    def test_configured(self):
        self.state.set_name("memory.example")
        c = self._client()
        resp = c.get("/setup")
        self.assertIn("https://memory.example:8443", resp.text)
        self.assertIn(self.state.status()["ca_fingerprint"], resp.text)
        self.assertIn("NODE_EXTRA_CA_CERTS", resp.text)
        self.assertIn("almost ready", resp.text)
        self.state.confirm("memory.example", "id", "id")
        self.assertIn("is required", c.get("/setup").text)
        ca = c.get("/ca.crt")
        self.assertEqual(ca.status_code, 200)
        self.assertEqual(ca.headers["content-type"], "application/x-pem-file")
        self.assertIn('filename="mnemomatic-ca.crt"', ca.headers["content-disposition"])
        parsed = x509.load_pem_x509_certificate(ca.content)
        self.assertEqual(parsed.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value,
                         "Mnem-O-matic CA (memory.example)")

    def test_html_escapes_name(self):
        # Names are validated, but the page must not trust the record blindly.
        # The status must have a fingerprint, or the page never shows the name.
        from mnemomatic.spa import _setup_html
        page = _setup_html({"state": "pending", "name": "<b>x</b>", "ca_fingerprint": "AB:CD",
                            "https_url": "https://<i>y</i>:8443"})
        self.assertIn("valid for <code>&lt;b&gt;x&lt;/b&gt;</code>", page)
        self.assertNotIn("<b>x</b>", page)
        self.assertNotIn("<i>y</i>", page)


if __name__ == "__main__":
    unittest.main()
