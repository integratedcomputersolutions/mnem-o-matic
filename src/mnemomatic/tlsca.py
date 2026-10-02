"""Built-in TLS: a private CA, a server certificate, and the state machine
that decides when HTTPS becomes the required entry point.

Out of the box the server listens on plain HTTP. An administrator names the
host (``memory.example`` — a DNS name, never an IP), which mints a CA scoped
by name constraints to exactly that name and a server certificate under it,
and starts the HTTPS listener. HTTPS is only *enforced* after the admin's
browser has fetched this process's instance id from the new HTTPS origin and
posted it back: proof that the name resolves to this server and not to some
other machine on the LAN. From then on the plain port serves only what is
needed to get onto HTTPS — the setup page, the CA download, the liveness
probe — and tells API and MCP callers where to go.

The CA's name constraints matter: a client that trusts this CA is trusting
it for one hostname only. A leaked ``ca.key`` cannot sign a certificate the
client would accept for anything else, and IP literals are excluded
outright, which is also why the name may not be an address.

An operator who already has a certificate drops ``custom.crt`` and
``custom.key`` into the TLS directory; they win over the built-in pair, no
CA is offered, and HTTPS counts as active at once.

Certificates are ECDSA P-256. The CA lives ten years, the leaf 397 days and
is reissued when under thirty remain — picked up by live connections
through the SNI callback without a restart.
"""

import ipaddress
import logging
import os
import re
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger("mnemomatic")

CA_LIFETIME = timedelta(days=3650)
LEAF_LIFETIME = timedelta(days=397)
RENEW_BEFORE = timedelta(days=30)
RENEWAL_CHECK_SECONDS = 24 * 3600

CA_CERT, CA_KEY = "ca.crt", "ca.key"
LEAF_CERT, LEAF_KEY = "leaf.crt", "leaf.key"
CUSTOM_CERT, CUSTOM_KEY = "custom.crt", "custom.key"
SETTINGS_KEY = "https"

_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")


class TlsError(Exception):
    """A refused TLS operation, carrying the API's error code and status."""

    def __init__(self, code: str, status: int, details: str):
        super().__init__(details)
        self.code, self.status, self.details = code, status, details


def validate_name(raw: str) -> str:
    """A lowercase DNS name. Single labels (LAN hostnames) are fine; IP
    literals are not, because the CA cannot vouch for them and the browser
    would not accept it if it did."""
    name = (raw or "").strip().lower().rstrip(".")
    if not name or len(name) > 253:
        raise TlsError("invalid_name", 400, "Enter the DNS name clients will use, e.g. memory.example.")
    try:
        ipaddress.ip_address(name.strip("[]"))
    except ValueError:
        pass
    else:
        raise TlsError("invalid_name", 400,
                       "Use a DNS name, not an IP address — the certificate is bound to a name.")
    if not all(_LABEL.match(label) for label in name.split(".")):
        raise TlsError("invalid_name", 400, "That is not a valid DNS name.")
    return name


def fingerprint(cert: x509.Certificate) -> str:
    return ":".join(f"{b:02X}" for b in cert.fingerprint(hashes.SHA256()))


# ── Certificate generation ──────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_private(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _write_public(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _pem_key(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def make_ca(name: str) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"Mnem-O-matic CA ({name})"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Mnem-O-matic"),
    ])
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + CA_LIFETIME)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
            encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.NameConstraints(
            permitted_subtrees=[x509.DNSName(name)],
            excluded_subtrees=[x509.IPAddress(ipaddress.ip_network("0.0.0.0/0")),
                               x509.IPAddress(ipaddress.ip_network("::/0"))]), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return cert, key


def make_leaf(name: str, ca_cert: x509.Certificate, ca_key: ec.EllipticCurvePrivateKey,
              lifetime: timedelta = LEAF_LIFETIME) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + lifetime)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False,
            encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    return cert, key


def _load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _load_key(path: Path):
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def _ca_name(ca_cert: x509.Certificate) -> str | None:
    """The one DNS name a built-in CA is constrained to."""
    try:
        nc = ca_cert.extensions.get_extension_for_class(x509.NameConstraints).value
    except x509.ExtensionNotFound:
        return None
    for sub in nc.permitted_subtrees or []:
        if isinstance(sub, x509.DNSName):
            return sub.value
    return None


# ── Live certificate holder ─────────────────────────────────────────────────

class CertHolder:
    """The SSLContext uvicorn serves from, swappable without a restart.

    uvicorn asks the factory once, at startup, for a context. We hand it a
    front context whose SNI callback redirects every handshake to whatever
    the current context is; renewing means building a new current context
    and (for the rare SNI-less client) reloading the front's chain as well.
    """

    def __init__(self):
        self._front: ssl.SSLContext | None = None
        self._current: ssl.SSLContext | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._current is not None

    @staticmethod
    def _context(certfile: Path, keyfile: Path) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(str(certfile), str(keyfile))
        return ctx

    def load(self, certfile: Path, keyfile: Path) -> None:
        with self._lock:
            self._current = self._context(certfile, keyfile)
            if self._front is None:
                front = self._context(certfile, keyfile)
                front.sni_callback = self._sni
                self._front = front
            else:
                self._front.load_cert_chain(str(certfile), str(keyfile))

    def _sni(self, sslobj, server_name, context):
        sslobj.context = self._current

    def factory(self, config=None, default_factory=None) -> ssl.SSLContext:
        """uvicorn's `ssl_context_factory` hook."""
        if self._front is None:
            raise RuntimeError("no certificate loaded")
        return self._front


# ── State machine ───────────────────────────────────────────────────────────

class TlsState:
    """What HTTPS is doing right now, backed by the files in `tls_dir` and
    the `https` row in the settings table.

    States reported by status():
      unconfigured  no name yet (or disabled); plain HTTP serves everything
      pending       name set, certs issued, HTTPS listening, not yet confirmed
      active        confirmed; plain HTTP only hands clients over to HTTPS
      external      custom.crt/custom.key present; active, no CA to offer
    """

    def __init__(self, db_getter, tls_dir: Path, https_port: int, *, public_host: str = "",
                 on_ready=None):
        self._db = db_getter
        self.dir = Path(tls_dir)
        self.port = https_port
        self.public_host = public_host
        self.on_ready = on_ready          # called when a listener should (re)start
        self.holder = CertHolder()
        self._lock = threading.Lock()

    # ── persisted record ──

    def _record(self) -> dict:
        import json
        raw = self._db().get_setting(SETTINGS_KEY)
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {}
        return {"name": data.get("name") or None, "confirmed": bool(data.get("confirmed"))}

    def _save(self, name: str | None, confirmed: bool) -> None:
        import json
        self._db().set_setting(SETTINGS_KEY, json.dumps({"name": name, "confirmed": confirmed}))

    # ── files ──

    def _p(self, filename: str) -> Path:
        return self.dir / filename

    def custom(self) -> bool:
        return self._p(CUSTOM_CERT).exists() and self._p(CUSTOM_KEY).exists()

    def _builtin_ready(self) -> bool:
        return all(self._p(f).exists() for f in (CA_CERT, CA_KEY, LEAF_CERT, LEAF_KEY))

    def ca_cert(self) -> x509.Certificate | None:
        if self.custom() or not self._p(CA_CERT).exists():
            return None
        return _load_cert(self._p(CA_CERT))

    def ca_pem(self) -> bytes | None:
        if self.custom() or not self._p(CA_CERT).exists():
            return None
        return self._p(CA_CERT).read_bytes()

    def _leaf(self) -> x509.Certificate | None:
        path = self._p(CUSTOM_CERT) if self.custom() else self._p(LEAF_CERT)
        return _load_cert(path) if path.exists() else None

    # ── derived state ──

    @property
    def name(self) -> str | None:
        if self.custom():
            leaf = self._leaf()
            if leaf is not None:
                try:
                    san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
                    names = san.get_values_for_type(x509.DNSName)
                    if names:
                        return names[0]
                except x509.ExtensionNotFound:
                    pass
                cn = leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
                return cn[0].value if cn else None
        return self._record()["name"]

    def state(self) -> str:
        if self.custom():
            return "external"
        rec = self._record()
        if not rec["name"] or not self._builtin_ready():
            return "unconfigured"
        ca = self.ca_cert()
        if rec["confirmed"] and ca is not None and _ca_name(ca) == rec["name"]:
            return "active"
        return "pending"

    def active(self) -> bool:
        return self.state() in ("active", "external")

    def https_url(self) -> str | None:
        name = self.name
        if not name or self.state() == "unconfigured":
            return None
        port = "" if self.port == 443 else f":{self.port}"
        return f"https://{name}{port}"

    def pending_origin(self) -> str | None:
        """The HTTPS origin the browser may contact during confirmation."""
        return self.https_url() if self.state() == "pending" else None

    def status(self) -> dict:
        state = self.state()
        ca = self.ca_cert()
        leaf = self._leaf() if state != "unconfigured" else None
        return {
            "state": state,
            "name": self.name,
            "https_url": self.https_url(),
            "https_port": self.port,
            "ca_fingerprint": fingerprint(ca) if ca is not None else None,
            "leaf_not_after": leaf.not_valid_after_utc.isoformat() if leaf is not None else None,
            "custom": self.custom(),
        }

    # ── transitions ──

    def _load_into_holder(self) -> None:
        if self.custom():
            self.holder.load(self._p(CUSTOM_CERT), self._p(CUSTOM_KEY))
        elif self._builtin_ready():
            self.holder.load(self._p(LEAF_CERT), self._p(LEAF_KEY))

    def bootstrap(self) -> None:
        """At startup: pick up existing certificates, seed the name from
        MNEMOMATIC_PUBLIC_HOST when nothing is configured yet, log the CA
        fingerprint so it can be checked against the download."""
        if self.custom():
            self._load_into_holder()
            logger.info("TLS: using custom certificate from %s (name %s)", self.dir, self.name)
            return
        rec = self._record()
        if not rec["name"] and self.public_host:
            try:
                self.set_name(self.public_host, notify=False)
            except TlsError as e:
                logger.error("MNEMOMATIC_PUBLIC_HOST rejected: %s", e.details)
            return
        if rec["name"] and self._builtin_ready():
            self._load_into_holder()
            ca = self.ca_cert()
            logger.info("TLS: %s for %s — CA fingerprint %s", self.state(), rec["name"],
                        fingerprint(ca) if ca else "?")
        else:
            logger.info("TLS: not configured yet — name the host on the HTTPS page to enable it")

    def set_name(self, raw_name: str, *, notify: bool = True) -> dict:
        """Issue (or re-issue) certificates for `raw_name` and start listening.
        A changed name means a new CA — the old one's constraints would not
        cover it — and a return to the pending state."""
        if self.custom():
            raise TlsError("external_certificate", 409,
                           "A custom certificate is in use; remove custom.crt/custom.key to use the built-in CA.")
        name = validate_name(raw_name)
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.dir, 0o700)
            ca = self.ca_cert()
            if ca is None or _ca_name(ca) != name or not self._p(CA_KEY).exists():
                ca, ca_key = make_ca(name)
                _write_private(self._p(CA_KEY), _pem_key(ca_key))
                _write_public(self._p(CA_CERT), ca.public_bytes(serialization.Encoding.PEM))
                logger.info("TLS: new CA for %s — fingerprint %s", name, fingerprint(ca))
            else:
                ca_key = _load_key(self._p(CA_KEY))
            self._issue_leaf(name, ca, ca_key)
            confirmed = self._record()["confirmed"] and self._record()["name"] == name
            self._save(name, confirmed)
            self._load_into_holder()
        if notify and self.on_ready:
            self.on_ready()
        return self.status()

    def _issue_leaf(self, name: str, ca: x509.Certificate, ca_key) -> None:
        leaf, key = make_leaf(name, ca, ca_key)
        _write_private(self._p(LEAF_KEY), _pem_key(key))
        _write_public(self._p(LEAF_CERT),
                      leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM))
        logger.info("TLS: server certificate for %s valid until %s", name, leaf.not_valid_after_utc.date())

    def confirm(self, raw_name: str, instance_id: str, expected_id: str) -> dict:
        """The browser reached https://<name> and read our instance id back."""
        if self.state() != "pending":
            raise TlsError("not_pending", 409, "There is no HTTPS name awaiting confirmation.")
        name = validate_name(raw_name)
        if name != self._record()["name"]:
            raise TlsError("name_mismatch", 400, "That is not the name currently pending.")
        if not instance_id or instance_id != expected_id:
            raise TlsError("instance_mismatch", 400,
                           "That name reaches a different server. Check DNS and try again.")
        self._save(name, True)
        logger.info("TLS: HTTPS confirmed for %s — plain HTTP now redirects", name)
        return self.status()

    def disable(self) -> dict:
        """Back to pending: plain HTTP serves everything again. Certificates
        stay, so re-confirming is one click."""
        if self.custom():
            raise TlsError("external_certificate", 409, "Remove custom.crt/custom.key to turn HTTPS off.")
        rec = self._record()
        if rec["name"]:
            self._save(rec["name"], False)
        return self.status()

    def renew_if_needed(self, now: datetime | None = None) -> bool:
        """Reissue the server certificate when it is close to expiry. True
        when a new one was written and loaded."""
        if self.custom() or not self._builtin_ready():
            return False
        leaf = self._leaf()
        if leaf is None or leaf.not_valid_after_utc - (now or _now()) > RENEW_BEFORE:
            return False
        name = self._record()["name"]
        if not name:
            return False
        with self._lock:
            self._issue_leaf(name, _load_cert(self._p(CA_CERT)), _load_key(self._p(CA_KEY)))
            self._load_into_holder()
        logger.info("TLS: renewed the server certificate for %s", name)
        return True


def start_renewal_thread(state: TlsState, interval: float = RENEWAL_CHECK_SECONDS) -> threading.Thread:
    def loop():
        while True:
            time.sleep(interval)
            try:
                state.renew_if_needed()
            except Exception as e:  # keep checking tomorrow
                logger.warning("TLS renewal check failed: %s: %s", type(e).__name__, e)

    thread = threading.Thread(target=loop, name="tls-renewal", daemon=True)
    thread.start()
    return thread
