"""Serving the browser: the single-page app, the setup page, the CA download,
and the gate that moves clients from plain HTTP to HTTPS once it is confirmed.

The SPA is a Vite build in ``static/app/`` next to this file (produced by
``npm --prefix web run build``; shipped inside the wheel). Hashed assets
under ``/assets`` are immutable forever; ``index.html`` is never cached and
is the answer for every unknown path, so the client router owns the URL
space — except for the reserved prefixes (``/api``, ``/mcp``, …) and for
requests that do not want HTML at all: an MCP client probing
``/.well-known/oauth-*`` after a 401 must get a 404, not a page.
"""

import html
import logging
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("mnemomatic")

APP_DIR = Path(__file__).parent / "static" / "app"

# First path segments that are never the SPA's to answer.
RESERVED = frozenset({"api", "mcp", "health", "export", "setup", "ca.crt", ".well-known", "assets"})

# Plain-HTTP paths that stay reachable once HTTPS is enforced: enough to find
# out that HTTPS is required and to get the CA that makes it work.
_PLAIN_ALLOWED = frozenset({"/health", "/setup", "/ca.crt", "/api/instance-id"})


# ── Small ASGI wrappers ─────────────────────────────────────────────────────

class CacheControl:
    """Add one Cache-Control value to every response of the wrapped app."""

    def __init__(self, app: ASGIApp, value: str):
        self.app = app
        self._value = value.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def guarded(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                headers.append((b"cache-control", self._value))
                message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, guarded)


class ListenerTag:
    """Mark each request with which listener it came in on, so the gate
    below can tell plain from TLS without trusting any header."""

    def __init__(self, app: ASGIApp, tag: str):
        self.app = app
        self.tag = tag

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["listener"] = self.tag
        await self.app(scope, receive, send)


class PlainPortGate:
    """Once HTTPS is confirmed, the plain port stops serving the application.

    Liveness, the setup page, the CA, and the instance id stay (they are how
    a client gets onto HTTPS). API and MCP calls get a 403 naming the HTTPS
    URL — never a redirect, which would silently replay a credential over
    plain HTTP. Everything else goes to /setup.
    """

    def __init__(self, app: ASGIApp, tls_state):
        self.app = app
        self._tls = tls_state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] != "http" or self._tls is None
                or scope.get("state", {}).get("listener") != "plain"
                or not self._tls.active()):
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path in _PLAIN_ALLOWED:
            await self.app(scope, receive, send)
            return
        https_url = self._tls.https_url() or ""
        if path.startswith(("/api", "/mcp", "/export")):
            response = JSONResponse(
                {"error": "https_required", "details": f"Use {https_url}{path} — plain HTTP is closed.",
                 "https_url": https_url},
                status_code=403, headers={"Cache-Control": "no-store"},
            )
        else:
            response = RedirectResponse("/setup", status_code=302)
        await response(scope, receive, send)


# ── /setup and /ca.crt ──────────────────────────────────────────────────────

_SETUP_CSS = """
:root{color-scheme:dark}body{margin:0;background:#160F26;color:#F1ECFA;font:15px/1.55 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:48px 24px}h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px;color:#C9BEE3}
.sub{color:#9A8CBF;margin:0 0 28px}.card{background:#241A40;border:1px solid #3A2C5E;border-radius:8px;padding:18px 20px;margin:12px 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}pre{background:#2C2050;border:1px solid #3A2C5E;border-radius:6px;padding:12px 14px;overflow-x:auto;margin:8px 0 0}
a{color:#B9A3E6}a.btn{display:inline-block;background:#422B6F;color:#fff;text-decoration:none;padding:10px 16px;border-radius:6px;border:1px solid #5B3F94;font-weight:600}
.fp{word-break:break-all;background:#2C2050;border-radius:6px;padding:10px 12px;display:block;margin-top:8px}ol{padding-left:22px}li{margin:6px 0}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:28px;color:#9A8CBF;font-size:13px}.mark{background:#422B6F;color:#fff;font-weight:800;padding:4px 10px;border-radius:5px;letter-spacing:1px}
"""


def _setup_html(status: dict) -> str:
    state = status.get("state")
    name = html.escape(status.get("name") or "")
    url = html.escape(status.get("https_url") or "")
    fp = html.escape(status.get("ca_fingerprint") or "")
    head = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Mnem-O-matic — HTTPS setup</title><style>{_SETUP_CSS}</style><main>"
            f"<div class=brand><span class=mark>ICS</span><span>Mnem-O-matic · Boston AI, a Division of ICS</span></div>")
    if state in ("unconfigured", "off") or not status.get("ca_fingerprint"):
        body = ("<h1>HTTPS is not set up yet</h1><p class=sub>Plain HTTP is still serving everything.</p>"
                "<div class=card>An administrator can enable HTTPS from the web UI under "
                "<b>Admin → HTTPS</b>. Once confirmed, this page explains how to trust the server's certificate.</div>"
                "<p><a class=btn href='/'>Open the web UI</a></p>")
        if state == "external":
            body = ("<h1>HTTPS is served with your own certificate</h1><p class=sub>No private CA to install.</p>"
                    f"<p><a class=btn href='{url}/'>Open {url}</a></p>")
        return head + body + "</main>"
    verb = "is required" if state == "active" else "is almost ready"
    body = f"""
<h1>HTTPS {verb}</h1>
<p class=sub>This server has its own certificate authority, valid for <code>{name}</code> only. Trust it once per device, then use <a href="{url}/">{url}</a>.</p>
<div class=card><b>1. Download the CA certificate</b><br>
<p><a class=btn href="/ca.crt" download="mnemomatic-ca.crt">Download mnemomatic-ca.crt</a></p>
Check its SHA-256 fingerprint before trusting it:<code class=fp>{fp}</code>
<pre>openssl x509 -in mnemomatic-ca.crt -noout -fingerprint -sha256</pre></div>
<div class=card><b>2. Trust it</b>
<h2>macOS</h2><pre>sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain mnemomatic-ca.crt</pre>
<h2>Windows (PowerShell, as administrator)</h2><pre>Import-Certificate -FilePath mnemomatic-ca.crt -CertStoreLocation Cert:\\LocalMachine\\Root</pre>
<h2>Debian / Ubuntu</h2><pre>sudo cp mnemomatic-ca.crt /usr/local/share/ca-certificates/ &amp;&amp; sudo update-ca-certificates</pre>
<h2>Fedora / RHEL</h2><pre>sudo cp mnemomatic-ca.crt /etc/pki/ca-trust/source/anchors/ &amp;&amp; sudo update-ca-trust</pre>
<h2>Firefox</h2><p>Firefox keeps its own store: Settings → Privacy &amp; Security → Certificates → View Certificates → Authorities → Import.</p>
<h2>Tools that ignore the system store</h2>
<pre>export NODE_EXTRA_CA_CERTS=$PWD/mnemomatic-ca.crt     # Claude Code, Claude Desktop, Cursor, other Node apps
export SSL_CERT_FILE=$PWD/mnemomatic-ca.crt           # Python, including mnemomatic-cli (or --ca-cert)
export REQUESTS_CA_BUNDLE=$PWD/mnemomatic-ca.crt      # Python requests
curl --cacert mnemomatic-ca.crt {url}/health</pre></div>
<div class=card><b>3. Switch your clients</b><br>Point them at <code>{url}/mcp</code>. Tokens are unchanged; only the URL moves.</div>
<p><a class=btn href="{url}/">Open {url}</a></p>
"""
    return head + body + "</main>"


def build_tls_routes(tls_state) -> list[Route]:
    """/setup and /ca.crt. Both are public and served on both listeners."""

    async def setup(request: Request) -> Response:
        status = tls_state.status() if tls_state is not None else {"state": "off"}
        return HTMLResponse(_setup_html(status), headers={"Cache-Control": "no-store"})

    async def ca(request: Request) -> Response:
        pem = tls_state.ca_pem() if tls_state is not None else None
        if pem is None:
            return JSONResponse({"error": "not_found", "details": "No built-in CA is configured."}, status_code=404)
        return Response(pem, media_type="application/x-pem-file",
                        headers={"Content-Disposition": 'attachment; filename="mnemomatic-ca.crt"',
                                 "Cache-Control": "no-store"})

    return [Route("/setup", setup, methods=["GET"]), Route("/ca.crt", ca, methods=["GET"])]


# ── The SPA ─────────────────────────────────────────────────────────────────

def build_spa_routes(app_dir: Path = APP_DIR) -> list:
    """Routes for the built front end. Append them after every other route:
    the last one is a catch-all."""
    app_dir = Path(app_dir)
    index_path = app_dir / "index.html"
    if not index_path.exists():
        logger.warning("Web UI build not found at %s — run `npm --prefix web run build`", app_dir)

    def wants_html(request: Request) -> bool:
        accept = request.headers.get("accept", "")
        return "text/html" in accept or "*/*" in accept and "application/json" not in accept

    async def index(request: Request) -> Response:
        path = request.path_params.get("path", "")
        first = path.split("/", 1)[0]
        if first in RESERVED or not wants_html(request):
            return JSONResponse({"error": "not_found", "details": "No such endpoint."}, status_code=404,
                                headers={"Cache-Control": "no-store"})
        # Root-level files from the build (favicon, logos): exact names only,
        # directly inside the app directory, never a subpath.
        if path and "/" not in path and not path.startswith("."):
            candidate = app_dir / path
            if candidate.is_file() and candidate.parent.resolve() == app_dir.resolve():
                return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
        if not index_path.exists():
            return PlainTextResponse(
                "The web UI has not been built. Run `npm --prefix web run build` (or use the Docker image).",
                status_code=503, headers={"Cache-Control": "no-store"},
            )
        return FileResponse(index_path, media_type="text/html", headers={"Cache-Control": "no-cache"})

    async def no_assets(scope, receive, send):
        await JSONResponse({"error": "not_found", "details": "No such asset."}, status_code=404)(scope, receive, send)

    assets_dir = app_dir / "assets"
    # StaticFiles insists on an existing directory (it checks on first use);
    # a tree without a build gets a plain 404 app so the 503 below can explain.
    assets = StaticFiles(directory=assets_dir) if assets_dir.is_dir() else no_assets

    return [
        Mount("/assets", app=CacheControl(assets, "public, max-age=31536000, immutable")),
        Route("/", index, methods=["GET", "HEAD"]),
        Route("/{path:path}", index, methods=["GET", "HEAD"]),
    ]
