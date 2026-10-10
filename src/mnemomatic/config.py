"""Runtime configuration, read once from the environment at import.

Everything the server is told at startup lives here, so a reader can see the
whole knob surface in one place and the modules that use it stay about their
own job.

Values are read at import, which makes them constants for the process. Modules
that consume a value tests replace — the thresholds and prefixes below — must
reach it through this module (`config.SIMILAR_THRESHOLD`) rather than binding
it at import time, or patching it will have no effect.
"""

import logging
import os

from mcp.types import ToolAnnotations

from mnemomatic import model_config

logger = logging.getLogger("mnemomatic")

DB_PATH = os.environ.get("MNEMOMATIC_DB_PATH", "/data/mnemomatic.db")
HOST = os.environ.get("MNEMOMATIC_HOST", "0.0.0.0")
PORT = int(os.environ.get("MNEMOMATIC_PORT", "8000"))
CORS_ORIGINS = os.environ.get("MNEMOMATIC_CORS_ORIGINS", "")
EMBED_URL = os.environ.get("MNEMOMATIC_EMBED_URL", "")
EMBED_MODEL = os.environ.get("MNEMOMATIC_EMBED_MODEL", "")
# The commit the image was built from (Docker build arg). Empty in local and
# source builds, which then report only the package version.
BUILD = os.environ.get("MNEMOMATIC_BUILD", "").strip() or None
MAX_SEARCH_LIMIT = 100
MAX_LIST_LIMIT = 200

# Trusted-proxy sign-in. A gateway that authenticates people itself can vouch
# for them: it sends this secret as the bearer token on /mcp and /export plus a
# header naming the user, and the user is created here on first sight. Off
# unless the secret is set. The mechanics are in auth.ProxyAuth.
PROXY_SECRET = os.environ.get("MNEMOMATIC_PROXY_SECRET", "").strip() or None
PROXY_SECRET_MIN_LEN = 32
PROXY_USER_HEADER = os.environ.get("MNEMOMATIC_PROXY_USER_HEADER", "").strip() or "X-Mnemomatic-User"
PROXY_NAME_HEADER = os.environ.get("MNEMOMATIC_PROXY_NAME_HEADER", "").strip() or None
PROXY_ROLE_HEADER = os.environ.get("MNEMOMATIC_PROXY_ROLE_HEADER", "").strip() or None

# 2.x credentials the server no longer reads. Left in a compose file from
# before the upgrade they would be silently ignored, which is worse than one
# line saying what replaced them.
for _stale, _replacement in (
    ("MNEMOMATIC_API_KEY", "agents use personal tokens (mnm_…) from the web UI, "
                           "or a trusted proxy vouches for users via MNEMOMATIC_PROXY_SECRET"),
    ("MNEMOMATIC_UI_TOKEN", "the web UI is behind a sign-in"),
):
    if os.environ.get(_stale, "").strip():
        logger.warning("%s is set but no longer used since 3.0 — %s", _stale, _replacement)

# First-run bootstrap. Only read while the users table is empty: it creates
# the initial `admin` without the setup-code dance, for compose files and CI.
# With users present it is ignored (and logged as such), never re-applied.
ADMIN_PASSWORD = os.environ.get("MNEMOMATIC_ADMIN_PASSWORD", "").strip() or None


# Built-in TLS. "auto" (default) runs the built-in CA and a second HTTPS
# listener once an admin names the host; "off" is for deployments that
# terminate TLS in their own reverse proxy (pair it with TRUSTED_PROXIES).
def _tls_mode() -> str:
    raw = os.environ.get("MNEMOMATIC_TLS", "auto").strip().lower()
    if raw in ("auto", ""):
        return "auto"
    if raw == "off":
        return "off"
    logger.warning("Ignoring unrecognised MNEMOMATIC_TLS=%r — expected 'auto' or 'off'; using 'auto'", raw)
    return "auto"


TLS_MODE = _tls_mode()
HTTPS_PORT = int(os.environ.get("MNEMOMATIC_HTTPS_PORT", "8443"))
# The DNS name the certificate must cover. Pre-seeds the HTTPS page; the admin
# still confirms from a browser before HTTPS becomes the required entry point.
PUBLIC_HOST = os.environ.get("MNEMOMATIC_PUBLIC_HOST", "").strip()
# Where the CA and server certificate live — beside the database by default,
# so the one mounted volume carries everything that must survive a restart.
TLS_DIR = os.environ.get("MNEMOMATIC_TLS_DIR", "").strip() or os.path.join(os.path.dirname(DB_PATH) or ".", "tls")

# Largest JSON body the browser API accepts. Nothing it takes is bigger than
# a login form or a token name; the MCP endpoint has its own, larger limit.
API_MAX_BODY = 1024 * 1024


def _trusted_proxies() -> list[str]:
    """Reverse proxies whose X-Forwarded-For / X-Forwarded-Proto are believed.

    Comma-separated IPs or CIDRs, or "*" when the server port is only reachable
    from the proxy (the shipped compose file). Empty — the default — means the
    socket peer is the client, which is right for direct connections; behind an
    untrusted proxy it means every client shares the proxy's address for
    throttling and the audit log. The list goes to uvicorn, which does the
    trust check itself.
    """
    raw = os.environ.get("MNEMOMATIC_TRUSTED_PROXIES", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


TRUSTED_PROXIES = _trusted_proxies()

# Largest request body the server will read. The biggest legitimate request is
# a store_document call at the model limits (100 KB of content plus 50 metadata
# values of 10 KB each), which stays well under 1 MB even after JSON escaping,
# so this leaves room to spare. Deliberately not configurable: the validation
# limits are the contract, and nothing legitimate needs a larger body.
MAX_BODY_BYTES = 4 * 1024 * 1024

# Scheduled backups of the export archive — disabled unless a directory is set.
BACKUP_DIR = os.environ.get("MNEMOMATIC_BACKUP_DIR", "").strip()
BACKUP_INTERVAL_HOURS = float(os.environ.get("MNEMOMATIC_BACKUP_INTERVAL", "24"))
BACKUP_KEEP = int(os.environ.get("MNEMOMATIC_BACKUP_KEEP", "7"))

# Cosine similarity at or above which two items count as near-duplicates —
# used by the `similar` field on store responses and by consolidation_report's
# clustering. Correct-but-distinct search hits typically score 0.3–0.6 across
# the bundled models, near-duplicates well above 0.8. 0 disables the store-time
# check entirely.
SIMILAR_THRESHOLD = float(os.environ.get("MNEMOMATIC_SIMILAR_THRESHOLD", "0.8"))
SIMILAR_LIMIT = 3

# Task prefixes for asymmetric embedding models. When the bundled model is
# trained with task prompts (e.g. EmbeddingGemma, multilingual-e5), the Docker
# build records them in model_config.json and they apply by default when
# embedding locally. External endpoints (MNEMOMATIC_EMBED_URL) default to no
# prefix since their model is unknown. Explicit env vars always win. Prefixes
# are baked into stored vectors, so changing them (like changing models)
# requires re-embedding existing content.
_DEFAULT_QUERY_PREFIX = "" if EMBED_URL else model_config.CONFIG.get("query_prefix", "")
_DEFAULT_DOC_PREFIX = "" if EMBED_URL else model_config.CONFIG.get("doc_prefix", "")
EMBED_QUERY_PREFIX = os.environ.get("MNEMOMATIC_EMBED_QUERY_PREFIX", _DEFAULT_QUERY_PREFIX)
EMBED_DOC_PREFIX = os.environ.get("MNEMOMATIC_EMBED_DOC_PREFIX", _DEFAULT_DOC_PREFIX)


# How startup responds to a stored index built by a different embedder than the
# one configured now:
#   "off"   (default) — refuse to start, naming what changed. The safe default:
#                       an unintended model or prefix change should stop the
#                       server, not silently rebuild the index.
#   "auto"  — rebuild and re-embed, but only when something actually changed.
#             A no-op otherwise, so it is safe to leave set permanently.
#   "force" — rebuild and re-embed on every boot regardless. For forcing a
#             re-embed that the identity check would not catch.
def _reindex_mode() -> str:
    raw = os.environ.get("MNEMOMATIC_REINDEX", "").strip().lower()
    if raw == "auto":
        return "auto"
    if raw in ("1", "true", "yes"):
        return "force"
    if raw not in ("", "0", "false", "no"):
        logger.warning(
            "Ignoring unrecognised MNEMOMATIC_REINDEX=%r — expected 'auto', '1', or unset. "
            "Treating it as unset, so a changed embedder will refuse startup rather than "
            "silently re-embedding.", raw,
        )
    return "off"


REINDEX_MODE = _reindex_mode()
# Both modes let the database defer an index-invalidating change instead of
# raising, so startup can rebuild rather than fail.
REINDEX = REINDEX_MODE in ("auto", "force")


def embed_identity() -> dict[str, str]:
    """What the database records about which embedder built its vector index.

    The model name comes from the bundled model's config, or from
    MNEMOMATIC_EMBED_MODEL for an external endpoint. An empty name means the
    identity is unknown (FTS-only, or an external endpoint that never named its
    model), which disables the startup identity check — see Database.
    """
    return {
        "embed_model": model_config.CONFIG.get("model") or EMBED_MODEL or "",
        "embed_query_prefix": EMBED_QUERY_PREFIX,
        "embed_doc_prefix": EMBED_DOC_PREFIX,
    }


# Tool annotation presets, declaring each tool's effects to the client.
ANN_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
ANN_STORE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
ANN_UPDATE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
ANN_DELETE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)
ANN_TAG = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
