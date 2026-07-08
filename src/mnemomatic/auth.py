"""Authentication middleware for MCP server.

Supports optional Bearer token authentication via the Authorization header.
When api_key is empty, authentication is disabled but logging still tracks all requests.
"""

import hmac
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mnemomatic.throttle import FailureThrottle

logger = logging.getLogger("mnemomatic")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """HTTP middleware for optional Bearer token authentication.

    When initialized with api_key="", authentication is disabled but request
    logging is still performed. This allows a single code path regardless of
    auth configuration.

    Repeated invalid-key attempts from the same client are throttled: after a
    burst of failures the client gets 429 responses (with Retry-After) until
    the lockout expires, regardless of what credential it presents.
    """

    def __init__(self, app, api_key: str = "", exempt_ui: bool = False):
        """Initialize middleware.

        Args:
            app: ASGI application
            api_key: API key for Bearer token validation. If empty, auth is disabled.
            exempt_ui: Skip Bearer validation for /ui paths. Only set this when the
                web viewer is registered — it carries its own shared-secret gate.
        """
        super().__init__(app)
        self.api_key = api_key.strip()
        self.auth_enabled = bool(self.api_key)
        self.exempt_ui = exempt_ui
        self._throttle = FailureThrottle()

        if self.auth_enabled:
            logger.info("Authentication enabled (Bearer token required)")
        else:
            logger.warning("Authentication disabled — server is running without API key validation")

    def _reject(self, reason: str, error: str, details: str, status: int,
                method: str, path: str, client_ip: str,
                headers: dict | None = None) -> JSONResponse:
        """Log an unauthorized request and build its JSON error response."""
        logger.warning(
            "Unauthorized request: %s (%s %s from %s)", reason, method, path, client_ip,
        )
        return JSONResponse({"error": error, "details": details}, status_code=status, headers=headers)

    async def dispatch(self, request: Request, call_next):
        """Process request and enforce authentication if enabled.

        Args:
            request: HTTP request
            call_next: ASGI callable to proceed to next middleware/handler

        Returns:
            Response (either error or result from next handler)
        """
        # Extract Authorization header
        auth_header = request.headers.get("authorization", "").strip()

        # Get client IP for logging
        client_ip = request.client[0] if request.client else "unknown"
        method = request.method
        path = request.url.path

        # The web viewer under /ui carries its own shared-secret gate, so the
        # MCP Bearer token does not apply to it. Only honored when the viewer
        # is actually registered — otherwise /ui is protected like any path.
        if self.exempt_ui and (path == "/ui" or path.startswith("/ui/")):
            return await call_next(request)

        # If auth is disabled, just log and proceed
        if not self.auth_enabled:
            response = await call_next(request)
            logger.debug(
                "Request: %s %s from %s (auth disabled)",
                method, path, client_ip,
            )
            return response

        # Locked-out clients are refused before any credential check
        wait = self._throttle.retry_after(client_ip)
        if wait:
            return self._reject(
                "throttled (too many failed attempts)",
                "Too many failed authentication attempts",
                f"Retry after {wait} seconds",
                429, method, path, client_ip,
                headers={"Retry-After": str(wait)},
            )

        # Auth is enabled — validate token
        if not auth_header:
            return self._reject(
                "missing Authorization header",
                "Missing Authorization header",
                "Required format: 'Authorization: Bearer <token>'",
                401, method, path, client_ip,
            )

        # Validate header format
        if not auth_header.lower().startswith("bearer "):
            return self._reject(
                "invalid Authorization header format",
                "Invalid Authorization header format",
                "Required format: 'Authorization: Bearer <token>'",
                401, method, path, client_ip,
            )

        # Extract token
        token = auth_header[7:].strip()  # Remove "Bearer " prefix

        if not token:
            return self._reject(
                "empty token",
                "Invalid Authorization header",
                "Token is empty",
                401, method, path, client_ip,
            )

        # Validate token using constant-time comparison (prevents timing attacks)
        if not hmac.compare_digest(token, self.api_key):
            self._throttle.record_failure(client_ip)
            return self._reject(
                "invalid API key",
                "Invalid API key",
                "The provided token does not match the server's API key",
                403, method, path, client_ip,
            )

        # Authentication successful
        self._throttle.record_success(client_ip)
        logger.debug(
            "Authenticated request: %s %s from %s",
            method, path, client_ip,
        )
        response = await call_next(request)
        return response
