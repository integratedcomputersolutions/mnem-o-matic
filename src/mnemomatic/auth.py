"""Authentication middleware for MCP server.

Supports optional Bearer token authentication via the Authorization header.
When api_key is empty, authentication is disabled but logging still tracks all requests.
"""

import hmac
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("mnemomatic")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """HTTP middleware for optional Bearer token authentication.

    When initialized with api_key="", authentication is disabled but request
    logging is still performed. This allows a single code path regardless of
    auth configuration.
    """

    def __init__(self, app, api_key: str = ""):
        """Initialize middleware.

        Args:
            app: ASGI application
            api_key: API key for Bearer token validation. If empty, auth is disabled.
        """
        super().__init__(app)
        self.api_key = api_key.strip()
        self.auth_enabled = bool(self.api_key)

        if self.auth_enabled:
            logger.info("Authentication enabled (Bearer token required)")
        else:
            logger.warning("Authentication disabled — server is running without API key validation")

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

        # If auth is disabled, just log and proceed
        if not self.auth_enabled:
            response = await call_next(request)
            logger.debug(
                "Request: %s %s from %s (auth disabled)",
                method, path, client_ip,
            )
            return response

        # Auth is enabled — validate token
        if not auth_header:
            logger.warning(
                "Unauthorized request: missing Authorization header (%s %s from %s)",
                method, path, client_ip,
            )
            return JSONResponse(
                {
                    "error": "Missing Authorization header",
                    "details": "Required format: 'Authorization: Bearer <token>'",
                },
                status_code=401,
            )

        # Validate header format
        if not auth_header.lower().startswith("bearer "):
            logger.warning(
                "Unauthorized request: invalid Authorization header format (%s %s from %s)",
                method, path, client_ip,
            )
            return JSONResponse(
                {
                    "error": "Invalid Authorization header format",
                    "details": "Required format: 'Authorization: Bearer <token>'",
                },
                status_code=401,
            )

        # Extract token
        try:
            token = auth_header[7:].strip()  # Remove "Bearer " prefix
        except (IndexError, AttributeError):
            logger.warning(
                "Unauthorized request: malformed Authorization header (%s %s from %s)",
                method, path, client_ip,
            )
            return JSONResponse(
                {
                    "error": "Malformed Authorization header",
                    "details": "Token is missing or empty",
                },
                status_code=401,
            )

        if not token:
            logger.warning(
                "Unauthorized request: empty token (%s %s from %s)",
                method, path, client_ip,
            )
            return JSONResponse(
                {
                    "error": "Invalid Authorization header",
                    "details": "Token is empty",
                },
                status_code=401,
            )

        # Validate token using constant-time comparison (prevents timing attacks)
        if not hmac.compare_digest(token, self.api_key):
            logger.warning(
                "Unauthorized request: invalid API key (%s %s from %s)",
                method, path, client_ip,
            )
            return JSONResponse(
                {
                    "error": "Invalid API key",
                    "details": "The provided token does not match the server's API key",
                },
                status_code=403,
            )

        # Authentication successful
        logger.debug(
            "Authenticated request: %s %s from %s",
            method, path, client_ip,
        )
        response = await call_next(request)
        return response
