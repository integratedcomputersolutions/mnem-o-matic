"""Reject request bodies larger than the server will ever need.

The largest legitimate request is a store_document call at the model limits
(100 KB of content, 50 metadata values of 10 KB each), which is well under a
megabyte once JSON-escaped. Anything bigger is a mistake or an attack, and it
is cheaper to refuse it here than to let the MCP layer buffer it — the JSON
body is read whole before anything validates it.

Pure ASGI (no BaseHTTPMiddleware) so streamed responses pass through
untouched, and the body is never buffered here: the declared Content-Length is
checked up front, and the streamed bytes are counted as the application reads
them, so a chunked upload that declares no length cannot slip past either.

Over the limit, this middleware answers 413 itself and reports a disconnect to
the application. It cannot raise instead: every ASGI app in the stack wraps
itself in error handling that would turn the exception into a 500 long before
it got back here. Whatever the application makes of the truncated read is
dropped, so a body that stops being read mid-request can never take effect.
"""

from starlette.types import ASGIApp, Receive, Scope, Send

_TOO_LARGE = b'{"error": "Request body too large"}'


class BodyLimitMiddleware:
    """Answer 413 to any request whose body exceeds `max_bytes`."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        over_limit = False       # the body went past the ceiling
        answered = False         # the 413 above came from here
        response_started = False

        async def limited_receive():
            nonlocal received, over_limit, answered
            if over_limit:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    over_limit = True
                    # A response already on the wire has spent the status line,
                    # so there is no 413 to send — an application that streams
                    # while still reading gets its response cut short instead.
                    if not response_started:
                        answered = True
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if answered:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # A read that stops mid-body surfaces as a disconnect error, which
            # is expected once the client has its 413. Anything raised before
            # the limit was hit is a real failure and still propagates.
            if not over_limit:
                raise

    @staticmethod
    async def _reject(send: Send) -> None:
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_TOO_LARGE)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": _TOO_LARGE})
