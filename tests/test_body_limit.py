"""Tests for the request body size limit (mnemomatic.bodylimit).

Two paths matter and they are independent: a declared Content-Length is
refused before the body is read at all, and a chunked upload that declares no
length is refused once the streamed bytes pass the ceiling. A store_document
call at the model's own maxima must still get through, since that is what the
limit is sized for.
"""

import json
import unittest

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic import config
from mnemomatic.bodylimit import BodyLimitMiddleware

LIMIT = 1024


async def _echo(request):
    body = await request.body()
    return PlainTextResponse(f"read {len(body)}")


async def _stream(request):
    async def chunks():
        for _ in range(3):
            yield b"event: ping\n\n"

    return StreamingResponse(chunks(), media_type="text/event-stream")


def _client(max_bytes=LIMIT):
    app = Starlette(routes=[
        Route("/mcp", _echo, methods=["GET", "POST"]),
        Route("/sse", _stream, methods=["GET"]),
    ])
    return TestClient(BodyLimitMiddleware(app, max_bytes=max_bytes))


class TestDeclaredLength(unittest.TestCase):
    def test_oversized_content_length_refused(self):
        resp = _client().post("/mcp", content=b"x" * (LIMIT + 1))
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.json()["error"], "Request body too large")

    def test_body_at_the_limit_passes(self):
        resp = _client().post("/mcp", content=b"x" * LIMIT)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, f"read {LIMIT}")

    def test_a_lying_content_length_does_not_admit_the_body(self):
        # The declared length is checked first, but the streamed count is what
        # actually bounds the read.
        resp = _client().request(
            "POST", "/mcp",
            content=iter([b"x" * (LIMIT + 1)]),
            headers={"content-length": "10"},
        )
        self.assertEqual(resp.status_code, 413)


class TestChunkedBody(unittest.TestCase):
    """An iterator body makes httpx use Transfer-Encoding: chunked, so there is
    no Content-Length to check up front."""

    def test_oversized_chunked_body_refused(self):
        resp = _client().request(
            "POST", "/mcp", content=iter([b"x" * 512] * 4),
        )
        self.assertNotIn("content-length", {k.lower() for k in resp.request.headers})
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.json()["error"], "Request body too large")

    def test_small_chunked_body_passes(self):
        resp = _client().request("POST", "/mcp", content=iter([b"x" * 100] * 2))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "read 200")


class TestPassThrough(unittest.TestCase):
    def test_bodyless_get_unaffected(self):
        self.assertEqual(_client().get("/mcp").status_code, 200)

    def test_streamed_response_unaffected(self):
        resp = _client().get("/sse")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "event: ping\n\n" * 3)


class TestConfiguredLimit(unittest.TestCase):
    def test_largest_legitimate_request_fits(self):
        """A store_document call at the validation maxima, JSON-escaped, must
        pass the shipped limit — that sizing is the reason it is not a knob."""
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "store_document",
                "arguments": {
                    "namespace": "n" * 100,
                    "title": "t" * 500,
                    "content": "c" * 100_000,
                    "tags": ["tag" * 16] * 100,
                    "metadata": {f"k{i}": "v" * 10_000 for i in range(50)},
                },
            },
        }).encode()
        self.assertLess(len(body), config.MAX_BODY_BYTES)
        resp = _client(config.MAX_BODY_BYTES).post("/mcp", content=body)
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
