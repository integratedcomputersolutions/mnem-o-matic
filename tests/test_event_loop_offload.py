"""Slow work must not run on the event loop.

FastMCP calls a synchronous tool or resource inline, and a Starlette handler
that calls blocking code holds the loop for as long as it takes. Either way
a few concurrent calls (a remote embedder can take its whole timeout) would
stall every other request. These pin that every MCP tool and resource is
registered as an offloading wrapper, and that the HTTP routes doing search
and export hand their work to a thread.
"""

import asyncio
import unittest
from contextvars import ContextVar
from unittest.mock import MagicMock, patch

from starlette.applications import Starlette
from starlette.routing import Route

import mnemomatic.server as server
from mnemomatic import runtime, tools_admin, tools_search
from mnemomatic.api import build_api_routes
from mnemomatic.auth import COOKIE_NAME, AuthMiddleware
from mnemomatic.identity import FirstRun
from mnemomatic.models import Note
from tests._support import CookieClient, IdentityFixture


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class TestMcpRegistration(unittest.TestCase):
    def test_every_tool_is_offloaded(self):
        tools = server.mcp._tool_manager.list_tools()
        self.assertTrue(tools)
        for t in tools:
            with self.subTest(tool=t.name):
                self.assertTrue(t.is_async)
                self.assertTrue(hasattr(t.fn, "__wrapped__"), "registered with @mcp.tool, not runtime.tool")

    def test_every_resource_is_offloaded(self):
        manager = server.mcp._resource_manager
        resources = list(manager._resources.values()) + list(manager._templates.values())
        self.assertTrue(resources)
        for r in resources:
            with self.subTest(resource=str(getattr(r, "uri", None) or r.uri_template)):
                self.assertTrue(hasattr(r.fn, "__wrapped__"), "registered with @mcp.resource, not runtime.resource")

    def test_tool_runs_on_a_worker_thread_and_keeps_context(self):
        var = ContextVar("probe", default=None)
        seen = {}

        def fn(x: int) -> dict:
            seen["loop"] = _on_event_loop()
            seen["var"] = var.get()
            return {"x": x}

        async def call():
            var.set("request-meta")
            return await runtime._offloaded(fn)(x=3)

        self.assertEqual(asyncio.run(call()), {"x": 3})
        self.assertFalse(seen["loop"])
        self.assertEqual(seen["var"], "request-meta")

    def test_decorators_return_the_plain_function(self):
        # Modules and tests call tools directly; they must stay synchronous.
        from mnemomatic import tools_content, tools_history
        for fn in (tools_content.store_note, tools_search.search, tools_history.consolidation_report,
                   tools_admin.embedding_info, tools_search.list_namespaces):
            with self.subTest(fn=fn.__name__):
                self.assertFalse(asyncio.iscoroutinefunction(fn))


class TestHttpRoutes(unittest.TestCase):
    def setUp(self):
        self.fx = IdentityFixture()
        self.addCleanup(self.fx.close)

    def test_api_search_and_related_run_off_the_loop(self):
        seen = []

        def fake_search(*a, **kw):
            seen.append(_on_event_loop())
            return [], False

        def fake_related(*a, **kw):
            seen.append(_on_event_loop())
            return []

        mount = build_api_routes(identity=lambda: self.fx.identity, db_getter=lambda: self.fx.db,
                                 settings_info=dict, first_run=FirstRun(), https=None)
        client = CookieClient(AuthMiddleware(Starlette(routes=[mount]), identity=lambda: self.fx.identity))
        cookies = {COOKIE_NAME: self.fx.session_for(self.fx.user)}
        note, _ = self.fx.db.store_note(Note(namespace="proj", title="t", content="c"), embedding=None)
        with patch.object(tools_search, "_search", fake_search), \
                patch.object(tools_search, "_related", fake_related):
            self.assertEqual(client.get("/api/search?q=x", cookies=cookies).status_code, 200)
            self.assertEqual(client.get(f"/api/items/note/{note.id}/related", cookies=cookies).status_code, 200)
        self.assertEqual(seen, [False, False])

    def test_export_runs_off_the_loop(self):
        seen = []

        def fake_export(namespace):
            seen.append(_on_event_loop())
            return b"zip", "export.zip"

        db = MagicMock()
        db.list_namespaces.side_effect = lambda: seen.append(_on_event_loop()) or ["proj"]
        app = Starlette(routes=[Route("/export", tools_admin._export_route, methods=["GET"])])
        client = CookieClient(AuthMiddleware(app, identity=lambda: self.fx.identity))
        with patch.object(tools_admin, "_make_export", fake_export), \
                patch.object(tools_admin, "_audit"), patch.object(runtime, "_db", return_value=db):
            resp = client.get("/export?namespace=proj", headers={"Authorization": f"Bearer {self.fx.user_token}"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen, [False, False])


if __name__ == "__main__":
    unittest.main()
