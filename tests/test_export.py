"""Tests for the zip export (mnemomatic.export) and its /export route.

Covers the archive layout (namespace/type folders, body files, metadata.json
sidecars, root manifest), filename sanitization and collision handling, the
namespace filter, and Bearer-auth enforcement on the HTTP route.
"""

import io
import json
import unittest
import zipfile
from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemomatic.auth import BearerAuthMiddleware
from mnemomatic.db import Database
from mnemomatic.export import EXPORT_FORMAT, _safe_name, _unique, build_export_zip
from mnemomatic.models import Document, Knowledge, Note


def _open(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def _build(db, namespace=None):
    return build_export_zip(db, namespace, server_version="0.0.0-test")


class TestNames(unittest.TestCase):
    def test_invalid_characters_replaced(self):
        # Invalid chars become underscores; a trailing underscore is trimmed.
        self.assertEqual(_safe_name('a/b:c*d?"e"', "x"), "a_b_c_d__e")

    def test_spaces_become_underscores(self):
        self.assertEqual(_safe_name("Design Doc v2", "x"), "Design_Doc_v2")

    def test_windows_trailing_dots_and_spaces_stripped(self):
        self.assertEqual(_safe_name("  name. ", "x"), "name")

    def test_long_names_capped(self):
        self.assertEqual(len(_safe_name("a" * 500, "x")), 100)

    def test_empty_falls_back(self):
        # Dots and spaces are stripped entirely; nothing survives → fallback.
        self.assertEqual(_safe_name(" .. ", "fallback"), "fallback")

    def test_collisions_get_id_suffix_case_insensitively(self):
        used: set = set()
        self.assertEqual(_unique("Doc", used, "aaaabbbb-1"), "Doc")
        # Same name, different case: still a collision (zip archives get
        # extracted onto case-insensitive filesystems).
        self.assertEqual(_unique("doc", used, "ccccdddd-2"), "doc--ccccdddd")


class TestArchive(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_layout_bodies_and_sidecars(self):
        doc, _ = self.db.store_document(
            Document(namespace="proj", title="Design Doc", content="# Heading\nbody",
                     tags=["a"], metadata={"k": "v"}),
            embedding=None,
        )
        know, _, _ = self.db.store_knowledge(
            Knowledge(namespace="proj", subject="auth", fact="Uses JWT.", confidence=0.9),
            embedding=None,
        )
        note, _ = self.db.store_note(
            Note(namespace="proj", title="Scratch", content="rough idea"), embedding=None,
        )
        data, filename = _build(self.db)
        self.assertRegex(filename, r"^mnemomatic-export-\d{4}-\d{2}-\d{2}\.zip$")

        with _open(data) as zf:
            names = set(zf.namelist())
            self.assertEqual(names, {
                "export-info.json",
                "proj/documents/Design_Doc.md", "proj/documents/metadata.json",
                "proj/knowledge/auth.md", "proj/knowledge/metadata.json",
                "proj/notes/Scratch.md", "proj/notes/metadata.json",
            })
            # Bodies are the content alone, byte-faithful.
            self.assertEqual(zf.read("proj/documents/Design_Doc.md").decode(), "# Heading\nbody")
            self.assertEqual(zf.read("proj/knowledge/auth.md").decode(), "Uses JWT.")

            sidecar = json.loads(zf.read("proj/documents/metadata.json"))
            record = sidecar["Design_Doc.md"]
            self.assertEqual(record["id"], doc.id)
            self.assertEqual(record["title"], "Design Doc")
            self.assertEqual(record["tags"], ["a"])
            self.assertEqual(record["metadata"], {"k": "v"})
            self.assertIn("created_at", record)

            k_record = json.loads(zf.read("proj/knowledge/metadata.json"))["auth.md"]
            self.assertEqual(k_record["confidence"], 0.9)
            self.assertEqual(k_record["id"], know.id)
            n_record = json.loads(zf.read("proj/notes/metadata.json"))["Scratch.md"]
            self.assertEqual(n_record["id"], note.id)

            manifest = json.loads(zf.read("export-info.json"))
            self.assertEqual(manifest["format"], EXPORT_FORMAT)
            self.assertEqual(manifest["counts"], {"documents": 1, "knowledge": 1, "notes": 1})
            self.assertNotIn("embedding", manifest)  # nothing embedder-related is exported
            self.assertEqual(manifest["namespaces"], {"proj": "proj"})

    def test_mime_type_extension_mapping(self):
        for mime, ext in (("text/plain", ".txt"), ("application/json", ".json"),
                          ("application/x-unknown", ".md")):
            self.db.store_document(
                Document(namespace="m", title=f"file {ext}", content="x", mime_type=mime),
                embedding=None,
            )
        data, _ = _build(self.db)
        with _open(data) as zf:
            names = set(zf.namelist())
        self.assertIn("m/documents/file_.txt.txt", names)
        self.assertIn("m/documents/file_.json.json", names)
        self.assertIn("m/documents/file_.md.md", names)

    def test_title_sanitization_and_collision(self):
        # Both titles sanitize to "a_b"; the second gets an id suffix.
        first, _ = self.db.store_document(
            Document(namespace="ns", title="a/b", content="one"), embedding=None)
        second, _ = self.db.store_document(
            Document(namespace="ns", title="a*b", content="two"), embedding=None)
        data, _ = _build(self.db)
        with _open(data) as zf:
            sidecar = json.loads(zf.read("ns/documents/metadata.json"))
        self.assertEqual(len(sidecar), 2)
        titles = {record["title"] for record in sidecar.values()}
        self.assertEqual(titles, {"a/b", "a*b"})
        # Exact original titles recoverable; filenames unique.
        self.assertEqual(len(set(sidecar)), 2)

    def test_namespace_with_path_characters(self):
        self.db.store_note(
            Note(namespace='a/b?c d"e', title="n", content="x"), embedding=None)
        data, _ = _build(self.db)
        with _open(data) as zf:
            manifest = json.loads(zf.read("export-info.json"))
            folder = next(iter(manifest["namespaces"]))
            self.assertEqual(manifest["namespaces"][folder], 'a/b?c d"e')
            self.assertNotIn("/", folder)
            self.assertIn(f"{folder}/notes/n.md", zf.namelist())

    def test_namespace_filter(self):
        self.db.store_note(Note(namespace="keep", title="k", content="x"), embedding=None)
        self.db.store_note(Note(namespace="drop", title="d", content="y"), embedding=None)
        data, _ = _build(self.db, namespace="keep")
        with _open(data) as zf:
            names = zf.namelist()
            manifest = json.loads(zf.read("export-info.json"))
        self.assertTrue(any(n.startswith("keep/") for n in names))
        self.assertFalse(any(n.startswith("drop/") for n in names))
        self.assertEqual(manifest["namespace_filter"], "keep")
        self.assertEqual(manifest["counts"]["notes"], 1)

    def test_empty_store_exports_manifest_only(self):
        data, _ = _build(self.db)
        with _open(data) as zf:
            self.assertEqual(zf.namelist(), ["export-info.json"])


class TestExportRoute(unittest.TestCase):
    """The /export route as the server wires it: behind BearerAuthMiddleware."""

    def setUp(self):
        # File-backed db: the TestClient serves requests on a worker thread,
        # and each thread gets its own connection — a ":memory:" database
        # would be empty there.
        import tempfile
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = Database(self._tmp.name)
        self.db.store_note(Note(namespace="proj", title="n", content="x"), embedding=None)

        async def export_route(request):
            from starlette.responses import JSONResponse, Response
            namespace = request.query_params.get("namespace") or None
            if namespace and namespace not in self.db.list_namespaces():
                return JSONResponse({"error": "Namespace not found"}, status_code=404)
            data, filename = _build(self.db, namespace)
            return Response(data, media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

        app = Starlette(routes=[Route("/export", export_route, methods=["GET"])])
        self.client = TestClient(BearerAuthMiddleware(app, api_key="k3y"), follow_redirects=False)

    def tearDown(self):
        self.db.close()
        Path(self._tmp.name).unlink(missing_ok=True)

    def test_requires_bearer(self):
        self.assertEqual(self.client.get("/export").status_code, 401)

    def test_downloads_zip_with_auth(self):
        resp = self.client.get("/export", headers={"Authorization": "Bearer k3y"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/zip")
        self.assertIn('filename="mnemomatic-export-', resp.headers["content-disposition"])
        self.assertEqual(resp.content[:2], b"PK")  # zip magic
        with _open(resp.content) as zf:
            self.assertIn("proj/notes/n.md", zf.namelist())

    def test_unknown_namespace_404(self):
        resp = self.client.get("/export?namespace=nope",
                               headers={"Authorization": "Bearer k3y"})
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
