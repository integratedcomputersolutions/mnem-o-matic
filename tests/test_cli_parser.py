"""Tests for the CLI's argument surface: each subcommand parses, and its flags
reach the server as the right tool (or resource) call.

The MCP client is replaced with a mock, so no server is needed; what is
checked is exactly the name and arguments `main()` hands to it.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from mnemomatic_cli import cli


class _CLITestCase(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(cli, "_CONFIG_PATH", Path("/nonexistent/mnemomatic.toml")),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(cli, "MCPClient"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = cli.MCPClient.return_value
        self.client.call_tool.return_value = {"ok": True}
        self.client.read_resource.return_value = "text"

    def run_cli(self, *argv, stdin=None):
        with mock.patch.object(sys, "argv", ["mnemomatic-cli", *argv]), \
                mock.patch.object(sys, "stdin", io.StringIO(stdin or "")), \
                redirect_stdout(io.StringIO()):
            cli.main()

    def assertTool(self, name, arguments):
        self.client.call_tool.assert_called_once_with(name, arguments)

    def assertResource(self, uri):
        self.client.read_resource.assert_called_once_with(uri)


class TestConnection(_CLITestCase):
    def test_server_url_and_token_reach_the_client(self):
        self.run_cli("--server-url", "https://mem.example/", "--token", "mnm_x",
                     "delete", "note", "n1")
        cli.MCPClient.assert_called_once_with(
            base_url="https://mem.example/mcp", api_key="mnm_x", ssl_context=None)

    def test_api_key_is_an_alias_for_token(self):
        self.run_cli("--api-key", "mnm_y", "delete", "note", "n1")
        self.assertEqual(cli.MCPClient.call_args.kwargs["api_key"], "mnm_y")


class TestSearch(_CLITestCase):
    def test_defaults(self):
        self.run_cli("search", "hello")
        self.assertTool("search", {
            "query": "hello", "limit": 10, "mode": "hybrid", "content_type": "all"})

    def test_all_flags(self):
        self.run_cli("search", "hello", "-n", "ns", "-t", "notes", "-l", "3",
                     "-m", "fulltext", "--tag", "a", "--tag", "b",
                     "--updated-after", "2026-01-01")
        self.assertTool("search", {
            "query": "hello", "limit": 3, "mode": "fulltext", "content_type": "notes",
            "namespace": "ns", "tags": ["a", "b"], "updated_after": "2026-01-01"})

    def test_mode_from_environment(self):
        with mock.patch.dict(os.environ, {"MNEMOMATIC_SEARCH_MODE": "semantic"}):
            self.run_cli("search", "hello")
        self.assertEqual(self.client.call_tool.call_args.args[1]["mode"], "semantic")


class TestStore(_CLITestCase):
    def test_document_minimal(self):
        self.run_cli("store", "document", "ns", "T", "body")
        self.assertTool("store_document", {
            "namespace": "ns", "title": "T", "content": "body"})

    def test_document_all_flags(self):
        self.run_cli("store", "document", "ns", "T", "body", "--mime-type", "text/plain",
                     "--tag", "a", "--tag", "b", "--meta", "k=v", "--meta", "x=y=z")
        self.assertTool("store_document", {
            "namespace": "ns", "title": "T", "content": "body", "mime_type": "text/plain",
            "tags": ["a", "b"], "metadata": {"k": "v", "x": "y=z"}})

    def test_document_content_from_stdin(self):
        self.run_cli("store", "document", "ns", "T", "-", stdin="piped")
        self.assertEqual(self.client.call_tool.call_args.args[1]["content"], "piped")

    def test_knowledge_minimal(self):
        self.run_cli("store", "knowledge", "ns", "subj", "a fact")
        self.assertTool("store_knowledge", {
            "namespace": "ns", "subject": "subj", "fact": "a fact"})

    def test_knowledge_all_flags(self):
        self.run_cli("store", "knowledge", "ns", "subj", "a fact", "--confidence", "0.5",
                     "--source", "me", "--tag", "a", "--meta", "k=v")
        self.assertTool("store_knowledge", {
            "namespace": "ns", "subject": "subj", "fact": "a fact", "confidence": 0.5,
            "source": "me", "tags": ["a"], "metadata": {"k": "v"}})

    def test_note_minimal(self):
        # The server's default source ("text") applies; None would be rejected.
        self.run_cli("store", "note", "ns", "T", "body")
        self.assertTool("store_note", {"namespace": "ns", "title": "T", "content": "body"})

    def test_note_all_flags(self):
        self.run_cli("store", "note", "ns", "T", "-", "--source", "voice",
                     "--tag", "a", "--meta", "k=v", stdin="piped")
        self.assertTool("store_note", {
            "namespace": "ns", "title": "T", "content": "piped", "source": "voice",
            "tags": ["a"], "metadata": {"k": "v"}})

    def test_bad_meta_is_refused(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            self.run_cli("store", "document", "ns", "T", "body", "--meta", "novalue")
        self.client.call_tool.assert_not_called()


class TestUpdate(_CLITestCase):
    def test_only_given_fields_are_sent(self):
        for item_type in cli._ITEM_TYPES:
            with self.subTest(item_type):
                self.client.call_tool.reset_mock()
                self.run_cli("update", item_type, "id1")
                self.assertTool(f"update_{item_type}", {"id": "id1"})

    def test_document_all_flags(self):
        self.run_cli("update", "document", "id1", "--title", "T", "--content", "-",
                     "--mime-type", "text/plain", "--tag", "a", "--meta", "k=v",
                     stdin="piped")
        self.assertTool("update_document", {
            "id": "id1", "title": "T", "content": "piped", "mime_type": "text/plain",
            "tags": ["a"], "metadata": {"k": "v"}})

    def test_knowledge_all_flags(self):
        self.run_cli("update", "knowledge", "id1", "--subject", "S", "--fact", "F",
                     "--confidence", "0.25", "--source", "me", "--tag", "a", "--meta", "k=v")
        self.assertTool("update_knowledge", {
            "id": "id1", "subject": "S", "fact": "F", "confidence": 0.25, "source": "me",
            "tags": ["a"], "metadata": {"k": "v"}})

    def test_note_all_flags(self):
        self.run_cli("update", "note", "id1", "--title", "T", "--content", "C",
                     "--source", "voice", "--tag", "a", "--meta", "k=v")
        self.assertTool("update_note", {
            "id": "id1", "title": "T", "content": "C", "source": "voice",
            "tags": ["a"], "metadata": {"k": "v"}})

    def test_namespace_is_not_updatable(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            self.run_cli("update", "note", "id1", "--namespace", "other")


class TestByIdCommands(_CLITestCase):
    def test_delete(self):
        for item_type in cli._ITEM_TYPES:
            with self.subTest(item_type):
                self.client.call_tool.reset_mock()
                self.run_cli("delete", item_type, "id1")
                self.assertTool(f"delete_{item_type}", {"id": "id1"})

    def test_read(self):
        for item_type in cli._ITEM_TYPES:
            with self.subTest(item_type):
                self.client.call_tool.reset_mock()
                self.run_cli("read", item_type, "id1")
                self.assertTool("read", {"item_type": item_type, "id": "id1"})

    def test_get(self):
        expected = {
            "document": "mnemomatic://document/id1",
            "knowledge": "mnemomatic://knowledge-entry/id1",
            "note": "mnemomatic://note/id1",
        }
        for item_type, uri in expected.items():
            with self.subTest(item_type):
                self.client.read_resource.reset_mock()
                self.run_cli("get", item_type, "id1")
                self.assertResource(uri)

    def test_tag(self):
        self.run_cli("tag", "id1", "note", "--add", "a", "--add", "b", "--remove", "c")
        self.assertTool("tag", {
            "item_id": "id1", "item_type": "note",
            "add_tags": ["a", "b"], "remove_tags": ["c"]})

    def test_tag_without_changes(self):
        self.run_cli("tag", "id1", "document")
        self.assertTool("tag", {"item_id": "id1", "item_type": "document"})


class TestNamespace(_CLITestCase):
    def test_list(self):
        self.run_cli("namespace", "list")
        self.assertResource("mnemomatic://namespaces")

    def test_rename(self):
        self.run_cli("namespace", "rename", "old", "new")
        self.assertTool("rename_namespace", {"old_namespace": "old", "new_namespace": "new"})

    def test_delete_with_yes(self):
        self.run_cli("namespace", "delete", "ns", "--yes")
        self.assertTool("delete_namespace", {"namespace": "ns"})

    def test_delete_confirmed_interactively(self):
        with mock.patch("builtins.input", return_value="ns"):
            self.run_cli("namespace", "delete", "ns")
        self.assertTool("delete_namespace", {"namespace": "ns"})

    def test_delete_aborts_on_mismatch(self):
        with mock.patch("builtins.input", return_value="nope"), \
                mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli("namespace", "delete", "ns")
        self.client.call_tool.assert_not_called()


class TestList(_CLITestCase):
    def test_resource_listing(self):
        for resource in cli._RESOURCE_TYPES:
            with self.subTest(resource):
                self.client.read_resource.reset_mock()
                self.run_cli("list", resource, "ns")
                self.assertResource(f"mnemomatic://{resource}/ns")

    def test_paginated_listing(self):
        for resource, item_type in zip(cli._RESOURCE_TYPES, cli._ITEM_TYPES):
            with self.subTest(resource):
                self.client.call_tool.reset_mock()
                self.run_cli("list", resource, "ns", "-l", "5", "-o", "10")
                self.assertTool("list_items", {
                    "item_type": item_type, "namespace": "ns", "limit": 5, "offset": 10})


class TestExport(_CLITestCase):
    def test_export_bypasses_mcp(self):
        with mock.patch.object(cli, "_cmd_export") as export:
            self.run_cli("--server-url", "http://h:1", "--token", "t",
                         "export", "-n", "ns", "-o", "out/")
        cli.MCPClient.assert_not_called()
        args, server_url, token, ssl_context = export.call_args.args
        self.assertEqual((args.namespace, args.output, server_url, token, ssl_context),
                         ("ns", "out/", "http://h:1", "t", None))



class TestServerUrlScheme(_CLITestCase):
    def test_non_http_schemes_refused_for_every_command(self):
        # export went straight to urlopen, which reads file:// and fetches ftp://.
        for argv in (("export", "-o", "-"), ("search", "x")):
            for url in ("file:///etc/passwd", "ftp://host/x"):
                with self.subTest(argv=argv[0], url=url), \
                        mock.patch.object(cli, "_err", side_effect=SystemExit) as err, \
                        mock.patch.object(cli, "_cmd_export") as export:
                    with self.assertRaises(SystemExit):
                        self.run_cli("--server-url", url, *argv)
                    self.assertIn("Unsupported URL scheme", err.call_args.args[0])
                    export.assert_not_called()
                    cli.MCPClient.assert_not_called()


class TestCaCert(unittest.TestCase):
    def test_adds_to_the_system_store_rather_than_replacing_it(self):
        # create_default_context(cafile=...) skips the system store entirely.
        real = cli.ssl.create_default_context
        calls = []

        def spy(*a, **kw):
            calls.append(kw)
            return real(*a, **kw)

        with mock.patch.object(cli.ssl, "create_default_context", side_effect=spy), \
                mock.patch.object(cli.ssl.SSLContext, "load_verify_locations") as load:
            ctx = cli._ssl_context("/path/to/ca.crt")
        self.assertEqual(calls, [{}])                       # system store loaded
        load.assert_called_once_with(cafile="/path/to/ca.crt")
        self.assertIsInstance(ctx, cli.ssl.SSLContext)

    def test_unreadable_ca_is_reported(self):
        with mock.patch.object(cli, "_err", side_effect=SystemExit) as err:
            with self.assertRaises(SystemExit):
                cli._ssl_context("/nonexistent/ca.crt")
        self.assertIn("cannot load CA certificate /nonexistent/ca.crt", err.call_args.args[0])

    def test_no_ca_means_default_context(self):
        self.assertIsNone(cli._ssl_context(None))


if __name__ == "__main__":
    unittest.main()
