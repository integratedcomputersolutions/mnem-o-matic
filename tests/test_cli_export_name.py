"""Tests for the export filename the CLI takes from the server.

`mnemomatic-cli export -o <dir>` names the file from the response's
Content-Disposition. The user has chosen to trust the server with the contents
of the export; that is not the same as trusting it to choose a path on their
filesystem, so only the basename survives.
"""

import unittest

from mnemomatic_cli.cli import _suggested_name

FALLBACK = "mnemomatic-export.zip"


class TestSuggestedName(unittest.TestCase):
    def test_plain_filename(self):
        self.assertEqual(
            _suggested_name('attachment; filename="mnemomatic-export-20260902.zip"'),
            "mnemomatic-export-20260902.zip",
        )

    def test_traversal_reduced_to_its_basename(self):
        self.assertEqual(_suggested_name('attachment; filename="../../evil"'), "evil")

    def test_absolute_path_reduced_to_its_basename(self):
        self.assertEqual(_suggested_name('attachment; filename="/etc/cron.d/x"'), "x")

    def test_trailing_separator_still_yields_the_name(self):
        self.assertEqual(_suggested_name('attachment; filename="backup/"'), "backup")

    def test_missing_header_falls_back(self):
        self.assertEqual(_suggested_name(""), FALLBACK)

    def test_header_without_a_filename_falls_back(self):
        self.assertEqual(_suggested_name("attachment"), FALLBACK)

    def test_names_that_reduce_to_nothing_fall_back(self):
        # Each of these has no usable basename, so it must not become the
        # output directory itself or its parent.
        for raw in ("..", ".", "../", "/", "../.."):
            with self.subTest(raw=raw):
                self.assertEqual(_suggested_name(f'attachment; filename="{raw}"'), FALLBACK)


if __name__ == "__main__":
    unittest.main()
