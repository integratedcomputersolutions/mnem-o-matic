"""Pytest configuration for the test suite.

The package under test is normally importable already — `uv run` installs it
in editable mode, which covers both CI and the documented commands. This adds
`src/` to the path as a fallback so a plain `pytest` run outside uv still
works, and keeps the individual test modules free of import boilerplate.

`cli/src` joins it so the CLI's own pure-logic tests run in the ordinary unit
pass. The CLI is a separate, dependency-free package, and only the end-to-end
tests in test_mcp_api.py need it actually installed.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT / "src", _ROOT / "cli" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
