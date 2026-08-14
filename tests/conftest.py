"""Pytest configuration for the test suite.

The package under test is normally importable already — `uv run` installs it
in editable mode, which covers both CI and the documented commands. This adds
`src/` to the path as a fallback so a plain `pytest` run outside uv still
works, and keeps the individual test modules free of import boilerplate.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
