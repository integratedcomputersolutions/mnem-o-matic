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

# The server reads its configuration from MNEMOMATIC_* variables, and the
# bundled model's settings from /app/model, once, at import. Tests must not
# depend on the machine they run on (a developer's shell, the Docker image),
# so drop the variables and point the model files at a path that cannot
# exist, before anything imports mnemomatic. MNEMOMATIC_TOKEN stays: the
# end-to-end tests in test_mcp_api.py take the live server's token from it.
import os  # noqa: E402

for _name in [n for n in os.environ if n.startswith("MNEMOMATIC_")]:
    if _name != "MNEMOMATIC_TOKEN":
        del os.environ[_name]
_NOWHERE = str(_ROOT / "tests" / "no-such-model")
os.environ["MNEMOMATIC_MODEL_PATH"] = _NOWHERE + "/model.onnx"
os.environ["MNEMOMATIC_TOKENIZER_PATH"] = _NOWHERE + "/tokenizer.json"
os.environ["MNEMOMATIC_MODEL_CONFIG_PATH"] = _NOWHERE + "/model_config.json"

# Importing the tool modules is what registers them with the one FastMCP
# instance, in import order. Load the server first, so the published order is
# server.py's (test_tool_registration pins it) whichever test module happens
# to import a tool module first.
import mnemomatic.server  # noqa: E402,F401
