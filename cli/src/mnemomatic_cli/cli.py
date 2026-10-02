"""mnemomatic-cli — shell interface to a running Mnem-O-matic MCP server."""

import argparse
import json
import os
import re
import ssl
import stat
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from mnemomatic_cli._mcp_client import MCPClient, _describe_http_error

_DEFAULT_URL = "http://localhost:8000"
_DEFAULT_MODE = "hybrid"
_CONFIG_PATH = Path.home() / ".config" / "mnemomatic" / "config.toml"
_ITEM_TYPES = ("document", "knowledge", "note")       # singular — used in tool calls
_RESOURCE_TYPES = ("documents", "knowledge", "notes")  # plural  — used in resource URIs
_SINGULAR = dict(zip(_RESOURCE_TYPES, _ITEM_TYPES))


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    """Load TOML config; return empty dict if missing, warn on parse error."""
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        print(f"Warning: could not parse config file {path}: {exc}", file=sys.stderr)
        return {}
    server = cfg.get("server", {})
    if server.get("token") or server.get("api_key"):
        mode = path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            print(f"Warning: {path} contains a token and is readable by others (mode {oct(mode)[-3:]})", file=sys.stderr)
    return cfg


def _resolve(cli_val, env_var: str, cfg_section: str, cfg_key: str, cfg: dict, default):
    """Merge priority: CLI flag > env var > config file > default."""
    if cli_val is not None:
        return cli_val
    env = os.environ.get(env_var)
    if env is not None:
        return env
    section = cfg.get(cfg_section, {})
    if cfg_key in section:
        return section[cfg_key]
    return default


def _resolve_token(cli_val, cfg: dict) -> str:
    """The bearer token: --token, MNEMOMATIC_TOKEN, or [server] token.

    3.0 renamed the credential from the shared API key to a per-user token.
    The old spellings (MNEMOMATIC_API_KEY, [server] api_key) are still read
    for one release so an upgrade does not break scripts, with a reminder.
    """
    token = _resolve(cli_val, "MNEMOMATIC_TOKEN", "server", "token", cfg, None)
    if token is not None:
        return token
    legacy = _resolve(None, "MNEMOMATIC_API_KEY", "server", "api_key", cfg, None)
    if legacy is not None:
        print("Warning: MNEMOMATIC_API_KEY / [server] api_key are deprecated; "
              "use MNEMOMATIC_TOKEN / [server] token", file=sys.stderr)
        return legacy
    return ""


def _ssl_context(ca_cert: str | None) -> ssl.SSLContext | None:
    """A verifying TLS context that also trusts `ca_cert` (the server's own
    CA, downloaded from /ca.crt). None means the system trust store alone."""
    if not ca_cert:
        return None
    try:
        return ssl.create_default_context(cafile=ca_cert)
    except (OSError, ssl.SSLError) as exc:
        _err(f"cannot load CA certificate {ca_cert}: {exc}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _out(data, pretty: bool):
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, indent=2 if pretty else None))


def _err(msg: str):
    print(json.dumps({"error": msg}), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _parse_meta(items: list[str] | None) -> dict:
    """Parse ["KEY=VALUE", ...] into {"KEY": "VALUE"}."""
    if not items:
        return {}
    result = {}
    for item in items:
        if "=" not in item:
            _err(f"--meta requires KEY=VALUE format, got: {item!r}")
        k, v = item.split("=", 1)
        result[k] = v
    return result


def _read_content(value: str) -> str:
    """Return *value* as-is, or read stdin when value is '-'."""
    if value == "-":
        return sys.stdin.read()
    return value


# ---------------------------------------------------------------------------
# Store / update: one field spec per item type
# ---------------------------------------------------------------------------

_CONTENT_HELP = "Content text, or '-' to read from stdin"

# Per type: the fields `store` takes as positionals (after the namespace) and
# `update` as flags, then the fields both take as flags. Tags and metadata are
# common to every type. Defaults belong to the server, so a flag left out is
# not sent; namespace is fixed at creation, so `update` takes an id instead.
_FIELDS = {
    "document": (("title", "content"), {
        "mime_type": {"metavar": "TYPE", "help": "default: text/markdown"},
    }),
    "knowledge": (("subject", "fact"), {
        "confidence": {"type": float, "metavar": "0.0-1.0", "help": "default: 1.0"},
        "source": {"metavar": "SRC", "help": "default: unknown"},
    }),
    "note": (("title", "content"), {
        "source": {"metavar": "SRC", "help": "default: text"},
    }),
}


def _add_item_args(parser: argparse.ArgumentParser, item_type: str, *, store: bool):
    main_fields, flag_fields = _FIELDS[item_type]
    parser.add_argument("namespace" if store else "id")
    for field in main_fields:
        kwargs = {"help": _CONTENT_HELP} if field == "content" else {}
        if store:
            parser.add_argument(field, **kwargs)
        else:
            parser.add_argument(f"--{field}", metavar=field[0].upper(), **kwargs)
    for field, kwargs in flag_fields.items():
        if not store:  # an omitted flag leaves the field as it is, not at the default
            kwargs = {k: v for k, v in kwargs.items() if k != "help"}
        parser.add_argument("--" + field.replace("_", "-"), **kwargs)
    parser.add_argument("--tag", action="append", metavar="TAG")
    parser.add_argument("--meta", action="append", metavar="KEY=VALUE")


def _item_params(args) -> dict:
    """The tool arguments for a store/update command: every field given."""
    main_fields, flag_fields = _FIELDS[args.item_type]
    params = {}
    for field in ("id", "namespace", *main_fields, *flag_fields):
        val = getattr(args, field, None)
        if val is not None:
            params[field] = val
    if "content" in params:
        params["content"] = _read_content(params["content"])
    if args.tag:
        params["tags"] = args.tag
    meta = _parse_meta(args.meta)
    if meta:
        params["metadata"] = meta
    return params


# ---------------------------------------------------------------------------
# Export (plain HTTP download, no MCP involved)
# ---------------------------------------------------------------------------

def _suggested_name(disposition: str) -> str:
    """The filename to use from a Content-Disposition header.

    Only the basename is kept. The server is trusted with the data it returns,
    but not with where that data lands: a `filename="../../.bashrc"` would
    otherwise write outside the directory the user named with -o.
    """
    match = re.search(r'filename="([^"]+)"', disposition)
    suggested = Path(match.group(1)).name if match else ""
    if not suggested or suggested in (".", ".."):
        return "mnemomatic-export.zip"
    return suggested


def _cmd_export(args, server_url: str, token: str, ssl_context=None) -> None:
    """Download the zip export; write it where -o points, atomically.

    -o accepts a directory (server-suggested filename inside it), a file
    path, or '-' for stdout. The zip lands as <name>.part first and is
    renamed only on a complete download, so an interrupted run never
    replaces an existing backup with a truncated file.
    """
    url = server_url.rstrip("/") + "/export"
    if args.namespace:
        url += "?" + urllib.parse.urlencode({"namespace": args.namespace})
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=300, context=ssl_context) as resp:
            data = resp.read()
            disposition = resp.headers.get("Content-Disposition", "")
    except urllib.error.HTTPError as exc:
        _err(f"export failed: {_describe_http_error(exc)}")
    except urllib.error.URLError as exc:
        _err(f"cannot reach server at {url}: {exc.reason}")

    if args.output == "-":
        sys.stdout.buffer.write(data)
        return

    suggested = _suggested_name(disposition)
    target = Path(args.output)
    # A trailing separator means "directory" even if it doesn't exist yet.
    if target.is_dir() or args.output.endswith(os.sep):
        target = target / suggested
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(data)
    part.replace(target)
    print(str(target))


# ---------------------------------------------------------------------------
# Resource URI mapping for the 'get' command
# ---------------------------------------------------------------------------

_GET_URI = {
    "document": "mnemomatic://document/{id}",
    "knowledge": "mnemomatic://knowledge-entry/{id}",
    "note": "mnemomatic://note/{id}",
}


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="mnemomatic-cli",
        description="Shell interface to a running Mnem-O-matic MCP server.",
    )
    root.add_argument("--server-url", metavar="URL", default=None,
                      help="Server base URL (env: MNEMOMATIC_SERVER_URL, default: http://localhost:8000)")
    root.add_argument("--token", metavar="TOKEN", default=None,
                      help="Your API token, mnm_... (env: MNEMOMATIC_TOKEN — preferred over this flag to avoid exposure in process list)")
    root.add_argument("--api-key", dest="token", metavar="KEY", default=None, help=argparse.SUPPRESS)
    root.add_argument("--ca-cert", metavar="FILE", default=None,
                      help="CA certificate to trust for HTTPS, e.g. the server's /ca.crt (env: MNEMOMATIC_CA_CERT)")
    root.add_argument("--config", metavar="FILE", default=None,
                      help=f"Config file path (default: {_CONFIG_PATH})")
    root.add_argument("--pretty", action="store_true",
                      help="Indent JSON output")

    sub = root.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- search ---------------------------------------------------------------
    p_search = sub.add_parser("search", help="Search stored content")
    p_search.add_argument("query")
    p_search.add_argument("-n", "--namespace", metavar="NS")
    p_search.add_argument("-t", "--type", metavar="TYPE",
                          choices=["all", *_RESOURCE_TYPES], default="all")
    p_search.add_argument("-l", "--limit", type=int, default=10, metavar="N")
    p_search.add_argument("-m", "--mode", metavar="MODE",
                          choices=["hybrid", "fulltext", "semantic"], default=None,
                          help="Search mode (overrides config/default)")
    p_search.add_argument("--tag", metavar="TAG", action="append", dest="tags",
                          help="Only items with this tag; repeat to require several")
    p_search.add_argument("--updated-after", metavar="DATE",
                          help="Only items updated at or after this ISO date/datetime")

    def per_type(command: str, help: str, item_help: str, types=_ITEM_TYPES):
        """A `command TYPE ...` parser; returns the per-type subparsers."""
        type_sub = sub.add_parser(command, help=help).add_subparsers(
            dest="item_type", metavar="TYPE", required=True)
        return {t: type_sub.add_parser(t, help=item_help.format(t)) for t in types}

    # -- store / update -------------------------------------------------------
    for item_type, p in per_type("store", "Store content", "Store a {}").items():
        _add_item_args(p, item_type, store=True)
    for item_type, p in per_type("update", "Update stored content", "Update a {}").items():
        _add_item_args(p, item_type, store=False)

    # -- delete / read / get --------------------------------------------------
    for command, help, item_help in (
        ("delete", "Delete stored content", "Delete a {}"),
        ("read", "Read full content of an item by ID", "Read a {} by ID"),
        ("get", "Get a single item by ID (via resource URI)", "Get a {} by ID"),
    ):
        for p in per_type(command, help, item_help).values():
            p.add_argument("id")

    # -- tag ------------------------------------------------------------------
    p_tag = sub.add_parser("tag", help="Add/remove tags on an item")
    p_tag.add_argument("id")
    p_tag.add_argument("type", choices=_ITEM_TYPES)
    p_tag.add_argument("--add", action="append", metavar="TAG")
    p_tag.add_argument("--remove", action="append", metavar="TAG")

    # -- namespace ------------------------------------------------------------
    p_ns = sub.add_parser("namespace", help="Manage namespaces")
    ns_sub = p_ns.add_subparsers(dest="ns_action", metavar="ACTION")
    ns_sub.required = True
    ns_sub.add_parser("list", help="List all namespaces")
    p_ns_rename = ns_sub.add_parser("rename", help="Rename a namespace")
    p_ns_rename.add_argument("old_namespace")
    p_ns_rename.add_argument("new_namespace")
    p_ns_delete = ns_sub.add_parser("delete", help="Delete all items in a namespace")
    p_ns_delete.add_argument("namespace")
    p_ns_delete.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt (for scripts and agents)")

    # -- export ---------------------------------------------------------------
    p_export = sub.add_parser("export", help="Download a zip export of stored content")
    p_export.add_argument("-n", "--namespace", metavar="NS",
                          help="Export a single namespace (default: all)")
    p_export.add_argument("-o", "--output", metavar="PATH", default=".",
                          help="Target directory or file path; '-' writes the zip to "
                               "stdout (default: current directory)")

    # -- list -----------------------------------------------------------------
    for p in per_type("list", "List content in a namespace", "List {} in a namespace",
                      _RESOURCE_TYPES).values():
        p.add_argument("namespace")
        p.add_argument("-l", "--limit", type=int, default=None, metavar="N",
                       help="Page size — switches to the paginated list_items tool (max 200)")
        p.add_argument("-o", "--offset", type=int, default=0, metavar="N",
                       help="Items to skip, for fetching subsequent pages (requires --limit)")

    return root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve config file path
    config_path = Path(args.config) if args.config else _CONFIG_PATH
    if args.config and not config_path.exists():
        _err(f"Config file not found: {config_path}")
    cfg = _load_config(config_path)

    # Resolve connection settings
    server_url = _resolve(args.server_url, "MNEMOMATIC_SERVER_URL", "server", "url", cfg, _DEFAULT_URL)
    token = _resolve_token(args.token, cfg)
    ssl_context = _ssl_context(_resolve(args.ca_cert, "MNEMOMATIC_CA_CERT", "server", "ca_cert", cfg, None))

    # Resolve default search mode
    default_mode = _resolve(None, "MNEMOMATIC_SEARCH_MODE", "search", "mode", cfg, _DEFAULT_MODE)

    # Apply default search mode to search command (CLI flag still overrides)
    if args.command == "search" and args.mode is None:
        args.mode = default_mode

    # Export is a plain HTTP download — no MCP session needed.
    if args.command == "export":
        _cmd_export(args, server_url, token, ssl_context)
        return

    base_url = server_url.rstrip("/") + "/mcp"

    try:
        client = MCPClient(base_url=base_url, api_key=token, ssl_context=ssl_context)
    except (RuntimeError, ValueError) as exc:
        _err(str(exc))

    try:
        result = _dispatch(args, client)
    except RuntimeError as exc:
        _err(str(exc))
    _out(result, args.pretty)


def _dispatch(args, client: MCPClient):
    """Make the one tool call or resource read *args* asks for."""
    match args.command:
        case "search":
            params = {
                "query": args.query,
                "limit": args.limit,
                "mode": args.mode,
                "content_type": args.type,
            }
            if args.namespace:
                params["namespace"] = args.namespace
            if args.tags:
                params["tags"] = args.tags
            if args.updated_after:
                params["updated_after"] = args.updated_after
            return client.call_tool("search", params)
        case "store" | "update":
            return client.call_tool(f"{args.command}_{args.item_type}", _item_params(args))
        case "delete":
            return client.call_tool(f"delete_{args.item_type}", {"id": args.id})
        case "read":
            return client.call_tool("read", {"item_type": args.item_type, "id": args.id})
        case "get":
            return client.read_resource(_GET_URI[args.item_type].format(id=args.id))
        case "tag":
            params = {"item_id": args.id, "item_type": args.type}
            if args.add:
                params["add_tags"] = args.add
            if args.remove:
                params["remove_tags"] = args.remove
            return client.call_tool("tag", params)
        case "namespace":
            match args.ns_action:
                case "list":
                    return client.read_resource("mnemomatic://namespaces")
                case "rename":
                    return client.call_tool("rename_namespace", {
                        "old_namespace": args.old_namespace,
                        "new_namespace": args.new_namespace})
                case "delete":
                    if not args.yes:
                        try:
                            confirm = input(
                                f"This will permanently delete all items in '{args.namespace}'.\n"
                                f"Type the namespace name to confirm: "
                            )
                        except EOFError:
                            _err("Confirmation required. Use --yes to skip (for scripts and agents).")
                        if confirm != args.namespace:
                            _err("Aborted: namespace name did not match.")
                    return client.call_tool("delete_namespace", {"namespace": args.namespace})
        case "list":
            if args.limit is not None:
                # Paginated path via the list_items tool (server >= 1.2).
                return client.call_tool("list_items", {
                    "item_type": _SINGULAR[args.item_type],
                    "namespace": args.namespace,
                    "limit": args.limit,
                    "offset": args.offset,
                })
            return client.read_resource(f"mnemomatic://{args.item_type}/{args.namespace}")
