"""mnemomatic-cli — shell interface to a running Mnem-O-matic MCP server."""

import argparse
import json
import os
import stat
import sys
import tomllib
from pathlib import Path

from mnemomatic_cli._mcp_client import MCPClient

_DEFAULT_URL = "http://localhost:8000"
_DEFAULT_MODE = "hybrid"
_CONFIG_PATH = Path.home() / ".config" / "mnemomatic" / "config.toml"
_ITEM_TYPES = ("document", "knowledge", "note")       # singular — used in tool calls
_RESOURCE_TYPES = ("documents", "knowledge", "notes")  # plural  — used in resource URIs


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
    if cfg.get("server", {}).get("api_key"):
        mode = path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            print(f"Warning: {path} contains api_key and is readable by others (mode {oct(mode)[-3:]})", file=sys.stderr)
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


def _run(fn, pretty: bool):
    try:
        result = fn()
        _out(result, pretty)
    except RuntimeError as exc:
        _err(str(exc))


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
# Generic command helpers
# ---------------------------------------------------------------------------

def _collect_params(args, fields: list[str], *, required: bool) -> dict:
    """Build a params dict from *args* for the given field names.

    When *required* is True all fields are included unconditionally (store).
    When False only non-None fields are included (update), and 'id' is always
    added first.
    """
    params: dict = {}
    if not required:
        params["id"] = args.id
    for field in fields:
        val = getattr(args, field.replace("-", "_"), None)
        if required or val is not None:
            params[field] = val
    if getattr(args, "tag", None):
        params["tags"] = args.tag
    meta = _parse_meta(getattr(args, "meta", None))
    if meta:
        params["metadata"] = meta
    return params


# Field lists per item type — shared by store and update command builders.
_DOCUMENT_FIELDS = ["namespace", "title", "content", "mime_type"]
_KNOWLEDGE_FIELDS = ["namespace", "subject", "fact", "confidence", "source"]
_NOTE_FIELDS = ["namespace", "title", "content", "source"]

# Update uses the same fields minus 'namespace' (immutable after creation).
_UPDATE_FIELDS = {
    "document": [f for f in _DOCUMENT_FIELDS if f != "namespace"],
    "knowledge": [f for f in _KNOWLEDGE_FIELDS if f != "namespace"],
    "note": [f for f in _NOTE_FIELDS if f != "namespace"],
}

_STORE_FIELDS = {
    "document": _DOCUMENT_FIELDS,
    "knowledge": _KNOWLEDGE_FIELDS,
    "note": _NOTE_FIELDS,
}


def _cmd_tool(tool: str, args, client: MCPClient, pretty: bool,
              fields: list[str], required: bool):
    params = _collect_params(args, fields, required=required)
    if "content" in params and isinstance(params["content"], str):
        params["content"] = _read_content(params["content"])
    _run(lambda: client.call_tool(tool, params), pretty)


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
    root.add_argument("--api-key", metavar="KEY", default=None,
                      help="Bearer token (env: MNEMOMATIC_API_KEY — preferred over this flag to avoid exposure in process list)")
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
                          choices=["all", "documents", "knowledge", "notes"], default="all")
    p_search.add_argument("-l", "--limit", type=int, default=10, metavar="N")
    p_search.add_argument("-m", "--mode", metavar="MODE",
                          choices=["hybrid", "fulltext", "semantic"], default=None,
                          help="Search mode (overrides config/default)")

    # -- store ----------------------------------------------------------------
    p_store = sub.add_parser("store", help="Store content")
    store_sub = p_store.add_subparsers(dest="store_type", metavar="TYPE")
    store_sub.required = True

    p_store_doc = store_sub.add_parser("document", help="Store a document")
    p_store_doc.add_argument("namespace")
    p_store_doc.add_argument("title")
    p_store_doc.add_argument("content", help="Content text, or '-' to read from stdin")
    p_store_doc.add_argument("--mime-type", default="text/markdown", metavar="TYPE")
    p_store_doc.add_argument("--tag", action="append", metavar="TAG")
    p_store_doc.add_argument("--meta", action="append", metavar="KEY=VALUE")

    p_store_know = store_sub.add_parser("knowledge", help="Store a knowledge entry")
    p_store_know.add_argument("namespace")
    p_store_know.add_argument("subject")
    p_store_know.add_argument("fact")
    p_store_know.add_argument("--confidence", type=float, default=1.0, metavar="0.0-1.0")
    p_store_know.add_argument("--source", default="unknown", metavar="SRC")
    p_store_know.add_argument("--tag", action="append", metavar="TAG")

    p_store_note = store_sub.add_parser("note", help="Store a note")
    p_store_note.add_argument("namespace")
    p_store_note.add_argument("title")
    p_store_note.add_argument("content", help="Content text, or '-' to read from stdin")
    p_store_note.add_argument("--source", metavar="SRC")
    p_store_note.add_argument("--tag", action="append", metavar="TAG")

    # -- update ---------------------------------------------------------------
    p_update = sub.add_parser("update", help="Update stored content")
    update_sub = p_update.add_subparsers(dest="update_type", metavar="TYPE")
    update_sub.required = True

    p_upd_doc = update_sub.add_parser("document", help="Update a document")
    p_upd_doc.add_argument("id")
    p_upd_doc.add_argument("--title", metavar="T")
    p_upd_doc.add_argument("--content", metavar="C", help="Content text, or '-' to read from stdin")
    p_upd_doc.add_argument("--mime-type", metavar="TYPE")
    p_upd_doc.add_argument("--tag", action="append", metavar="TAG")
    p_upd_doc.add_argument("--meta", action="append", metavar="KEY=VALUE")

    p_upd_know = update_sub.add_parser("knowledge", help="Update a knowledge entry")
    p_upd_know.add_argument("id")
    p_upd_know.add_argument("--subject", metavar="S")
    p_upd_know.add_argument("--fact", metavar="F")
    p_upd_know.add_argument("--confidence", type=float, metavar="0.0-1.0")
    p_upd_know.add_argument("--source", metavar="SRC")

    p_upd_note = update_sub.add_parser("note", help="Update a note")
    p_upd_note.add_argument("id")
    p_upd_note.add_argument("--title", metavar="T")
    p_upd_note.add_argument("--content", metavar="C", help="Content text, or '-' to read from stdin")
    p_upd_note.add_argument("--source", metavar="SRC")

    # -- delete ---------------------------------------------------------------
    p_delete = sub.add_parser("delete", help="Delete stored content")
    delete_sub = p_delete.add_subparsers(dest="delete_type", metavar="TYPE")
    delete_sub.required = True

    for dtype in _ITEM_TYPES:
        p_del = delete_sub.add_parser(dtype, help=f"Delete a {dtype}")
        p_del.add_argument("id")

    # -- read -----------------------------------------------------------------
    p_read = sub.add_parser("read", help="Read full content of an item by ID")
    read_sub = p_read.add_subparsers(dest="read_type", metavar="TYPE")
    read_sub.required = True
    for rtype in _ITEM_TYPES:
        p_r = read_sub.add_parser(rtype, help=f"Read a {rtype} by ID")
        p_r.add_argument("id")

    # -- get ------------------------------------------------------------------
    p_get = sub.add_parser("get", help="Get a single item by ID (via resource URI)")
    get_sub = p_get.add_subparsers(dest="get_type", metavar="TYPE")
    get_sub.required = True

    for gtype in _ITEM_TYPES:
        p_g = get_sub.add_parser(gtype, help=f"Get a {gtype} by ID")
        p_g.add_argument("id")

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

    # -- list -----------------------------------------------------------------
    p_list = sub.add_parser("list", help="List content in a namespace")
    list_sub = p_list.add_subparsers(dest="list_type", metavar="TYPE")
    list_sub.required = True

    for ltype in _RESOURCE_TYPES:
        p_ls = list_sub.add_parser(ltype, help=f"List {ltype} in a namespace")
        p_ls.add_argument("namespace")

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
    api_key = _resolve(args.api_key, "MNEMOMATIC_API_KEY", "server", "api_key", cfg, "")

    # Resolve default search mode
    default_mode = _resolve(None, "MNEMOMATIC_SEARCH_MODE", "search", "mode", cfg, _DEFAULT_MODE)

    # Apply default search mode to search command (CLI flag still overrides)
    if args.command == "search" and args.mode is None:
        args.mode = default_mode

    base_url = server_url.rstrip("/") + "/mcp"

    try:
        client = MCPClient(base_url=base_url, api_key=api_key)
    except (RuntimeError, ValueError) as exc:
        _err(str(exc))

    pretty = args.pretty

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
            _run(lambda: client.call_tool("search", params), pretty)
        case "store":
            tool = f"store_{args.store_type}"
            _cmd_tool(tool, args, client, pretty,
                      _STORE_FIELDS[args.store_type], required=True)
        case "update":
            tool = f"update_{args.update_type}"
            _cmd_tool(tool, args, client, pretty,
                      _UPDATE_FIELDS[args.update_type], required=False)
        case "delete":
            tool = f"delete_{args.delete_type}"
            _run(lambda: client.call_tool(tool, {"id": args.id}), pretty)
        case "read":
            _run(lambda: client.call_tool("read", {
                "item_type": args.read_type, "id": args.id}), pretty)
        case "get":
            uri = _GET_URI[args.get_type].format(id=args.id)
            _run(lambda: client.read_resource(uri), pretty)
        case "tag":
            params: dict = {"item_id": args.id, "item_type": args.type}
            if args.add:
                params["add_tags"] = args.add
            if args.remove:
                params["remove_tags"] = args.remove
            _run(lambda: client.call_tool("tag", params), pretty)
        case "namespace":
            match args.ns_action:
                case "list":
                    _run(lambda: client.read_resource("mnemomatic://namespaces"), pretty)
                case "rename":
                    _run(lambda: client.call_tool("rename_namespace", {
                        "old_namespace": args.old_namespace,
                        "new_namespace": args.new_namespace}), pretty)
        case "list":
            _run(lambda: client.read_resource(
                f"mnemomatic://{args.list_type}/{args.namespace}"), pretty)
