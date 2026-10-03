"""Human-readable export of the store as a zip archive.

Layout: one folder per namespace, one subfolder per content type, one file
per item. File bodies are the content alone — a document's/note's ``content``
byte-faithfully, a knowledge entry's ``fact`` — so the files port cleanly
into any other system. Everything else (exact title/subject, ids, tags,
timestamps, per-item metadata) lives in a ``metadata.json`` sidecar in each
type folder, keyed by filename. ``export-info.json`` at the archive root
carries the manifest.

Vectors, document chunks, and FTS rows are derived data and deliberately not
exported: excluding them keeps the archive independent of the embedding
model, and an import re-embeds on the target server.

Filenames are sanitized titles; sanitization can collide (and zip archives
are routinely extracted onto case-insensitive filesystems), so collisions get
an id-prefix suffix. The exact original names are always recoverable from the
sidecars and the manifest's namespace map.
"""

import io
import json
import re
import zipfile
from datetime import datetime, timezone

from mnemomatic.db import _SPECS

# Windows-forbidden characters plus control chars; the superset is safe everywhere.
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_EXT_BY_MIME = {
    "text/markdown": ".md",
    "text/plain": ".txt",
    "application/json": ".json",
}
_MAX_NAME_LEN = 100

EXPORT_FORMAT = 1


def _safe_name(raw: str, fallback: str) -> str:
    """A filesystem-safe name derived from *raw*, or *fallback* if nothing survives.

    Spaces become underscores so the names are shell- and URL-friendly.
    """
    name = _INVALID_CHARS.sub("_", raw).strip(" .")
    name = name.replace(" ", "_")[:_MAX_NAME_LEN].rstrip("._")
    return name or fallback


def _unique(base: str, used: set[str], item_id: str) -> str:
    """*base*, suffixed with an id prefix when it collides case-insensitively."""
    if base.casefold() not in used:
        used.add(base.casefold())
        return base
    suffixed = f"{base}--{item_id[:8]}"
    used.add(suffixed.casefold())
    return suffixed


def _add_file(zf: zipfile.ZipFile, path: str, body: str, when: datetime) -> None:
    """Write one archive member, stamped with the item's updated_at."""
    info = zipfile.ZipInfo(path, date_time=when.timetuple()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, body)


def _export_section(zf: zipfile.ZipFile, folder: str,
                    items: list[tuple[str, str, str, str, dict, datetime]]) -> None:
    """Write one type folder: an extension-suffixed file per item + metadata.json.

    *items* rows are (id, display_name, extension, body, meta, updated_at);
    meta is the sidecar record (exact title/subject, tags, timestamps, ...).
    """
    used: set[str] = set()
    sidecar: dict[str, dict] = {}
    for item_id, display_name, ext, body, meta, updated in items:
        stem = _unique(_safe_name(display_name, item_id), used, item_id)
        filename = f"{stem}{ext}"
        _add_file(zf, f"{folder}/{filename}", body, updated)
        sidecar[filename] = meta
    _add_file(zf, f"{folder}/metadata.json",
              json.dumps(sidecar, indent=2, ensure_ascii=False),
              datetime.now(timezone.utc))


def _extension(item) -> str:
    """A document's extension follows its MIME type; everything else is Markdown."""
    return _EXT_BY_MIME.get(getattr(item, "mime_type", None), ".md")


def _sidecar_meta(spec, item) -> dict:
    """The sidecar record: every column but the body (which is the file) and
    the temporal bookkeeping (exports hold current items only), in column order."""
    meta = {}
    for col in spec.columns:
        if col not in (spec.snippet_field, "valid_until", "superseded_by"):
            value = getattr(item, col)
            meta[col] = value.isoformat() if isinstance(value, datetime) else value
    return meta


def build_export_zip(db, namespace: str | None = None, *,
                     server_version: str) -> tuple[bytes, str]:
    """Build the archive for one namespace (or all) and suggest a filename.

    Returns (zip bytes, filename). Type folders without items are omitted;
    a namespace with no items simply contributes nothing. Every read comes
    from one snapshot: exports run on a worker thread alongside writes, and
    a rename landing between two namespaces' reads would otherwise list an
    item twice, or not at all.
    """
    with db.snapshot():
        return _build_export_zip(db, namespace, server_version=server_version)


def _build_export_zip(db, namespace: str | None, *, server_version: str) -> tuple[bytes, str]:
    now = datetime.now(timezone.utc)
    namespaces = [namespace] if namespace else db.list_namespaces()

    counts = dict.fromkeys(_SPECS, 0)
    folder_by_ns: dict[str, str] = {}
    used_folders: set[str] = set()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ns in namespaces:
            by_table = {t: db.list_items(spec.item_type, ns) for t, spec in _SPECS.items()}
            if not any(by_table.values()):
                continue
            # Namespace folder names collide the same way filenames do.
            folder = _unique(_safe_name(ns, "namespace"), used_folders, ns)
            folder_by_ns[folder] = ns

            for table, items in by_table.items():
                if not items:
                    continue
                spec = _SPECS[table]
                counts[table] += len(items)
                _export_section(zf, f"{folder}/{table}", [
                    (it.id, getattr(it, spec.title_field), _extension(it),
                     getattr(it, spec.snippet_field), _sidecar_meta(spec, it), it.updated_at)
                    for it in items
                ])

        manifest = {
            "format": EXPORT_FORMAT,
            "exported_at": now.isoformat(),
            "server_version": server_version,
            "namespace_filter": namespace,
            "counts": counts,
            "namespaces": folder_by_ns,
        }
        _add_file(zf, "export-info.json",
                  json.dumps(manifest, indent=2, ensure_ascii=False), now)

    filename = f"mnemomatic-export-{now.date().isoformat()}.zip"
    return buf.getvalue(), filename
