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


def build_export_zip(db, namespace: str | None = None, *,
                     server_version: str) -> tuple[bytes, str]:
    """Build the archive for one namespace (or all) and suggest a filename.

    Returns (zip bytes, filename). Type folders without items are omitted;
    a namespace with no items simply contributes nothing.
    """
    now = datetime.now(timezone.utc)
    namespaces = [namespace] if namespace else db.list_namespaces()

    counts = {"documents": 0, "knowledge": 0, "notes": 0}
    folder_by_ns: dict[str, str] = {}
    used_folders: set[str] = set()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ns in namespaces:
            docs = db.list_documents(ns)
            knowledge = db.list_knowledge(ns)
            notes = db.list_notes(ns)
            if not (docs or knowledge or notes):
                continue
            # Namespace folder names collide the same way filenames do.
            folder = _unique(_safe_name(ns, "namespace"), used_folders, ns)
            folder_by_ns[folder] = ns

            if docs:
                counts["documents"] += len(docs)
                _export_section(zf, f"{folder}/documents", [
                    (d.id, d.title, _EXT_BY_MIME.get(d.mime_type, ".md"), d.content,
                     {"id": d.id, "namespace": ns, "title": d.title,
                      "mime_type": d.mime_type, "tags": d.tags, "metadata": d.metadata,
                      "created_at": d.created_at.isoformat(),
                      "updated_at": d.updated_at.isoformat()},
                     d.updated_at)
                    for d in docs
                ])
            if knowledge:
                counts["knowledge"] += len(knowledge)
                _export_section(zf, f"{folder}/knowledge", [
                    (k.id, k.subject, ".md", k.fact,
                     {"id": k.id, "namespace": ns, "subject": k.subject,
                      "confidence": k.confidence, "source": k.source,
                      "tags": k.tags, "metadata": k.metadata,
                      "created_at": k.created_at.isoformat(),
                      "updated_at": k.updated_at.isoformat()},
                     k.updated_at)
                    for k in knowledge
                ])
            if notes:
                counts["notes"] += len(notes)
                _export_section(zf, f"{folder}/notes", [
                    (n.id, n.title, ".md", n.content,
                     {"id": n.id, "namespace": ns, "title": n.title,
                      "source": n.source, "tags": n.tags, "metadata": n.metadata,
                      "created_at": n.created_at.isoformat(),
                      "updated_at": n.updated_at.isoformat()},
                     n.updated_at)
                    for n in notes
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
