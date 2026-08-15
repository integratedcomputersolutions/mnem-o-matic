"""Administration: namespace operations, health, export, and settings."""
import json
import logging
from importlib.metadata import PackageNotFoundError, version

from mnemomatic import config, runtime
from mnemomatic.db import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_THRESHOLD
from mnemomatic.runtime import _audit, mcp

logger = logging.getLogger("mnemomatic")


@mcp.resource("mnemomatic://health")
def health() -> str:
    """Health check endpoint. Returns server status and configuration."""
    embedder = runtime._embedder()
    embedding_mode = embedder.mode if embedder is not None else "FTS-only (no embedder)"

    return json.dumps({
        "status": "ok",
        "version": _server_version(),
        "embedding_mode": embedding_mode,
        "auth_enabled": bool(config.API_KEY),
    })


def _server_version() -> str:
    # Distribution name is "mnemomatic-server"; fall back gracefully when
    # running from a source tree with no installed metadata.
    try:
        return version("mnemomatic-server")
    except PackageNotFoundError:
        return "unknown"


def _make_export(namespace: str | None) -> tuple[bytes, str]:
    """Build the export archive; shared by /export and the web viewer."""
    from mnemomatic.export import build_export_zip

    return build_export_zip(runtime._db(), namespace, server_version=_server_version())


async def _export_route(request):
    """GET /export[?namespace=...] — zip download, behind the Bearer middleware."""
    from starlette.responses import JSONResponse, Response

    namespace = request.query_params.get("namespace") or None
    if namespace and namespace not in runtime._db().list_namespaces():
        return JSONResponse(
            {"error": "Namespace not found", "details": f"No items in namespace {namespace!r}"},
            status_code=404,
        )
    data, filename = _make_export(namespace)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Hugging Face model cards for the models the Docker build can bake in, keyed
# by the exact names it writes to model_config.json. Links point at the source
# model card (weights provenance, benchmarks, license), not the ONNX mirror
# the image downloads from. Unknown/external models simply get no link.
_HF_MODEL_PAGES = {
    "all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "gte-multilingual-base": "https://huggingface.co/Alibaba-NLP/gte-multilingual-base",
    "embeddinggemma-300m": "https://huggingface.co/google/embeddinggemma-300m",
    "amaretto-embed-148m": "https://huggingface.co/AmarettoLabs/amaretto-embed-148m",
}


def _settings_info() -> dict:
    """Configuration snapshot for the web viewer's settings page.

    Reads EMBEDDING_DIM through the db module so it reflects the value the
    running Database actually used (tests patch it there).
    """
    from mnemomatic import db as db_module
    from mnemomatic.embeddings import EMBED_API, MODEL_MAX_TOKENS

    embedder = runtime._embedder()
    model_name = config.embed_identity()["embed_model"] or None
    info = {
        "version": _server_version(),
        "mode": embedder.mode if embedder is not None else "FTS-only (no embedder)",
        "model": model_name,
        "model_url": _HF_MODEL_PAGES.get(model_name),
        "dim_configured": db_module.EMBEDDING_DIM,
        "dim_database": runtime._db().stored_embed_dim(),
        "model_database": runtime._db().stored_embed_identity().get("embed_model") or None,
        "query_prefix": config.EMBED_QUERY_PREFIX,
        "doc_prefix": config.EMBED_DOC_PREFIX,
        "chunk_threshold": CHUNK_THRESHOLD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    if config.EMBED_URL:
        info["endpoint_url"] = config.EMBED_URL
        info["wire_api"] = EMBED_API
    else:
        info["max_tokens"] = MODEL_MAX_TOKENS
    return info


@mcp.tool(annotations=config.ANN_DELETE)
def delete_namespace(namespace: str) -> dict:
    """Delete all items in a namespace.

    Removes every document, knowledge entry, and note in the given namespace in
    a single atomic operation. Each item's final state is kept as a revision, so
    individual items can be recovered via list_revisions + restore — but there
    is no one-call undo for the whole namespace, so treat this as destructive.
    If you only want to reorganize content, use rename_namespace instead.

    Args:
        namespace: The namespace to delete.
    """
    counts = runtime._db().delete_namespace(namespace)
    _audit("delete_namespace", namespace=namespace, deleted=sum(counts.values()))
    return {
        "namespace": namespace,
        "deleted": counts,
        "total": sum(counts.values()),
    }


@mcp.tool(annotations=config.ANN_UPDATE)
def rename_namespace(old_namespace: str, new_namespace: str) -> dict:
    """Rename a namespace across all documents, knowledge entries, and notes.

    Moves every item in old_namespace to new_namespace atomically. If
    new_namespace already exists this acts as a merge: on a title/subject
    collision the moved item replaces the target's item (the same upsert
    semantics as the store tools). Check `replaced` in the response to see
    how many target items were overwritten.

    Args:
        old_namespace: The namespace to rename.
        new_namespace: The new name for the namespace. Must differ from old_namespace.
    """
    try:
        counts, replaced = runtime._db().rename_namespace(old_namespace, new_namespace)
    except ValueError as e:
        return {"error": str(e)}
    _audit("rename_namespace", namespace=old_namespace, new_namespace=new_namespace,
           moved=sum(counts.values()), replaced=sum(replaced.values()))
    return {
        "old_namespace": old_namespace,
        "new_namespace": new_namespace,
        "renamed": counts,
        "replaced": replaced,
        "total": sum(counts.values()),
    }
