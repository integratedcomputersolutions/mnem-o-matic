# syntax=docker/dockerfile:1

# Built-in embedding model for the full image. One of:
#   arctic-embed-xs       — Snowflake Arctic Embed xs (default): 384 dims,
#                           English, fastest (~10-15 ms/embed), tiny (~23 MB
#                           INT8). CLS pooling + a query prefix
#   gte-multilingual-base — 768 dims, ~70 languages, near-MiniLM query speed
#                           (~12 ms/embed), 8192-token context, ~325 MB
#   embeddinggemma        — EmbeddingGemma-300m: 768 dims, best retrieval
#                           quality, multilingual, ~200 ms/embed, ~330 MB
#   amaretto-embed-148m   — sliced EmbeddingGemma distillation: 768 dims,
#                           8 Latin-script languages + code, near-Gemma quality
#                           at ~130 ms/embed and ~297 MB (weight-only INT8)
# Selecting a model bakes its weights and a model_config.json (dimension, task
# prefixes, token limit) into the image — no runtime configuration needed.
# Switching models on an existing database requires a reindex — set
# MNEMOMATIC_REINDEX=auto (see docs/installation.md, "Switching Embedding
# Models"). Same-dimension swaps are caught by the recorded model identity.
ARG EMBED_MODEL=arctic-embed-xs

# ── Builder base ──────────────────────────────────────────────────────────────
# Shared setup: system tools and source code only

FROM python:3.11-slim AS builder-base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc binutils && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# ── Model download ─────────────────────────────────────────────────────────────
# Isolated stage: downloads the ONNX embedding model selected by EMBED_MODEL,
# never copied to lite. Alongside the weights it writes model_config.json
# (model name, embedding dimension, token limit, task prefixes), which the
# server reads for its defaults — so the build arg is the only knob.
#
# Every download is pinned to an immutable revision and verified against a
# known SHA-256 digest so a compromised or moved upstream file fails the build
# instead of shipping.
#
# Every model here publishes a pre-quantized INT8 variant, so nothing is
# quantized at build time. EmbeddingGemma's graph references its weights by the
# fixed name `model_quantized.onnx_data`, which must keep that name next to
# model.onnx; its export also bakes the full sentence-transformers stack (mean pooling,
# dense projection layers, normalization) into the graph as a
# `sentence_embedding` output.

FROM builder-base AS model-builder

ARG EMBED_MODEL

RUN python3 << 'PYTHON_EOF'
import hashlib, json, os, urllib.request

MODELS = {
    # Snowflake's retrain of all-MiniLM-L6-v2, which this replaces: same
    # architecture, same 384 dims, same ~23 MB INT8 footprint, and +8 nDCG@10
    # on MTEB Retrieval-15 (50.15 vs 41.95). Two differences the config below
    # carries: it pools the CLS token rather than the mean, and queries take a
    # prefix while documents take none.
    "arctic-embed-xs": {
        "repo": "Snowflake/snowflake-arctic-embed-xs",
        "revision": "d8c86521100d3556476a063fc2342036d45c106f",
        "files": [
            ("onnx/model_int8.onnx", "model.onnx",
             "e6aa5e656466a73d7c3111e9a3378bd13e5b93af30eaac2b3f13fd56692589a1"),
            ("tokenizer.json", "tokenizer.json",
             "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"),
        ],
        "config": {"model": "snowflake-arctic-embed-xs", "dim": 384, "max_tokens": 512,
                   "pooling": "cls",
                   "query_prefix": "Represent this sentence for searching relevant passages: ",
                   "doc_prefix": ""},
    },
    "gte-multilingual-base": {
        "repo": "onnx-community/gte-multilingual-base",
        "revision": "2edbf5e672aab465f9ed4c154a8b61791c082c69",
        "files": [
            ("onnx/model_quantized.onnx", "model.onnx",
             "ab2bd164ebd8ca9003dc49a981b611e849b5d326f504c8873ba76e07fa6c0082"),
            ("tokenizer.json", "tokenizer.json",
             "3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20"),
        ],
        "config": {"model": "gte-multilingual-base", "dim": 768, "max_tokens": 8192,
                   "query_prefix": "", "doc_prefix": ""},
    },
    "embeddinggemma": {
        "repo": "onnx-community/embeddinggemma-300m-ONNX",
        "revision": "5090578d9565bb06545b4552f76e6bc2c93e4a66",
        "files": [
            ("onnx/model_quantized.onnx", "model.onnx",
             "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8"),
            ("onnx/model_quantized.onnx_data", "model_quantized.onnx_data",
             "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0"),
            ("tokenizer.json", "tokenizer.json",
             "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"),
        ],
        "config": {"model": "embeddinggemma-300m", "dim": 768, "max_tokens": 2048,
                   "query_prefix": "task: search result | query: ",
                   "doc_prefix": "title: none | text: "},
    },
    # Weight-only INT8 (MatMulNBits): activations stay FP32, so fidelity holds
    # at long inputs (cos vs torch >= 0.999 through 2048 tokens) and the weights
    # stay quantized in RAM. The op is com.microsoft domain — fine for the
    # bundled onnxruntime, not portable to other ONNX runtimes.
    "amaretto-embed-148m": {
        "repo": "AmarettoLabs/amaretto-embed-148m-ONNX",
        "revision": "f27ee11523834d96eb43e293afb878bf943701bf",  # v1.1.0
        "files": [
            ("model_int8.onnx", "model.onnx",
             "8bf1cc9663913c7fc7c3d3787fc42869d5a120ebeef297c3e663f4bf5a5221ef"),
            ("tokenizer.json", "tokenizer.json",
             "557d686df474db4ed5612819752c7b1e9996e697170f9ae74577ee616cb4179c"),
        ],
        "config": {"model": "amaretto-embed-148m", "dim": 768, "max_tokens": 2048,
                   "query_prefix": "task: search result | query: ",
                   "doc_prefix": "title: none | text: "},
    },
}

choice = os.environ.get("EMBED_MODEL", "")
if choice not in MODELS:
    raise SystemExit(
        f"Unknown EMBED_MODEL {choice!r} — expected one of: {', '.join(MODELS)}"
    )
model = MODELS[choice]

os.makedirs("/app/model", exist_ok=True)
for remote, local, expected in model["files"]:
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{remote}"
    dest = f"/app/model/{local}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise SystemExit(f"SHA256 mismatch for {remote}: got {h.hexdigest()}, expected {expected}")
    print(f"  OK {local}: {os.path.getsize(dest) / 1024 / 1024:.1f} MB, sha256 {expected}")

with open("/app/model/model_config.json", "w") as f:
    json.dump(model["config"], f, indent=2)
print(f"Wrote model_config.json: {model['config']}")
PYTHON_EOF


# ── Full builder ───────────────────────────────────────────────────────────────
# Installs all deps including the ML stack (onnxruntime, numpy, tokenizers)

FROM builder-base AS builder-full

RUN pip install --no-cache-dir --no-compile --prefix=/install ".[onnx]"

# Strip onnxruntime extras not needed for CPU inference
RUN find /install/lib/python3.11/site-packages/onnxruntime -maxdepth 1 -type d \
    \( -name 'transformers' -o -name 'quantization' -o -name 'tools' -o -name 'datasets' \) \
    -exec rm -rf {} + 2>/dev/null || true

# Strip pip/setuptools (not needed at runtime)
RUN rm -rf /install/lib/python3.11/site-packages/pip* \
           /install/lib/python3.11/site-packages/setuptools* \
           /install/lib/python3.11/site-packages/*.dist-info/RECORD

# Strip packages not needed at runtime:
#   sympy/mpmath  — onnxruntime optional deps for shape inference (build-time only)
#   flatbuffers/packaging/protobuf(google) — onnxruntime declares these but only its
#     training/quantization/conversion tooling (already removed above) imports them; a
#     real InferenceSession loads and runs the model entirely in native C++.
#   huggingface_hub/hf_xet/fsspec/pyyaml/tqdm/filelock — pulled in by fastembed for model
#     download only (tqdm = progress bars, filelock = cache locking); unused at runtime.
#   rich/pygments/markdown_it/mdurl/typer/shellingham — pulled in transitively via the MCP
#     SDK's CLI deps and unused by the server (which logs via the stdlib). rich is removed,
#     not just pygments: rich imports pygments lazily when rendering tracebacks, so stripping
#     pygments alone leaves rich to crash at runtime with "No module named 'pygments'".
#     With rich gone, the SDK's optional-import falls back cleanly to a plain log handler.
RUN rm -rf \
    /install/lib/python3.11/site-packages/sympy \
    /install/lib/python3.11/site-packages/sympy-*.dist-info \
    /install/lib/python3.11/site-packages/mpmath \
    /install/lib/python3.11/site-packages/mpmath-*.dist-info \
    /install/lib/python3.11/site-packages/flatbuffers \
    /install/lib/python3.11/site-packages/flatbuffers-*.dist-info \
    /install/lib/python3.11/site-packages/packaging \
    /install/lib/python3.11/site-packages/packaging-*.dist-info \
    /install/lib/python3.11/site-packages/google \
    /install/lib/python3.11/site-packages/protobuf-*.dist-info \
    /install/lib/python3.11/site-packages/huggingface_hub \
    /install/lib/python3.11/site-packages/huggingface_hub-*.dist-info \
    /install/lib/python3.11/site-packages/hf_xet \
    /install/lib/python3.11/site-packages/hf_xet-*.dist-info \
    /install/lib/python3.11/site-packages/fsspec \
    /install/lib/python3.11/site-packages/fsspec-*.dist-info \
    /install/lib/python3.11/site-packages/tqdm \
    /install/lib/python3.11/site-packages/tqdm-*.dist-info \
    /install/lib/python3.11/site-packages/filelock \
    /install/lib/python3.11/site-packages/filelock-*.dist-info \
    /install/lib/python3.11/site-packages/yaml \
    /install/lib/python3.11/site-packages/PyYAML-*.dist-info \
    /install/lib/python3.11/site-packages/pygments \
    /install/lib/python3.11/site-packages/Pygments-*.dist-info \
    /install/lib/python3.11/site-packages/rich \
    /install/lib/python3.11/site-packages/rich-*.dist-info \
    /install/lib/python3.11/site-packages/markdown_it \
    /install/lib/python3.11/site-packages/markdown_it_py-*.dist-info \
    /install/lib/python3.11/site-packages/mdurl \
    /install/lib/python3.11/site-packages/mdurl-*.dist-info \
    /install/lib/python3.11/site-packages/typer \
    /install/lib/python3.11/site-packages/typer-*.dist-info \
    /install/lib/python3.11/site-packages/shellingham \
    /install/lib/python3.11/site-packages/shellingham-*.dist-info

# Strip debug symbols from all shared libraries
RUN find /install -name '*.so*' -type f -exec strip --strip-debug {} + 2>/dev/null || true

# Strip __pycache__ and .pyc files
RUN find /install -name '*.pyc' -delete && \
    find /install -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ── Lite builder ───────────────────────────────────────────────────────────────
# Installs only core deps — no ML stack. Semantic search requires MNEMOMATIC_EMBED_URL.

FROM builder-base AS builder-lite

RUN pip install --no-cache-dir --no-compile --prefix=/install .

# Strip pip/setuptools (not needed at runtime)
RUN rm -rf /install/lib/python3.11/site-packages/pip* \
           /install/lib/python3.11/site-packages/setuptools* \
           /install/lib/python3.11/site-packages/*.dist-info/RECORD

# Strip packages not needed at runtime:
#   rich/pygments/markdown_it/mdurl/typer/shellingham — MCP SDK CLI deps, unused by the
#     server. rich is removed too (not just pygments): rich imports pygments lazily when
#     rendering tracebacks, so stripping pygments alone leaves rich to crash at runtime.
RUN rm -rf \
    /install/lib/python3.11/site-packages/pygments \
    /install/lib/python3.11/site-packages/Pygments-*.dist-info \
    /install/lib/python3.11/site-packages/rich \
    /install/lib/python3.11/site-packages/rich-*.dist-info \
    /install/lib/python3.11/site-packages/markdown_it \
    /install/lib/python3.11/site-packages/markdown_it_py-*.dist-info \
    /install/lib/python3.11/site-packages/mdurl \
    /install/lib/python3.11/site-packages/mdurl-*.dist-info \
    /install/lib/python3.11/site-packages/typer \
    /install/lib/python3.11/site-packages/typer-*.dist-info \
    /install/lib/python3.11/site-packages/shellingham \
    /install/lib/python3.11/site-packages/shellingham-*.dist-info

# Strip debug symbols from all shared libraries
RUN find /install -name '*.so*' -type f -exec strip --strip-debug {} + 2>/dev/null || true

# Strip __pycache__ and .pyc files
RUN find /install -name '*.pyc' -delete && \
    find /install -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ── Runtime: full ──────────────────────────────────────────────────────────────

FROM gcr.io/distroless/python3-debian12 AS full

WORKDIR /app

COPY --from=builder-full /install /usr/local
COPY --from=model-builder /app/model /app/model

ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
ENV MNEMOMATIC_DB_PATH=/data/mnemomatic.db
ENV MNEMOMATIC_HOST=0.0.0.0
ENV MNEMOMATIC_PORT=8000

EXPOSE 8000

CMD ["-c", "from mnemomatic.server import main; main()"]

# ── Runtime: lite ──────────────────────────────────────────────────────────────

FROM gcr.io/distroless/python3-debian12 AS lite

WORKDIR /app

COPY --from=builder-lite /install /usr/local

ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
ENV MNEMOMATIC_DB_PATH=/data/mnemomatic.db
ENV MNEMOMATIC_HOST=0.0.0.0
ENV MNEMOMATIC_PORT=8000

EXPOSE 8000

CMD ["-c", "from mnemomatic.server import main; main()"]
