# syntax=docker/dockerfile:1

# Built-in embedding model for the full image. One of:
#   minilm                — all-MiniLM-L6-v2 (default): 384 dims, English,
#                           fastest (~20 ms/embed), tiny (~23 MB INT8)
#   multilingual-e5-small — 384 dims, ~100 languages, ~40 ms/embed, ~115 MB
#   embeddinggemma        — EmbeddingGemma-300m: 768 dims, best retrieval
#                           quality, multilingual, ~200 ms/embed, ~330 MB
# Selecting a model bakes its weights and a model_config.json (dimension, task
# prefixes, token limit) into the image — no runtime configuration needed.
# Switching models on an existing database requires one MNEMOMATIC_REINDEX=1
# restart (see docs/installation.md, "Switching Embedding Models").
ARG EMBED_MODEL=minilm

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
# minilm is downloaded as FP32 and quantized to INT8 here at build time — the
# same pipeline previous releases used, keeping its vectors compatible with
# databases embedded by those releases. The other models publish INT8 variants
# pre-made. EmbeddingGemma's graph references its weights by the fixed name
# `model_quantized.onnx_data`, which must keep that name next to model.onnx;
# its export also bakes the full sentence-transformers stack (mean pooling,
# dense projection layers, normalization) into the graph as a
# `sentence_embedding` output.

FROM builder-base AS model-builder

ARG EMBED_MODEL

# Quantization toolchain, only needed for the minilm FP32→INT8 conversion
RUN if [ "$EMBED_MODEL" = "minilm" ]; then \
        pip install --no-cache-dir --no-compile --target=/tmp/quant onnx onnxruntime sympy; \
    fi

RUN python3 << 'PYTHON_EOF'
import hashlib, json, os, urllib.request

MODELS = {
    "minilm": {
        "repo": "Qdrant/all-MiniLM-L6-v2-onnx",
        "revision": "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079",
        "files": [
            ("model.onnx", "model.onnx",
             "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5"),
            ("tokenizer.json", "tokenizer.json",
             "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
        ],
        "quantize": True,
        "config": {"model": "all-MiniLM-L6-v2", "dim": 384, "max_tokens": 512,
                   "query_prefix": "", "doc_prefix": ""},
    },
    "multilingual-e5-small": {
        "repo": "Xenova/multilingual-e5-small",
        "revision": "761b726dd34fb83930e26aab4e9ac3899aa1fa78",
        "files": [
            ("onnx/model_quantized.onnx", "model.onnx",
             "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193"),
            ("tokenizer.json", "tokenizer.json",
             "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
        ],
        "quantize": False,
        "config": {"model": "multilingual-e5-small", "dim": 384, "max_tokens": 512,
                   "query_prefix": "query: ", "doc_prefix": "passage: "},
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
        "quantize": False,
        "config": {"model": "embeddinggemma-300m", "dim": 768, "max_tokens": 2048,
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

# FP32 → INT8 for minilm (~4x smaller, 2-3x faster on CPU). The quantization
# toolchain lives in /tmp and is never copied to the runtime image.
RUN if [ "$EMBED_MODEL" = "minilm" ]; then \
        PYTHONPATH=/tmp/quant python3 -c "\
from onnxruntime.quantization import quantize_dynamic, QuantType; import os; \
orig = os.path.getsize('/app/model/model.onnx'); \
quantize_dynamic('/app/model/model.onnx', '/app/model/model_int8.onnx', weight_type=QuantType.QUInt8); \
quant = os.path.getsize('/app/model/model_int8.onnx'); \
os.replace('/app/model/model_int8.onnx', '/app/model/model.onnx'); \
print(f'Quantized: {orig/1024/1024:.1f}MB -> {quant/1024/1024:.1f}MB')"; \
    fi

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
