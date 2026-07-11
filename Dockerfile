# syntax=docker/dockerfile:1
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
# Isolated stage: downloads the ONNX embedding model, never copied to lite.
#
# EmbeddingGemma-300m, community ONNX export with the full sentence-transformers
# stack (mean pooling, dense projection layers, normalization) baked into the
# graph as a `sentence_embedding` output. The INT8-quantized variant is
# published pre-made, so no quantization step is needed here. The graph file
# references its weights by the fixed name `model_quantized.onnx_data`, which
# must therefore keep that name next to model.onnx.
#
# Pinned to an immutable revision and verified against known SHA-256 digests so
# a compromised or moved upstream file fails the build instead of shipping.

FROM builder-base AS model-builder

RUN python3 << 'PYTHON_EOF'
import hashlib, os, urllib.request

REPO = "onnx-community/embeddinggemma-300m-ONNX"
REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
FILES = [
    ("onnx/model_quantized.onnx", "model.onnx",
     "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8"),
    ("onnx/model_quantized.onnx_data", "model_quantized.onnx_data",
     "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0"),
    ("tokenizer.json", "tokenizer.json",
     "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"),
]

os.makedirs("/app/model", exist_ok=True)
for remote, local, expected in FILES:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
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
