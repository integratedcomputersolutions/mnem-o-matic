# syntax=docker/dockerfile:1
# ── Builder base ──────────────────────────────────────────────────────────────
# Shared setup: system tools and source code only

FROM python:3.11-slim AS builder-base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc binutils && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# ── Model download + quantization ─────────────────────────────────────────────
# Isolated stage: downloads and quantizes the ONNX model, never copied to lite

FROM builder-base AS model-builder

# Download the embedding model using fastembed (isolated — never copied to runtime)
RUN pip install --no-cache-dir --no-compile --target=/tmp/dl fastembed
ENV FASTEMBED_CACHE_PATH=/app/fastembed-cache
RUN PYTHONPATH=/tmp/dl \
    python3 -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

# Extract only model.onnx + tokenizer.json from wherever fastembed cached them
RUN python3 << 'PYTHON_EOF'
import glob, os, shutil, hashlib

os.makedirs('/app/model', exist_ok=True)
onnx = glob.glob('/app/fastembed-cache/**/*.onnx', recursive=True)
tok  = glob.glob('/app/fastembed-cache/**/tokenizer.json', recursive=True)
shutil.copy(onnx[0], '/app/model/model.onnx')
shutil.copy(tok[0],  '/app/model/tokenizer.json')

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

onnx_hash = sha256_file('/app/model/model.onnx')
tok_hash = sha256_file('/app/model/tokenizer.json')
print(f'ONNX: {onnx[0]}')
print(f'ONNX SHA256: {onnx_hash}')
print(f'Tokenizer: {tok[0]}')
print(f'Tokenizer SHA256: {tok_hash}')
PYTHON_EOF

# Quantize the model from FP32 to INT8: ~4x smaller, ~2-3x faster inference on CPU
# onnx package is only needed here at build time, never copied to the runtime image
RUN pip install --no-cache-dir --no-compile --target=/tmp/quant onnx && \
    PYTHONPATH=/tmp/dl:/tmp/quant \
    python3 -c "from onnxruntime.quantization import quantize_dynamic, QuantType; import os; orig=os.path.getsize('/app/model/model.onnx'); quantize_dynamic('/app/model/model.onnx','/app/model/model_int8.onnx',weight_type=QuantType.QUInt8); quant=os.path.getsize('/app/model/model_int8.onnx'); os.replace('/app/model/model_int8.onnx','/app/model/model.onnx'); print(f'Quantized: {orig/1024/1024:.1f}MB -> {quant/1024/1024:.1f}MB')"

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
#   huggingface_hub/hf_xet/fsspec/pyyaml — pulled in by fastembed for model download only
#   pygments — syntax highlighting pulled in by rich, not useful in a server
RUN rm -rf \
    /install/lib/python3.11/site-packages/sympy \
    /install/lib/python3.11/site-packages/sympy-*.dist-info \
    /install/lib/python3.11/site-packages/mpmath \
    /install/lib/python3.11/site-packages/mpmath-*.dist-info \
    /install/lib/python3.11/site-packages/huggingface_hub \
    /install/lib/python3.11/site-packages/huggingface_hub-*.dist-info \
    /install/lib/python3.11/site-packages/hf_xet \
    /install/lib/python3.11/site-packages/hf_xet-*.dist-info \
    /install/lib/python3.11/site-packages/fsspec \
    /install/lib/python3.11/site-packages/fsspec-*.dist-info \
    /install/lib/python3.11/site-packages/yaml \
    /install/lib/python3.11/site-packages/PyYAML-*.dist-info \
    /install/lib/python3.11/site-packages/pygments \
    /install/lib/python3.11/site-packages/Pygments-*.dist-info

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

# Strip packages not needed at runtime
RUN rm -rf \
    /install/lib/python3.11/site-packages/pygments \
    /install/lib/python3.11/site-packages/Pygments-*.dist-info

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
