# Installation

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [mkcert](https://github.com/FiloSottile/mkcert) — for generating locally-trusted TLS certificates (LAN deployments)

## Deployment Profiles

| Profile           | Image size | Embeddings                          | Semantic search |
| ----------------- | ---------- | ----------------------------------- | --------------- |
| `full` (default)  | ~320–650 MB | Built-in ONNX model (CPU), selectable at build time | Yes |
| `lite` + Ollama   | ~120 MB    | External via `MNEMOMATIC_EMBED_URL` | Yes             |
| `lite` (FTS-only) | ~120 MB    | None                                | No              |

Choose the profile that fits your setup. The `full` image is self-contained and works out of the box; its bundled embedding model is chosen with the `EMBED_MODEL` build argument (see [Choosing the built-in embedding model](#choosing-the-built-in-embedding-model)). The `lite` image is significantly smaller and delegates embedding to an Ollama instance (or any compatible API), or runs keyword-only search if no embedder is configured.

## TLS Setup (LAN deployments)

When running on a machine that other devices on your network will connect to, use the included Caddy reverse proxy for HTTPS. Caddy terminates TLS and proxies to the Mnem-O-matic container — the app itself stays HTTP-only on the internal Docker network.

### 1. Install mkcert and create a local CA

[mkcert](https://github.com/FiloSottile/mkcert) creates a local certificate authority that your operating system trusts, so generated certificates work without browser/client warnings.

```bash
# macOS
brew install mkcert
mkcert -install

# Linux (Debian/Ubuntu)
sudo apt install mkcert
mkcert -install

# Windows
winget install FiloSottile.mkcert
mkcert -install
```

`mkcert -install` adds the CA to your system trust store. **Repeat this on every client device** that will connect to Mnem-O-matic.

### 2. Generate a certificate for your server

On the **server machine**, generate a certificate covering its hostname and/or IP address:

```bash
cd mnemomatic
mkcert -cert-file certs/cert.pem -key-file certs/key.pem \
    your-server-hostname your-server-ip 192.168.1.x localhost 127.0.0.1
```

Replace `your-server-hostname` and `your-server-ip` with the actual hostname and LAN IP of the server machine. Include all names clients might use to reach it. The generated files go into the `certs/` directory (gitignored).

### 3. Trust the CA on client devices

Copy the mkcert root CA certificate to each client device and trust it:

```bash
# On the server, find the CA location
mkcert -CAROOT
# e.g. /home/user/.local/share/mkcert

# Copy rootCA.pem to each client and trust it
# macOS: double-click → Keychain → set to "Always Trust"
# Windows: double-click → Install Certificate → Trusted Root CAs
# Linux: copy to /usr/local/share/ca-certificates/ and run update-ca-certificates
```

Alternatively, install mkcert on each client machine and run `mkcert -install` — they will share the same CA if you copy the `rootCA.pem` and `rootCA-key.pem` files from the server's CAROOT directory to the client's CAROOT directory first.

## Quick Start (Pre-built Images)

Pre-built images for `linux/amd64` and `linux/arm64` are published to the GitHub Container Registry on every release. No build step required.

### Full image (recommended)

```bash
# Generate TLS certificates (see TLS Setup above)
mkdir -p certs
mkcert -cert-file certs/cert.pem -key-file certs/key.pem your-server-hostname your-server-ip

# Create a data directory
mkdir -p data

# Pull and run
docker run -d \
  --name mnemomatic \
  -p 8000:8000 \
  -v "$(pwd)/data:/data" \
  -e MNEMOMATIC_API_KEY=your-secret-key \
  ghcr.io/integratedcomputersolutions/mnem-o-matic:latest-full
```

Or with `docker-compose.yml` — replace the `build:` block with the pre-built image:

```yaml
services:
  mnemomatic:
    image: ghcr.io/integratedcomputersolutions/mnem-o-matic:latest-full
    volumes:
      - ./data:/data
    environment:
      - MNEMOMATIC_API_KEY=your-secret-key
```

Then:

```bash
docker compose up -d
```

### Lite image with Ollama (pre-built)

```yaml
services:
  mnemomatic:
    image: ghcr.io/integratedcomputersolutions/mnem-o-matic:latest-lite
    volumes:
      - ./data:/data
    environment:
      - MNEMOMATIC_API_KEY=your-secret-key
      - MNEMOMATIC_EMBED_URL=http://host.docker.internal:11434/v1/embeddings
      - MNEMOMATIC_EMBED_MODEL=nomic-embed-text
      - MNEMOMATIC_EMBED_DIM=768
```

Any OpenAI-compatible embedding endpoint works the same way — Ollama's `/v1/embeddings` (shown above), llama.cpp's `llama-server --embeddings`, vLLM, or LM Studio. For Ollama's native `/api/embeddings` endpoint, add `MNEMOMATIC_EMBED_API=ollama`.

### Available image tags

| Tag | Description |
|-----|-------------|
| `latest-full` | Latest release, built-in ONNX embeddings |
| `latest-lite` | Latest release, no ML stack |
| `1.2.3-full` / `1.2.3-lite` | Exact version |
| `1.2-full` / `1.2-lite` | Minor floating tag |
| `1-full` / `1-lite` | Major floating tag |

---

## Build and Run

If you prefer to build from source (required for local development or unreleased changes):

### Full image (default)

```bash
# Clone the repository
git clone git@github.com:integratedcomputersolutions/mnem-o-matic.git
cd mnem-o-matic

# Generate TLS certificates (see TLS Setup above)
mkcert -cert-file certs/cert.pem -key-file certs/key.pem your-server-hostname your-server-ip

# Build and start
docker compose up --build
```

The server is accessible at `https://your-server-hostname/mcp`.

The first build takes a few minutes — it downloads the embedding model (checksum-verified; ~90–330 MB depending on `EMBED_MODEL`). Subsequent builds use the cached layer.

### Choosing the built-in embedding model

The `full` image bundles one of four embedding models, selected with the `EMBED_MODEL` build argument:

| `EMBED_MODEL` | Dimensions | Languages | Query embed (CPU) | Model size | RAM (running) | Notes |
| ------------- | ---------- | --------- | ----------------- | ---------- | ------------- | ----- |
| `arctic-embed-xs` (default) | 384 | English | ~10–15 ms | ~23 MB | ~240 MB | Snowflake Arctic Embed xs — fastest and smallest; Apache-2.0. CLS pooling and a query prefix, both applied automatically |
| `amaretto-embed-148m` | 768 | 8 Latin-script + code | ~130 ms | ~297 MB | ~470 MB | A vocabulary-sliced distillation of EmbeddingGemma — most of its retrieval quality at ~40% of the RAM; 2048-token context; weights under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| `gte-multilingual-base` | 768 | ~70 | ~12 ms | ~325 MB | ~880 MB | Strong multilingual retrieval at near-arctic query speed; 8192-token context; no task prefixes |
| `embeddinggemma` | 768 | 100+ | ~160–225 ms | ~330 MB | ~1.2 GB | EmbeddingGemma-300m — best retrieval quality, 2048-token context; weights under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

RAM figures are steady-state container usage measured on the same production deployment (they include the server itself plus SQLite's page cache and memory-mapped database file, not just the model). The INT8 weights are compact on disk, but `onnxruntime` expands parts of the larger models at load time, so their resident memory runs well above the model file size.

#### Which one should I pick?

The models were compared on the same real-world corpus (~90 items of technical notes and documents) on a modest x86 server, measuring end-to-end MCP search latency and ranking quality on probe queries:

- **`arctic-embed-xs` — English content, smallest image.** The lightest option: ~240 MB of memory and a query embedded in ~10–15 ms. Snowflake's retrain of all-MiniLM-L6-v2 on the same architecture, scoring **50.15 vs 41.95** on MTEB Retrieval-15 — a ~20% relative gain for the same footprint, which is why it replaced MiniLM as the default. English only. One known limit, found on a real store rather than a benchmark: on corpora where most content covers closely related subject matter it discriminates poorly, scoring everything in a narrow band. If your store is dense, homogeneous technical material, prefer `amaretto-embed-148m` or `embeddinggemma`.
- **`amaretto-embed-148m` — near-EmbeddingGemma quality on a smaller budget.** A vocabulary-sliced distillation of EmbeddingGemma (262k → 60.5k token vocabulary, 148M encoder parameters) that retains ~99.5% of the teacher's retrieval nDCG@10 while using roughly 40% of its RAM and about half its query latency. It takes the same task prefixes as EmbeddingGemma and, having been distilled to preserve the teacher's vector space, ranks much like it: on technical content probes it put the correct chunk first with clear margins, and Italian queries retrieved English content at least as well as the English equivalents did. Two limits worth knowing: coverage is 8 Latin-script languages plus code — non-Latin scripts fall back to byte-level tokens and degrade badly — and on generic, low-jargon queries its score range compresses, so rank order among near-ties is less dependable than EmbeddingGemma's.
- **`gte-multilingual-base` — multilingual content without the latency cost.** Queries embed in ~12 ms — as fast as arctic-embed-xs — because most of its parameters sit in the vocabulary matrix, which costs little for short inputs (longer chunks take ~150 ms each at store time). Cross-lingual retrieval is strong: in probes, an Italian query separated the correct English answer from a distractor *better* than the equivalent English query did (0.71 vs 0.32). No task prefixes needed. Its trade-off is image size (~325 MB, Gemma-class). *(`multilingual-e5-small` was evaluated for this slot and dropped: on English content it underperformed the then-default MiniLM, and its compressed score range made rankings unreliable.)*
- **`embeddinggemma` — best retrieval quality, paraphrase-robust.** The quality winner: it separates relevant from irrelevant results by an order of magnitude larger margins and reliably resolves zero-word-overlap queries ("how do I get back into my account?" → password-reset content) that smaller models rank flat or miss. The cost is CPU time: ~200 ms per query embedding (imperceptible in agent workflows) and ~0.5–1 s per chunk when storing large documents — a 20-chunk document takes ~10–20 s to store. Pick it unless storage throughput or very weak hardware is a concern.

```bash
# Plain docker build
docker build --target full --build-arg EMBED_MODEL=embeddinggemma -t mnemomatic .
```

Or in `docker-compose.yml`:

```yaml
    build:
      context: .
      target: full
      args:
        EMBED_MODEL: embeddinggemma
```

The build bakes the model weights plus a `model_config.json` (dimension, token limit, task prefixes) into the image, and the server reads its defaults from that file — no runtime environment changes are needed when picking a different model. **Changing the model for an existing database requires a reindex** (see [Switching Embedding Models](#switching-embedding-models)) — set `MNEMOMATIC_REINDEX=auto` to have the server handle it on the next start. Until then it refuses to start rather than searching against another model's vectors.

### Lite image with Ollama

Edit `docker-compose.yml` to target the lite build and point at your Ollama instance:

```yaml
services:
  mnemomatic:
    build:
      context: .
      target: lite
    environment:
      - MNEMOMATIC_EMBED_URL=http://host.docker.internal:11434/v1/embeddings
      - MNEMOMATIC_EMBED_MODEL=nomic-embed-text
      - MNEMOMATIC_EMBED_DIM=768
```

Then:

```bash
docker compose up --build
```

### Lite image (FTS-only)

Set `target: lite` and omit `MNEMOMATIC_EMBED_URL`. Fulltext search works normally; semantic and hybrid search return an error indicating no embedder is available.

### Background and stop

```bash
# Run in the background
docker compose up --build -d

# Stop
docker compose down
```

## Configuration

Environment variables (set in `docker-compose.yml` or passed to Docker):

| Variable                    | Default                     | Description                                              |
| --------------------------- | --------------------------- | -------------------------------------------------------- |
| `MNEMOMATIC_DB_PATH`        | `/data/mnemomatic.db`       | Path to the SQLite database file                         |
| `MNEMOMATIC_HOST`           | `0.0.0.0`                   | Server bind address                                      |
| `MNEMOMATIC_PORT`           | `8000`                      | Server port (inside container)                           |
| `MNEMOMATIC_API_KEY`        | *(unset)*                   | API key for Bearer token auth. Auth disabled when unset. |
| `MNEMOMATIC_UI_TOKEN`       | *(unset)*                   | Shared secret for the read-only web viewer at `/ui`. Viewer disabled when unset. |
| `MNEMOMATIC_BACKUP_DIR`     | *(unset)*                   | Directory for scheduled export-zip backups. Backups disabled when unset. |
| `MNEMOMATIC_BACKUP_INTERVAL` | `24`                       | Hours between scheduled backups                          |
| `MNEMOMATIC_BACKUP_KEEP`    | `7`                         | Scheduled backup archives to retain; older ones are pruned |
| `MNEMOMATIC_REVISIONS_KEEP` | `10`                        | Prior versions retained per item (captured on update/delete) for the `restore` tool. `0` disables revision capture. |
| `MNEMOMATIC_SIMILAR_THRESHOLD` | `0.8`                    | Cosine similarity at which stored items count as near-duplicates (`similar` field on store responses, `consolidation_report` clustering). `0` disables the store-time check. |
| `MNEMOMATIC_AUDIT_KEEP_DAYS` | `730`                      | Audit-log retention in days; older events are pruned as new ones are appended. `0` keeps the trail forever. |
| `MNEMOMATIC_EMBED_URL`      | *(unset)*                   | External embedding endpoint (takes priority over the built-in model) |
| `MNEMOMATIC_EMBED_API`      | `openai`                    | Endpoint wire format: `openai` (llama.cpp, vLLM, LM Studio, Ollama `/v1/embeddings`) or `ollama` (native `/api/embeddings`) |
| `MNEMOMATIC_EMBED_MODEL`    | *(empty)*                   | Model name passed to the external embedder               |
| `MNEMOMATIC_EMBED_CONCURRENCY` | `8`                      | Parallel requests to the external embedder when embedding chunked documents |
| `MNEMOMATIC_EMBED_DIM`      | *(bundled model's; else 384)* | Embedding dimension — must match the model's output. Defaults to the bundled model's dimension from `model_config.json`; 384 without a config file. |
| `MNEMOMATIC_EMBED_QUERY_PREFIX` | *(bundled model's; else empty)* | Task prefix prepended to search queries before embedding (asymmetric models). Defaults from `model_config.json`; empty when `MNEMOMATIC_EMBED_URL` is set. |
| `MNEMOMATIC_EMBED_DOC_PREFIX` | *(bundled model's; else empty)* | Task prefix prepended to stored content before embedding (asymmetric models). Defaults from `model_config.json`; empty when `MNEMOMATIC_EMBED_URL` is set. |
| `MNEMOMATIC_REINDEX`        | *(unset)*                   | `auto` re-embeds only when the configured embedder differs from the one that built the index — a no-op otherwise, so it is safe to leave set. `1` rebuilds on every startup; remove it afterwards. Unset refuses to start on a mismatch. |
| `MNEMOMATIC_MODEL_PATH`     | `/app/model/model.onnx`     | Path to the ONNX model file (full image only)            |
| `MNEMOMATIC_TOKENIZER_PATH` | `/app/model/tokenizer.json` | Path to the tokenizer file (full image only)             |
| `MNEMOMATIC_MODEL_CONFIG_PATH` | `/app/model/model_config.json` | Path to the bundled model's metadata file, written by the Docker build |
| `MNEMOMATIC_MODEL_MAX_TOKENS` | *(bundled model's; else 512)* | Token truncation limit for the built-in model (2048 for `embeddinggemma`, 512 for the others) |

> **Changing `MNEMOMATIC_EMBED_DIM`:** the embedding dimension is baked into the database's vector tables at creation. The server records it and refuses to start on a mismatch rather than corrupting the index — unless `MNEMOMATIC_REINDEX` is set to `auto` or `1`, in which case it rebuilds the index at the new dimension (see below).

> **Asymmetric embedding models:** some models are trained with task prefixes that differ between queries and stored content — e.g. EmbeddingGemma expects `task: search result | query: ` on queries and `title: none | text: ` on documents, and multilingual-e5 expects `query: ` / `passage: `. For the built-in model the correct prompts are recorded in `model_config.json` at build time and apply automatically. When `MNEMOMATIC_EMBED_URL` points at an external endpoint, both prefixes default to empty and must be set explicitly for asymmetric models (include the trailing space). Prefixes are applied at embedding time only and never appear in stored content or search snippets. Because the document prefix is baked into stored vectors, changing prefixes — like changing models — requires re-embedding existing content.

## Switching Embedding Models

Changing the embedding model, dimension, or task prefixes invalidates every stored vector — old and new embeddings live in different spaces and must not be compared. The database records which embedder built its index, so the server can tell a deliberate switch from an accidental one.

`MNEMOMATIC_REINDEX` decides what happens when they disagree:

| Value | Behaviour |
| ----- | --------- |
| *(unset — the default)* | Refuse to start, naming what changed. An unintended model or prefix change stops the server instead of quietly rebuilding the index. |
| `auto` | Rebuild and re-embed, but **only** when the embedder actually changed. A no-op on every other start, so it is safe to leave set permanently. |
| `1` | Rebuild and re-embed on **every** start, whether or not anything changed. For forcing a re-embed the identity check would not catch. |

**Switching between built-in models** — set `MNEMOMATIC_REINDEX=auto` once in `docker-compose.yml` and leave it there, then rebuild with a different `EMBED_MODEL`:

```bash
docker compose build --build-arg EMBED_MODEL=embeddinggemma
docker compose up -d
```

The server notices the new model on startup, re-embeds everything, and starts serving. Later restarts on the same model do nothing. The bundled `model_config.json` carries the new dimension and prefixes, so no other settings change.

Without `auto`, the same switch takes three restarts: one that refuses, one with `MNEMOMATIC_REINDEX=1`, and one with the flag removed.

**Switching to an external embedder:**

1. Update the embedder settings — e.g. for EmbeddingGemma served by llama.cpp
   (`llama-server -m embeddinggemma-300M-Q8_0.gguf --embeddings --pooling mean`)
   or Ollama:
   ```yaml
   environment:
     - MNEMOMATIC_EMBED_URL=http://your-embed-host:8181/v1/embeddings
     - MNEMOMATIC_EMBED_MODEL=embeddinggemma
     - MNEMOMATIC_EMBED_DIM=768
     - "MNEMOMATIC_EMBED_QUERY_PREFIX=task: search result | query: "
     - "MNEMOMATIC_EMBED_DOC_PREFIX=title: none | text: "
     - MNEMOMATIC_REINDEX=auto
   ```
2. Restart the server. Startup rebuilds the vector index at the new dimension and re-embeds every document, chunk, knowledge entry, and note with the new model before serving. Content and timestamps are untouched; progress and a final count are logged, and the run is recorded in the audit log.

With `auto` there is no third step — it stays inert until the embedder changes again. (With `MNEMOMATIC_REINDEX=1` you must remove the flag and restart once more, or the re-embed repeats on every boot.)

A reindex needs a working embedder. If one is not available, startup refuses rather than emptying an index it cannot rebuild.

Items whose embedding fails during the run are logged and remain findable via fulltext search; re-run the reindex to retry them. Fulltext search is unaffected throughout.

### Forgetting the reindex

The database records which embedder built its vector index — the model name and both task prefixes, alongside the dimension — and refuses to start when the configured embedder disagrees. The error names each field that changed:

```
Embedding identity mismatch: model was 'embeddinggemma-300m', now 'amaretto-embed-148m'.
The stored vectors were produced with a different embedding configuration, so searching
against them returns wrong results with no error to notice. Restore the previous settings
to keep the existing index, or set MNEMOMATIC_REINDEX=auto to rebuild the index and re-embed
all content on startup.
```

This matters most for swaps that keep the same dimension — most built-in models are 768-dim, so the dimension check alone would let the swap through. The resulting index isn't broken in any visible way: queries embedded by the new model, searched against the old model's vectors, return plausible but degraded results and nothing errors.

Either way out works: restore the previous setting, or reindex.

> **Upgrading an existing database:** the first start after upgrading to a release with this check records whatever embedder is configured then, logs a warning, and starts normally — nothing breaks. It cannot detect a swap that already happened, so if you changed models earlier without reindexing, run `MNEMOMATIC_REINDEX=1` once now (`auto` will not fire — as far as the record is concerned nothing has changed). Running FTS-only (no embedder configured) neither triggers the check nor disturbs a recorded identity.

> **Upgrading from a release that defaulted to `minilm`:** the default model changed, so a database built by an earlier image needs one reindex. Set `MNEMOMATIC_REINDEX=auto` and restart — the recorded embedding identity detects the change and rebuilds the index once. Both models are 384-dim, so the dimension check alone would not catch this; the identity check is what does.

## Schema Migrations

The database schema is versioned (`PRAGMA user_version`). On startup the server migrates older databases forward automatically — for example, databases created before v1 have their vector tables rebuilt with a namespace partition key (embeddings are preserved; no re-embedding needed). Migrations run in a single transaction: if one fails, the database is left untouched.

## Data Portability

The entire database is a single SQLite file. The default setup bind-mounts `./data:/data`, so `mnemomatic.db` lives directly in your project directory where you can back it up, copy it to another machine, or open it with any SQLite tool.

```bash
# Back up the database
cp data/mnemomatic.db ~/backups/mnemomatic-$(date +%Y%m%d).db

# Open with SQLite CLI
sqlite3 data/mnemomatic.db ".tables"
```

## Development

To run locally without Docker:

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the server in development mode with the ONNX embedding stack
pip install -e ".[onnx]"

# Or without the ML stack (FTS-only, or set MNEMOMATIC_EMBED_URL)
pip install -e .

# Install the CLI in development mode (separate package, no deps)
pip install -e ./cli

# Set the database path to a local file
export MNEMOMATIC_DB_PATH=./mnemomatic.db

# Run the server
mnemomatic

# Use the CLI
mnemomatic-cli --help
```

## Tests

### Unit tests

Unit tests cover the database layer directly using an in-memory SQLite database — no Docker required.

```bash
uv run python -m unittest tests/test_db.py -v
```

### Integration tests

Integration tests run against the live MCP server over HTTP. They use Python's built-in `unittest` module — no extra dependencies required.

```bash
# Start the server
docker compose up --build -d

# Run the tests
uv run python -m unittest tests/test_mcp_api.py -v

# Stop when done
docker compose down
```

The integration tests cover storing, reading, upserting, and deleting documents, knowledge entries, and notes over the live MCP API.

## Project Structure

```
mnemomatic/
├── pyproject.toml              # mnemomatic-server package (server deps)
├── Dockerfile                  # Multi-stage build: full (ONNX) and lite (no ML stack)
├── docker-compose.yml          # Container orchestration (Caddy + Mnem-O-matic)
├── Caddyfile                   # Caddy reverse proxy config (TLS termination)
├── LICENSE                     # Apache License 2.0
├── certs/                      # TLS certificates (generated by mkcert, gitignored)
│   ├── cert.pem
│   └── key.pem
├── cli/
│   ├── pyproject.toml          # mnemomatic-cli package (no dependencies)
│   └── src/mnemomatic_cli/
│       ├── cli.py              # CLI entry point and command definitions
│       └── _mcp_client.py      # Minimal MCP HTTP client (stdlib only)
├── src/mnemomatic/
│   ├── server.py               # MCP server — tools and resources
│   ├── db.py                   # SQLite schema, CRUD, and search
│   ├── embeddings.py           # OnnxEmbedder (built-in) and HttpEmbedder (external)
│   ├── auth.py                 # Bearer token authentication middleware
│   └── models.py               # Pydantic data models with input validation
└── tests/
    ├── test_db.py              # Database CRUD and search (in-memory SQLite)
    ├── test_db_corruption.py   # JSON corruption graceful handling
    ├── test_authentication.py  # Auth middleware unit tests
    ├── test_input_validation.py # Pydantic model validation
    ├── test_embedder_errors.py  # Embedder error handling and fallback
    └── test_mcp_api.py         # Integration tests (run against live server)
```
