# Installation

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [mkcert](https://github.com/FiloSottile/mkcert) — for generating locally-trusted TLS certificates (LAN deployments)

## Deployment Profiles

| Profile           | Image size | Embeddings                          | Semantic search |
| ----------------- | ---------- | ----------------------------------- | --------------- |
| `full` (default)  | ~316 MB    | Built-in ONNX model (CPU)           | Yes             |
| `lite` + Ollama   | ~120 MB    | External via `MNEMOMATIC_EMBED_URL` | Yes             |
| `lite` (FTS-only) | ~120 MB    | None                                | No              |

Choose the profile that fits your setup. The `full` image is self-contained and works out of the box. The `lite` image is significantly smaller and delegates embedding to an Ollama instance (or any compatible API), or runs keyword-only search if no embedder is configured.

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
      - MNEMOMATIC_EMBED_URL=http://host.docker.internal:11434/api/embeddings
      - MNEMOMATIC_EMBED_MODEL=nomic-embed-text
      - MNEMOMATIC_EMBED_DIM=768
```

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

The first build takes a few minutes — it downloads and quantizes the embedding model. Subsequent builds use the cached layer.

### Lite image with Ollama

Edit `docker-compose.yml` to target the lite build and point at your Ollama instance:

```yaml
services:
  mnemomatic:
    build:
      context: .
      target: lite
    environment:
      - MNEMOMATIC_EMBED_URL=http://host.docker.internal:11434/api/embeddings
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
| `MNEMOMATIC_EMBED_URL`      | *(unset)*                   | Ollama-compatible embedding endpoint (lite image)        |
| `MNEMOMATIC_EMBED_MODEL`    | *(empty)*                   | Model name passed to the external embedder               |
| `MNEMOMATIC_EMBED_DIM`      | `384`                       | Embedding dimension — must match the model's output      |
| `MNEMOMATIC_MODEL_PATH`     | `/app/model/model.onnx`     | Path to the ONNX model file (full image only)            |
| `MNEMOMATIC_TOKENIZER_PATH` | `/app/model/tokenizer.json` | Path to the tokenizer file (full image only)             |

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

# Install in development mode with the ONNX embedding stack
pip install -e ".[onnx]"

# Or without the ML stack (FTS-only, or set MNEMOMATIC_EMBED_URL)
pip install -e .

# Set the database path to a local file
export MNEMOMATIC_DB_PATH=./mnemomatic.db

# Run the server
mnemomatic
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
├── pyproject.toml              # Project metadata and dependencies
├── Dockerfile                  # Multi-stage build: full (ONNX) and lite (no ML stack)
├── docker-compose.yml          # Container orchestration (Caddy + Mnem-O-matic)
├── Caddyfile                   # Caddy reverse proxy config (TLS termination)
├── LICENSE                     # Apache License 2.0
├── certs/                      # TLS certificates (generated by mkcert, gitignored)
│   ├── cert.pem
│   └── key.pem
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
