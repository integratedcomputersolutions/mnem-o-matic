# Tech Stack

## Python + MCP SDK

Python has the most mature official MCP SDK ([modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)). FastMCP provides a decorator-based API for defining tools and resources with minimal boilerplate. Python also has the best ecosystem for ML/embeddings if we need to extend capabilities later.

## SQLite + FTS5 + sqlite-vec

SQLite was chosen for transportability — the entire database is a single file. No server process, no connection strings, no migrations infrastructure. Copy the file to back it up, move it to another machine, or sync it between devices.

FTS5 is SQLite's built-in full-text search engine. It handles keyword and phrase matching with no external dependencies.

[sqlite-vec](https://github.com/asg017/sqlite-vec) adds vector search to SQLite. This enables semantic search — finding results by meaning rather than exact word matches. A search for "authentication" will find entries about "JWT login tokens" even though the words don't overlap.

## Embeddings — built-in or external

Mnem-O-matic supports two embedding backends:

**Built-in (full image)** — The `full` Docker image bundles `all-MiniLM-L6-v2` as an INT8-quantized ONNX model that runs locally on CPU. No external services required. Inference via `onnxruntime` and tokenization via the Rust-backed `tokenizers` library — no PyTorch or full ML framework needed.

The Docker build downloads Qdrant's ONNX export of `all-MiniLM-L6-v2`, then **quantizes it from FP32 to INT8** before copying it into the runtime image (~80 MB → ~20 MB, 2–3× faster inference). The `onnx` package used for quantization is installed only in the build stage and never copied to the runtime image.

**External (lite image)** — The `lite` image ships without the ML stack (~120 MB vs ~316 MB). Point it at any Ollama-compatible embedding endpoint via `MNEMOMATIC_EMBED_URL` and it will call out for embeddings. If no URL is configured, the server runs in FTS-only mode — fulltext search works, semantic and hybrid search are unavailable.

## Streamable HTTP Transport

The MCP server runs as an HTTP service, which means multiple LLM clients can connect simultaneously. This is what makes it a _shared_ memory — Claude Code and Copilot can both be connected at the same time, reading and writing to the same knowledge base.

## Concurrency

Mnem-O-matic is designed to handle up to 10 simultaneous LLM clients safely.

### How it works

Each request handler thread gets its own SQLite connection via `threading.local()`. This avoids thread-safety issues entirely — Python's `sqlite3` connections are not safe to share across threads, so each thread operates on an independent connection.

All connections run in WAL (Write-Ahead Logging) mode, which allows unlimited concurrent readers alongside a single writer. Individual write operations (store, update, delete) are sub-millisecond, so write serialization is a non-issue at this scale.

A 5-second `busy_timeout` is configured on every connection. If a write is attempted while another write is in progress, SQLite retries automatically for up to 5 seconds instead of immediately failing with a "database is locked" error.

### Why not Postgres?

SQLite in WAL mode comfortably handles the concurrency level of a personal Mnem-O-matic instance (5-10 LLM clients). Switching to a client-server database would add deployment complexity and eliminate SQLite's main advantage: the entire database is a single portable file you can copy, back up, or move between machines.

## Performance

### Embedding cache

For both the built-in ONNX model and the external HTTP embedder, identical text inputs are cached in memory (LRU, up to 256 entries) — re-storing the same content via upsert skips recomputation or a network round-trip entirely.

When using the `full` image, the ONNX model is also pre-warmed at server startup so the first request doesn't pay the model load cost (~1–2s).

### SQLite tuning

Every connection is configured with three PRAGMAs beyond the defaults:

| PRAGMA        | Value                | Effect                                                                           |
| ------------- | -------------------- | -------------------------------------------------------------------------------- |
| `synchronous` | `NORMAL`             | Safe with WAL mode; skips redundant fsync calls on every write                   |
| `cache_size`  | `-64000` (64 MB)     | Keeps 64 MB of database pages in memory, reducing disk reads on repeated queries |
| `mmap_size`   | `268435456` (256 MB) | Memory-maps up to 256 MB of the database file for faster sequential reads        |

`synchronous=NORMAL` is safe because WAL mode guarantees that a crash cannot corrupt the database — at worst, the last committed transaction is lost, which is acceptable for a personal memory store.
