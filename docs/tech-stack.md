# Tech Stack

## Python + MCP SDK

Python has the most mature official MCP SDK ([modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)). FastMCP provides a decorator-based API for defining tools and resources with minimal boilerplate. Python also has the best ecosystem for ML/embeddings if we need to extend capabilities later.

## SQLite + FTS5 + sqlite-vec

SQLite was chosen for transportability — the entire database is a single file. No server process, no connection strings. Copy the file to back it up, move it to another machine, or sync it between devices.

The schema is versioned via `PRAGMA user_version`: on startup, databases from older versions are migrated forward automatically in a single transaction (see [Schema Migrations](installation.md#schema-migrations)). The database also records the embedding dimension it was created with, and the server fails fast on a mismatch instead of corrupting the index.

FTS5 is SQLite's built-in full-text search engine. It handles keyword and phrase matching with no external dependencies. User queries are escaped defensively: anything that isn't plain words is quoted into a literal phrase, so FTS5 operator syntax (`AND`, `NEAR`, `:` column filters, stray `?`) can never break a search.

[sqlite-vec](https://github.com/asg017/sqlite-vec) adds vector search to SQLite. This enables semantic search — finding results by meaning rather than exact word matches. A search for "authentication" will find entries about "JWT login tokens" even though the words don't overlap. The vec0 tables declare the item's namespace as a **partition key**, so namespace-filtered searches run the nearest-neighbor scan inside that namespace's partition — a small namespace always yields its own best matches instead of being drowned out by larger ones.

## Embeddings — built-in or external

Mnem-O-matic supports two embedding backends:

**Built-in (full image)** — The `full` Docker image bundles [EmbeddingGemma-300m](https://ai.google.dev/gemma/docs/embeddinggemma) as an INT8-quantized ONNX model that runs locally on CPU. No external services required. Inference via `onnxruntime` and tokenization via the Rust-backed `tokenizers` library — no PyTorch or full ML framework needed. The model produces 768-dimensional embeddings, accepts up to 2048 tokens per text, is multilingual (trained on 100+ languages), and is asymmetric: its query/document task prompts are applied automatically (see prefixes below).

The Docker build downloads the [community ONNX export](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) — pinned to an immutable revision and verified against SHA-256 digests, so a moved or tampered upstream file fails the build instead of shipping. The export bakes the full sentence-transformers stack (mean pooling, dense projection layers, normalization) into the graph as a `sentence_embedding` output, which the embedder uses directly — pooling token embeddings by hand would skip the projection layers and produce garbage vectors. The INT8-quantized variant is published pre-made (~310 MB vs ~1.2 GB FP32), so no build-time quantization is needed. EmbeddingGemma weights are distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

At 300 M parameters, CPU inference costs roughly 0.2 s for a short query and up to ~1 s for a full 1000-character chunk — an order of magnitude more than a MiniLM-class model, traded for substantially better retrieval quality and multilingual coverage. Search adds one query embedding per request; document storage embeds each chunk sequentially.

**External (lite image)** — The `lite` image ships without the ML stack and bundled model (~120 MB vs ~650 MB). Point `MNEMOMATIC_EMBED_URL` at an embedding endpoint and the server calls out for embeddings. Two wire formats are supported via `MNEMOMATIC_EMBED_API`: **`openai`** (the default — llama.cpp's `llama-server`, vLLM, LM Studio, Ollama's `/v1/embeddings`, and hosted APIs) and **`ollama`** (the native `/api/embeddings` endpoint). Setting `MNEMOMATIC_EMBED_URL` takes priority over the built-in model, so the `full` image can also delegate to an external embedder. If no URL is configured and no local model is present, the server runs in FTS-only mode — fulltext search works, semantic and hybrid search are unavailable.

The external path is deliberately model-agnostic:

- **Normalization** — every embedding returned by an external endpoint is L2-normalized before storage or search. The score math (L2 distance → cosine similarity) is only correct for unit vectors, and for non-unit vectors even the *ranking* would be wrong; normalizing makes any model safe, including Matryoshka-truncated dimensions.
- **Task prefixes** — asymmetric models (like EmbeddingGemma) are trained with different prompts for queries and stored content. `MNEMOMATIC_EMBED_QUERY_PREFIX` / `MNEMOMATIC_EMBED_DOC_PREFIX` apply these at embedding time only; stored text, snippets, and fulltext search never see them. The built-in model gets its EmbeddingGemma prompts by default; external endpoints default to no prefix since their model is unknown.
- **Concurrent chunk embedding** — a chunked document embeds its chunks with up to `MNEMOMATIC_EMBED_CONCURRENCY` (default 8) requests in flight, rather than one sequential round trip per chunk. (The built-in ONNX model intentionally embeds sequentially: benchmarks showed padded batch inference is neutral-to-slower on CPU, where onnxruntime already parallelizes single runs across cores.)
- **Reindexing** — switching model, dimension, or prefixes invalidates all stored vectors. `MNEMOMATIC_REINDEX=1` rebuilds the vector index and re-embeds every item at startup, making the whole swap a config change plus one flagged restart (see [Switching Embedding Models](installation.md#switching-embedding-models)).

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

When using the `full` image, the ONNX model is also pre-warmed at server startup so the first request doesn't pay the model load cost (a few seconds).

### SQLite tuning

Every connection is configured with three PRAGMAs beyond the defaults:

| PRAGMA        | Value                | Effect                                                                           |
| ------------- | -------------------- | -------------------------------------------------------------------------------- |
| `synchronous` | `NORMAL`             | Safe with WAL mode; skips redundant fsync calls on every write                   |
| `cache_size`  | `-64000` (64 MB)     | Keeps 64 MB of database pages in memory, reducing disk reads on repeated queries |
| `mmap_size`   | `268435456` (256 MB) | Memory-maps up to 256 MB of the database file for faster sequential reads        |

`synchronous=NORMAL` is safe because WAL mode guarantees that a crash cannot corrupt the database — at worst, the last committed transaction is lost, which is acceptable for a personal memory store.
