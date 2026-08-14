# Tech Stack

## Python + MCP SDK

Python has the most mature official MCP SDK ([modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)). FastMCP provides a decorator-based API for defining tools and resources with minimal boilerplate. Python also has the best ecosystem for ML/embeddings if we need to extend capabilities later.

## SQLite + FTS5 + sqlite-vec

SQLite was chosen for transportability — the entire database is a single file. No server process, no connection strings. Copy the file to back it up, move it to another machine, or sync it between devices.

The schema is versioned via `PRAGMA user_version`: on startup, databases from older versions are migrated forward automatically in a single transaction (see [Schema Migrations](installation.md#schema-migrations)). The database also records which embedder built its vector index — the dimension, the model name, and both task prefixes — and the server fails fast on a mismatch instead of searching against vectors from a different embedding space. The dimension alone is not enough: models of equal dimension would otherwise swap silently, leaving an index that returns degraded results with nothing to signal it.

FTS5 is SQLite's built-in full-text search engine. It handles keyword and phrase matching with no external dependencies. User queries are escaped defensively: anything that isn't plain words is quoted into a literal phrase, so FTS5 operator syntax (`AND`, `NEAR`, `:` column filters, stray `?`) can never break a search.

[sqlite-vec](https://github.com/asg017/sqlite-vec) adds vector search to SQLite. This enables semantic search — finding results by meaning rather than exact word matches. A search for "authentication" will find entries about "JWT login tokens" even though the words don't overlap. The vec0 tables declare the item's namespace as a **partition key**, so namespace-filtered searches run the nearest-neighbor scan inside that namespace's partition — a small namespace always yields its own best matches instead of being drowned out by larger ones.

## Embeddings — built-in or external

Mnem-O-matic supports two embedding backends:

**Built-in (full image)** — The `full` Docker image bundles an INT8-quantized ONNX embedding model that runs locally on CPU. No external services required. Inference via `onnxruntime` and tokenization via the Rust-backed `tokenizers` library — no PyTorch or full ML framework needed. The model is chosen at build time with the `EMBED_MODEL` build argument:

- **`minilm` (default)** — `all-MiniLM-L6-v2`: 384 dims, English, ~10–15 ms per embed, ~23 MB on disk / ~240 MB RAM. Downloaded as FP32 and quantized to INT8 at build time — the same pipeline earlier releases used, so existing databases stay compatible. The smallest image and the safe default for English-only content.
- **`amaretto-embed-148m`** — [amaretto-embed-148m](https://huggingface.co/AmarettoLabs/amaretto-embed-148m): 768 dims, 8 Latin-script languages + code, 2048-token context, ~297 MB on disk / ~470 MB RAM. A vocabulary-sliced distillation of EmbeddingGemma (60,497-token vocabulary, 148M encoder parameters) that keeps ~99.5% of the teacher's retrieval nDCG@10 at ~40% of its RAM and ~130 ms per query embed, and takes the same task prefixes. Non-Latin scripts fall back to byte-level tokens and degrade sharply. Shipped as weight-only INT8 (`MatMulNBits`): activations stay FP32, so fidelity holds at long inputs (cosine ≥ 0.999 against the PyTorch reference through 2048 tokens) — the operator is in the `com.microsoft` domain, fine for the bundled `onnxruntime` but not portable to other ONNX runtimes. Weights under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **`gte-multilingual-base`** — 768 dims, ~70 languages, 8192-token context, ~325 MB on disk / ~880 MB RAM, Apache-2.0. Queries embed in ~12 ms — near-MiniLM speed, since most of its parameters are vocabulary embeddings that cost little for short inputs — while longer chunks take ~150 ms. Strong cross-lingual retrieval (a query in one language finds content stored in another) with no task prefixes.
- **`embeddinggemma`** — [EmbeddingGemma-300m](https://ai.google.dev/gemma/docs/embeddinggemma): 768 dims, 100+ languages, 2048-token context, ~330 MB on disk / ~1.2 GB RAM, and decisively the best retrieval quality — an order of magnitude larger relevant-vs-irrelevant score margins, and it resolves zero-word-overlap paraphrase queries the smaller models miss. The cost: ~160–225 ms for a short query and up to ~1 s per 1000-character chunk. Weights under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

(Latency figures measured end to end on a modest x86 server; see [Choosing the built-in embedding model](installation.md#choosing-the-built-in-embedding-model) for per-model guidance.)

Alongside the weights, the build writes a `model_config.json` recording the model's dimension, token limit, and task prefixes; the server reads its defaults from it, so the build argument is the only knob. All downloads are pinned to immutable revisions and verified against SHA-256 digests — a moved or tampered upstream file fails the build instead of shipping.

The embedder handles two ONNX graph shapes. Sentence-transformers exports like EmbeddingGemma's bake the full pooling stack (mean pooling, dense projection layers, normalization) into the graph as a `sentence_embedding` output, which is used directly — pooling token embeddings by hand would skip the projection layers and produce garbage vectors. Plain transformer exports (MiniLM, multilingual-e5) only produce token embeddings, which are mean-pooled and L2-normalized in Python.

**External (lite image)** — The `lite` image ships without the ML stack and bundled model (~120 MB vs ~320–650 MB depending on `EMBED_MODEL`). Point `MNEMOMATIC_EMBED_URL` at an embedding endpoint and the server calls out for embeddings. Two wire formats are supported via `MNEMOMATIC_EMBED_API`: **`openai`** (the default — llama.cpp's `llama-server`, vLLM, LM Studio, Ollama's `/v1/embeddings`, and hosted APIs) and **`ollama`** (the native `/api/embeddings` endpoint). Setting `MNEMOMATIC_EMBED_URL` takes priority over the built-in model, so the `full` image can also delegate to an external embedder. If no URL is configured and no local model is present, the server runs in FTS-only mode — fulltext search works, semantic and hybrid search are unavailable.

The external path is deliberately model-agnostic:

- **Normalization** — every embedding returned by an external endpoint is L2-normalized before storage or search. The score math (L2 distance → cosine similarity) is only correct for unit vectors, and for non-unit vectors even the *ranking* would be wrong; normalizing makes any model safe, including Matryoshka-truncated dimensions.
- **Task prefixes** — asymmetric models (like EmbeddingGemma and multilingual-e5) are trained with different prompts for queries and stored content. `MNEMOMATIC_EMBED_QUERY_PREFIX` / `MNEMOMATIC_EMBED_DOC_PREFIX` apply these at embedding time only; stored text, snippets, and fulltext search never see them. The built-in model's prompts come from `model_config.json` automatically; external endpoints default to no prefix since their model is unknown.
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
