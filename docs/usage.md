# Usage

## Connecting LLM Clients

### Claude Code

```bash
claude mcp add --transport http mnemomatic https://your-server-hostname/mcp \
  -H "Authorization: Bearer your-secret-key-here"
```

Replace `your-server-hostname` with the hostname or IP you used when generating the TLS certificate. The client device must have the mkcert CA trusted (see [TLS Setup](installation.md#tls-setup-lan-deployments)).

### Other MCP Clients

Point any MCP-compatible client to `https://your-server-hostname/mcp` using the Streamable HTTP transport. Include the `Authorization: Bearer <key>` header with every request.

### Small-Context Models (SLMs)

Verbose tool descriptions can consume a significant portion of a small model's context window. Appending `?compact=true` to the endpoint URL switches `tools/list` responses to concise one-line descriptions and strips verbose parameter descriptions, keeping only short hints for parameters with constrained valid values (`mode`, `content_type`, `item_type`, `confidence`).

| Client | URL |
|--------|-----|
| Full-context (Claude, GPT-4, etc.) | `https://your-server-hostname/mcp` |
| Small-context (7B–13B local models) | `https://your-server-hostname/mcp?compact=true` |

Both endpoints share the same server instance, database, and authentication. The compact descriptions are tuned in `src/mnemomatic/compact.py` (`_COMPACT_DESCRIPTIONS` and `_COMPACT_PARAMS`).

## Authentication

Authentication is **optional** and uses the Bearer token scheme. Requests that fail authentication are rejected before any MCP processing.

### Enabling Authentication

Set `MNEMOMATIC_API_KEY` to enable token validation:

```yaml
# docker-compose.yml
services:
  mnemomatic:
    environment:
      - MNEMOMATIC_API_KEY=your-secret-key-here
```

When authentication is enabled, all requests must include the `Authorization` header:

```
Authorization: Bearer <your-secret-key-here>
```

### Without Authentication

If `MNEMOMATIC_API_KEY` is not set or is empty, the server runs without authentication. This is suitable for local development and trusted networks. The server logs a warning at startup:

```
WARNING  mnemomatic: Authentication disabled — server is running without API key validation
```

For LAN deployments with TLS, the API key is **required** — it is the only per-request credential. On TLS alone, any device on your network that trusts the CA could connect without it.

### Best Practices

**For production deployments:**

1. **Use a strong, random key** — At least 32 characters. Example:
   ```bash
   openssl rand -base64 32
   # Zn8p7xQvJ9kL2mN3bC4dE5fG6hI7jK8lMnOpQrStUvW=
   ```

2. **Never commit keys to version control** — Use environment variables, secrets managers (e.g., Docker Secrets, Kubernetes Secrets), or `.env` files (excluded from git).

3. **Use HTTPS in production** — Deploy behind a reverse proxy (nginx, Caddy, or similar) with TLS encryption. Authentication headers are transmitted in the `Authorization` header, which should be encrypted in transit.

4. **Rotate keys periodically** — If a key is compromised or exposed:
   - Update `MNEMOMATIC_API_KEY`
   - Restart the server: `docker compose down && docker compose up -d`
   - Update all clients with the new key

5. **Log authentication events** — Mnem-O-matic logs all authentication attempts (both successful and failed) at WARNING and DEBUG levels. Monitor these logs for suspicious activity.

### Error Responses

| Status | Error | Reason |
|--------|-------|--------|
| 401 | Missing Authorization header | No `Authorization` header sent with request |
| 401 | Invalid Authorization header format | Header format is not `Bearer <token>` |
| 401 | Malformed Authorization header | Token is missing or header is incomplete |
| 401 | Invalid Authorization header (empty token) | Token is present but empty |
| 403 | Invalid API key | Token was sent but does not match `MNEMOMATIC_API_KEY` |
| 429 | Too many failed authentication attempts | Repeated invalid keys from the same client triggered a temporary lockout; retry after the `Retry-After` header. Missing/malformed headers don't count toward the lockout. |

All error responses include a `details` field explaining the exact issue.

### Troubleshooting

**"Missing Authorization header"**
- Ensure you're sending the `Authorization` header with every request
- Verify the format: `Bearer <key>` (note the space after `Bearer`)

**"Invalid API key"**
- Check that the key in your request matches `MNEMOMATIC_API_KEY` exactly
- Keys are case-sensitive
- Verify there's no leading/trailing whitespace

**"Invalid Authorization header format"**
- Ensure the header starts with `Bearer ` (case-insensitive)
- The format must be: `Authorization: Bearer <token>`
- Common mistake: using `Token` or `Basic` instead of `Bearer`

**Server starts with "Authentication disabled"**
- `MNEMOMATIC_API_KEY` is not set or is empty
- Set it in `docker-compose.yml` or pass it via `-e` flag:
  ```bash
  docker compose up -e MNEMOMATIC_API_KEY=your-key
  ```

## Web Viewer

A minimal, **read-only** web viewer is available at `/ui` for browsing stored documents, knowledge, and notes. It has no create, edit, or delete functionality.

The viewer is **disabled by default**. Enable it by setting a shared secret:

```yaml
services:
  mnemomatic:
    environment:
      - MNEMOMATIC_UI_TOKEN=your-viewer-secret
```

Then open `https://your-host/ui` (or `http://your-host:8000/ui` for direct access), enter the token once, and browse by namespace.

A **Settings** page (`/ui/settings`, linked from the navbar) shows the configuration the server is running with — first section covers the embedding model: mode (built-in / external endpoint / FTS-only), model name (linked to its Hugging Face card for the built-in models), embedding dimension, and the model and dimension the vector index was actually built with — with a warning when either disagrees with the running server, and an explicit "not recorded" where the index predates identity tracking — plus token truncation limit or endpoint URL, task prefixes, and the document chunking settings. The `embedding_info` tool reports the same state to an agent.

Notes:
- There are **no user accounts** — access is a single shared secret, separate from `MNEMOMATIC_API_KEY` (the viewer is exempt from MCP Bearer auth and uses its own gate; the exemption only exists while the viewer is enabled).
- The session cookie is HttpOnly and stores a value **derived** from the token with a per-process key — never the token itself. It is marked `Secure` when the connection is HTTPS (directly or via a proxy that sets `X-Forwarded-Proto`). Restarting the server invalidates existing sessions, so viewers re-enter the token.
- Repeated wrong tokens from the same client trigger a temporary lockout (HTTP 429). The same applies to repeated invalid MCP API keys.
- The viewer is served on the same host/port as the MCP endpoint. Because that port is typically bound to `0.0.0.0`, the shared secret is what keeps it private — choose a strong token, or additionally restrict the port at the network level (VPN, reverse proxy, firewall).
- When `MNEMOMATIC_UI_TOKEN` is unset, `/ui` is not registered at all.

## Export

`GET /export` downloads the entire store (or one namespace with `?namespace=...`) as a **human-readable zip archive** — for backups, or for porting content into another system:

```
mnemomatic-export-2026-08-02.zip
├── export-info.json          # manifest: format version, date, counts, namespace map
└── <namespace>/
    ├── documents/
    │   ├── <title>.md        # the document content, byte-faithful — nothing injected
    │   └── metadata.json     # filename → exact title, id, tags, timestamps, metadata
    ├── knowledge/            # one .md per entry containing the fact
    └── notes/
```

File names are sanitized titles (collisions get an id suffix); the exact originals are always in the `metadata.json` sidecars, and the manifest maps folder names back to exact namespace names. Document extensions follow the mime type (`.md`, `.txt`, `.json`). Embeddings, chunks, and full-text indexes are **not** exported — they are derived data, and excluding them keeps the archive independent of the embedding model. Superseded knowledge entries and item revisions are not exported either: the archive carries the store's current state.

Three ways to trigger it:

```bash
# curl (the endpoint honors the same Bearer auth as MCP)
curl -H "Authorization: Bearer $KEY" -OJ https://your-host/export

# CLI — writes atomically (never leaves a truncated zip over a previous backup)
mnemomatic-cli export -o /backups/          # directory: server-suggested, date-based name
mnemomatic-cli export -o memory.zip         # exact file path
mnemomatic-cli export -o -                  # raw zip to stdout, for piping
mnemomatic-cli export -n myproject          # single namespace

# Web viewer: Settings → Export → "Download export"
```

The date-based default filename means a daily cron job gets one file per day, and re-running the same day safely replaces that day's file (the CLI downloads to `<name>.part` and renames only on success).

### Scheduled backups

The server can also write the export archive itself, on a schedule — no cron or CLI on the host required. Point `MNEMOMATIC_BACKUP_DIR` at a directory (in Docker, somewhere under the mounted data volume):

```yaml
    environment:
      - MNEMOMATIC_BACKUP_DIR=/data/backups
      - MNEMOMATIC_BACKUP_INTERVAL=24   # hours between backups (default 24)
      - MNEMOMATIC_BACKUP_KEEP=7        # archives to retain (default 7)
```

Backups are full exports (all namespaces) named `mnemomatic-backup-YYYYMMDD-HHMMSS.zip` (UTC), written atomically. Once more than `MNEMOMATIC_BACKUP_KEEP` exist, the oldest are deleted — pruning only ever touches that filename pattern, so manual exports stored in the same directory are never removed. The schedule survives restarts: the next backup is due one interval after the newest existing archive, not after boot, so restarting the server neither skips a backup nor churns the retention window. When `MNEMOMATIC_BACKUP_DIR` is unset, nothing runs.

The CLI + cron path above remains the right choice when the backup needs to leave the machine or be encrypted (e.g. piping `export -o -` through `gpg`).

## Usage Tracking & Revisions

Two always-on recording mechanisms make the store safer to mutate and lay the groundwork for memory-review workflows:

**Usage tracking** — every item carries a `retrieval_count` and `last_accessed`, bumped when the item is fetched with the `read` tool (or an MCP resource) and when a search surfaces it in results. Browsing does **not** count: `list_items`, the web viewer, exports, and backups never touch the counters, so they measure genuine retrieval, not housekeeping. The counters appear in `read` output and `list_items` summaries; `updated_at` is never affected. There is no ranking impact yet — the data accumulates first, so any future ranking blend can be tuned against real numbers.

**Revisions** — every update and delete first saves the item's prior state, including upsert overwrites (`store_*` on an existing title/subject), tag edits, `delete_namespace`, and items replaced by a `rename_namespace` merge. The server keeps the newest `MNEMOMATIC_REVISIONS_KEEP` revisions per item (default 10; `0` disables capture). Two tools work with them:

```
list_revisions [item_type] [item_id] [namespace] [limit]   # newest first; op is "update" or "delete"
restore <revision_id>                                       # roll back / undelete
```

`restore` semantics:
- If the item still exists, its content rolls back to the revision's state through the normal update path — the pre-restore state is captured as a new revision first, so **a restore can itself be undone**.
- If the item was deleted, it is recreated with its original id and `created_at`. When another item has since taken the same namespace + title/subject, the restore refuses (naming the occupant) instead of overwriting it.
- Restored content is re-embedded immediately, so search reflects it right away.

Revisions store content and metadata, not embeddings — like the export archive, they stay independent of the embedding model. Note that deleting an item does **not** purge its revisions: recovering exactly that data is what they are for. Set `MNEMOMATIC_REVISIONS_KEEP=0` if items must be gone the moment they are deleted.

## Audit Log

Every successful write operation is recorded in an **append-only audit log** — the event trail that complements revisions: revisions hold what an item *was* (for restore, pruned per item), the audit log holds what *happened* (for accountability, never pruned).

Each event carries the timestamp, operation (`store`, `update`, `supersede`, `delete`, `tag`, `restore`, `rename_namespace`, `delete_namespace`), the item's type/id/namespace/title, op-specific detail (e.g. which fields an update touched, which entry a supersession closed), and three request-identity fields:

| Field | Source | Trust |
|-------|--------|-------|
| `actor` | The client's `X-Mnemomatic-Actor` request header, if it sends one | Self-declared — fine among cooperating clients, not authenticated |
| `client` | The `User-Agent` header | What the connecting software reports |
| `ip` | The connection's peer address | Behind a reverse proxy this is the proxy's address |

To label a client, add the header to its MCP configuration:

```bash
claude mcp add --transport http mnemomatic https://your-host/mcp \
  -H "Authorization: Bearer your-key" \
  -H "X-Mnemomatic-Actor: matt-laptop"
```

Query the trail with the `list_audit` tool — filter by item, namespace, or operation:

```
list_audit(namespace="myproject")                  # recent activity in a project
list_audit(item_id="abc-123")                      # everything that happened to one item
list_audit(op="delete")                            # all deletions, store-wide
```

Reads are deliberately not audited (usage tracking covers retrieval); failed operations are not recorded; and a failing audit write never breaks the operation it describes. With a single shared API key the `actor` is self-reported — per-key authenticated attribution would come with scoped API keys, which the schema already accommodates.

Retention is time-based: events older than `MNEMOMATIC_AUDIT_KEEP_DAYS` (default 730 — two years) are pruned as new ones are appended; set `0` to keep the trail forever. Events are a couple of hundred bytes each (titles and ids, never content), so even the default retention stays in the low tens of MB on a busy store.

## Temporal Facts

Knowledge entries answer questions like "what is our auth method?" — and the answer changes over time. So knowledge is **temporal**: when a fact changes, the old entry is *superseded* rather than overwritten. It stays in the store with `valid_until` (when it stopped being the current answer) and `superseded_by` (the id of its replacement), answering "what did we believe before, and until when?"

How a fact changes:
- `store_knowledge` with an existing subject and a **different** fact → the current entry is closed, the new fact becomes a new entry, and the response carries `"superseded": "<old-id>"`. Re-storing the **same** fact just refreshes the entry in place (no history spam from agents re-storing what they know).
- `update_knowledge` changing `fact` → same supersession; changing only `confidence`/`source`/`tags`/`metadata` edits the current entry in place (captured as a revision, like documents and notes).

Superseded entries are **excluded from search, listings, counts, and exports** — only the current answer surfaces. They remain readable by id and through the dedicated tool:

```
fact_history(namespace="webapp", subject="auth method")
→ {"count": 3, "history": [ current entry, then superseded versions newest first ]}
```

History is immutable: updating a superseded entry returns an error (correct the current fact instead). Deleting one is allowed (pruning history). Deleting the *current* entry ends the chain — the next `store_knowledge` for that subject starts a fresh one, and `fact_history` still shows everything ever held for the subject.

The division of labor with [revisions](#usage-tracking--revisions): fact changes are *history* (first-class, queryable, permanent); everything else — in-place edits, deletes, document/note changes — is *undo* (revisions, capped per item).

## Memory Hygiene: Duplicates, Consolidation, Prompts

Mnem-O-matic never needs its own LLM for memory upkeep — every MCP client already is one. The server does the mechanical part (vector math, usage statistics) and hands the judgment to the connected agent:

**`similar` on store responses** — when newly stored content is nearly identical (cosine ≥ `MNEMOMATIC_SIMILAR_THRESHOLD`, default 0.8) to items already in the namespace, the store response includes a `similar` list (id, title, score). The agent that is mid-write is the best judge: merge, supersede, or ignore. Requires an embedder; chunked documents (no whole-document vector) are skipped; `0` disables the check.

**`consolidation_report` tool** — mechanical consolidation candidates for a namespace: same-type near-duplicate clusters computed from the stored vectors, plus stale items (never retrieved since usage tracking began and not updated in `stale_days` days, default 90). Pure vector math and SQL — the report only *flags*.

**Prompts** — two MCP prompts turn the report into workflows (in Claude Code they appear as slash commands):

- `consolidate(namespace)` — walks the agent through the report: read every cluster member, merge duplicates (fold unique details in, delete the copy — recoverable via revisions), let conflicting facts supersede through `update_knowledge`, review stale items (keep / tag `deprecated` / delete), and report actions taken. Conservative by instruction: nothing is deleted unread.
- `briefing(task, [namespace])` — memory that shows up prepared: the agent derives several search queries from a task description, reads what's relevant, checks `fact_history` where an answer may have changed, and answers with a briefing (constraints, references, gaps) instead of a search log.

For scheduled upkeep, run the consolidation from cron via a headless agent — it uses your existing subscription, no API keys:

```
claude -p "Use the mnemomatic consolidate prompt on namespace 'myproject' and apply its workflow."
```

A note on early reports: usage counters only accumulate from the moment this feature is deployed, so "never retrieved" on a fresh upgrade means "not retrieved *yet*" — give the data a few weeks before trusting the stale list.

## CLI Interface

`mnemomatic-cli` provides shell access to a running Mnem-O-matic server for agents and users without MCP support.

### Installation

```bash
git clone https://github.com/integratedcomputersolutions/mnem-o-matic.git
cd mnem-o-matic
uv tool install ./cli
```

This installs `mnemomatic-cli` into an isolated environment with no extra dependencies. Verify with:

```bash
mnemomatic-cli --help
```

To uninstall: `uv tool uninstall mnemomatic-cli`

For development (runs from source without installing):

```bash
uv run --project cli mnemomatic-cli --help
```

### Configuration

Settings resolve with this priority: **CLI flags > environment variables > config file > defaults**.

| Setting | CLI flag | Environment variable | Config key | Default |
|---------|----------|---------------------|------------|---------|
| Server URL | `--server-url` | `MNEMOMATIC_SERVER_URL` | `server.url` | `http://localhost:8000` |
| API key | `--api-key` | `MNEMOMATIC_API_KEY` | `server.api_key` | *(none)* |
| Search mode | `-m` / `--mode` | `MNEMOMATIC_SEARCH_MODE` | `search.mode` | `hybrid` |

The config file lives at `~/.config/mnemomatic/config.toml`:

```toml
[server]
url = "https://your-server-hostname"
api_key = "your-secret-key-here"

[search]
mode = "fulltext"
```

> **Security:** Prefer the environment variable or config file for the API key — CLI flags are visible in the process list. The CLI warns if the config file is readable by other users.

### Commands

```bash
# Search
mnemomatic-cli search "authentication"
mnemomatic-cli search "JWT tokens" -n webapp -m semantic -l 5
mnemomatic-cli search "deploy" --tag runbook --updated-after 2026-08-01

# Store
mnemomatic-cli store document myproject "API spec" "Full API specification text"
mnemomatic-cli store knowledge myproject "auth method" "Uses JWT with RS256"
mnemomatic-cli store note myproject "Quick thought" "Consider adding rate limiting"

# Read from stdin (use '-' as content)
cat spec.md | mnemomatic-cli store document myproject "API spec" -

# Update
mnemomatic-cli update document <id> --content "Updated content"
mnemomatic-cli update knowledge <id> --fact "Migrated to session cookies"

# Delete individual items
mnemomatic-cli delete document <id>
mnemomatic-cli delete knowledge <id>
mnemomatic-cli delete note <id>

# Read full content by ID (after a search)
mnemomatic-cli read document <id>
mnemomatic-cli read knowledge <id>
mnemomatic-cli read note <id>

# Get full content by ID (via resource URI)
mnemomatic-cli get document <id>

# Tags
mnemomatic-cli tag <id> document --add prod --add critical --remove draft

# Browse content in a namespace
mnemomatic-cli list documents myproject
mnemomatic-cli list knowledge myproject
mnemomatic-cli list notes myproject

# Paginated listing for large namespaces (uses the list_items tool)
mnemomatic-cli list documents myproject --limit 20
mnemomatic-cli list documents myproject --limit 20 --offset 20

# Namespace management
mnemomatic-cli namespace list
mnemomatic-cli namespace rename old-project new-project
mnemomatic-cli namespace delete old-project           # prompts for confirmation
mnemomatic-cli namespace delete old-project --yes     # skip prompt (scripts/agents)

# Export (see the Export section)
mnemomatic-cli export -o /backups/                    # all namespaces, into a directory
mnemomatic-cli export -n myproject -o project.zip     # one namespace, exact filename
mnemomatic-cli export -o - | gpg -e -r me@example.com > backup.zip.gpg   # stream to stdout
```

All output is JSON. Use `--pretty` for indented output:

```bash
mnemomatic-cli --pretty search "auth"
```

## Available Tools

Once connected, your LLM has access to these tools:

| Tool                 | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `store_document`     | Save a document (code, spec, config)                 |
| `store_knowledge`    | Save a fact, decision, or observation                |
| `store_note`         | Save a quick thought, idea, or transcript            |
| `update_document`    | Modify an existing document                          |
| `update_knowledge`   | Modify an existing knowledge entry                   |
| `update_note`        | Modify an existing note                              |
| `delete_document`    | Remove a document                                    |
| `delete_knowledge`   | Remove a knowledge entry                             |
| `delete_note`        | Remove a note                                        |
| `tag`                | Add or remove tags on any entry                      |
| `search`             | Search across all stored data; optional `tags` / `updated_after` filters (see Search Filters) |
| `related`            | Items most similar to an existing item — "more like this" (see Related Items) |
| `read`               | Fetch full content of an item by ID                  |
| `list_items`         | List item summaries in a namespace, newest first, paginated with `limit`/`offset` (response includes `total`) |
| `rename_namespace`   | Rename a namespace atomically across all item types. Merges into an existing target: on title/subject collisions the moved item replaces the target's (upsert semantics); the response reports `replaced` counts. |
| `delete_namespace`   | Permanently delete all items in a namespace          |
| `list_revisions`     | List saved prior versions of items (captured on every update and delete), newest first — filter by type, item, or namespace |
| `restore`            | Restore an item to a revision: roll back an update or recreate a deleted item |
| `fact_history`       | The timeline of a knowledge fact: the current entry, then every superseded version (see Temporal Facts) |
| `consolidation_report` | Consolidation candidates for a namespace: near-duplicate clusters and stale never-retrieved items (see Memory Hygiene) |
| `list_audit`         | The append-only audit trail of write operations, newest first — filter by item, namespace, or operation (see Audit Log) |
| `embedding_info`     | Which embedding model is in use, whether it matches the one that built the index, and whether semantic search is available (see Embedding Info) |

### Input Validation & Limits

Mnem-O-matic validates all inputs to prevent silent failures:

| Constraint | Limit | Impact |
|-----------|-------|--------|
| **Namespace length** | ≤ 100 chars | Used for grouping related entries |
| **Content length** | ≤ 100,000 chars | Documents, notes, facts |
| **Title length** | ≤ 500 chars | Document/note titles, knowledge subjects |
| **Search query** | Non-empty, ≤ 10,000 chars | Empty queries rejected; very long queries capped |
| **Search results** | ≤ 100 results | Limited to prevent memory exhaustion; use smaller limits for faster results |
| **Tags per entry** | ≤ 100 tags | Too many tags degrade performance |
| **Tag length** | ≤ 50 chars each | Keep tags short and descriptive |
| **Metadata keys** | ≤ 50 keys | Avoid excessive metadata |
| **Metadata value** | ≤ 10,000 chars | Keep values reasonably sized |
| **Confidence (knowledge)** | 0.0 to 1.0 | Must be a valid probability |
| **Embedding dimension** | Must match embedder | Mismatch causes search errors; server warns at startup |

If validation fails, tools return an error with details — fix the input and retry.

### Deduplication

Store tools use upsert semantics — if an entry with the same namespace and title (for documents) or namespace and subject (for knowledge) already exists, it is updated rather than creating a duplicate.

This matters because LLMs don't track what's already stored. Without deduplication, restarting a session and re-storing the same facts would create duplicate rows. Documents and notes update in place (`"created": false`). Knowledge is temporal (see [Temporal Facts](#temporal-facts)): re-storing the *same* fact refreshes the entry in place, while storing a *different* fact for an existing subject supersedes it — the old entry is kept as queryable history:

```
# First call — creates a new entry
store_knowledge(namespace="webapp", subject="auth method", fact="Uses JWT with RS256")
→ {"id": "abc-123", "created": true}

# Same fact again — refreshes in place, no history entry
store_knowledge(namespace="webapp", subject="auth method", fact="Uses JWT with RS256")
→ {"id": "abc-123", "created": false}

# The fact changed — the old entry is closed and kept as history
store_knowledge(namespace="webapp", subject="auth method", fact="Migrated to session cookies")
→ {"id": "def-456", "created": true, "superseded": "abc-123"}
```

### Chunked Retrieval for Large Documents

Documents longer than `MNEMOMATIC_CHUNK_THRESHOLD` (default: 2000 chars) are automatically split into overlapping chunks at store time. Each chunk gets its own vector embedding, so semantic search returns the most relevant passage rather than a whole-document match.

When a search result comes from a chunk, the response includes `"partial": true`. This signals that only part of the document was returned — call `read` with the same `id` to retrieve the full content.

```
# Search returns a relevant passage from a large document
search("authentication flow")
→ {"id": "abc-123", "title": "API spec", "snippet": "...JWT tokens are validated by...", "partial": true}

# Fetch the full document when needed
read(item_type="document", id="abc-123")
→ {"content": "...full document..."}
```

Chunking is transparent: documents are split and indexed automatically, and search results use the same `id` as the parent document. Small documents, knowledge entries, and notes are unaffected.

Existing documents stored before upgrading continue to work via their whole-document embeddings. They transition to chunk-based retrieval automatically the next time they are stored or updated.

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMOMATIC_CHUNK_THRESHOLD` | `2000` | Document length in chars above which chunking is applied |
| `MNEMOMATIC_CHUNK_SIZE` | `1000` | Target chunk size in chars |
| `MNEMOMATIC_CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks in chars |

### Search Modes

The `search` tool supports three modes:

- **fulltext** — keyword and phrase matching via SQLite FTS5
- **semantic** — meaning-based search via vector embeddings
- **hybrid** (default) — combines both, ranked by a blended score

### Search Filters

Two optional filters narrow any mode, and compose with `namespace` and `content_type`:

- **`tags`** — only items carrying **all** the listed tags (exact matches, not prefixes)
- **`updated_after`** — only items updated at or after an ISO date or datetime (`"2026-08-01"`, `"2026-08-01T12:00:00"`)

```
search("deployment", tags=["runbook"])                     # tagged runbooks only
search("auth", updated_after="2026-08-01")                  # what changed recently
search("cache", tags=["decision"], updated_after="2026-07-01", namespace="webapp")
```

Filtering never changes the ranking — results still come back by relevance, and only qualifying items are considered. From the CLI:

```bash
mnemomatic-cli search "deployment" --tag runbook --tag current
mnemomatic-cli search "auth" --updated-after 2026-08-01
```

### Related Items

`related(item_type, id)` returns the items most similar to one you already have — "more like this", without composing a query:

```
related(item_type="document", id="abc-123")            # neighbors across all types
related(item_type="knowledge", id="def-456", namespace="webapp", limit=10)
```

Results span all content types, ranked by embedding similarity, and never include the item itself. It needs an embedder (semantic search); chunked documents work through the centroid of their chunk vectors. Items stored while no embedder was configured have no vector and return an error suggesting a `MNEMOMATIC_REINDEX=1` restart.

### Embedding Info

`embedding_info()` reports the state semantic search depends on:

```json
{
  "semantic_search": true,
  "mode": "built-in ONNX (amaretto-embed-148m)",
  "model": "amaretto-embed-148m",
  "dimensions": 768,
  "index_model": "amaretto-embed-148m",
  "index_dimensions": 768,
  "matches_index": true,
  "query_prefix": "task: search result | query: ",
  "doc_prefix": "title: none | text: ",
  "max_tokens": 2048,
  "model_url": "https://huggingface.co/AmarettoLabs/amaretto-embed-148m"
}
```

The field worth checking is **`matches_index`**. Search only works when the model embedding your query is the one that embedded the stored content — query a model against another model's vectors and results come back plausible but wrong, with no error to notice. `false` means the index needs rebuilding (see [Switching Embedding Models](installation.md#switching-embedding-models)) and similarity scores mean little until it is.

`null` means unknowable rather than mismatched: the database was written before the server began recording which model built the index. `semantic_search: false` means no embedder is available at all — `semantic` mode will error and `hybrid` falls back to fulltext.

External endpoints report `endpoint` and `wire_api` in place of `max_tokens`.

### Example Usage

After connecting Claude Code, you can interact naturally:

> "Store a knowledge entry in the 'webapp' namespace: the API uses JWT with RS256 signing for authentication"

> "Search for anything related to authentication"

> "Store this deployment config as a document in the 'infra' namespace"

> "What do you know about the database setup?"

## HTTP Endpoints

Alongside the MCP transport, the server exposes two plain HTTP routes:

| Route | Auth | Purpose |
| ----- | ---- | ------- |
| `GET /health` | **none** | Liveness — `{"status": "ok"}`. Used by the images' `HEALTHCHECK`; see [Health Endpoint](installation.md#health-endpoint) |

Both routes are served on the server's own port (8000 inside the container). With the bundled Caddy setup that port is not published — reach them through the proxy: `https://your-server-hostname/export`, and `/health` on either `https://your-server-hostname/health` or plain `http://your-server-hostname/health`, which Caddy serves without the HTTPS redirect so probes work without TLS.
| `GET /export` | Bearer | The full store as a zip; optional `?namespace=` filter (see [Export](#export)) |

## Available Resources

MCP resources provide read-only access to browse stored data:

| Resource URI                         | Description                    |
| ------------------------------------ | ------------------------------ |
| `mnemomatic://namespaces`            | List all namespaces            |
| `mnemomatic://documents/{namespace}` | List documents in a namespace  |
| `mnemomatic://knowledge/{namespace}` | List knowledge in a namespace  |
| `mnemomatic://notes/{namespace}`     | List notes in a namespace      |
| `mnemomatic://document/{id}`         | Get a specific document        |
| `mnemomatic://knowledge-entry/{id}`  | Get a specific knowledge entry |
| `mnemomatic://note/{id}`             | Get a specific note            |
