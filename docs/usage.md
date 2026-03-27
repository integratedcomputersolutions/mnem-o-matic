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

## CLI Interface

`mnemomatic-cli` provides shell access to a running Mnem-O-matic server for agents and users without MCP support.

### Installation

```bash
git clone https://github.com/integratedcomputersolutions/mnem-o-matic.git
cd mnem-o-matic
uv tool install .
```

This installs `mnemomatic-cli` into an isolated environment. Verify with:

```bash
mnemomatic-cli --help
```

To uninstall: `uv tool uninstall mnemomatic`

For development (runs from source without installing):

```bash
uv run mnemomatic-cli --help
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

# Store
mnemomatic-cli store document myproject "API spec" "Full API specification text"
mnemomatic-cli store knowledge myproject "auth method" "Uses JWT with RS256"
mnemomatic-cli store note myproject "Quick thought" "Consider adding rate limiting"

# Read from stdin (use '-' as content)
cat spec.md | mnemomatic-cli store document myproject "API spec" -

# Update
mnemomatic-cli update document <id> --content "Updated content"
mnemomatic-cli update knowledge <id> --fact "Migrated to session cookies"

# Delete
mnemomatic-cli delete document <id>

# Get full content by ID
mnemomatic-cli get document <id>

# Tags
mnemomatic-cli tag <id> document --add prod --add critical --remove draft

# Browse
mnemomatic-cli namespaces
mnemomatic-cli list documents myproject
```

All output is JSON. Use `--pretty` for indented output:

```bash
mnemomatic-cli --pretty search "auth"
```

## Available Tools

Once connected, your LLM has access to these tools:

| Tool               | Description                               |
| ------------------ | ----------------------------------------- |
| `store_document`   | Save a document (code, spec, config)      |
| `store_knowledge`  | Save a fact, decision, or observation     |
| `store_note`       | Save a quick thought, idea, or transcript |
| `update_document`  | Modify an existing document               |
| `update_knowledge` | Modify an existing knowledge entry        |
| `update_note`      | Modify an existing note                   |
| `delete_document`  | Remove a document                         |
| `delete_knowledge` | Remove a knowledge entry                  |
| `delete_note`      | Remove a note                             |
| `tag`              | Add or remove tags on any entry           |
| `search`           | Search across all stored data             |

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

Store tools use upsert semantics — if an entry with the same namespace and title (for documents) or namespace and subject (for knowledge) already exists, it is updated in place rather than creating a duplicate.

This matters because LLMs don't track what's already stored. Without deduplication, restarting a session and re-storing the same facts would create duplicate rows. With upsert, the second call updates the existing entry and the response includes `"created": false` so the caller knows it was an update. Notes deduplicate on namespace + title.

```
# First call — creates a new entry
store_knowledge(namespace="webapp", subject="auth method", fact="Uses JWT with RS256")
→ {"id": "abc-123", "created": true}

# Second call — same namespace + subject, updates in place
store_knowledge(namespace="webapp", subject="auth method", fact="Migrated to session cookies")
→ {"id": "abc-123", "created": false}
```

### Search Modes

The `search` tool supports three modes:

- **fulltext** — keyword and phrase matching via SQLite FTS5
- **semantic** — meaning-based search via vector embeddings
- **hybrid** (default) — combines both, ranked by a blended score

### Example Usage

After connecting Claude Code, you can interact naturally:

> "Store a knowledge entry in the 'webapp' namespace: the API uses JWT with RS256 signing for authentication"

> "Search for anything related to authentication"

> "Store this deployment config as a document in the 'infra' namespace"

> "What do you know about the database setup?"

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
