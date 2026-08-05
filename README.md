<div align="center">
<img src="assets/mnem-o-matic.png" alt="Mnem-O-matic logo" width="512" height="512">

[![CI](https://github.com/integratedcomputersolutions/mnem-o-matic/actions/workflows/ci.yml/badge.svg)](https://github.com/integratedcomputersolutions/mnem-o-matic/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://github.com/integratedcomputersolutions/mnem-o-matic/pkgs/container/mnem-o-matic)

</div>

Shared memory layer for LLMs. Store documents, knowledge and notes in a single portable database and access them from any MCP-compatible client — Claude Code, VS Code Copilot, ChatGPT, Mistral Vibe, custom agents, or anything that speaks MCP.

Runs privately in a Docker container or natively. Your data never leaves your machine.

## The Problem

Every LLM session starts from scratch. Claude doesn't know what ChatGPT learned yesterday. Your Copilot session can't access the architectural decisions you discussed with Claude last week. Each tool operates in complete isolation.

Mnem-O-matic fixes this by providing a shared, persistent memory that any LLM can read from and write to.

## What It Stores

**Documents** — reference material, code snippets, specs, configs, notes. Anything you want LLMs to have access to.

**Knowledge** — discrete facts, decisions, and observations. "The auth system uses JWT with RS256." "We chose Postgres over SQLite for the main database." "The deploy pipeline runs on GitHub Actions."

**Notes** — quick thoughts, ideas, observations, and voice transcripts. Informal content that LLMs should be aware of but that isn't structured enough to be a document or atomic enough to be a knowledge entry.

All types support namespaces (per-project or global), tags, and metadata. Everything is searchable via full-text and semantic search, narrowed when you need it by tag or by "updated since". Large documents are automatically split into chunks at store time, so search returns the most relevant passage rather than the entire file — giving agents focused context without burning their context window.

## A Memory, Not a Filing Cabinet

Content has history, mistakes are reversible, and the store helps keep itself tidy:

- **Temporal facts** — knowledge answers questions whose answers change. When a fact changes, the old entry is superseded rather than overwritten: search returns only the current answer, and `fact_history` shows what was believed before, and until when. [More →](docs/usage.md#temporal-facts)
- **Undo & recovery** — every update and delete first saves the item's prior state as a revision; `restore` rolls back a bad edit or recreates a deleted item under its original id. [More →](docs/usage.md#usage-tracking--revisions)
- **Duplicate awareness & consolidation** — storing near-identical content gets flagged in the store response, and `consolidation_report` clusters look-alike items and lists stale, never-retrieved ones. The bundled `consolidate` and `briefing` prompts turn review into one-command workflows — no server-side LLM involved, the connected agent is the judge. [More →](docs/usage.md#memory-hygiene-duplicates-consolidation-prompts)
- **Associative recall** — `related` returns an item's nearest neighbors across all content types, so an agent that just read one thing can pull in the surrounding context it didn't know to search for. [More →](docs/usage.md#related-items)
- **Usage tracking** — items carry retrieval counters, bumped only when something is genuinely read or surfaced by search. The raw material for spotting what earns its place. [More →](docs/usage.md#usage-tracking--revisions)
- **Audit trail** — every write lands in an append-only log: what changed, when, from which client and address, and — when clients send an `X-Mnemomatic-Actor` header — who. Two-year retention by default. [More →](docs/usage.md#audit-log)

## Backups & Export

The whole store downloads as a **human-readable zip** — one folder per namespace, one Markdown file per item, metadata in sidecars — via `GET /export`, the web viewer, or the CLI. Your memory stays portable and is never locked in. The server can also write that archive on a schedule with rotation: set `MNEMOMATIC_BACKUP_DIR` and backups happen with no host-side cron. [More →](docs/usage.md#export)

## Embedding Model

Semantic search runs on a local embedding model bundled into the Docker image — nothing leaves your machine. Three models are selectable at build time via the `EMBED_MODEL` build argument: **MiniLM** (the default) is the smallest and fastest but also the most limited — English only, and the weakest at paraphrased queries; **gte-multilingual-base** adds strong multilingual retrieval at near-MiniLM query speed; **EmbeddingGemma** has the best retrieval quality of the three — it resolves paraphrased queries that share no words with the stored content — at a higher CPU and memory cost. You can also bypass the built-in model and point `MNEMOMATIC_EMBED_URL` at any OpenAI-compatible embedding endpoint. See [choosing the built-in embedding model](docs/installation.md#choosing-the-built-in-embedding-model) for the full comparison.

## Agent Skill

A sample agent skill file is included at `skills/mnemomatic/SKILL.md`. It teaches an agent how to use Mnem-O-matic effectively — when to reach for memory at all, which search mode to pick, what content type to store, how facts supersede, and how to undo mistakes.

The skill is written for Claude Code but can be adapted to any agent framework that supports custom instructions or skill files. Tailor the wording, triggers, and examples to match your agent's terminology and workflow.

To install for Claude Code:

```bash
# Personal (available in all your projects)
mkdir -p ~/.claude/skills && cp -r skills/mnemomatic ~/.claude/skills/mnemomatic

# Project-only (available in the current project)
mkdir -p .claude/skills && cp -r skills/mnemomatic .claude/skills/mnemomatic
```

## Web Viewer

A built-in, read-only web viewer lets you browse stored documents, knowledge, and notes in the browser — no MCP client required. It's view-only: no creating, editing, or deleting.

<div align="center">
<table>
<tr>
<td align="center"><a href="assets/mnemomatic-ui-login.png"><img src="assets/mnemomatic-ui-login.png" alt="Shared-secret login" width="360"></a><br><sub>Shared-secret login</sub></td>
<td align="center"><a href="assets/mnemomatic-ui-namespaces.png"><img src="assets/mnemomatic-ui-namespaces.png" alt="Namespaces overview" width="360"></a><br><sub>Namespaces</sub></td>
</tr>
<tr>
<td align="center"><a href="assets/mnemomatic-ui-content.png"><img src="assets/mnemomatic-ui-content.png" alt="Browsing a namespace" width="360"></a><br><sub>Browsing a namespace</sub></td>
<td align="center"><a href="assets/mnemomatic-ui-content-details.png"><img src="assets/mnemomatic-ui-content-details.png" alt="Item detail" width="360"></a><br><sub>Item detail</sub></td>
</tr>
</table>
<sub><i>Click any image to view full size.</i></sub>
</div>

The viewer is disabled by default. Set a shared secret to enable it:

```bash
docker run -e MNEMOMATIC_UI_TOKEN=your-viewer-secret ...
```

Then open `http://your-host:8000/ui` and enter the token once. There are no user accounts — access is a single shared secret, kept separate from the MCP API key. When `MNEMOMATIC_UI_TOKEN` is unset, `/ui` is not served at all.

A **Settings** page shows the configuration the server is running with — embedding model (linked to its model card), dimensions, task prefixes, chunking — and offers the export download.

See the [Usage Guide](docs/usage.md#web-viewer) for details and security notes.

## Documentation

- [Installation Guide](docs/installation.md) — prerequisites, Docker profiles, TLS setup, configuration, development
- [Usage Guide](docs/usage.md) — connecting clients, authentication, tools, search, resources, web viewer
- [Tech Stack](docs/tech-stack.md) — architecture decisions, embeddings, concurrency, performance

## License

[Apache License 2.0](LICENSE)
