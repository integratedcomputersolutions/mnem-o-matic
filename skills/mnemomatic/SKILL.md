---
name: mnemomatic
description: >
  Use the Mnem-O-matic shared memory MCP server to load context, search stored knowledge,
  and persist information across sessions. Invoke at session start to load relevant context,
  before answering questions that may have prior history, and after discovering information
  worth remembering. Aliases: memory server, mnemomatic, mnemo, mcp memory.
---

# Mnem-O-matic — Shared Memory for LLMs

Mnem-O-matic is a persistent shared memory layer. Use it to recall past decisions, load
project context, and store anything worth remembering for future sessions.

## Session Start

Always search before answering questions or starting work — a prior session may have already
captured relevant context:

```
search("current project goals")
search("recent decisions")
search("<topic of the user's request>")
```

## Searching

Three modes — pick the right one:

| Mode | When to use |
|------|-------------|
| `hybrid` (default) | General purpose — catches exact matches and conceptually related content |
| `fulltext` | Looking for a specific name, term, or exact phrase |
| `semantic` | Concept or question where stored content may use different words |

Search results include a `snippet` (preview) and a `resource_uri`. To get the full content
of a result, call the `read` tool with the item's `item_type` and `id`:

```
search("authentication") → result with resource_uri: "mnemomatic://document/abc-123"
read(item_type="document", id="abc-123") → full content
```

## What to Store

| Type | Use for | Deduplication key |
|------|---------|-------------------|
| `store_document` | Long-form reference material: specs, documentation, configs, runbooks, code files, API schemas | namespace + title |
| `store_knowledge` | Single atomic facts: decisions, technology choices, conventions, constraints | namespace + subject |
| `store_note` | Informal content: rough thoughts, meeting notes, observations, transcripts, temporary items | namespace + title |

All three use **upsert semantics** — storing with the same key updates in place. Check `created`
in the response: `true` = new entry, `false` = updated existing.

**Rule of thumb:**
- More than two sentences? → document or note
- A confirmed fact or decision? → knowledge
- Still rough or exploratory? → note
- Structured and reusable? → document

## Namespaces

Namespaces scope content to a project or context. Use consistent names:

- `global` — cross-project facts, conventions, user preferences
- `<project-name>` — project-specific content
- `personal` — user-specific notes not tied to a project

When searching, omit `namespace` to search globally across all namespaces.

## Storing Good Knowledge Entries

```
store_knowledge(
    namespace="myproject",
    subject="auth mechanism",        # short label — the deduplication key
    fact="Uses JWT with RS256 signing, tokens expire after 1 hour",
    source="code-review",
    confidence=1.0
)
```

Use `confidence < 1.0` for inferred or tentative facts.

## Updating and Tagging

Use `update_*` to change specific fields without rewriting the whole entry.
Use `tag` to add/remove tags without touching other fields — prefer this over `update_*`
when only tags need to change.

## What NOT to Store

- Information already in the codebase, git history, or documentation
- Duplicates — always search first to avoid creating redundant entries
