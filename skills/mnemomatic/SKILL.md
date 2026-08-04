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

Large documents are split into overlapping chunks at store time, so a document hit's
`snippet` is the single most relevant passage — not the whole file. This keeps search
focused without flooding your context with entire documents. Always `read` the document
to retrieve its complete content before relying on it; the matched passage alone may omit
context elsewhere in the file.

## What to Store

| Type | Use for | Deduplication key |
|------|---------|-------------------|
| `store_document` | Long-form reference material: specs, documentation, configs, runbooks, code files, API schemas | namespace + title |
| `store_knowledge` | Single atomic facts: decisions, technology choices, conventions, constraints | namespace + subject |
| `store_note` | Informal content: rough thoughts, meeting notes, observations, transcripts, temporary items | namespace + title |

All three deduplicate on their key. Documents and notes **upsert** — storing with the same
key updates in place (`created: false`). Knowledge is **temporal** — see the next section.

**Respond to `similar`:** when a store response includes a `similar` list, the server found
near-identical existing items. Don't ignore it — read them and either merge your content into
the existing item, let the existing fact be superseded, or delete your redundant addition.

## Temporal Knowledge

Facts change, and Mnem-O-matic keeps the timeline. `store_knowledge` on an existing subject:

- **Same fact again** → refreshes the entry in place (`created: false`). Safe to re-store
  what you already know.
- **Different fact** → the old entry is closed as history and your fact becomes its
  successor: `created: true` plus `"superseded": "<old-id>"`.

So to correct or update a fact, just store it — never delete-and-recreate. Search and
listings only surface the current answer. To see what was believed before (and until when):

```
fact_history(namespace="myproject", subject="auth mechanism")
→ current entry first, then superseded versions with valid_until / superseded_by
```

History entries are immutable — update the current fact, not a superseded one. Changing
only `confidence`, `source`, `tags`, or `metadata` via `update_knowledge` edits in place
without creating history.

## Undo & Recovery

Every update and delete first saves the item's prior state as a **revision** (a limited
number are kept per item). Mistakes are recoverable:

```
list_revisions(item_id="abc-123")            # one item's history
list_revisions(namespace="myproject")        # what changed recently, incl. deletes
restore(revision_id=42)                      # roll back an update, or undelete
```

`restore` on a live item rolls its content back (and is itself undoable); on a deleted item
it recreates it with the original id. Prefer fixing a bad edit with `restore` over
re-typing content from memory.

## Memory Hygiene

`consolidation_report(namespace)` returns near-duplicate clusters (from embeddings) and
stale items (never retrieved, not recently updated) — candidates only; reviewing and acting
is your job. Two server prompts package the workflows (as slash commands in Claude Code):
`consolidate` (walk the report; read everything before merging or deleting) and `briefing`
(assemble memory context for a task before starting work).

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

To browse or inventory a namespace (rather than search by topic), use `list_items` — it
returns paginated summaries, newest first:

```
list_items(item_type="document", namespace="myproject", limit=20, offset=0)
```

The response's `total` tells you when there are more pages (`offset + len(items) < total`).
Summaries omit document/note bodies; `read` an item for its full content.

Use `rename_namespace` to rename a namespace atomically across all content types:

```
rename_namespace(old_namespace="old-name", new_namespace="new-name")
```

This also works as a merge — if `new_namespace` already exists, items from `old_namespace`
are moved into it. On a title/subject conflict the moved item replaces the target's item
(the same upsert semantics as the store tools); the response's `replaced` counts tell you
how many target items were overwritten. If both versions matter, rename the colliding
items first.

Use `delete_namespace` to remove all items in a namespace at once:

```
delete_namespace(namespace="old-project")
```

Prefer `rename_namespace` if you only want to reorganize content. Deleted items can be
recovered individually via `list_revisions`/`restore` while their revisions last, but
there is no one-call undo for a whole namespace.

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
- Duplicates — always search first to avoid creating redundant entries (and act on the
  `similar` field when a store response includes it)
