---
name: mnemomatic
description: >
  Use the Mnem-O-matic shared memory MCP server to load context, search stored knowledge,
  and persist information across sessions. Invoke at session start to load relevant context,
  before answering questions that may have prior history, after discovering information
  worth remembering, and to fix memory mistakes (restore) or tidy a namespace (consolidate).
  Aliases: memory server, mnemomatic, mnemo, mcp memory.
---

# Mnem-O-matic — Shared Memory for LLMs

A persistent shared memory layer: recall past decisions, load project context, and store
anything worth remembering for future sessions.

## When to Reach for Memory

**Read** when:
- starting work in a project you haven't touched this session
- the user references past work — "like we did before", "what did we decide about X"
- you're about to say "I don't know" or "there's no prior decision on this" — check first
- a task depends on conventions, constraints, or choices that aren't in the code

**Write** when:
- the user says to remember something (store it as stated, don't paraphrase away details)
- a decision is made or reversed in conversation — capture the *why*, not just the outcome
- you solved a non-obvious problem, or learned a user preference or correction
- a task ends — pause and store what a future session would need before the context is gone

**Never store**: secrets, tokens, or credentials (this is shared memory); mid-task
ephemera that won't matter tomorrow; anything the codebase or git history already records.

## Recall

**Search before answering questions or starting work** — a prior session may have already
captured relevant context — and search again before storing, to avoid duplicates:

```
search("<topic of the user's request>")
search("recent decisions")
```

| Mode | When to use |
|------|-------------|
| `hybrid` (default) | General purpose — catches exact matches and conceptually related content |
| `fulltext` | A specific name, term, or exact phrase |
| `semantic` | A concept or question where stored content may use different words |

Results carry a `snippet` preview; call `read(item_type, id)` for full content before
relying on an item. A document hit with `partial: true` matched one chunk of a larger
document — the snippet is the best passage, not the whole file, so always `read` it.

For a thorough context load at task start, the server's `briefing` prompt (a slash command
in Claude Code) packages the whole workflow: multi-query search → read → summarize.

## Store

| Type | Use for | Deduplication key |
|------|---------|-------------------|
| `store_document` | Long-form reference material: specs, documentation, configs, runbooks, code files, API schemas | namespace + title |
| `store_knowledge` | Single atomic facts: decisions, technology choices, conventions, constraints | namespace + subject |
| `store_note` | Informal content: rough thoughts, meeting notes, observations, transcripts | namespace + title |

Rule of thumb: a confirmed fact or decision → knowledge; structured and reusable →
document; rough or exploratory → note.

```
store_knowledge(
    namespace="myproject",
    subject="auth mechanism",        # short label — the deduplication key
    fact="Uses JWT with RS256 signing, tokens expire after 1 hour",
    source="code-review",
    confidence=1.0                   # use < 1.0 for inferred or tentative facts
)
```

Storing with an existing key never creates a duplicate: documents and notes update in
place (`created: false`); knowledge is temporal (next section).

**Respond to `similar`:** when a store response includes a `similar` list, the server
found near-identical existing items. Read them and either merge your content into the
existing item, store the corrected fact so it supersedes, or delete your redundant
addition — don't leave both.

## Temporal Knowledge

Facts change, and Mnem-O-matic keeps the timeline. `store_knowledge` on an existing subject:

- **Same fact again** → refreshes the entry in place. Safe to re-store what you know.
- **Different fact** → the old entry is closed as history and yours becomes the current
  answer (`created: true` plus `"superseded": "<old-id>"`).

**To correct a fact, store it — never delete-and-recreate** (that destroys the history
supersession would keep). Search and listings surface only current answers; the past is
one call away:

```
fact_history(namespace="myproject", subject="auth mechanism")
→ current entry first, then superseded versions with valid_until / superseded_by
```

History entries are immutable — always update the current entry. Changing only
`confidence`, `source`, `tags`, or `metadata` edits in place without creating history.

## Edit, Undo, Recover

- `update_*` changes specific fields without rewriting the whole entry.
- `tag` adds/removes tags without touching anything else — prefer it over `update_*` for
  tag-only changes.
- Every update and delete first saves the prior state as a **revision** (a limited number
  per item), so mistakes are recoverable — prefer `restore` over re-typing content:

```
list_revisions(item_id="abc-123")       # one item's history
list_revisions(namespace="myproject")   # recent changes, including deletes
restore(revision_id=42)                 # roll back an update, or undelete
```

`restore` on a live item rolls its content back (and is itself undoable); on a deleted
item it recreates it with its original id.

## Namespaces

Namespaces scope content: `global` for cross-project facts and preferences,
`<project-name>` for project content, `personal` for user notes. Omit `namespace` in
search to look everywhere.

- Browse instead of search: `list_items(item_type, namespace, limit, offset)` — paginated
  summaries, newest first (`offset + len(items) < total` means more pages).
- Reorganize: `rename_namespace(old, new)` — atomic; merges into an existing target, where
  a title/subject conflict means the moved item replaces the target's (`replaced` counts
  in the response). Rename colliding items first if both versions matter.
- Remove: `delete_namespace(namespace)` — items remain individually recoverable via
  revisions, but there is no bulk undo, so treat it as destructive.

## Maintenance

`consolidation_report(namespace)` flags near-duplicate clusters and stale never-retrieved
items — candidates only; reviewing and acting is your job. The server's `consolidate`
prompt walks the workflow: read every flagged item before merging, superseding, tagging,
or deleting.

Every write is recorded in an append-only audit trail — `list_audit(namespace=...)` shows
recent activity (who changed what, when), `list_audit(item_id=...)` traces one item.
