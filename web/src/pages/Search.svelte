<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import TypeBadge from '../components/TypeBadge.svelte';
  import { api, qs } from '../lib/api.js';
  import { action } from '../lib/load.svelte.js';
  import { route, navigate, itemHref } from '../lib/router.svelte.js';

  let q = $state(route.query.get('q') || '');
  let namespace = $state(route.query.get('namespace') || '');
  let type = $state(route.query.get('type') || 'all');
  let mode = $state(route.query.get('mode') || 'hybrid');
  let namespaces = $state([]);
  let result = $state(null);
  const search = action();

  $effect(() => {
    api.get('/api/namespaces').then((r) => (namespaces = r.namespaces)).catch(() => {});
  });
  $effect(() => {
    if (q.trim()) run();
  });

  // Reads its inputs before the first await, so the $effect above re-runs on any of them.
  const run = () => search.run(async () => {
    try {
      result = await api.get(`/api/search${qs({ q: q.trim(), namespace, type, mode, limit: 30 })}`);
      navigate(`/search${qs({ q: q.trim(), namespace, type, mode })}`, { replace: true });
    } catch (e) {
      result = null;
      throw e;
    }
  });
  function submit(e) {
    e.preventDefault();
    if (q.trim()) run();
  }
</script>

<PageHeader title="Search" subtitle="Keyword, meaning, or both — the same search the agents use." />

<Card>
  <form class="form" onsubmit={submit}>
    <input class="input q" bind:value={q} placeholder="What are you looking for?" />
    <select class="select" bind:value={namespace} aria-label="Namespace">
      <option value="">All namespaces</option>
      {#each namespaces as n (n.name)}<option value={n.name}>{n.name}</option>{/each}
    </select>
    <select class="select" bind:value={type} aria-label="Type">
      <option value="all">All types</option><option value="documents">Documents</option>
      <option value="knowledge">Knowledge</option><option value="notes">Notes</option>
    </select>
    <select class="select" bind:value={mode} aria-label="Mode">
      <option value="hybrid">Hybrid</option><option value="fulltext">Full text</option><option value="semantic">Semantic</option>
    </select>
    <button class="btn primary" type="submit" disabled={search.busy || !q.trim()}>{search.busy ? '…' : 'Search'}</button>
  </form>
</Card>

<div class="mt"><ErrorBox error={search.error} /></div>

{#if result}
  {#if result.degraded}<div class="alert warn mt">Semantic search is unavailable, so these are keyword matches only.</div>{/if}
  {#if result.results.length === 0}
    <div class="mt"><Empty text="No matches." /></div>
  {:else}
    <div class="stack mt">
      {#each result.results as r (r.id)}
        <a class="hit" href={itemHref(r.namespace, r.type, r.id)}>
          <div class="row between">
            <span class="row"><TypeBadge type={r.type} /><b>{r.title}</b></span>
            <span class="muted small nowrap">{r.namespace} · score {Math.round(r.score * 1000) / 1000}{#if r.partial} · excerpt{/if}</span>
          </div>
          <div class="snippet">{r.snippet}</div>
          {#if r.tags?.length}<div class="pill-list mt-s">{#each r.tags as t}<span class="badge">{t}</span>{/each}</div>{/if}
        </a>
      {/each}
    </div>
  {/if}
{/if}

<style>
  .form { display: grid; grid-template-columns: 1fr auto auto auto auto; gap: 8px; align-items: center; }
  @media (max-width: 860px) { .form { grid-template-columns: 1fr 1fr; } .q { grid-column: 1 / -1; } }
  .hit { display: block; background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 12px 16px; color: inherit; }
  .hit:hover { text-decoration: none; border-color: var(--line-strong); background: var(--hover); }
  .snippet { color: var(--ink-2); font-size: 14px; margin-top: 6px; white-space: pre-wrap; word-break: break-word;
             display: -webkit-box; -webkit-line-clamp: 4; line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
  .mt-s { margin-top: 8px; }
</style>
