<script>
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import TypeBadge from '../components/TypeBadge.svelte';
  import StatusBadge from '../components/StatusBadge.svelte';
  import { api } from '../lib/api.js';
  import { seg } from '../lib/router.svelte.js';
  import { fmtDate, fmtNumber, itemTitle, typeLabel } from '../lib/format.js';

  let { ns, type, id } = $props();
  let item = $state(null);
  let revisions = $state([]);
  let related = $state(null);
  let error = $state(null);

  $effect(() => {
    api.get(`/api/items/${type}/${seg(id)}`).then((r) => (item = r.item)).catch((e) => (error = e));
    api.get(`/api/items/${type}/${seg(id)}/revisions`).then((r) => (revisions = r.revisions)).catch(() => {});
    api.get(`/api/items/${type}/${seg(id)}/related`).then((r) => (related = r)).catch(() => {});
  });

  const body = $derived(item ? (type === 'knowledge' ? item.fact : item.content) : '');
  const metaEntries = $derived(item ? Object.entries(item.metadata || {}) : []);
</script>

<PageHeader title={item ? itemTitle(item) : '…'} subtitle={typeLabel(type)}>
  <p class="small mt"><a href="/browse">Browse</a> › <a href={`/browse/${seg(ns)}`}>{ns}</a> › <a href={`/browse/${seg(ns)}/${type}`}>{typeLabel(type)}</a></p>
</PageHeader>
<ErrorBox {error} />

{#if item}
  <div class="layout">
    <div class="stack">
      <Card title={type === 'knowledge' ? 'Fact' : 'Content'}>
        {#if type === 'knowledge' && item.valid_until}
          <div class="alert warn mb">This fact was superseded on {fmtDate(item.valid_until)}{#if item.superseded_by} by <a href={`/browse/${seg(ns)}/knowledge/${seg(item.superseded_by)}`}>a newer entry</a>{/if}.</div>
        {/if}
        <pre class="content">{body}</pre>
      </Card>
      {#if metaEntries.length}
        <Card title="Metadata">
          <dl class="kv">
            {#each metaEntries as [k, v] (k)}
              <dt>{k}</dt><dd><code>{typeof v === 'string' ? v : JSON.stringify(v)}</code></dd>
            {/each}
          </dl>
        </Card>
      {/if}
      {#if revisions.length}
        <Card title="Revisions" subtitle="Earlier states captured on update or delete" flush>
          <table class="table">
            <thead><tr><th>#</th><th>Change</th><th>Title then</th><th>Captured</th></tr></thead>
            <tbody>
              {#each revisions as r (r.id)}
                <tr><td class="muted">{r.id}</td><td><code>{r.op}</code></td><td>{r.title}</td><td class="muted nowrap">{fmtDate(r.revised_at)}</td></tr>
              {/each}
            </tbody>
          </table>
        </Card>
      {/if}
    </div>

    <div class="stack">
      <Card title="Details">
        <dl class="kv">
          <dt>Type</dt><dd><TypeBadge {type} /></dd>
          <dt>Namespace</dt><dd><a href={`/browse/${seg(ns)}`}>{item.namespace}</a></dd>
          <dt>ID</dt><dd><code class="small">{item.id}</code></dd>
          {#if type === 'document'}<dt>MIME</dt><dd>{item.mime_type}</dd>{/if}
          {#if type !== 'document'}<dt>Source</dt><dd>{item.source}</dd>{/if}
          {#if type === 'knowledge'}<dt>Confidence</dt><dd>{item.confidence}</dd>{/if}
          <dt>Tags</dt><dd><span class="pill-list">{#each item.tags || [] as t}<span class="badge">{t}</span>{:else}<span class="muted">none</span>{/each}</span></dd>
          <dt>Created</dt><dd>{fmtDate(item.created_at)}</dd>
          <dt>Updated</dt><dd>{fmtDate(item.updated_at)}</dd>
          <dt>Retrieved</dt><dd>{fmtNumber(item.retrieval_count)} times{#if item.last_accessed}, last {fmtDate(item.last_accessed)}{/if}</dd>
        </dl>
      </Card>
      <Card title="Related" subtitle="Nearest by embedding">
        {#if !related}
          <span class="muted">…</span>
        {:else if related.unavailable}
          <StatusBadge tone="neutral" label={related.unavailable} />
        {:else if related.related.length === 0}
          <span class="muted">Nothing similar enough.</span>
        {:else}
          <ul class="rel">
            {#each related.related as r (r.id)}
              <li>
                <a href={`/browse/${seg(r.namespace)}/${r.type}/${seg(r.id)}`}>{r.title}</a>
                <span class="muted small"> · {r.namespace} · {Math.round(r.score * 100) / 100}</span>
              </li>
            {/each}
          </ul>
        {/if}
      </Card>
    </div>
  </div>
{/if}

<style>
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; align-items: start; }
  @media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
  .content { max-height: 70vh; overflow: auto; font-size: 13.5px; }
  .rel { margin: 0; padding-left: 18px; }
  .rel li { margin: 4px 0; }
</style>
