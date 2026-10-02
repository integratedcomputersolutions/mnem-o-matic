<script>
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import Pager from '../components/Pager.svelte';
  import { api, qs } from '../lib/api.js';
  import { navigate, seg } from '../lib/router.svelte.js';
  import { fmtRelative, fmtNumber, truncate } from '../lib/format.js';

  let { ns, type = 'document' } = $props();
  const TYPES = [['document', 'Documents'], ['knowledge', 'Knowledge'], ['note', 'Notes']];
  const LIMIT = 50;

  let offset = $state(0);
  let page = $state(null);
  let error = $state(null);

  $effect(() => {
    const url = `/api/items${qs({ namespace: ns, type, limit: LIMIT, offset })}`;
    api.get(url).then((r) => (page = r)).catch((e) => (error = e));
  });

  function switchType(t) {
    offset = 0;
    navigate(`/browse/${seg(ns)}/${t}`);
  }
</script>

<PageHeader title={ns} subtitle="Namespace">
  <p class="small mt"><a href="/browse">← All namespaces</a></p>
</PageHeader>
<ErrorBox {error} />

<div class="tabs" role="tablist">
  {#each TYPES as [t, label] (t)}
    <button type="button" role="tab" class="tab" class:on={type === t} aria-selected={type === t} onclick={() => switchType(t)}>{label}</button>
  {/each}
</div>

{#if page}
  {#if page.items.length === 0}
    <Empty text={`No ${type === 'knowledge' ? 'knowledge' : type + 's'} in this namespace.`} />
  {:else}
    <Card flush>
      <div class="table-wrap"><table class="table">
        <thead>
          <tr>
            <th>{type === 'knowledge' ? 'Subject' : 'Title'}</th>
            {#if type === 'knowledge'}<th>Fact</th><th class="num">Confidence</th>{:else}<th>{type === 'document' ? 'Type' : 'Source'}</th>{/if}
            <th>Tags</th><th>Updated</th><th class="num">Reads</th>
          </tr>
        </thead>
        <tbody>
          {#each page.items as it (it.id)}
            <tr>
              <td><a href={`/browse/${seg(ns)}/${type}/${seg(it.id)}`}>{it.title || it.subject}</a></td>
              {#if type === 'knowledge'}
                <td class="dim">{truncate(it.fact, 140)}</td>
                <td class="num">{it.confidence}</td>
              {:else}
                <td class="muted small">{it.mime_type || it.source || '—'}</td>
              {/if}
              <td><span class="pill-list">{#each it.tags || [] as t}<span class="badge">{t}</span>{/each}</span></td>
              <td class="nowrap muted small">{fmtRelative(it.updated_at)}</td>
              <td class="num muted">{fmtNumber(it.retrieval_count)}</td>
            </tr>
          {/each}
        </tbody>
      </table></div>
      <div style="padding:0 16px 12px"><Pager total={page.total} limit={LIMIT} bind:offset /></div>
    </Card>
  {/if}
{/if}

<style>
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
  .tab { background: none; border: 0; border-bottom: 2px solid transparent; margin-bottom: -1px; color: var(--ink-2);
         padding: 8px 12px; font: inherit; font-weight: 550; cursor: pointer; }
  .tab.on { color: var(--ink); border-bottom-color: var(--brand-bright); }
</style>
