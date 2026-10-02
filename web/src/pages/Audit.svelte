<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import Pager from '../components/Pager.svelte';
  import { api, qs } from '../lib/api.js';
  import { seg } from '../lib/router.svelte.js';
  import { fmtDate } from '../lib/format.js';

  const LIMIT = 50;
  let filters = $state({ op: '', actor: '', namespace: '', item_type: '' });
  let applied = $state({ op: '', actor: '', namespace: '', item_type: '' });
  let offset = $state(0);
  let page = $state(null);
  let error = $state(null);

  $effect(() => {
    api.get(`/api/audit${qs({ ...applied, limit: LIMIT, offset })}`).then((r) => (page = r)).catch((e) => (error = e));
  });

  function apply(e) {
    e.preventDefault();
    offset = 0;
    applied = { ...filters };
  }
  function detailText(d) {
    if (!d) return '';
    const copy = { ...d };
    if (copy.token) { copy.token = copy.token.hint; }
    const parts = Object.entries(copy).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`);
    return parts.join('  ');
  }
</script>

<PageHeader title="Activity" subtitle="Every write and every sign-in, with who did it. Read-only here." />
<ErrorBox {error} />

<Card>
  <form class="filters" onsubmit={apply}>
    <input class="input" bind:value={filters.actor} placeholder="Actor (username)" />
    <input class="input" bind:value={filters.op} placeholder="Operation, e.g. store or auth.login" />
    <input class="input" bind:value={filters.namespace} placeholder="Namespace" />
    <select class="select" bind:value={filters.item_type} aria-label="Item type">
      <option value="">Any item type</option>
      {#each ['document', 'knowledge', 'note', 'user', 'token', 'https', 'schema'] as t}<option value={t}>{t}</option>{/each}
    </select>
    <button class="btn" type="submit">Filter</button>
  </form>
</Card>

{#if page}
  {#if page.events.length === 0}
    <div class="mt"><Empty text="No matching events." /></div>
  {:else}
    <div class="mt"><Card flush>
      <div class="table-wrap"><table class="table">
        <thead><tr><th>When</th><th>Actor</th><th>Operation</th><th>Item</th><th>Via</th><th>Detail</th></tr></thead>
        <tbody>
          {#each page.events as e (e.id)}
            <tr>
              <td class="nowrap muted small">{fmtDate(e.ts)}</td>
              <td>{e.actor || '—'}</td>
              <td><code>{e.op}</code></td>
              <td style="max-width:260px" class="truncate">
                {#if e.namespace && ['document','knowledge','note'].includes(e.item_type) && e.item_id}
                  <a href={`/browse/${seg(e.namespace)}/${e.item_type}/${seg(e.item_id)}`}>{e.title || e.item_id}</a>
                  <span class="muted small"> · {e.namespace}</span>
                {:else}
                  {e.title || e.item_id || e.namespace || ''}{#if e.item_type}<span class="muted small"> ({e.item_type})</span>{/if}
                {/if}
              </td>
              <td class="muted small nowrap">{e.detail?.token ? `token ${e.detail.token.name || e.detail.token.hint}` : (e.client ? e.client.split(' ')[0] : '—')}{#if e.ip}&nbsp;· {e.ip}{/if}</td>
              <td class="muted small mono">{detailText(e.detail)}</td>
            </tr>
          {/each}
        </tbody>
      </table></div>
      <div style="padding:0 16px 12px"><Pager total={page.total} limit={LIMIT} bind:offset /></div>
    </Card></div>
  {/if}
{/if}

<style>
  .filters { display: grid; grid-template-columns: repeat(4, 1fr) auto; gap: 8px; }
  @media (max-width: 860px) { .filters { grid-template-columns: 1fr 1fr; } }
</style>
