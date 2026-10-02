<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import StatTile from '../components/StatTile.svelte';
  import StatusBadge from '../components/StatusBadge.svelte';
  import Card from '../components/Card.svelte';
  import TableCard from '../components/TableCard.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import Empty from '../components/Empty.svelte';
  import { api } from '../lib/api.js';
  import { remote } from '../lib/load.svelte.js';
  import { session, isAdmin } from '../lib/session.svelte.js';
  import { seg, itemHref } from '../lib/router.svelte.js';
  import { fmtNumber, fmtRelative } from '../lib/format.js';

  const overview = remote({ namespaces: [], settings: null, events: [] });
  $effect(() => {
    overview.load(async () => {
      const [n, s, a] = await Promise.all([api.get('/api/namespaces'), api.get('/api/settings'), api.get('/api/audit?limit=8')]);
      return { namespaces: n.namespaces, settings: s, events: a.events };
    });
  });

  const namespaces = $derived(overview.data.namespaces);
  const settings = $derived(overview.data.settings);
  const events = $derived(overview.data.events);
  const totals = $derived(namespaces.reduce(
    (t, n) => ({ documents: t.documents + n.documents, knowledge: t.knowledge + n.knowledge, notes: t.notes + n.notes }),
    { documents: 0, knowledge: 0, notes: 0 },
  ));
  const indexOk = $derived(settings && settings.dim_database != null
    ? settings.dim_configured === settings.dim_database && (!settings.model_database || settings.model_database === settings.model)
    : null);
  const https = $derived(session.https || settings?.tls || { state: 'off' });
  const httpsTone = $derived({ active: 'good', external: 'good', pending: 'warn', unconfigured: 'neutral', off: 'neutral' }[https.state] || 'neutral');
  const httpsLabel = $derived({ active: 'HTTPS active', external: 'HTTPS (own certificate)', pending: 'HTTPS pending confirmation',
    unconfigured: 'HTTPS not set up', off: 'TLS handled externally' }[https.state] || https.state);
</script>

<PageHeader title="Dashboard" subtitle={`Welcome back, ${session.user?.display_name || session.user?.username}.`} />
<ErrorBox error={overview.error} />

<div class="grid mb">
  <StatTile label="Documents" value={fmtNumber(totals.documents)} href="/browse" />
  <StatTile label="Knowledge" value={fmtNumber(totals.knowledge)} href="/browse" />
  <StatTile label="Notes" value={fmtNumber(totals.notes)} href="/browse" />
  <StatTile label="Namespaces" value={fmtNumber(namespaces.length)} href="/browse" />
</div>

<div class="grid two">
  <Card title="Server">
    <div class="stack">
      <div class="row between">
        <span class="dim">Embeddings</span>
        <StatusBadge tone={settings ? (settings.model ? 'good' : 'warn') : 'neutral'} label={settings?.mode || '…'} />
      </div>
      <div class="row between">
        <span class="dim">Index</span>
        {#if indexOk === null}<StatusBadge tone="neutral" label="No identity recorded" />
        {:else if indexOk}<StatusBadge tone="good" label={`Matches ${settings.model || 'configured model'} · ${settings.dim_database} dims`} />
        {:else}<StatusBadge tone="bad" label="Does not match the configured embedder" />{/if}
      </div>
      <div class="row between">
        <span class="dim">Transport</span>
        {#if isAdmin()}<a href="/admin/https"><StatusBadge tone={httpsTone} label={httpsLabel} /></a>
        {:else}<StatusBadge tone={httpsTone} label={httpsLabel} />{/if}
      </div>
      <div class="row between"><span class="dim">Version</span><span class="mono">{settings?.version || '…'}</span></div>
    </div>
    <p class="help mt">Agents connect with a personal token — see <a href="/connect">Connect an agent</a>.</p>
  </Card>

  <Card title="Recent activity" subtitle="The audit trail, newest first" flush>
    {#if events.length === 0}
      <div style="padding:16px"><Empty text="No activity recorded yet." /></div>
    {:else}
      <div class="table-wrap"><table class="table">
        <thead><tr><th>When</th><th>Who</th><th>What</th><th>Item</th></tr></thead>
        <tbody>
          {#each events as e (e.id)}
            <tr>
              <td class="nowrap muted">{fmtRelative(e.ts)}</td>
              <td>{e.actor || '—'}{#if e.detail?.token?.name}<span class="muted small"> · {e.detail.token.name}</span>{/if}</td>
              <td><code>{e.op}</code></td>
              <td class="truncate" style="max-width:260px">
                {#if e.namespace && e.item_type && e.item_id && ['document','knowledge','note'].includes(e.item_type)}
                  <a href={itemHref(e.namespace, e.item_type, e.item_id)}>{e.title || e.item_id}</a>
                {:else}{e.title || e.item_id || e.namespace || ''}{/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table></div>
      <div style="padding:10px 18px"><a href="/audit" class="small">All activity →</a></div>
    {/if}
  </Card>
</div>

{#if namespaces.length}
  <TableCard title="Namespaces">
    <thead><tr><th>Namespace</th><th class="num">Documents</th><th class="num">Knowledge</th><th class="num">Notes</th></tr></thead>
    <tbody>
      {#each namespaces as n (n.name)}
        <tr>
          <td><a href={`/browse/${seg(n.name)}`}>{n.name}</a></td>
          <td class="num">{fmtNumber(n.documents)}</td><td class="num">{fmtNumber(n.knowledge)}</td><td class="num">{fmtNumber(n.notes)}</td>
        </tr>
      {/each}
    </tbody>
  </TableCard>
{/if}

<style>
  :global(.grid.two) { margin-bottom: 14px; }
</style>
