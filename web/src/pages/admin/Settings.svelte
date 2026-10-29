<script>
  let { params = {} } = $props();
  import PageHeader from '../../components/PageHeader.svelte';
  import Card from '../../components/Card.svelte';
  import ErrorBox from '../../components/ErrorBox.svelte';
  import StatusBadge from '../../components/StatusBadge.svelte';
  import { api } from '../../lib/api.js';
  import { remote, action } from '../../lib/load.svelte.js';
  import { fmtVersion } from '../../lib/format.js';

  const settings = remote();
  $effect(() => { settings.load(() => api.get('/api/settings')); });
  const s = $derived(settings.data);
  const dimOk = $derived(s && (s.dim_database == null || s.dim_database === s.dim_configured));
  const modelOk = $derived(s && (!s.model_database || s.model_database === s.model));

  // The re-index: its status is polled while a run is on, and the embeddings card reloads at
  // the end (the index's model and dimension change with a successful run).
  const reindex = remote();
  const start = action();
  const r = $derived(reindex.data);
  let confirming = $state(false);
  $effect(() => { reindex.load(() => api.get('/api/admin/reindex')); });
  $effect(() => {
    if (!r?.running) return;
    const t = setInterval(async () => {
      const next = await api.get('/api/admin/reindex');
      reindex.data = next;
      if (!next.running) settings.load(() => api.get('/api/settings'));
    }, 2000);
    return () => clearInterval(t);
  });
  const runReindex = () => start.run(async () => {
    confirming = false;
    reindex.data = await api.post('/api/admin/reindex');
  });
  const total = $derived(r?.counts ? r.counts.documents + r.counts.knowledge + r.counts.notes : 0);
</script>

<PageHeader title="Settings" subtitle="What this server is running with. Changed through the environment, shown here.">
  {#snippet actions()}<a class="btn" href="/export" download>Download export (.zip)</a>{/snippet}
</PageHeader>
<ErrorBox error={settings.error} />

{#if s}
  <div class="grid two">
    <Card title="Embeddings">
      <dl class="kv">
        <dt>Mode</dt><dd>{s.mode}</dd>
        <dt>Model</dt><dd>{#if s.model_url}<a href={s.model_url} target="_blank" rel="noopener">{s.model}</a>{:else}{s.model || '—'}{/if}</dd>
        <dt>Dimensions</dt>
        <dd>{s.dim_configured} configured · {s.dim_database ?? '—'} in the index
          {#if !dimOk}<div><StatusBadge tone="bad" label="Mismatch: a reindex is needed" /></div>{/if}</dd>
        <dt>Index built by</dt>
        <dd>{s.model_database || 'not recorded'}
          {#if !modelOk}<div><StatusBadge tone="bad" label="Different model than configured" /></div>{/if}</dd>
        {#if s.endpoint_url}<dt>Endpoint</dt><dd><code>{s.endpoint_url}</code> ({s.wire_api})</dd>{/if}
        {#if s.max_tokens}<dt>Max tokens</dt><dd>{s.max_tokens}</dd>{/if}
        <dt>Query prefix</dt><dd><code>{s.query_prefix || '∅'}</code></dd>
        <dt>Document prefix</dt><dd><code>{s.doc_prefix || '∅'}</code></dd>
      </dl>
    </Card>

    <Card title="Semantic index">
      <ErrorBox error={start.error || reindex.error} />
      {#if r}
        <dl class="kv">
          <dt>State</dt>
          <dd>
            {#if r.running}
              <StatusBadge tone="warn" label="Re-embedding…" />
              <div class="small dim mt">{r.done} of {r.total} items</div>
            {:else if !dimOk || !modelOk}
              <StatusBadge tone="bad" label="Index and embedder disagree" />
            {:else}
              <StatusBadge tone="good" label="Up to date" />
            {/if}
          </dd>
          {#if r.finished_at}
            <dt>Last run</dt>
            <dd>
              {new Date(r.finished_at).toLocaleString()}{#if r.actor} by {r.actor}{:else if r.trigger === 'startup'} at startup{/if}
              {#if r.error}
                <div><StatusBadge tone="bad" label="Stopped" /> <span class="small dim">{r.error}</span></div>
              {:else if r.counts}
                <div class="small dim">{total} items re-embedded{#if r.counts.failed}, {r.counts.failed} failed (fulltext only){/if}</div>
              {/if}
            </dd>
          {/if}
        </dl>
        <p class="help mt">Re-embedding rebuilds the vector index with the current embedder and embeds every document, knowledge entry and note again. Content, timestamps, revisions and the audit log are untouched. Searches keep working meanwhile, fulltext only until it's done.</p>
        {#if confirming}
          <div class="row mt">
            <button class="btn danger" onclick={runReindex} disabled={start.busy}>Re-embed everything now</button>
            <button class="btn" onclick={() => (confirming = false)} disabled={start.busy}>Cancel</button>
          </div>
        {:else}
          <button class="btn mt" onclick={() => (confirming = true)} disabled={r.running || start.busy}>Re-embed everything…</button>
        {/if}
      {/if}
    </Card>

    <Card title="Chunking">
      <dl class="kv">
        <dt>Threshold</dt><dd>{s.chunk_threshold} chars</dd>
        <dt>Chunk size</dt><dd>{s.chunk_size} chars</dd>
        <dt>Overlap</dt><dd>{s.chunk_overlap} chars</dd>
      </dl>
      <p class="help mt">Documents longer than the threshold are embedded in overlapping chunks; search returns the matching chunk as an excerpt.</p>
    </Card>

    <Card title="Server">
      <dl class="kv">
        <dt>Version</dt><dd class="mono" title={s.build || undefined}>{fmtVersion(s.version, s.build)}</dd>
        <dt>TLS</dt><dd>{s.tls?.state}{#if s.tls?.name} · {s.tls.name}{/if} {#if s.tls?.state !== 'off'}<a href="/admin/https" class="small">manage</a>{/if}</dd>
        <dt>Trusted proxies</dt><dd>{s.trusted_proxies?.length ? s.trusted_proxies.join(', ') : 'none'}</dd>
        <dt>Audit retention</dt><dd>{s.audit_keep_days ? `${s.audit_keep_days} days` : 'forever'}</dd>
        <dt>Backups</dt><dd>{#if s.backup}every {s.backup.interval_hours} h to <code>{s.backup.dir}</code>, keeping {s.backup.keep}{:else}not scheduled{/if}</dd>
      </dl>
    </Card>

    <Card title="Recovery">
      <p class="dim small">If no administrator can sign in, run inside the container:</p>
      <pre>docker exec &lt;container&gt; /usr/bin/python3 -m mnemomatic.admin_cli reset-password &lt;username&gt;</pre>
      <p class="help mt">Prints a temporary password and re-enables the account. <code>create-admin &lt;username&gt;</code> makes a new one.</p>
    </Card>
  </div>
{/if}
