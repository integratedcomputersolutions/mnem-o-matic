<script>
  let { params = {} } = $props();
  import PageHeader from '../../components/PageHeader.svelte';
  import Card from '../../components/Card.svelte';
  import ErrorBox from '../../components/ErrorBox.svelte';
  import StatusBadge from '../../components/StatusBadge.svelte';
  import { api } from '../../lib/api.js';
  import { remote } from '../../lib/load.svelte.js';

  const settings = remote();
  $effect(() => { settings.load(() => api.get('/api/settings')); });
  const s = $derived(settings.data);
  const dimOk = $derived(s && (s.dim_database == null || s.dim_database === s.dim_configured));
  const modelOk = $derived(s && (!s.model_database || s.model_database === s.model));
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
        <dt>Version</dt><dd class="mono">{s.version}</dd>
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
