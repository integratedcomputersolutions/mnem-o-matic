<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import Tabs from '../components/Tabs.svelte';
  import CopyField from '../components/CopyField.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import Stepper from '../components/Stepper.svelte';
  import { api } from '../lib/api.js';
  import { remote } from '../lib/load.svelte.js';
  import { session, isAdmin } from '../lib/session.svelte.js';
  import { snippets, connectUrl } from '../lib/snippets.js';

  const info = remote();
  $effect(() => { info.load(() => api.get('/api/connect')); });
  let active = $state('claude-code');

  const list = $derived(info.data ? snippets(info.data, session.freshToken) : []);
  const current = $derived(list.find((s) => s.id === active) || list[0]);
  const tabs = $derived(list.map((s) => [s.id, s.label]));
</script>

<PageHeader title="Connect an agent" subtitle="Give a client this server's address and your token. Nothing else is needed." />
<ErrorBox error={info.error} />

{#if info.data}
  {@const i = info.data}
  {#if session.freshToken}
    <div class="alert good mb">The token you just created is filled into the examples below. It is held in this tab only; reload and it is gone.</div>
  {:else}
    <div class="alert mb">The examples use the placeholder <code>mnm_your_token_here</code>. <a href="/tokens">Create a token</a> and it is filled in for you.</div>
  {/if}

  {#if i.https?.state === 'pending'}
    <div class="alert warn mb">HTTPS is set up but not yet confirmed{#if isAdmin()} — finish it under <a href="/admin/https">Admin → HTTPS</a>{/if}. Until then the URLs below use plain HTTP.</div>
  {/if}

  <div class="grid two mb">
    <Card title="Server address">
      <CopyField label="MCP endpoint" value={connectUrl(i)} />
      <div class="mt"><CopyField label="Trimmed tool list for small-context models" value={connectUrl(i).replace('/mcp', '/mcp?compact=true')} /></div>
    </Card>
    <Card title="Authentication">
      <p class="dim small">Every request carries <code>Authorization: Bearer &lt;token&gt;</code>. The server records what each token does under your name; revoke a token on <a href="/tokens">My tokens</a> and only that agent stops.</p>
      <p class="dim small">Examples that put the token on a command line (<code>claude mcp add</code>, <code>export</code>, <code>curl</code>) leave it in your shell history. Clear that entry afterwards, or use the config-file form where the client has one.</p>
      {#if session.freshToken}<CopyField label="Your new token" value={session.freshToken} secret />{/if}
    </Card>
  </div>

  {#if i.builtin_ca}
    <Card title="Trust this server's certificate first" subtitle="One time per device. The CA is valid for this hostname only.">
      <Stepper steps={['Download the CA', 'Check the fingerprint', 'Trust it', 'Point clients at HTTPS']} current={0} />
      <div class="grid two">
        <div class="stack">
          <a class="btn primary" href={i.ca_url} download="mnemomatic-ca.crt">Download mnemomatic-ca.crt</a>
          <CopyField label="SHA-256 fingerprint" value={i.ca_fingerprint} multiline />
        </div>
        <div class="dim small">
          <p>Per-OS and per-browser steps are on the <a href="/setup" target="_blank" rel="noopener">setup page</a>. Tools that keep their own trust store need an environment variable:</p>
          <CopyField multiline value={`export NODE_EXTRA_CA_CERTS=$HOME/mnemomatic-ca.crt   # Claude Code, Claude Desktop, Cursor\nexport SSL_CERT_FILE=$HOME/mnemomatic-ca.crt         # Python, mnemomatic-cli`} />
        </div>
      </div>
    </Card>
    <div class="mb"></div>
  {/if}

  <Card>
    <Tabs {tabs} bind:active>
      {#if current}
        <p class="dim">{current.intro}</p>
        <div class="stack">
          {#each current.blocks as b (b.title)}
            <CopyField label={b.title} value={b.code} multiline />
          {/each}
        </div>
        {#if current.notes?.length}
          <ul class="notes">{#each current.notes as n}<li>{n}</li>{/each}</ul>
        {/if}
      {/if}
    </Tabs>
  </Card>
{/if}

<style>
  .notes { margin: 14px 0 0; padding-left: 18px; color: var(--ink-2); font-size: 14px; }
  .notes li { margin: 4px 0; }
</style>
