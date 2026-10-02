<script>
  let { params = {} } = $props();
  import PageHeader from '../../components/PageHeader.svelte';
  import Card from '../../components/Card.svelte';
  import Stepper from '../../components/Stepper.svelte';
  import ErrorBox from '../../components/ErrorBox.svelte';
  import CopyField from '../../components/CopyField.svelte';
  import StatusBadge from '../../components/StatusBadge.svelte';
  import { api } from '../../lib/api.js';
  import { session } from '../../lib/session.svelte.js';
  import { fmtDay } from '../../lib/format.js';

  let status = $state(null);
  let name = $state('');
  let busy = $state(false);
  let error = $state(null);
  let probe = $state(null);          // null | 'checking' | 'unreachable' | 'mismatch' | 'ok'

  async function load() {
    try {
      status = await api.get('/api/admin/https');
      // Pre-fill with the hostname the browser used, unless it is an IP literal (the CA refuses those).
      const h = window.location.hostname;
      if (!name) name = status.name || (/^[\d.]+$|:/.test(h) ? '' : h);
      session.https = status;
    } catch (e) {
      error = e;
    }
  }
  $effect(() => { load(); });

  const step = $derived(!status ? 0 : status.state === 'unconfigured' ? 0 : status.state === 'pending' ? 1 : 2);

  async function setName(e) {
    e.preventDefault();
    busy = true;
    error = null;
    probe = null;
    try {
      status = await api.post('/api/admin/https/name', { name: name.trim() });
      session.https = status;
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }

  async function confirm() {
    busy = true;
    error = null;
    probe = 'checking';
    let id = null;
    try {
      // Cross-origin on purpose: this proves the name resolves to this very
      // server. It only works once this browser trusts the CA.
      const r = await fetch(`${status.https_url}/api/instance-id`, { mode: 'cors', credentials: 'omit' });
      id = (await r.json()).instance_id;
    } catch {
      probe = 'unreachable';
      busy = false;
      return;
    }
    try {
      status = await api.post('/api/admin/https/confirm', { name: status.name, instance_id: id });
      session.https = status;
      probe = 'ok';
    } catch (err) {
      probe = err.code === 'instance_mismatch' ? 'mismatch' : null;
      error = err;
    } finally {
      busy = false;
    }
  }

  async function disable() {
    busy = true;
    error = null;
    try {
      status = await api.post('/api/admin/https/disable');
      session.https = status;
      probe = null;
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<PageHeader title="HTTPS" subtitle="A private certificate authority, bound to one hostname, confirmed from your browser." />
<ErrorBox {error} />

{#if status}
  {#if status.state === 'off'}
    <div class="alert">Built-in TLS is off (<code>MNEMOMATIC_TLS=off</code>). This deployment terminates TLS in its own reverse proxy; set <code>MNEMOMATIC_TRUSTED_PROXIES</code> so client addresses are right.</div>
  {:else if status.state === 'external'}
    <Card title="Your own certificate is in use">
      <p>Serving <b>{status.name}</b> from <code>custom.crt</code> / <code>custom.key</code>. No CA to distribute. Expires {fmtDay(status.leaf_not_after)}.</p>
      <CopyField label="HTTPS address" value={status.https_url} />
    </Card>
  {:else}
    <Stepper steps={['Name the host', 'Trust the CA and confirm', 'HTTPS enforced']} current={step} />

    <div class="grid two">
      <Card title="1 · Hostname">
        <form class="stack" onsubmit={setName}>
          <div class="field">
            <label for="hn">DNS name clients will use</label>
            <input id="hn" class="input mono" bind:value={name} placeholder="memory.example" required />
            <div class="help">A name, not an IP: the certificate is bound to it. Changing it later issues a new CA.</div>
          </div>
          <div class="row">
            <button class="btn primary" type="submit" disabled={busy || !name.trim()}>
              {status.state === 'unconfigured' ? 'Issue certificates' : 'Re-issue for this name'}
            </button>
            {#if status.state !== 'unconfigured'}
              <StatusBadge tone={status.state === 'active' ? 'good' : 'warn'} label={status.state === 'active' ? 'Active' : 'Pending confirmation'} />
            {/if}
          </div>
        </form>
      </Card>

      {#if status.state !== 'unconfigured'}
        <Card title="2 · Trust and confirm">
          <p class="dim small">HTTPS is already listening at <a href={status.https_url} target="_blank" rel="noopener">{status.https_url}</a>. Before this browser can reach it you must trust the CA:</p>
          <div class="stack">
            <a class="btn" href="/ca.crt" download="mnemomatic-ca.crt">Download mnemomatic-ca.crt</a>
            <CopyField label="SHA-256 fingerprint" value={status.ca_fingerprint || ''} multiline />
            <p class="dim small">Install steps per OS and browser are on the <a href="/setup" target="_blank" rel="noopener">setup page</a>. Then:</p>
            {#if status.state === 'pending'}
              <button class="btn primary" onclick={confirm} disabled={busy}>{probe === 'checking' ? 'Checking…' : 'Check and confirm HTTPS'}</button>
              {#if probe === 'unreachable'}
                <div class="alert warn">This browser could not reach <code>{status.https_url}</code>. Either the CA is not trusted here yet, or the name does not resolve to this server from this machine.</div>
              {:else if probe === 'mismatch'}
                <div class="alert bad">That name reaches a <em>different</em> server. Check DNS before confirming.</div>
              {/if}
            {:else}
              <div class="alert good">Confirmed. Plain HTTP now only serves the setup page; everything else lives at <a href={status.https_url}>{status.https_url}</a>.</div>
              <button class="btn danger sm" onclick={disable} disabled={busy}>Turn HTTPS enforcement off</button>
            {/if}
          </div>
        </Card>
      {/if}
    </div>

    {#if status.state !== 'unconfigured'}
      <p class="help mt">Server certificate valid until {fmtDay(status.leaf_not_after)}; renewed automatically a month before. The CA lives in the data volume under <code>tls/</code>.</p>
    {/if}
  {/if}
{/if}
