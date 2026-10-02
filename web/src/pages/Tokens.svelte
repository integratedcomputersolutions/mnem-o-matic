<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import Modal from '../components/Modal.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import CopyField from '../components/CopyField.svelte';
  import StatusBadge from '../components/StatusBadge.svelte';
  import { api } from '../lib/api.js';
  import { session } from '../lib/session.svelte.js';
  import { navigate } from '../lib/router.svelte.js';
  import { fmtDate, fmtRelative } from '../lib/format.js';

  let tokens = $state([]);
  let error = $state(null);
  let createOpen = $state(false);
  let name = $state('');
  let expires = $state('0');
  let busy = $state(false);
  let created = $state(null);      // {name, token} shown once
  let confirmRevoke = $state(null);

  async function load() {
    try {
      tokens = (await api.get('/api/me/tokens')).tokens;
    } catch (e) {
      error = e;
    }
  }
  $effect(() => { load(); });

  function status(t) {
    if (t.revoked_at) return ['neutral', 'Revoked'];
    if (t.expires_at && new Date(t.expires_at) < new Date()) return ['warn', 'Expired'];
    return ['good', 'Active'];
  }

  async function create(e) {
    e.preventDefault();
    busy = true;
    error = null;
    try {
      const r = await api.post('/api/me/tokens', { name: name.trim(), expires_in_days: Number(expires) });
      created = r;
      session.freshToken = r.token;       // memory only, for the Connect page
      createOpen = false;
      name = '';
      await load();
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }

  async function revoke(t) {
    try {
      await api.del(`/api/me/tokens/${t.id}`);
      if (session.freshToken?.startsWith(t.hint)) session.freshToken = null;
      confirmRevoke = null;
      await load();
    } catch (e) {
      error = e;
    }
  }
</script>

<PageHeader title="My tokens" subtitle="Each agent gets its own token. Revoke one and only that agent stops.">
  {#snippet actions()}<button class="btn primary" onclick={() => (createOpen = true)}>New token</button>{/snippet}
</PageHeader>
<ErrorBox {error} />

{#if created}
  <div class="alert good mb">
    <b>Token “{created.name}” created.</b> Copy it now — it is not shown again.
    <div class="mt"><CopyField value={created.token} secret /></div>
    <div class="row mt">
      <button class="btn primary sm" onclick={() => navigate('/connect')}>Use it on the Connect page</button>
      <button class="btn ghost sm" onclick={() => (created = null)}>Done</button>
    </div>
  </div>
{/if}

{#if tokens.length === 0}
  <Empty text="No tokens yet. Create one, then paste it into your agent's configuration." />
{:else}
  <Card flush>
    <div class="table-wrap"><table class="table">
      <thead><tr><th>Name</th><th>Hint</th><th>Status</th><th>Created</th><th>Expires</th><th>Last used</th><th></th></tr></thead>
      <tbody>
        {#each tokens as t (t.id)}
          {@const [tone, label] = status(t)}
          <tr>
            <td><b>{t.name}</b></td>
            <td class="mono muted">{t.hint}…</td>
            <td><StatusBadge {tone} {label} /></td>
            <td class="muted small nowrap">{fmtDate(t.created_at)}</td>
            <td class="muted small nowrap">{t.expires_at ? fmtDate(t.expires_at) : 'never'}</td>
            <td class="muted small nowrap">{t.last_used_at ? fmtRelative(t.last_used_at) : 'never'}</td>
            <td class="right">{#if !t.revoked_at}<button class="btn danger sm" onclick={() => (confirmRevoke = t)}>Revoke</button>{/if}</td>
          </tr>
        {/each}
      </tbody>
    </table></div>
  </Card>
{/if}

<Modal bind:open={createOpen} title="New API token">
  <form class="stack" onsubmit={create} id="create-token">
    <div class="field">
      <label for="tn">Name</label>
      <input id="tn" class="input" bind:value={name} placeholder="e.g. laptop · Claude Code" maxlength="64" required />
      <div class="help">Only for your own reference — which agent or machine holds it.</div>
    </div>
    <div class="field">
      <label for="te">Expires</label>
      <select id="te" class="select" bind:value={expires}>
        <option value="0">Never</option><option value="30">In 30 days</option>
        <option value="90">In 90 days</option><option value="365">In a year</option>
      </select>
    </div>
  </form>
  {#snippet footer()}
    <button class="btn ghost" type="button" onclick={() => (createOpen = false)}>Cancel</button>
    <button class="btn primary" type="submit" form="create-token" disabled={busy || !name.trim()}>{busy ? 'Creating…' : 'Create'}</button>
  {/snippet}
</Modal>

<Modal open={!!confirmRevoke} title="Revoke token?" onclose={() => (confirmRevoke = null)}>
  <p>Agents using <b>{confirmRevoke?.name}</b> ({confirmRevoke?.hint}…) will be refused from now on. This cannot be undone; create a new token instead.</p>
  {#snippet footer()}
    <button class="btn ghost" onclick={() => (confirmRevoke = null)}>Keep it</button>
    <button class="btn danger" onclick={() => revoke(confirmRevoke)}>Revoke</button>
  {/snippet}
</Modal>
