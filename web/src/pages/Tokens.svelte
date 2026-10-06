<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import TableCard from '../components/TableCard.svelte';
  import Modal from '../components/Modal.svelte';
  import ConfirmModal from '../components/ConfirmModal.svelte';
  import OneTimeSecret from '../components/OneTimeSecret.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import StatusBadge from '../components/StatusBadge.svelte';
  import { api } from '../lib/api.js';
  import { remote, action } from '../lib/load.svelte.js';
  import { session } from '../lib/session.svelte.js';
  import { navigate } from '../lib/router.svelte.js';
  import { fmtDate, fmtRelative } from '../lib/format.js';

  const tokens = remote([]);
  const op = action();
  let createOpen = $state(false);
  let name = $state('');
  let expires = $state('0');
  let scope = $state('');          // no default: the person picks read or write
  let created = $state(null);      // {name, token} shown once
  let confirmRevoke = $state(null);

  const load = () => tokens.load(async () => (await api.get('/api/me/tokens')).tokens);
  $effect(() => { load(); });

  function status(t) {
    if (t.revoked_at) return ['neutral', 'Revoked'];
    if (t.expires_at && new Date(t.expires_at) < new Date()) return ['warn', 'Expired'];
    return ['good', 'Active'];
  }

  function create(e) {
    e.preventDefault();
    op.run(async () => {
      const r = await api.post('/api/me/tokens', { name: name.trim(), scope, expires_in_days: Number(expires) });
      created = r;
      session.freshToken = r.token;       // memory only, for the Connect page
      createOpen = false;
      name = '';
      scope = '';
      await load();
    });
  }

  const revoke = (t) => op.run(async () => {
    await api.del(`/api/me/tokens/${t.id}`);
    if (session.freshToken?.startsWith(t.hint)) session.freshToken = null;
    confirmRevoke = null;
    await load();
  });
</script>

<PageHeader title="My tokens" subtitle="Each agent gets its own token. Revoke one and only that agent stops.">
  {#snippet actions()}<button class="btn primary" onclick={() => (createOpen = true)}>New token</button>{/snippet}
</PageHeader>
<ErrorBox error={op.error || tokens.error} />

{#if created}
  <OneTimeSecret value={created.token} ondone={() => (created = null)}>
    <b>Token “{created.name}” created.</b> Copy it now — it is not shown again.
    {#snippet actions()}<button class="btn primary sm" onclick={() => navigate('/connect')}>Use it on the Connect page</button>{/snippet}
  </OneTimeSecret>
{/if}

{#if tokens.data.length === 0}
  <Empty text="No tokens yet. Create one, then paste it into your agent's configuration." />
{:else}
  <TableCard>
    <thead><tr><th>Name</th><th>Hint</th><th>Access</th><th>Status</th><th>Created</th><th>Expires</th><th>Last used</th><th></th></tr></thead>
    <tbody>
      {#each tokens.data as t (t.id)}
        {@const [tone, label] = status(t)}
        <tr>
          <td><b>{t.name}</b></td>
          <td class="mono muted">{t.hint}…</td>
          <td>{#if t.scope === 'read'}<StatusBadge tone="neutral" label="Read only" />{:else}<StatusBadge tone="brand" label="Read & write" />{/if}</td>
          <td><StatusBadge {tone} {label} /></td>
          <td class="muted small nowrap">{fmtDate(t.created_at)}</td>
          <td class="muted small nowrap">{t.expires_at ? fmtDate(t.expires_at) : 'never'}</td>
          <td class="muted small nowrap">{t.last_used_at ? fmtRelative(t.last_used_at) : 'never'}</td>
          <td class="right">{#if !t.revoked_at}<button class="btn danger sm" onclick={() => (confirmRevoke = t)}>Revoke</button>{/if}</td>
        </tr>
      {/each}
    </tbody>
  </TableCard>
{/if}

<Modal bind:open={createOpen} title="New API token">
  <form class="stack" onsubmit={create} id="create-token">
    <div class="field">
      <label for="tn">Name</label>
      <input id="tn" class="input" bind:value={name} placeholder="e.g. laptop · Claude Code" maxlength="64" required />
      <div class="help">Only for your own reference — which agent or machine holds it.</div>
    </div>
    <div class="field">
      <label for="ts">Access</label>
      <select id="ts" class="select" bind:value={scope} required>
        <option value="" disabled>Choose…</option>
        <option value="read">Read only: search and read, never change anything</option>
        <option value="write">Read &amp; write: everything you can do</option>
      </select>
      <div class="help">Give agents that only recall memory a read-only token. If it leaks, nothing can be deleted or overwritten with it.</div>
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
    <button class="btn primary" type="submit" form="create-token" disabled={op.busy || !name.trim() || !scope}>{op.busy ? 'Creating…' : 'Create'}</button>
  {/snippet}
</Modal>

<ConfirmModal open={!!confirmRevoke} title="Revoke token?" cancel="Keep it" confirm="Revoke"
  onconfirm={() => revoke(confirmRevoke)} onclose={() => (confirmRevoke = null)}>
  <p>Agents using <b>{confirmRevoke?.name}</b> ({confirmRevoke?.hint}…) will be refused from now on. This cannot be undone; create a new token instead.</p>
</ConfirmModal>
