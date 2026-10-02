<script>
  let { params = {} } = $props();
  import PageHeader from '../../components/PageHeader.svelte';
  import Card from '../../components/Card.svelte';
  import Modal from '../../components/Modal.svelte';
  import ErrorBox from '../../components/ErrorBox.svelte';
  import CopyField from '../../components/CopyField.svelte';
  import StatusBadge from '../../components/StatusBadge.svelte';
  import { api } from '../../lib/api.js';
  import { session } from '../../lib/session.svelte.js';
  import { fmtDate } from '../../lib/format.js';

  let users = $state([]);
  let error = $state(null);
  let createOpen = $state(false);
  let form = $state({ username: '', display_name: '', role: 'user' });
  let busy = $state(false);
  let issued = $state(null);          // {username, password, expires_at} shown once
  let confirmDelete = $state(null);

  async function load() {
    try {
      users = (await api.get('/api/admin/users')).users;
    } catch (e) {
      error = e;
    }
  }
  $effect(() => { load(); });

  const me = $derived(session.user?.id);

  async function act(fn) {
    error = null;
    busy = true;
    try {
      await fn();
      await load();
    } catch (e) {
      error = e;
    } finally {
      busy = false;
    }
  }

  function create(e) {
    e.preventDefault();
    act(async () => {
      const r = await api.post('/api/admin/users', { ...form, username: form.username.trim(), display_name: form.display_name.trim() });
      issued = { username: r.user.username, password: r.temporary_password, expires_at: r.expires_at };
      createOpen = false;
      form = { username: '', display_name: '', role: 'user' };
    });
  }
  const reset = (u) => act(async () => {
    const r = await api.post(`/api/admin/users/${u.id}/reset-password`);
    issued = { username: u.username, password: r.temporary_password, expires_at: r.expires_at };
  });
  const setActive = (u, active) => act(() => api.post(`/api/admin/users/${u.id}/active`, { active }));
  const setRole = (u, role) => act(() => api.post(`/api/admin/users/${u.id}/role`, { role }));
  const remove = (u) => act(async () => { await api.del(`/api/admin/users/${u.id}`); confirmDelete = null; });
</script>

<PageHeader title="Users" subtitle="People who can sign in here and mint tokens for their agents.">
  {#snippet actions()}<button class="btn primary" onclick={() => (createOpen = true)}>Add user</button>{/snippet}
</PageHeader>
<ErrorBox {error} />

{#if issued}
  <div class="alert good mb">
    <b>Temporary password for {issued.username}.</b> Hand it over out of band; it must be changed at first sign-in and expires {fmtDate(issued.expires_at)}.
    <div class="mt"><CopyField value={issued.password} secret /></div>
    <div class="mt"><button class="btn ghost sm" onclick={() => (issued = null)}>Done</button></div>
  </div>
{/if}

<Card flush>
  <div class="table-wrap"><table class="table">
    <thead><tr><th>User</th><th>Role</th><th>Status</th><th class="num">Tokens</th><th>Created</th><th></th></tr></thead>
    <tbody>
      {#each users as u (u.id)}
        <tr>
          <td><b>{u.username}</b>{#if u.display_name}<div class="muted small">{u.display_name}</div>{/if}</td>
          <td>
            {#if u.id === me}<span class="badge brand">admin · you</span>
            {:else}
              <select class="select sm" value={u.role} disabled={busy} onchange={(e) => setRole(u, e.currentTarget.value)} aria-label="Role">
                <option value="user">user</option><option value="admin">admin</option>
              </select>
            {/if}
          </td>
          <td>
            {#if !u.active}<StatusBadge tone="bad" label="Disabled" />
            {:else if u.must_change_password}<StatusBadge tone="warn" label="Temporary password" />
            {:else}<StatusBadge tone="good" label="Active" />{/if}
          </td>
          <td class="num">{u.token_count}</td>
          <td class="muted small nowrap">{fmtDate(u.created_at)}</td>
          <td class="right nowrap">
            {#if u.id !== me}
              <button class="btn sm" disabled={busy} onclick={() => reset(u)}>Reset password</button>
              {#if u.active}<button class="btn sm" disabled={busy} onclick={() => setActive(u, false)}>Disable</button>
              {:else}<button class="btn sm" disabled={busy} onclick={() => setActive(u, true)}>Enable</button>{/if}
              <button class="btn danger sm" disabled={busy} onclick={() => (confirmDelete = u)}>Delete</button>
            {:else}<span class="muted small">Change your own password on the Account page.</span>{/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table></div>
</Card>
<p class="help mt">Disabling a user ends their sessions and refuses their tokens immediately; enabling them brings the tokens back. The last active administrator cannot be removed or demoted.</p>

<Modal bind:open={createOpen} title="Add a user">
  <form class="stack" onsubmit={create} id="create-user">
    <div class="field">
      <label for="un">Username</label>
      <input id="un" class="input" bind:value={form.username} pattern="[a-zA-Z0-9][a-zA-Z0-9._-]{'{'}1,31{'}'}" autocapitalize="none" required />
      <div class="help">Lowercase letters, digits, dots, underscores, dashes.</div>
    </div>
    <div class="field">
      <label for="dn">Display name</label>
      <input id="dn" class="input" bind:value={form.display_name} />
    </div>
    <div class="field">
      <label for="rl">Role</label>
      <select id="rl" class="select" bind:value={form.role}><option value="user">user</option><option value="admin">admin</option></select>
    </div>
    <p class="help">A temporary password is generated; the user picks their own at first sign-in.</p>
  </form>
  {#snippet footer()}
    <button class="btn ghost" type="button" onclick={() => (createOpen = false)}>Cancel</button>
    <button class="btn primary" type="submit" form="create-user" disabled={busy}>Create</button>
  {/snippet}
</Modal>

<Modal open={!!confirmDelete} title="Delete user?" onclose={() => (confirmDelete = null)}>
  <p>Delete <b>{confirmDelete?.username}</b>, their sessions and all their tokens. Audit history keeps the username.</p>
  {#snippet footer()}
    <button class="btn ghost" onclick={() => (confirmDelete = null)}>Cancel</button>
    <button class="btn danger" onclick={() => remove(confirmDelete)}>Delete</button>
  {/snippet}
</Modal>

<style>
  .select.sm { width: auto; padding: 4px 8px; font-size: 13px; }
</style>
