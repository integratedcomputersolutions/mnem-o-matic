<script>
  let { params = {} } = $props();
  import PageHeader from '../../components/PageHeader.svelte';
  import TableCard from '../../components/TableCard.svelte';
  import Modal from '../../components/Modal.svelte';
  import ConfirmModal from '../../components/ConfirmModal.svelte';
  import OneTimeSecret from '../../components/OneTimeSecret.svelte';
  import ErrorBox from '../../components/ErrorBox.svelte';
  import StatusBadge from '../../components/StatusBadge.svelte';
  import { api } from '../../lib/api.js';
  import { remote, action } from '../../lib/load.svelte.js';
  import { session } from '../../lib/session.svelte.js';
  import { fmtDate } from '../../lib/format.js';

  const users = remote([]);
  const op = action();
  let createOpen = $state(false);
  let form = $state({ username: '', display_name: '', role: 'user' });
  let issued = $state(null);          // {username, password, expires_at} shown once
  let confirmDelete = $state(null);

  const load = () => users.load(async () => (await api.get('/api/admin/users')).users);
  $effect(() => { load(); });

  const me = $derived(session.user?.id);

  const act = (fn) => op.run(async () => { await fn(); await load(); });

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
<ErrorBox error={op.error || users.error} />

{#if issued}
  <OneTimeSecret value={issued.password} ondone={() => (issued = null)}>
    <b>Temporary password for {issued.username}.</b> Hand it over out of band; it must be changed at first sign-in and expires {fmtDate(issued.expires_at)}.
  </OneTimeSecret>
{/if}

<TableCard>
  <thead><tr><th>User</th><th>Role</th><th>Status</th><th class="num">Tokens</th><th>Created</th><th></th></tr></thead>
  <tbody>
    {#each users.data as u (u.id)}
      <tr>
        <td><b>{u.username}</b>{#if u.display_name}<div class="muted small">{u.display_name}</div>{/if}{#if u.external_id && u.external_id !== u.display_name}<div class="muted small">via proxy: {u.external_id}</div>{/if}</td>
        <td>
          {#if u.id === me}<span class="badge brand">admin · you</span>
          {:else}
            <select class="select sm" value={u.role} disabled={op.busy} onchange={(e) => setRole(u, e.currentTarget.value)} aria-label="Role">
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
            <button class="btn sm" disabled={op.busy} onclick={() => reset(u)}>Reset password</button>
            {#if u.active}<button class="btn sm" disabled={op.busy} title="Signs them out and revokes all their tokens" onclick={() => setActive(u, false)}>Disable</button>
            {:else}<button class="btn sm" disabled={op.busy} onclick={() => setActive(u, true)}>Enable</button>{/if}
            <button class="btn danger sm" disabled={op.busy} onclick={() => (confirmDelete = u)}>Delete</button>
          {:else}<span class="muted small">Change your own password on the Account page.</span>{/if}
        </td>
      </tr>
    {/each}
  </tbody>
</TableCard>
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
    <button class="btn primary" type="submit" form="create-user" disabled={op.busy}>Create</button>
  {/snippet}
</Modal>

<ConfirmModal open={!!confirmDelete} title="Delete user?" confirm="Delete"
  onconfirm={() => remove(confirmDelete)} onclose={() => (confirmDelete = null)}>
  <p>Delete <b>{confirmDelete?.username}</b>, their sessions and all their tokens. Audit history keeps the username.</p>
</ConfirmModal>

<style>
  .select.sm { width: auto; padding: 4px 8px; font-size: 13px; }
</style>
