<script>
  import AuthCard from '../components/AuthCard.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import { api } from '../lib/api.js';
  import { session, refresh, logout } from '../lib/session.svelte.js';

  // forced: the full-screen gate for temporary passwords. Otherwise an inline
  // form (the Account page).
  let { forced = false } = $props();

  let current = $state('');
  let next = $state('');
  let confirm = $state('');
  let busy = $state(false);
  let error = $state(null);
  let done = $state(false);

  async function submit(e) {
    e.preventDefault();
    error = null;
    done = false;
    if (next !== confirm) {
      error = 'The two new passwords differ.';
      return;
    }
    busy = true;
    try {
      await api.post('/api/password', { current_password: current, new_password: next });
      current = next = confirm = '';
      done = true;
      await refresh();
    } catch (err) {
      error = err.message;
    } finally {
      busy = false;
    }
  }
</script>

{#snippet form()}
  <form class="stack" onsubmit={submit}>
    <ErrorBox {error} />
    {#if done}<div class="alert good">Password changed.</div>{/if}
    <div class="field">
      <label for="cur">{forced ? 'Temporary password' : 'Current password'}</label>
      <input id="cur" class="input" type="password" bind:value={current} autocomplete="current-password" required />
    </div>
    <div class="field">
      <label for="new">New password</label>
      <input id="new" class="input" type="password" bind:value={next} autocomplete="new-password" minlength="10" required />
      <div class="help">At least 10 characters, different from the current one.</div>
    </div>
    <div class="field">
      <label for="conf">Confirm new password</label>
      <input id="conf" class="input" type="password" bind:value={confirm} autocomplete="new-password" required />
    </div>
    <div class="row">
      <button class="btn primary" type="submit" disabled={busy}>{busy ? 'Saving…' : 'Change password'}</button>
      {#if forced}<button class="btn ghost" type="button" onclick={logout}>Sign out</button>{/if}
    </div>
  </form>
{/snippet}

{#if forced}
  <AuthCard title="Choose a password" subtitle={`Hello ${session.user?.display_name || session.user?.username}. Your temporary password must be replaced before you continue.`}>
    {@render form()}
  </AuthCard>
{:else}
  {@render form()}
{/if}
