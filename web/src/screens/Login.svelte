<script>
  import AuthCard from '../components/AuthCard.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import { session, login } from '../lib/session.svelte.js';

  let username = $state('');
  let password = $state('');
  let busy = $state(false);
  let error = $state(null);

  async function submit(e) {
    e.preventDefault();
    busy = true;
    error = null;
    try {
      await login(username.trim(), password);
    } catch (err) {
      error = err.status === 429 ? `Too many attempts. ${err.message}` : err.message;
      password = '';
    } finally {
      busy = false;
    }
  }
</script>

<AuthCard title="Sign in" subtitle="Shared memory for your agents.">
  <form class="stack" onsubmit={submit}>
    {#if session.error}<div class="alert warn">{session.error}</div>{/if}
    <ErrorBox {error} />
    <div class="field">
      <label for="u">Username</label>
      <input id="u" class="input" bind:value={username} autocomplete="username" autocapitalize="none" required />
    </div>
    <div class="field">
      <label for="p">Password</label>
      <input id="p" class="input" type="password" bind:value={password} autocomplete="current-password" required />
    </div>
    <button class="btn primary" type="submit" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
    <p class="help">Forgot the password? An administrator can reset it. If nobody can sign in, see the recovery command in the documentation.</p>
  </form>
</AuthCard>
