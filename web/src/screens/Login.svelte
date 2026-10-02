<script>
  import AuthCard from '../components/AuthCard.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import { session, login } from '../lib/session.svelte.js';
  import { action } from '../lib/load.svelte.js';

  let username = $state('');
  let password = $state('');
  const op = action();

  function submit(e) {
    e.preventDefault();
    op.run(async () => {
      try {
        await login(username.trim(), password);
      } catch (err) {
        password = '';
        throw err.status === 429 ? new Error(`Too many attempts. ${err.message}`) : err;
      }
    });
  }
</script>

<AuthCard title="Sign in" subtitle="Shared memory for your agents.">
  <form class="stack" onsubmit={submit}>
    {#if session.error}<div class="alert warn">{session.error}</div>{/if}
    <ErrorBox error={op.error} />
    <div class="field">
      <label for="u">Username</label>
      <input id="u" class="input" bind:value={username} autocomplete="username" autocapitalize="none" required />
    </div>
    <div class="field">
      <label for="p">Password</label>
      <input id="p" class="input" type="password" bind:value={password} autocomplete="current-password" required />
    </div>
    <button class="btn primary" type="submit" disabled={op.busy}>{op.busy ? 'Signing in…' : 'Sign in'}</button>
    <p class="help">Forgot the password? An administrator can reset it. If nobody can sign in, see the recovery command in the documentation.</p>
  </form>
</AuthCard>
