<script>
  import AuthCard from '../components/AuthCard.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import { api } from '../lib/api.js';
  import { session } from '../lib/session.svelte.js';
  import { action } from '../lib/load.svelte.js';

  let code = $state('');
  let username = $state('admin');
  let displayName = $state('');
  let password = $state('');
  let confirm = $state('');
  const op = action();

  function submit(e) {
    e.preventDefault();
    op.run(async () => {
      if (password !== confirm) throw new Error('The two passwords differ.');
      const r = await api.post('/api/first-run', {
        setup_code: code, username: username.trim(), display_name: displayName.trim(), password,
      });
      session.user = r.user;
      session.firstRun = false;
    });
  }
</script>

<AuthCard title="Create the first administrator" subtitle="No users exist yet. The setup code is in the server's log.">
  <form class="stack" onsubmit={submit}>
    <ErrorBox error={op.error} />
    <div class="field">
      <label for="c">Setup code</label>
      <input id="c" class="input mono" bind:value={code} placeholder="XXXX-XXXX-XXXX" autocomplete="off" spellcheck="false" required />
      <div class="help">Printed when the server started, e.g. <code>docker logs mnemomatic</code>.</div>
    </div>
    <div class="field">
      <label for="u">Username</label>
      <input id="u" class="input" bind:value={username} autocomplete="username" autocapitalize="none" required />
    </div>
    <div class="field">
      <label for="d">Display name <span class="muted">(optional)</span></label>
      <input id="d" class="input" bind:value={displayName} autocomplete="name" />
    </div>
    <div class="field">
      <label for="p">Password</label>
      <input id="p" class="input" type="password" bind:value={password} autocomplete="new-password" minlength="10" required />
      <div class="help">At least 10 characters.</div>
    </div>
    <div class="field">
      <label for="p2">Confirm password</label>
      <input id="p2" class="input" type="password" bind:value={confirm} autocomplete="new-password" required />
    </div>
    <button class="btn primary" type="submit" disabled={op.busy}>{op.busy ? 'Creating…' : 'Create administrator'}</button>
  </form>
</AuthCard>
