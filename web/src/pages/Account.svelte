<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import ChangePassword from '../screens/ChangePassword.svelte';
  import { session, viaProxy } from '../lib/session.svelte.js';
  import { fmtDate } from '../lib/format.js';
  const u = $derived(session.user);
</script>

<PageHeader title="Account" />
<div class="grid two">
  <Card title="You">
    <dl class="kv">
      <dt>Username</dt><dd>{u.username}</dd>
      <dt>Display name</dt><dd>{u.display_name || '—'}</dd>
      <dt>Role</dt><dd>{u.role}</dd>
      <dt>Member since</dt><dd>{fmtDate(u.created_at)}</dd>
    </dl>
    <p class="help mt">Display names and roles are set by an administrator.</p>
  </Card>
  {#if viaProxy()}
    <Card title="Sign-in" subtitle="Your portal signs you in here.">
      <p class="help">This account has no password of its own: the portal vouches for you on every request, and signing out happens there. Your identity for the audit trail is <b>{u.external_id || u.username}</b>.</p>
    </Card>
  {:else}
    <Card title="Change password" subtitle="Other browser sessions are signed out; your tokens keep working.">
      <ChangePassword />
    </Card>
  {/if}
</div>
