<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import { route } from '../lib/router.svelte.js';
  import { viaProxy } from '../lib/session.svelte.js';
</script>

<PageHeader title="Not found" subtitle={`Nothing lives at ${route.path}.`} />
{#if viaProxy()}<p class="help">Pages about personal tokens and passwords are not shown when your portal signs you in.</p>{/if}
<p><a href="/">Back to the dashboard</a></p>
