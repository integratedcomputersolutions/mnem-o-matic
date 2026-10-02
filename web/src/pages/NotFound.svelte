<script>
  let { params = {} } = $props();
  import PageHeader from '../components/PageHeader.svelte';
  import { route } from '../lib/router.svelte.js';
</script>

<PageHeader title="Not found" subtitle={`Nothing lives at ${route.path}.`} />
<p><a href="/">Back to the dashboard</a></p>
