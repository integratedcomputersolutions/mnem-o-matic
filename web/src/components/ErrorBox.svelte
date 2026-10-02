<script>
  import { errorText } from '../lib/format.js';
  let { error = null } = $props();
</script>

{#if error}
  <div class="alert bad" role="alert">{errorText(error)}</div>
{/if}
