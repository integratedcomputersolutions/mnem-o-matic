<script>
  import { typeLabel } from '../lib/format.js';
  let { type } = $props();
</script>

<span class="badge type-{type}">{typeLabel(type)}</span>

<style>
  .type-document { border-color: rgba(139, 111, 196, 0.6); }
  .type-knowledge { border-color: rgba(95, 201, 167, 0.5); }
  .type-note { border-color: rgba(224, 179, 90, 0.5); }
</style>
