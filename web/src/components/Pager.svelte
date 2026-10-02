<script>
  import { fmtNumber } from '../lib/format.js';
  let { total = 0, limit = 50, offset = $bindable(0) } = $props();
  const from = $derived(total === 0 ? 0 : offset + 1);
  const to = $derived(Math.min(total, offset + limit));
</script>

<div class="pager">
  <span class="muted small">{fmtNumber(from)}–{fmtNumber(to)} of {fmtNumber(total)}</span>
  <span class="row">
    <button type="button" class="btn sm" disabled={offset === 0} onclick={() => (offset = Math.max(0, offset - limit))}>Previous</button>
    <button type="button" class="btn sm" disabled={offset + limit >= total} onclick={() => (offset = offset + limit)}>Next</button>
  </span>
</div>

<style>
  /* Sits at the foot of a flush table card, so it brings its own inset. */
  .pager { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 16px 12px; }
</style>
