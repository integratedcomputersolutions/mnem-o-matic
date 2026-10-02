<script>
  // steps: ['Name the host', 'Trust the CA', ...]; current is the 0-based active index.
  let { steps = [], current = 0, onselect = undefined } = $props();
</script>

<ol class="stepper">
  {#each steps as step, i (step)}
    <li class:done={i < current} class:on={i === current}>
      <button type="button" class="step" onclick={() => onselect?.(i)} disabled={!onselect}>
        <span class="n">{i < current ? '✓' : i + 1}</span>
        <span>{step}</span>
      </button>
    </li>
  {/each}
</ol>

<style>
  .stepper { list-style: none; margin: 0 0 18px; padding: 0; display: flex; gap: 6px; flex-wrap: wrap; }
  .step { display: inline-flex; align-items: center; gap: 8px; background: none; border: 1px solid var(--line);
          color: var(--muted); border-radius: 999px; padding: 5px 12px 5px 6px; font: inherit; font-size: 13.5px; }
  .step:not(:disabled) { cursor: pointer; }
  .n { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; background: var(--raised);
       font-size: 12px; font-weight: 700; color: var(--ink-2); }
  .on .step { border-color: var(--brand-bright); color: var(--ink); }
  .on .n { background: var(--brand); color: #fff; }
  .done .step { color: var(--ink-2); }
  .done .n { background: var(--good-soft); color: var(--good); }
</style>
