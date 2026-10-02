<script>
  let { label, value = '', note = '', href = '', children = undefined } = $props();
</script>

<svelte:element this={href ? 'a' : 'div'} class="tile" {href}>
  <div class="label">{label}</div>
  <div class="value">{#if children}{@render children()}{:else}{value}{/if}</div>
  {#if note}<div class="note">{note}</div>{/if}
</svelte:element>

<style>
  .tile { display: block; background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
          padding: 14px 16px; color: inherit; }
  a.tile:hover { border-color: var(--line-strong); text-decoration: none; background: var(--hover); }
  .label { font-size: 12.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
  .value { font-size: 26px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .note { font-size: 13px; color: var(--ink-2); margin-top: 2px; }
</style>
