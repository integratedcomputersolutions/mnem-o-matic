<script>
  // tabs: [[id, label], ...]. `active` may be bound, or driven from outside
  // (a URL) with `onchange` asked to navigate.
  let { tabs = [], active = $bindable(), onchange = undefined, children = undefined } = $props();
  $effect(() => {
    if (active === undefined && tabs.length) active = tabs[0][0];
  });
  function pick(id) {
    active = id;
    onchange?.(id);
  }
</script>

<div class="tabs" role="tablist">
  {#each tabs as [id, label] (id)}
    <button type="button" role="tab" aria-selected={active === id} class="tab" class:on={active === id}
            onclick={() => pick(id)}>{label}</button>
  {/each}
</div>
{@render children?.()}

<style>
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
  .tab { background: none; border: 0; border-bottom: 2px solid transparent; margin-bottom: -1px; color: var(--ink-2);
         padding: 8px 12px; font: inherit; font-weight: 550; cursor: pointer; border-radius: 6px 6px 0 0; }
  .tab:hover { color: var(--ink); background: var(--hover); }
  .tab.on { color: var(--ink); border-bottom-color: var(--brand-bright); }
</style>
