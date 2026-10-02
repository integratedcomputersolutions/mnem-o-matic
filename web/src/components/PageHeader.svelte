<script>
  let { title, subtitle = '', actions = undefined, children = undefined } = $props();
</script>

<div class="ph">
  <div>
    <h1>{title}</h1>
    {#if subtitle}<div class="muted">{subtitle}</div>{/if}
    {@render children?.()}
  </div>
  {#if actions}<div class="row">{@render actions()}</div>{/if}
</div>

<style>
  .ph { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
</style>
