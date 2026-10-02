<script>
  let { title = '', subtitle = '', actions = undefined, children = undefined, flush = false } = $props();
</script>

<section class="card" class:flush>
  {#if title || actions}
    <header class="head">
      <div>
        {#if title}<h2>{title}</h2>{/if}
        {#if subtitle}<div class="muted small">{subtitle}</div>{/if}
      </div>
      {#if actions}<div class="row">{@render actions()}</div>{/if}
    </header>
  {/if}
  <div class="body">{@render children?.()}</div>
</section>

<style>
  .card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
  .head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
          padding: 14px 18px; border-bottom: 1px solid var(--line); background: var(--card-head); }
  .body { padding: 16px 18px; }
  .flush .body { padding: 0; }
</style>
