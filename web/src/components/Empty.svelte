<script>
  let { text = 'Nothing here yet.', children = undefined } = $props();
</script>

<div class="empty">{#if children}{@render children()}{:else}{text}{/if}</div>

<style>
  .empty { padding: 28px 16px; text-align: center; color: var(--muted); border: 1px dashed var(--line-strong);
           border-radius: var(--radius); }
</style>
