<script>
  let { open = $bindable(false), title = '', wide = false, locked = false, onclose = undefined, children = undefined, footer = undefined } = $props();

  function close() {
    if (locked) return;
    open = false;
    onclose?.();
  }
  function onkey(e) {
    if (open && e.key === 'Escape') close();
  }
  function onbackdrop(e) {
    if (e.target === e.currentTarget) close();
  }
</script>

<svelte:window onkeydown={onkey} />

{#if open}
  <div class="backdrop" role="presentation" onclick={onbackdrop} onkeydown={onkey}>
    <div class="modal" class:wide role="dialog" aria-modal="true" aria-label={title}>
      <header>
        <h2>{title}</h2>
        {#if !locked}<button class="btn ghost sm" onclick={close} aria-label="Close">✕</button>{/if}
      </header>
      <div class="body">{@render children?.()}</div>
      {#if footer}<footer>{@render footer()}</footer>{/if}
    </div>
  </div>
{/if}

<style>
  .backdrop { position: fixed; inset: 0; background: rgba(10, 5, 24, 0.7); display: grid; place-items: center;
              padding: 20px; z-index: 50; }
  .modal { width: 100%; max-width: 480px; background: var(--card); border: 1px solid var(--line-strong);
           border-radius: 12px; box-shadow: var(--shadow); }
  .modal.wide { max-width: 720px; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px;
           border-bottom: 1px solid var(--line); }
  .body { padding: 18px; }
  footer { display: flex; justify-content: flex-end; gap: 10px; padding: 12px 18px; border-top: 1px solid var(--line); }
</style>
