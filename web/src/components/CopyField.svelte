<script>
  // A read-only value with a copy button. The clipboard API only exists in
  // secure contexts (HTTPS or localhost), so the button appears only there;
  // the text stays selectable everywhere. `secret` hides the value until asked.
  let { value = '', label = '', secret = false, multiline = false } = $props();
  const canCopy = typeof navigator !== 'undefined' && !!navigator.clipboard;
  let copied = $state(false);
  let revealed = $state(false);
  const shown = $derived(!secret || revealed);
  const display = $derived(shown ? value : '•'.repeat(Math.min(24, Math.max(8, value.length))));

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      copied = false;
    }
  }
</script>

<div class="cf">
  {#if label}<div class="label">{label}</div>{/if}
  <div class="box" class:multiline>
    {#if multiline}<pre class="val">{display}</pre>{:else}<code class="val">{display}</code>{/if}
    <div class="acts">
      {#if secret}<button type="button" class="btn ghost sm" onclick={() => (revealed = !revealed)}>{revealed ? 'Hide' : 'Show'}</button>{/if}
      {#if canCopy}<button type="button" class="btn sm" onclick={copy}>{copied ? 'Copied' : 'Copy'}</button>{/if}
    </div>
  </div>
</div>

<style>
  .cf { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .box { display: flex; align-items: flex-start; gap: 8px; background: var(--page); border: 1px solid var(--line-strong);
         border-radius: var(--radius-sm); padding: 8px 8px 8px 12px; }
  .val { flex: 1; min-width: 0; background: none; border: 0; padding: 0; word-break: break-all; white-space: pre-wrap;
         font-family: var(--mono); font-size: 13px; line-height: 1.5; color: var(--ink); user-select: all; }
  .acts { display: flex; gap: 6px; flex-shrink: 0; }
</style>
