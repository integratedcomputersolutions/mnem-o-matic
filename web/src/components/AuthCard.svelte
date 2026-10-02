<script>
  import Logo from './Logo.svelte';
  let { title, subtitle = '', children = undefined, footer = undefined } = $props();
</script>

<div class="wrap">
  <div class="card">
    <div class="head"><Logo size={40} /></div>
    <h1>{title}</h1>
    {#if subtitle}<p class="muted">{subtitle}</p>{/if}
    <div class="body">{@render children?.()}</div>
    {#if footer}<div class="foot-slot">{@render footer()}</div>{/if}
    <div class="foot">
      <img class="ics" src="/ics-logo.svg" alt="ICS" />
      <span class="sep" aria-hidden="true"></span>
      <img class="bai" src="/bostonai-logo.png" alt="Boston AI" />
    </div>
  </div>
</div>

<style>
  .wrap { min-height: 100vh; display: grid; place-items: center; padding: 24px;
          background: radial-gradient(900px 500px at 50% -10%, rgba(66, 43, 111, 0.55), transparent 70%), var(--page); }
  .card { width: 100%; max-width: 420px; background: var(--card); border: 1px solid var(--line);
          border-radius: 12px; padding: 28px 28px 20px; box-shadow: var(--shadow); }
  .head { margin-bottom: 18px; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .body { margin-top: 16px; }
  .foot-slot { margin-top: 14px; }
  .foot { margin-top: 22px; padding-top: 14px; border-top: 1px solid var(--line); display: flex; align-items: center;
          gap: 10px; color: var(--muted); font-size: 12.5px; }
  .foot img { flex: 0 0 auto; width: auto; object-fit: contain; }
  .foot .bai { height: 26px; }
  .foot .ics { height: 24px; border-radius: 4px; }
  .sep { width: 1px; height: 20px; background: var(--line-strong); }
</style>
