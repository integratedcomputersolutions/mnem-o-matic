<script>
  let { size = 28, wordmark = true } = $props();
</script>

<span class="logo">
  <img src="/mnemomatic.png" alt="" width={size} height={size} />
  {#if wordmark}<span class="word">Mnem-O-matic</span>{/if}
</span>

<style>
  .logo { display: inline-flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
  img { border-radius: 7px; display: block; }
</style>
