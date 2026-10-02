<script>
  // tone: good | warn | bad | neutral | brand. Always carries a label, so the
  // colour is reinforcement, never the only signal.
  let { tone = 'neutral', label } = $props();
</script>

<span class="badge {tone}"><span class="dot" aria-hidden="true"></span>{label}</span>
