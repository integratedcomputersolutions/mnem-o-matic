<script>
  // A flush card holding one scrollable table. `children` is the table's
  // thead/tbody; `footer` renders under the table, inside the card.
  import Card from './Card.svelte';
  let { title = '', children, footer = undefined } = $props();
</script>

<Card {title} flush>
  <div class="table-wrap"><table class="table">{@render children()}</table></div>
  {@render footer?.()}
</Card>
