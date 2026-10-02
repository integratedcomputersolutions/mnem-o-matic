<script>
  // "Are you sure?" for one destructive action. Closing it any way but the
  // confirm button calls onclose.
  import Modal from './Modal.svelte';
  let { open = false, title = '', confirm = 'Confirm', cancel = 'Cancel', onconfirm, onclose, children = undefined } = $props();
</script>

<Modal {open} {title} {onclose}>
  {@render children?.()}
  {#snippet footer()}
    <button class="btn ghost" onclick={() => onclose()}>{cancel}</button>
    <button class="btn danger" onclick={() => onconfirm()}>{confirm}</button>
  {/snippet}
</Modal>
