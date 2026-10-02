<script>
  // A just-issued secret the server will never show again: a message, the
  // value to copy, optional extra actions and a Done button.
  import CopyField from './CopyField.svelte';
  let { value, ondone, children, actions = undefined } = $props();
</script>

<div class="alert good mb">
  {@render children()}
  <div class="mt"><CopyField {value} secret /></div>
  <div class={actions ? 'row mt' : 'mt'}>
    {@render actions?.()}
    <button class="btn ghost sm" onclick={() => ondone()}>Done</button>
  </div>
</div>
