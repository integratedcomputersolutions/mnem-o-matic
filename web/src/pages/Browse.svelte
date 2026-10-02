<script>
  import BrowseNamespaces from './BrowseNamespaces.svelte';
  import BrowseItems from './BrowseItems.svelte';
  import ItemDetail from './ItemDetail.svelte';
  let { params = {} } = $props();
</script>

{#if params.ns && params.type && params.id}
  <ItemDetail ns={params.ns} type={params.type} id={params.id} />
{:else if params.ns}
  <BrowseItems ns={params.ns} type={params.type || 'document'} />
{:else}
  <BrowseNamespaces />
{/if}
