<script>
  import PageHeader from '../components/PageHeader.svelte';
  import TableCard from '../components/TableCard.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import { api } from '../lib/api.js';
  import { remote } from '../lib/load.svelte.js';
  import { seg } from '../lib/router.svelte.js';
  import { fmtNumber } from '../lib/format.js';

  const namespaces = remote();
  $effect(() => { namespaces.load(() => api.get('/api/namespaces').then((r) => r.namespaces)); });
</script>

<PageHeader title="Browse" subtitle="Everything the agents have stored, by namespace." />
<ErrorBox error={namespaces.error} />
{#if namespaces.data && namespaces.data.length === 0}
  <Empty text="The store is empty. Connect an agent and let it remember something." />
{:else if namespaces.data}
  <TableCard>
    <thead><tr><th>Namespace</th><th class="num">Documents</th><th class="num">Knowledge</th><th class="num">Notes</th></tr></thead>
    <tbody>
      {#each namespaces.data as n (n.name)}
        <tr>
          <td><a href={`/browse/${seg(n.name)}`}><b>{n.name}</b></a></td>
          <td class="num"><a href={`/browse/${seg(n.name)}/document`}>{fmtNumber(n.documents)}</a></td>
          <td class="num"><a href={`/browse/${seg(n.name)}/knowledge`}>{fmtNumber(n.knowledge)}</a></td>
          <td class="num"><a href={`/browse/${seg(n.name)}/note`}>{fmtNumber(n.notes)}</a></td>
        </tr>
      {/each}
    </tbody>
  </TableCard>
{/if}
