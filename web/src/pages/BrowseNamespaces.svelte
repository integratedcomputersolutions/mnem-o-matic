<script>
  import PageHeader from '../components/PageHeader.svelte';
  import Card from '../components/Card.svelte';
  import Empty from '../components/Empty.svelte';
  import ErrorBox from '../components/ErrorBox.svelte';
  import { api } from '../lib/api.js';
  import { seg } from '../lib/router.svelte.js';
  import { fmtNumber } from '../lib/format.js';

  let namespaces = $state(null);
  let error = $state(null);
  $effect(() => {
    api.get('/api/namespaces').then((r) => (namespaces = r.namespaces)).catch((e) => (error = e));
  });
</script>

<PageHeader title="Browse" subtitle="Everything the agents have stored, by namespace." />
<ErrorBox {error} />
{#if namespaces && namespaces.length === 0}
  <Empty text="The store is empty. Connect an agent and let it remember something." />
{:else if namespaces}
  <Card flush>
    <div class="table-wrap"><table class="table">
      <thead><tr><th>Namespace</th><th class="num">Documents</th><th class="num">Knowledge</th><th class="num">Notes</th></tr></thead>
      <tbody>
        {#each namespaces as n (n.name)}
          <tr>
            <td><a href={`/browse/${seg(n.name)}`}><b>{n.name}</b></a></td>
            <td class="num"><a href={`/browse/${seg(n.name)}/document`}>{fmtNumber(n.documents)}</a></td>
            <td class="num"><a href={`/browse/${seg(n.name)}/knowledge`}>{fmtNumber(n.knowledge)}</a></td>
            <td class="num"><a href={`/browse/${seg(n.name)}/note`}>{fmtNumber(n.notes)}</a></td>
          </tr>
        {/each}
      </tbody>
    </table></div>
  </Card>
{/if}
