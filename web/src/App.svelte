<script>
  import { onMount } from 'svelte';
  import { session, refresh } from './lib/session.svelte.js';
  import { route, installRouter } from './lib/router.svelte.js';
  import { resolve } from './lib/pages.js';
  import Sidebar from './components/Sidebar.svelte';
  import Loading from './screens/Loading.svelte';
  import FirstRun from './screens/FirstRun.svelte';
  import Login from './screens/Login.svelte';
  import ChangePassword from './screens/ChangePassword.svelte';

  onMount(() => {
    refresh();
    return installRouter();
  });

  const resolved = $derived(resolve(route.path, session.user));
  const Page = $derived(resolved.entry.page);
</script>

<svelte:head>
  <title>Mnem-O-matic</title>
</svelte:head>

{#if session.loading}
  <Loading />
{:else if !session.user && session.firstRun}
  <FirstRun />
{:else if !session.user}
  <Login />
{:else if session.user.must_change_password}
  <!-- The shell is not mounted until a password is chosen, so no URL skips the gate. -->
  <ChangePassword forced />
{:else}
  <div class="shell">
    <Sidebar />
    <main class="main">
      <div class="container">
        {#key route.path}
          <Page params={resolved.params} />
        {/key}
      </div>
    </main>
  </div>
{/if}
