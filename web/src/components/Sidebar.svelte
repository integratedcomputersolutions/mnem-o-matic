<script>
  import Logo from './Logo.svelte';
  import { session, logout } from '../lib/session.svelte.js';
  import { route } from '../lib/router.svelte.js';
  import { sidebarGroups, activeFor } from '../lib/pages.js';

  const groups = $derived(sidebarGroups(session.user));
  const active = $derived(activeFor(route.path));
  let open = $state(false);
</script>

<aside class="sidebar" class:open>
  <div class="top">
    <a href="/" class="brand"><Logo size={30} /></a>
    <button type="button" class="btn ghost sm menu" onclick={() => (open = !open)} aria-label="Menu">☰</button>
  </div>
  <nav>
    {#each groups as g (g.name)}
      <div class="group">
        <div class="gname">{g.name}</div>
        {#each g.items as item (item.path)}
          <a href={item.path} class="item" class:on={active === item} onclick={() => (open = false)}>{item.label}</a>
        {/each}
      </div>
    {/each}
  </nav>
  <div class="bottom">
    <div class="who">
      <div class="truncate"><b>{session.user?.display_name || session.user?.username}</b></div>
      <div class="muted small truncate">{session.user?.username} · {session.user?.role}</div>
    </div>
    <button type="button" class="btn ghost sm" onclick={logout}>Sign out</button>
    <div class="brands">
      <img class="bai" src="/bostonai-logo.png" alt="Boston AI" height="22" />
      <div class="ics">
        <img src="/ics-logo.svg" alt="ICS" height="18" />
        <span>A Division of ICS{#if session.version}&nbsp;· v{session.version}{/if}</span>
      </div>
    </div>
  </div>
</aside>

<style>
  .sidebar { background: var(--sidebar); border-right: 1px solid var(--line); display: flex; flex-direction: column;
             position: sticky; top: 0; height: 100vh; overflow-y: auto; }
  .top { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; height: var(--topbar-h); }
  .brand:hover { text-decoration: none; }
  .menu { display: none; }
  nav { flex: 1; padding: 4px 10px 16px; }
  .group { margin-top: 12px; }
  .gname { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); padding: 6px 8px; font-weight: 700; }
  .item { display: block; padding: 7px 10px; border-radius: var(--radius-sm); color: var(--ink-2); font-weight: 500; }
  .item:hover { background: var(--hover); color: var(--ink); text-decoration: none; }
  .item.on { background: var(--brand); color: #fff; }
  .bottom { padding: 12px 16px 16px; border-top: 1px solid var(--line); display: flex; flex-direction: column; gap: 8px; }
  .brands { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; padding-top: 10px; border-top: 1px solid var(--line); }
  .ics { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 11.5px; }
  .ics img { border-radius: 3px; }
  @media (max-width: 860px) {
    .sidebar { position: static; height: auto; }
    .menu { display: inline-flex; }
    nav, .bottom { display: none; }
    .sidebar.open nav, .sidebar.open .bottom { display: block; }
    .sidebar.open .bottom { display: flex; }
  }
</style>
