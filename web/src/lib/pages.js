// The one table that drives the sidebar and the route guard. A page is shown
// in the sidebar when it has a label and the user's role allows it; a path
// resolves to a page only when the role allows it, otherwise NotFound.

import { matchPath } from './router.svelte.js';

import Dashboard from '../pages/Dashboard.svelte';
import Browse from '../pages/Browse.svelte';
import Search from '../pages/Search.svelte';
import Audit from '../pages/Audit.svelte';
import Connect from '../pages/Connect.svelte';
import Tokens from '../pages/Tokens.svelte';
import Account from '../pages/Account.svelte';
import Users from '../pages/admin/Users.svelte';
import Https from '../pages/admin/Https.svelte';
import Settings from '../pages/admin/Settings.svelte';
import NotFound from '../pages/NotFound.svelte';

export const pages = [
  { path: '/', page: Dashboard, role: 'any', label: 'Dashboard', group: 'Memory' },
  { path: '/browse', page: Browse, role: 'any', label: 'Browse', group: 'Memory' },
  { path: '/browse/:ns', page: Browse, role: 'any' },
  { path: '/browse/:ns/:type', page: Browse, role: 'any' },
  { path: '/browse/:ns/:type/:id', page: Browse, role: 'any' },
  { path: '/search', page: Search, role: 'any', label: 'Search', group: 'Memory' },
  { path: '/audit', page: Audit, role: 'any', label: 'Activity', group: 'Memory' },
  { path: '/connect', page: Connect, role: 'any', label: 'Connect an agent', group: 'Agents' },
  { path: '/tokens', page: Tokens, role: 'any', label: 'My tokens', group: 'Agents' },
  { path: '/account', page: Account, role: 'any', label: 'Account', group: 'You' },
  { path: '/admin/users', page: Users, role: 'admin', label: 'Users', group: 'Admin' },
  { path: '/admin/https', page: Https, role: 'admin', label: 'HTTPS', group: 'Admin' },
  { path: '/admin/settings', page: Settings, role: 'admin', label: 'Settings', group: 'Admin' },
];

export function allowed(entry, user) {
  return entry.role === 'any' || (user && user.role === entry.role);
}

// Sidebar groups, in first-seen order, holding only labelled pages the user may see.
export function sidebarGroups(user) {
  const groups = [];
  for (const entry of pages) {
    if (!entry.label || !allowed(entry, user)) continue;
    let g = groups.find((x) => x.name === entry.group);
    if (!g) groups.push((g = { name: entry.group, items: [] }));
    g.items.push(entry);
  }
  return groups;
}

export function resolve(path, user) {
  for (const entry of pages) {
    const params = matchPath(entry.path, path);
    if (params) {
      if (!allowed(entry, user)) break;
      return { entry, params };
    }
  }
  return { entry: { path, page: NotFound, role: 'any' }, params: {} };
}

// The sidebar item to highlight for a path: the longest labelled prefix.
export function activeFor(path) {
  let best = null;
  for (const entry of pages) {
    if (!entry.label) continue;
    const hit = entry.path === '/' ? path === '/' : path === entry.path || path.startsWith(entry.path + '/');
    if (hit && (!best || entry.path.length > best.path.length)) best = entry;
  }
  return best;
}
