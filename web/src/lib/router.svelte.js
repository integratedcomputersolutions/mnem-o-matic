// A small history-API router. The page table (pages.js) decides what a path
// shows; this module only tracks the path and turns same-origin link clicks
// into navigations. Paths the server owns are left alone.

export const route = $state({ path: location.pathname, query: new URLSearchParams(location.search) });

function sync() {
  route.path = location.pathname;
  route.query = new URLSearchParams(location.search);
}

export function navigate(to, { replace = false } = {}) {
  if (replace) history.replaceState(null, '', to);
  else history.pushState(null, '', to);
  sync();
  window.scrollTo(0, 0);
}

const SERVER_PATHS = ['/api/', '/mcp', '/export', '/ca.crt', '/setup', '/health'];

function serverOwned(pathname) {
  return SERVER_PATHS.some((p) => pathname === p || pathname === p.replace(/\/$/, '') || pathname.startsWith(p));
}

export function installRouter() {
  window.addEventListener('popstate', sync);
  const onClick = (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target instanceof Element ? e.target.closest('a[href]') : null;
    if (!a || a.target || a.hasAttribute('download') || a.origin !== location.origin) return;
    if (serverOwned(a.pathname)) return;
    if (a.pathname === location.pathname && a.search === location.search && a.hash) return;
    e.preventDefault();
    navigate(a.pathname + a.search + a.hash);
  };
  document.addEventListener('click', onClick);
  return () => {
    window.removeEventListener('popstate', sync);
    document.removeEventListener('click', onClick);
  };
}

// '/browse/:ns/:type/:id' against '/browse/proj/document/abc' → {ns, type, id}
export function matchPath(pattern, path) {
  const pp = pattern.split('/').filter(Boolean);
  const sp = path.split('/').filter(Boolean);
  if (pp.length !== sp.length) return null;
  const params = {};
  for (let i = 0; i < pp.length; i++) {
    if (pp[i].startsWith(':')) {
      try {
        params[pp[i].slice(1)] = decodeURIComponent(sp[i]);
      } catch {
        return null;
      }
    } else if (pp[i] !== sp[i]) {
      return null;
    }
  }
  return params;
}

export const seg = (s) => encodeURIComponent(s);

// The detail page of one stored item.
export const itemHref = (ns, type, id) => `/browse/${seg(ns)}/${type}/${seg(id)}`;
