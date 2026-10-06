import { api, setSignedOutHandler } from './api.js';

// Who is signed in, and what the server told us about itself. Exported as
// one mutable object so every importer sees the same state.
export const session = $state({
  loading: true,
  user: null,
  firstRun: false,
  version: null,
  build: null,
  https: null,
  error: null,
  freshToken: null,   // a token just minted, for the Connect page; never persisted
});

export async function refresh() {
  try {
    const s = await api.get('/api/session');
    session.user = s.authenticated ? s.user : null;
    session.firstRun = !!s.first_run;
    session.version = s.version;
    session.build = s.build;
    session.https = s.https;
    session.error = null;
  } catch (e) {
    session.error = e.message;
  } finally {
    session.loading = false;
  }
}

// Everything tied to the person signed in. A shared browser tab must not
// carry one person's freshly minted token into the next person's session.
export function signOutLocally() {
  session.user = null;
  session.freshToken = null;
}
setSignedOutHandler(signOutLocally);

export async function login(username, password) {
  const r = await api.post('/api/login', { username, password });
  signOutLocally();
  session.user = r.user;
  session.firstRun = false;
  return r.user;
}

export async function logout() {
  try {
    await api.post('/api/logout');
  } finally {
    signOutLocally();
  }
}

export const isAdmin = () => !!session.user && session.user.role === 'admin';
