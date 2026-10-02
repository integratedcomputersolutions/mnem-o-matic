// The one way the UI talks to the server. Relative URLs, same-origin
// cookies, JSON in and out. A 401 anywhere but the login call means the
// session is gone, and the app drops back to the sign-in screen.

export class ApiError extends Error {
  constructor(status, message, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
    this.code = body && body.error ? body.error : null;
  }
}

let onSignedOut = () => {};
export function setSignedOutHandler(fn) {
  onSignedOut = fn;
}

async function request(method, path, body) {
  /** @type {RequestInit} */
  const init = { method, credentials: 'same-origin', headers: { Accept: 'application/json' } };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let resp;
  try {
    resp = await fetch(path, init);
  } catch (e) {
    throw new ApiError(0, 'Cannot reach the server.', null);
  }
  const text = await resp.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: 'invalid_response', details: text.slice(0, 200) };
    }
  }
  if (!resp.ok) {
    if (resp.status === 401 && path !== '/api/login') onSignedOut();
    const message = (data && (data.details || data.error)) || `Request failed (${resp.status})`;
    throw new ApiError(resp.status, message, data);
  }
  return data;
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body = {}) => request('POST', path, body),
  del: (path) => request('DELETE', path),
};

// Build a query string, skipping empty values.
export function qs(params) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') u.set(k, String(v));
  }
  const s = u.toString();
  return s ? `?${s}` : '';
}
