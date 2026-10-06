// Display helpers. Unknown or empty values render as an em dash so a table
// never shows "null" or "Invalid Date".

const DASH = '—';

export function fmtDate(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return DASH;
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

export function fmtDay(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return DASH;
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

export function fmtRelative(iso) {
  if (!iso) return DASH;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return DASH;
  const s = Math.round((Date.now() - then) / 1000);
  const abs = Math.abs(s);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  if (abs < 60) return rtf.format(-s, 'second');
  if (abs < 3600) return rtf.format(-Math.round(s / 60), 'minute');
  if (abs < 86400) return rtf.format(-Math.round(s / 3600), 'hour');
  if (abs < 86400 * 30) return rtf.format(-Math.round(s / 86400), 'day');
  return fmtDay(iso);
}

export function fmtNumber(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return DASH;
  return Number(n).toLocaleString();
}

export function truncate(text, n = 160) {
  if (!text) return '';
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}

export const TYPE_LABEL = { document: 'Document', knowledge: 'Knowledge', note: 'Note' };

export function typeLabel(t) {
  return TYPE_LABEL[t] || t || DASH;
}

// The human-facing title of an item of any type.
export function itemTitle(item) {
  return item?.title || item?.subject || item?.id || DASH;
}

export function errorText(e) {
  if (!e) return '';
  return e.message || String(e);
}

// "3.0.0 (c4e171b)" for CI images, plain "3.0.0" for local builds.
export function fmtVersion(version, build) {
  if (!version) return '';
  return build ? `${version} (${build.slice(0, 7)})` : version;
}
