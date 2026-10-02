// One shape for "fetch something and show it": data, the error if the fetch
// failed, and whether it is in flight. Pages call `load` from an $effect so
// it re-runs when its inputs change, and render from `data` / `error`.
//
//   const items = remote([]);
//   $effect(() => { items.load(() => api.get(`/api/items${qs({ namespace })}`).then((r) => r.items)); });

export function remote(initial = null) {
  const r = $state({
    data: initial,
    error: null,
    busy: false,
    async load(fn) {
      r.busy = true;
      r.error = null;
      try {
        r.data = await fn();
      } catch (e) {
        r.error = e;
      } finally {
        r.busy = false;
      }
      return r.data;
    },
  });
  return r;
}

// The same shape for "do something on click/submit": whether it is running
// and the error if it failed. `run` returns whatever `fn` returns, or
// undefined when it threw.
//
//   const save = action();
//   const submit = () => save.run(() => api.post('/api/things', form));

export function action() {
  const a = $state({
    error: null,
    busy: false,
    async run(fn) {
      a.busy = true;
      a.error = null;
      try {
        return await fn();
      } catch (e) {
        a.error = e;
      } finally {
        a.busy = false;
      }
    },
  });
  return a;
}
