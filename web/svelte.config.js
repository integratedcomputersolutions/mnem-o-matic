import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  // Runes everywhere: legacy `export let` / `on:click` syntax is an error,
  // not a silent fallback, so the whole app stays on one model.
  compilerOptions: { runes: true },
};
