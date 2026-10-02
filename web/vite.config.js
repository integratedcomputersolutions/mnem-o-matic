import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// In development the Python server runs plain on :8000 and Vite serves the
// SPA on :5173. These paths are proxied through unchanged — changeOrigin is
// off on purpose so the request keeps Host: localhost:5173, which is what
// the API's Origin-must-match-Host check compares against. Cookies are
// host-scoped (not port-scoped), so the session set by :5173 works for :8000.
const backend = process.env.MNEMOMATIC_DEV_BACKEND || 'http://127.0.0.1:8000';
const proxied = ['/api', '/mcp', '/export', '/ca.crt', '/setup', '/health'];

export default defineConfig({
  plugins: [svelte()],
  build: {
    // Straight into the Python package: `pip install .`, the wheel, and
    // `uv run mnemomatic` all serve from there. Gitignored; the Docker build
    // produces it in its own stage.
    outDir: '../src/mnemomatic/static/app',
    emptyOutDir: true,
    assetsInlineLimit: 0,
    sourcemap: false,
    chunkSizeWarningLimit: 150,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(proxied.map((p) => [p, { target: backend, changeOrigin: false }])),
  },
});
