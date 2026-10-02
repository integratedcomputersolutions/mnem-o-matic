// Client-specific "connect" snippets, filled from /api/connect. Each entry is
// {id, label, intro, blocks: [{title, code}], notes: [...]}. The token is a
// placeholder unless the Tokens page handed one over for this session.

const PLACEHOLDER = 'mnm_your_token_here';

export function connectUrl(info) {
  return info.https_mcp_url && info.https?.state === 'active' ? info.https_mcp_url : info.mcp_url;
}

export function snippets(info, token) {
  const url = connectUrl(info);
  const compact = url.replace('/mcp', '/mcp?compact=true');
  const origin = url.replace(/\/mcp$/, '');
  const tok = token || PLACEHOLDER;
  const ca = info.builtin_ca;
  const caNote = ca
    ? `This server uses its own certificate authority. Download it from ${info.ca_url} (fingerprint ${info.ca_fingerprint}) and trust it first; the HTTPS page has per-OS steps.`
    : null;

  return [
    {
      id: 'claude-code',
      label: 'Claude Code',
      intro: 'One command registers the server for your user.',
      blocks: [
        ...(ca ? [{ title: 'Trust the CA for Node-based tools (once per shell profile)', code: `export NODE_EXTRA_CA_CERTS=$HOME/mnemomatic-ca.crt` }] : []),
        { title: 'Add the server', code: `claude mcp add --transport http mnemomatic ${url} -H "Authorization: Bearer ${tok}"` },
        { title: 'Small-context models: a trimmed tool list', code: `claude mcp add --transport http mnemomatic ${compact} -H "Authorization: Bearer ${tok}"` },
      ],
      notes: ['Run `/mcp` inside Claude Code to confirm the connection.', caNote].filter(Boolean),
    },
    {
      id: 'claude-desktop',
      label: 'Claude Desktop',
      intro: 'Desktop launches local commands; mcp-remote bridges to an HTTP server.',
      blocks: [
        {
          title: 'claude_desktop_config.json',
          code: JSON.stringify({
            mcpServers: {
              mnemomatic: {
                command: 'npx',
                args: ['-y', 'mcp-remote', url, '--header', `Authorization: Bearer ${tok}`],
                ...(ca ? { env: { NODE_EXTRA_CA_CERTS: '/path/to/mnemomatic-ca.crt' } } : {}),
              },
            },
          }, null, 2),
        },
      ],
      notes: ['Settings → Developer → Edit Config opens the file. Restart Desktop afterwards.', caNote].filter(Boolean),
    },
    {
      id: 'cursor',
      label: 'Cursor',
      intro: 'Project-level .cursor/mcp.json or the global ~/.cursor/mcp.json.',
      blocks: [
        {
          title: '.cursor/mcp.json',
          code: JSON.stringify({ mcpServers: { mnemomatic: { url, headers: { Authorization: `Bearer ${tok}` } } } }, null, 2),
        },
      ],
      notes: [ca ? 'Cursor is a Node application: start it with NODE_EXTRA_CA_CERTS pointing at the CA file, or trust the CA system-wide.' : null].filter(Boolean),
    },
    {
      id: 'opencode',
      label: 'OpenCode',
      intro: 'opencode.json in the project, or ~/.config/opencode/opencode.json.',
      blocks: [
        {
          title: 'opencode.json',
          code: JSON.stringify({ mcp: { mnemomatic: { type: 'remote', url, headers: { Authorization: `Bearer ${tok}` } } } }, null, 2),
        },
      ],
      notes: [caNote].filter(Boolean),
    },
    {
      id: 'codex',
      label: 'Codex CLI',
      intro: 'Codex reads the token from an environment variable rather than the config file.',
      blocks: [
        { title: 'Shell', code: `export MNEMOMATIC_TOKEN=${tok}${ca ? '\nexport NODE_EXTRA_CA_CERTS=$HOME/mnemomatic-ca.crt' : ''}` },
        { title: '~/.codex/config.toml', code: `[mcp_servers.mnemomatic]\nurl = "${url}"\nbearer_token_env_var = "MNEMOMATIC_TOKEN"` },
      ],
      notes: ['Key names follow the Codex MCP configuration reference; check it if your version differs.'],
    },
    {
      id: 'llama-cpp',
      label: 'llama.cpp web UI',
      intro: 'A browser-based client: the browser itself calls this server, which makes it a cross-origin request.',
      blocks: [
        { title: 'In the web UI: Settings → MCP servers', code: `URL:    ${url}\nHeader: Authorization: Bearer ${tok}` },
        { title: 'On this server: allow that origin (restart after)', code: `MNEMOMATIC_CORS_ORIGINS=http://<llama-host>:8080` },
      ],
      notes: [
        'Origins match literally — scheme, host and port — so list every address you use to open the UI.',
        ca ? 'The browser running the UI must trust the CA too (import it in that browser or its OS).' : null,
      ].filter(Boolean),
    },
    {
      id: 'cli',
      label: 'mnemomatic-cli',
      intro: 'The shell client. Config file, environment, or flags — in that order of convenience.',
      blocks: [
        {
          title: '~/.config/mnemomatic/config.toml (chmod 600)',
          code: `[server]\nurl = "${origin}"\ntoken = "${tok}"${ca ? '\nca_cert = "/path/to/mnemomatic-ca.crt"' : ''}`,
        },
        { title: 'Or environment variables', code: `export MNEMOMATIC_SERVER_URL=${origin}\nexport MNEMOMATIC_TOKEN=${tok}${ca ? '\nexport MNEMOMATIC_CA_CERT=/path/to/mnemomatic-ca.crt' : ''}` },
        { title: 'Try it', code: `mnemomatic-cli search "deployment notes"\nmnemomatic-cli export -o ./backups/` },
      ],
      notes: [],
    },
    {
      id: 'curl',
      label: 'curl',
      intro: 'A raw initialize call, useful to prove the token and URL before configuring a client.',
      blocks: [
        {
          title: 'Initialize',
          code: `curl -sS ${ca ? '--cacert mnemomatic-ca.crt ' : ''}-X POST ${url} \\\n  -H "Authorization: Bearer ${tok}" \\\n  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \\\n  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'`,
        },
      ],
      notes: ['A 200 with a server capabilities block means the token works; 401 or 403 means it does not.'],
    },
  ];
}
