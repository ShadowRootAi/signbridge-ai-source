/** Loopback-only static review server with the proposed response headers. */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(web, '../publication-review/dist');
const headers = JSON.parse(await readFile(path.join(web, 'deployment/headers.json'), 'utf8'));
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.txt': 'text/plain; charset=utf-8', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.task': 'application/octet-stream' };
const server = createServer(async (req, res) => {
  for (const [name, value] of Object.entries(headers)) res.setHeader(name, value);
  if (!['GET', 'HEAD'].includes(req.method ?? '')) { res.writeHead(405, { Allow: 'GET, HEAD' }); res.end(); return; }
  try {
    const route = decodeURIComponent(new URL(req.url, 'http://127.0.0.1:5176').pathname);
    if (route === '/_headers' || route.startsWith('/__signbridge/') || route.startsWith('/@fs/')) { res.writeHead(404); res.end(); return; }
    const target = path.resolve(root, `.${route === '/' ? '/index.html' : route}`);
    const relative = path.relative(root, target);
    if (relative.startsWith('..') || path.isAbsolute(relative)) { res.writeHead(403); res.end(); return; }
    const body = await readFile(target);
    res.writeHead(200, { 'Content-Type': mime[path.extname(target)] ?? 'application/octet-stream', 'Content-Length': body.length });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch { res.writeHead(404); res.end(); }
});
server.listen(5176, '127.0.0.1', () => console.log('Unpublished static review: http://127.0.0.1:5176'));
