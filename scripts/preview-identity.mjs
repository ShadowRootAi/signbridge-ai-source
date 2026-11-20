import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createConnection } from 'node:net';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
async function portIsAbsent(origin) {
  const url = new URL(origin);
  return new Promise(resolve => {
    const socket = createConnection({ host: url.hostname, port: Number(url.port || 80) });
    let finished = false;
    const finish = absent => { if (!finished) { finished = true; socket.destroy(); resolve(absent); } };
    socket.once('connect', () => finish(false));
    socket.once('error', error => finish(error.code === 'ECONNREFUSED'));
    socket.setTimeout(750, () => finish(false));
  });
}
export async function expectedPreview(dist) {
  return Object.fromEntries(await Promise.all([
    ['/', 'index.html'], ['/models/model_config.json', 'models/model_config.json'], ['/models/model.onnx', 'models/model.onnx'],
  ].map(async ([route, file]) => [route, digest(await readFile(path.join(dist, file)))])));
}
/** Read-only identity probe. An unrelated process is never stopped/replaced. */
export async function inspectPreview(origin, expected) {
  for (const [route, hash] of Object.entries(expected)) {
    let response;
    try { response = await fetch(new URL(route, origin), { cache: 'no-store', redirect: 'error', signal: AbortSignal.timeout(2000) }); }
    catch (error) {
      // A pooled HTTP socket may close differently from ECONNREFUSED. Confirm
      // absence with a fresh TCP connection; timeout/ambiguous failures conflict.
      if (route === '/' && await portIsAbsent(origin)) return { state: 'absent' };
      return { state: 'incompatible', reason: `Existing port did not provide a verifiable response for ${route}.` };
    }
    if (!response.ok) return { state: 'incompatible', reason: `${route} returned HTTP ${response.status}.` };
    const bytes = Buffer.from(await response.arrayBuffer());
    if (route === '/') {
      const html = bytes.toString('utf8');
      if (html.includes('/@vite/client') || html.includes('/src/main.tsx') || !html.includes('/assets/')) {
        return { state: 'incompatible', reason: 'Port serves development or unrelated HTML instead of the built app.' };
      }
    }
    if (digest(bytes) !== hash) return { state: 'incompatible', reason: `${route} does not match the current dist artifact.` };
  }
  return { state: 'matching' };
}
