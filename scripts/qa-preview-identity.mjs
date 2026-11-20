/** HTTP identity tests only; no model execution or simulated camera. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expectedPreview, inspectPreview } from './preview-identity.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'web/dist');
const expected = await expectedPreview(dist);
const bodies = Object.fromEntries(await Promise.all(Object.keys(expected).map(async route => [route, await readFile(path.join(dist, route === '/' ? 'index.html' : route))])));
let scenario = 'matching';
const server = createServer((req, res) => {
  let bytes = bodies[req.url];
  if (scenario === 'development' && req.url === '/') bytes = Buffer.from('<script src="/@vite/client"></script><script src="/src/main.tsx"></script>');
  if (scenario === 'stale-config' && req.url === '/models/model_config.json') bytes = Buffer.from('{"modelVersion":"signbridge-old"}');
  if (scenario === 'wrong-model' && req.url === '/models/model.onnx') bytes = Buffer.from('Invalid identity fixture; never passed to inference.');
  if (scenario === 'unrelated') { res.writeHead(404); res.end(); return; }
  res.writeHead(200); res.end(bytes);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const cases = [];
try {
  for (scenario of ['matching', 'development', 'stale-config', 'wrong-model', 'unrelated']) {
    const result = await inspectPreview(origin, expected);
    assert.equal(result.state, scenario === 'matching' ? 'matching' : 'incompatible');
    assert.ok(server.listening, 'Read-only probe must not stop an occupied process.');
    cases.push({ scenario, ...result });
  }
} finally { await new Promise(resolve => server.close(resolve)); }
const absent = await inspectPreview(origin, expected); assert.equal(absent.state, 'absent'); cases.push({ scenario: 'closed-port', ...absent });
await mkdir(path.join(root, 'publication-review/qa'), { recursive: true });
await writeFile(path.join(root, 'publication-review/qa/launcher-identity.json'), JSON.stringify({ passed: true, cases, cameraOrModelInferenceUsed: false }, null, 2));
console.log(JSON.stringify({ passed: true, cases }));
