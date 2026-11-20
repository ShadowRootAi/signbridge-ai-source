import { chromium } from 'playwright';
import { appendFile, readFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative, isAbsolute, sep, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(web, '..');
const option = (flag, fallback) => {
  const i = process.argv.indexOf(flag);
  if (i < 0) return fallback;
  if (process.argv[i + 1] === undefined || process.argv[i + 1].startsWith('--')) throw new Error(`Missing value for ${flag}`);
  return process.argv[i + 1];
};
const base = new URL(option('--url', 'http://127.0.0.1:5175'));
if (base.protocol !== 'http:' || base.hostname !== '127.0.0.1') throw new Error('Use a loopback Vite origin.');
const manifestPath = resolve(root, option('--manifest', 'data/continuous-v1/manifest.json'));
const output = resolve(root, option('--output', 'data/continuous-v1/features.jsonl'));
const stride = Number(option('--stride', 2));
const parallel = Number(option('--parallel', 2));
if (!Number.isInteger(stride) || stride < 1 || !Number.isInteger(parallel) || parallel < 1 || parallel > 4) throw new Error('Invalid stride or worker count');

const hash = createHash('sha256').update(`stride=${stride}\n`);
for (const file of ['src/continuous-extract.ts', 'src/recognizer.worker.ts', 'src/features.ts', 'public/models/holistic_landmarker.task', 'package-lock.json']) {
  hash.update(file).update(await readFile(resolve(web, file)));
}
const extractor_sha256 = hash.digest('hex');
const { samples } = JSON.parse(await readFile(manifestPath, 'utf8'));
if (!Array.isArray(samples) || !samples.length || new Set(samples.map(row => row.id)).size !== samples.length) throw new Error('Missing samples or duplicate IDs');
const sourceKey = (sample) => createHash('sha256').update(JSON.stringify([sample.fps, sample.frame_paths])).digest('hex');
const keys = new Map(samples.map(sample => [sample.id, sourceKey(sample)]));
const done = new Set();
try {
  for (const line of (await readFile(output, 'utf8')).split('\n').filter(Boolean)) {
    const row = JSON.parse(line);
    if (row.extractor_sha256 !== extractor_sha256 || row.source_key !== keys.get(row.id) || done.has(row.id)) {
      throw new Error('Existing features use a different extractor, source manifest, or contain duplicate IDs. Use a new output file.');
    }
    done.add(row.id);
  }
} catch (error) { if (error.code !== 'ENOENT') throw error; }
await mkdir(dirname(output), { recursive: true });

const remaining = samples.filter(row => !done.has(row.id));
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-background-timer-throttling'] });
let next = 0;
let writes = Promise.resolve();
try {
  await Promise.all(Array.from({ length: Math.min(parallel, remaining.length) }, async () => {
    const page = await browser.newPage();
    // The local app may be edited during a long extraction. Vite's hot reload
    // must not navigate this research page while a detector worker has a clip.
    await page.routeWebSocket(url => url.host === base.host, () => {});
    page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
    await page.goto(new URL('/extract.html', base).href);
    for (;;) {
      const sample = remaining[next++];
      if (!sample) break;
      const urls = sample.frame_paths.map((path) => {
        const absolute = resolve(root, path);
        const fromRoot = relative(root, absolute);
        if (!fromRoot || fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot)) throw new Error(`Source path outside the project: ${path}`);
        return `/@fs/${absolute.replaceAll('\\', '/').split('/').map((part, i) => i === 0 && /^[A-Z]:$/i.test(part) ? part : encodeURIComponent(part)).join('/')}`;
      });
      const extracted = await page.evaluate(async ({ urls, fps, stride }) => {
        const { extractContinuousFrames } = await import('/src/continuous-extract.ts');
        return extractContinuousFrames(urls, fps, stride);
      }, { urls, fps: sample.fps, stride });
      const row = { id: sample.id, extractor_sha256, source_key: keys.get(sample.id), ...extracted };
      writes = writes.then(() => appendFile(output, JSON.stringify(row) + '\n', 'utf8'));
      await writes;
      done.add(sample.id);
      console.log(JSON.stringify({ done: done.size, total: samples.length, id: row.id, frames: row.features.length, quality: row.quality, ms: Math.round(row.elapsed_ms) }));
    }
  }));
} finally { await browser.close(); await writes; }
console.log(JSON.stringify({ output, samples: done.size, extractor_sha256 }));
