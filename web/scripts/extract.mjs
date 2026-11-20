import { chromium } from 'playwright';
import { readFile, writeFile, appendFile, mkdir, access } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(web, '..');
const args = process.argv.slice(2);
const get = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const base = get('--url', 'http://127.0.0.1:5173');
const limit = Number(get('--limit', '100000'));
const shard = Number(get('--shard', '0'));
const shards = Number(get('--shards', '1'));
if (!Number.isInteger(shard) || !Number.isInteger(shards) || shards < 1 || shard < 0 || shard >= shards) throw new Error('Invalid shard parameters');
const manifestPath = resolve(root, get('--manifest', 'data/manifest.json'));
const outputPath = resolve(root, get('--output', `data/features/extracted-isolated${shards > 1 ? `-${shard}` : ''}.jsonl`));
const reportPath = resolve(root, get('--report', `reports/extraction${shards > 1 ? `-${shard}` : ''}.json`));
const quiet = args.includes('--quiet');
const featureHash = createHash('sha256').update(await readFile(resolve(web, 'src/features.ts'))).digest('hex');
const pipelineHash = createHash('sha256');
for (const name of ['src/features.ts', 'src/pipeline.ts', 'src/recognizer.worker.ts']) {
  pipelineHash.update(name).update(await readFile(resolve(web, name)));
}
const extractionHash = pipelineHash.digest('hex');
await mkdir(dirname(outputPath), { recursive: true });
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const samples = manifest.samples.filter((_, i) => i % shards === shard).slice(0, limit);
const done = new Set();
try {
  for (const line of (await readFile(outputPath, 'utf8')).trim().split('\n').filter(Boolean)) {
    const record = JSON.parse(line);
    if (record.feature_sha256 !== featureHash || record.pipeline_sha256 !== extractionHash) throw new Error('Cached extractor differs; use a new output file.');
    done.add(record.id);
  }
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const rawGroups = new Set();
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-background-timer-throttling'] });
const page = await browser.newPage();
page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
page.setDefaultTimeout(120000);
try {
  for (let attempt = 0; attempt < 3; attempt++) {
    await page.goto(`${base}/extract.html`);
    await page.waitForFunction(() => Boolean(window.signbridge));
    console.log('Initializing real MediaPipe Holistic CPU in browser worker...');
    try { await page.evaluate(() => window.signbridge.warmupLandmarks()); break; }
    catch (error) {
      // A first Vite dependency optimization may reload the harness exactly once.
      if (!String(error).includes('Execution context was destroyed') || attempt === 2) throw error;
    }
  }
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    if (done.has(sample.id)) continue;
    const paths = sample.frame_paths.map(p => resolve(root, p));
    const waitStart = Date.now();
    while (true) {
      try { await Promise.all(paths.map(p => access(p))); break; }
      catch { if (Date.now() - waitStart > 180000) throw new Error(`Source clip not ready: ${sample.id}`); await new Promise(r => setTimeout(r, 1000)); }
    }
    const urls = paths.map(p => `${base}/@fs/${p.replaceAll('\\', '/')}`);
    const group = `${sample.split}:${sample.raw_label}`;
    const includeRaw = !rawGroups.has(group);
    const result = await page.evaluate(async ({ urls, fps, includeRaw }) => window.signbridge.extractImagePaths(urls, fps, includeRaw), { urls, fps: sample.fps ?? 30, includeRaw });
    rawGroups.add(group);
    const record = { id: sample.id, feature_sha256: featureHash, pipeline_sha256: extractionHash, ...result };
    await appendFile(outputPath, JSON.stringify(record) + '\n', 'utf8');
    done.add(sample.id);
    if (!quiet || done.size % 10 === 0 || done.size === samples.length) {
      console.log(JSON.stringify(quiet ? { completed: done.size, total: samples.length } : { completed: done.size, total: samples.length, id: sample.id, quality: result.quality, frames: result.frames, ms: Math.round(result.elapsedMs) }));
    }
  }
} finally { await browser.close(); }
const report = { samples: done.size, requested: samples.length, shard, shards, feature_sha256: featureHash, pipeline_sha256: extractionHash, output: outputPath, extractor: '@mediapipe/tasks-vision 1.2.2 Holistic CPU IMAGE, new Chrome Web Worker per clip', noSyntheticLandmarks: true };
await mkdir(dirname(reportPath), { recursive: true });
await writeFile(reportPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
