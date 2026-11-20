import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = dirname(web);
const [manifestArgument, outputArgument, ...options] = process.argv.slice(2);
assert(manifestArgument && outputArgument && (!options.length || (options.length === 3 && options[0] === '--reuse')),
  'Usage: node web/scripts/extract-recognition-scale-pilot.mjs <manifest.json> <output.jsonl> [--reuse <manifest.json> <features.jsonl>]');
const manifestPath = resolve(manifestArgument);
const outputPath = resolve(outputArgument);
assert.notEqual(manifestPath, outputPath);
const manifestBytes = await readFile(manifestPath);
const manifest = JSON.parse(manifestBytes);
const { samples } = manifest;
assert(Array.isArray(samples) && samples.length > 0, 'Manifest samples must be a nonempty list');
const ids = new Set();
const fingerprints = new Set();
for (const sample of samples) {
  assert(typeof sample.id === 'string' && sample.id.trim() && !ids.has(sample.id), 'Missing or duplicate sample ID');
  assert(typeof sample.path === 'string' && /\.mp4$/i.test(sample.path), `Expected an MP4 for ${sample.id}`);
  assert(/^[a-f0-9]{64}$/.test(sample.sha256) && !fingerprints.has(sample.sha256), `Missing or duplicate MP4 checksum for ${sample.id}`);
  ids.add(sample.id);
  fingerprints.add(sample.sha256);
}

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const extractor = createHash('sha256');
for (const name of ['pipeline.ts', 'recognizer.worker.ts', 'features.ts', 'tracking.ts', 'openset.ts', 'local-video.ts']) {
  extractor.update(name).update(await readFile(resolve(web, 'src', name)));
}
const extractorSha256 = extractor.digest('hex');
const manifestSha256 = digest(manifestBytes);
const expectedFeatureSchema = 'signbridge-landmarks-xy-v1';
const seekOffsetMs = 2;

async function loadReusableFeatures() {
  if (!options.length) return new Map();
  assert.equal(manifest.complete, true, 'Only a complete frozen manifest can rebind earlier features');
  const priorManifestPath = resolve(options[1]);
  const priorFeaturesPath = resolve(options[2]);
  assert(![priorManifestPath, priorFeaturesPath].includes(outputPath), 'Write new features to a separate path');
  const [manifestInput, featuresInput] = await Promise.all([readFile(priorManifestPath), readFile(priorFeaturesPath)]);
  const prior = JSON.parse(manifestInput);
  assert(prior.complete === true && Array.isArray(prior.samples), 'Invalid prior manifest');
  const priorById = new Map(prior.samples.map(sample => [sample.id, sample]));
  assert.equal(priorById.size, prior.samples.length, 'Duplicate prior manifest sample ID');
  const priorManifestSha256 = digest(manifestInput);
  const priorFeaturesSha256 = digest(featuresInput);
  const rows = new Map();
  const seen = new Set();
  for (const line of featuresInput.toString('utf8').trim().split(/\r?\n/)) {
    const row = JSON.parse(line);
    const sample = priorById.get(row.id);
    assert(sample && !seen.has(row.id) && row.fingerprint === sample.sha256
      && row.manifestSha256 === priorManifestSha256, `Unbound prior feature row ${row.id}`);
    seen.add(row.id);
    if (row.extractorSha256 !== extractorSha256 || row.featureSchema !== expectedFeatureSchema
      || row.seekOffsetMs !== seekOffsetMs) continue;
    assert(Array.isArray(row.x) && row.x.length === 48 && row.x.every(frame =>
      Array.isArray(frame) && frame.length === 224 && frame.every(Number.isFinite)), `Invalid prior features ${row.id}`);
    assert(typeof row.browserVersion === 'string' && Number.isInteger(row.frames) && row.frames >= 2 && row.frames <= 48
      && [row.quality, row.duration, row.elapsedMs, row.tracking?.eitherHandDetection, row.motion?.motionBodyRms]
        .every(Number.isFinite), `Invalid prior measurements ${row.id}`);
    rows.set(row.id, { ...row, manifestSha256, reusedFrom: { manifestSha256: priorManifestSha256, featuresSha256: priorFeaturesSha256 } });
  }
  assert.equal(seen.size, priorById.size, 'Prior feature file is incomplete');
  return rows;
}

async function readClip(sample) {
  const bytes = await readFile(resolve(root, sample.path));
  assert.equal(digest(bytes), sample.sha256, `MP4 checksum mismatch for ${sample.id}`);
  return bytes;
}

const reusable = await loadReusableFeatures();
await mkdir(dirname(outputPath), { recursive: true });
const stagingPath = `${outputPath}.${randomUUID()}.tmp`;
const origin = 'http://127.0.0.1:5328';
// The existing Vite runtime routes resolve packaged model assets from the web folder.
process.chdir(web);
const browsers = [];
let server;
let output;
try {
  const externalRequests = [];
  async function openExtractor() {
    // Independent headless browser processes avoid background-tab video timing throttling.
    const browser = await chromium.launch({ channel: 'chrome', headless: true,
      args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
    browsers.push(browser);
    const context = await browser.newContext({ serviceWorkers: 'block' });
    await context.route(/^https?:/, route => {
      const url = new URL(route.request().url());
      if (url.origin === origin) return route.continue();
      externalRequests.push(url.origin);
      return route.abort();
    });
    const page = await context.newPage();
    await page.goto(`${origin}/extract.html`);
    await page.waitForFunction(() => Boolean(window.signbridge));
    await page.evaluate(async () => {
      const [{ collectVideoLandmarks }, { preprocess, TIMESTEPS, FEATURE_COUNT, FEATURE_SCHEMA }, { measureRawTracking },
        { measureQualityMotion }, { loadLocalVideo, localVideoUrl }] = await Promise.all([
        import('/src/pipeline.ts'), import('/src/features.ts'), import('/src/tracking.ts'),
        import('/src/openset.ts'), import('/src/local-video.ts'),
      ]);
      const input = document.createElement('input');
      input.type = 'file';
      document.body.append(input);
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      document.body.append(video);
      window.extractMp4 = async () => {
        const [file] = input.files;
        if (!file) throw new Error('No MP4 supplied');
        const signal = AbortSignal.timeout(120_000);
        const url = localVideoUrl(file);
        try {
          await loadLocalVideo(video, url, signal);
          const { frames, start } = await collectVideoLandmarks(video, undefined, signal, true);
          const { x, quality } = preprocess(frames);
          if (x.length !== TIMESTEPS || x.some(frame => frame.length !== FEATURE_COUNT || !frame.every(Number.isFinite))) {
            throw new Error('Invalid landmark feature tensor');
          }
          return { x, quality, frames: frames.length, tracking: measureRawTracking(frames), motion: measureQualityMotion(x),
            duration: video.duration, elapsedMs: performance.now() - start, featureSchema: FEATURE_SCHEMA };
        } finally {
          input.value = '';
          video.removeAttribute('src');
          video.load();
          URL.revokeObjectURL(url);
        }
      };
    });
    return { page, browserVersion: browser.version() };
  }
  async function extract({ page, browserVersion }, sample) {
    const bytes = await readClip(sample);
    await page.locator('input[type=file]').setInputFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: bytes });
    const result = await page.evaluate(() => window.extractMp4());
    return { id: sample.id, fingerprint: sample.sha256, manifestSha256, extractorSha256, browserVersion, seekOffsetMs, ...result };
  }
  let completed = 0;
  async function append(row) {
    await output.writeFile(`${JSON.stringify(row)}\n`);
    console.log(JSON.stringify({ completed: ++completed, total: samples.length, id: row.id, reused: Boolean(row.reusedFrom),
      frames: row.frames, tracking: row.tracking.eitherHandDetection, elapsedMs: Math.round(row.elapsedMs) }));
  }
  output = await open(stagingPath, 'wx');
  const missing = [];
  for (const sample of samples) {
    const row = reusable.get(sample.id);
    if (row?.fingerprint === sample.sha256) {
      await readClip(sample);
      await append(row);
    } else {
      missing.push(sample);
    }
  }
  if (missing.length) {
    server = await createServer({
      root: web,
      configFile: resolve(web, 'vite.config.ts'),
      logLevel: 'error',
      server: { host: '127.0.0.1', port: 5328, strictPort: true, hmr: false, watch: null },
    });
    await server.listen();
    // Initialize one worker first so Vite's cold optimization cannot reload the other page mid-clip.
    const firstPage = await openExtractor();
    await append(await extract(firstPage, missing[0]));
    const pages = missing.length > 2 ? [firstPage, await openExtractor()] : [firstPage];
    for (let start = 1; start < missing.length; start += pages.length) {
      const rows = await Promise.all(missing.slice(start, start + pages.length).map((sample, index) => extract(pages[index], sample)));
      for (const row of rows) await append(row);
    }
  }
  assert.deepEqual(externalRequests, [], 'Expected local-only browser network traffic');
  await output.close();
  output = undefined;
  await rename(stagingPath, outputPath);
} finally {
  try {
    if (output) await output.close();
    await rm(stagingPath, { force: true });
  } finally {
    try { await Promise.all(browsers.map(browser => browser.close())); }
    finally { await server?.close(); }
  }
}
