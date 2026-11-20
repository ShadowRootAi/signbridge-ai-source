import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = dirname(web);
const edition = resolve(root, 'publication-judge-demo');
const output = resolve(root, '.runtime-tmp/judge-final-rgb/poses.jsonl');
const staging = `${output}.${randomUUID()}.tmp`;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const origin = 'http://127.0.0.1:5330';
const modelPath = '/__signbridge/openhands/';

const audit = JSON.parse(await readFile(resolve(edition, 'library-audit.json')));
assert.equal(audit.status, 'local_judge_demo');
assert.equal(audit.openhands_frame_boundary_offset_ms, 2);
const pinnedModels = new Map();
for (const name of ['stgcn.onnx', 'lstm.onnx', 'stgcn-onnx.json', 'policy.json']) {
  const route = modelPath + name;
  const bytes = await readFile(resolve(edition, 'dist', route.slice(1)));
  const sha256 = digest(bytes);
  assert.equal(sha256, audit.runtime_files[route.slice(1)], `Judge asset checksum changed: ${name}`);
  pinnedModels.set(route, { bytes, sha256, contentType: name.endsWith('.onnx') ? 'application/octet-stream' : 'application/json' });
}
const policyAsset = pinnedModels.get(modelPath + 'policy.json');
const roster = new Set(JSON.parse(policyAsset.bytes).supportedLabels);
assert.equal(roster.size, 200);

const samples = [];
for (const [manifestRelativePath, expectedKnown] of [
  ['reports/openhands-engine/replay-manifest.json', true],
  ['data/isolated-sartinas-outside-roster/manifest.json', false],
]) {
  const manifestPath = resolve(root, manifestRelativePath);
  const bytes = await readFile(manifestPath);
  const manifestSha256 = digest(bytes);
  const { clips, policySha256 } = JSON.parse(bytes);
  assert.equal(clips.length, 24);
  if (expectedKnown) assert.equal(policySha256, policyAsset.sha256);
  for (const clip of clips) {
    assert.equal(clip.on_roster, expectedKnown);
    assert.equal(roster.has(clip.gloss), expectedKnown);
    assert.match(clip.sha256, /^[a-f0-9]{64}$/);
    assert.match(clip.path, /\.mp4$/i);
    samples.push({ ...clip, manifestSha256, file: resolve(dirname(manifestPath), clip.path) });
  }
}
assert.equal(new Set(samples.map(clip => clip.id)).size, 48);
assert.equal(new Set(samples.map(clip => clip.sha256)).size, 48);

// The existing Vite preview helpers look up local detector assets relative to web/.
process.chdir(web);
await mkdir(dirname(output), { recursive: true });
const server = await createServer({ root: web, configFile: resolve(web, 'vite.config.ts'), logLevel: 'error',
  server: { host: '127.0.0.1', port: 5330, strictPort: true, hmr: false, watch: null } });
const browsers = [];
const externalRequests = [];
let destination;

async function openExtractor() {
  // Use separate headless browsers to avoid throttling a background tab's frame callbacks.
  const browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
  browsers.push(browser);
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route(/^https?:/, route => {
    const request = new URL(route.request().url());
    if (request.origin !== origin) {
      externalRequests.push(request.origin);
      return route.abort();
    }
    const asset = pinnedModels.get(request.pathname);
    return asset ? route.fulfill({ body: asset.bytes, contentType: asset.contentType }) : route.continue();
  });
  const page = await context.newPage();
  await page.goto(`${origin}/extract.html`);
  await page.waitForFunction(() => Boolean(window.signbridge));
  await page.evaluate(async modelPath => {
    const [{ collectVideoLandmarks }, { loadLocalVideo, localVideoUrl }, { loadOpenHandsRecognizer, prepareOpenHandsPose }] =
      await Promise.all([import('/src/pipeline.ts'), import('/src/local-video.ts'), import('/src/openhands-recognizer.ts')]);
    const recognizer = await loadOpenHandsRecognizer({ modelUrl: `${modelPath}stgcn.onnx`, configUrl: `${modelPath}stgcn-onnx.json`,
      policyUrl: `${modelPath}policy.json`, secondaryModelUrl: `${modelPath}lstm.onnx` });
    const input = document.createElement('input');
    input.type = 'file';
    document.body.append(input);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    document.body.append(video);
    window.extractJudgePose = async () => {
      const file = input.files[0];
      if (!file) throw new Error('No recorded video supplied');
      const signal = AbortSignal.timeout(120_000);
      const url = localVideoUrl(file);
      try {
        await loadLocalVideo(video, url, signal);
        const { frames, start } = await collectVideoLandmarks(video, undefined, signal, true);
        const pose = prepareOpenHandsPose(frames);
        if (pose && (pose.length !== 2 * frames.length * 27 || !pose.every(Number.isFinite))) {
          throw new Error('Invalid OpenHands pose tensor');
        }
        const base = await recognizer.classify(frames, true);
        return { pose: pose && Array.from(pose), frames: frames.length, base, duration: video.duration,
          elapsedMs: performance.now() - start };
      } finally {
        input.value = '';
        video.removeAttribute('src');
        video.load();
        URL.revokeObjectURL(url);
      }
    };
  }, modelPath);
  return { page, browserVersion: browser.version() };
}

async function extract({ page, browserVersion }, { file, id, gloss, group, signer, on_roster, sha256, manifestSha256 }) {
  const bytes = await readFile(file);
  assert.equal(digest(bytes), sha256, `MP4 checksum changed: ${id}`);
  await page.locator('input[type=file]').setInputFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: bytes });
  const result = await page.evaluate(() => window.extractJudgePose());
  return { id, gloss, group, signer, on_roster, sha256, manifestSha256, browserVersion, seekOffsetMs: 2,
    policySha256: policyAsset.sha256, ...result };
}

let count = 0;
async function append(row) {
  await destination.writeFile(`${JSON.stringify(row)}\n`);
  console.log(JSON.stringify({ completed: ++count, total: samples.length, id: row.id, frames: row.frames,
    baseline: row.base.rejectionCode ?? row.base.acceptedBy, elapsedMs: Math.round(row.elapsedMs) }));
}

try {
  await server.listen();
  destination = await open(staging, 'wx');
  const first = await openExtractor();
  await append(await extract(first, samples[0]));
  const pages = [first, await openExtractor()];
  for (let start = 1; start < samples.length; start += pages.length) {
    const rows = await Promise.all(samples.slice(start, start + pages.length).map((clip, index) => extract(pages[index], clip)));
    for (const row of rows) await append(row);
  }
  assert.deepEqual(externalRequests, [], 'No data was expected to leave the browser');
  await destination.close();
  destination = undefined;
  await rename(staging, output);
} finally {
  try {
    if (destination) await destination.close();
    await rm(staging, { force: true });
  } finally {
    try { await Promise.all(browsers.map(browser => browser.close())); }
    finally { await server.close(); }
  }
}
