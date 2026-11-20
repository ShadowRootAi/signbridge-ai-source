/** Diagnostic only: repeat already-seen real data; never changes a model/gate. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(webRoot, '..');
const base = (process.argv[2] ?? 'http://127.0.0.1:5174').replace(/\/$/, '');
const mapping = JSON.parse(await readFile(path.join(root, 'data/clips/mapping.json'), 'utf8'));
const manifest = JSON.parse(await readFile(path.join(root, 'data/manifest.json'), 'utf8'));
const [id, clip] = Object.entries(mapping.clips).find(([, sample]) => !sample.is_known);
const source = manifest.samples.find((sample) => sample.id === id);
assert.ok(source, 'The real clip must have an original frame manifest.');
const urlFor = (relative) => new URL(`/@fs/${path.resolve(root, relative).replaceAll('\\', '/')}`, base).href;
const videoUrl = urlFor(clip.path);
const jpegUrls = source.frame_paths.map(urlFor);
const samePage = [];
const freshBrowser = [];
const jpeg = [];
let failed;

const hashFeatures = (x) => createHash('sha256').update(Buffer.from(Float32Array.from(x.flat()).buffer)).digest('hex');
function difference(a, b) {
  const left = a.flat(), right = b.flat();
  assert.equal(left.length, right.length);
  let maximum = 0, squares = 0, changed = 0;
  for (let i = 0; i < left.length; i++) {
    const delta = Math.abs(left[i] - right[i]);
    maximum = Math.max(maximum, delta);
    squares += delta * delta;
    if (delta !== 0) changed++;
  }
  return { maximum_absolute_difference: maximum, rms_difference: Math.sqrt(squares / left.length), changed_values: changed, values: left.length };
}
function compareRuns(a, b) {
  return {
    logits: difference(a.result.logits, b.result.logits),
    features: difference(a.features, b.features),
    changed_pixel_frames: a.pixels.flatMap((frame, index) => frame.sha256 !== b.pixels[index]?.sha256 ? [{ index, first_currentTime: frame.currentTime, second_currentTime: b.pixels[index]?.currentTime }] : []),
    quality_difference: Math.abs(a.result.trackingQuality - b.result.trackingQuality),
    first_frames: a.result.frames,
    second_frames: b.result.frames,
  };
}
function summarize(run) {
  return { result: run.result, pixels: run.pixels, features_sha256: hashFeatures(run.features) };
}

async function openBrowser() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  page.setDefaultTimeout(120000);
  await page.goto(`${base}/extract.html`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !!window.signbridge);
  await page.evaluate(async () => {
    window.repeatDiag = { pixels: [], inputs: [] };
    const originalDraw = OffscreenCanvasRenderingContext2D.prototype.drawImage;
    OffscreenCanvasRenderingContext2D.prototype.drawImage = function (...args) {
      const result = originalDraw.apply(this, args);
      const sourceVideo = args[0];
      if (sourceVideo instanceof HTMLVideoElement) {
        const pixels = this.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
        const sha = crypto.subtle.digest('SHA-256', pixels).then((buffer) => Array.from(new Uint8Array(buffer), (value) => value.toString(16).padStart(2, '0')).join(''));
        window.repeatDiag.pixels.push({ currentTime: sourceVideo.currentTime, readyState: sourceVideo.readyState, seeking: sourceVideo.seeking, width: sourceVideo.videoWidth, height: sourceVideo.videoHeight, sha });
      }
      return result;
    };
    const originalPost = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (...args) {
      const message = args[0];
      if (message?.kind === 'infer') window.repeatDiag.inputs.push(structuredClone(message.x));
      return originalPost.apply(this, args);
    };
    const status = await window.signbridge.getModelStatus();
    if (!status.ready) throw new Error(status.message);
    await window.signbridge.warmupLandmarks();
  });
  return { browser, page };
}

async function runVideo(page) {
  return page.evaluate(async (src) => {
    let video = window.repeatVideo;
    if (!video) {
      video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      document.body.appendChild(video);
      window.repeatVideo = video;
      await new Promise((resolve, reject) => {
        video.addEventListener('loadeddata', resolve, { once: true });
        video.addEventListener('error', () => reject(new Error('Real MP4 decoding failed')), { once: true });
        video.src = src;
        video.load();
      });
    }
    window.repeatDiag.pixels = [];
    window.repeatDiag.inputs = [];
    const result = await window.signbridge.analyzeVideo(video);
    const pixels = await Promise.all(window.repeatDiag.pixels.map(async ({ sha, ...state }) => ({ ...state, sha256: await sha })));
    return { result, pixels, features: window.repeatDiag.inputs.at(-1) };
  }, videoUrl);
}

try {
  const session = await openBrowser();
  try {
    for (let i = 0; i < 3; i++) {
      samePage.push(await runVideo(session.page));
      console.log(`Same page real MP4 ${i + 1}/3: score=${samePage.at(-1).result.confidence}`);
    }
    for (let i = 0; i < 2; i++) {
      const extraction = await session.page.evaluate((urls) => window.signbridge.extractImagePaths(urls, 30, true), jpegUrls);
      const result = await session.page.evaluate(({ x, quality }) => window.signbridge.classifyFeatures(x, quality), extraction);
      jpeg.push({ extraction, result });
      console.log(`Same JPEG sequence ${i + 1}/2: score=${result.confidence}`);
    }
  } finally { await session.browser.close(); }
  for (let i = 0; i < 3; i++) {
    const session = await openBrowser();
    try {
      freshBrowser.push(await runVideo(session.page));
      console.log(`Fresh browser real MP4 ${i + 1}/3: score=${freshBrowser.at(-1).result.confidence}`);
    } finally { await session.browser.close(); }
  }
} catch (error) {
  failed = error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) };
  process.exitCode = 1;
}

let cached;
for (const line of (await readFile(path.join(root, 'data/features/extracted.jsonl'), 'utf8')).split('\n').filter(Boolean)) {
  const record = JSON.parse(line);
  if (record.id === id) { cached = record; break; }
}
const report = {
  status: failed ? 'failed' : 'completed', failure: failed, sample_id: id, true_class: clip.true_class,
  diagnostic_only: true, new_untouched_test: false,
  instrumentation: 'Observed actual OffscreenCanvas video pixels by SHA-256 and original worker infer input without modifying either. Pixel readback can affect timing; compare with the separate uninstrumented QA.',
  same_page_worker: samePage.map(summarize),
  fresh_browser_worker: freshBrowser.map(summarize),
  same_page_comparisons: samePage.slice(1).map((run, index) => ({ first: 0, second: index + 1, ...compareRuns(samePage[0], run) })),
  fresh_browser_comparisons: freshBrowser.slice(1).map((run, index) => ({ first: 0, second: index + 1, ...compareRuns(freshBrowser[0], run) })),
  first_same_vs_first_fresh: samePage.length && freshBrowser.length ? compareRuns(samePage[0], freshBrowser[0]) : undefined,
  jpeg_runs: jpeg.map(({ extraction, result }) => ({ result, quality: extraction.quality, frames: extraction.frames, features_sha256: hashFeatures(extraction.x) })),
  jpeg_feature_comparison: jpeg.length === 2 ? difference(jpeg[0].extraction.x, jpeg[1].extraction.x) : undefined,
  jpeg_logit_comparison: jpeg.length === 2 ? difference(jpeg[0].result.logits, jpeg[1].result.logits) : undefined,
  jpeg_vs_cached_training_extraction: jpeg.length && cached ? difference(jpeg[0].extraction.x, cached.x) : undefined,
  jpeg_vs_mp4_feature_comparison: jpeg.length && samePage.length ? difference(jpeg[0].extraction.x, samePage[0].features) : undefined,
  limitations: ['Only the already-seen mapped unknown clip is used; this is not a new accuracy test.', 'Original JPEG versus lossy MP4 features are not expected to be numerically identical.', 'No retraining, threshold tuning, fake webcam or synthetic sign data.'],
};
await mkdir(path.join(webRoot, 'qa-artifacts'), { recursive: true });
await writeFile(path.join(webRoot, 'qa-artifacts/repeatability.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ status: report.status, failure: failed, same_page_comparisons: report.same_page_comparisons, fresh_browser_comparisons: report.fresh_browser_comparisons, jpeg_feature_comparison: report.jpeg_feature_comparison, jpeg_vs_cached_training_extraction: report.jpeg_vs_cached_training_extraction }, null, 2));
