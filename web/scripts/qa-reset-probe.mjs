/** Bounded diagnostic of clip isolation. Uses the frozen model and seen data. */
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(web, '..');
const base = (process.argv[2] ?? 'http://127.0.0.1:5174').replace(/\/$/, '');
const mapping = JSON.parse(await readFile(path.join(root, 'data/clips/mapping.json'), 'utf8'));
const manifest = JSON.parse(await readFile(path.join(root, 'data/manifest.json'), 'utf8'));
const [id, clip] = Object.entries(mapping.clips).find(([, item]) => !item.is_known);
const source = manifest.samples.find((item) => item.id === id);
const fsUrl = (relative) => new URL(`/@fs/${path.resolve(root, relative).replaceAll('\\', '/')}`, base).href;
const jpegUrls = source.frame_paths.map(fsUrl);
const videoUrl = fsUrl(clip.path);
const jpeg = [], videos = [];
let failure;
function diff(a, b) {
  const left = a.flat(), right = b.flat();
  assert.equal(left.length, right.length);
  return Math.max(...left.map((value, i) => Math.abs(value - right[i])));
}
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.goto(`${base}/extract.html`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !!window.signbridge);
  for (let i = 0; i < 2; i++) {
    const extraction = await page.evaluate((urls) => window.signbridge.extractImagePaths(urls, 30), jpegUrls);
    const result = await page.evaluate(({ x, quality }) => window.signbridge.classifyFeatures(x, quality), extraction);
    jpeg.push({ extraction, result });
    console.log(`Isolated JPEG run ${i + 1}: ${result.confidence}`);
  }
  for (let i = 0; i < 2; i++) {
    videos.push(await page.evaluate(async (src) => {
      let video = window.resetProbeVideo;
      if (!video) {
        video = document.createElement('video');
        video.muted = true;
        document.body.appendChild(video);
        window.resetProbeVideo = video;
        await new Promise((resolve, reject) => {
          video.addEventListener('loadeddata', resolve, { once: true });
          video.addEventListener('error', reject, { once: true });
          video.src = src;
          video.load();
        });
      }
      return window.signbridge.analyzeVideo(video);
    }, videoUrl));
    console.log(`Isolated MP4 run ${i + 1}: ${videos.at(-1).confidence}`);
  }
} catch (error) {
  failure = error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) };
  process.exitCode = 1;
} finally {
  await browser.close();
}
const report = {
  status: failure ? 'failed' : 'completed', failure, diagnostic_only: true, source_id: id,
  modification: 'Fresh Holistic graph for every clip and forced initial seek to zero; frozen classifier unchanged.',
  jpeg_features_max_absolute_difference: jpeg.length === 2 ? diff(jpeg[0].extraction.x, jpeg[1].extraction.x) : null,
  jpeg_logits_max_absolute_difference: jpeg.length === 2 ? diff(jpeg[0].result.logits, jpeg[1].result.logits) : null,
  jpeg_runs: jpeg.map(({ extraction, result }) => ({ frames: extraction.frames, quality: extraction.quality, result })),
  mp4_logits_max_absolute_difference: videos.length === 2 ? diff(videos[0].logits, videos[1].logits) : null,
  mp4_runs: videos,
  limitations: ['Only one already-seen diagnostic clip; not a fresh model accuracy test.', 'JPEG and lossy MP4 are not expected to have identical feature vectors.'],
};
await mkdir(path.join(web, 'qa-artifacts'), { recursive: true });
await writeFile(path.join(web, 'qa-artifacts/repeatability-v2-reset-probe.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (!failure) {
  assert.equal(report.jpeg_features_max_absolute_difference, 0, 'Identical JPEG features must be bit-identical after graph reset.');
  assert.equal(report.jpeg_logits_max_absolute_difference, 0);
  assert.equal(report.mp4_logits_max_absolute_difference, 0, 'Identical MP4 logits must match after reset and forced seek.');
}
