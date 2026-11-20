/**
 * Real artifact / real video QA. Run only after training has produced artifacts.
 * This script never creates predictions, replaces model responses, or uses a
 * fake webcam. Label correctness is recorded, not required for a pipeline pass.
 *
 * node scripts/qa-real-model.mjs --url http://127.0.0.1:5174
 * Optional: --fixtures <path> --config <path> --model <path> --mapping <path>
 *           --skip-videos --skip-ui --timeout-ms 180000
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const projectRoot = path.resolve(webRoot, '..');
const outputRoot = path.join(webRoot, 'qa-artifacts');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const baseUrl = option('--url', 'http://127.0.0.1:5174').replace(/\/$/, '');
const fixturePath = path.resolve(option('--fixtures', path.join(projectRoot, 'artifacts', 'parity-fixtures.json')));
const configPath = path.resolve(option('--config', path.join(projectRoot, 'artifacts', 'config.json')));
const modelPath = path.resolve(option('--model', path.join(projectRoot, 'artifacts', 'model.onnx')));
const mappingPath = path.resolve(option('--mapping', path.join(projectRoot, 'data', 'clips', 'mapping.json')));
const timeoutMs = Number(option('--timeout-ms', '180000'));
const checks = [];
const parity = [];
const videos = [];
const uiUploads = [];
let cancellation;
const appRequests = new Set();
const pageErrors = [];
let browser;
let context;
let failure;
let modelVersion;
let modelSha256;

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const jsonFile = async (filename) => JSON.parse(await readFile(filename, 'utf8'));
function bounded(promise, description) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${description} exceeded ${timeoutMs} ms`)), timeoutMs); }),
  ]).finally(() => clearTimeout(timer));
}

function compare(actual, expected, atol, rtol, description) {
  assert.equal(actual.length, expected.length, `${description}: output dimension`);
  let maximumAbsoluteError = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `${description}: finite logit ${i}`);
    const error = Math.abs(actual[i] - expected[i]);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
    assert.ok(error <= atol + rtol * Math.abs(expected[i]), `${description}: logit ${i}, actual=${actual[i]}, expected=${expected[i]}, error=${error}`);
  }
  return maximumAbsoluteError;
}

function validateResult(result, description) {
  assert.equal(typeof result.accepted, 'boolean', `${description}: acceptance flag`);
  assert.ok(Number.isFinite(result.frames) && result.frames > 0, `${description}: actual decoded frames`);
  assert.ok(Number.isFinite(result.elapsedMs) && result.elapsedMs > 0, `${description}: elapsed time`);
  assert.ok(result.confidence === null || (Number.isFinite(result.confidence) && result.confidence >= 0 && result.confidence <= 1), `${description}: finite score`);
  assert.equal(result.modelVersion, modelVersion, `${description}: loaded model version`);
  assert.ok(typeof result.reason === 'string' && result.reason.length > 0, `${description}: explanation`);
}

try {
  const [fixtures, config, localModel] = await Promise.all([jsonFile(fixturePath), jsonFile(configPath), readFile(modelPath)]);
  modelVersion = config.modelVersion;
  modelSha256 = sha256(localModel);
  assert.ok(typeof modelVersion === 'string' && modelVersion.length > 0, 'A real model version is required.');
  assert.ok(Array.isArray(config.labels) && config.labels.length > 0, 'Known class labels are required.');
  assert.deepEqual(fixtures.shape.slice(1), [config.T, config.F], 'Fixture and model feature shapes must agree.');
  assert.equal(fixtures.x.length, fixtures.shape[0]);
  assert.equal(fixtures.pytorch_logits.length, fixtures.x.length);
  assert.equal(fixtures.onnx_logits.length, fixtures.x.length);
  assert.ok(fixtures.x.length > 0, 'Real parity cases must exist.');
  const atol = fixtures.atol ?? 1e-5;
  const rtol = fixtures.rtol ?? 1e-4;
  assert.ok(Number.isFinite(atol) && atol > 0 && Number.isFinite(rtol) && rtol >= 0);
  checks.push(`Read real fixture/config/model artifacts; ${fixtures.x.length} parity cases.`);

  browser = await chromium.launch({ channel: 'chrome', headless: true });
  context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
  context.on('request', (request) => appRequests.add(request.url()));
  const page = await context.newPage();
  page.setDefaultTimeout(timeoutMs);
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const servedConfigResponse = await context.request.get(`${baseUrl}/models/model_config.json`);
  assert.equal(servedConfigResponse.ok(), true, 'Served trained configuration must exist.');
  const servedConfig = await servedConfigResponse.json();
  assert.deepEqual(servedConfig, config, 'Browser configuration must be the exact trained artifact.');
  const servedModelResponse = await context.request.get(`${baseUrl}/models/model.onnx`);
  assert.equal(servedModelResponse.ok(), true, 'Served trained ONNX model must exist.');
  assert.equal(sha256(await servedModelResponse.body()), modelSha256, 'Served model bytes must match the trained artifact.');
  checks.push('Served model SHA-256 and configuration match the actual training artifacts.');

  await page.goto(`${baseUrl}/extract.html`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => typeof window.signbridge?.classifyFeatures === 'function');
  const status = await bounded(page.evaluate(() => window.signbridge.getModelStatus()), 'Real model loading');
  assert.equal(status.ready, true, status.message);
  assert.equal(status.modelVersion, modelVersion);
  assert.equal(status.rejectionCalibrated, config.rejection?.calibrated === true);

  for (let i = 0; i < fixtures.x.length; i++) {
    assert.equal(fixtures.x[i].length, config.T);
    for (const row of fixtures.x[i]) {
      assert.equal(row.length, config.F);
      assert.ok(row.every(Number.isFinite));
    }
    // quality=1 only controls rejection. It is not a claim of measured tracking
    // quality; these fixture checks intentionally test model logits alone.
    const prediction = await bounded(page.evaluate((x) => window.signbridge.classifyFeatures(x, 1), fixtures.x[i]), `Parity case ${i}`);
    const pytorchError = compare(prediction.logits, fixtures.pytorch_logits[i], atol, rtol, `Browser vs PyTorch case ${i}`);
    const onnxError = compare(prediction.logits, fixtures.onnx_logits[i], atol, rtol, `Browser vs desktop ONNX case ${i}`);
    parity.push({ index: i, source_id: fixtures.source_id?.[i], split: fixtures.split?.[i], maximum_absolute_error_vs_pytorch: pytorchError, maximum_absolute_error_vs_desktop_onnx: onnxError, browser_accepted: prediction.accepted, browser_label: prediction.label, browser_confidence: prediction.confidence, atol, rtol });
  }
  checks.push(`All ${parity.length} actual browser logit vectors match PyTorch and desktop ONNX within fixture tolerances.`);

  cancellation = await page.evaluate(async () => {
    const controller = new AbortController();
    const start = performance.now();
    const pending = window.signbridge.warmupLandmarks(controller.signal);
    controller.abort();
    try {
      await pending;
      return { rejected: false, elapsedMs: performance.now() - start };
    } catch (error) {
      return { rejected: error instanceof DOMException && error.name === 'AbortError', elapsedMs: performance.now() - start };
    }
  });
  assert.equal(cancellation.rejected, true, 'Real worker warmup request must reject when cancelled.');
  assert.ok(cancellation.elapsedMs < 1000, 'Cancellation must promptly return control, independent of background warmup.');
  checks.push('Cancellation promptly rejected a real landmark warmup request; no camera or fake model was used.');

  let samples = [];
  if (!args.includes('--skip-videos') || !args.includes('--skip-ui')) {
    const mapping = await jsonFile(mappingPath);
    assert.ok(mapping.clips && typeof mapping.clips === 'object', 'Real clip mapping is required.');
    samples = Object.entries(mapping.clips).map(([id, clip]) => ({ id, ...clip }));
    assert.ok(samples.length > 0, 'No real sample videos found.');
    for (const clip of samples) {
      const absolute = path.resolve(projectRoot, clip.path);
      const relative = path.relative(projectRoot, absolute);
      assert.ok(relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'Clip must be inside the project.');
      assert.equal(sha256(await readFile(absolute)), clip.mp4_sha256, `${clip.id}: original mapping checksum`);
      clip.absolutePath = absolute;
    }
    checks.push('Real sample MP4 checksums match the dataset-derived clip mapping.');
  }

  if (!args.includes('--skip-videos')) {
    for (const clip of samples) {
      const videoUrl = new URL(`/@fs/${clip.absolutePath.replaceAll('\\', '/')}`, baseUrl).href;
      const result = await bounded(page.evaluate(async (src) => {
        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        document.body.appendChild(video);
        let lastProgress;
        try {
          await new Promise((resolve, reject) => {
            video.addEventListener('loadeddata', resolve, { once: true });
            video.addEventListener('error', () => reject(new Error('Real sample could not be decoded.')), { once: true });
            video.src = src;
            video.load();
          });
          return await window.signbridge.analyzeVideo(video, (value) => { lastProgress = value; });
        } catch (error) {
          throw new Error(`${error instanceof Error ? error.message : String(error)}; videoState=${JSON.stringify({ src, currentTime: video.currentTime, duration: video.duration, readyState: video.readyState, seeking: video.seeking, width: video.videoWidth, height: video.videoHeight, mediaError: video.error?.message, lastProgress })}`);
        } finally {
          video.pause();
          video.removeAttribute('src');
          video.load();
          video.remove();
        }
      }, videoUrl), `Real video ${clip.id}`);
      validateResult(result, clip.id);
      videos.push({ id: clip.id, true_class: clip.true_class, signer_id: clip.signer_id, split: clip.split, is_known: clip.is_known, accepted: result.accepted, label: result.label, confidence: result.confidence, reason: result.reason, frames: result.frames, elapsedMs: result.elapsedMs, trackingQuality: result.trackingQuality, modelVersion: result.modelVersion,
        accepted_and_correct: clip.is_known ? result.accepted && result.label === clip.true_class : null,
        unknown_false_accept: clip.is_known ? null : result.accepted });
    }
    checks.push(`Completed actual MP4 decoding → landmarks → features → ONNX for ${videos.length} real held-out clips; correctness is recorded without forcing success.`);
  }

  if (!args.includes('--skip-ui')) {
    await page.goto(baseUrl, { waitUntil: 'networkidle' });
    await page.getByText('Το εκπαιδευμένο ONNX μοντέλο φορτώθηκε τοπικά.', { exact: true }).waitFor();
    const selected = [samples.find((clip) => clip.is_known), samples.find((clip) => !clip.is_known)].filter(Boolean);
    assert.equal(selected.length, 2, 'UI QA requires one known and one unknown real sample.');
    for (const clip of selected) {
      await page.locator('#video-file').setInputFiles(clip.absolutePath);
      const analyze = page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true });
      await page.waitForFunction(() => {
        const button = Array.from(document.querySelectorAll('button')).find((element) => element.textContent === 'Ανάλυση βίντεο');
        return button && !button.disabled;
      });
      await analyze.click();
      await page.locator('.result').waitFor();
      const rendered = await page.locator('.result').innerText();
      assert.equal(await page.getByRole('alert').count(), 0, 'UI must not hide a pipeline exception.');
      const accepted = await page.locator('.result.accepted').count() === 1;
      uiUploads.push({ id: clip.id, true_class: clip.true_class, is_known: clip.is_known, accepted, rendered_result: rendered });
    }
    await mkdir(outputRoot, { recursive: true });
    await page.screenshot({ path: path.join(outputRoot, 'ui-real-model.png'), fullPage: true });
    checks.push('Actual UI file uploads completed for one known and one unknown held-out video.');
  }

  const origin = new URL(baseUrl).origin;
  const externalRequests = [...appRequests].filter((url) => /^https?:/.test(url) && new URL(url).origin !== origin);
  assert.deepEqual(externalRequests, [], 'Observed application requests should remain on the local origin.');
  assert.deepEqual(pageErrors, [], 'No unhandled page errors.');
  checks.push('All observed application HTTP requests stayed on the local origin; no unhandled page errors.');
} catch (error) {
  failure = error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) };
  process.exitCode = 1;
} finally {
  await context?.close();
  await browser?.close();
  await mkdir(outputRoot, { recursive: true });
  const report = {
    status: failure ? 'failed' : 'passed', modelVersion, modelSha256, checks, failure,
    parity, cancellation, videos, uiUploads, observed_application_requests: [...appRequests], pageErrors,
    limitations: [
      'No human webcam session was available or tested. No fake webcam was used.',
      'Numerical parity verifies the exported model, not sign-language accuracy or feature extraction equivalence.',
      'Real clip outcomes are recorded including wrong predictions and rejections. A completed pipeline is not an accuracy pass.',
      'Fixture quality=1 is a rejection-control input, not measured video tracking quality.',
      'The small preselected held-out clip sample does not establish generalization to new live users.',
      'Network observation applies to application requests seen during this run, not every browser background process.',
    ],
  };
  await writeFile(path.join(outputRoot, 'real-model-qa.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, modelVersion, checks, failure, parityCases: parity.length, realVideos: videos.length, uiUploads: uiUploads.length }, null, 2));
}
