/** Independent Python/raw tracking parity plus actual worker enforcement. No fitting. */
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const url = args.includes('--url') ? args[args.indexOf('--url') + 1] : 'http://127.0.0.1:5173';
const path = resolve(root, 'reports/v4/tracking-parity-fixtures.json');
const text = await readFile(path, 'utf8'), fixtures = JSON.parse(text);
const report = { scope: 'Structural parity and six already-exposed real raw fixtures, not new evaluation',
  fixtureSha256: createHash('sha256').update(text).digest('hex'), policy: fixtures.policy,
  parity: [], worker: [], videos: [], humanWebcamTested: false, noModelOrThresholdChanges: true, passed: false };
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'networkidle' });
  for (const item of fixtures.cases) {
    const actual = await page.evaluate(async raw => {
      const { measureRawTracking } = await import('/src/tracking.ts');
      return measureRawTracking(raw);
    }, item.raw);
    assert.deepEqual(actual, item.expected, item.id);
    report.parity.push({ id: item.id, kind: item.kind, exactMatch: true, usable: actual.usable });
  }
  for (const item of fixtures.cases.filter(c => c.kind === 'already_seen_real_raw_fixture')) {
    const result = await page.evaluate(async raw => {
      const { classifyLandmarkFrames } = await import('/src/pipeline.ts');
      return classifyLandmarkFrames(raw);
    }, item.raw);
    assert.deepEqual(result.tracking, item.expected);
    if (!item.expected.usable) {
      assert.equal(result.accepted, false); assert.equal(result.label, null);
      assert.equal(result.classificationScore, null); assert.equal(result.confidence, null);
      assert.equal(result.classifierExecuted, false);
    } else {
      assert.equal(result.classifierExecuted, true);
      assert.equal(typeof result.classificationScore, 'number');
      assert.equal(result.classificationScore, result.confidence);
    }
    report.worker.push({ id: item.id, inputUsable: item.expected.usable, accepted: result.accepted, label: result.label,
      classifierExecuted: result.classifierExecuted, classificationScore: result.classificationScore,
      tracking: result.tracking, reason: result.reason, modelVersion: result.modelVersion });
  }
  const failed = report.worker.find(r => r.id === 'health4_signer3_rep2_sentences__sentences0021');
  assert.equal(failed.tracking.eitherHandDetection, 1); assert.equal(failed.tracking.rightHandCompleteness, .6);
  assert.equal(failed.classifierExecuted, false);
  const mapping = JSON.parse(await readFile(resolve(root, 'data/open-set-v2/clips/mapping.json'), 'utf8'));
  for (const id of ['kep1_signer3_rep1_sentences__sentences0000', 'kep1_signer7_rep3_sentences__sentences0000',
    'health4_signer3_rep1_sentences__sentences0021']) {
    const clip = mapping.clips[id]; assert.ok(clip, id);
    const file = resolve(root, clip.path);
    assert.equal(createHash('sha256').update(await readFile(file)).digest('hex'), clip.mp4_sha256);
    const result = await page.evaluate(async source => {
      const { analyzeVideo } = await import('/src/pipeline.ts');
      const video = document.createElement('video'); video.muted = true; video.playsInline = true;
      document.body.append(video);
      try {
        await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = () => reject(new Error('MP4 decode failed')); video.src = source; video.load(); });
        return await analyzeVideo(video);
      } finally { video.pause(); video.removeAttribute('src'); video.load(); video.remove(); }
    }, `/@fs/${file.replaceAll('\\', '/')}`);
    assert.ok(result.tracking); assert.equal(result.tracking.source, 'raw_landmark_frames');
    assert.equal(result.tracking.frameCount, result.frames);
    assert.equal(result.trackingQuality, result.tracking.requiredDataCompleteness);
    if (!result.tracking.usable) {
      assert.equal(result.accepted, false); assert.equal(result.classifierExecuted, false);
      assert.equal(result.label, null); assert.equal(result.classificationScore, null);
    } else { assert.equal(result.classifierExecuted, true); }
    report.videos.push({ id, rawLabel: clip.raw_label, role: 'already_exposed_integration_fixture_not_accuracy_evaluation',
      mp4Sha256: clip.mp4_sha256, ...result });
  }
  assert.equal(report.videos.find(r => r.id === 'kep1_signer3_rep1_sentences__sentences0000').classifierExecuted, true);
  assert.equal(report.videos.find(r => r.id === 'kep1_signer7_rep3_sentences__sentences0000').classifierExecuted, false);
  await page.evaluate(async () => { const { cancelPipeline } = await import('/src/pipeline.ts'); cancelPipeline(); });
  report.passed = true;
} finally {
  await browser.close();
  await mkdir(resolve(root, 'reports/v4'), { recursive: true });
  await writeFile(resolve(root, 'reports/v4/tracking-browser-parity.json'), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ passed: report.passed, parity: report.parity.length, realWorkerCases: report.worker.length, realMp4Cases: report.videos.length }));
