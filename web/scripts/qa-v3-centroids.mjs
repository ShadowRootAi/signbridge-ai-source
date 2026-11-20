/** Separate research QA: actual exported centroids, no deployment or fitting. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(web, '..'), output = path.join(root, 'reports/v3');
const args = process.argv.slice(2), option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const base = option('--url', 'http://127.0.0.1:5173').replace(/\/$/, '');
const read = async p => JSON.parse(await readFile(p, 'utf8'));
const hash = b => createHash('sha256').update(b).digest('hex');
const local = p => new URL(`/@fs/${p.replaceAll('\\', '/')}`, base).href;
const rejectReasons = reasons => Object.keys(reasons).filter(key => reasons[key]).sort();
const diagnosticsVector = d => [d.motionBodyRms, d.motionLocalRms, d.validTrackedPairFraction, ...d.meanPresence];
function stats(values) {
  const sorted = [...values].sort((a, b) => a - b), n = sorted.length;
  return n ? { n, minMs: sorted[0], medianMs: (sorted[Math.floor((n - 1) / 2)] + sorted[Math.ceil((n - 1) / 2)]) / 2,
    p95Ms: sorted[Math.max(0, Math.ceil(n * .95) - 1)], maxMs: sorted[n - 1], meanMs: values.reduce((a, b) => a + b, 0) / n } : { n: 0 };
}
function compare(actual, expected, atol, rtol, name) {
  assert.equal(actual.length, expected.length, `${name}: shape`);
  let max = 0;
  actual.forEach((value, i) => {
    assert.ok(Number.isFinite(value) && Number.isFinite(expected[i]), `${name}: finite ${i}`);
    const delta = Math.abs(value - expected[i]); max = Math.max(max, delta);
    assert.ok(delta <= atol + rtol * Math.abs(expected[i]), `${name}[${i}] delta ${delta}: ${value} vs ${expected[i]}`);
  });
  return max;
}
const report = { role: 'adaptive_development_parity_and_exposed_V2_MP4_regression_NOT_new_final_test', createdAt: new Date().toISOString(),
  deploymentChanged: false, parametersFittedByThisScript: false, humanWebcamTested: false,
  warning: 'Methods/categories were chosen after earlier V2 results. These same MP4 examples are exposed regression fixtures. Improvements cannot confirm performance on new users or unseen classes.',
  validationLimitation: 'The validation-selected augmented_pool rejected all11 ΓΕΙΑ validation clips; it must not be described as retaining reliable coverage of all3 supported signs.',
  parity: [], classifierTiming: [], videos: [], errors: [] };
const requests = new Set(); let browser, failure;
async function save() { await mkdir(output, { recursive: true }); await writeFile(path.join(output, 'browser-comparison.json'), JSON.stringify(report, null, 2)); }
try {
  const exportPath = path.join(root, 'artifacts/v3-centroid/browser/export.locked.json');
  const exported = await read(exportPath), fitted = await read(path.join(root, 'artifacts/v3-centroid/fit.locked.json'));
  assert.equal(exported.fitLockSha256, hash(await readFile(path.join(root, 'artifacts/v3-centroid/fit.locked.json'))));
  report.exportLockSha256 = hash(await readFile(exportPath)); report.fitLockSha256 = exported.fitLockSha256;
  report.selectedDevelopmentCandidate = fitted.selected_development_candidate;
  const development = await read(path.join(root, 'reports/v3/centroid-development-results.json'));
  const selectedValidation = development.methods[fitted.selected_development_candidate]?.validation.metrics;
  report.selectedValidation = selectedValidation;
  const methods = [];
  for (const [name, identity] of Object.entries(exported.methods)) {
    const dir = path.join(root, 'artifacts/v3-centroid/browser', name);
    const configFile = path.join(dir, 'browser-config.json'), modelFile = path.join(dir, 'model.onnx'), fixtureFile = path.join(dir, 'validation-parity.json');
    assert.equal(hash(await readFile(configFile)), identity.browserConfigSha256);
    assert.equal(hash(await readFile(modelFile)), identity.modelSha256);
    assert.equal(hash(await readFile(fixtureFile)), identity.fixtureSha256);
    const config = await read(configFile), fixture = await read(fixtureFile);
    assert.equal(config.output.embeddingName, undefined, 'Centroid browser config cannot request a fictitious embedding output.');
    methods.push({ name, config, fixture, identity, source: { modelUrl: local(modelFile), configUrl: local(configFile) } });
  }
  assert.equal(methods.length, 4);
  const sharedIds = methods[0].fixture.cases.map(c => c.id);
  for (const method of methods) assert.deepEqual(method.fixture.cases.map(c => c.id), sharedIds, 'Every method must use identical validation examples.');
  report.artifacts = methods.map(({ name, identity }) => ({ name, ...identity }));
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
  context.on('request', req => requests.add(req.url()));
  const page = await context.newPage(); page.setDefaultTimeout(150000);
  page.on('pageerror', error => report.errors.push(error.message));
  await page.goto(`${base}/extract.html`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => typeof window.signbridge?.setModelSource === 'function');
  await page.evaluate(async () => { window.openSetMath = await import('/src/openset.ts'); });
  report.runtime = { browser: browser.version(), node: process.version, platform: process.platform, architecture: process.arch,
    backend: 'ONNX Runtime Web WASM CPU single thread', origin: base, sharedDevelopmentServer: true,
    note: 'Sequential measurements on one local computer. Not public-network or mobile performance; no concurrent-load guarantee.' };
  for (const method of methods) {
    const loaded = await page.evaluate(async source => {
      window.signbridge.setModelSource(source); const start = performance.now();
      const status = await window.signbridge.getModelStatus(); return { status, elapsedMs: performance.now() - start };
    }, method.source);
    assert.equal(loaded.status.ready, true, loaded.status.message); assert.equal(loaded.status.modelVersion, method.config.modelVersion);
    for (const fixture of method.fixture.cases) {
      const actual = await page.evaluate(({ x, quality }) => window.signbridge.classifyFeatures(x, quality), fixture);
      const atol = method.fixture.numericAtol, rtol = method.fixture.numericRtol;
      const numeric = {
        numpyLogits: compare(actual.logits, fixture.numpy_logits, atol, rtol, 'Browser vs NumPy'),
        pytorchLogits: compare(actual.logits, fixture.pytorch_logits, atol, rtol, 'Browser vs PyTorch'),
        onnxLogits: compare(actual.logits, fixture.onnx_logits, atol, rtol, 'Browser vs desktop ONNX'),
        diagnostics: compare(diagnosticsVector(actual.diagnostics), diagnosticsVector(fixture.diagnostics), atol, rtol, 'Quality/motion'),
        confidence: compare([actual.confidence], [fixture.confidence], atol, rtol, 'Gate confidence'),
      };
      assert.equal(actual.accepted, fixture.accepted, `${method.name}/${fixture.id}: acceptance`);
      assert.equal(actual.predicted, fixture.predicted, `${method.name}/${fixture.id}: predicted`);
      assert.deepEqual(rejectReasons(actual.reasons), [...fixture.reasons].sort(), `${method.name}/${fixture.id}: exact gate reasons`);
      assert.equal(actual.embedding, undefined);
      report.parity.push({ method: method.name, id: fixture.id, rawLabel: fixture.rawLabel, split: fixture.split,
        numericMaxErrors: numeric, exactDecisionAgreement: true, accepted: actual.accepted, label: actual.label, reasons: rejectReasons(actual.reasons) });
    }
    const timedFixture = [...method.fixture.cases].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)[0];
    const timings = await page.evaluate(async fixture => {
      for (let i = 0; i < 5; i++) await window.signbridge.classifyFeatures(fixture.x, fixture.quality);
      const perCallMs = []; let last;
      for (let i = 0; i < 50; i++) { const start = performance.now(); last = await window.signbridge.classifyFeatures(fixture.x, fixture.quality); perCallMs.push(performance.now() - start); }
      return { perCallMs, accepted: last.accepted, logits: last.logits };
    }, timedFixture);
    compare(timings.logits, timedFixture.numpy_logits, method.fixture.numericAtol, method.fixture.numericRtol, 'Timed real model logits');
    assert.equal(timings.accepted, timedFixture.accepted);
    report.classifierTiming.push({ method: method.name, modelLoadMs: loaded.elapsedMs, fixtureId: timedFixture.id, explicitWarmupCalls: 5,
      priorParityCalls: method.fixture.cases.length, scope: 'Warm features-to-decision roundtrip: main-thread serialization, worker request, ONNX inference, diagnostics and frozen gate. Excludes decoding/landmarks/model loading.',
      perCallMs: timings.perCallMs, ...stats(timings.perCallMs) });
    console.log(`Validation ${method.name}: ${method.fixture.cases.length} parity cases; warm classifier median ${stats(timings.perCallMs).medianMs.toFixed(3)} ms`);
    await save();
  }
  if (!args.includes('--skip-videos')) {
    const regressionPath = path.join(root, 'reports/v3/centroid-exposed-regression.json');
    const regression = await read(regressionPath);
    assert.equal(regression.fit_lock_sha256, exported.fitLockSha256);
    report.regressionReportSha256 = hash(await readFile(regressionPath));
    const mappingPath = path.join(root, 'data/open-set-v2/clips/mapping.json'), mapping = await read(mappingPath);
    report.mappingSha256 = hash(await readFile(mappingPath));
    const clips = Object.entries(mapping.clips).map(([id, c]) => ({ id, ...c })).sort((a, b) => a.source_path < b.source_path ? -1 : a.source_path > b.source_path ? 1 : 0);
    assert.equal(clips.length, 12);
    for (const clip of clips) assert.equal(hash(await readFile(path.resolve(root, clip.path))), clip.mp4_sha256);
    for (const method of methods) {
      await page.evaluate(source => window.signbridge.setModelSource(source), method.source);
      for (const clip of clips) {
        const prediction = await page.evaluate(async src => {
          const video = document.createElement('video'); video.muted = true; video.playsInline = true; document.body.append(video);
          try {
            await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = () => reject(new Error('Real regression clip decode failed')); video.src = src; video.load(); });
            return await window.signbridge.analyzeVideo(video);
          } finally { video.pause(); video.removeAttribute('src'); video.load(); video.remove(); }
        }, local(path.resolve(root, clip.path)));
        assert.equal(prediction.frames, Math.min(48, clip.frame_count));
        assert.equal(prediction.modelVersion, method.config.modelVersion); assert.ok(prediction.elapsedMs > 0);
        assert.ok(Number.isFinite(prediction.confidence));
        report.videos.push({ method: method.name, id: clip.id, rawLabel: clip.raw_label, trueClass: clip.true_class, isKnown: clip.is_known, signer: clip.signer_id,
          role: 'previously_exposed_V2_MP4_regression_only', mp4Sha256: clip.mp4_sha256, ...prediction,
          acceptedCorrect: clip.is_known ? prediction.accepted && prediction.label === clip.true_class : null,
          unknownFalseAccept: clip.is_known ? null : prediction.accepted });
        console.log(`MP4 ${method.name} ${clip.id}: ${prediction.label ?? 'REJECT'} (${prediction.elapsedMs.toFixed(0)} ms)`);
        await save();
      }
    }
  }
  report.summary = Object.fromEntries(methods.map(method => {
    const rows = report.videos.filter(row => row.method === method.name), known = rows.filter(row => row.isKnown), unknown = rows.filter(row => !row.isKnown);
    return [method.name, { parityCases: report.parity.filter(p => p.method === method.name).length,
      mp4KnownN: known.length, mp4AcceptedKnownCorrect: known.filter(row => row.acceptedCorrect).length, mp4RejectedKnown: known.filter(row => !row.accepted).length,
      mp4UnknownN: unknown.length, mp4FalseAcceptedUnknown: unknown.filter(row => row.accepted).length,
      fullPipelineTiming: { ...stats(rows.map(row => row.elapsedMs)), scope: 'Fresh landmark worker per clip; includes warmup, video seeks/snapshotting, feature extraction, model loading/inference and gate. Local cached static assets.' },
      warmClassifierTiming: report.classifierTiming.find(row => row.method === method.name) }];
  }));
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) { failure = error; report.passed = false; report.failure = error.stack ?? String(error); }
finally {
  report.appHttpOrigins = [...new Set([...requests].filter(url => /^https?:/.test(url)).map(url => new URL(url).origin))];
  await save(); await browser?.close();
}
console.log(JSON.stringify({ passed: report.passed, parity: report.parity.length, videos: report.videos.length, failure: failure?.message, report: path.join(output, 'browser-comparison.json') }));
if (failure) process.exitCode = 1;
