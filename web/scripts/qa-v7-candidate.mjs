/**
 * Isolated V7 browser numerical/guard QA. Never imports or changes the product.
 * From project root: node web/scripts/qa-v7-candidate.mjs
 * Options --model --config --fixtures --output are project-relative or absolute.
 *
 * Fixture contract:
 * {modelSha256,configSha256,featureSchema,T:48,F:224,atol?:1e-5,rtol?:1e-4,
 *  cases:[{id,sourceId,split,origin:'original'|'fused',classId:-1|0|1|2,
 *    x:number[48][224],expectedLogits:number[4],
 *    expected:{predictedIndex,accepted,classificationScore,reasons:string[]},
 *    rawFrames?:LandmarkFrame[],expectedTracking?:TrackingDiagnostics}],
 *  rawCases?:[{id,sourceId,split,origin,classId,rawFrames,expectedTracking,
 *    x?,expectedLogits?,expected?}]}
 * Config uses labels/backgroundIndex/input.name/output.name and
 * rejection:{temperature,thresholds:[number,number,number]}.
 * Acceptance is strict best-known>OTHER AND score>=predicted-class threshold.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import { stripTypeScriptTypes } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  if (index < 0) return fallback;
  assert.ok(args[index + 1] && !args[index + 1].startsWith('--'), `${name} requires a value`);
  return args[index + 1];
};
const paths = {
  model: resolve(root, option('--model', 'artifacts/v7/browser/model.onnx')),
  config: resolve(root, option('--config', 'artifacts/v7/browser/model-config.json')),
  fixtures: resolve(root, option('--fixtures', 'artifacts/v7/browser/parity-fixtures.json')),
  output: resolve(root, option('--output', 'reports/v7/browser-candidate-qa.json')),
};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const report = {
  scope: 'Isolated browser ONNX numerical parity and raw tracking guard; selected fixtures are not an accuracy benchmark',
  productChanged: false, productIntegrationTested: false, humanWebcamTested: false,
  actualBrowserUsed: false, passed: false, tensorParity: [], rawPaths: [], structuralGuards: [], pageErrors: [],
};
let browser, server;
let failure;

function compare(actual, expected, atol, rtol, description) {
  assert.equal(actual.length, expected.length, `${description}: size`);
  let maxError = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `${description}: finite ${i}`);
    const error = Math.abs(actual[i] - expected[i]);
    maxError = Math.max(maxError, error);
    assert.ok(error <= atol + rtol * Math.abs(expected[i]), `${description}: index ${i}, error ${error}`);
  }
  return maxError;
}

function summarize(rows) {
  const summary = {};
  for (const origin of ['original', 'fused']) {
    const group = rows.filter(row => row.origin === origin);
    summary[origin] = {
      fixtureN: group.length,
      supportedN: group.filter(row => row.classId >= 0).length,
      unknownN: group.filter(row => row.classId === -1).length,
      classifierExecutedN: group.filter(row => row.classifierExecuted).length,
      blockedBeforeClassifierN: group.filter(row => row.classifierExecuted === false).length,
      acceptedN: group.filter(row => row.accepted).length,
      acceptedCorrectSupportedN: group.filter(row => row.accepted && row.classId >= 0 && row.predictedIndex === row.classId).length,
      acceptedUnknownN: group.filter(row => row.accepted && row.classId === -1).length,
    };
  }
  return summary;
}

try {
  const [modelBytes, configBytes, fixtureBytes, trackingSource, featureSource, ortPackageBytes] = await Promise.all([
    readFile(paths.model), readFile(paths.config), readFile(paths.fixtures),
    readFile(resolve(root, 'web/src/tracking.ts'), 'utf8'), readFile(resolve(root, 'web/src/features.ts'), 'utf8'),
    readFile(resolve(root, 'web/node_modules/onnxruntime-web/package.json')),
  ]);
  const config = JSON.parse(configBytes), fixtures = JSON.parse(fixtureBytes), ortPackage = JSON.parse(ortPackageBytes);
  assert.equal(ortPackage.version, '1.31.1', 'Use the pinned ORT-Web runtime');
  assert.equal(fixtures.modelSha256, sha(modelBytes), 'Fixture/model provenance');
  assert.equal(fixtures.configSha256, sha(configBytes), 'Fixture/config provenance');
  assert.equal(config.featureSchema, 'signbridge-landmarks-xy-v1');
  assert.equal(fixtures.featureSchema, config.featureSchema);
  assert.equal(config.T, 48); assert.equal(config.F, 224);
  assert.equal(fixtures.T, 48); assert.equal(fixtures.F, 224);
  assert.equal(config.labels.length, 3); assert.equal(config.backgroundIndex, 3);
  assert.ok(typeof config.input?.name === 'string' && typeof config.output?.name === 'string');
  assert.ok(Number.isFinite(config.rejection.temperature) && config.rejection.temperature > 0);
  assert.ok(Array.isArray(config.rejection.thresholds) && config.rejection.thresholds.length === 3);
  assert.ok(config.rejection.thresholds.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
  assert.ok(Array.isArray(fixtures.cases) && fixtures.cases.length > 0);
  assert.equal(new Set(fixtures.cases.map(row => row.id)).size, fixtures.cases.length, 'Unique fixture IDs');
  const atol = fixtures.atol ?? 1e-5, rtol = fixtures.rtol ?? 1e-4;
  assert.ok(Number.isFinite(atol) && atol > 0 && Number.isFinite(rtol) && rtol >= 0);
  const runtimeFiles = ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
  const routes = new Map();
  const runtimeHashes = {};
  for (const filename of runtimeFiles) {
    const content = await readFile(resolve(root, 'web/node_modules/onnxruntime-web/dist', filename));
    runtimeHashes[filename] = sha(content);
    routes.set(`/runtime/${filename}`, { content, type: filename.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' });
  }
  routes.set('/tracking.mjs', { content: stripTypeScriptTypes(trackingSource), type: 'text/javascript' });
  routes.set('/features.mjs', { content: stripTypeScriptTypes(featureSource), type: 'text/javascript' });
  routes.set('/model.onnx', { content: modelBytes, type: 'application/octet-stream' });
  routes.set('/', { content: '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Isolated V7 scientific QA</title></head><body><main>Local numerical test harness</main></body></html>', type: 'text/html' });
  report.provenance = { modelSha256: sha(modelBytes), configSha256: sha(configBytes), fixturesSha256: sha(fixtureBytes),
    trackingSourceSha256: sha(trackingSource), featureSourceSha256: sha(featureSource),
    scriptSha256: sha(await readFile(fileURLToPath(import.meta.url))), runtimeVersion: ortPackage.version, runtimeHashes };
  report.modelVersion = config.modelVersion;
  server = createServer((request, response) => {
    const resource = routes.get(new URL(request.url, 'http://127.0.0.1').pathname);
    if (!resource || !['GET', 'HEAD'].includes(request.method)) { response.writeHead(404); response.end(); return; }
    response.writeHead(200, { 'Content-Type': resource.type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    response.end(request.method === 'HEAD' ? undefined : resource.content);
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  const address = server.address();
  await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'load', timeout: 30000 });
  const initial = await page.evaluate(async config => {
    const ort = await import('/runtime/ort.wasm.min.mjs');
    const { measureRawTracking, TRACKING_POLICY } = await import('/tracking.mjs');
    const { preprocess } = await import('/features.mjs');
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = '/runtime/';
    const bytes = new Uint8Array(await (await fetch('/model.onnx')).arrayBuffer());
    const modelDigest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    let inferenceCalls = 0;
    const decide = (logits, thresholds = config.rejection.thresholds) => {
      if (logits.length !== 4 || logits.some(v => !Number.isFinite(v))) throw new Error('Invalid model logits');
      const scaled = logits.map(value => value / config.rejection.temperature), maximum = Math.max(...scaled);
      const exponentials = scaled.map(value => Math.exp(value - maximum)), total = exponentials.reduce((a, b) => a + b, 0);
      const probabilities = exponentials.map(value => value / total);
      let predictedIndex = 0;
      for (let index = 1; index < 3; index++) if (probabilities[index] > probabilities[predictedIndex]) predictedIndex = index;
      const classificationScore = probabilities[predictedIndex];
      const reasons = [];
      if (!(classificationScore > probabilities[3])) reasons.push('background');
      if (classificationScore < thresholds[predictedIndex]) reasons.push('confidence');
      return { predictedIndex, classificationScore, probabilities, accepted: reasons.length === 0, reasons,
        thresholdMargin: classificationScore - thresholds[predictedIndex],
        backgroundMargin: classificationScore - probabilities[3],
        label: reasons.length ? null : config.labels[predictedIndex] };
    };
    const infer = async x => {
      if (x.length !== 48 || x.some(row => row.length !== 224 || row.some(v => !Number.isFinite(v)))) throw new Error('Invalid model tensor');
      const start = performance.now();
      inferenceCalls++;
      const outputs = await session.run({ [config.input.name]: new ort.Tensor('float32', Float32Array.from(x.flat()), [1, 48, 224]) });
      const output = outputs[config.output.name];
      if (!output || output.data.length !== 4) throw new Error('Expected four logits');
      const logits = Array.from(output.data);
      return { ...decide(logits), logits, classifierExecuted: true, elapsedMs: performance.now() - start };
    };
    const guarded = async raw => {
      const before = inferenceCalls;
      const tracking = measureRawTracking(raw);
      if (!tracking.usable) return { accepted: false, label: null, classificationScore: null,
        predictedIndex: null, classifierExecuted: false, inferenceCalls: inferenceCalls - before, tracking,
        reasons: tracking.reasons, logits: null };
      const processed = preprocess(raw);
      const result = await infer(processed.x);
      return { ...result, tracking, inferenceCalls: inferenceCalls - before, x: processed.x };
    };
    const boundaryChecks = () => {
      const logits = [3, 1, 0, -2], exact = decide(logits).classificationScore;
      return {
        equality: decide(logits, [exact, exact, exact]),
        thresholdJustAboveScore: decide(logits, [exact + 1e-12, exact + 1e-12, exact + 1e-12]),
        thresholdJustBelowScore: decide(logits, [exact - 1e-12, exact - 1e-12, exact - 1e-12]),
        backgroundTie: decide([0, 0, 0, 0], [.25, .25, .25]),
      };
    };
    window.v7QA = { infer, guarded, boundaryChecks, release: () => session.release() };
    return { modelDigest, inputNames: session.inputNames, outputNames: session.outputNames, policy: TRACKING_POLICY };
  }, config);
  report.actualBrowserUsed = true;
  assert.equal(initial.modelDigest, fixtures.modelSha256, 'Actual browser fetched the intended model');
  assert.ok(initial.inputNames.includes(config.input.name)); assert.ok(initial.outputNames.includes(config.output.name));
  report.trackingPolicy = initial.policy;
  report.boundaryChecks = await page.evaluate(() => window.v7QA.boundaryChecks());
  assert.equal(report.boundaryChecks.equality.accepted, true, 'Threshold equality is accepted');
  assert.equal(report.boundaryChecks.thresholdJustAboveScore.accepted, false);
  assert.deepEqual(report.boundaryChecks.thresholdJustAboveScore.reasons, ['confidence']);
  assert.equal(report.boundaryChecks.thresholdJustBelowScore.accepted, true);
  assert.equal(report.boundaryChecks.backgroundTie.accepted, false, 'Strict OTHER tie must reject');
  assert.deepEqual(report.boundaryChecks.backgroundTie.reasons, ['background']);
  assert.equal(report.boundaryChecks.backgroundTie.predictedIndex, 0, 'First known index wins a tie');
  for (const item of fixtures.cases) {
    assert.ok(['original', 'fused'].includes(item.origin), `${item.id}: explicit feature origin`);
    assert.ok([-1, 0, 1, 2].includes(item.classId), `${item.id}: known/unknown class`);
    assert.ok(item.sourceId && item.split, `${item.id}: provenance`);
    const result = await page.evaluate(x => window.v7QA.infer(x), item.x);
    const maxLogitError = compare(result.logits, item.expectedLogits, atol, rtol, item.id);
    const classificationScoreError = compare([result.classificationScore], [item.expected.classificationScore], atol, rtol, `${item.id}: score`);
    assert.equal(result.predictedIndex, item.expected.predictedIndex, `${item.id}: top known class`);
    assert.equal(result.accepted, item.expected.accepted, `${item.id}: decision`);
    assert.deepEqual([...result.reasons].sort(), [...item.expected.reasons].sort(), `${item.id}: reasons`);
    report.tensorParity.push({ id: item.id, sourceId: item.sourceId, split: item.split, origin: item.origin,
      classId: item.classId, maxLogitError, classificationScoreError, ...result });
  }
  const rawCases = [...fixtures.cases.filter(item => item.rawFrames), ...(fixtures.rawCases ?? [])];
  assert.equal(new Set(rawCases.map(row => row.id)).size, rawCases.length, 'Unique raw fixture IDs');
  for (const item of rawCases) {
      assert.ok(['original', 'fused'].includes(item.origin), `${item.id}: explicit raw origin`);
      assert.ok([-1, 0, 1, 2].includes(item.classId), `${item.id}: raw known/unknown class`);
      assert.ok(item.sourceId && item.split, `${item.id}: raw provenance`);
      assert.ok(item.expectedTracking && item.rawFrames, `${item.id}: raw fixtures need Python tracking expectations`);
      const raw = await page.evaluate(frames => window.v7QA.guarded(frames), item.rawFrames);
      assert.deepEqual(raw.tracking, item.expectedTracking, `${item.id}: Python/browser raw completeness`);
      if (!raw.tracking.usable) {
        assert.equal(raw.classifierExecuted, false); assert.equal(raw.inferenceCalls, 0);
        assert.equal(raw.accepted, false); assert.equal(raw.label, null); assert.equal(raw.classificationScore, null);
      } else {
        assert.equal(raw.classifierExecuted, true); assert.equal(raw.inferenceCalls, 1);
        assert.ok(Number.isFinite(raw.classificationScore) && raw.logits.every(Number.isFinite));
        if (item.x) compare(raw.x.flat(), item.x.flat(), 1e-6, 1e-6, `${item.id}: raw preprocessing`);
        if (item.expectedLogits) compare(raw.logits, item.expectedLogits, atol, rtol, `${item.id}: raw logits`);
        if (item.expected) {
          assert.equal(raw.accepted, item.expected.accepted);
          assert.equal(raw.predictedIndex, item.expected.predictedIndex);
          compare([raw.classificationScore], [item.expected.classificationScore], atol, rtol, `${item.id}: raw score`);
          assert.deepEqual([...raw.reasons].sort(), [...item.expected.reasons].sort());
        }
      }
      delete raw.x;
      report.rawPaths.push({ id: item.id, sourceId: item.sourceId, split: item.split, origin: item.origin, classId: item.classId,
        numericalReferenceProvided: Boolean(item.expectedLogits), ...raw });
  }
  assert.ok(report.tensorParity.some(row => row.classId >= 0), 'Include supported fixtures');
  assert.ok(report.tensorParity.some(row => row.classId === -1), 'Include unknown fixtures');
  assert.ok(report.rawPaths.length > 0, 'Raw guard must actually execute');
  const base = rawCases.find(item => item.rawFrames?.length >= 4);
  assert.ok(base, 'A real raw fixture is required for structural guard controls');
  const structural = [
    { id: 'no_hands', raw: base.rawFrames.map(frame => ({ ...frame, left: [], right: [] })), reason: 'incomplete_required_landmarks' },
    { id: 'too_few_frames', raw: base.rawFrames.slice(0, 3), reason: 'too_few_frames' },
    { id: 'duplicate_timestamps', raw: base.rawFrames.map((frame, index) => ({ ...frame, t: index ? base.rawFrames[0].t : frame.t })), reason: 'invalid_frame_metadata' },
  ];
  for (const item of structural) {
    const actual = await page.evaluate(raw => window.v7QA.guarded(raw), item.raw);
    assert.equal(actual.classifierExecuted, false); assert.equal(actual.inferenceCalls, 0);
    assert.equal(actual.accepted, false); assert.equal(actual.label, null); assert.equal(actual.classificationScore, null);
    assert.ok(actual.reasons.includes(item.reason));
    report.structuralGuards.push({ id: item.id, role: 'synthetic structural control, not an observed sign', ...actual });
  }
  await page.evaluate(() => window.v7QA.release());
  assert.deepEqual(report.pageErrors, []);
  report.tensorFixtureCounts = summarize(report.tensorParity);
  report.rawFixtureCounts = summarize(report.rawPaths);
  report.minimumAbsoluteThresholdMargin = Math.min(...report.tensorParity.map(row => Math.abs(row.thresholdMargin)));
  report.minimumAbsoluteBackgroundMargin = Math.min(...report.tensorParity.map(row => Math.abs(row.backgroundMargin)));
  report.maximumLogitError = Math.max(...report.tensorParity.map(row => row.maxLogitError));
  report.maximumClassificationScoreError = Math.max(...report.tensorParity.map(row => row.classificationScoreError));
  report.numericToleranceNeverRelaxesDecisions = true;
  report.passed = true;
} catch (error) {
  failure = error;
  report.error = { message: error.message, stack: error.stack };
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
  await mkdir(dirname(paths.output), { recursive: true });
  await writeFile(paths.output, `${JSON.stringify(report, null, 2)}\n`);
}
console.log(JSON.stringify({ passed: report.passed, actualBrowserUsed: report.actualBrowserUsed,
  tensorFixtureN: report.tensorParity.length, rawFixtureN: report.rawPaths.length,
  structuralGuardN: report.structuralGuards.length, output: paths.output }));
if (failure) throw failure;
