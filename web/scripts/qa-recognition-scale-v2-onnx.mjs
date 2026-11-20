/** Reproduce frozen held-clip decisions in Chrome using real ONNX Runtime Web. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const [modelPath, fixturePath, reportPath] = process.argv.slice(2).map(path => resolve(path));
assert(modelPath && fixturePath && reportPath,
  'Usage: node web/scripts/qa-recognition-scale-v2-onnx.mjs model.onnx fixture.json output.json');
assert.equal(new Set([modelPath, fixturePath, reportPath]).size, 3, 'Input and output paths must be distinct');
const script = fileURLToPath(import.meta.url);
const runtime = resolve(dirname(script), '../node_modules/onnxruntime-web');
const origin = 'https://signbridge-scale-v2-qa.invalid';
const heads = ['logits', 'temporal', 'motion'];
const tolerance = .0003;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = {
  passed: false, actualChrome: false, executionProvider: 'wasm', numThreads: 1, tolerance,
  scope: 'Chrome ONNX parity on previously extracted landmarks from real held MP4 clips. Video extraction and application integration are outside this check.',
  cases: [], blockedRequests: [], pageErrors: [],
};

function compare(actual, expected, label) {
  assert(actual.length === expected.length && actual.every(Number.isFinite) && expected.every(Number.isFinite), label);
  const maxAbsError = Math.max(...actual.map((value, i) => Math.abs(value - expected[i])));
  assert(maxAbsError <= tolerance, `${label} maximum absolute error ${maxAbsError}`);
  return maxAbsError;
}

let browser;
let failure;
try {
  const [model, fixtureBytes, packageBytes] = await Promise.all([
    readFile(modelPath), readFile(fixturePath), readFile(resolve(runtime, 'package.json')),
  ]);
  const { rows, classes, policy } = JSON.parse(fixtureBytes);
  assert(Array.isArray(rows) && rows.length && new Set(rows.map(row => row.id)).size === rows.length,
    'Fixture must contain unique clips');
  assert(Array.isArray(classes) && classes.length >= 2 && classes.every(label => typeof label === 'string' && label.trim())
    && new Set(classes).size === classes.length, 'Invalid class labels');
  assert(Number.isFinite(policy.temperature) && policy.temperature > 0 && Number.isFinite(policy.weight)
    && policy.weight >= 0 && policy.weight <= 1 && Number.isFinite(policy.confidence_threshold)
    && Number.isFinite(policy.min_quality) && typeof policy.motion_agreement === 'boolean', 'Invalid policy');
  const assets = new Map([
    ['/', { body: '<!doctype html><title>SignBridge scale V2 ONNX parity</title>', type: 'text/html' }],
    ['/model.onnx', { body: model, type: 'application/octet-stream' }],
  ]);
  const runtimeHashes = {};
  for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
    const body = await readFile(resolve(runtime, 'dist', name));
    assets.set(`/runtime/${name}`, { body, type: name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' });
    runtimeHashes[name] = digest(body);
  }
  report.provenance = { modelSha256: digest(model), fixturesSha256: digest(fixtureBytes),
    scriptSha256: digest(await readFile(script)), runtimeVersion: JSON.parse(packageBytes).version, runtimeHashes,
    transport: 'Exact local assets fulfilled at a private intercepted origin' };
  report.policy = policy;
  report.classes = classes;

  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.actualChrome = true;
  report.browserVersion = browser.version();
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const asset = url.origin === origin && request.method() === 'GET' ? assets.get(url.pathname) : undefined;
    if (!asset) {
      report.blockedRequests.push({ url: request.url(), method: request.method() });
      return route.abort('blockedbyclient');
    }
    return route.fulfill({ status: 200, body: asset.body, contentType: asset.type,
      headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:" } });
  });
  const page = await context.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.goto(origin, { timeout: 15000 });
  report.initialization = await page.evaluate(async () => {
    const ort = await import('/runtime/ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = '/runtime/';
    const response = await fetch('/model.onnx');
    if (!response.ok) throw new Error(`Model request failed: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
    const start = performance.now();
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    window.scaleQA = { ort, session };
    return { modelSha256: hash, elapsedMs: performance.now() - start, inputNames: session.inputNames,
      outputNames: session.outputNames };
  });
  assert.equal(report.initialization.modelSha256, digest(model));
  assert.deepEqual(report.initialization.inputNames, ['landmarks']);
  assert.deepEqual(report.initialization.outputNames, heads);

  for (const row of rows) {
    assert(row.landmarks.length === 48 && row.landmarks.every(frame => frame.length === 224 && frame.every(Number.isFinite)), row.id);
    assert(Number.isFinite(row.quality) && row.quality >= 0 && row.quality <= 1, `${row.id} tracking quality`);
    assert(row.expectedLabel === null || classes.includes(row.expectedLabel), `${row.id} reference decision`);
    const result = await page.evaluate(async ({ landmarks, classes, policy, quality }) => {
      const { ort, session } = window.scaleQA;
      const argmax = values => values.reduce((best, value, index) => value > values[best] ? index : best, 0);
      const input = new ort.Tensor('float32', Float32Array.from(landmarks.flat()), [1, 48, 224]);
      const start = performance.now();
      const tensors = await session.run({ landmarks: input });
      const elapsedMs = performance.now() - start;
      const outputs = Object.fromEntries(Object.entries(tensors).map(([name, tensor]) => [name, Array.from(tensor.data)]));
      const count = classes.length;
      const winner = argmax(outputs.logits.slice(0, count));
      const otherWins = argmax(outputs.logits) === count;
      const agree = winner === argmax(outputs.temporal.slice(0, count)) && winner === argmax(outputs.motion.slice(0, count));
      const peak = Math.max(...outputs.logits);
      const likelihood = outputs.logits.map(value => Math.exp((value - peak) / policy.temperature));
      const confidence = likelihood[winner] / likelihood.reduce((sum, value) => sum + value, 0);
      const accepted = !otherWins && quality >= Math.max(.6, policy.min_quality)
        && (!policy.motion_agreement || agree) && confidence >= policy.confidence_threshold;
      return { outputs, confidence, otherWins, agree, decision: accepted ? classes[winner] : null, elapsedMs };
    }, { landmarks: row.landmarks, classes, policy, quality: row.quality });
    const errors = Object.fromEntries(heads.map(name => {
      assert.equal(row[name].length, classes.length + 1, `${row.id} ${name} output includes OTHER`);
      return [name, compare(result.outputs[name], row[name], `${row.id} ${name}`)];
    }));
    const fusion = result.outputs.temporal.map((value, index) => value * (1 - policy.weight) + result.outputs.motion[index] * policy.weight);
    errors.fusion = compare(result.outputs.logits, fusion, `${row.id} branch fusion`);
    assert.equal(result.decision, row.expectedLabel, `${row.id} browser decision differs from Python`);
    report.cases.push({ id: row.id, decision: result.decision, confidence: result.confidence, otherWins: result.otherWins,
      motionAgreement: result.agree, quality: row.quality, elapsedMs: result.elapsedMs, maxAbsErrors: errors });
  }
  await page.evaluate(async () => { await window.scaleQA.session.release(); delete window.scaleQA; });
  assert.deepEqual(report.pageErrors, []);
  assert.deepEqual(report.blockedRequests, []);
  const times = report.cases.map(row => row.elapsedMs).sort((a, b) => a - b);
  const middle = Math.floor(times.length / 2);
  report.summary = { cases: report.cases.length, accepted: report.cases.filter(row => row.decision !== null).length,
    rejected: report.cases.filter(row => row.decision === null).length,
    maximumAbsError: Math.max(...report.cases.flatMap(row => Object.values(row.maxAbsErrors))),
    classifierMedianMs: (times[Math.ceil(times.length / 2) - 1] + times[middle]) / 2,
    classifierP95Ms: times[Math.ceil(.95 * times.length) - 1] };
  report.passed = true;
} catch (error) {
  failure = error;
  report.failure = String(error.stack || error);
} finally {
  try { await browser?.close(); }
  finally {
    await mkdir(dirname(reportPath), { recursive: true });
    await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
  }
}

console.log(JSON.stringify({ passed: report.passed, ...report.summary, browser: report.browserVersion, reportPath }));
if (failure) throw failure;
