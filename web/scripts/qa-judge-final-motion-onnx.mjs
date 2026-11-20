/** Compare the research motion head to PyTorch using Chrome ONNX Runtime Web. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const script = fileURLToPath(import.meta.url);
const web = resolve(dirname(script), '..');
const root = dirname(web);
const artifacts = resolve(root, 'artifacts/judge-final-motion');
const fixturePath = resolve(root, '.runtime-tmp/judge-final-rgb/motion-onnx-fixture.json');
const output = resolve(root, 'reports/judge-final/browser-motion-parity.json');
const runtime = resolve(web, 'node_modules/onnxruntime-web');
const origin = 'https://signbridge-judge-motion-qa.invalid';
const inputShape = [1, 2, 32, 27];
const outputShape = [1, 201];
const tolerance = .0002;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const argmax = values => values.reduce((best, value, index) => value > values[best] ? index : best, 0);
const report = { passed: false, actualChrome: false, executionProvider: 'wasm', numThreads: 1, tolerance,
  scope: 'Logit and winning-index comparison on six prepared landmark tensors, using the frozen research motion head. The app is outside this test.',
  inputShape, outputShape, cases: [], blockedRequests: [], pageErrors: [] };

let browser;
let failure;
try {
  const [model, freezeBytes, fixtureBytes, packageBytes] = await Promise.all([
    readFile(resolve(artifacts, 'motion.onnx')),
    readFile(resolve(artifacts, 'freeze.json')),
    readFile(fixturePath),
    readFile(resolve(runtime, 'package.json')),
  ]);
  const { rows } = JSON.parse(fixtureBytes);
  const freeze = JSON.parse(freezeBytes);
  assert.equal(digest(model), freeze.files['motion.onnx'], 'Expected the frozen ONNX bytes');
  assert.equal(freeze.classes.length + 1, outputShape[1], 'Expected 200 classes plus OTHER');
  assert(Array.isArray(rows) && rows.length === 6 && new Set(rows.map(row => row.id)).size === 6, 'Expected six distinct cases');
  const elements = inputShape.reduce((total, size) => total * size, 1);
  for (const row of rows) {
    assert(typeof row.id === 'string' && row.id.trim() && Array.isArray(row.features)
      && row.features.length === elements && row.features.every(Number.isFinite), 'Invalid fixture ID or input tensor');
    assert(Array.isArray(row.logits) && row.logits.length === outputShape[1] && row.logits.every(Number.isFinite),
      `${row.id}: invalid reference logits`);
  }

  const assets = new Map([
    ['/', { body: '<!doctype html><title>SignBridge motion head parity</title>', type: 'text/html' }],
    ['/model.onnx', { body: model, type: 'application/octet-stream' }],
  ]);
  const runtimeHashes = {};
  for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
    const body = await readFile(resolve(runtime, 'dist', name));
    assets.set(`/runtime/${name}`, { body, type: name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' });
    runtimeHashes[name] = digest(body);
  }
  report.provenance = { modelSha256: digest(model), freezeSha256: digest(freezeBytes), fixtureSha256: digest(fixtureBytes),
    scriptSha256: digest(await readFile(script)), runtimeVersion: JSON.parse(packageBytes).version, runtimeHashes,
    transport: 'Only exact local bytes at a private intercepted origin' };

  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.actualChrome = true;
  report.browserVersion = browser.version();
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route('**/*', route => {
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
  const results = await page.evaluate(async ({ rows, shape }) => {
    const ort = await import('/runtime/ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = '/runtime/';
    const response = await fetch('/model.onnx');
    if (!response.ok) throw new Error(`Motion model request failed: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
    const init = performance.now();
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    try {
      const initialization = { modelSha256: hash, elapsedMs: performance.now() - init,
        inputNames: session.inputNames, outputNames: session.outputNames };
      const cases = [];
      for (const { id, features } of rows) {
        const input = new ort.Tensor('float32', Float32Array.from(features), shape);
        const start = performance.now();
        const { logits } = await session.run({ landmarks: input });
        cases.push({ id, dims: logits.dims, elapsedMs: performance.now() - start, values: Array.from(logits.data) });
      }
      return { initialization, cases };
    } finally {
      await session.release();
    }
  }, { rows: rows.map(({ id, features }) => ({ id, features })), shape: inputShape });
  report.initialization = results.initialization;
  assert.equal(report.initialization.modelSha256, digest(model));
  assert.deepEqual(report.initialization.inputNames, ['landmarks']);
  assert.deepEqual(report.initialization.outputNames, ['logits']);
  for (const [index, { id, values, dims, elapsedMs }] of results.cases.entries()) {
    const expected = rows[index];
    assert.equal(id, expected.id);
    assert.deepEqual(dims, outputShape);
    assert(values.length === outputShape[1] && values.every(Number.isFinite));
    const maxAbsError = Math.max(...values.map((value, i) => Math.abs(value - expected.logits[i])));
    assert(maxAbsError <= tolerance, `${id}: maximum logit error ${maxAbsError}`);
    const winner = argmax(values);
    assert.equal(winner, argmax(expected.logits), `${id}: winning class changed`);
    report.cases.push({ id, winner, maxAbsError, elapsedMs });
  }
  assert.equal(report.cases.length, rows.length);
  assert.deepEqual(report.pageErrors, []);
  assert.deepEqual(report.blockedRequests, []);
  const times = report.cases.map(row => row.elapsedMs).sort((a, b) => a - b);
  report.summary = { cases: times.length, maximumAbsError: Math.max(...report.cases.map(row => row.maxAbsError)),
    classifierMedianMs: (times[2] + times[3]) / 2 };
  report.passed = true;
} catch (error) {
  failure = error;
  report.failure = String(error.stack || error);
} finally {
  try { await browser?.close(); }
  finally {
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  }
}
console.log(JSON.stringify({ passed: report.passed, browser: report.browserVersion, ...report.summary }));
if (failure) throw failure;
