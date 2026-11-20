/** Actual Chrome/WASM branch-only parity; no product import or release changes. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const artifact = resolve(root, 'artifacts/isolated-cross-signer-metric');
const output = resolve(root, 'reports/isolated-cross-signer-metric/browser-onnx-parity.json');
const origin = 'https://signbridge-metric-qa.invalid';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = {
  passed: false, actualChrome: false, executionProvider: 'wasm', numThreads: 1,
  scope: 'Real ORT-Web execution of learned evidence branch on12 development tensors; anchor logits come from frozen Python fixtures.',
  productChanged: false, rawVideoExtractionTested: false, fullModelBrowserTested: false,
  unseenSignRejectionTested: false, cases: [], pageErrors: [], blockedRequests: [],
};
let browser;
let failure;

function compare(actual, expected, tolerance, name) {
  assert.equal(actual.length, expected.length, `${name} length`);
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `${name} finite`);
    const error = Math.abs(actual[i] - expected[i]);
    maximum = Math.max(maximum, error);
    assert.ok(error <= tolerance, `${name} index${i} error${error}`);
  }
  return maximum;
}

function decision(values, config) {
  const order = values.map((value, i) => [value, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  return { index: order[0][1], accepted: order[0][0] >= config.policy.minimum_similarity
    && order[0][0] - order[1][0] >= config.policy.minimum_margin };
}

try {
  const [model, configBytes, fixtureBytes, packageBytes] = await Promise.all([
    readFile(resolve(artifact, 'metric-evidence.onnx')),
    readFile(resolve(artifact, 'metric-evidence-config.json')),
    readFile(resolve(artifact, 'metric-evidence-fixtures.json')),
    readFile(resolve(root, 'web/node_modules/onnxruntime-web/package.json')),
  ]);
  const config = JSON.parse(configBytes), fixtures = JSON.parse(fixtureBytes);
  assert.equal(config.model_sha256, digest(model));
  assert.equal(JSON.parse(packageBytes).version, '1.31.1');
  assert.deepEqual(config.input.shape, ['batch', 16, 189]);
  assert.equal(fixtures.cases.length, 12);
  const routes = new Map();
  const runtimeHashes = {};
  for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
    const bytes = await readFile(resolve(root, 'web/node_modules/onnxruntime-web/dist', name));
    routes.set(`/runtime/${name}`, { body: bytes, type: name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' });
    runtimeHashes[name] = digest(bytes);
  }
  routes.set('/model.onnx', { body: model, type: 'application/octet-stream' });
  routes.set('/', { body: '<!doctype html><html lang="en"><title>Research metric evidence parity</title><body><main>Isolated numerical harness; not the SignBridge product.</main></body></html>', type: 'text/html' });
  report.provenance = { modelSha256: digest(model), configSha256: digest(configBytes), fixturesSha256: digest(fixtureBytes),
    scriptSha256: digest(await readFile(fileURLToPath(import.meta.url))), runtimeVersion: '1.31.1', runtimeHashes,
    transport: 'Playwright fulfills exact local asset bytes at private intercepted origin; no prediction mocking or live server' };
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.actualChrome = true;
  report.browserVersion = browser.version();
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const resource = url.origin === origin && request.method() === 'GET' ? routes.get(url.pathname) : undefined;
    if (!resource) {
      report.blockedRequests.push({ url: request.url(), method: request.method() });
      return route.abort('blockedbyclient');
    }
    return route.fulfill({ status: 200, body: resource.body, contentType: resource.type,
      headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:" } });
  });
  const page = await context.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.goto(origin, { waitUntil: 'load', timeout: 15000 });
  report.initialization = await page.evaluate(async () => {
    const ort = await import('/runtime/ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = '/runtime/';
    const bytes = new Uint8Array(await (await fetch('/model.onnx')).arrayBuffer());
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
    const start = performance.now();
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    window.metricQA = { ort, session };
    return { modelSha256: hash, sessionMs: performance.now() - start,
      inputNames: session.inputNames, outputNames: session.outputNames };
  });
  assert.equal(report.initialization.modelSha256, digest(model));
  assert.deepEqual(report.initialization.inputNames, ['sequence']);
  assert.deepEqual(report.initialization.outputNames, ['metric_cosines']);
  for (const fixture of fixtures.cases) {
    assert.equal(fixture.sequence.length, 16);
    assert.ok(fixture.sequence.every(row => row.length === 189 && row.every(Number.isFinite)));
    const result = await page.evaluate(async ({ fixture, weight }) => {
      const { ort, session } = window.metricQA;
      const start = performance.now();
      const outputs = await session.run({ sequence: new ort.Tensor('float32', Float32Array.from(fixture.sequence.flat()), [1, 16, 189]) });
      const cosine = Array.from(outputs.metric_cosines.data);
      const mean = cosine.reduce((sum, value) => sum + value, 0) / cosine.length;
      const deviation = Math.max(.02, Math.sqrt(cosine.reduce((sum, value) => sum + (value - mean) ** 2, 0) / cosine.length));
      const logits = cosine.map((value, i) => fixture.anchor_logits[i] + weight * (value - mean) / deviation);
      const peak = Math.max(...logits);
      const exp = logits.map(value => Math.exp(value - peak));
      const total = exp.reduce((sum, value) => sum + value, 0);
      return { cosine, probability: exp.map(value => value / total), elapsedMs: performance.now() - start };
    }, { fixture, weight: config.mixture_weight });
    const cosineError = compare(result.cosine, fixture.metric_cosines, 3e-6, fixture.id + ' cosine');
    const probabilityError = compare(result.probability, fixture.fused_probabilities, 3e-6, fixture.id + ' probability');
    const actual = decision(result.probability, config), expected = decision(fixture.fused_probabilities, config);
    assert.deepEqual(actual, expected, fixture.id + ' exact winner/acceptance');
    report.cases.push({ id: fixture.id, reference: fixture.reference, predicted: config.labels[actual.index],
      accepted: actual.accepted, cosineError, probabilityError, elapsedMs: result.elapsedMs });
  }
  await page.evaluate(async () => { await window.metricQA.session.release(); delete window.metricQA; });
  assert.equal(report.pageErrors.length, 0);
  assert.equal(report.blockedRequests.length, 0, 'No unexpected page requests');
  report.maximumCosineError = Math.max(...report.cases.map(row => row.cosineError));
  report.maximumProbabilityError = Math.max(...report.cases.map(row => row.probabilityError));
  report.passed = true;
} catch (error) {
  failure = error;
  report.failure = String(error.stack || error);
} finally {
  await browser?.close();
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ passed: report.passed, cases: report.cases.length, browser: report.browserVersion,
  maxCosineError: report.maximumCosineError, maxProbabilityError: report.maximumProbabilityError, output }));
if (failure) throw failure;
