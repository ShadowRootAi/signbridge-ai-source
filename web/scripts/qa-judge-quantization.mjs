/** Benchmark two local motion-head graphs in Chrome ONNX Runtime Web. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const script = fileURLToPath(import.meta.url);
const web = resolve(dirname(script), '..');
const root = dirname(web);
const runtime = resolve(web, 'node_modules/onnxruntime-web');
const output = resolve(root, 'reports/judge-ai-development/quantization-browser.json');
const origin = 'https://signbridge-judge-quantization.invalid';
const variants = [
  { id: 'original', path: 'artifacts/judge-final-motion/motion.onnx' },
  { id: 'linearInt8', path: 'artifacts/judge-ai-development/motion-int8-linear.onnx' },
];
const shape = [1, 2, 32, 27];
const outputShape = [1, 201];
const tolerance = .0002;
const warmupRounds = 3;
const measuredRounds = 24;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const winner = values => values.reduce((best, value, index) => value > values[best] ? index : best, 0);
const difference = (left, right) => Math.max(...left.map((value, index) => Math.abs(value - right[index])));
const report = {
  passed: false, actualChrome: false, executionProvider: 'wasm', threads: 1,
  scope: 'Paired motion-head inference on six prepared tensors from real videos. Model-only timing after warm-up and before tensor disposal. These cases measure output agreement, not recognition accuracy.',
  warmupRounds, measuredRounds, cases: [], blockedRequests: [], pageErrors: [],
};

function timing(values) {
  assert(values.length === measuredRounds * 6 && values.every(value => Number.isFinite(value) && value >= 0));
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length / 2;
  return { samples: values.length, medianMs: (sorted[middle - 1] + sorted[middle]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * .95) - 1] };
}

let browser;
let failure;
try {
  const [fixtureBytes, runtimePackage, code, ...models] = await Promise.all([
    readFile(resolve(root, '.runtime-tmp/judge-final-rgb/motion-onnx-fixture.json')),
    readFile(resolve(runtime, 'package.json')),
    readFile(script),
    ...variants.map(({ path }) => readFile(resolve(root, path))),
  ]);
  const fixture = JSON.parse(fixtureBytes);
  const { rows } = fixture;
  assert.equal(digest(models[0]), fixture.model_sha256, 'Original graph must match the fixture SHA-256');
  assert.notEqual(digest(models[0]), digest(models[1]), 'Expected two distinct graphs');
  assert(Array.isArray(rows) && rows.length === 6 && new Set(rows.map(row => row.id)).size === 6,
    'Expected six distinct landmark fixtures');
  const elements = shape.reduce((product, extent) => product * extent, 1);
  for (const { id, features, logits } of rows) {
    assert(typeof id === 'string' && id && Array.isArray(features) && features.length === elements
      && features.every(Number.isFinite), 'Invalid fixture input');
    assert(Array.isArray(logits) && logits.length === outputShape[1] && logits.every(Number.isFinite),
      `${id}: invalid fixture logits`);
  }

  const assets = new Map([['/', { body: '<!doctype html><title>SignBridge quantization QA</title>', type: 'text/html' }]]);
  report.models = Object.fromEntries(variants.map(({ id, path }, index) => {
    const body = models[index];
    assets.set(`/${id}.onnx`, { body, type: 'application/octet-stream' });
    return [id, { path, bytes: body.length, sha256: digest(body) }];
  }));
  const runtimeHashes = {};
  for (const file of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
    const body = await readFile(resolve(runtime, 'dist', file));
    assets.set(`/runtime/${file}`, { body, type: file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' });
    runtimeHashes[file] = digest(body);
  }
  report.provenance = { fixtureSha256: digest(fixtureBytes), scriptSha256: digest(code),
    runtimeVersion: JSON.parse(runtimePackage).version, runtimeHashes,
    transport: 'Exact local assets fulfilled at a private intercepted origin' };

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
  const measured = await page.evaluate(async ({ modelIds, features, shape, warmupRounds, measuredRounds }) => {
    const ort = await import('/runtime/ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = '/runtime/';
    const sessions = [];
    const input = features.map(values => new ort.Tensor('float32', Float32Array.from(values), shape));
    const times = modelIds.map(() => []);
    const hashes = [];
    try {
      for (const id of modelIds) {
        const response = await fetch(`/${id}.onnx`);
        if (!response.ok) throw new Error(`Model ${id} returned ${response.status}`);
        const graph = new Uint8Array(await response.arrayBuffer());
        const hash = await crypto.subtle.digest('SHA-256', graph);
        hashes.push(Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join(''));
        sessions.push(await ort.InferenceSession.create(graph, { executionProviders: ['wasm'] }));
      }
      for (let round = 0; round < warmupRounds + measuredRounds; round++) {
        for (const [index, tensor] of input.entries()) {
          // Equalize order effects while using the same input consecutively for both graphs.
          const first = (round + index) % 2;
          for (const model of [first, 1 - first]) {
            const start = performance.now();
            const { logits } = await sessions[model].run({ landmarks: tensor });
            const elapsed = performance.now() - start;
            if (round >= warmupRounds) times[model].push(elapsed);
            logits.dispose();
          }
        }
      }
      const scores = [];
      for (const tensor of input) {
        const pair = [];
        for (const session of sessions) {
          const { logits } = await session.run({ landmarks: tensor });
          pair.push({ dims: logits.dims, values: Array.from(logits.data) });
          logits.dispose();
        }
        scores.push(pair);
      }
      return { hashes, times, scores };
    } finally {
      for (const tensor of input) tensor.dispose();
      for (const session of sessions) await session.release();
    }
  }, { modelIds: variants.map(({ id }) => id), features: rows.map(row => row.features), shape, warmupRounds, measuredRounds });

  for (const [index, { id }] of variants.entries()) {
    assert.equal(measured.hashes[index], report.models[id].sha256);
    report.models[id].latency = timing(measured.times[index]);
  }
  for (const [index, [original, quantized]] of measured.scores.entries()) {
    const row = rows[index];
    for (const result of [original, quantized]) {
      assert.deepEqual(result.dims, outputShape);
      assert(result.values.length === outputShape[1] && result.values.every(Number.isFinite));
    }
    const originalParity = difference(original.values, row.logits);
    assert(originalParity <= tolerance && winner(original.values) === winner(row.logits),
      `${row.id}: original Chrome output differs from the pinned reference`);
    report.cases.push({ id: row.id, originalWinner: winner(original.values), int8Winner: winner(quantized.values),
      maxAbsLogitDifference: difference(original.values, quantized.values), originalParity });
  }
  assert.equal(report.cases.length, rows.length);
  assert.deepEqual(report.blockedRequests, []);
  assert.deepEqual(report.pageErrors, []);
  const { original, linearInt8 } = report.models;
  report.summary = { cases: rows.length, winnerChanges: report.cases.filter(row => row.originalWinner !== row.int8Winner).length,
    maxAbsLogitDifference: Math.max(...report.cases.map(row => row.maxAbsLogitDifference)),
    savedBytes: original.bytes - linearInt8.bytes, reductionPercent: (1 - linearInt8.bytes / original.bytes) * 100,
    originalMedianMs: original.latency.medianMs, int8MedianMs: linearInt8.latency.medianMs };
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
