/** Verify the built app through its real UI; no extraction harness or fake AI. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(web, '..');
const base = (process.argv[2] ?? 'http://127.0.0.1:5174').replace(/\/$/, '');
const output = path.join(web, 'qa-artifacts');
const config = JSON.parse(await readFile(path.join(root, 'artifacts/config.json'), 'utf8'));
const mapping = JSON.parse(await readFile(path.join(root, 'data/clips/mapping.json'), 'utf8'));
const [sampleId, clip] = Object.entries(mapping.clips).find(([, item]) => item.is_known);
const hash = (value) => createHash('sha256').update(value).digest('hex');
const modelSha256 = hash(await readFile(path.join(root, 'artifacts/model.onnx')));
const checks = [], pageErrors = [], networkErrors = [], failedResponses = [], wasmResponses = [];
const observedRequests = new Set();
let failure, result, reportMetrics, headers, navigatorInfo;
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const runtime = { browserVersion: browser.version(), hostPlatform: process.platform, hostArchitecture: process.arch, nodeVersion: process.version };
try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
  context.on('request', (request) => observedRequests.add(request.url()));
  context.on('requestfailed', (request) => networkErrors.push({ url: request.url(), failure: request.failure() }));
  context.on('response', (response) => {
    if (response.status() >= 400) failedResponses.push({ url: response.url(), status: response.status() });
    if (new URL(response.url()).pathname.endsWith('.wasm')) wasmResponses.push({ url: response.url(), status: response.status(), contentType: response.headers()['content-type'] });
  });
  const page = await context.newPage();
  page.setDefaultTimeout(180000);
  page.on('pageerror', (error) => pageErrors.push(error.message));

  const navigation = await page.goto(base, { waitUntil: 'networkidle' });
  navigatorInfo = await page.evaluate(() => ({ userAgent: navigator.userAgent, platform: navigator.platform }));
  assert.ok(navigation?.ok(), 'Built app must return a successful HTML response.');
  headers = navigation.headers();
  assert.equal(headers['cross-origin-opener-policy'], 'same-origin');
  assert.equal(headers['cross-origin-embedder-policy'], 'require-corp');
  const html = await navigation.text();
  assert.ok(!html.includes('/src/main.tsx') && html.includes('/assets/'), 'HTML must reference a production bundle, not Vite source entry.');
  checks.push('Built HTML loads with COOP/COEP headers and production bundle references.');

  const configuration = await context.request.get(`${base}/models/model_config.json`);
  assert.ok(configuration.ok());
  assert.deepEqual(await configuration.json(), config);
  const model = await context.request.get(`${base}/models/model.onnx`);
  assert.ok(model.ok());
  assert.equal(hash(await model.body()), modelSha256);
  const technicalReport = await context.request.get(`${base}/reports/technical-report.json`);
  assert.ok(technicalReport.ok());
  const technical = await technicalReport.json();
  reportMetrics = technical.metrics?.test;
  assert.ok(reportMetrics && Number.isInteger(reportMetrics.unknown_n), 'Technical report must contain actual test counts.');
  checks.push('Production model and config match the trained artifacts; actual technical report is served.');

  await page.getByText('Το εκπαιδευμένο ONNX μοντέλο φορτώθηκε τοπικά.', { exact: true }).waitFor();
  await page.getByRole('link', { name: 'Πραγματική τεχνική αναφορά (JSON)' }).waitFor();
  const expectedFailureCount = `${reportMetrics.unknown_false_acceptance_n}/${reportMetrics.unknown_n}`;
  await page.getByText(`Στο ανεξάρτητο test έγιναν ${expectedFailureCount} λανθασμένες αποδοχές άγνωστων νοημάτων.`, { exact: true }).waitFor();
  checks.push('UI displays the unknown-sign failure counts derived from the actual report.');

  await page.locator('#video-file').setInputFiles(path.resolve(root, clip.path));
  await page.waitForFunction(() => {
    const button = Array.from(document.querySelectorAll('button')).find((element) => element.textContent === 'Ανάλυση βίντεο');
    return button && !button.disabled;
  });
  await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click();
  await page.locator('.result').waitFor();
  assert.equal(await page.getByRole('alert').count(), 0);
  result = await page.locator('.result').evaluate((element) => ({
    accepted: element.classList.contains('accepted'),
    title: element.querySelector('.result-title')?.textContent,
    rendered: element.textContent,
    metrics: Object.fromEntries(Array.from(element.querySelectorAll('.metrics > div')).map((item) => [item.querySelector('dt')?.textContent, item.querySelector('dd')?.textContent])),
  }));
  assert.equal(Number(result.metrics['Επεξεργασμένα καρέ']), Math.min(48, clip.frame_count));
  checks.push('One preselected real held-out MP4 completed through the production UI and bundled worker/WASM. Prediction correctness was not forced.');
  await mkdir(output, { recursive: true });
  await page.screenshot({ path: path.join(output, 'production-ui.png'), fullPage: true });

  await page.locator('#video-file').setInputFiles({ name: 'invalid-production-check.mp4', mimeType: 'video/mp4', buffer: Buffer.from('Deliberately invalid media for a decoding-error check.') });
  await page.getByRole('alert').filter({ hasText: 'δεν μπορεί να διαβάσει' }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).isDisabled(), true);
  checks.push('Undecodable input is rejected and cannot trigger recognition.');

  assert.ok(wasmResponses.length > 0, 'Production inference must actually load WASM assets.');
  for (const response of wasmResponses) {
    assert.equal(response.status, 200);
    assert.match(response.contentType ?? '', /application\/wasm/);
  }
  assert.deepEqual(pageErrors, []);
  assert.deepEqual(networkErrors.filter((entry) => /^https?:/.test(entry.url)), []);
  assert.deepEqual(failedResponses.filter((entry) => /^https?:/.test(entry.url)), []);
  const localOrigin = new URL(base).origin;
  assert.deepEqual([...observedRequests].filter((url) => /^https?:/.test(url) && new URL(url).origin !== localOrigin), []);
  checks.push('WASM assets have correct MIME types; no observed app HTTP/page errors or off-origin requests.');
  await context.close();
} catch (error) {
  failure = error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) };
  process.exitCode = 1;
} finally {
  await browser.close();
  await mkdir(output, { recursive: true });
  const report = {
    status: failure ? 'failed' : 'passed', failure, modelVersion: config.modelVersion, modelSha256,
    runtime: { ...runtime, navigator: navigatorInfo }, checks, sample_id: sampleId, true_class: clip.true_class, result, test_metrics: reportMetrics,
    headers, wasmResponses, pageErrors, networkErrors, failedResponses, observed_application_requests: [...observedRequests],
    limitations: ['Single real upload verifies production integration, not dataset accuracy.', 'No human webcam session was available or tested.', 'No fake model responses or successful recognition mocks were used.'],
  };
  await writeFile(path.join(output, 'production-qa.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, modelVersion: report.modelVersion, checks, failure, result }, null, 2));
}
