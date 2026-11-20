import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const origin = new URL(process.argv[2] ?? 'http://127.0.0.1:5191').origin;
const fixtures = resolve(project, 'reports/v8/app-ui-qa/fixtures');
const reports = resolve(project, 'reports/pretrained-gsl');
const localOrigin = ['localhost', '127.0.0.1'].includes(new URL(origin).hostname);
const releaseStage = localOrigin ? 'staged' : new URL(origin).hostname.startsWith('openhands-preview.') ? 'preview' : 'public';
const output = process.argv[4] ? resolve(process.argv[4]) : resolve(reports, `${releaseStage}-openhands-qa.json`);
const frameOffset = process.argv.slice(5).includes('--frame-offset');
const assetsPath = '/__signbridge/openhands/';
const assets = ['stgcn.onnx', 'stgcn-onnx.json', 'policy.json'];
const expectedPolicy = JSON.parse(await readFile(
  process.argv[3] ? resolve(process.argv[3]) : resolve(reports, 'stgcn-gate-config.json'), 'utf8',
));
const { clips: fixtureManifest } = JSON.parse(await readFile(resolve(fixtures, '../fixtures.json'), 'utf8'));
const outsideRoster = fixtureManifest.find(clip => clip.role === 'hard_unknown');
assert(outsideRoster?.rawLabel && !expectedPolicy.supportedLabels.includes(outsideRoster.rawLabel), 'The negative fixture must be outside the supported roster');
const report = {
  passed: false, origin, humanWebcamTested: false, model: {}, uploadedClips: [], camera: {},
  remoteRequests: [], writeRequests: [], pageErrors: [], browserAssetRequests: [],
};

async function verifyAssets(context) {
  const [model, metadata, policy] = await Promise.all(assets.map(name =>
    context.request.get(`${origin}${assetsPath}${name}`, { timeout: 30_000 })));
  for (const [i, response] of [model, metadata, policy].entries()) {
    assert.equal(response.status(), 200, assets[i]);
    assert.equal(new URL(response.url()).origin, origin, `${assets[i]} must stay on this origin`);
  }
  assert.match(model.headers()['content-type'] ?? '', /^(?:application|binary)\/(?:octet-stream|(?:x-)?onnx)(?:;|$)/i);
  for (const response of [metadata, policy]) assert.match(response.headers()['content-type'] ?? '', /^application\/json(?:;|$)/i);
  const bytes = await model.body();
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  assert.equal(sha256, expectedPolicy.runtimeModelSha256, 'Unexpected OpenHands ONNX bytes');
  const config = await metadata.json();
  const actualPolicy = await policy.json();
  assert.equal(config.model_sha256, sha256);
  assert.equal(config.source_checkpoint_sha256, expectedPolicy.sourceModelSha256);
  assert.deepEqual(actualPolicy, expectedPolicy, 'Hosted acceptance policy must match the staged policy');
  assert.equal(new Set(config.labels).size, 310);
  assert.equal(new Set(actualPolicy.supportedLabels).size, expectedPolicy.supportedLabels.length);
  assert(actualPolicy.supportedLabels.every(label => config.labels.includes(label)));
  report.model = { bytes: bytes.length, sha256, classes: config.labels.length, vocabulary: actualPolicy.supportedLabels.length };
  if (expectedPolicy.agreementGate) {
    const verifier = await context.request.get(`${origin}${assetsPath}lstm.onnx`);
    assert.equal(verifier.status(), 200);
    assert.match(verifier.headers()['content-type'] ?? '', /^application\/octet-stream(?:;|$)/i);
    const body = await verifier.body();
    const verifierSha256 = createHash('sha256').update(body).digest('hex');
    assert.equal(verifierSha256, expectedPolicy.agreementGate.modelSha256);
    report.model.verifier = { bytes: body.length, sha256: verifierSha256 };
  }
}

function observe(context, page) {
  context.on('request', request => {
    if (!/^https?:/.test(request.url())) return;
    const url = new URL(request.url());
    if (url.origin !== origin) report.remoteRequests.push(url.href);
    if (!['GET', 'HEAD'].includes(request.method())) report.writeRequests.push({ url: url.href, method: request.method() });
    if (url.pathname.startsWith(assetsPath)) report.browserAssetRequests.push(url.pathname.slice(assetsPath.length));
  });
  page.on('pageerror', error => report.pageErrors.push(error.message));
}

async function openRoute(page, route, title) {
  const response = await page.goto(`${origin}/#/${route}`);
  if (response) {
    assert.equal(response.status(), 200);
    assert.match(response.headers()['content-security-policy'] ?? '', /connect-src 'self' blob:/);
    assert.match(response.headers()['permissions-policy'] ?? '', /camera=\(self\).*microphone=\(\)/);
  }
  await page.getByRole('heading', { name: title, level: 1 }).waitFor();
  assert.equal(await page.evaluate(() => window.crossOriginIsolated), true);
  assert.equal(await page.locator('.recognition-workspace .recognition-capture').count(), 1);
  assert.equal(await page.locator('.recognition-workspace .recognition-message').count(), 1);
  await page.getByRole('list', { name: 'Διαθέσιμα νοήματα' })
    .getByText(`${expectedPolicy.supportedLabels.length} μεμονωμένα νοήματα σε δοκιμή`, { exact: true }).waitFor();
  assert.equal(await page.locator('a[href*="openhands-webcam"]').count(), 0, 'Research lab exposed on public app');
}

async function analyze(page) {
  await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click({ timeout: 25_000 });
  const result = page.locator('#recognition-result');
  const error = page.locator('.recognition-error');
  await result.or(error).waitFor({ timeout: 70_000 });
  assert.equal(await error.count(), 0, (await error.allTextContents()).join('\n'));
  const text = (await result.innerText()).trim();
  const accepted = (await result.getAttribute('data-accepted')) === 'true';
  if (!accepted) assert.equal(text, 'Δεν αναγνωρίστηκε');
  return { text, accepted };
}

await mkdir(reports, { recursive: true });
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: [
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${resolve(fixtures, 'real-gsl-simulated-camera.y4m')}`,
  ] });
  report.browser = browser.version();
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, reducedMotion: 'reduce' });
  await verifyAssets(context);
  const page = await context.newPage();
  observe(context, page);

  await openRoute(page, 'try', 'Αναγνώριση από βίντεο');
  for (const [button, expected] of [['γεια', 'Γεια'], ['ευχαριστώ', 'Ευχαριστώ'], ['εντάξει', 'Εντάξει']]) {
    await page.getByRole('button', { name: button, exact: true }).click();
    const result = await analyze(page);
    report.uploadedClips.push({ source: 'catalog', expected, ...result });
    assert(!result.accepted || result.text === expected, `Wrong accepted label for ${button}`);
  }
  assert(report.uploadedClips.some(result => result.accepted), 'No known example produced a correct acceptance');
  if (expectedPolicy.agreementGate?.temporal) {
    assert(report.uploadedClips.some(row => row.expected === 'Εντάξει' && row.accepted && row.text === row.expected),
      'The original Entaxei MP4 must be correctly recovered');
    // With the corrected MP4 decoding this same clip can pass the primary gate.
    if (!frameOffset) assert(report.browserAssetRequests.includes('lstm.onnx'), 'The browser never requested the recovery model');
  }
  for (const asset of assets) assert(report.browserAssetRequests.includes(asset), `Browser did not request ${asset}`);
  assert.equal(report.browserAssetRequests.filter(name => name === 'stgcn.onnx').length, 1, 'Repeated analysis should reuse the loaded ONNX');
  await page.locator('#video-file').setInputFiles(resolve(fixtures, 'hard_unknown.mp4'));
  const unknown = await analyze(page);
  report.uploadedClips.push({ source: 'file', file: 'hard_unknown.mp4', rawLabel: outsideRoster.rawLabel, ...unknown });
  assert.equal(unknown.accepted, false, 'The fixture outside the supported roster should abstain');

  await context.grantPermissions(['camera'], { origin });
  await openRoute(page, 'sign', 'Νόημα σε κείμενο');
  const open = page.getByRole('button', { name: 'Άνοιγμα κάμερας', exact: true });
  assert.equal(await open.isEnabled(), false, 'Camera should require the user consent checkbox');
  assert.equal(await page.locator('#live-camera').evaluate(video => video.srcObject === null), true);
  await page.getByRole('checkbox', { name: /άδεια για καταγραφή εικόνας/i }).check();
  await open.click();
  await page.waitForFunction(() => document.querySelector('#live-camera')?.srcObject?.getVideoTracks().length > 0);
  const stream = await page.locator('#live-camera').evaluateHandle(video => video.srcObject);
  assert.equal(await stream.evaluate(media => media.getAudioTracks().length), 0, 'Capture included audio');
  await page.getByRole('button', { name: 'Καταγραφή', exact: true }).click();
  report.camera = { source: 'Chrome fake camera replaying the repository GSL fixture', ...await analyze(page) };
  report.camera.tracksEnded = await stream.evaluate(media => media.getTracks().every(track => track.readyState === 'ended'));
  assert.equal(report.camera.tracksEnded, true);
  assert.equal(await page.locator('#recognition-video').evaluate(video => video.currentSrc.startsWith('blob:')), true);
  await stream.dispose();
  if (releaseStage === 'public' && !process.argv[4]) await page.screenshot({ path: resolve(reports, 'public-openhands-camera.png') });

  await page.setViewportSize({ width: 390, height: 844 });
  for (const [route, title] of [['sign', 'Νόημα σε κείμενο'], ['try', 'Αναγνώριση από βίντεο']]) {
    await openRoute(page, route, title);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true, `${route} at 390px`);
  }
  for (const path of ['/openhands-webcam.html', '/v8-pilot.html', '/__signbridge/webcam-capture/status']) {
    const response = await context.request.get(`${origin}${path}`);
    assert.equal(response.status(), 404, `Private endpoint exposed: ${path}`);
  }
  assert.deepEqual(report.remoteRequests, []);
  assert.deepEqual(report.writeRequests, []);
  assert.deepEqual(report.pageErrors, []);
  report.passed = true;
  console.log(`Public OpenHands QA passed at ${origin}`);
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser?.close();
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(output);
}
