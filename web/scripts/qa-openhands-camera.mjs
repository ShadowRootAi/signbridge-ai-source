import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const origin = 'http://127.0.0.1:5175';
const report = { humanWebcamTested: false, app: {}, lab: {}, remoteRequests: [], pageErrors: [] };

async function useCamera(label, extraArgs, run) {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: [
    '--use-fake-device-for-media-stream', ...extraArgs,
  ] });
  try {
    const context = await browser.newContext({ acceptDownloads: true });
    await context.grantPermissions(['camera'], { origin });
    const requests = [];
    context.on('request', request => {
      if (!/^https?:/.test(request.url())) return;
      const url = new URL(request.url());
      if (url.origin !== origin) report.remoteRequests.push(url.href);
      else requests.push({ method: request.method(), path: url.pathname });
    });
    const page = await context.newPage();
    const consoleErrors = [];
    page.on('console', event => { if (event.type() === 'error') consoleErrors.push(event.text()); });
    page.on('pageerror', error => report.pageErrors.push({ label, message: error.message }));
    try { await run(page, requests); }
    catch (error) {
      console.error(JSON.stringify({ label, url: page.url(), requests: requests.slice(-18), pageErrors: report.pageErrors, consoleErrors: consoleErrors.slice(-8),
        text: (await page.locator('body').innerText()).slice(-1300) }));
      throw error;
    }
    assert(!requests.some(request => request.method === 'POST'), 'Camera workflow must keep data in the browser');
  } finally {
    await browser.close();
  }
}

const fixture = resolve(root, 'reports/v8/app-ui-qa/fixtures/real-gsl-simulated-camera.y4m');
await useCamera('app', [`--use-file-for-fake-video-capture=${fixture}`], async (page, requests) => {
  await page.goto(`${origin}/?model=openhands#/try`);
  await page.getByRole('link', { name: 'Δοκιμή 120 νοημάτων' }).waitFor();
  const outputs = [];
  const analyzed = async () => {
    const result = page.locator('#recognition-result');
    const error = page.locator('.recognition-error');
    await result.or(error).waitFor({ timeout: 30_000 });
    assert.equal(await error.count(), 0, await error.allTextContents());
    return result;
  };
  for (const label of ['Γεια', 'Ευχαριστώ', 'Εντάξει']) {
    await page.getByRole('button', { name: label.toLocaleLowerCase('el'), exact: true }).click();
    await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click();
    const result = await analyzed();
    outputs.push({ expected: label, output: (await result.innerText()).trim(), accepted: await result.getAttribute('data-accepted') === 'true' });
  }
  assert(outputs.some(row => row.accepted && row.expected === row.output), 'Known clips did not exercise a successful acceptance');
  assert(outputs.every(row => !row.accepted || row.expected === row.output), 'A curated known clip produced a wrong acceptance');
  const loads = requests.filter(request => request.path === '/__signbridge/openhands/stgcn.onnx').length;
  assert.equal(loads, 1, 'Model worker should reuse its verified local model for repeated clips');
  report.app.fileRecognition = outputs;
  report.app.modelLoadsForThreeClips = loads;
  console.log(JSON.stringify({ fileRecognition: outputs, modelLoads: loads }));

  await page.goto(`${origin}/?model=openhands#/sign`);
  await page.getByRole('checkbox', { name: /Δίνω άδεια/ }).check();
  await page.getByRole('button', { name: 'Άνοιγμα κάμερας', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#live-camera')?.srcObject?.getVideoTracks().length > 0);
  await page.evaluate(() => { window.qaTracks = document.querySelector('#live-camera').srcObject.getTracks(); });
  await page.getByRole('button', { name: 'Καταγραφή', exact: true }).click();
  await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click();
  const result = await analyzed();
  assert(await page.evaluate(() => window.qaTracks.every(track => track.readyState === 'ended')));
  report.app.simulatedCamera = { source: 'Existing GSL video via Chrome fake camera', output: (await result.innerText()).trim(), cameraTracksEnded: true };
  console.log(JSON.stringify({ appSimulatedCamera: report.app.simulatedCamera }));
});

await useCamera('lab', [], async page => {
  await page.goto(`${origin}/openhands-webcam.html`);
  await page.locator('#start:enabled').waitFor();
  await page.locator('#start').click();
  assert.match(await page.locator('#progress-label').innerText(), /1 από 144/);
  assert(await page.locator('#live').isHidden(), 'Camera must wait for an explicit user action');
  assert(await page.locator('#summary').isHidden(), 'Scores must stay hidden during the session');
  await page.locator('#skip').click();
  await page.reload();
  await page.waitForFunction(() => /2 από 144/.test(document.querySelector('#progress-label')?.textContent ?? ''));

  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download').click()]);
  const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
  const known = exported.trials.filter(row => row.kind === 'known');
  const outside = exported.trials.filter(row => row.kind === 'held_out_sign');
  const controls = exported.trials.filter(row => row.kind === 'no_sign');
  assert.equal(new Set(known.map(row => row.gloss)).size, 120);
  assert.equal(outside.length, 16);
  assert.equal(controls.length, 8);
  assert(outside.every(row => !known.some(knownRow => knownRow.gloss === row.gloss)));
  assert.equal(exported.summary.skipped, 1);
  report.lab.fullProtocol = { known: known.length, outsideRoster: outside.length, noSign: controls.length, resumedAfterReload: true, exportVerified: true };

  page.once('dialog', dialog => dialog.accept());
  await page.locator('#reset').click();
  await page.locator('#mode').selectOption('controls');
  await page.locator('#start').click();
  await page.locator('#open-camera').click();
  await page.locator('#capture:enabled').waitFor();
  await page.evaluate(() => { window.qaTracks = document.querySelector('#live').srcObject.getTracks(); });
  await page.locator('#capture').click();
  await page.locator('#save-trial').waitFor({ timeout: 60_000 });
  assert(await page.locator('#summary').isHidden());
  await page.locator('#save-trial').click();
  const [negativeDownload] = await Promise.all([page.waitForEvent('download'), page.locator('#download').click()]);
  const negative = JSON.parse(await readFile(await negativeDownload.path(), 'utf8'));
  assert.equal(negative.attempts[0].outcome, 'rejected');
  assert.equal(negative.attempts[0].rejectionCode, 'tracking');
  assert(negative.attempts[0].frames >= 4);
  await page.locator('#close-camera').click();
  assert(await page.evaluate(() => window.qaTracks.every(track => track.readyState === 'ended')));
  report.lab.simulatedNegative = { source: 'Chrome synthetic camera, no person', attempt: negative.attempts[0], cameraTracksEnded: true };
});

assert.deepEqual(report.remoteRequests, []);
assert.deepEqual(report.pageErrors, []);
const output = resolve(root, 'reports/pretrained-gsl/local-camera-qa.json');
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, ...report }));
