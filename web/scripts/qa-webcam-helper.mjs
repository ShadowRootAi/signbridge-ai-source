import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
const base = process.argv[2] ?? 'http://127.0.0.1:5174';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const checks = [];
let failure;
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const beforeResponse = await context.request.get(`${base}/__signbridge/webcam-capture/status`);
  assert.equal(beforeResponse.status(), 200);
  const before = await beforeResponse.json();
  assert.equal(before.localOnly, true);
  await page.goto(`${base}/webcam-check.html`, { waitUntil: 'networkidle' });
  await page.getByText('Έτοιμο για προαιρετική τοπική καταγραφή.', { exact: false }).waitFor();
  assert.equal(await page.locator('#open-camera').isDisabled(), true);
  assert.equal(await page.locator('#record').isDisabled(), true);
  assert.equal(await page.locator('#save').isDisabled(), true);
  assert.equal(await page.evaluate(() => document.querySelector('#live-video').srcObject), null);
  await page.locator('#consent').check();
  assert.equal(await page.locator('#open-camera').isDisabled(), false);
  assert.equal(await page.locator('#record').isDisabled(), true);
  checks.push('Camera is not opened automatically; explicit consent gates camera activation.');
  assert.equal(await page.locator('#trial-kind option').count(), 4);
  for (const value of ['idle', 'ordinary_hand_motion', 'hands_out_of_view', 'position_or_light_change']) {
    await page.locator('#trial-kind').selectOption(value);
    assert.ok((await page.locator('#instructions').innerText()).length > 20);
  }
  checks.push('All four prespecified non-signing observational categories have instructions.');
  const withoutConsent = await context.request.post(`${base}/__signbridge/webcam-capture`, { headers: { Origin: base }, data: { metadata: { trialKind: 'idle' } } });
  assert.equal(withoutConsent.status(), 400);
  const crossOrigin = await context.request.post(`${base}/__signbridge/webcam-capture`, { headers: { Origin: 'https://example.invalid' }, data: {} });
  assert.equal(crossOrigin.status(), 403);
  const invalidVideo = await context.request.post(`${base}/__signbridge/webcam-capture`, { headers: { Origin: base }, data: { metadata: { trialKind: 'idle', consentLocalRecording: true, labelConfirmed: true, noIntentionalSignLanguage: true, requestedDurationMs: 1500, actualDurationMs: 1500 }, mimeType: 'video/webm', videoBase64: 'bm90IGEgdmlkZW8=' } });
  assert.equal(invalidVideo.status(), 400);
  const after = await (await context.request.get(`${base}/__signbridge/webcam-capture/status`)).json();
  assert.equal(after.savedCount, before.savedCount);
  checks.push('Missing consent, external origin, and invalid media are rejected; saved recording count is unchanged.');
  assert.deepEqual(errors, []);
  checks.push('No unhandled page JavaScript errors. No camera, recorder, or valid fixture upload was used.');
  await mkdir('qa-artifacts', { recursive: true });
  await page.screenshot({ path: 'qa-artifacts/webcam-helper.png', fullPage: true });
  await context.close();
} catch (error) { failure = error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) }; process.exitCode = 1; }
finally {
  await browser.close();
  await mkdir('qa-artifacts', { recursive: true });
  const report = { status: failure ? 'failed' : 'passed', checks, failure, humanCapturePerformed: false, validRecordingFixtureUploaded: false };
  await writeFile('qa-artifacts/webcam-helper-qa.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
