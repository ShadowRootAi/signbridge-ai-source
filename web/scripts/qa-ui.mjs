import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const url = process.argv[2] ?? 'http://127.0.0.1:5173';
const checks = [];
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
const page = await context.newPage();
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));

try {
  // Simulate a real missing-file failure, never a successful model/prediction.
  await context.route('**/models/model_config.json', (route) => route.fulfill({ status: 404, body: 'not found' }));
  await context.route('**/reports/technical-report.json', (route) => route.fulfill({ status: 404, body: 'not found' }));
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: 'SignBridge AI', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Νέος έλεγχος', exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: 'Έναρξη δοκιμής', exact: true }).isDisabled(), true);
  assert.equal(await page.locator('.result').count(), 0);
  assert.equal(await page.getByRole('link', { name: 'Πραγματική τεχνική αναφορά (JSON)' }).count(), 0);
  checks.push('Missing trained model disables inference and produces no prediction.');
  checks.push('Missing technical report produces no broken report link.');

  await page.getByRole('button', { name: 'Νέος έλεγχος', exact: true }).click();
  await page.getByRole('button', { name: 'Νέος έλεγχος', exact: true }).waitFor();
  assert.equal(await page.locator('.result').count(), 0);
  checks.push('Retry after a missing model remains an honest unavailable state.');

  assert.equal(await page.locator('#capture-duration').inputValue(), '1000');
  await page.locator('#capture-duration').selectOption('600');
  assert.equal(await page.locator('#capture-duration').inputValue(), '600');
  checks.push('Camera interval defaults to one second and can be changed.');

  await page.locator('#video-file').setInputFiles({ name: 'not-a-video.txt', mimeType: 'text/plain', buffer: Buffer.from('This is not a video.') });
  assert.match(await page.getByRole('alert').innerText(), /Επιλέξτε ένα αρχείο βίντεο/);
  checks.push('Non-video input is rejected with an accessible error.');
  await page.locator('#video-file').setInputFiles({ name: 'broken.mp4', mimeType: 'video/mp4', buffer: Buffer.from('Deliberately invalid video bytes for decoding error test.') });
  await page.getByRole('alert').filter({ hasText: 'δεν μπορεί να διαβάσει' }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).isDisabled(), true);
  checks.push('Undecodable video disables analysis and reports a real decoding error.');

  const notice = await context.request.get(`${url}/NOTICE.txt`);
  assert.equal(notice.ok(), true);
  assert.match(await notice.text(), /10\.5281\/zenodo\.4756317/);
  checks.push('Attribution notice is served and identifies the source dataset record.');

  await mkdir('qa-artifacts', { recursive: true });
  await page.screenshot({ path: 'qa-artifacts/ui-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.screenshot({ path: 'qa-artifacts/ui-mobile.png', fullPage: true });
  checks.push('390px layout has no horizontal overflow.');
  assert.deepEqual(errors, []);
  checks.push('No unhandled page JavaScript errors during these checks.');
  await writeFile('qa-artifacts/ui-smoke.json', JSON.stringify({ status: 'passed', checks, limitations: ['No fake camera or successful prediction was introduced.', 'This checks UI failure handling, not model accuracy, camera performance, or signer generalization.'] }, null, 2));
  console.log(JSON.stringify({ status: 'passed', checks }, null, 2));
} finally {
  await context.close();
  await browser.close();
}
