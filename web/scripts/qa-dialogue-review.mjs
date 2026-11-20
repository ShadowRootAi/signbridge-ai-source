import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';

const [target, fixture, output] = process.argv.slice(2);
assert(target && fixture && output, 'Usage: qa-dialogue-review.mjs <origin> <fixture.json> <report.json>');
const origin = new URL(target);
const source = JSON.parse(await readFile(fixture, 'utf8'));
const report = { scope: source.scope, origin: origin.origin, turns: [], externalRequests: [], writes: [], passed: false };
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-background-timer-throttling'] });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
page.on('request', request => {
  if (!/^https?:/.test(request.url())) return;
  if (new URL(request.url()).origin !== origin.origin) report.externalRequests.push(request.url());
  if (!['GET', 'HEAD'].includes(request.method())) report.writes.push(request.method());
});
try {
  await page.goto(origin.origin + '/' + origin.search + '#/sequence');
  await page.getByLabel('Επιλογή συνεχούς βίντεο', { exact: true }).setInputFiles(resolve(dirname(fixture), source.file));
  const player = page.locator('#sequence-video');
  const analyze = page.getByRole('button', { name: 'Ανάλυση φράσης', exact: true });
  const confirm = page.getByRole('button', { name: 'Επιβεβαίωση ατάκας', exact: true });
  const transcript = page.getByRole('list', { name: 'Επιβεβαιωμένες ατάκες', exact: true });
  const start = page.getByLabel('Αρχή σε δευτερόλεπτα', { exact: true });
  const end = page.getByLabel('Τέλος σε δευτερόλεπτα', { exact: true });
  await page.getByText('Χρονικά όρια αποσπάσματος', { exact: true }).click();
  for (const [index, turn] of source.turns.toReversed().entries()) {
    await start.fill(String(turn.startSeconds));
    await end.fill(String(turn.endSeconds));
    await analyze.click();
    await page.locator('#sequence-text, #sequence-review, .recognition-error').waitFor({ timeout: 90_000 });
    const recognized = await page.locator('#sequence-text').allTextContents();
    const errors = await page.locator('.recognition-error').allTextContents();
    assert.deepEqual(errors, []);
    assert.equal(recognized[0], turn.expected, 'The real cropped frames should recover the phrase: ' + turn.id);
    assert.equal(await transcript.getByRole('listitem').count(), index, 'No turn before human confirmation');
    const draft = page.getByLabel('Κείμενο ατάκας', { exact: true });
    assert.equal(await draft.inputValue(), turn.expected);
    await draft.fill(turn.expected + (turn.id === 'poio-provlima' ? ';' : '.'));
    await confirm.click();
    await page.getByText('Η ατάκα προστέθηκε στο κείμενο.', { exact: true }).waitFor();
    report.turns.push({ ...turn, recognized: recognized[0] });
  }
  const transcriptText = await transcript.getByRole('listitem').locator('span').allTextContents();
  assert.deepEqual(transcriptText, source.turns.map(turn => turn.expected + (turn.id === 'poio-provlima' ? ';' : '.')));
  await transcript.getByRole('listitem').last().getByRole('button').first().click();
  const videoTime = await player.evaluate(video => video.currentTime);
  assert(Math.abs(videoTime - source.turns.at(-1).startSeconds) < .02);
  const remove = transcript.getByRole('listitem').last().getByRole('button', { name: /Αφαίρεση ατάκας/ });
  await remove.click();
  assert.equal(await transcript.getByRole('listitem').count(), source.turns.length - 1);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  assert.deepEqual(report.externalRequests, []);
  assert.deepEqual(report.writes, []);
  report.passed = true;
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser.close();
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
}

