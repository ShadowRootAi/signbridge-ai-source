import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';

const [origin, mode, manifestPath, output] = process.argv.slice(2);
assert(origin && ['sequence', 'teach'].includes(mode) && manifestPath && output);
const manifestBytes = await readFile(manifestPath);
const manifest = JSON.parse(manifestBytes);
const clips = manifest.clips.map(row => ({ ...row, file: resolve(dirname(manifestPath), row.path) }));
assert(clips.length > 0, 'Need at least one evaluation clip');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
for (const clip of clips) assert.equal(hash(await readFile(clip.file)), clip.mp4Sha256 ?? clip.sha256, clip.id);
const report = { passed: false, origin, mode, scope: manifest.scope, manifestSha256: hash(manifestBytes), rows: [], external: [], writes: [], errors: [] };
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-background-timer-throttling'] });
const page = await browser.newPage();
const base = new URL(origin);
page.on('request', request => {
  if (!/^https?:/.test(request.url())) return;
  if (new URL(request.url()).origin !== base.origin) report.external.push(request.url());
  if (!['GET', 'HEAD'].includes(request.method())) report.writes.push(request.method());
});
page.on('pageerror', error => report.errors.push(error.message));

const norm = text => text.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase('el').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
function edits(reference, predicted) {
  let row = Array.from({ length: predicted.length + 1 }, (_, index) => index);
  for (const [index, token] of reference.entries()) {
    const next = [index + 1];
    for (const [column, candidate] of predicted.entries()) {
      next.push(Math.min(row[column + 1] + 1, next[column] + 1, row[column] + Number(candidate !== token)));
    }
    row = next;
  }
  return row.at(-1);
}

async function sequence() {
  await page.addInitScript(() => {
    const StandardWorker = window.Worker;
    window.Worker = class extends StandardWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (Array.isArray(data?.result?.segments)) window.qaSequence = data.result;
        });
      }
    };
  });
  await page.goto(base.origin + '/' + base.search + '#/sequence');
  const choose = page.getByLabel('Επιλογή συνεχούς βίντεο', { exact: true });
  const analyze = page.getByRole('button', { name: 'Ανάλυση φράσης', exact: true });
  for (const clip of clips) {
    await page.evaluate(() => { window.qaSequence = null; });
    await choose.setInputFiles(clip.file);
    await analyze.click({ timeout: 25_000 });
    await page.waitForFunction(() => window.qaSequence || document.querySelector('.recognition-error'), undefined, { timeout: 95_000 });
    const prediction = await page.evaluate(() => window.qaSequence);
    assert(prediction, (await page.locator('.recognition-error').allTextContents()).join(' '));
    await page.locator(prediction.text ? '#sequence-text' : '#sequence-review').waitFor();
    const displayed = await page.locator('#sequence-text').allTextContents();
    assert.equal(displayed[0] ?? null, prediction.text, 'UI and model must show the same phrase');
    if (!prediction.text) await page.locator('#sequence-review').waitFor();
    const duration = await page.locator('#sequence-video').evaluate(video => video.duration * 1000);
    assert(prediction.segments.every(segment => segment.startMs >= 0 && segment.endMs > segment.startMs && segment.endMs <= duration + 1));
    assert(prediction.segments.every((segment, index) => index === 0 || segment.startMs >= prediction.segments[index - 1].endMs - .001));
    const correct = Boolean(prediction.text && norm(clip.referenceText) === norm(prediction.text));
    const row = {
      id: clip.id, group: clip.group, signer: clip.signer, reference: clip.referenceText, referenceGlosses: clip.referenceGlosses,
      ...prediction, errors: edits(clip.referenceGlosses, prediction.glosses), exactGloss: JSON.stringify(clip.referenceGlosses) === JSON.stringify(prediction.glosses),
      correct,
    };
    report.rows.push(row);
    console.log(clip.id, clip.group, JSON.stringify({ text: prediction.text, correct, edits: row.errors, segments: prediction.segments.length }));
  }
  const summarize = items => ({
    clips: items.length, correctText: items.filter(row => row.correct).length,
    wrongText: items.filter(row => row.text && !row.correct).length,
    review: items.filter(row => !row.text).length,
    exactGloss: items.filter(row => row.exactGloss).length,
    glossWer: items.reduce((n, row) => n + row.errors, 0) / items.reduce((n, row) => n + row.referenceGlosses.length, 0),
  });
  report.summary = summarize(report.rows);
  report.byGroup = Object.fromEntries([...new Set(report.rows.map(row => row.group))].map(group =>
    [group, summarize(report.rows.filter(row => row.group === group))]));
  if (manifest.sequenceScope !== 'off_scope') assert(report.summary.correctText > 0 && report.summary.review > 0);
  const newClip = report.rows.find(row => !row.text);
  if (newClip) assert(newClip.segments.length > 0, 'Abstention must still expose a timeline');
}

async function teach() {
  const label = manifest.displayLabel ?? 'Εντάξει';
  const path = base.origin + '/' + base.search + '#/teach';
  const enrollment = clips.filter(row => row.role === 'enrollment');
  const positive = clips.filter(row => row.role === 'positive_check');
  const [first, second, third] = clips.filter(row => row.role === 'negative_check');
  const held = clips.filter(row => row.role === 'final_replay');
  await page.goto(path);
  await page.getByLabel('Σημασία του νοήματος').fill(label);
  await page.getByLabel('3 λήψεις του νοήματος για εκμάθηση').setInputFiles(enrollment.map(row => row.file));
  await page.getByLabel('2 νέες λήψεις του ίδιου νοήματος για έλεγχο').setInputFiles(positive.map(row => row.file));
  await page.getByLabel('2 λήψεις διαφορετικών νοημάτων για έλεγχο').setInputFiles([first.file, second.file]);
  await page.getByRole('button', { name: 'Έλεγχος και αποθήκευση', exact: true }).click();
  await page.getByText(/αποθηκεύτηκε αφού πέρασε|χρειάζεται νέα παραδείγματα/).or(page.getByRole('alert')).waitFor({ timeout: 150_000 });
  report.enrollment = await page.locator('[role=status], [role=alert]').allTextContents();
  report.saved = await page.getByRole('button', { name: 'Διαγραφή ' + label, exact: true }).count() === 1;
  if (typeof manifest.expectEnrollment === 'boolean') assert.equal(report.saved, manifest.expectEnrollment);
  if (!report.saved) return;
  await page.reload();
  await page.getByRole('button', { name: 'Διαγραφή ' + label, exact: true }).waitFor();
  report.persisted = true;
  for (const row of [...held, third]) {
    await page.getByLabel('Δοκίμασε νέο προσωπικό νόημα').setInputFiles(row.file);
    await page.locator('.teach-result').or(page.getByRole('alert')).waitFor({ timeout: 55_000 });
    const error = await page.getByRole('alert').allTextContents();
    assert.equal(error.length, 0, error.join(' '));
    const matched = (await page.locator('.teach-result').innerText()) === label;
    report.rows.push({ id: row.id, role: row.role, expectedMatch: row.role === 'final_replay', matched });
  }
  if (manifest.expectEnrollment === true) assert(report.rows.every(row => row.expectedMatch === row.matched), 'Personal sign replay mismatch');
  await page.getByRole('button', { name: 'Διαγραφή ' + label, exact: true }).click();
  await page.reload();
  await page.getByText('Τα νοήματα που περνούν τον έλεγχο εμφανίζονται εδώ.').waitFor();
  report.removed = true;
}

try {
  await (mode === 'sequence' ? sequence() : teach());
  await page.setViewportSize({ width: 360, height: 800 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Mobile horizontal overflow');
  assert.deepEqual(report.external, []);
  assert.deepEqual(report.writes, []);
  assert.deepEqual(report.errors, []);
  report.passed = true;
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser.close();
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ output, passed: report.passed, summary: report.summary, groups: report.byGroup, saved: report.saved, teach: mode === 'teach' ? report.rows : undefined }));
}

