import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const { values, positionals: addresses } = parseArgs({ allowPositionals: true,
  options: { 'frame-offset': { type: 'boolean' }, mobile: { type: 'boolean' }, 'output-dir': { type: 'string' } } });
const frameOffset = values['frame-offset'] ?? false;
const mobile = values.mobile ?? false;
assert(addresses.length <= 1, 'Pass at most one loopback preview address');
const output = values['output-dir'] ? resolve(values['output-dir'])
  : resolve(project, 'reports', frameOffset ? 'openhands-engine' : 'judge-robustness');
const reportName = `${frameOffset ? 'judge-panel-frame-offset' : 'judge-panel'}${mobile ? '-mobile' : ''}`;
const origin = new URL(addresses[0] ?? 'http://127.0.0.1:5199').origin;
assert(['localhost', '127.0.0.1'].includes(new URL(origin).hostname), 'The judges’ diagnostic build is local only');

const samples = [
  { name: 'Εντάξει', gate: frameOffset ? 'primary' : 'temporal', accepted: true,
    result: 'Εντάξει', classification: true, tracking: 1, candidate: 'ΕΝΤΑΞΕΙ' },
  { name: 'Παγωμένο καρέ', gate: 'no_motion', result: 'Δεν αναγνωρίστηκε', classification: false, tracking: 1 },
  { name: 'Μαύρη εικόνα', gate: 'tracking', result: 'Δεν αναγνωρίστηκε', classification: false, tracking: 0 },
  { name: 'Εκτός λεξιλογίου', gate: 'outside_roster', result: 'Δεν αναγνωρίστηκε', classification: true,
    tracking: 1, candidate: frameOffset ? 'ΑΚΟΥΩ' : 'ΚΑΤΑΛΑΒΑΙΝΩ' },
];
const percent = new Intl.NumberFormat('el-GR', { style: 'percent', maximumFractionDigits: 1 });
const score = new Intl.NumberFormat('el-GR', { minimumFractionDigits: 6, maximumFractionDigits: 6 });
const report = {
  passed: false, origin, frameOffset, mobile, samples: [], requestsOutsideOrigin: [], writeRequests: [], pageErrors: [],
  scope: 'Four local browser decision traces using prepared MP4s. Mobile mode uses touch and a narrow viewport in desktop Chrome.',
};

await mkdir(output, { recursive: true });
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.browser = browser.version();
  const viewport = mobile ? { width: 390, height: 844 } : { width: 1600, height: 1000 };
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce', isMobile: mobile, hasTouch: mobile });
  report.viewport = viewport;
  context.on('request', request => {
    const url = new URL(request.url());
    if (!/^https?:$/.test(url.protocol)) return;
    if (url.origin !== origin) report.requestsOutsideOrigin.push(url.href);
    if (!['GET', 'HEAD'].includes(request.method())) report.writeRequests.push({ method: request.method(), url: url.href });
  });
  const page = await context.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  // Read worker messages without changing the values or the executed path.
  await page.addInitScript(() => {
    const WorkerImplementation = window.Worker;
    window.judgeQaResults = [];
    window.Worker = class extends WorkerImplementation {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (Array.isArray(data?.result?.topCandidates)) window.judgeQaResults.push(data.result);
        });
      }
    };
  });

  const heading = page.getByRole('heading', { level: 1, name: 'Αναγνώριση από βίντεο' });
  const trace = page.getByRole('region', { name: 'Ίχνος απόφασης AI' });
  const result = page.locator('#recognition-result');
  const status = trace.locator('[data-decision-gate]');
  const scores = trace.locator('details').filter({ has: page.getByText('Βαθμολογίες ST-GCN (softmax)', { exact: true }) });
  const recovery = trace.locator('details').filter({ has: page.getByText('Έλεγχος δεύτερου μοντέλου', { exact: true }) });
  const screenshot = resolve(output, `${reportName}.png`);

  await page.goto(origin + '/#/try');
  await heading.waitFor();
  assert.equal(await trace.count(), 0, 'The standard file route must hide the judges’ trace');
  assert.equal(await page.getByRole('heading', { name: 'Δείγματα επίδειξης' }).count(), 0);
  report.standardRouteClean = true;

  await page.goto(origin + '/?judge=1#/try');
  await heading.waitFor();
  await trace.waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Recognition has horizontal overflow');
  assert.equal(await status.count(), 0, 'No model decision exists before analysis');
  assert.equal(await scores.count(), 0);
  const demoSamples = page.locator('[aria-labelledby="judge-samples-title"]');
  for (const sample of samples) {
    await demoSamples.getByRole('button', { name: sample.name, exact: true }).click();
    await page.getByText(`Παράδειγμα · ${sample.name}`, { exact: true }).waitFor();
    const analyze = page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true });
    await analyze.waitFor();
    assert.equal(await status.count(), 0, 'Selecting another clip clears the previous decision');
    assert.equal(await scores.count(), 0, 'Selecting another clip clears the previous candidates');
    assert.equal(await result.count(), 0, 'Selecting another clip clears the previous recognized text');
    const started = performance.now();
    await analyze.click();
    await status.waitFor({ timeout: 60_000 });
    const elapsedMs = Math.round(performance.now() - started);
    const diagnostics = await page.evaluate(() => window.judgeQaResults.at(-1));
    assert(diagnostics, 'The analysis must emit a genuine worker result');
    assert.equal(diagnostics.rejectionCode ?? diagnostics.acceptedBy, sample.gate);
    assert.equal(diagnostics.classifierExecuted, sample.classification);
    assert.equal(diagnostics.trackingQuality, sample.tracking);
    assert.equal(await status.getAttribute('data-decision-gate'), sample.gate);
    assert.equal(await trace.locator('[data-classifier-executed]').getAttribute('data-classifier-executed'), String(sample.classification));
    assert.equal((await result.innerText()).trim(), sample.result);
    assert.equal(await result.getAttribute('data-accepted'), String(Boolean(sample.accepted)));
    const coverageLine = await trace.locator('dd').first().innerText();
    assert.match(coverageLine, /^[1-9]\d* καρέ/);
    assert(coverageLine.includes(`${percent.format(sample.tracking)} με ώμους και τουλάχιστον ένα χέρι`), coverageLine);
    if (sample.classification) {
      await scores.getByText('Βαθμολογίες ST-GCN (softmax)', { exact: true }).click();
      const [best] = diagnostics.topCandidates;
      const firstCandidate = scores.locator('li').first();
      assert.equal(best.gloss, sample.candidate);
      assert.deepEqual(await firstCandidate.locator('span').allTextContents(), [best.gloss, score.format(best.score)]);
      assert.equal(await scores.locator('li').count(), 3);
    } else {
      assert.equal(await scores.count(), 0, 'Skipped inference must hide model scores');
      assert.deepEqual(diagnostics.topCandidates, []);
    }
    if (sample.gate === 'temporal') {
      const evidence = diagnostics.recoveryEvidence;
      assert(evidence.primaryScore < evidence.primaryThreshold, 'This known clip missed the primary acceptance floor');
      assert.deepEqual(evidence.temporalChecks.map(check => check.factor), [.65, .8]);
      assert(evidence.temporalChecks.every(check => check.frames >= 4 && check.score > 0), 'Both resampled checks ran');
      await recovery.getByText('Έλεγχος δεύτερου μοντέλου', { exact: true }).click();
      const expected = [
        ['Πρώτη βαθμολογία / όριο', `${score.format(evidence.primaryScore)} / ${score.format(evidence.primaryThreshold)}`],
        ['Στήριξη LSTM', score.format(evidence.secondarySupport)],
        ...evidence.temporalChecks.map(check => [`ST-GCN · ${percent.format(check.factor)} · ${check.frames} καρέ`, score.format(check.score)]),
      ];
      assert.deepEqual(await recovery.locator('li').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('span')].map(span => span.textContent))), expected);
    } else {
      assert.equal(await recovery.count(), 0, 'Samples that do not use recovery must not show recovery scores');
    }
    report.samples.push({ name: sample.name, output: sample.result, elapsedMs, ...diagnostics });
    if (sample.accepted) {
      if (sample.gate === 'temporal') await scores.getByText('Βαθμολογίες ST-GCN (softmax)', { exact: true }).click();
      await page.locator('.recognition-workspace').screenshot({ path: screenshot });
      console.log(screenshot);
    }
    if (sample.gate === 'no_motion') {
      await page.locator('.recognition-workspace').screenshot({ path: resolve(output, `${reportName}-frozen.png`) });
    }
  }

  await page.getByRole('button', { name: 'Καθαρισμός', exact: true }).click();
  assert.equal(await status.count(), 0);
  assert.equal(await scores.count(), 0);
  assert.equal(await result.count(), 0);

  if (mobile) {
    const menu = page.getByRole('button', { name: 'Άνοιγμα μενού', exact: true });
    await menu.click();
    assert.equal(await page.getByRole('button', { name: 'Κλείσιμο μενού', exact: true }).getAttribute('aria-expanded'), 'true');
  }
  await page.getByRole('link', { name: 'Βιβλιοθήκη', exact: true }).click();
  const phrase = 'η τηλεόραση έχει βλάβη';
  const search = page.getByRole('searchbox', { name: 'Ελληνική λέξη ή φράση' });
  await page.waitForFunction(() => {
    const heading = document.querySelector('.library-list-heading');
    const badge = heading?.querySelector('span')?.textContent;
    return badge === '3000' || (badge === '3.000+' && heading?.querySelector('[role="status"]')?.textContent === '3.000 βίντεο');
  });
  await search.fill(phrase);
  await search.press('Enter');
  const viewer = page.locator('#library-video');
  await page.waitForFunction(() => document.querySelector('#library-video')?.currentSrc.endsWith('/elementary-glossa-c2-0198.mp4'));
  await page.getByRole('button', { name: 'Αναπαραγωγή από την αρχή', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#library-video')?.currentTime > .1);
  assert.equal((await page.locator('#library-player-heading').innerText()).trim(), 'Η τηλεόραση έχει βλάβη');
  const attribution = page.locator('.library-source');
  await attribution.getByText('Πηγή και άδεια', { exact: true }).click();
  assert.equal(await attribution.getByRole('link', { name: 'Αρχική πηγή' }).getAttribute('href'), 'https://zenodo.org/records/7847052');
  assert.equal(await attribution.getByRole('link', { name: 'Creative Commons Attribution 4.0 International' }).getAttribute('href'), 'https://creativecommons.org/licenses/by/4.0/');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Library has horizontal overflow');
  report.library = { phrase, video: await viewer.getAttribute('src'), playback: true, attribution: 'https://zenodo.org/records/7847052', catalogCount: 3000 };
  assert.deepEqual(report.requestsOutsideOrigin, []);
  assert.deepEqual(report.writeRequests, []);
  assert.deepEqual(report.pageErrors, []);
  report.screenshots = Object.fromEntries(await Promise.all([`${reportName}.png`, `${reportName}-frozen.png`].map(async file => [
    file, createHash('sha256').update(await readFile(resolve(output, file))).digest('hex'),
  ])));
  report.clearResetsTrace = true;
  report.passed = true;
  console.log(`Verified the opt-in judges’ trace and all ${samples.length} browser decisions.`);
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser?.close();
  const evidence = resolve(output, `${reportName}.json`);
  await writeFile(evidence, `${JSON.stringify(report, null, 2)}\n`);
  console.log(evidence);
}
