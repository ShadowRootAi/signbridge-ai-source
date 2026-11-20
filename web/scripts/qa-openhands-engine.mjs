import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const frameOffset = args.includes('--frame-offset');
const addresses = args.filter(arg => arg !== '--frame-offset');
assert(addresses.length <= 2, 'Pass at most two loopback preview addresses');
const [baselineAddress = 'http://127.0.0.1:5196', candidateAddress = 'http://127.0.0.1:5199'] = addresses;
const baselineOrigin = localOrigin(baselineAddress);
const candidateOrigin = localOrigin(candidateAddress);
assert.notEqual(baselineOrigin, candidateOrigin, 'Use distinct, unmodified baseline and candidate servers');

const reportPath = resolve(project, 'reports/openhands-engine', frameOffset ? 'frame-offset-guards.json' : 'guards.json');
const manifestPath = resolve(project, 'reports/judge-robustness/manifest.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const fixtures = [
  { id: 'no-tracking', gate: 'tracking', label: null, coverage: 0 },
  { id: 'frozen', gate: 'no_motion', label: null, coverage: 1 },
  { id: 'geia', gate: 'primary', label: 'ΓΕΙΑ' },
  { id: 'efcharisto', gate: 'primary', label: 'ΕΥΧΑΡΙΣΤΩ' },
  { id: 'entaxei', gate: 'temporal', label: 'ΕΝΤΑΞΕΙ' },
  { id: 'unknown', gate: 'outside_roster', label: null },
];
const models = {
  primary: '/__signbridge/openhands/stgcn.onnx',
  secondary: '/__signbridge/openhands/lstm.onnx',
};
const report = {
  passed: false,
  scope: 'Paired Chrome file-upload regression on four existing annotated recordings and two explicitly derived controls.',
  baselineOrigin, candidateOrigin, frameOffset,
  manifestSha256: sha256(await readFile(manifestPath)),
};
const runtimeFiles = paths => paths.filter(path => /(?:onnxruntime|ort-wasm).*\.(?:wasm|mjs)$/.test(path));

function localOrigin(address) {
  const url = new URL(address);
  assert.equal(url.protocol, 'http:');
  assert(['127.0.0.1', 'localhost'].includes(url.hostname));
  return url.origin;
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

for (const fixture of fixtures) {
  const clip = manifest.clips.find(clip => clip.id === fixture.id);
  assert(clip, `Missing fixture ${fixture.id}`);
  fixture.path = resolve(dirname(manifestPath), clip.path);
  fixture.sha256 = sha256(await readFile(fixture.path));
  assert.equal(fixture.sha256, clip.sha256, `Unexpected video bytes for ${fixture.id}`);
}

async function scan(browser, origin, target) {
  const details = { rows: [], requests: [], pageErrors: [] };
  const context = await browser.newContext({ reducedMotion: 'reduce' });
  context.on('request', request => {
    const url = new URL(request.url());
    if (/^https?:$/.test(url.protocol)) details.requests.push({ path: url.pathname, origin: url.origin, method: request.method() });
  });
  const page = await context.newPage();
  page.on('pageerror', error => details.pageErrors.push(error.message));
  await page.addInitScript(() => {
    const WorkerImplementation = window.Worker;
    window.engineQaResults = [];
    window.Worker = class extends WorkerImplementation {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (Array.isArray(data?.result?.topCandidates)) window.engineQaResults.push(data.result);
        });
      }
    };
  });
  report[target] = details;
  try {
    await page.goto(`${origin}/#/try`);
    await page.getByRole('heading', { level: 1, name: 'Αναγνώριση από βίντεο' }).waitFor();
    for (const fixture of fixtures) {
      const requestStart = fixture.id === 'no-tracking' ? 0 : details.requests.length;
      const video = page.locator('#recognition-video');
      const previous = await video.evaluate(element => element.currentSrc);
      await page.locator('#video-file').setInputFiles(fixture.path);
      await page.waitForFunction(before => {
        const current = document.querySelector('#recognition-video');
        return current?.currentSrc.startsWith('blob:') && current.currentSrc !== before && current.readyState >= 2;
      }, previous);
      const result = page.locator('#recognition-result');
      const error = page.locator('.recognition-error');
      assert.equal(await result.count(), 0);
      await page.evaluate(() => { window.engineQaResults.length = 0; });
      const clicked = performance.now();
      await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click();
      await result.or(error).waitFor({ timeout: 60_000 });
      assert.equal(await error.count(), 0, (await error.allTextContents()).join('\n'));
      const elapsedMs = Math.round(performance.now() - clicked);
      const diagnostics = await page.evaluate(() => window.engineQaResults.at(-1));
      assert(diagnostics, `No OpenHands result for ${fixture.id}`);
      const { accepted, label, trackingQuality, rejectionCode, acceptedBy, classifierExecuted } = diagnostics;
      const associatedRequests = details.requests.slice(requestStart).map(request => request.path);
      const weights = associatedRequests.filter(path => Object.values(models).includes(path));
      const outcome = {
        id: fixture.id, sha256: fixture.sha256, elapsedMs,
        text: (await result.innerText()).trim(), accepted, label, trackingQuality, classifierExecuted,
        gate: rejectionCode ?? acceptedBy, weights, runtimeAssets: runtimeFiles(associatedRequests), diagnostics,
      };
      details.rows.push(outcome);
      assert.equal(label, fixture.label, fixture.id);
      assert.equal(accepted, fixture.label !== null, fixture.id);
      assert.equal(await result.getAttribute('data-accepted'), String(accepted), fixture.id);
      const validGates = target === 'candidate' && frameOffset && fixture.id === 'entaxei'
        ? ['primary', 'temporal'] : [fixture.gate];
      assert(validGates.includes(outcome.gate), `${fixture.id}: unexpected gate ${outcome.gate}`);
      if (fixture.id === 'entaxei' && outcome.gate === 'temporal' && typeof diagnostics.recoveryEvidence === 'object') {
        assert.deepEqual(diagnostics.recoveryEvidence.temporalChecks.map(check => check.frames), [12, 15],
          'The 19-frame recording must retain its two distinct short views');
      }
      if (fixture.coverage !== undefined) {
        assert.equal(trackingQuality, fixture.coverage, fixture.id);
        assert.equal(classifierExecuted, false, fixture.id);
      } else {
        assert.equal(classifierExecuted, true, fixture.id);
      }
      console.log(`${target} ${fixture.id}: ${outcome.gate}, ${elapsedMs}ms, weights ${weights.join(', ') || 'none'}`);
    }
    assert.deepEqual(details.pageErrors, []);
    assert(details.requests.every(request => request.origin === origin && ['GET', 'HEAD'].includes(request.method)));
    return details;
  } finally {
    await context.close();
  }
}

function comparable({ label, accepted, classifierExecuted, topCandidates, rejectionCode, acceptedBy, trackingQuality }) {
  return { label, accepted, classifierExecuted, topCandidates, rejectionCode, acceptedBy, trackingQuality };
}

async function verifyRetry(browser) {
  const context = await browser.newContext({ reducedMotion: 'reduce' });
  try {
    let requests = 0;
    await context.route(candidateOrigin + models.primary, async route => {
      requests++;
      if (requests === 1) await route.abort('failed');
      else await route.continue();
    });
    const page = await context.newPage();
    await page.goto(candidateOrigin + '/#/try');
    await page.getByRole('heading', { level: 1, name: 'Αναγνώριση από βίντεο' }).waitFor();
    await page.locator('#video-file').setInputFiles(fixtures.find(fixture => fixture.id === 'geia').path);
    const analyze = page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true });
    const result = page.locator('#recognition-result');
    const error = page.locator('.recognition-error');
    await analyze.click();
    await error.waitFor({ timeout: 60_000 });
    assert.equal(await result.count(), 0);
    assert.equal(requests, 1);
    await analyze.click();
    await result.waitFor({ timeout: 60_000 });
    assert.equal(await result.getAttribute('data-accepted'), 'true');
    assert.equal((await result.innerText()).trim(), 'Γεια');
    assert.equal(await error.count(), 0);
    assert.equal(requests, 2, 'One interrupted download followed by a new request');
    return { interruptedFirstDownload: true, confirmedAfterRetry: true, requests };
  } finally {
    await context.close();
  }
}

await mkdir(dirname(reportPath), { recursive: true });
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.browser = browser.version();
  const baseline = await scan(browser, baselineOrigin, 'baseline');
  const candidate = await scan(browser, candidateOrigin, 'candidate');
  const weightRequests = (details, file) => details.requests.filter(request => request.path === file).length;
  const verifierLoads = candidate.rows.find(row => row.id === 'entaxei').gate === 'temporal' ? [models.secondary] : [];
  assert.deepEqual(baseline.rows[0].weights, [models.primary], 'Baseline must request weights on the first invalid clip');
  assert.deepEqual(candidate.rows.map(row => row.weights), [[], [], [models.primary], [], verifierLoads, []],
    'The engine should download the primary at first eligible input, and the verifier only during recovery');
  assert.equal(weightRequests(baseline, models.primary), 1);
  assert.equal(weightRequests(baseline, models.secondary), 1);
  assert.equal(weightRequests(candidate, models.primary), 1);
  assert.equal(weightRequests(candidate, models.secondary), verifierLoads.length);
  const changes = [];
  for (const [index, current] of candidate.rows.entries()) {
    const previous = baseline.rows[index];
    assert.equal(current.text, previous.text, `File upload must preserve visible output for ${current.id}`);
    if (!frameOffset) assert.deepEqual(comparable(current.diagnostics), comparable(previous.diagnostics),
      `Changing model startup must preserve exact scores and gates for ${current.id}`);
    if (current.gate !== previous.gate) changes.push({ id: current.id, from: previous.gate, to: current.gate });
  }
  assert.deepEqual(candidate.rows.slice(0, 2).flatMap(row => row.runtimeAssets), []);
  report.coldInvalidPrimaryRequests = { baseline: 1, candidate: 0 };
  const weightHead = await fetch(candidateOrigin + models.primary, { method: 'HEAD' });
  assert(weightHead.ok, 'ONNX weights must remain available');
  const weightsBytes = Number(weightHead.headers.get('content-length'));
  assert(Number.isSafeInteger(weightsBytes) && weightsBytes > 0, 'ONNX weights byte count must be measurable');
  report.coldInvalidWeightsBytesAvoided = weightsBytes;
  report.visibleDecisionsMatch = true;
  if (!frameOffset) report.decisionsAndScoresMatch = true;
  report.gateChanges = changes;
  report.modelDownloadRetry = await verifyRetry(browser);
  report.passed = true;
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser?.close();
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(reportPath);
}
