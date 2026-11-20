import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const web = resolve(root, 'web');
const [fixturePath, diagnosticArgument, outputPath, enrollmentMode] = process.argv.slice(2);
const output = resolve(root, outputPath ?? (fixturePath ? 'reports/personal-gestures/custom-restart-integration.json'
  : 'reports/personal-gestures/restart-integration.json'));
const manifestPath = resolve(root, fixturePath ?? '.runtime-tmp/teach-me-rgb-audit/thanks-signer3/manifest.json');
const auditManifest = resolve(root, '.runtime-tmp/recognition-rgb-audit/manifest.json');
const dist = '../publication-local-assistant/dist';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const port = 5292;
const origin = `http://127.0.0.1:${port}`;
const recognition = `${origin}/#/try`;
const manifestBytes = await readFile(manifestPath);
const manifest = JSON.parse(manifestBytes);
const label = manifest.displayLabel;
const expectEnrollment = enrollmentMode ?? manifest.expectedEnrollment ?? manifest.expectEnrollment ?? true;
assert([true, false, 'optional'].includes(expectEnrollment));
assert(label && manifest.targetGloss);
const clips = manifest.clips.map(row => ({ ...row, file: resolve(dirname(manifestPath), row.path) }));
const byRole = role => clips.filter(row => row.role === role);
const training = byRole('enrollment'), checks = byRole('positive_check'), negatives = byRole('negative_check');
const replays = byRole('final_replay');
const [held] = replays;
assert.deepEqual([training.length, checks.length], [3, 2]);
assert(negatives.length >= 2 && held);
assert([...training, ...checks, ...replays].every(row => row.gloss === manifest.targetGloss));
assert(negatives.every(row => row.gloss !== manifest.targetGloss));
assert.equal(new Set(clips.map(row => row.sha256)).size, clips.length);

const diagnosticId = diagnosticArgument ?? (fixturePath ? undefined : 'health4_signer7_rep1_sentences__sentences0025');
let diagnostic;
if (diagnosticId && diagnosticId !== '-') {
  const audit = JSON.parse(await readFile(auditManifest, 'utf8'));
  const row = audit.clips.find(item => item.id === diagnosticId);
  assert(row && row.gloss === manifest.targetGloss);
  diagnostic = { ...row, file: resolve(dirname(auditManifest), row.path) };
}
for (const clip of diagnostic ? [...clips, diagnostic] : clips) {
  assert.equal(digest(await readFile(clip.file)), clip.sha256, clip.id);
}

const report = {
  passed: false,
  scope: manifest.scope ?? 'Paired Chrome restart test with previously exposed CERTH studio recordings.',
  source: manifest.source, license: manifest.license, manifestSha256: digest(manifestBytes),
  assistantLayout: 'The main recognition route is used with default query settings.',
  serverProcessIds: [], browserLaunches: 0, differentOrigin: [], networkWrites: [], errors: [],
};
await mkdir(dirname(output), { recursive: true });
const profileRoot = await mkdtemp(resolve(root, '.runtime-tmp/teach-restart-'));
const profile = resolve(profileRoot, 'chrome-profile');
const localDatabase = resolve(profileRoot, 'isolated-assistant.sqlite3');
let server;
let context;

async function stopServer() {
  if (!server) return;
  const child = server;
  server = undefined;
  if (child.exitCode !== null || child.signalCode !== null) return;
  const stopped = once(child, 'exit');
  child.kill();
  assert.notEqual(await Promise.race([stopped, delay(5000, 'timeout', { ref: false })]), 'timeout');
}

async function startServer() {
  try {
    await fetch(origin, { headers: { Connection: 'close' }, signal: AbortSignal.timeout(500) });
    throw new Error(`Test port ${port} is already occupied`);
  } catch (error) {
    if (error.cause?.code !== 'ECONNREFUSED') throw error;
  }
  const child = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview',
    '--config', 'vite.v8-public.config.ts', '--mode', 'signbridge-local-assistant', '--outDir', dist,
    '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
    cwd: web, env: { ...process.env, SIGNBRIDGE_ASSISTANT_DB: localDatabase }, windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server = child;
  let startupError;
  let recentOutput = '';
  child.on('error', error => { startupError = error; });
  for (const pipe of [child.stdout, child.stderr]) pipe.on('data', chunk => { recentOutput = (recentOutput + chunk).slice(-5000); });
  for (let attempt = 0; attempt < 40; attempt++) {
    if (startupError) throw startupError;
    if (child.exitCode !== null) throw new Error(`Test preview failed: ${recentOutput}`);
    try {
      const response = await fetch(`${origin}/__signbridge/local-assistant/status`, {
        headers: { Connection: 'close' }, signal: AbortSignal.timeout(600),
      });
      if (response.ok && (await response.json()).service === 'signbridge-local-assistant') {
        report.serverProcessIds.push(child.pid);
        return;
      }
    } catch (error) {
      if (error.cause?.code !== 'ECONNREFUSED' && error.name !== 'TimeoutError') throw error;
    }
    await delay(250);
  }
  throw new Error(`Test preview did not become ready: ${recentOutput}`);
}

async function openBrowser() {
  context = await chromium.launchPersistentContext(profile, { channel: 'chrome', headless: true,
    args: ['--disable-background-timer-throttling'] });
  const browser = context.browser();
  report.browser = browser?.version() ?? 'Chrome';
  report.browserLaunches++;
  context.on('request', request => {
    const url = new URL(request.url());
    if (!['http:', 'https:'].includes(url.protocol)) return;
    if (url.origin !== origin) report.differentOrigin.push(url.href);
    if (!['GET', 'HEAD'].includes(request.method())) report.networkWrites.push(`${request.method()} ${url.pathname}`);
  });
  const page = context.pages()[0] ?? await context.newPage();
  page.on('pageerror', error => report.errors.push(error.message));
  await page.goto(recognition);
  await page.locator('#recognition-message-title').waitFor();
  return page;
}

async function analyze(page, clip) {
  const result = page.locator('#recognition-result');
  const previous = await page.locator('#recognition-video').evaluate(node => node.currentSrc);
  await page.locator('#video-file').setInputFiles(clip.file);
  await page.waitForFunction(before => {
    const video = document.querySelector('#recognition-video');
    return video?.currentSrc.startsWith('blob:') && video.currentSrc !== before && video.readyState >= 2;
  }, previous);
  const failure = page.getByRole('alert');
  assert.equal(await result.count(), 0, 'Previous recognition should clear after opening a clip');
  await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click();
  await result.or(failure).waitFor({ timeout: 90_000 });
  assert.equal(await failure.count(), 0, (await failure.allTextContents()).join(' '));
  return { id: clip.id, signer: clip.signer, gloss: clip.gloss, sha256: clip.sha256,
    source: await result.getAttribute('data-source'), text: (await result.innerText()).trim(),
    accepted: (await result.getAttribute('data-accepted')) === 'true' };
}

async function closeBrowser() {
  if (!context) return;
  const active = context;
  context = undefined;
  await active.close();
}

async function enroll(page) {
  const handoff = page.getByRole('button', { name: 'Διόρθωσε στο Teach Me', exact: true });
  if (expectEnrollment === 'optional' && !await handoff.count()) {
    report.enrollment = 'The first enrollment clip did not pass the handoff quality gate.';
    return false;
  }
  await handoff.click();
  await page.waitForURL('**/#/teach');
  await page.getByTestId('teach-carried-sample').waitFor();
  const name = page.getByLabel('Σημασία του νοήματος');
  assert.equal(await name.inputValue(), '', 'The correction label should come from the user');
  await name.fill(label);
  const enrollmentFiles = page.locator('input[name=training]');
  await page.locator('input[name=checks]').setInputFiles(checks.map(row => row.file));
  await page.locator('input[name=negatives]').setInputFiles(negatives.slice(0, 2).map(row => row.file));
  const submit = page.getByRole('button', { name: 'Έλεγχος και αποθήκευση', exact: true });
  if (expectEnrollment !== 'optional') {
    await enrollmentFiles.setInputFiles(training.slice(0, 2).map(row => row.file));
    await submit.click();
    const duplicate = page.getByText(/χρησιμοποίησε διαφορετική λήψη.+χωρίς αντίγραφα/iu);
    await duplicate.waitFor({ timeout: 150_000 });
    report.duplicateCarriedSampleRefused = (await duplicate.innerText()).trim();
    assert.equal(await page.getByRole('button', { name: `Διαγραφή ${label}`, exact: true }).count(), 0);
    await page.getByTestId('teach-carried-sample').waitFor();
  }
  await enrollmentFiles.setInputFiles(training.slice(1).map(row => row.file));
  await submit.click();
  const enrollment = page.getByText(/αποθηκεύτηκε αφού πέρασε|χρειάζεται νέα παραδείγματα/).or(page.getByRole('alert'));
  await enrollment.waitFor({ timeout: 150_000 });
  report.enrollment = await enrollment.innerText();
  const deleteSign = page.getByRole('button', { name: `Διαγραφή ${label}`, exact: true });
  if (expectEnrollment === true) await deleteSign.waitFor();
  if (expectEnrollment === false) {
    assert.match(report.enrollment, /χρειάζεται νέα παραδείγματα/u);
    assert.equal(await deleteSign.count(), 0);
    report.enrollmentRefusedAsExpected = true;
  }
  return await deleteSign.count() > 0;
}

try {
  await startServer();
  let page = await openBrowser();
  report.baseline = await analyze(page, held);
  assert.equal(report.baseline.source, 'model');
  if (expectEnrollment === 'optional') {
    report.activationCheckBaseline = await analyze(page, checks[0]);
    assert.equal(report.activationCheckBaseline.source, 'model');
  }
  report.trainingStart = await analyze(page, training[0]);
  assert.equal(report.trainingStart.source, 'model');
  report.trainingIds = training.map(row => row.id);
  report.positiveCheckIds = checks.map(row => row.id);
  report.negativeCheckIds = negatives.slice(0, 2).map(row => row.id);
  report.enrolled = await enroll(page);
  await closeBrowser();
  await stopServer();

  await startServer();
  assert.notEqual(report.serverProcessIds[0], report.serverProcessIds[1]);
  page = await openBrowser();
  report.afterRestart = await analyze(page, held);
  if (expectEnrollment !== 'optional' || !report.enrolled) {
    assert.equal(report.afterRestart.source, report.enrolled ? 'personal' : 'model');
  }
  if (report.afterRestart.source === 'personal') {
    assert.equal(report.afterRestart.text, label);
    assert.equal(report.afterRestart.accepted, true);
  }
  if (negatives[2]) {
    report.differentSign = await analyze(page, negatives[2]);
    assert.equal(report.differentSign.source, 'model');
  }
  if (diagnostic) report.otherSigner = await analyze(page, diagnostic);
  if (replays.length > 1) {
    report.additionalReplays = [];
    for (const clip of replays.slice(1)) report.additionalReplays.push(await analyze(page, clip));
  }
  if (expectEnrollment === 'optional' && report.enrolled) {
    report.activationCheckReplays = [];
    for (const clip of checks) report.activationCheckReplays.push(await analyze(page, clip));
  }

  if (report.enrolled) {
    await page.goto(`${origin}/#/teach`);
    await page.getByRole('button', { name: `Διαγραφή ${label}`, exact: true }).click();
    await page.reload();
    await page.getByText('Τα νοήματα που περνούν τον έλεγχο εμφανίζονται εδώ.', { exact: true }).waitFor();
    await page.goto(recognition);
    report.afterDelete = await analyze(page, held);
    assert.equal(report.afterDelete.source, 'model');
  }
  assert.deepEqual(report.differentOrigin, []);
  assert.deepEqual(report.networkWrites, []);
  assert.deepEqual(report.errors, []);
  report.passed = true;
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  try {
    await closeBrowser();
  } finally {
    try { await stopServer(); }
    finally { await writeFile(output, `${JSON.stringify(report, null, 2)}\n`); }
  }
}

console.log(output);
