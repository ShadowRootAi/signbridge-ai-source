import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdir, mkdtemp, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const web = resolve(root, 'web');
const port = 5208;
const origin = `http://127.0.0.1:${port}`;
const endpoint = '/__signbridge/local-assistant/chat';
const demo = `${origin}/?assistant=1#/try`;
const original = 'ραντεβού Δευτέρα στις 17:00';
const replacement = 'ραντεβού Πέμπτη στις 18:00';
const question = `Η παλιά σημείωση γράφει «${original}». Πότε είναι το ραντεβού σύμφωνα με τη διόρθωσή μου;`;
const runtime = resolve(root, '.runtime-tmp');
const reportPath = resolve(root, 'reports/local-assistant/ui-restart.json');
await mkdir(runtime, { recursive: true });
await mkdir(dirname(reportPath), { recursive: true });
const database = resolve(await mkdtemp(resolve(runtime, 'assistant-ui-qa-')), 'memory.sqlite3');
const report = { passed: false, model: 'qwen3:4b', browser: '', isolatedDatabase: true,
  serverProcessIds: [], sameOriginRequests: true, unrelatedCorrectionMatches: [], recognitionLabel: '' };
let activeServer;
let activeBrowser;

async function stopServer() {
  if (!activeServer) return;
  const process = activeServer;
  activeServer = undefined;
  if (process.exitCode !== null || process.signalCode !== null) return;
  const stopped = once(process, 'exit');
  process.kill();
  const timeout = delay(5000, 'timeout', { ref: false });
  assert.notEqual(await Promise.race([stopped, timeout]), 'timeout', 'Could not shut down the test preview');
}

async function startServer() {
  try {
    await fetch(origin, { headers: { Connection: 'close' }, signal: AbortSignal.timeout(500) });
    throw new Error(`The test port ${port} already has a server`);
  } catch (error) {
    if (error.cause?.code !== 'ECONNREFUSED') throw error;
  }
  const child = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--config', 'vite.v8-public.config.ts',
    '--mode', 'signbridge-local-assistant', '--outDir', '../publication-local-assistant/dist', '--host', '127.0.0.1',
    '--port', String(port), '--strictPort'], {
    cwd: web, env: { ...process.env, SIGNBRIDGE_ASSISTANT_DB: database }, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
  });
  activeServer = child;
  let output = '';
  let launchError;
  child.on('error', error => { launchError = error; });
  for (const pipe of [child.stdout, child.stderr]) pipe.on('data', chunk => { output = (output + chunk).slice(-5000); });
  for (let attempt = 0; attempt < 40; attempt++) {
    if (launchError) throw launchError;
    if (child.exitCode !== null) throw new Error(`Local assistant preview failed: ${output}`);
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
  throw new Error(`Local assistant preview did not become ready: ${output}`);
}

async function openBrowser() {
  activeBrowser = await chromium.launch({ channel: 'chrome', headless: true });
  report.browser = activeBrowser.version();
  const page = await activeBrowser.newPage();
  page.on('request', request => {
    const url = new URL(request.url());
    if (['http:', 'https:'].includes(url.protocol) && url.origin !== origin) report.sameOriginRequests = false;
  });
  return page;
}

async function openAssistant(page) {
  await page.getByRole('button', { name: 'Τοπικός βοηθός' }).click();
  const dialog = page.getByRole('dialog', { name: 'Τοπικός βοηθός' });
  await dialog.waitFor();
  return dialog;
}

async function submit(page, action, message) {
  const isRequest = value => value.url() === `${origin}${endpoint}`;
  const [request, response] = await Promise.all([
    page.waitForRequest(value => isRequest(value) && value.method() === 'POST'),
    page.waitForResponse(value => isRequest(value) && value.request().method() === 'POST', { timeout: 125_000 }),
    action.click(),
  ]);
  const data = await response.json();
  assert(response.ok(), `${message}: ${data.error ?? response.status()}`);
  const reply = page.locator('[data-local-assistant-reply]');
  await reply.waitFor();
  assert.equal(await reply.textContent(), data.reply, message);
  return { sent: request.postDataJSON(), data };
}

async function chat(page, dialog, message) {
  await dialog.getByRole('textbox', { name: 'Μήνυμα', exact: true }).fill(message);
  return submit(page, dialog.getByRole('button', { name: 'Αποστολή', exact: true }), 'Chat response not displayed');
}

try {
  await startServer();
  let page = await openBrowser();
  await page.goto(`${origin}/#/try`);
  await page.locator('#recognition-message-title').waitFor();
  assert.equal(await page.getByRole('button', { name: 'Τοπικός βοηθός' }).count(), 0);
  report.normalRouteHasNoAssistant = true;
  await page.goto(demo);
  let dialog = await openAssistant(page);
  const before = await chat(page, dialog, question);
  assert.deepEqual(before.data.appliedCorrections, []);
  assert(!(/πέμπτη/iu.test(before.data.reply) && before.data.reply.includes('18:00')));
  report.beforeCorrection = before.data.reply;
  await dialog.getByText('Αποθήκευση διόρθωσης', { exact: true }).click();
  await dialog.getByRole('textbox', { name: 'Αρχική φράση', exact: true }).fill(original);
  await dialog.getByRole('textbox', { name: 'Σωστή φράση', exact: true }).fill(replacement);
  const saved = await submit(page, dialog.getByRole('button', { name: 'Αποθήκευση', exact: true }), 'Correction save confirmation missing');
  assert.equal(saved.data.remembered, true);
  assert.deepEqual(saved.sent.correction, { from: original, to: replacement });
  assert((await stat(database)).size > 0);
  await dialog.getByRole('button', { name: 'Κλείσιμο', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
  await activeBrowser.close();
  activeBrowser = undefined;
  await stopServer();

  await startServer();
  assert.notEqual(report.serverProcessIds[0], report.serverProcessIds[1]);
  page = await openBrowser();
  await page.goto(demo);
  dialog = await openAssistant(page);
  const after = await chat(page, dialog, question);
  assert.deepEqual(after.data.appliedCorrections, [{ from: original, to: replacement }]);
  assert(/πέμπτη/iu.test(after.data.reply) && after.data.reply.includes('18:00'), 'Qwen did not answer with the corrected day and time');
  assert(!/δευτέρα|17:00/iu.test(after.data.reply), 'Qwen repeated the superseded day or time');
  const visibleCorrection = dialog.getByRole('list', { name: 'Διορθώσεις που χρησιμοποιήθηκαν' });
  assert((await visibleCorrection.innerText()).includes(replacement));
  report.afterRestart = after.data.reply;
  report.appliedCorrections = after.data.appliedCorrections;

  const unrelated = await chat(page, dialog, 'Πες καλημέρα σε δύο λέξεις.');
  assert.deepEqual(unrelated.data.appliedCorrections, []);
  assert.equal(await visibleCorrection.count(), 0);
  report.unrelatedCorrectionMatches = unrelated.data.appliedCorrections;
  await dialog.getByRole('button', { name: 'Κλείσιμο', exact: true }).click();

  await page.locator('#video-file').setInputFiles(resolve(root, 'publication-local-assistant/dist/library/geia.mp4'));
  await page.getByRole('button', { name: 'Ανάλυση βίντεο' }).click();
  await page.waitForFunction(() => document.querySelector('#recognition-result')?.getAttribute('data-accepted') === 'true', null, { timeout: 90_000 });
  assert.equal(await page.locator('#recognition-result').innerText(), 'Γεια');
  dialog = await openAssistant(page);
  const recognition = await chat(page, dialog, 'Ποιο νόημα αναγνώρισε η εφαρμογή;');
  assert.deepEqual(recognition.sent.recognition, { accepted: true, label: 'Γεια' });
  assert(/γεια/iu.test(recognition.data.reply), 'Qwen did not use the recognition label in its response');
  assert.equal(await page.locator('#recognition-result').innerText(), 'Γεια');
  report.recognitionLabel = recognition.sent.recognition.label;
  report.recognitionReply = recognition.data.reply;
  await dialog.getByRole('button', { name: 'Εκκαθάριση μνήμης', exact: true }).click();
  const cleared = await submit(page, dialog.getByRole('button', { name: 'Διαγραφή όλων', exact: true }),
    'Memory clear confirmation missing');
  assert.deepEqual(cleared.sent, { memory: 'clear' });
  assert.deepEqual((await chat(page, dialog, question)).data.appliedCorrections, []);
  report.clearConfirmed = true;
  assert(report.sameOriginRequests);
  report.passed = true;
} catch (error) {
  report.error = error.message;
  throw error;
} finally {
  try {
    await activeBrowser?.close();
    await stopServer();
  } finally {
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  }
}

console.log(JSON.stringify(report, null, 2));
