/**
 * Actual-browser V8 app and debug QA using existing development RGB clips.
 * No recognition mocks. Chrome's camera is explicitly a simulated device fed
 * with an actual GSL clip, not a human webcam evaluation.
 * node web/scripts/qa-v8-pilot.mjs --url http://127.0.0.1:5173
 * --prepare-only creates local MP4/Y4M fixtures without opening the pilot.
 */
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { chromium } from 'playwright';

const exec = promisify(execFile);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const option = (key, fallback) => { const i = args.indexOf(key); return i < 0 ? fallback : args[i + 1]; };
const base = option('--url', 'http://127.0.0.1:5173').replace(/\/$/, '');
const appUrl = `${base}/v8-pilot.html`;
const url = `${appUrl}?debug=1`;
const out = path.resolve(root, option('--output-dir', 'reports/v8/app-ui-qa'));
const fixturesDir = path.join(out, 'fixtures');
const sha = value => createHash('sha256').update(value).digest('hex');
const relative = filename => path.relative(root, filename).replaceAll('\\', '/');
const report = { createdAt: new Date().toISOString(), passed: false, url, appUrl,
  scope: 'Local app and debug UI/inference integration on previously exposed GSL development clips',
  humanWebcamTested: false, recognitionMocked: false, productionChanged: false,
  simulatedCamera: { used: false, source: 'Real development GSL clip repeated by Chrome fake camera',
    limitation: 'Repeated file footage has artificial boundaries and is not a naturally signed webcam utterance.' },
  checks: [], realVideoRuns: [], cameraRuns: [], screenshots: [], errors: [] };
let browser, context, page, failure;

async function prepareFixtures() {
  await mkdir(fixturesDir, { recursive: true });
  const planPath = path.join(root, 'data/v7-recovery/extraction-protocol.json');
  const planBytes = await readFile(planPath), plan = JSON.parse(planBytes);
  const rows = [
    { role: 'known', source: 'kep1_signer6_rep3_sentences/sentences0000', expected: 'ΓΕΙΑ' },
    { role: 'known_thanks', source: 'police1_signer6_rep1_sentences/sentences0013', expected: 'ΕΥΧΑΡΙΣΤΩ' },
    { role: 'known_okay', source: 'police1_signer6_rep1_sentences/sentences0011', expected: 'ΕΝΤΑΞΕΙ' },
    { role: 'hard_unknown', source: 'health4_signer6_rep4_sentences/sentences0021', expected: null },
    { role: 'incomplete', source: 'police2_signer6_rep1_sentences/sentences0022', expected: null },
  ];
  const python = path.join(root, '.venv/Scripts/python.exe');
  const { stdout } = await exec(python, ['-X', 'utf8', '-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], { cwd: root });
  const ffmpeg = stdout.trim();
  assert.ok(ffmpeg && path.isAbsolute(ffmpeg));
  const ffmpegVersion = (await exec(ffmpeg, ['-version'])).stdout.split(/\r?\n/)[0];
  for (const item of rows) {
    const source = plan.samples.find(row => row.source_id === item.source);
    assert.ok(source && source.split === 'val' && Number(source.signer) === 6, 'Only the declared non-final person6 clips');
    assert.equal(source.paths.length, source.frameSha256.length);
    const first = path.resolve(root, source.paths[0]);
    for (let i = 0; i < source.paths.length; i++) {
      const filename = path.resolve(root, source.paths[i]);
      assert.ok(!path.relative(root, filename).startsWith('..') && !path.isAbsolute(path.relative(root, filename)));
      assert.equal(filename, path.join(path.dirname(first), `frame_${String(i).padStart(4, '0')}.jpg`));
      assert.equal(sha(await readFile(filename)), source.frameSha256[i], 'Original RGB digest');
    }
    item.path = path.join(fixturesDir, `${item.role}.mp4`);
    const ffmpegArgs = ['-hide_banner', '-loglevel', 'error', '-y', '-framerate', '30', '-start_number', '0',
      '-i', path.join(path.dirname(first), 'frame_%04d.jpg'), '-frames:v', String(source.paths.length),
      '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', item.path];
    await exec(ffmpeg, ffmpegArgs, { maxBuffer: 2 ** 20 });
    Object.assign(item, { frameCount: source.paths.length, rawLabel: source.raw_label, sourceId: source.source_id,
      sourceContentSha256: source.sourceContentSha256, sha256: sha(await readFile(item.path)),
      encoding: 'H.264 CRF0 yuv420p, original30fps/order, no resizing. RGB/chroma conversion can still differ from source JPEG extraction.' });
  }
  const fakeVideo = path.join(fixturesDir, 'real-gsl-simulated-camera.y4m');
  await exec(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-i', rows[0].path, '-an',
    '-pix_fmt', 'yuv420p', '-f', 'yuv4mpegpipe', fakeVideo], { maxBuffer: 2 ** 20 });
  const result = { extractionProtocolSha256: sha(planBytes), ffmpegVersion,
    clips: rows, fakeVideo, fakeVideoSha256: sha(await readFile(fakeVideo)),
    noOld129Used: true, noNewHumanRecording: true };
  await writeFile(path.join(out, 'fixtures.json'), `${JSON.stringify(result, null, 2)}\n`);
  return result;
}

async function screenshot(name) {
  const target = path.join(out, name);
  await page.screenshot({ path: target, fullPage: true, animations: 'disabled' });
  report.screenshots.push(relative(target));
}

async function downloadReport(name) {
  const pending = page.waitForEvent('download');
  await page.locator('#save-report').click();
  const download = await pending;
  const filename = path.join(out, `${name}.json`);
  await download.saveAs(filename);
  const data = JSON.parse(await readFile(filename, 'utf8'));
  assert.equal(data.schema, 'signbridge-v8-pilot-v1');
  assert.ok(data.clip_id && ['file', 'webcam'].includes(data.source));
  assert.ok(data.media?.sha256 && data.media.byte_length > 0 && data.media.duration_ms > 0);
  assert.ok(data.media.width > 0 && data.media.height > 0);
  for (const method of ['baseline', 'v8']) {
    const decision = data.comparison[method];
    assert.equal(typeof decision.accepted, 'boolean');
    assert.ok(decision.classificationScore === null || Number.isFinite(decision.classificationScore) && decision.classificationScore >= 0 && decision.classificationScore <= 1);
    if (decision.accepted) assert.ok(['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ'].includes(decision.label));
    else assert.equal(decision.label, null);
    assert.equal(await page.locator(`#${method === 'baseline' ? 'baseline' : 'v8'}-result`).getAttribute('data-accepted'), String(decision.accepted));
    const domLabel = await page.locator(`#${method === 'baseline' ? 'baseline' : 'v8'}-result`).getAttribute('data-label');
    assert.equal(domLabel || null, decision.label);
  }
  assert.ok(data.comparison.originalTracking && data.comparison.fusedTracking);
  assert.equal(data.comparison.frames, data.comparison.fusedTracking.frameCount);
  assert.ok(data.comparison.frames >= 4);
  assert.equal(typeof data.comparison.classifierExecuted, 'boolean');
  for (const value of Object.values(data.comparison.timings)) assert.ok(Number.isFinite(value) && value >= 0);
  assert.ok(data.comparison.timings.totalMs > 0);
  if (!data.comparison.fusedTracking.usable) {
    assert.equal(data.comparison.classifierExecuted, false);
    assert.equal(data.comparison.logits, null);
    assert.equal(data.comparison.baseline.classificationScore, null);
    assert.equal(data.comparison.v8.classificationScore, null);
  } else assert.equal(data.comparison.classifierExecuted, true);
  assert.ok(!data.comparison.v8.accepted || data.comparison.baseline.accepted, 'Veto cannot create new acceptance');
  return { filename: relative(filename), data };
}

async function completeAnalysis(name) {
  await page.locator('#analyze').waitFor({ state: 'visible' });
  await page.waitForFunction(() => !document.querySelector('#analyze').disabled);
  const start = Date.now();
  await page.locator('#analyze').click();
  await page.waitForFunction(() => document.querySelector('#results')?.dataset.state === 'ready'
    || Boolean(document.querySelector('#error')?.textContent.trim()), null, { timeout: 180000 });
  assert.equal((await page.locator('#error').innerText()).trim(), '', `${name}: no analysis error`);
  assert.equal(await page.locator('#results').getAttribute('data-state'), 'ready');
  return start;
}

async function analyze(name, source, mediaSha256) {
  const start = await completeAnalysis(name);
  const saved = await downloadReport(name);
  assert.equal(saved.data.source, source);
  if (mediaSha256) assert.equal(saved.data.media.sha256, mediaSha256, 'Actual input bytes bound to report');
  return { role: name, wallMs: Date.now() - start, ...saved };
}

async function setFile(filename) {
  await page.locator('#file-input').setInputFiles(filename);
  await page.waitForFunction(() => !document.querySelector('#analyze').disabled);
  assert.notEqual(await page.locator('#results').getAttribute('data-state'), 'ready', 'Changing input clears prior result');
}

function compareDecisions(first, second) {
  for (const method of ['baseline', 'v8']) {
    assert.equal(second.comparison[method].accepted, first.comparison[method].accepted, `${method}: capture/replay acceptance`);
    assert.equal(second.comparison[method].label, first.comparison[method].label, `${method}: capture/replay label`);
    const a = first.comparison[method].classificationScore, b = second.comparison[method].classificationScore;
    if (a === null || b === null) assert.equal(a, b);
    else assert.ok(Math.abs(a - b) < 1e-5, `${method}: same-byte replay score`);
  }
}

async function checkPublicView(fixtures, expectedCsp) {
  const response = await page.goto(appUrl, { waitUntil: 'networkidle', timeout: 60000 });
  assert.ok(response?.ok());
  assert.equal(response.headers()['content-security-policy'], expectedCsp);
  report.networkPolicyHeaders.push({ url: appUrl, contentSecurityPolicy: response.headers()['content-security-policy'] });
  assert.equal(await page.locator('html').getAttribute('data-mode'), 'app');
  assert.equal(await page.locator('#consent').isChecked(), false);
  assert.equal(await page.locator('#open-camera').isDisabled(), true);
  assert.equal(await page.locator('#live-video').evaluate(video => video.srcObject), null);
  assert.equal(await page.locator('#public-result-area').isVisible(), false);
  assert.equal(await page.locator('#file-input').isEnabled(), true);
  const layoutAt = async (width, height) => {
    await page.setViewportSize({ width, height });
    const layout = await page.evaluate(() => ({ viewportWidth: innerWidth, documentWidth: document.documentElement.scrollWidth }));
    assert.ok(layout.documentWidth <= layout.viewportWidth + 1, `Public view fits ${width}px viewport`);
    return layout;
  };
  const desktop = await layoutAt(1400, 1050);
  await screenshot('app-desktop.png');

  const expectedRuns = [
    [fixtures.clips.find(row => row.role === 'known'), 'ΓΕΙΑ', 'app-known-desktop.png'],
    [fixtures.clips.find(row => row.role === 'hard_unknown'), 'Δεν αναγνωρίστηκε', 'app-rejection-mobile.png'],
  ];
  report.app = { desktop, runs: [] };
  for (const [fixture, expected, imageName] of expectedRuns) {
    if (fixture.role === 'hard_unknown') {
      await page.emulateMedia({ reducedMotion: 'reduce' });
      report.app.mobile = await layoutAt(390, 844);
    }
    await setFile(fixture.path);
    const start = await completeAnalysis(fixture.role);
    assert.equal(await page.locator('#public-result-area').isVisible(), true);
    assert.equal(await page.locator('#public-result').innerText(), expected);
    const technicalSelectors = ['#blind', '#collection-fields', '#mirror', '#visible-results', '#baseline-result',
      '#v8-result', '#tracking-result', '#timing-result', '#media-info', '#save-report', '#save-video'];
    for (const selector of technicalSelectors) {
      assert.equal(await page.locator(selector).isVisible(), false, `${selector} stays out of the public view`);
    }
    const renderedDebugNodes = await page.locator('[data-debug-only]').evaluateAll(elements =>
      elements.filter(element => element.getClientRects().length).map(element => element.id || element.tagName));
    assert.deepEqual(renderedDebugNodes, []);
    for (const selector of ['#baseline-result', '#v8-result', '#tracking-result', '#timing-result', '#media-info']) {
      assert.equal((await page.locator(selector).textContent()).trim(), '', `${selector} contains no public diagnostic values`);
    }
    const text = await page.locator('body').innerText();
    assert.doesNotMatch(text, /V4 με ανάκτηση|V8 με όριο|αναφοράς JSON|Λειτουργία συλλογής/);
    report.app.runs.push({ role: fixture.role, inputSha256: fixture.sha256, output: expected, wallMs: Date.now() - start });
    await screenshot(imageName);
  }
  report.app.narrow = await layoutAt(320, 700);
  report.app.reducedMotion = await page.evaluate(() => ({
    requested: matchMedia('(prefers-reduced-motion: reduce)').matches,
    runningAnimations: document.getAnimations().filter(animation => animation.playState === 'running').length,
  }));
  assert.deepEqual(report.app.reducedMotion, { requested: true, runningAnimations: 0 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.setViewportSize({ width: 1400, height: 1050 });
  report.checks.push('Public view shows one real known result and one real rejection without comparative outputs, downloads or diagnostics. Desktop and mobile layouts fit and honor reduced motion.');

  await page.locator('#clear').click();
  await page.locator('#consent').check();
  await page.locator('#open-camera').click();
  await page.waitForFunction(() => document.querySelector('#live-video').videoWidth > 0, null, { timeout: 15000 });
  await page.locator('#live-video').evaluate(video => { window.__v8AppTracks = [...video.srcObject.getVideoTracks()]; });
  await page.locator('#record').click();
  await page.waitForFunction(() => !document.querySelector('#stop-recording').disabled, null, { timeout: 10000 });
  await page.waitForFunction(() => !document.querySelector('#analyze').disabled, null, { timeout: 25000 });
  assert.equal(await page.locator('#live-video').evaluate(video => video.srcObject), null);
  assert.equal(await page.evaluate(() => window.__v8AppTracks.length > 0 && window.__v8AppTracks.every(track => track.readyState === 'ended')), true);
  const duration = await page.locator('#preview-video').evaluate(video => video.duration);
  assert.ok(duration >= 2.5 && duration <= 3.5, `Public automatic recording duration: ${duration}`);
  await completeAnalysis('public-simulated-camera');
  assert.equal(await page.locator('#public-result-area').isVisible(), true);
  const cameraResult = await page.locator('#public-result').innerText();
  assert.ok(['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ', 'Δεν αναγνωρίστηκε'].includes(cameraResult));
  report.app.camera = { simulatedDevice: true, duration, cameraReleasedAfterRecording: true, output: cameraResult };
  report.checks.push('Public camera records the simulated real-file device, releases its tracks automatically and analyzes the recorded clip.');
}

try {
  await mkdir(out, { recursive: true });
  const fixtures = await prepareFixtures();
  report.fixtureManifestSha256 = sha(await readFile(path.join(out, 'fixtures.json')));
  if (args.includes('--prepare-only')) {
    console.log(JSON.stringify({ prepared: true, fixtures: fixtures.clips.map(row => ({ role: row.role, path: row.path, sha256: row.sha256 })), fakeVideo: fixtures.fakeVideo }));
  } else {
    browser = await chromium.launch({ channel: 'chrome', headless: true,
      args: ['--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${fixtures.fakeVideo}`] });
    report.browserVersion = browser.version();
    context = await browser.newContext({ viewport: { width: 1400, height: 1050 }, acceptDownloads: true });
    page = await context.newPage(); page.setDefaultTimeout(120000);
    page.on('pageerror', error => report.errors.push(error.message));
    const workerUrls = new Set(), cspMessages = [];
    page.on('worker', worker => workerUrls.add(worker.url()));
    page.on('console', message => { if (/Content Security Policy|connect-src/.test(message.text())) cspMessages.push(message.text()); });
    const requests = new Set(), requestMethods = [], requestRecords = new Map();
    context.on('request', request => { if (/^https?:/.test(request.url())) {
      const body = request.postDataBuffer();
      const record = { method: request.method(), url: request.url(), resourceType: request.resourceType(),
        bodyByteLength: body?.length ?? 0, bodySha256: body ? sha(body) : null, responseStatus: null, failure: null };
      requests.add(request.url()); requestMethods.push(record); requestRecords.set(request, record);
    } });
    context.on('response', response => { const record = requestRecords.get(response.request()); if (record) record.responseStatus = response.status(); });
    context.on('requestfailed', request => { const record = requestRecords.get(request); if (record) record.failure = request.failure()?.errorText ?? 'unknown'; });
    const response = await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
    assert.ok(response?.ok());
    // Vite can reload the document after dependency optimization; fetch the served
    // HTML independently so a discarded navigation response cannot break QA.
    const servedHtml = await context.request.get(url);
    assert.ok(servedHtml.ok()); report.servedHtmlSha256 = sha(await servedHtml.body());
    report.servedHtmlHashSource = 'Independent same-origin GET after navigation';
    const expectedCsp = "connect-src 'self' blob: ws://127.0.0.1:* ws://localhost:*";
    assert.equal(servedHtml.headers()['content-security-policy'], expectedCsp);
    report.networkPolicyHeaders = [{ url, contentSecurityPolicy: servedHtml.headers()['content-security-policy'] }];
    report.sourceFileSha256 = {};
    for (const source of ['web/v8-pilot.html', 'web/src/v8-pilot.ts', 'web/src/v8-pilot-core.ts', 'web/src/v8-pilot.css',
      'web/scripts/qa-v8-pilot.mjs', 'web/vite.config.ts', 'artifacts/v8/browser/model-config.json']) {
      report.sourceFileSha256[source] = sha(await readFile(path.join(root, source)));
    }
    await page.locator('#file-input').waitFor({ state: 'attached' });
    assert.equal(await page.locator('html').getAttribute('data-mode'), 'debug');
    assert.equal(await page.locator('#consent').isChecked(), false);
    assert.equal(await page.locator('#open-camera').isDisabled(), true, 'No camera before consent');
    assert.equal(await page.locator('#live-video').evaluate(video => video.srcObject), null);
    report.checks.push('Camera is closed and unavailable before explicit consent.');

    for (const fixture of fixtures.clips) {
      console.log(`Real development MP4: ${fixture.role}`);
      await setFile(fixture.path);
      const run = await analyze(`mp4-${fixture.role}`, 'file', fixture.sha256);
      report.realVideoRuns.push({ ...run, sourceId: fixture.sourceId, expectedRawLabel: fixture.rawLabel });
      await screenshot(`mp4-${fixture.role}.png`);
      if (fixture.role.startsWith('known')) {
        assert.equal(run.data.comparison.baseline.accepted, true);
        assert.equal(run.data.comparison.v8.accepted, true); assert.equal(run.data.comparison.v8.label, fixture.expected);
      } else if (fixture.role === 'hard_unknown') {
        assert.equal(run.data.comparison.baseline.accepted, true, 'Hard unknown reaches classifier; not trivial raw-guard rejection');
        assert.equal(run.data.comparison.v8.accepted, false, 'V8 rejects the hard unknown');
      } else {
        assert.equal(run.data.comparison.baseline.accepted, false); assert.equal(run.data.comparison.v8.accepted, false);
        assert.equal(run.data.comparison.baseline.classificationScore, null); assert.equal(run.data.comparison.v8.classificationScore, null);
        assert.equal(run.data.comparison.classifierExecuted, false);
      }
    }
    report.checks.push('Five real non-final MP4s exercise all three supported signs, a hard-unknown veto and incomplete-input rejection.');

    // Identical bytes under an unrelated filename cannot select another prediction.
    await page.locator('#file-input').setInputFiles({ name: 'unrelated-entaxei-name.mp4', mimeType: 'video/mp4', buffer: await readFile(fixtures.clips[0].path) });
    const renamed = await analyze('mp4-renamed-known', 'file', fixtures.clips[0].sha256);
    compareDecisions(report.realVideoRuns[0].data, renamed.data); report.realVideoRuns.push(renamed);
    await page.locator('#clear').click();
    assert.notEqual(await page.locator('#results').getAttribute('data-state'), 'ready');

    const permission = await context.newCDPSession(page);
    const targetInfo = await permission.send('Target.getTargetInfo');
    const browserContextId = targetInfo.targetInfo.browserContextId;
    assert.ok(browserContextId, 'Permission override targets the actual incognito context');
    console.log('Camera: explicit permission denial');
    await permission.send('Browser.setPermission', { browserContextId, permission: { name: 'camera' }, setting: 'denied', origin: new URL(base).origin });
    await page.locator('#consent').check(); await page.locator('#open-camera').click();
    await page.waitForFunction(() => Boolean(document.querySelector('#error')?.textContent.trim()), null, { timeout: 15000 });
    assert.equal(await page.locator('#live-video').evaluate(video => video.srcObject), null);
    report.permissionDeniedMessage = await page.locator('#error').innerText();
    report.checks.push('Actual Chrome permission denial leaves the camera closed and displays an error.');
    console.log('Camera: granting simulated real-file device');
    await permission.send('Browser.setPermission', { browserContextId, permission: { name: 'camera' }, setting: 'granted', origin: new URL(base).origin });
    await page.locator('#open-camera').click();
    await page.waitForFunction(() => { const video = document.querySelector('#live-video'); return video.srcObject && video.videoWidth > 0; }, null, { timeout: 15000 });
    report.simulatedCamera.used = true; report.simulatedCamera.fakeVideoSha256 = fixtures.fakeVideoSha256;
    report.simulatedCamera.initialTracks = await page.locator('#live-video').evaluate(video => {
      window.__v8ObservedTracks = [...video.srcObject.getTracks()];
      return window.__v8ObservedTracks.map(track => ({ kind: track.kind, readyState: track.readyState, settings: track.getSettings() }));
    });
    assert.ok(report.simulatedCamera.initialTracks.length > 0);
    assert.ok(report.simulatedCamera.initialTracks.every(track => track.kind === 'video' && track.readyState === 'live'));
    await page.locator('#mirror').check();
    const mirrored = await page.locator('#live-video').evaluate(video => getComputedStyle(video).transform);
    await page.locator('#mirror').uncheck();
    const plain = await page.locator('#live-video').evaluate(video => getComputedStyle(video).transform);
    assert.notEqual(mirrored, plain, 'Preview mirror changes CSS'); report.previewMirror = { mirrored, plain };
    console.log('Camera: countdown, capture and manual stop');
    await page.locator('#record').click();
    await page.waitForFunction(() => !document.querySelector('#stop-recording').disabled, null, { timeout: 10000 });
    await page.waitForFunction(() => { const video = document.querySelector('#live-video');
      if (window.__v8CaptureStart === undefined) { window.__v8CaptureStart = video.currentTime; return false; }
      return video.currentTime - window.__v8CaptureStart >= 1.0; }, null, { timeout: 5000 });
    await page.locator('#stop-recording').click();
    await page.waitForFunction(() => !document.querySelector('#save-video').disabled && !document.querySelector('#analyze').disabled, null, { timeout: 25000 });
    const captured = await analyze('simulated-camera-capture', 'webcam');
    assert.ok(captured.data.camera); assert.equal(captured.data.camera.preview_mirrored, false);
    report.cameraRuns.push(captured);
    await screenshot('simulated-camera-result.png');
    const pendingVideo = page.waitForEvent('download'); await page.locator('#save-video').click();
    const videoDownload = await pendingVideo;
    const extension = path.extname(videoDownload.suggestedFilename()); assert.ok(['.webm', '.mp4'].includes(extension));
    const recordedFile = path.join(out, `simulated-camera-recording${extension}`);
    await videoDownload.saveAs(recordedFile);
    assert.equal(sha(await readFile(recordedFile)), captured.data.media.sha256, 'Download contains exact analyzed camera bytes');
    report.simulatedCamera.recordingFile = relative(recordedFile);
    await page.locator('#close-camera').click();
    assert.equal(await page.evaluate(() => window.__v8ObservedTracks.every(track => track.readyState === 'ended')), true);
    assert.equal(await page.locator('#live-video').evaluate(video => video.srcObject), null);
    report.checks.push('Actual simulated-device recording, manual stop, media download and camera track shutdown pass.');
    console.log('Camera: exact downloaded-byte replay');
    await setFile(recordedFile);
    const replayed = await analyze('simulated-camera-byte-identical-replay', 'file', captured.data.media.sha256);
    compareDecisions(captured.data, replayed.data); report.cameraRuns.push(replayed);
    report.checks.push('The downloaded camera recording produces the same decisions/scores through file replay.');

    await page.locator('#clear').click();
    await page.locator('#open-camera').click();
    await page.waitForFunction(() => document.querySelector('#live-video').srcObject !== null, null, { timeout: 15000 });
    await page.locator('#live-video').evaluate(video => { window.__v8ObservedTracks = [...video.srcObject.getTracks()]; });
    console.log('Camera: automatic three-second stop');
    await page.locator('#record').click();
    await page.waitForFunction(() => !document.querySelector('#stop-recording').disabled, null, { timeout: 10000 });
    await page.waitForFunction(() => !document.querySelector('#analyze').disabled, null, { timeout: 25000 });
    const autoDuration = await page.locator('#preview-video').evaluate(video => video.duration);
    assert.ok(autoDuration >= 2.5 && autoDuration <= 3.5, `Actual auto-stopped media duration: ${autoDuration}`);
    report.simulatedCamera.autoStopDurationSeconds = autoDuration;
    await page.locator('#clear').click();
    await page.locator('#record').click();
    await page.locator('#close-camera').click();
    await page.waitForFunction(() => document.querySelector('#live-video').srcObject === null, null, { timeout: 5000 });
    assert.equal(await page.evaluate(() => window.__v8ObservedTracks.every(track => track.readyState === 'ended')), true);
    report.checks.push('Closing camera during countdown stops all real simulated-device tracks.');
    assert.notEqual(await page.locator('#results').getAttribute('data-state'), 'ready');
    // setInputFiles bypasses a disabled input, unlike a user. Wait for the
    // cancelled countdown task to settle before selecting the next file.
    await page.waitForFunction(() => !document.querySelector('#file-input').disabled, null, { timeout: 5000 });
    await page.locator('#file-input').setInputFiles({ name: 'broken.mp4', mimeType: 'video/mp4', buffer: Buffer.from('not a video') });
    await page.waitForFunction(() => Boolean(document.querySelector('#error')?.textContent.trim()), null, { timeout: 20000 });
    assert.equal(await page.locator('#analyze').isDisabled(), true);
    report.checks.push('Clear removes stale results; invalid media is rejected without recognition.');
    // UI contract test only: these synthetic declarations are not saved as a
    // participant report and must never enter an intake/evaluation collection.
    await setFile(fixtures.clips[0].path);
    await page.locator('#blind').check();
    await page.locator('#signer-id').fill('P999999');
    await page.locator('#session-id').fill('QA_UI_ONLY');
    await page.locator('#known-label').selectOption({ label: 'ΓΕΙΑ' });
    await page.locator('#collection-consent').check();
    await page.locator('#analyze').click();
    await page.waitForFunction(() => document.querySelector('#results')?.dataset.state === 'ready', null, { timeout: 180000 });
    assert.equal(await page.locator('#visible-results').isVisible(), false);
    assert.equal(await page.locator('#blind-result').isVisible(), true);
    for (const selector of ['#baseline-result', '#v8-result']) {
      assert.equal((await page.locator(selector).innerText()).trim(), '');
      assert.equal(await page.locator(selector).getAttribute('data-accepted'), null);
      assert.equal(await page.locator(selector).getAttribute('data-label'), null);
    }
    report.blindMode = { tested: true, predictionTextOrDataAttributesPresent: false,
      existingDatasetFixture: true, syntheticMetadataUiTestOnly: true, participantReportDownloaded: false };
    report.checks.push('Blind mode leaves prediction text/data attributes empty; synthetic UI metadata was not exported as participant evidence.');
    await page.setViewportSize({ width: 390, height: 844 });
    report.mobileLayout = await page.evaluate(() => ({ viewportWidth: innerWidth, documentWidth: document.documentElement.scrollWidth,
      overflowingElements: [...document.body.querySelectorAll('*')].flatMap(element => {
        const rect = element.getBoundingClientRect();
        return rect.width && (rect.right > innerWidth + 1 || rect.left < -1)
          ? [{ tag: element.tagName, id: element.id, left: rect.left, right: rect.right, width: rect.width }] : [];
      }) }));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await screenshot('mobile-blind-mode.png');
    await page.setViewportSize({ width: 1400, height: 1050 });
    await page.reload({ waitUntil: 'networkidle' });
    assert.equal(await page.locator('#consent').isChecked(), false);
    assert.equal(await page.locator('#live-video').evaluate(video => video.srcObject), null);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await screenshot('desktop-reloaded.png');
    report.checks.push('Desktop/mobile layouts have no horizontal overflow; reload starts without camera consent or a live stream.');
    console.log('Public view: actual recognition and rejection without debug controls');
    await checkPublicView(fixtures, expectedCsp);
    console.log('Integrity: incorrect config hash fails closed');
    let modifiedConfigResponses = 0;
    await page.route('**/artifacts/v8/browser/model-config.json*', async route => {
      const response = await route.fetch();
      if ((response.headers()['content-type'] ?? '').includes('application/json')) {
        modifiedConfigResponses++;
        await route.fulfill({ response, body: Buffer.concat([await response.body(), Buffer.from('\n ')]) });
      } else await route.fulfill({ response });
    });
    await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
    await setFile(fixtures.clips[0].path);
    await page.locator('#analyze').click();
    await page.waitForFunction(() => Boolean(document.querySelector('#error')?.textContent.trim()), null, { timeout: 20000 });
    assert.equal(modifiedConfigResponses, 1, 'Intercept only the actual frozen config JSON');
    assert.notEqual(await page.locator('#results').getAttribute('data-state'), 'ready');
    const integrityError = await page.locator('#error').innerText();
    assert.match(integrityError, /παγωμένη έκδοση/);
    report.configIntegrityNegative = { modifiedConfigResponses, resultProduced: false, error: integrityError,
      responseOnlyMutation: true, diskFilesChanged: false, recognitionResultMocked: false };
    await screenshot('wrong-config-hash.png');
    report.checks.push('Changing only the served frozen config JSON bytes fails SHA verification and produces no recognition result.');
    await page.goto(appUrl, { waitUntil: 'networkidle', timeout: 60000 });
    await setFile(fixtures.clips[0].path);
    await page.locator('#analyze').click();
    await page.waitForFunction(() => Boolean(document.querySelector('#error')?.textContent.trim()), null, { timeout: 20000 });
    assert.equal(modifiedConfigResponses, 2, 'The same integrity failure reached both UI modes');
    assert.notEqual(await page.locator('#results').getAttribute('data-state'), 'ready');
    const appIntegrityError = await page.locator('#error').innerText();
    assert.equal(appIntegrityError, 'Δεν ήταν δυνατή η ανάλυση. Δοκιμάστε ξανά.');
    report.app.configIntegrityNegative = { resultProduced: false, publicError: appIntegrityError };
    report.checks.push('Public mode fails closed on the same modified configuration and shows a concise non-technical error.');
    report.requests = [...requests];
    report.requestMetadata = requestMethods;
    assert.ok(workerUrls.size >= 2, 'Classifier and recovery workers were actually created');
    for (const workerUrl of workerUrls) {
      assert.equal(new URL(workerUrl).origin, new URL(base).origin, 'Every observed worker entry is local');
      const workerResponse = await context.request.head(workerUrl);
      assert.ok(workerResponse.ok());
      assert.equal(workerResponse.headers()['content-security-policy'], expectedCsp, 'Worker response enforces the same network policy');
      report.networkPolicyHeaders.push({ url: workerUrl, contentSecurityPolicy: workerResponse.headers()['content-security-policy'] });
    }
    const externalRequests = requestMethods.filter(request => new URL(request.url).origin !== new URL(base).origin);
    report.cspConsoleMessages = cspMessages;
    report.externalRequestAttempts = externalRequests.map(request => ({ ...request,
      explicitlyBlockedByCsp: Boolean(request.failure && request.responseStatus === null &&
        (/BLOCKED_BY_CSP/.test(request.failure) || cspMessages.some(message => message.includes(request.url) && /Content Security Policy|connect-src/.test(message)))) }));
    assert.ok(report.externalRequestAttempts.every(request => request.explicitlyBlockedByCsp), 'Every external attempt must have explicit browser CSP-block evidence and no response');
    assert.ok(requestMethods.filter(request => new URL(request.url).origin === new URL(base).origin).every(request => ['GET', 'HEAD'].includes(request.method)), 'No local media or metadata POSTs');
    report.noMediaOrMetadataPosts = true;
    report.noUnblockedExternalRequests = true;
    report.checks.push('Pilot HTML and every observed worker entry enforce local connect-src; any telemetry attempt is explicitly blocked by browser CSP with no response.');
    report.sourceFilesUnchangedDuringRun = true;
    for (const [source, digest] of Object.entries(report.sourceFileSha256)) {
      assert.equal(sha(await readFile(path.join(root, source))), digest, `Test source changed while running: ${source}`);
    }
    assert.deepEqual(report.errors, []);
    report.passed = true;
  }
} catch (error) {
  failure = error; report.failure = String(error.stack ?? error);
  if (page && !page.isClosed()) report.failureUiState = await page.evaluate(() => ({
    status: document.querySelector('#status')?.textContent, error: document.querySelector('#error')?.textContent,
    cameraOpen: Boolean(document.querySelector('#live-video')?.srcObject),
    resultsState: document.querySelector('#results')?.dataset.state,
  })).catch(() => null);
  if (page) await screenshot('failure.png').catch(() => {});
} finally {
  if (page && !page.isClosed()) await page.locator('#live-video').evaluate(video => {
    if (video.srcObject) for (const track of video.srcObject.getTracks()) track.stop();
  }).catch(() => {});
  if (browser) await browser.close();
  if (!args.includes('--prepare-only') || failure) {
    await mkdir(out, { recursive: true });
    await writeFile(path.join(out, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  }
}
if (!args.includes('--prepare-only') || failure) console.log(JSON.stringify({ passed: report.passed, realMp4Runs: report.realVideoRuns.length,
  appRuns: report.app?.runs.length ?? 0, cameraRuns: report.cameraRuns.length, humanWebcamTested: false,
  report: path.join(out, 'report.json'), failure: report.failure }));
if (failure) throw failure;
