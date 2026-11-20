import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const origin = new URL(process.argv[2] ?? 'http://127.0.0.1:5191').origin;
const fixtures = resolve(project, 'reports/v8/app-ui-qa/fixtures');
const evidence = resolve(project, 'reports/app-release');
const audit = await readFile(resolve(project, 'publication-app/publication-audit.json'));
const report = {
  passed: false,
  buildAuditSha256: createHash('sha256').update(audit).digest('hex'),
  origin,
  browser: '',
  simulatedCamera: 'Real GSL file replayed by the Chrome fake camera device',
  humanWebcamTested: false,
  navigation: [], library: [], recognition: [], camera: {}, remoteRequests: [], pageErrors: [],
};
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: [
  '--use-fake-device-for-media-stream',
  `--use-file-for-fake-video-capture=${resolve(fixtures, 'real-gsl-simulated-camera.y4m')}`,
] });

try {
  report.browser = browser.version();
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  await context.grantPermissions(['camera'], { origin });
  context.on('request', request => {
    if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== origin) report.remoteRequests.push(request.url());
  });
  page.on('pageerror', error => report.pageErrors.push(error.message));
  const navigate = async name => {
    const link = page.getByRole('navigation', { name: 'Κύρια πλοήγηση' }).getByRole('link', { name, exact: true });
    if (!await link.isVisible()) await page.getByRole('button', { name: 'Άνοιγμα μενού' }).click();
    await link.click();
    await page.locator('#main-content h1').waitFor();
    await page.waitForFunction(label => document.querySelector('#main-navigation [aria-current="page"]')?.textContent === label, name);
  };
  const screenshot = name => page.screenshot({ path: resolve(evidence, `${name}.png`), fullPage: true });

  const start = await page.goto(`${origin}/?debug=1`);
  assert.equal(start.status(), 200);
  assert.match(start.headers()['content-security-policy'], /connect-src 'self' blob:/);
  assert.equal(await page.evaluate(() => crossOriginIsolated), true);
  await page.getByRole('heading', { level: 1, name: /Η επικοινωνία/ }).waitFor();
  await screenshot('home-desktop');

  await navigate('Βιβλιοθήκη');
  await page.locator('[data-library-entry]').first().waitFor();
  assert.equal(await page.locator('[data-library-entry]').count(), 3);
  for (const id of ['geia', 'efcharisto', 'entaxei']) {
    const response = await context.request.get(`${origin}/library/${id}.mp4`);
    assert.equal(response.status(), 200);
    assert.match(response.headers()['content-type'], /video\/mp4/);
    report.library.push({ id, bytes: (await response.body()).length });
  }
  await page.locator('#phrase').fill('ΕΥΧΑΡΙΣΤΏ');
  await page.getByRole('button', { name: 'Αναζήτηση', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#library-video')?.currentSrc.endsWith('/efcharisto.mp4'));
  await page.getByRole('button', { name: 'Αναπαραγωγή από την αρχή' }).click();
  await page.waitForFunction(() => document.querySelector('#library-video')?.currentTime > 0);
  await page.getByText('Πηγή και άδεια', { exact: true }).click();
  assert.ok(await page.getByRole('link', { name: /Creative Commons/ }).isVisible());
  await screenshot('library-desktop');
  await page.locator('#phrase').fill('φράση εκτός καταλόγου');
  await page.getByRole('button', { name: 'Αναζήτηση', exact: true }).click();
  await page.getByText('Δεν βρέθηκε απόδοση', { exact: true }).waitFor();

  await navigate('Δοκιμή βίντεο');
  const analysis = page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true });
  for (const [file, accepted, label] of [
    ['known.mp4', true, /^Γεια$/i],
    ['hard_unknown.mp4', false, /^Δεν αναγνωρίστηκε$/],
  ]) {
    await page.locator('#video-file').setInputFiles(resolve(fixtures, file));
    await analysis.click();
    const result = page.locator('#recognition-result');
    await result.waitFor({ timeout: 60_000 });
    assert.equal(await result.getAttribute('data-accepted'), String(accepted));
    const text = (await result.innerText()).trim();
    assert.match(text, label);
    report.recognition.push({ file, accepted, text });
    assert.doesNotMatch(await page.locator('main').innerText(), /Βαθμολογία μοντέλου|Λήψη αναφοράς JSON|Πληρότητα δεδομένων λήψης/);
    if (accepted) await screenshot('recognition-desktop');
  }

  // Pages can serve MP4 requests as full HTTP 200 responses without byte ranges.
  const clipUrl = `${origin}/library/entaxei.mp4`;
  const clipBytes = await readFile(resolve(project, 'web/public/library/entaxei.mp4'));
  const exampleRequests = [];
  const serveWholeClip = route => {
    const request = route.request();
    exampleRequests.push({ type: request.resourceType(), range: request.headers().range ?? null });
    return route.fulfill({ status: 200, contentType: 'video/mp4', body: clipBytes });
  };
  await page.route(clipUrl, serveWholeClip);
  const frameProbe = await page.evaluateHandle(() => {
    const original = OffscreenCanvas.prototype.convertToBlob;
    const frames = [];
    OffscreenCanvas.prototype.convertToBlob = function (...args) {
      const encoded = original.apply(this, args);
      const video = document.querySelector('#recognition-video');
      if (video?.currentSrc) {
        const time = video.currentTime;
        frames.push(encoded.then(blob => blob.arrayBuffer()).then(async bytes => ({
          time,
          hash: Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))).join(','),
        })));
      }
      return encoded;
    };
    return { frames, restore: () => { OffscreenCanvas.prototype.convertToBlob = original; } };
  });
  const recognizeClip = async choose => {
    const before = await page.locator('#recognition-video').evaluate(video => video.currentSrc);
    await choose();
    await page.waitForFunction(previous => {
      const video = document.querySelector('#recognition-video');
      return video?.currentSrc.startsWith('blob:') && video.currentSrc !== previous && video.readyState >= 2;
    }, before);
    await analysis.click();
    const result = page.locator('#recognition-result');
    await result.waitFor({ timeout: 60_000 });
    const frames = await frameProbe.evaluate(probe => Promise.all(probe.frames.splice(0)));
    assert.ok(frames.length > 1 && frames.every(({ time }, i) => i === 0 || time > frames[i - 1].time),
      'Inference must seek through the clip instead of staying on the first frame');
    assert.ok(new Set(frames.map(frame => frame.hash)).size > 1, 'Inference must receive distinct decoded images');
    return {
      accepted: await result.getAttribute('data-accepted'),
      text: (await result.innerText()).trim(),
      frames: frames.length,
    };
  };
  try {
    const example = await recognizeClip(() => page.getByRole('button', { name: 'εντάξει', exact: true }).click());
    assert.deepEqual(exampleRequests, [{ type: 'fetch', range: null }]);
    const file = await recognizeClip(() => page.locator('#video-file').setInputFiles({
      name: 'entaxei.mp4', mimeType: 'video/mp4', buffer: clipBytes,
    }));
    assert.deepEqual(example, file, 'The same MP4 bytes must give the same outcome as a catalog example and an upload');
    report.catalogRecognition = { wholeFileResponse: true, example, file };
  } finally {
    await frameProbe.evaluate(probe => probe.restore());
    await frameProbe.dispose();
    await page.unroute(clipUrl, serveWholeClip);
  }

  await navigate('Κάμερα');
  const open = page.getByRole('button', { name: 'Άνοιγμα κάμερας', exact: true });
  assert.equal(await open.isEnabled(), false);
  await page.getByRole('checkbox', { name: /άδεια για καταγραφή εικόνας/i }).check();
  await open.click();
  await page.waitForFunction(() => !!document.querySelector('#live-camera')?.srcObject);
  await page.evaluate(() => { window.qaTracks = document.querySelector('#live-camera').srcObject.getTracks(); });
  await page.getByRole('button', { name: 'Καταγραφή', exact: true }).click();
  await analysis.click({ timeout: 20_000 });
  const cameraResult = page.locator('#recognition-result');
  await cameraResult.waitFor({ timeout: 60_000 });
  report.camera = {
    result: (await cameraResult.innerText()).trim(),
    tracksEnded: await page.evaluate(() => window.qaTracks.every(track => track.readyState === 'ended')),
  };
  assert.equal(report.camera.tracksEnded, true);

  await navigate('Βοήθεια');
  await page.getByText('Τι σημαίνει «Δεν αναγνωρίστηκε»;', { exact: true }).click();
  await page.getByText(/Η λήψη μπορεί να ήταν ασαφής/).waitFor();
  await screenshot('help-desktop');
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    for (const route of ['Αρχική', 'Κάμερα', 'Βιβλιοθήκη', 'Δοκιμή βίντεο', 'Βοήθεια']) {
      await navigate(route);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true, `${route} at ${width}px`);
      report.navigation.push({ route, width });
      if (route === 'Αρχική' && width === 390) await screenshot('home-mobile');
    }
  }
  assert.equal(await page.evaluate(() => document.getAnimations().length), 0);
  assert.equal((await context.request.get(`${origin}/NOTICE.txt`)).status(), 200);
  for (const privatePath of ['/v8-pilot.html', '/reports/technical-report.json', '/__signbridge/webcam-capture/status']) {
    assert.equal((await context.request.get(`${origin}${privatePath}`)).status(), 404);
  }
  assert.deepEqual(report.remoteRequests, []);
  assert.deepEqual(report.pageErrors, []);
  report.passed = true;
  console.log('Full app passed Chrome QA with all five routes, the library, V8 file recognition and simulated camera.');
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser.close();
  await writeFile(resolve(evidence, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
}
