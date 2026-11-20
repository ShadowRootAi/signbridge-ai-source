/** Diagnostic replay of actual decoded frames through the frozen OpenHands browser worker. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const output = resolve(root, 'reports/openhands-decoding');
const media = resolve(output, 'fixtures');
const dist = resolve(root, 'publication-recognition-200/dist');
const host = '127.0.0.1';
const port = Number(process.argv[2] ?? 5203);
const origin = `http://${host}:${port}`;
assert(Number.isSafeInteger(port) && port > 0 && port < 65536);
const headers = JSON.parse(await readFile(resolve(root, 'web/deployment/headers.json'), 'utf8'));
const cases = JSON.parse(await readFile(resolve(root, 'reports/judge-robustness/manifest.json'), 'utf8')).clips;
const selected = cases.filter(row => ['geia', 'efcharisto', 'entaxei', 'unknown', 'frozen'].includes(row.id));
assert.equal(selected.length, 5);
const arms = [
  { name: 'current', offset: 0, dedup: false },
  { name: 'seek-plus-2ms', offset: .002, dedup: false },
  { name: 'seek-plus-2ms-collapse-identical-pixels', offset: .002, dedup: true },
];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
await mkdir(media, { recursive: true });
const ffmpeg = execFileSync(resolve(root, '.venv/Scripts/python.exe'), ['-X', 'utf8', '-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
  { encoding: 'utf8', windowsHide: true }).trim();
const fixtures = new Map();
for (const row of selected) {
  const original = resolve(root, 'reports/judge-robustness', row.path);
  assert.equal(digest(await readFile(original)), row.sha256);
  // Any lower-frame-rate variants are engineering controls paired to their own
  // transcoded copy. They are not additional independent utterances.
  for (const fps of ['geia', 'entaxei', 'unknown'].includes(row.id) ? [30, 24, 15] : [30]) {
    const filename = `${row.id}-${fps}.mp4`;
    const target = fps === 30 ? original : resolve(media, filename);
    if (fps !== 30) execFileSync(ffmpeg, ['-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', original, '-an',
      '-vf', `fps=${fps}`, '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', target],
    { windowsHide: true });
    // A trusted decoder gives the expected number of presented frames per file.
    const diagnostic = spawnSync(ffmpeg, ['-nostdin', '-hide_banner', '-loglevel', 'info', '-i', target, '-an',
      '-vf', 'showinfo', '-f', 'null', '-'], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'], encoding: 'utf8' });
    assert.equal(diagnostic.status, 0, diagnostic.stderr);
    const pts = [...diagnostic.stderr.matchAll(/\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([\d.eE+-]+)/g)]
      .map(item => Number(item[1]));
    assert(pts.length >= 2);
    fixtures.set(filename, { ...row, filename, fps, bytes: await readFile(target), pts });
  }
}

const types = new Map([['.html', 'text/html'], ['.js', 'text/javascript'], ['.mjs', 'text/javascript'],
  ['.css', 'text/css'], ['.mp4', 'video/mp4'], ['.json', 'application/json'], ['.wasm', 'application/wasm']]);
const server = createServer(async (req, res) => {
  try {
    assert(['GET', 'HEAD'].includes(req.method));
    const path = new URL(req.url, origin).pathname;
    if (path.startsWith('/qa-decoding/')) {
      const sample = fixtures.get(basename(path));
      if (!sample) throw new Error('Missing fixture');
      res.writeHead(200, { ...headers, 'Content-Type': 'video/mp4', 'Content-Length': sample.bytes.length });
      res.end(req.method === 'HEAD' ? undefined : sample.bytes);
      return;
    }
    const file = resolve(dist, '.' + (path === '/' ? '/index.html' : path));
    if (!file.startsWith(dist + sep)) throw new Error('Outside release path');
    const stats = await stat(file);
    if (!stats.isFile()) throw new Error('Not a release file');
    res.writeHead(200, { ...headers, 'Content-Type': types.get(extname(path === '/' ? 'index.html' : path))
      ?? 'application/octet-stream', 'Content-Length': stats.size });
    if (req.method === 'HEAD') res.end();
    else createReadStream(file).pipe(res);
  } catch {
    if (!res.headersSent) res.writeHead(404);
    res.end();
  }
});
await new Promise((ok, fail) => { server.once('error', fail); server.listen(port, host, ok); });
const results = [];
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const requested = new Set();
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== origin) requested.add(request.url());
  });
  await page.addInitScript(() => {
    const time = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'currentTime');
    Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
      ...time, set(seconds) {
        const probe = window.decodingProbe;
        if (probe?.active && this === document.querySelector('#recognition-video')) {
          seconds = Math.min(seconds + probe.arm.offset, this.duration - .000001);
        }
        time.set.call(this, seconds);
      },
    });
    const draw = OffscreenCanvasRenderingContext2D.prototype.drawImage;
    OffscreenCanvasRenderingContext2D.prototype.drawImage = function (...args) {
      const response = draw.apply(this, args);
      const probe = window.decodingProbe;
      const video = args[0];
      if (probe?.active && video instanceof HTMLVideoElement && video.id === 'recognition-video') {
        const rgba = this.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
        const hash = crypto.subtle.digest('SHA-256', rgba)
          .then(bytes => Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join(''));
        probe.pixels.push({ targetTime: video.currentTime, hash });
      }
      return response;
    };
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message, ...args) {
      const probe = window.decodingProbe;
      if (probe?.active && Array.isArray(message?.frames) && typeof message?.allowRecovery === 'boolean') {
        const instance = this;
        const observe = event => {
          if (event.data?.id !== message.id) return;
          instance.removeEventListener('message', observe);
          probe.classification = event.data.result ?? { error: event.data.error };
        };
        const send = (pixels) => {
          probe.payload = { pixels, frameTimes: message.frames.map(f => f.t) };
          const frames = probe.arm.dedup
            ? message.frames.filter((_frame, i) => i === 0 || pixels[i].hash !== pixels[i - 1].hash)
            : message.frames;
          probe.classifierFrameTimes = frames.map(f => f.t);
          instance.addEventListener('message', observe);
          post.call(instance, { ...message, frames }, ...args);
        };
        Promise.all(probe.pixels.map(async row => ({ targetTime: row.targetTime, hash: await row.hash })))
          .then(pixels => {
            if (pixels.length !== message.frames.length) throw new Error('Decoded frames do not match worker input');
            send(pixels);
          }).catch(error => { probe.instrumentationError = String(error); });
      } else {
        if (probe?.active && message.kind === 'resetDetector') {
          const started = performance.now();
          const listener = event => {
            if (event.data?.id !== message.id) return;
            this.removeEventListener('message', listener);
            probe.detectorInitializationMs = performance.now() - started;
          };
          this.addEventListener('message', listener);
        }
        post.call(this, message, ...args);
      }
    };
  });
  await page.goto(origin + '/#/try');
  const result = page.locator('#recognition-result');
  const analyze = page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true });
  for (const fixture of fixtures.values()) {
    for (const arm of arms) {
      await page.locator('#video-file').setInputFiles({
        name: fixture.filename, mimeType: 'video/mp4', buffer: fixture.bytes,
      });
      await page.getByText(fixture.filename, { exact: true }).waitFor();
      await page.evaluate(arm => {
        window.decodingProbe = { active: true, arm, pixels: [] };
      }, arm);
      const t0 = performance.now();
      await analyze.click();
      await page.waitForFunction(() => !!window.decodingProbe.classification || !!window.decodingProbe.instrumentationError,
        null, { timeout: 35000 });
      await result.waitFor();
      const elapsedMs = Math.round(performance.now() - t0);
      const observed = await page.evaluate(() => {
        const probe = window.decodingProbe;
        probe.active = false;
        return { ...probe.payload, classifierFrameTimes: probe.classifierFrameTimes,
          classification: probe.classification, instrumentationError: probe.instrumentationError,
          detectorInitializationMs: probe.detectorInitializationMs };
      });
      assert(!observed.instrumentationError, observed.instrumentationError);
      assert(!observed.classification.error, observed.classification.error);
      const duplicates = observed.pixels.slice(1)
        .reduce((n, frame, i) => n + Number(frame.hash === observed.pixels[i].hash), 0);
      const label = await result.innerText();
      assert.equal(await result.getAttribute('data-accepted'), String(observed.classification.accepted));
      const record = {
        fixture: fixture.filename, fps: fixture.fps, sourceId: fixture.source_id, sourceSha256: fixture.sha256,
        referenceGloss: fixture.gloss,
        fixtureSha256: digest(fixture.bytes), comparison: arm.name, output: label, adjacentIdenticalPixels: duplicates,
        framesInFfmpeg: fixture.pts.length, analyzedFrames: observed.pixels.length,
        classifierInputFrames: observed.classifierFrameTimes.length,
        distinctPixelHashes: new Set(observed.pixels.map(row => row.hash)).size, elapsedMs, ...observed,
      };
      results.push(record);
      console.log(JSON.stringify({ fixture: record.fixture, arm: record.comparison, output: record.output,
        frames: record.analyzedFrames, duplicates, candidate: observed.classification.topCandidates[0]?.gloss }));
    }
  }
  assert.deepEqual([...requested], []);
} finally {
  await browser?.close();
  server.closeAllConnections();
  await new Promise(ok => server.close(ok));
  await writeFile(resolve(output, 'browser-sampling.json'), JSON.stringify({
    status: results.length === arms.length * fixtures.size ? 'complete' : 'partial',
    browser: browser?.version(), origin, results,
    interpretation: 'Reused fixtures and explicitly re-encoded control copies. Pixel readback affects timings. No generalized accuracy or camera claim.',
  }, null, 2) + '\n');
}
