/** Paired Chrome test on the unchanged OpenHands model and policy. */
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const script = fileURLToPath(import.meta.url);
const web = resolve(dirname(script), '..');
const root = dirname(web);
const edition = resolve(root, 'publication-judge-demo');
const reportPath = resolve(root, 'reports/judge-ai-development/temporal.json');
const origin = 'http://127.0.0.1:5332';
const modelPath = '/__signbridge/openhands/';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const run = promisify(execFile);
const report = { passed: false, scope: 'Paired test on previously examined labeled GSL MP4s, author-timed Sartinas excerpts, '
  + 'and clearly labeled derived controls. All real predictions use the frozen judge model and acceptance thresholds. '
  + 'The crop uses landmark motion only and the shuffled landmark sequence preserves timestamps and tracking availability. '
  + 'No generalization or webcam accuracy is estimated.', rows: [], sourceManifests: {}, externalRequests: [], pageErrors: [] };
const padding = 'tpad=start_mode=clone:start_duration=0.5:stop_mode=clone:stop_duration=0.5';
const browsers = [];
let server;
let failure;

async function loadManifest(path, subgroup, count) {
  const file = resolve(root, path);
  const bytes = await readFile(file);
  const rows = JSON.parse(bytes).clips;
  assert.equal(rows.length, count);
  report.sourceManifests[path] = digest(bytes);
  return rows.map(row => ({ ...row, subgroup, file: resolve(dirname(file), row.path) }));
}

async function derivedPadding(samples) {
  const work = await mkdtemp(resolve(root, '.runtime-tmp/judge-temporal-'));
  const { stdout } = await run(resolve(root, '.venv/Scripts/python.exe'),
    ['-B', '-X', 'utf8', '-c', 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'], { windowsHide: true });
  const ffmpeg = stdout.trim();
  const padded = [];
  for (const row of samples.filter(row => ['geia', 'efcharisto', 'entaxei', 'unknown'].includes(row.id))) {
    assert.equal(digest(await readFile(row.file)), row.sha256);
    const file = resolve(work, `${row.id}-padded.mp4`);
    await run(ffmpeg, ['-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror', '-y', '-i', row.file, '-an',
      '-vf', padding, '-r', '30', '-c:v', 'libx264', '-threads', '1', '-crf', '17', '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart', file], { timeout: 30000, windowsHide: true });
    padded.push({ ...row, subgroup: row.on_roster ? 'padded_known' : 'padded_unsupported', file, id: `${row.id}-padded`,
      sha256: digest(await readFile(file)), derivedFrom: { id: row.id, sha256: row.sha256, padding } });
  }
  return padded;
}

async function openHarness(assets) {
  const browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
  browsers.push(browser);
  report.browser = browser.version();
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route(/^https?:/, route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) {
      report.externalRequests.push(url.origin);
      return route.abort('blockedbyclient');
    }
    const asset = assets.get(url.pathname);
    return asset ? route.fulfill({ body: asset.body, contentType: asset.type }) : route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.goto(`${origin}/extract.html`);
  await page.waitForFunction(() => Boolean(window.signbridge));
  const parameters = await page.evaluate(async modelPath => {
    const [{ collectVideoLandmarks }, { loadLocalVideo, localVideoUrl }, { loadOpenHandsRecognizer },
      { shuffleLandmarkOrder, selectMotionWindow, temporalExperiment }, { measureRawTracking }] = await Promise.all([
      import('/src/pipeline.ts'), import('/src/local-video.ts'), import('/src/openhands-recognizer.ts'),
      import('/src/judge-temporal-experiment.ts'), import('/src/tracking.ts'),
    ]);
    const engine = await loadOpenHandsRecognizer({ modelUrl: modelPath + 'stgcn.onnx', secondaryModelUrl: modelPath + 'lstm.onnx',
      configUrl: modelPath + 'stgcn-onnx.json', policyUrl: modelPath + 'policy.json' });
    const input = document.createElement('input');
    input.type = 'file';
    document.body.append(input);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    document.body.append(video);
    window.testTemporal = async () => {
      const file = input.files[0];
      if (!file) throw new Error('Missing MP4');
      const signal = AbortSignal.timeout(120_000);
      const url = localVideoUrl(file);
      try {
        await loadLocalVideo(video, url, signal);
        const { frames, start } = await collectVideoLandmarks(video, undefined, signal, true);
        const landmarkMs = performance.now() - start;
        const tracking = JSON.stringify(measureRawTracking(frames));
        const classify = async frames => {
          const began = performance.now();
          const result = await engine.classify(frames, true);
          return { ...result, elapsedMs: performance.now() - began };
        };
        const baseline = await classify(frames);
        const shuffle = [];
        for (const seed of temporalExperiment.seeds) {
          const shuffled = shuffleLandmarkOrder(frames, seed);
          if (JSON.stringify(measureRawTracking(shuffled.frames)) !== tracking
            || new Set(shuffled.permutation).size !== frames.length
            || shuffled.frames.some((frame, slot) => frame.t !== frames[slot].t)) {
            throw new Error('Shuffle did not preserve the tracking evidence and time slots');
          }
          shuffle.push({ seed, changedSlots: shuffled.changedSlots, trackingPreserved: true,
            result: await classify(shuffled.frames) });
        }
        const window = selectMotionWindow(frames);
        const crop = window.reason === 'window' ? await classify(frames.slice(window.first, window.last + 1)) : null;
        const useCrop = Boolean(!baseline.accepted && baseline.classifierExecuted && crop?.accepted);
        return { frames: frames.length, duration: video.duration, landmarkMs, baseline, shuffle, window, crop,
          decision: { branch: useCrop ? 'crop' : 'baseline', result: useCrop ? crop : baseline } };
      } finally {
        input.value = '';
        video.removeAttribute('src');
        video.load();
        URL.revokeObjectURL(url);
      }
    };
    return temporalExperiment;
  }, modelPath);
  report.parameters = parameters;
  return page;
}

async function extract(page, { id, subgroup, signer, gloss, sha256, on_roster: known, file, derivedFrom }) {
  const bytes = await readFile(file);
  assert.equal(digest(bytes), sha256, `MP4 integrity: ${id}`);
  await page.locator('input[type=file]').setInputFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: bytes });
  const result = await page.evaluate(() => window.testTemporal());
  return { id, subgroup, signer, gloss, known, sha256, derivedFrom, ...result };
}

function summarize(rows, fetchResult) {
  const counts = { known: 0, unknown: 0, correct: 0, wrong: 0, falseAccept: 0, rejected: 0 };
  for (const row of rows) {
    if (row.known) counts.known++; else counts.unknown++;
    const result = fetchResult(row);
    if (!result.accepted) counts.rejected++;
    else if (!row.known) counts.falseAccept++;
    else if (result.label === row.gloss) counts.correct++;
    else counts.wrong++;
  }
  return counts;
}

try {
  const audit = JSON.parse(await readFile(resolve(edition, 'library-audit.json')));
  assert.equal(audit.status, 'local_judge_demo');
  assert.equal(audit.openhands_frame_boundary_offset_ms, 2);
  const assets = new Map();
  report.artifacts = {};
  for (const filename of ['stgcn.onnx', 'lstm.onnx', 'stgcn-onnx.json', 'policy.json']) {
    const path = modelPath + filename;
    const body = await readFile(resolve(edition, 'dist', path.slice(1)));
    const sha256 = digest(body);
    assert.equal(sha256, audit.runtime_files[path.slice(1)], filename);
    report.artifacts[filename] = sha256;
    assets.set(path, { body, type: filename.endsWith('.onnx') ? 'application/octet-stream' : 'application/json' });
  }
  report.experimentSha256 = digest(await readFile(resolve(web, 'src/judge-temporal-experiment.ts')));
  report.scriptSha256 = digest(await readFile(script));
  const fixtures = await loadManifest('reports/judge-robustness/manifest.json', 'reference_and_controls', 6);
  const regression = await loadManifest('reports/openhands-engine/replay-manifest.json', 'known_regression', 24);
  const outside = await loadManifest('data/isolated-sartinas-outside-roster/manifest.json', 'outside_regression', 24);
  const samples = [...fixtures, ...await derivedPadding(fixtures), ...regression, ...outside];
  const roster = new Set(JSON.parse(assets.get(modelPath + 'policy.json').body).supportedLabels);
  assert(samples.every(clip => roster.has(clip.gloss) === clip.on_roster));
  assert.equal(new Set(samples.map(row => row.id)).size, samples.length);
  assert.equal(new Set(samples.map(row => row.sha256)).size, samples.length);

  process.chdir(web);
  server = await createServer({ root: web, configFile: resolve(web, 'vite.config.ts'), logLevel: 'error',
    server: { host: '127.0.0.1', port: 5332, strictPort: true, hmr: false, watch: null } });
  await server.listen();
  const first = await openHarness(assets);
  report.rows.push(await extract(first, samples[0]));
  const pages = [first, await openHarness(assets)];
  for (let start = 1; start < samples.length; start += pages.length) {
    const rows = await Promise.all(samples.slice(start, start + pages.length).map((row, index) => extract(pages[index], row)));
    for (const row of rows) {
      report.rows.push(row);
      console.log(`${report.rows.length}/${samples.length} ${row.id}: baseline=${row.baseline.label ?? row.baseline.rejectionCode}; window=${row.window.reason}; crop=${row.crop?.label ?? row.crop?.rejectionCode ?? 'fallback'}`);
    }
  }
  assert.deepEqual(report.externalRequests, []);
  assert.deepEqual(report.pageErrors, []);
  const subgroups = [...new Set(report.rows.map(row => row.subgroup))];
  report.summary = Object.fromEntries(subgroups.map(group => {
    const rows = report.rows.filter(row => row.subgroup === group);
    return [group, { baseline: summarize(rows, row => row.baseline),
      shuffled: report.parameters.seeds.map((seed, index) => ({ seed, ...summarize(rows, row => row.shuffle[index].result) })),
      cropWithFallback: summarize(rows, row => row.crop ?? row.baseline),
      recoveryOnly: summarize(rows, row => row.decision.result), windows: rows.filter(row => row.window.reason === 'window').length,
      recoveries: rows.filter(row => row.decision.branch === 'crop').length }];
  }));
  report.passed = true;
} catch (error) {
  failure = error;
  report.failure = String(error.stack || error);
} finally {
  try { await Promise.all(browsers.map(browser => browser.close())); }
  finally {
    await server?.close();
    await mkdir(dirname(reportPath), { recursive: true });
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  }
}
console.log(JSON.stringify({ passed: report.passed, summary: report.summary, reportPath }));
if (failure) throw failure;
