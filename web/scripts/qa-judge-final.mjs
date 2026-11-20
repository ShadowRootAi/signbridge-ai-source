/** Run the frozen judge edition on a fresh loopback preview and report release QA. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const scripts = dirname(fileURLToPath(import.meta.url));
const web = dirname(scripts);
const project = dirname(web);
const edition = resolve(project, 'publication-judge-demo');
const catalog = resolve(edition, 'dist/library/catalog.json');
const output = resolve(project, 'reports/judge-final');
const port = Number(process.argv[2] ?? 5289);
assert(Number.isInteger(port) && port > 0 && port <= 65535, 'Pass a valid port');
const origin = `http://127.0.0.1:${port}`;
const libraryReport = resolve(output, 'library.json');
const readJson = async path => JSON.parse(await readFile(path));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { passed: false, origin, scope: 'Local frozen judge package. Real desktop Chrome, with mobile viewport/touch emulation. Preselected MP4s and catalog checks.' };
let preview;
let previewError;
let log = '';
let failure;
await mkdir(output, { recursive: true });

async function script(name, args) {
  const child = spawn(process.execPath, [resolve(scripts, name), ...args],
    { cwd: project, stdio: 'inherit', windowsHide: true });
  const [code, signal] = await once(child, 'exit');
  assert.equal(code, 0, `${name} failed (${signal ?? code})`);
}

try {
  const audit = await readJson(resolve(edition, 'library-audit.json'));
  assert.equal(audit.status, 'local_judge_demo');
  assert.equal(audit.entries, 3000);
  assert.equal(audit.openhands_frame_boundary_offset_ms, 2);

  preview = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview',
    '--config', 'vite.v8-public.config.ts', '--mode', 'signbridge-openhands',
    '--outDir', '../publication-judge-demo/dist', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  { cwd: web, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  preview.on('error', error => { previewError = error; });
  for (const stream of [preview.stdout, preview.stderr]) stream.on('data', chunk => { log = (log + chunk).slice(-5000); });
  for (let attempt = 0; attempt < 50 && !log.includes(origin); attempt++) {
    if (previewError) throw previewError;
    assert.equal(preview.exitCode, null, log);
    await delay(200);
  }
  assert(log.includes(origin) && preview.exitCode === null, `Fresh preview did not start: ${log}`);

  const assets = ['index.html', 'library/catalog.json', '__signbridge/judge/sources.json', '__signbridge/judge/frozen.mp4',
    '__signbridge/judge/no-tracking.mp4', '__signbridge/judge/unsupported.mp4', '__signbridge/openhands/policy.json',
    '__signbridge/openhands/stgcn.onnx', '__signbridge/openhands/lstm.onnx'];
  report.checkedAssets = Object.fromEntries(await Promise.all(assets.map(async path => {
    const expected = path === 'library/catalog.json' ? audit.catalog_sha256 : audit.runtime_files[path];
    const response = await fetch(`${origin}/${path}`, { redirect: 'error', signal: AbortSignal.timeout(10000) });
    assert(response.ok && typeof expected === 'string', path);
    const hash = digest(Buffer.from(await response.arrayBuffer()));
    assert.equal(hash, expected, `Served bytes do not match the packaged audit: ${path}`);
    return [path, hash];
  })));
  const policy = await readJson(resolve(edition, 'dist/__signbridge/openhands/policy.json'));
  report.activeLabels = policy.supportedLabels.length;
  assert.equal(report.activeLabels, 200);
  const judgeArgs = [origin, '--frame-offset', '--output-dir', output];
  await script('qa-judge-demo.mjs', judgeArgs);
  await script('qa-judge-demo.mjs', [...judgeArgs, '--mobile']);
  await script('qa-expanded-library.mjs', [origin, libraryReport, resolve(output, 'library.png'), catalog]);

  report.recognition = [];
  for (const file of ['judge-panel-frame-offset.json', 'judge-panel-frame-offset-mobile.json']) {
    const check = await readJson(resolve(output, file));
    assert(check.passed);
    report.recognition.push({ file, browser: check.browser, mobile: check.mobile, viewport: check.viewport,
      samples: check.samples.map(({ name, output, rejectionCode, acceptedBy, elapsedMs }) =>
        ({ name, output, decisionPath: rejectionCode ?? acceptedBy, elapsedMs })),
      clearResetsTrace: check.clearResetsTrace, standardRouteClean: check.standardRouteClean });
  }
  const library = await readJson(libraryReport);
  assert(library.passed && library.catalogCount === audit.entries && library.videos.length === audit.entries);
  report.library = { file: 'library.json', catalogCount: library.catalogCount, verifiedVideoHashes: library.videos.length,
    chromePlaybackChecks: library.playbackCount, mobile: library.mobile, filters: library.filtering,
    cameraCalls: library.cameraCalls, externalRequests: library.remoteRequests.length };
  report.passed = true;
} catch (error) {
  failure = error;
  report.failure = String(error.stack || error);
} finally {
  if (preview?.pid && preview.exitCode === null) {
    const stopped = once(preview, 'exit');
    preview.kill();
    await Promise.race([stopped, delay(5000, undefined, { ref: false })]);
  }
  const path = resolve(output, 'browser-qa.json');
  await writeFile(path, JSON.stringify(report, null, 2) + '\n');
  console.log(path);
}
if (failure) throw failure;
