/** User-invoked local launcher. No training, downloads, uploads or camera access. */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, open, access, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expectedPreview, inspectPreview } from './preview-identity.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const web = resolve(root, 'web');
const judgeDemo = process.argv.includes('--judge-demo');
const assistant = process.argv.includes('--local-assistant');
if (assistant && judgeDemo) throw new Error('Choose one local edition.');
const videoLibrary = judgeDemo || assistant || process.argv.includes('--video-library');
const port = assistant ? '5206' : judgeDemo ? '5199' : videoLibrary ? '5196' : '5174';
const url = `http://127.0.0.1:${port}/`;
const edition = assistant ? 'publication-local-assistant' : judgeDemo ? 'publication-judge-demo' : 'publication-video-library';
const dist = videoLibrary ? resolve(root, edition, 'dist') : resolve(web, 'dist');
const logName = assistant ? 'local-assistant-preview' : judgeDemo ? 'judge-demo-preview' : videoLibrary ? 'video-library-preview' : 'preview';
const runtime = resolve(root, '.runtime-tmp');
await access(resolve(dist, 'index.html'));
await mkdir(runtime, { recursive: true });
const expected = await expectedPreview(dist);
if (videoLibrary) {
  const catalog = await readFile(resolve(dist, 'library/catalog.json'));
  const audit = JSON.parse(await readFile(resolve(dist, '../library-audit.json'), 'utf8'));
  const hash = createHash('sha256').update(catalog).digest('hex');
  if (hash !== audit.catalog_sha256) throw new Error('The video catalog does not match its build audit.');
  expected['/library/catalog.json'] = hash;
  if (assistant) {
    if (audit.status !== 'local_assistant') throw new Error('The local assistant build is missing.');
    await access(resolve(root, '.venv/Scripts/python.exe'));
    expected['/__signbridge/local-assistant/status'] = createHash('sha256')
      .update(JSON.stringify({ service: 'signbridge-local-assistant', model: 'qwen3:4b' })).digest('hex');
  }
  if (judgeDemo) {
    if (audit.status !== 'local_judge_demo') throw new Error('The local judge build is missing.');
    for (const file of ['sources.json', 'frozen.mp4', 'no-tracking.mp4', 'unsupported.mp4']) {
      const asset = `__signbridge/judge/${file}`;
      const digest = createHash('sha256').update(await readFile(resolve(dist, asset))).digest('hex');
      if (digest !== audit.runtime_files[asset]) throw new Error(`The demo fixture ${file} does not match its build audit.`);
      expected[`/${asset}`] = digest;
    }
  }
}
const initial = await inspectPreview(url, expected);
if (initial.state === 'incompatible') throw new Error(`Port ${port} is occupied by an incompatible or stale server. ${initial.reason} Close that server yourself or use the correct current preview.`);
if (initial.state === 'absent') {
  const stdout = await open(resolve(runtime, `${logName}.stdout.log`), 'a');
  const stderr = await open(resolve(runtime, `${logName}.stderr.log`), 'a');
  const options = videoLibrary ? ['--config', 'vite.v8-public.config.ts', '--mode', assistant ? 'signbridge-local-assistant' : 'signbridge-openhands',
    '--outDir', `../${edition}/dist`] : [];
  const child = spawn(process.execPath, [resolve(web, 'node_modules/vite/bin/vite.js'), 'preview', '--host', '127.0.0.1', '--port', port, '--strictPort', ...options], {
    cwd: web, detached: true, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd],
  });
  let launchError;
  child.on('error', error => { launchError = error; });
  child.unref();
  await stdout.close(); await stderr.close();
  let found = false;
  for (let i = 0; i < 30; i++) {
    if (launchError) throw launchError;
    const current = await inspectPreview(url, expected);
    if (current.state === 'matching') { found = true; break; }
    if (current.state === 'incompatible') {
      child.kill(); // Only the child created by this invocation; never an existing server.
      throw new Error(`Preview identity check failed. ${current.reason} No existing process was stopped.`);
    }
    if (child.exitCode !== null) throw new Error(`Preview failed; inspect .runtime-tmp/${logName}.stderr.log. Port ${port} may be in use.`);
    await new Promise(r => setTimeout(r, 300));
  }
  if (!found) { child.kill(); throw new Error(`Preview did not start; inspect .runtime-tmp/${logName}.stderr.log.`); }
  await writeFile(resolve(runtime, `${logName}.pid`), String(child.pid));
}
const appUrl = assistant ? `${url}?assistant=1#/try` : judgeDemo ? `${url}?judge=1#/try` : videoLibrary ? `${url}#/library` : url;
console.log(`SignBridge AI: ${appUrl}`);
if (!videoLibrary) console.log('Research prototype: consult the measured open-set report. Human webcam accuracy is untested.');
if (process.platform === 'win32' && !process.argv.includes('--no-open')) {
  const opener = spawn('powershell.exe', ['-NoProfile', '-Command', `Start-Process '${appUrl}'`], { detached: true, windowsHide: true, stdio: 'ignore' });
  opener.unref();
}
