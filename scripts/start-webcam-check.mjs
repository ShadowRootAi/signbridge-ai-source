/** Explicit user launcher: hidden local dev helper, never automatic camera use. */
import { spawn } from 'node:child_process';
import { access, mkdir, open, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const web = resolve(root, 'web');
const runtime = resolve(root, '.runtime-tmp');
const origin = 'http://127.0.0.1:5175';
const pilot = process.argv.includes('--v8-pilot');
const openHands = process.argv.includes('--openhands');
if (pilot && openHands) throw new Error('Choose one test mode.');
const page = openHands ? 'openhands-webcam.html' : pilot ? 'v8-pilot.html' : 'webcam-check.html';
const url = `${origin}/${page}`;
await access(resolve(web, 'webcam-check.html'));
await access(resolve(web, 'node_modules/vite/bin/vite.js'));
if (openHands) {
  await access(resolve(web, page));
  await access(resolve(root, 'research/cache/openhands-gsl/stgcn.onnx'));
  await access(resolve(root, 'research/cache/openhands-gsl/stgcn-onnx.json'));
  await access(resolve(root, 'reports/pretrained-gsl/stgcn-gate-config.json'));
} else if (pilot) {
  await access(resolve(web, page));
  await access(resolve(root, 'artifacts/v8/browser/model-config.json'));
  await access(resolve(root, 'research/cache/mediapipe/hand_landmarker-float16-1.task'));
}
await mkdir(runtime, { recursive: true });

async function ready() {
  try {
    const response = await fetch(`${origin}/__signbridge/webcam-capture/status`, { signal: AbortSignal.timeout(1500) });
    if (!response.ok) return false;
    const status = await response.json();
    return status.ready === true && status.localOnly === true && status.directory === 'data/open-set-v2/webcam';
  } catch { return false; }
}

if (!await ready()) {
  const stdout = await open(resolve(runtime, 'webcam-helper.stdout.log'), 'a');
  const stderr = await open(resolve(runtime, 'webcam-helper.stderr.log'), 'a');
  const child = spawn(process.execPath, [resolve(web, 'node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '5175', '--strictPort'], {
    cwd: web, detached: true, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd],
  });
  let launchError;
  child.on('error', (error) => { launchError = error; });
  child.unref();
  await stdout.close();
  await stderr.close();
  let found = false;
  for (let attempt = 0; attempt < 30; attempt++) {
    if (launchError) throw launchError;
    if (await ready()) { found = true; break; }
    if (child.exitCode !== null) throw new Error('Webcam helper failed. Inspect .runtime-tmp/webcam-helper.stderr.log; port 5175 may be occupied.');
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  if (!found) throw new Error('Webcam helper did not start. Inspect .runtime-tmp/webcam-helper.stderr.log.');
  await writeFile(resolve(runtime, 'webcam-helper.pid'), String(child.pid));
}

if (openHands) {
  const [lab, policy, model] = await Promise.all([
    fetch(url, { signal: AbortSignal.timeout(5000) }),
    fetch(`${origin}/__signbridge/openhands/policy.json`, { signal: AbortSignal.timeout(5000) }),
    fetch(`${origin}/__signbridge/openhands/stgcn.onnx`, { method: 'HEAD', signal: AbortSignal.timeout(5000) }),
  ]);
  if (!lab.ok || !/\/src\/openhands-webcam\.ts/.test(await lab.text()) || !policy.ok
      || (await policy.json()).supportedLabels?.length !== 120 || !model.ok) {
    throw new Error('OpenHands local assets are unavailable on port 5175. Inspect the Vite logs.');
  }
  console.log(`OpenHands camera evaluation: ${url}`);
  console.log(`SignBridge with OpenHands: ${origin}/?model=openhands#/sign`);
} else if (pilot) {
  const response = await fetch(url, { signal: AbortSignal.timeout(4000) });
  if (!response.ok || !/\bsrc="\/src\/v8-pilot\.ts(?:\?[^"]*)?"/.test(await response.text())) {
    throw new Error('The local V8 pilot page is unavailable.');
  }
  const policy = "connect-src 'self' blob: ws://127.0.0.1:* ws://localhost:*";
  if (response.headers.get('content-security-policy') !== policy) {
    throw new Error('The local V8 connection policy is missing. Restart the project server before using the pilot.');
  }
  const pilotSources = [page, 'src/v8-pilot.ts', 'src/v8-pilot.css', 'src/v8-pilot-core.ts', 'src/features.ts',
    'src/tracking.ts', 'src/openset.ts', 'src/recognizer.worker.ts', 'src/v6_hand_recovery.worker.ts', 'src/v6_hand_detector.worker.ts'];
  await Promise.all(pilotSources.map(async (name) => {
    const source = await readFile(resolve(web, name), 'utf8');
    const served = await fetch(`${origin}/${name}?raw&import`, { signal: AbortSignal.timeout(5000) });
    const rawModule = (await served.text()).trim().replace(/;$/, '');
    if (!served.ok || rawModule !== `export default ${JSON.stringify(source)}`) {
      throw new Error(`Port 5175 does not serve this copy of the V8 pilot (${name}). No existing server was stopped.`);
    }
  }));
  console.log(`Local SignBridge V8 pilot: ${url}`);
  console.log('The pilot opens the camera only after consent and an explicit action. Reports and media are downloaded locally.');
} else {
  console.log(`Optional local webcam check: ${url}`);
  console.log('No camera is opened automatically. Each recording needs consent, manual capture, label confirmation and explicit local save.');
  console.log('These observations must remain separate from model training and threshold fitting.');
}
if (process.platform === 'win32' && !process.argv.includes('--no-open')) {
  const opener = spawn('powershell.exe', ['-NoProfile', '-Command', `Start-Process '${url}'`], { detached: true, windowsHide: true, stdio: 'ignore' });
  opener.unref();
}
