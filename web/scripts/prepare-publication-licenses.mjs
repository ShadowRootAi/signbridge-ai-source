/** Fetch legal notices only from authoritative upstreams; record exact bytes. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const destination = path.join(web, 'deployment/licenses');
await mkdir(destination, { recursive: true });
const records = [];
const hash = value => createHash('sha256').update(value).digest('hex');
for (const pkg of ['react', 'react-dom', 'scheduler']) {
  const meta = JSON.parse(await readFile(path.join(web, 'node_modules', pkg, 'package.json'), 'utf8'));
  const bytes = await readFile(path.join(web, 'node_modules', pkg, 'LICENSE'));
  const file = `${pkg}-LICENSE.txt`; await writeFile(path.join(destination, file), bytes);
  records.push({ file, package: pkg, version: meta.version, source: `Installed exact npm package ${pkg}@${meta.version}/LICENSE`, sha256: hash(bytes), exactPackageText: true });
}
const sources = [
  ['mediapipe-APACHE-2.0.txt', ['https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/LICENSE'], '@mediapipe/tasks-vision', '1.2.2'],
  ['onnxruntime-MIT.txt', ['https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/LICENSE', 'https://raw.githubusercontent.com/microsoft/onnxruntime/main/LICENSE'], 'onnxruntime-web / onnxruntime-common', '1.31.1'],
  ['onnxruntime-ThirdPartyNotices.txt', ['https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/ThirdPartyNotices.txt', 'https://raw.githubusercontent.com/microsoft/onnxruntime/main/ThirdPartyNotices.txt'], 'onnxruntime-web / onnxruntime-common', '1.31.1'],
  ['GSL-CC-BY-4.0.txt', ['https://creativecommons.org/licenses/by/4.0/legalcode.txt'], 'GSL dataset', 'Zenodo 4756317'],
];
for (const [file, urls, pkg, version] of sources) {
  let record;
  for (const url of urls) {
    const response = await fetch(url, { signal: AbortSignal.timeout(20000) });
    if (!response.ok) continue;
    const bytes = Buffer.from(await response.arrayBuffer());
    if (bytes.length < 100 || /<!doctype html/i.test(bytes.toString('utf8'))) continue;
    await writeFile(path.join(destination, file), bytes);
    record = { file, package: pkg, version, source: url, sha256: hash(bytes), retrievedAt: new Date().toISOString(), exactPackageText: url.includes('/v1.31.1/'), upstreamMainFallback: url.includes('/main/') };
    records.push(record); break;
  }
  if (!record) throw new Error(`No authoritative license text fetched for ${file}`);
}
const landmark = await readFile(path.join(web, 'public/models/holistic_landmarker.task'));
const landmarkNotice = `MediaPipe Holistic pretrained model bundle\n\nSource: https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task\nSHA-256: ${hash(landmark)}\nUnmodified downloaded artifact; no inference that the mutable latest URL is immutable.\n\nMediaPipe maintainer schmidt-sebastian stated on 2026-10-05 that MediaPipe source\nand models are Apache2 unless otherwise noted:\nhttps://github.com/google-ai-edge/mediapipe/issues/6355\nThe SDK package @mediapipe/tasks-vision@1.2.2 also declares Apache-2.0.\nThe Apache License text is included in mediapipe-APACHE-2.0.txt.\n\nThis records the upstream general license statement and our exact asset hash.\nAn artifact-specific SBOM/NOTICE for the installed compiled WASM was not present\nin its npm package; this bundle does not claim an independently reconciled\nthird-party build inventory.\n`;
await writeFile(path.join(destination, 'mediapipe-model-NOTICE.txt'), landmarkNotice);
records.push({ file: 'mediapipe-model-NOTICE.txt', source: 'https://github.com/google-ai-edge/mediapipe/issues/6355', modelSha256: hash(landmark), sha256: hash(landmarkNotice), generalUpstreamModelLicense: 'Apache-2.0 unless otherwise noted' });
await writeFile(path.join(destination, 'provenance.json'), JSON.stringify({ records, note: 'Preserved upstream texts and general upstream model-license statement, not a claim that every compiled transitive dependency notice has been independently reconciled.' }, null, 2));
console.log(JSON.stringify(records.map(({ file, exactPackageText, upstreamMainFallback }) => ({ file, exactPackageText, upstreamMainFallback }))));
