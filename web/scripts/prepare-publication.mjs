/** Isolated, local review artifact. Does not register, synchronize or publish a Site. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFile, readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const project = path.resolve(web, '..');
const output = path.join(project, 'publication-review');
const dist = path.join(output, 'dist');
const hash = b => createHash('sha256').update(b).digest('hex');
async function files(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    assert.ok(!entry.isSymbolicLink(), 'Symlinks are excluded from publication.');
    const name = prefix + entry.name;
    if (entry.isDirectory()) result.push(...await files(path.join(directory, entry.name), `${name}/`));
    else result.push(name);
  }
  return result.sort();
}
const config = JSON.parse(await readFile(path.join(web, 'dist/models/model_config.json'), 'utf8'));
const model = await readFile(path.join(web, 'dist/models/model.onnx'));
const selected = JSON.parse(await readFile(path.join(project, 'artifacts/open-set-v2/selection.locked.json'), 'utf8'));
const frozen = selected.candidates[selected.selected_candidate].export;
assert.equal(config.candidate, selected.selected_candidate, 'Current deployment must remain validation-selected B until root decision.');
assert.equal(hash(model), frozen.model_sha256);
assert.equal(hash(await readFile(path.join(web, 'dist/models/model_config.json'))), frozen.config_sha256);
const sourceFiles = await files(path.join(web, 'dist'));
const allowed = /^(index\.html|NOTICE\.txt|assets\/[A-Za-z0-9_.-]+\.(?:js|css|wasm)|models\/(?:model\.onnx|model_config\.json|holistic_landmarker\.task)|reports\/technical-report\.json|vendor\/(?:mediapipe|onnxruntime)\/[A-Za-z0-9_.-]+)$/;
assert.ok(sourceFiles.every(file => allowed.test(file)), `Unexpected dist file; review before publication: ${sourceFiles.filter(file => !allowed.test(file))}`);
await mkdir(dist, { recursive: true });
for (const name of sourceFiles) {
  const target = path.join(dist, name); await mkdir(path.dirname(target), { recursive: true });
  await copyFile(path.join(web, 'dist', name), target);
}
const sourceReportBytes = await readFile(path.join(web, 'dist/reports/technical-report.json'));
let redactedStringCount = 0;
function sanitize(value) {
  if (typeof value === 'string') {
    let next = value.replaceAll(project, '[project]').replaceAll(project.replaceAll('\\', '/'), '[project]');
    if (/[A-Za-z]:[\\/]/.test(next)) next = next.replace(/[A-Za-z]:[\\/][^\n]*/g, '[local-path-redacted]');
    if (next !== value) redactedStringCount++;
    return next;
  }
  if (Array.isArray(value)) return value.map(sanitize);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, sanitize(item)]));
  return value;
}
const report = sanitize(JSON.parse(sourceReportBytes));
report.publicationPreparation = { publicationStatus: 'not_published', sourceReportSha256: hash(sourceReportBytes), change: 'Local machine absolute paths replaced with project-relative provenance markers. Scientific measurements unchanged.' };
assert.equal(report.modelSha256, hash(model));
await writeFile(path.join(dist, 'reports/technical-report.json'), JSON.stringify(report, null, 2));
const licenseRoot = path.join(web, 'deployment/licenses');
const licenseFiles = await files(licenseRoot);
assert.ok(licenseFiles.includes('provenance.json'), 'Fetch/copy required license resources first.');
for (const name of licenseFiles) { await mkdir(path.dirname(path.join(dist, 'licenses', name)), { recursive: true }); await copyFile(path.join(licenseRoot, name), path.join(dist, 'licenses', name)); }
const notice = await readFile(path.join(dist, 'NOTICE.txt'), 'utf8');
await writeFile(path.join(dist, 'NOTICE.txt'), `${notice}\nPUBLICATION REVIEW MATERIAL\nFull runtime license texts and upstream provenance are included under /licenses/.\nThis is an unpublished review artifact. No raw GSL images, clips, human webcam\nrecordings, development capture endpoint or extraction harness are included.\n`);
const headers = JSON.parse(await readFile(path.join(web, 'deployment/headers.json'), 'utf8'));
await writeFile(path.join(dist, '_headers'), `/*\n${Object.entries(headers).map(([key, value]) => `  ${key}: ${value}`).join('\n')}\n`);
await writeFile(path.join(output, 'hosting-manifest.template.json'), JSON.stringify({ project_id: 'REQUIRES_NATIVE_SITES_REGISTRATION', static: { directory: 'dist' } }, null, 2));
const inventory = [];
const expectedFiles = new Set([...sourceFiles, ...licenseFiles.map(file => `licenses/${file}`), '_headers']);
for (const name of await files(dist)) {
  assert.ok(expectedFiles.has(name), `Unexpected stale file in review directory: ${name}; inspect it before regenerating.`);
  const bytes = await readFile(path.join(dist, name));
  assert.ok(!/(^|\/)(?:data|src|node_modules|\.git|\.env)(\/|$)|webcam-check|extract\.html|\.map$|\.(?:jpg|jpeg|mp4|webm)$/i.test(name), `Private/development material excluded: ${name}`);
  assert.ok(bytes.length <= 25 * 1024 * 1024, `${name} exceeds reference 25 MiB static asset limit.`);
  if (/\.(?:json|txt|html|js|css)$/.test(name)) assert.ok(!/[A-Za-z]:[\\/]Users[\\/]/.test(bytes.toString('utf8')), `Local user path leaked in ${name}`);
  inventory.push({ path: name, bytes: bytes.length, sha256: hash(bytes) });
}
const audit = { status: 'prepared_not_published', generatedAt: new Date().toISOString(), modelVersion: config.modelVersion, modelSha256: hash(model), configSha256: frozen.config_sha256, sourceReportSha256: hash(sourceReportBytes), redactedStringCount,
  totalBytes: inventory.reduce((sum, file) => sum + file.bytes, 0), largestFileBytes: Math.max(...inventory.map(file => file.bytes)), files: inventory,
  supportedLabels: config.labels, measuredUnknownFalseAcceptances: report.metrics.test.unknown_false_acceptance_n, measuredUnknownN: report.metrics.test.unknown_n,
  humanWebcamTested: false, speechEnabled: false, rawMediaIncluded: false, developmentCaptureEndpointIncluded: false, sitesRegistrationPerformed: false,
  outstanding: ['No public host/header verification yet.', 'No human webcam or non-Chromium/mobile validation.', 'Unknown rejection remains unreliable at 24/50 false acceptances.'],
  provenanceLimitations: ['MediaPipe general source/model Apache-2.0 statement is preserved with the exact model hash. Installed compiled WASM package has no artifact-specific SBOM/NOTICE; transitive build inventory was not independently reconciled.'] };
await writeFile(path.join(output, 'publication-audit.json'), JSON.stringify(audit, null, 2));
await copyFile(path.join(web, 'deployment/README.md'), path.join(output, 'README.md'));
console.log(JSON.stringify({ status: audit.status, directory: output, files: inventory.length, totalBytes: audit.totalBytes, redactedStringCount, modelSha256: audit.modelSha256 }));
