import { chromium } from 'playwright';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, relative, isAbsolute, sep, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const base = 'http://127.0.0.1:5175';
const manifest = JSON.parse(await readFile(resolve(root, 'data/manifest.json'), 'utf8'));
const cohort = manifest.samples.filter(row =>
  /^kep4_signer3_rep[12]_sentences__sentences(0006|0014|0018)$/.test(row.id),
).sort((a, b) => a.id.localeCompare(b.id));
assert.equal(cohort.length, 6, 'Need the prespecified two repetitions and three glosses');
const pathUrl = (file) => {
  const absolute = resolve(root, file);
  const rel = relative(root, absolute);
  assert(rel && rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
  return `/@fs/${absolute.replaceAll('\\', '/').split('/').map((part, i) =>
    i === 0 && /^[A-Z]:$/i.test(part) ? part : encodeURIComponent(part)).join('/')}`;
};
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const rows = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', error => console.error(error.message));
  await page.goto(`${base}/extract.html`);
  await page.evaluate(async ({ model, config }) => {
    const { loadOpenHandsProbe } = await import('/src/openhands-stgcn-probe.ts');
    window.openhandsProbe = await loadOpenHandsProbe(model, config);
  }, { model: pathUrl('research/cache/openhands-gsl/stgcn.onnx'),
    config: pathUrl('research/cache/openhands-gsl/stgcn-onnx.json') });
  for (const item of cohort) {
    const result = await page.evaluate(({ paths, fps }) => window.openhandsProbe(paths, fps),
      { paths: item.frame_paths.map(pathUrl), fps: item.fps ?? 30 });
    rows.push({ id: item.id, source_gloss: item.raw_label, ...result });
  }
} finally { await browser.close(); }
const report = {
  scope: 'Chrome MediaPipe Tasks IMAGE (instead of OpenHands legacy Holistic) + official ST-GCN ONNX; six original, pre-cut single-gloss sentence clips, one GSL signer, no fresh person or webcam claim',
  clips: rows.length,
  correct_top1: rows.filter(row => row.top5[0]?.gloss === row.source_gloss).length,
  correct_top5: rows.filter(row => row.top5.some(candidate => candidate.gloss === row.source_gloss)).length,
  rows,
};
const output = resolve(root, 'reports/pretrained-gsl/browser-smoke.json');
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, clips: report.clips, correct_top1: report.correct_top1, correct_top5: report.correct_top5 }));
