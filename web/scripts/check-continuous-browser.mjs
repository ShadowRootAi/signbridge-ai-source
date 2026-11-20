import { chromium } from 'playwright';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, relative, isAbsolute, sep, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const prefix = process.argv.includes('--expanded') ? 'continuous-v1-expanded' : 'continuous-v1';
const manifestDirs = prefix === 'continuous-v1-expanded' ? ['continuous-v1', 'continuous-v1-extra'] : ['continuous-v1'];
const base = 'http://127.0.0.1:5175';
const json = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const toVite = (path) => {
  const absolute = resolve(root, path);
  const fromRoot = relative(root, absolute);
  assert(fromRoot && fromRoot !== '..' && !fromRoot.startsWith(`..${sep}`) && !isAbsolute(fromRoot), 'File must be inside the project');
  return `/@fs/${absolute.replaceAll('\\', '/').split('/').map((part, i) => i === 0 && /^[A-Z]:$/i.test(part) ? part : encodeURIComponent(part)).join('/')}`;
};
const rows = new Map();
const features = new Map();
const expanded = prefix === 'continuous-v1-expanded';
for (const dir of expanded ? [...manifestDirs, 'continuous-v1-holdout'] : manifestDirs) {
  for (const row of (await json(`data/${dir}/manifest.json`)).samples) rows.set(row.id, row);
  for (const line of (await readFile(resolve(root, `data/${dir}/features.jsonl`), 'utf8')).split('\n').filter(Boolean)) {
    const row = JSON.parse(line);
    features.set(row.id, row);
  }
}
const { test_predictions: test } = await json(`reports/${prefix}/ctc.json`);
const sorted = [...test].sort((a, b) => a.frames - b.frames);
const fresh = new Set([sorted[0].id, sorted.at(-1).id]);
const holdout = expanded ? (await json(`reports/${prefix}/ctc-holdout.json`)).predictions : [];
// The first held-out signer and repetition, selected by ID, not by correctness.
const dialogue = expanded ? [...test, ...holdout].filter(row =>
  /^police1_signer6_rep1_sentences__sentences000[0-5]$/.test(row.id),
).sort((a, b) => a.id.localeCompare(b.id)) : [];
if (expanded) assert.equal(dialogue.length, 6, 'Need all six consecutive dialogue turns');
for (const row of dialogue) fresh.add(row.id);
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const output = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', (error) => console.error(error));
  await page.goto(`${base}/extract.html`);
  const model = `artifacts/${prefix}/browser`;
  await page.evaluate(async ({ config, model }) => {
    const { loadContinuousRecognizer } = await import('/src/continuous-inference.ts');
    window.continuousPredict = await loadContinuousRecognizer(config, model);
  }, { config: toVite(`${model}/sequence-config.json`), model: toVite(`${model}/sequence.onnx`) });
  for (const reference of [...test, ...holdout]) {
    const feature = features.get(reference.id);
    const source = rows.get(reference.id);
    assert(source && feature);
    const inputs = {
      features: feature.features, fresh: fresh.has(reference.id),
      paths: source.frame_paths.map(toVite), sourceFps: source.fps, stride: feature.stride,
    };
    const actual = await page.evaluate(async (input) => {
      if (!input.fresh) return window.continuousPredict(input.features);
      const { extractContinuousFrames } = await import('/src/continuous-extract.ts');
      const extraction = await extractContinuousFrames(input.paths, input.sourceFps, input.stride);
      return {
        ...await window.continuousPredict(extraction.features), extraction_ms: extraction.elapsed_ms,
        maximum_feature_error: Math.max(...extraction.features.map((row, i) =>
          Math.max(...row.map((value, j) => Math.abs(value - input.features[i][j]))))),
      };
    }, inputs);
    assert.deepEqual(actual.glosses, reference.predicted_glosses, reference.id);
    assert.equal(actual.text, reference.predicted_text, reference.id);
    assert(Math.abs(actual.confidence - reference.confidence) <= 0.0001, `Confidence differs for ${reference.id}`);
    if (inputs.fresh) assert.equal(actual.maximum_feature_error, 0, `RGB re-extraction differs for ${reference.id}`);
    output.push({ id: reference.id, from_original_rgb: inputs.fresh, frames: feature.features.length, ...actual });
  }
} finally { await browser.close(); }
const median = (values) => {
  values.sort((a, b) => a - b);
  return (values[Math.floor((values.length - 1) / 2)] + values[Math.ceil((values.length - 1) / 2)]) / 2;
};
const result = {
  engine: 'Chrome / ONNX Runtime Web WASM', matching_test_predictions: output.length,
  reextracted_and_predicted_from_rgb: output.filter(row => row.from_original_rgb).length,
  median_model_ms: median(output.map(row => row.inference_ms)),
  scripted_dialogue: dialogue.map((reference, turn) => ({
    turn: turn + 1, id: reference.id, dataset_reference: reference.reference_text,
    browser_output: output.find(row => row.id === reference.id).text,
  })),
  output,
};
await writeFile(resolve(root, `reports/${prefix}/browser-parity.json`), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ ...result, output: undefined }));
