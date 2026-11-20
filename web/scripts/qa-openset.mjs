/** Real frozen artifacts only. No threshold fitting, fake media, or webcam claims. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(web, '..');
const output = path.join(web, 'qa-artifacts/open-set-v2');
const args = process.argv.slice(2);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const base = option('--url', 'http://127.0.0.1:5174');
const phase = option('--phase', 'all');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const read = async p => JSON.parse(await readFile(p, 'utf8'));
const local = p => new URL(`/@fs/${p.replaceAll('\\', '/')}`, base).href;
const reasons = value => Object.entries(value).filter(([, reject]) => reject).map(([name]) => name).sort();
const report = { createdAt: new Date().toISOString(), phase, synthetic: false, humanWebcamTested: false, thresholdFitting: false, parity: [], videos: [], checks: [], errors: [] };
const stress = { createdAt: report.createdAt, derivedDiagnosticOnly: true, humanWebcamTested: false, thresholdFitting: false,
  note: 'Synthetic transformations of real source JPEGs. Horizontal flip has no assumed linguistic label preservation. These are score/decision sensitivity diagnostics, not transformed-sign accuracy.', rows: [] };
let browser, page, failure;
const requests = new Set();
function compare(actual, expected, atol, rtol, label) {
  assert.equal(actual.length, expected.length, `${label} length`);
  let max = 0;
  actual.forEach((value, i) => {
    assert.ok(Number.isFinite(value) && Number.isFinite(expected[i]), `${label} finite`);
    const delta = Math.abs(value - expected[i]); max = Math.max(max, delta);
    assert.ok(delta <= atol + rtol * Math.abs(expected[i]), `${label}[${i}]: ${value} vs ${expected[i]}, delta ${delta}`);
  });
  return max;
}
const diagVector = d => [d.motionBodyRms, d.motionLocalRms, d.validTrackedPairFraction, ...d.meanPresence];
async function save() {
  await mkdir(output, { recursive: true });
  await writeFile(path.join(output, 'browser-comparison.json'), JSON.stringify(report, null, 2));
  if (stress.rows.length || phase === 'stress') await writeFile(path.join(output, 'stress-checks.json'), JSON.stringify(stress, null, 2));
}
try {
  const selectionPath = path.join(root, 'artifacts/open-set-v2/selection.locked.json');
  const selection = await read(selectionPath);
  assert.equal(selection.selection_uses_final_test, false);
  report.selectionSha256 = sha(await readFile(selectionPath));
  report.selectedCandidate = selection.selected_candidate;
  const methods = [];
  for (const [name, locked] of Object.entries(selection.candidates)) {
    const dir = path.join(root, 'artifacts/open-set-v2', name);
    const configPath = path.join(dir, 'config.json'), modelPath = path.join(dir, 'model.onnx');
    assert.equal(sha(await readFile(configPath)), locked.export.config_sha256, `${name} frozen config`);
    assert.equal(sha(await readFile(modelPath)), locked.export.model_sha256, `${name} frozen model`);
    methods.push({ name, config: await read(configPath), fixture: await read(path.join(dir, 'gate-parity-fixtures.json')),
      source: { configUrl: local(configPath), modelUrl: local(modelPath) }, hashes: locked.export });
  }
  report.artifacts = methods.map(({ name, hashes }) => ({ name, ...hashes }));
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.runtime = { browser: browser.version(), platform: process.platform, architecture: process.arch, node: process.version, origin: base, backend: 'WASM single-thread CPU', freshLandmarkWorkerPerClip: true };
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
  context.on('request', req => requests.add(req.url()));
  page = await context.newPage();
  page.setDefaultTimeout(150000);
  page.on('pageerror', error => report.errors.push(error.message));
  await page.goto(`${base}/extract.html`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => typeof window.signbridge?.setModelSource === 'function');
  await page.evaluate(async () => { window.openSetMath = await import('/src/openset.ts'); });
  if (phase === 'all' || phase === 'parity') {
    for (const method of methods) {
      await page.evaluate(source => window.signbridge.setModelSource(source), method.source);
      const status = await page.evaluate(() => window.signbridge.getModelStatus());
      assert.equal(status.ready, true, status.message);
      assert.equal(status.modelVersion, method.config.modelVersion);
      for (const fixture of method.fixture.cases) {
        const result = await page.evaluate(async ({ fixture, config }) => {
          const diagnostics = window.openSetMath.measureQualityMotion(fixture.x);
          const direct = window.openSetMath.scoreOpenSet(fixture.logits, fixture.quality, diagnostics.motionBodyRms, config, fixture.embedding);
          const actual = await window.signbridge.classifyFeatures(fixture.x, fixture.quality);
          return { direct, diagnostics, actual };
        }, { fixture, config: method.config });
        const atol = method.fixture.numericAtol, rtol = method.fixture.numericRtol;
        const numeric = {
          diagnostics: compare(diagVector(result.diagnostics), diagVector(fixture.diagnostics), atol, rtol, 'Direct diagnostics'),
          workerDiagnostics: compare(diagVector(result.actual.diagnostics), diagVector(fixture.diagnostics), atol, rtol, 'Worker diagnostics'),
          logits: compare(result.actual.logits, fixture.logits, atol, rtol, 'Real ONNX logits'),
          embedding: compare(result.actual.embedding, fixture.embedding, atol, rtol, 'Real ONNX embedding'),
          directConfidence: compare([result.direct.confidence], [fixture.confidence], atol, rtol, 'Direct confidence'),
          actualConfidence: compare([result.actual.confidence], [fixture.confidence], atol, rtol, 'Actual confidence'),
        };
        for (const scored of [result.direct, result.actual]) {
          assert.equal(scored.accepted, fixture.accepted, `${method.name}/${fixture.id}: acceptance`);
          assert.equal(scored.predicted, fixture.predicted, `${method.name}/${fixture.id}: class`);
          assert.deepEqual(reasons(scored.reasons), [...fixture.reasons].sort(), `${method.name}/${fixture.id}: rejection reasons`);
        }
        report.parity.push({ candidate: method.name, id: fixture.id, numericMaxAbsoluteErrors: numeric, exactDecisionAgreement: true, accepted: result.actual.accepted, reasons: reasons(result.actual.reasons) });
      }
      console.log(`Parity ${method.name}: ${method.fixture.cases.length} real fixtures passed`);
    }
    report.invalidInputs = await page.evaluate(config => {
      const checks = [[ [NaN, 0, 0, 0], 1, 0 ], [[0, 0, 0, 0], NaN, 0], [[0, 0, 0, 0], 1.1, 0], [[0, 0, 0, 0], 1, -1]];
      return checks.map(([logits, quality, motion]) => { try { window.openSetMath.scoreOpenSet(logits, quality, motion, config); return false; } catch { return true; } });
    }, methods[0].config);
    assert.ok(report.invalidInputs.every(Boolean));
    await save();
  }
  if (phase !== 'parity') {
    // This prerequisite prevents browser diagnostics from opening sealed final data early.
    const finalPath = path.join(root, 'artifacts/open-set-v2/final-evaluation.json');
    await read(finalPath); report.finalEvaluationSha256 = sha(await readFile(finalPath));
    const mappingPath = path.join(root, 'data/open-set-v2/clips/mapping.json');
    const mapping = await read(mappingPath);
    report.mappingSha256 = sha(await readFile(mappingPath));
    const clips = Object.entries(mapping.clips).map(([id, clip]) => ({ id, ...clip })).sort((a, b) => a.source_path < b.source_path ? -1 : a.source_path > b.source_path ? 1 : 0);
    assert.equal(clips.length, 12);
    for (const clip of clips) assert.equal(sha(await readFile(path.resolve(root, clip.path))), clip.mp4_sha256);
    if (phase === 'all' || phase === 'videos') {
      for (const method of methods) {
        await page.evaluate(source => window.signbridge.setModelSource(source), method.source);
        for (const clip of clips) {
          const prediction = await page.evaluate(async src => {
            const video = document.createElement('video'); video.muted = true; video.playsInline = true; document.body.append(video);
            try {
              await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = () => reject(new Error('MP4 decode failed')); video.src = src; video.load(); });
              return await window.signbridge.analyzeVideo(video);
            } finally { video.pause(); video.removeAttribute('src'); video.load(); video.remove(); }
          }, local(path.resolve(root, clip.path)));
          assert.equal(prediction.modelVersion, method.config.modelVersion);
          assert.equal(prediction.frames, Math.min(48, clip.frame_count));
          assert.ok(prediction.elapsedMs > 0 && Number.isFinite(prediction.confidence));
          report.videos.push({ candidate: method.name, id: clip.id, rawLabel: clip.raw_label, trueClass: clip.true_class, isKnown: clip.is_known, signerId: clip.signer_id, mp4Sha256: clip.mp4_sha256,
            ...prediction, acceptedCorrect: clip.is_known ? prediction.accepted && prediction.label === clip.true_class : null, unknownFalseAccept: clip.is_known ? null : prediction.accepted });
          console.log(`MP4 ${method.name} ${clip.id}: ${prediction.accepted ? prediction.label : 'REJECT'} (${prediction.elapsedMs.toFixed(0)} ms)`);
          await save();
        }
      }
    }
    if (phase === 'all' || phase === 'stress') {
      const manifest = await read(path.join(root, 'data/open-set-v2/manifest.new.final.json'));
      const chosen = [...new Map(clips.map(c => c.raw_label).map(label => [label, clips.find(c => c.raw_label === label)])).values()];
      assert.equal(chosen.length, 6);
      stress.selection = 'First lexical source path per each of six final labels, fixed before any diagnostic outcomes';
      stress.runtime = report.runtime; stress.selectionSha256 = report.selectionSha256;
      for (const clip of chosen) {
        const sample = manifest.samples.find(s => s.id === clip.id); assert.ok(sample);
        for (let i = 0; i < sample.frame_paths.length; i++) assert.equal(sha(await readFile(path.resolve(root, sample.frame_paths[i]))), sample.frame_sha256[i]);
        for (const transformation of ['original', 'brightness_0.6', 'horizontal_flip', 'translate_right_0.1_width_black']) {
          const extracted = await page.evaluate(async ({ paths, transformation }) => {
            const urls = [];
            try {
              for (const src of paths) {
                const bitmap = await createImageBitmap(await (await fetch(src)).blob());
                const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext('2d');
                ctx.fillStyle = 'black'; ctx.fillRect(0, 0, canvas.width, canvas.height);
                if (transformation === 'brightness_0.6') ctx.filter = 'brightness(0.6)';
                if (transformation === 'horizontal_flip') { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
                if (transformation === 'translate_right_0.1_width_black') ctx.translate(Math.round(canvas.width * 0.1), 0);
                ctx.drawImage(bitmap, 0, 0); bitmap.close();
                urls.push(URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' })));
              }
              return await window.signbridge.extractImagePaths(urls, 30);
            } finally { urls.forEach(url => URL.revokeObjectURL(url)); }
          }, { paths: sample.frame_paths.map(p => local(path.resolve(root, p))), transformation });
          const predictions = [];
          for (const method of methods) {
            await page.evaluate(source => window.signbridge.setModelSource(source), method.source);
            const prediction = await page.evaluate(({ x, quality, frames, span }) => window.signbridge.classifyFeatures(x, quality, undefined, { frameCount: frames, observedSpanMs: span }),
              { x: extracted.x, quality: extracted.quality, frames: extracted.frames, span: extracted.observedSpanMs });
            predictions.push({ candidate: method.name, ...prediction });
          }
          stress.rows.push({ id: clip.id, sourceLabel: clip.raw_label, transformation, derived: transformation !== 'original', labelPreservationAssumed: false,
            sourceSequenceSha256: clip.source_frame_sequence_sha256, frames: extracted.frames, observedSpanMs: extracted.observedSpanMs, extractionMs: extracted.elapsedMs, quality: extracted.quality, predictions });
          console.log(`Stress ${clip.id} ${transformation}: ${predictions.map(p => `${p.candidate[0]}=${p.label ?? 'REJECT'}`).join(', ')}`);
          await save();
        }
      }
    }
  }
  report.summary = Object.fromEntries(methods.map(method => {
    const rows = report.videos.filter(v => v.candidate === method.name), known = rows.filter(v => v.isKnown), unknown = rows.filter(v => !v.isKnown);
    const times = rows.map(v => v.elapsedMs).sort((a,b) => a-b);
    return [method.name, { total: rows.length, known: known.length, acceptedKnownCorrect: known.filter(v => v.acceptedCorrect).length, rejectedKnown: known.filter(v => !v.accepted).length, unknown: unknown.length, falselyAcceptedUnknown: unknown.filter(v => v.unknownFalseAccept).length, medianElapsedMs: times.length ? (times[Math.floor((times.length-1)/2)] + times[Math.ceil((times.length-1)/2)]) / 2 : null }];
  }));
  assert.equal(report.errors.length, 0, 'No uncaught browser errors');
  report.passed = true; stress.passed = true;
} catch (error) {
  failure = error; report.passed = false; report.failure = error.stack ?? String(error); stress.passed = false;
} finally {
  report.appHttpOrigins = [...new Set([...requests].filter(url => /^https?:/.test(url)).map(url => new URL(url).origin))];
  await save(); await browser?.close();
}
console.log(JSON.stringify({ passed: report.passed, parity: report.parity.length, videos: report.videos.length, stress: stress.rows.length, output, failure: failure?.message }));
if (failure) process.exitCode = 1;
