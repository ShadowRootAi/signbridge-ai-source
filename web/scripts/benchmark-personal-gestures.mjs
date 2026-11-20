// Offline replay of the actual browser feature and teaching algorithms on cached MediaPipe frames.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(web, '..');
const manifests = ['data/manifest.json', 'data/open-set-v2/manifest.json', 'data/v3/manifest.json'];
const featureDirs = ['data/features', 'data/open-set-v2/features', 'data/v3/features'];
const thresholds = [
  { name: 'current', cap: .34, multiplier: 1.25 },
  { name: 'bounded', cap: .30, multiplier: 1.15 },
  { name: 'strict', cap: .26, multiplier: 1.10 },
];
const digest = buffer => createHash('sha256').update(buffer).digest('hex');
const source = name => resolve(root, name);

function combinations(items, count) {
  if (!count) return [[]];
  return items.flatMap((item, index) => combinations(items.slice(index + 1), count - 1).map(group => [item, ...group]));
}

function totals() {
  return { attempted: 0, fitted: 0, heldKnown: 0, heldKnownAccepted: 0, heldOffSign: 0,
    heldOffSignAccepted: 0, heldOtherSignerKnown: 0, heldOtherSignerKnownAccepted: 0 };
}

const server = await createServer({
  root: web, configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false },
  plugins: [{ name: 'expose-teaching-functions-for-local-benchmark', enforce: 'pre', transform(code, id) {
    if (id.replaceAll('\\', '/').endsWith('/src/personal-gestures.ts')) {
      return `${code}\nexport { compact, sequenceDistance };`;
    }
  } }],
});
try {
  const teaching = await server.ssrLoadModule('/src/personal-gestures.ts');
  const { preprocess } = await server.ssrLoadModule('/src/features.ts');
  const { measureRawTracking } = await server.ssrLoadModule('/src/tracking.ts');
  const { measureQualityMotion } = await server.ssrLoadModule('/src/openset.ts');
  const refs = new Map();
  for (const name of manifests) {
    const { samples } = JSON.parse(await readFile(source(name), 'utf8'));
    for (const row of samples) refs.set(row.id, row);
  }
  const files = (await Promise.all(featureDirs.map(async name =>
    (await readdir(source(name))).filter(file => file.endsWith('.jsonl')).map(file => `${name}/${file}`)))).flat().sort();
  const records = new Map();
  for (const name of files) {
    for (const line of (await readFile(source(name), 'utf8')).trim().split('\n')) {
      const row = JSON.parse(line);
      const ref = refs.get(row.id);
      if (row.raw?.length && ref && ref.raw_label === ref.annotation_raw_glosses && !records.has(row.id)) {
        records.set(row.id, { ref, frames: row.raw });
      }
    }
  }
  const clips = [];
  const skipped = {};
  const reject = reason => { skipped[reason] = (skipped[reason] ?? 0) + 1; };
  for (const { ref, frames } of [...records.values()].sort((a, b) => a.ref.id.localeCompare(b.ref.id))) {
    if (!(ref.duration_seconds >= .2 && ref.duration_seconds <= 3)) { reject('duration'); continue; }
    if (frames.length < 6 || measureRawTracking(frames).eitherHandDetection < .8) { reject('tracking'); continue; }
    const { x } = preprocess(frames);
    if (measureQualityMotion(x).motionBodyRms < .004) { reject('motion'); continue; }
    assert.match(ref.sha256, /^[a-f0-9]{64}$/);
    clips.push({ index: clips.length, id: ref.id, signer: String(ref.signer), label: ref.raw_label, scenario: ref.scenario,
      sample: { fingerprint: ref.sha256, sequence: teaching.compact(x) } });
  }
  assert.equal(new Set(clips.map(clip => clip.sample.fingerprint)).size, clips.length);
  const matrix = clips.map(() => new Float64Array(clips.length));
  for (const a of clips) for (const b of clips.slice(a.index + 1)) {
    matrix[a.index][b.index] = matrix[b.index][a.index] = teaching.sequenceDistance(a.sample, b.sample);
  }
  function distance(candidate, examples) {
    const [first, second] = examples.map(clip => matrix[candidate.index][clip.index]).sort((a, b) => a - b);
    return (first + second) / 2;
  }
  const groups = new Map();
  for (const clip of clips) {
    const key = `${clip.signer}/${clip.label}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(clip);
  }
  const aggregate = Object.fromEntries(thresholds.map(({ name }) => [name, totals()]));
  const byGroup = [];
  let parityFolds = 0;
  for (const [key, positives] of groups) {
    if (positives.length < 6) continue;
    const { signer, label } = positives[0];
    const offSign = clips.filter(clip => clip.signer === signer && clip.label !== label);
    const checksOffSign = [...new Map(offSign.map(clip => [clip.label, clip])).values()].slice(0, 2);
    if (checksOffSign.length !== 2) continue;
    const testOffSign = offSign.filter(clip => !checksOffSign.includes(clip));
    const testOtherSigner = clips.filter(clip => clip.signer !== signer && clip.label === label);
    const metrics = Object.fromEntries(thresholds.map(({ name }) => [name, totals()]));
    for (const training of combinations(positives, 3)) {
      const remaining = positives.filter(clip => !training.includes(clip));
      const leaveOneOut = Math.max(...training.map(clip => distance(clip, training.filter(other => other !== clip))));
      for (const checks of combinations(remaining, 2)) {
        const testKnown = remaining.filter(clip => !checks.includes(clip));
        for (const { name, cap, multiplier } of thresholds) {
          const threshold = Math.min(cap, multiplier * leaveOneOut);
          const correct = checks.filter(clip => distance(clip, training) <= threshold).length;
          const rejected = checksOffSign.filter(clip => distance(clip, training) > threshold).length;
          const fit = threshold > 0 && correct === 2 && rejected === 2;
          if (name === 'current' && metrics[name].attempted === 0) {
            const actual = teaching.fitPersonalSign(label, training.map(clip => clip.sample), checks.map(clip => clip.sample),
              checksOffSign.map(clip => clip.sample));
            assert.equal(actual.correct, correct);
            assert.equal(actual.rejected, rejected);
            assert.equal(Boolean(actual.sign), fit);
            if (actual.sign) assert.equal(actual.sign.threshold, threshold);
            parityFolds++;
          }
          for (const stats of [aggregate[name], metrics[name]]) {
            stats.attempted++;
            if (!fit) continue;
            stats.fitted++;
            stats.heldKnown += testKnown.length;
            stats.heldKnownAccepted += testKnown.filter(clip => distance(clip, training) <= threshold).length;
            stats.heldOffSign += testOffSign.length;
            stats.heldOffSignAccepted += testOffSign.filter(clip => distance(clip, training) <= threshold).length;
            stats.heldOtherSignerKnown += testOtherSigner.length;
            stats.heldOtherSignerKnownAccepted += testOtherSigner.filter(clip => distance(clip, training) <= threshold).length;
          }
        }
      }
    }
    byGroup.push({ key, signer, label, positiveIds: positives.map(clip => clip.id),
      negativeCheckIds: checksOffSign.map(clip => clip.id), negativeTestIds: testOffSign.map(clip => clip.id), metrics });
  }
  const output = source('reports/personal-gestures/cache-benchmark.json');
  const report = {
    scope: 'Existing annotated source-dataset recordings with cached MediaPipe landmarks. No new webcam input. '
      + 'Folds reuse clips across different combinations, so counts are not independent human trials.',
    selection: 'Within each same-signer raw-gloss group, exhaustively rotate 3 training, 2 positive validation, '
      + 'and at least 1 held recording. Two deterministic different-gloss recordings are negative validation. '
      + 'The remaining same-signer off-signs and all other-signer same-label clips are diagnostics only.',
    implementationSha256: digest(await readFile(resolve(web, 'src/personal-gestures.ts'))),
    inputRawClips: records.size, eligibleClips: clips.length, skipped, groupsScored: byGroup.length, parityFolds,
    thresholdVariants: thresholds, aggregate, byGroup,
  };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ output, inputRawClips: report.inputRawClips, eligible: clips.length, skipped,
    groups: byGroup.map(group => group.key), parityFolds, aggregate }, null, 2));
} finally {
  await server.close();
}
