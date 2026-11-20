import * as ort from 'onnxruntime-web/wasm';
import { FEATURE_COUNT, FEATURE_SCHEMA } from './features';

type SequenceConfig = {
  feature_schema: string;
  labels: (string | null)[];
  confidence_threshold: number;
  training_greek_references: { glosses: string[]; text: string }[];
};

export type ContinuousSegment = {
  kind: 'gloss' | 'blank' | 'gap';
  gloss: string | null;
  startMs: number;
  endMs: number;
  /** Geometric mean posterior of the CTC frames in this segment. */
  confidence: number | null;
  reason?: 'tracking' | 'unobserved' | 'low_confidence';
};

export type SequenceTimeline = {
  timestampsMs: (number | null)[];
  tracked: boolean[];
  durationMs: number;
};

type CtcRun = { label: number; start: number; end: number; confidence: number };

// This product display guard is not an estimate of correctness or open-set calibration.
const MINIMUM_GLOSS_CONFIDENCE = 0.6;
const MINIMUM_REFERENCE_TRACKING = 0.8;

function asset(path: string) {
  const url = new URL(path, globalThis.location.origin);
  if (url.origin !== globalThis.location.origin || url.username || url.password) {
    throw new Error('Only same-origin model artifacts are allowed.');
  }
  return fetch(url.href, { redirect: 'error' });
}

function greedyRuns(scores: Float32Array, count: number, width: number) {
  const runs: CtcRun[] = [];
  let label = 0, start = 0, sum = 0;
  function flush(end: number) {
    if (end > start) runs.push({ label, start, end, confidence: Math.min(1, Math.exp(sum / (end - start))) });
  }
  for (let index = 0; index < count; index++) {
    const offset = index * width;
    let best = 0;
    for (let col = 1; col < width; col++) if (scores[offset + col] > scores[offset + best]) best = col;
    if (best !== label) { flush(index); label = best; start = index; sum = 0; }
    sum += scores[offset + best];
  }
  flush(count);
  return runs;
}

/** Only absent interval boundaries are estimated between surrounding decoded frames. */
function intervalBoundaries({ timestampsMs, durationMs }: SequenceTimeline) {
  const values: number[] = [];
  let previousIndex = -1, previousTime = 0;
  for (let index = 0; index <= timestampsMs.length; index++) {
    const time = index === timestampsMs.length ? durationMs : timestampsMs[index];
    if (time === null) continue;
    for (let missing = previousIndex + 1; missing < index; missing++) {
      const fraction = (missing - previousIndex) / (index - previousIndex);
      values[missing] = previousTime + fraction * (time - previousTime);
    }
    values[index] = time;
    previousIndex = index;
    previousTime = time;
  }
  return values;
}

function decodeTimeline(runs: CtcRun[], scores: Float32Array, labels: SequenceConfig['labels'], timeline: SequenceTimeline) {
  const bounds = intervalBoundaries(timeline);
  const segments: ContinuousSegment[] = [];
  const glosses: string[] = [];
  let complete = timeline.timestampsMs.every(time => time !== null);
  const push = (segment: ContinuousSegment) => {
    if (segment.endMs <= segment.startMs) return;
    const previous = segments.at(-1);
    if (segment.kind === 'gap' && previous?.kind === 'gap' && previous.reason === segment.reason
      && previous.endMs === segment.startMs) {
      previous.endMs = segment.endMs;
    } else segments.push(segment);
  };
  if (bounds[0] > 0) push({ kind: 'gap', gloss: null, startMs: 0, endMs: bounds[0], confidence: null, reason: 'unobserved' });

  for (const run of runs) {
    const gloss = labels[run.label];
    const reasonAt = (index: number): ContinuousSegment['reason'] => {
      if (timeline.timestampsMs[index] === null) return 'unobserved';
      if (!timeline.tracked[index]) return 'tracking';
      if (gloss && run.confidence < MINIMUM_GLOSS_CONFIDENCE) return 'low_confidence';
    };
    let visible = false, start = run.start, reason = reasonAt(start), sum = 0;
    function flush(end: number) {
      if (reason && gloss) complete = false;
      if (!reason && gloss) visible = true;
      const confidence = reason ? null : Math.min(1, Math.exp(sum / (end - start)));
      push({ kind: reason ? 'gap' : gloss ? 'gloss' : 'blank', gloss: reason ? null : gloss,
        startMs: bounds[start], endMs: bounds[end], confidence, ...(reason ? { reason } : {}) });
    }
    for (let index = start; index < run.end; index++) {
      const next = reasonAt(index);
      if (next !== reason) { flush(index); start = index; sum = 0; reason = next; }
      sum += scores[index * labels.length + run.label];
    }
    flush(run.end);
    if (gloss && visible) glosses.push(gloss);
  }
  return { segments, glosses, complete };
}

export async function loadContinuousRecognizer(configUrl: string, modelUrl: string) {
  const response = await asset(configUrl);
  if (!response.ok) throw new Error('Continuous model config: HTTP ' + response.status);
  const config: SequenceConfig = await response.json();
  if (config.feature_schema !== FEATURE_SCHEMA || !Array.isArray(config.labels) || config.labels[0] !== null
    || config.labels.length < 2 || new Set(config.labels).size !== config.labels.length
    || config.labels.slice(1).some(label => typeof label !== 'string' || !label)
    || !Number.isFinite(config.confidence_threshold) || config.confidence_threshold < 0
    || !Array.isArray(config.training_greek_references)) throw new Error('Incompatible continuous model');
  const references = new Map<string, string>();
  for (const reference of config.training_greek_references) {
    if (!Array.isArray(reference.glosses) || !reference.glosses.length
      || reference.glosses.some(gloss => typeof gloss !== 'string' || !config.labels.includes(gloss))
      || typeof reference.text !== 'string' || !reference.text.trim()) throw new Error('Invalid continuous reference');
    const key = JSON.stringify(reference.glosses);
    if (references.has(key)) throw new Error('Ambiguous continuous reference');
    references.set(key, reference.text);
  }
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = '/vendor/onnxruntime/';
  const model = await asset(modelUrl);
  if (!model.ok) throw new Error('Continuous model: HTTP ' + model.status);
  const session = await ort.InferenceSession.create(await model.arrayBuffer(), { executionProviders: ['wasm'] });

  return async (features: number[][], timeline?: SequenceTimeline) => {
    const start = performance.now();
    if (!features.length || features.some(row => row.length !== FEATURE_COUNT || row.some(value => !Number.isFinite(value)))) {
      throw new Error('Invalid continuous feature sequence');
    }
    if (timeline) {
      let previous = -1;
      const { timestampsMs, durationMs, tracked } = timeline;
      if (!Number.isFinite(durationMs) || durationMs <= 0 || timestampsMs.length !== features.length
        || tracked.length !== features.length || timestampsMs.some(time => {
          if (time === null) return false;
          if (!Number.isFinite(time) || time <= previous || time < 0 || time >= durationMs) return true;
          previous = time;
          return false;
        })) throw new Error('Invalid decoded presentation timeline');
    }
    const inputs = new ort.Tensor('float32', Float32Array.from(features.flat()), [1, features.length, FEATURE_COUNT]);
    const result = await session.run({ frames: inputs });
    const output = result.log_probabilities;
    const scores = output?.data;
    const width = config.labels.length;
    if (!(scores instanceof Float32Array) || output.dims.join(',') !== [1, features.length, width].join(',')
      || scores.some(value => !Number.isFinite(value))) throw new Error('Unexpected CTC output');
    const runs = greedyRuns(scores, features.length, width);
    const emitted = runs.filter(run => run.label !== 0);
    const originalGlosses = emitted.map(run => config.labels[run.label]!);
    const confidence = emitted.length ? Math.min(...emitted.map(run => run.confidence)) : 0;
    const decoded = timeline ? decodeTimeline(runs, scores, config.labels, timeline) : undefined;
    const referenceEligible = !timeline || (decoded?.complete
      && timeline.tracked.filter(Boolean).length / features.length >= MINIMUM_REFERENCE_TRACKING);
    const text = referenceEligible && confidence >= config.confidence_threshold
      ? references.get(JSON.stringify(originalGlosses)) ?? null : null;
    return { glosses: decoded?.glosses ?? originalGlosses, text, confidence, segments: decoded?.segments ?? [],
      inference_ms: performance.now() - start };
  };
}
