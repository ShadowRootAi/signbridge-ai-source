/**
 * Open-set decision mathematics shared with ml/openset_core.py.
 * This module never fits thresholds or modifies preprocessed features.
 * Inputs originate as Float32 features/embeddings; calculations use JS doubles.
 */
export const OPENSET_DIAGNOSTIC_SCHEMA = 'signbridge-quality-motion-v1';

export interface ClipObservationMetadata {
  frameCount?: number | null;
  observedSpanMs?: number | null;
}

export interface QualityMotionDiagnostics {
  motionBodyRms: number;
  motionLocalRms: number;
  validTrackedPairFraction: number;
  /** Mean original fractional masks: left hand, right hand, pose, face. */
  meanPresence: [number, number, number, number];
}

export interface PrototypeGateConfig {
  embeddingName: string;
  embeddingSize: number;
  classOrder: string[];
  mean: number[];
  scale: number[];
  active: number[];
  activeCount: number;
  centroids: number[][];
  classDistanceCutoffs?: number[] | null;
  firstIndexWinsTies: true;
}

export interface OpenSetGateConfig {
  labels: string[];
  backgroundIndex: number;
  rejection: {
    temperature: number;
    confidenceThreshold: number;
    minQuality: number;
    minMotionBodyRms?: number;
  };
  prototypeGate?: PrototypeGateConfig | null;
}

export interface OpenSetGateResult {
  accepted: boolean;
  predicted: number;
  confidence: number;
  probabilities: number[];
  reasons: {
    quality: boolean;
    motion: boolean;
    confidence: boolean;
    background: boolean;
    prototypeDisagreement?: boolean;
    prototypeDistance?: boolean;
  };
  prototypeDistances?: number[];
  nearestPrototype?: number;
  predictedPrototypeDistance?: number;
}

function requireFinite(values: ArrayLike<number>, description: string) {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error(`${description} contains a non-finite value.`);
  }
}

/** Exactly 48x224 unstandardized features, before model-embedded normalization. */
export function measureQualityMotion(x: readonly (readonly number[])[]): QualityMotionDiagnostics {
  if (x.length !== 48 || x.some((row) => row.length !== 224)) {
    throw new Error('Expected a finite [48,224] preprocessed feature sequence.');
  }
  for (const row of x) requireFinite(row, 'Feature sequence');
  const meanPresence: [number, number, number, number] = [0, 0, 0, 0];
  for (let t = 0; t < 48; t++) for (let mask = 0; mask < 4; mask++) meanPresence[mask] += x[t][220 + mask] / 48;
  let bodySquare = 0, localSquare = 0, bodyPairs = 0, localPairs = 0;
  const present = (t: number, mask: number) => x[t][220 + mask] >= 1 - 1e-6;
  for (const [hand, bodyStart, localStart] of [[0, 0, 42], [1, 84, 126]] as const) {
    for (let t = 1; t < 48; t++) {
      const handPair = present(t - 1, hand) && present(t, hand);
      const bodyPair = handPair && present(t - 1, 2) && present(t, 2);
      if (handPair) localPairs++;
      if (bodyPair) bodyPairs++;
      for (let coordinate = 0; coordinate < 42; coordinate++) {
        if (bodyPair) {
          const delta = x[t][bodyStart + coordinate] - x[t - 1][bodyStart + coordinate];
          bodySquare += delta * delta;
        }
        if (handPair) {
          const delta = x[t][localStart + coordinate] - x[t - 1][localStart + coordinate];
          localSquare += delta * delta;
        }
      }
    }
  }
  return {
    motionBodyRms: bodyPairs > 0 ? Math.sqrt(bodySquare / (bodyPairs * 21)) : 0,
    motionLocalRms: localPairs > 0 ? Math.sqrt(localSquare / (localPairs * 21)) : 0,
    validTrackedPairFraction: bodyPairs / 94,
    meanPresence,
  };
}

function firstMaximum(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
function firstMinimum(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[best]) best = i;
  return best;
}

export function prototypeDistances(embedding: ArrayLike<number>, config: PrototypeGateConfig): number[] {
  const dimensions = config.embeddingSize;
  if (embedding.length !== dimensions || config.mean.length !== dimensions || config.scale.length !== dimensions
    || config.active.length !== dimensions || config.centroids.length !== config.classOrder.length
    || config.centroids.some((center) => center.length !== dimensions)) throw new Error('Prototype dimensions do not match.');
  requireFinite(embedding, 'Embedding');
  requireFinite(config.mean, 'Prototype mean');
  requireFinite(config.scale, 'Prototype scale');
  if (config.scale.some((scale) => scale <= 0) || config.active.some((active) => active !== 0 && active !== 1)) {
    throw new Error('Invalid prototype scale or active mask.');
  }
  const activeCount = config.active.reduce((sum, value) => sum + value, 0);
  if (!activeCount || activeCount !== config.activeCount) throw new Error('Prototype candidate has no valid active dimension count.');
  const normalized = Array.from({ length: dimensions }, (_, i) => (embedding[i] - config.mean[i]) / config.scale[i] * config.active[i]);
  return config.centroids.map((center) => {
    requireFinite(center, 'Prototype centroid');
    let sum = 0;
    for (let i = 0; i < dimensions; i++) {
      const delta = normalized[i] - center[i];
      sum += delta * delta;
    }
    return Math.sqrt(sum / activeCount);
  });
}

/** Apply already-frozen configuration. No test data can tune this function. */
export function scoreOpenSet(
  logits: ArrayLike<number>, trackingQuality: number, motionBodyRms: number,
  config: OpenSetGateConfig, embedding?: ArrayLike<number>,
): OpenSetGateResult {
  const known = config.labels.length;
  const gate = config.rejection;
  if (known < 1 || logits.length !== known + 1) throw new Error('Expected known-class logits and exactly one OTHER/background output.');
  requireFinite(logits, 'Logits');
  requireFinite([trackingQuality, motionBodyRms, gate.temperature, gate.confidenceThreshold, gate.minQuality, gate.minMotionBodyRms ?? 0], 'Gate input/configuration');
  if (gate.temperature <= 0) throw new Error('Gate temperature must be positive.');
  if (trackingQuality < 0 || trackingQuality > 1 || motionBodyRms < 0) throw new Error('Out-of-range quality/motion metadata.');
  if (embedding) requireFinite(embedding, 'Embedding');
  const scaled = Array.from(logits, (value) => value / gate.temperature);
  const maximum = Math.max(...scaled);
  const exp = scaled.map((value) => Math.exp(value - maximum));
  const total = exp.reduce((sum, value) => sum + value, 0);
  const probabilities = exp.map((value) => value / total);
  const predicted = firstMaximum(probabilities.slice(0, known));
  const confidence = probabilities[predicted];
  const result: OpenSetGateResult = {
    predicted, confidence, probabilities, accepted: false,
    reasons: {
      quality: trackingQuality < Math.max(0.25, gate.minQuality),
      motion: motionBodyRms < (gate.minMotionBodyRms ?? 0),
      confidence: confidence < gate.confidenceThreshold,
      background: firstMaximum(probabilities) >= known,
    },
  };
  const prototype = config.prototypeGate;
  if (prototype) {
    if (!embedding) throw new Error('Prototype gate requires the real model embedding.');
    if (prototype.classOrder.length !== known || prototype.classOrder.some((label, i) => label !== config.labels[i])) {
      throw new Error('Prototype class order does not match model labels.');
    }
    const distances = prototypeDistances(embedding, prototype);
    const nearest = firstMinimum(distances);
    const distance = distances[predicted];
    const cutoffs = prototype.classDistanceCutoffs;
    if (cutoffs && cutoffs.length !== known) throw new Error('Prototype cutoff dimensions do not match classes.');
    if (cutoffs) requireFinite(cutoffs, 'Prototype distance cutoffs');
    result.prototypeDistances = distances;
    result.nearestPrototype = nearest;
    result.predictedPrototypeDistance = distance;
    result.reasons.prototypeDisagreement = nearest !== predicted;
    result.reasons.prototypeDistance = cutoffs ? distance > cutoffs[predicted] : false;
  }
  result.accepted = !Object.values(result.reasons).some(Boolean);
  return result;
}
