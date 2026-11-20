/** Isolated Greek-sign recognition with OpenHands ST-GCN, independent of UI. */
import * as ort from 'onnxruntime-web/wasm';
import { preprocess } from './features';
import type { LandmarkFrame, Point } from './features';
import { measureQualityMotion } from './openset';
import type { RecognitionResult } from './pipeline-contract';
import { measureRawTracking } from './tracking';

const JOINTS = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74];
const STGCN_ONNX_SHA256 = '8ace9360ed396ecd6df2670a9c34d1bab271d6384808f2972456355c577fbd79';
const LSTM_ONNX_SHA256 = '78cec3333c33c1880de3da421d0ae4c8b914e7536827662f4db7d3aa630426d1';
const MIN_TRACKING_COVERAGE = 0.6;
// Engineering input check. The native GSL policy did not calibrate webcam motion.
const MIN_BODY_MOTION = 0.004;
// Extra evidence for recovery: a rigid translation of the hand alone is insufficient.
const MIN_RECOVERY_HAND_MOTION = 0.005;

interface OpenHandsConfig {
  labels: string[];
  input: string;
  input_shape: [number, number, string, number];
  output: string;
  source_checkpoint_sha256: string;
  model_sha256: string;
}

export interface OpenHandsPolicy {
  schemaVersion: 1;
  sourceModelSha256: string;
  runtimeModelSha256: string;
  supportedLabels: string[];
  unknownLabels: string[];
  minConfidence: number;
  minMargin: number;
  displayLabels: Record<string, string>;
  decisionGate?: {
    method: 'winningLogit';
    minimumConfidence: number;
    minimumEvidence: number;
  };
  expansionGate?: {
    labels: string[];
    minimumConfidence: number;
  };
  agreementGate?: {
    modelSha256: string;
    minimumConfidence: number;
    minimumSecondaryProbability: number;
    maximumSecondaryRank: number;
    temporal?: {
      minimumPrimaryConfidence: number;
      minimumPeakConfidence: number;
      minimumConsistencyConfidence: number;
      minimumSecondaryProbability: number;
    };
  };
}

export interface OpenHandsSources { modelUrl: string; configUrl: string; policyUrl: string; secondaryModelUrl?: string }
export type OpenHandsRejection = 'tracking' | 'no_motion' | 'outside_roster' | 'low_confidence' | 'low_margin' | 'low_evidence';
export interface OpenHandsCandidate { gloss: string; score: number }
type RecoveryEvidence = {
  primaryScore: number;
  primaryThreshold: number;
  secondarySupport: number;
  temporalChecks: { factor: number; frames: number; score: number }[];
};
export type OpenHandsRecognition = Omit<RecognitionResult, 'frames' | 'elapsedMs'> & {
  topCandidates: OpenHandsCandidate[];
  rejectionCode: OpenHandsRejection | null;
  acceptedBy?: 'primary' | 'agreement' | 'temporal';
  recoveryEvidence?: RecoveryEvidence;
};

function sameOrigin(path: string) {
  const url = new URL(path, globalThis.location.origin);
  if (url.origin !== globalThis.location.origin || url.username || url.password) {
    throw new Error('OpenHands model and policy must be same-origin');
  }
  return url.href;
}

async function fetchAsset(path: string) {
  const response = await fetch(sameOrigin(path), { cache: 'no-store', redirect: 'error' });
  if (!response.ok) throw new Error(`OpenHands asset HTTP ${response.status}`);
  return response;
}

function coordinates(frame: LandmarkFrame) {
  const part = (points: Point[], count: number) => {
    if (points.length < count || points.slice(0, count).some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
      return Array.from({ length: count }, () => [0, 0]);
    }
    return points.slice(0, count).map(p => [p.x, p.y]);
  };
  return [...part(frame.pose, 33), ...part(frame.left, 21), ...part(frame.right, 21)];
}

/** OpenHands [1,2,T,27] from original [body,left,right] XY landmarks. */
export function prepareOpenHandsPose(frames: LandmarkFrame[]): Float32Array | null {
  if (!frames.length) return null;
  const points = frames.map(coordinates);
  const center = [0, 0];
  let width = 0;
  for (const frame of points) {
    const left = frame[11], right = frame[12];
    center[0] += (left[0] + right[0]) / 2;
    center[1] += (left[1] + right[1]) / 2;
    width += Math.hypot(left[0] - right[0], left[1] - right[1]);
  }
  width /= points.length;
  if (!Number.isFinite(width) || width <= 0) return null;
  return Float32Array.from(center.flatMap((sum, channel) => {
    const origin = sum / points.length;
    return points.flatMap(frame => JOINTS.map(index => (frame[index][channel] - origin) / width));
  }));
}

function topCandidates(logits: Float32Array, labels: string[]): OpenHandsCandidate[] {
  if (logits.length !== labels.length || logits.some(value => !Number.isFinite(value))) {
    throw new Error('Invalid OpenHands logits');
  }
  const max = Math.max(...logits);
  const total = logits.reduce((sum, value) => sum + Math.exp(value - max), 0);
  return Array.from(logits, (value, index) => ({ gloss: labels[index], score: Math.exp(value - max) / total }))
    .sort((a, b) => b.score - a.score).slice(0, 5);
}

async function checkedModel(path: string, expectedChecksum: string) {
  const bytes = await fetchAsset(path).then(response => response.arrayBuffer());
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const checksum = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  if (checksum !== expectedChecksum) throw new Error('Απέτυχε ο έλεγχος ακεραιότητας του μοντέλου OpenHands.');
  return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
}

function forLstm(points: Float32Array, frames: number) {
  const joints = JOINTS.length;
  const channelSize = frames * joints;
  const input = new Float32Array(points.length);
  for (let frame = 0; frame < frames; frame++) {
    const start = frame * joints;
    input.set(points.subarray(start, start + joints), frame * 2 * joints);
    input.set(points.subarray(channelSize + start, channelSize + start + joints), (2 * frame + 1) * joints);
  }
  return new ort.Tensor('float32', input, [1, frames, joints * 2]);
}

function resampledPose(points: Float32Array, frames: number, count: number) {
  const joints = JOINTS.length;
  const sampled = new Float32Array(2 * count * joints);
  for (let channel = 0; channel < 2; channel++) {
    for (let i = 0; i < count; i++) {
      const time = i * (frames - 1) / (count - 1);
      const before = Math.floor(time);
      const after = Math.min(before + 1, frames - 1);
      for (let j = 0; j < joints; j++) {
        const a = points[(channel * frames + before) * joints + j];
        const b = points[(channel * frames + after) * joints + j];
        sampled[(channel * count + i) * joints + j] = a + (b - a) * (time - before);
      }
    }
  }
  return new ort.Tensor('float32', sampled, [1, 2, count, joints]);
}

function temporalViews(frames: number) {
  const views = [0.65, 0.8].map(factor => ({ factor, frames: Math.max(4, Math.round(frames * factor)) }));
  return views[0].frames < views[1].frames && views[1].frames < frames ? views : [];
}

export async function loadOpenHandsRecognizer({ modelUrl, configUrl, policyUrl, secondaryModelUrl }: OpenHandsSources) {
  const [modelConfig, policyData] = await Promise.all([
    fetchAsset(configUrl).then(response => response.json()),
    fetchAsset(policyUrl).then(response => response.json()),
  ]);
  const config: OpenHandsConfig = modelConfig;
  const policy: OpenHandsPolicy = policyData;
  if (config.input !== 'landmarks' || config.output !== 'logits' || config.model_sha256 !== STGCN_ONNX_SHA256
    || !Array.isArray(config.input_shape) || config.input_shape.join(',') !== '1,2,time,27'
    || !Array.isArray(config.labels) || config.labels.length !== 310 || new Set(config.labels).size !== 310
    || config.labels.some(label => typeof label !== 'string' || !label)) {
    throw new Error('Ασυμβατό αρχείο ρυθμίσεων OpenHands.');
  }
  const isScore = (value: number) => Number.isFinite(value) && value >= 0 && value <= 1;
  const policyLabels = [...(Array.isArray(policy.supportedLabels) ? policy.supportedLabels : []),
    ...(Array.isArray(policy.unknownLabels) ? policy.unknownLabels : [])];
  if (policy.schemaVersion !== 1 || policy.sourceModelSha256 !== config.source_checkpoint_sha256
    || policy.runtimeModelSha256 !== config.model_sha256
    || !Array.isArray(policy.supportedLabels) || !policy.supportedLabels.length
    || policyLabels.length !== config.labels.length || new Set(policyLabels).size !== config.labels.length
    || policyLabels.some(label => !config.labels.includes(label) || typeof policy.displayLabels?.[label] !== 'string')
    || !isScore(policy.minConfidence) || !isScore(policy.minMargin)) {
    throw new Error('Ασυμβίβαστο λεξιλόγιο ή όρια αποδοχής OpenHands.');
  }
  const gate = policy.decisionGate;
  if (gate !== undefined && (!gate || !isScore(gate.minimumConfidence) || !Number.isFinite(gate.minimumEvidence)
    || gate.method !== 'winningLogit')) throw new Error('Ασυμβίβαστος έλεγχος αποδοχής OpenHands.');
  const expansion = policy.expansionGate;
  if (expansion !== undefined && (!expansion || !gate || !Array.isArray(expansion.labels)
    || !expansion.labels.length || new Set(expansion.labels).size !== expansion.labels.length
    || expansion.labels.some(label => !policy.supportedLabels.includes(label))
    || !isScore(expansion.minimumConfidence))) {
    throw new Error('Ασυμβίβαστο διευρυμένο λεξιλόγιο OpenHands.');
  }
  const agreement = policy.agreementGate;
  if (agreement !== undefined && (!agreement || !gate || !secondaryModelUrl || agreement.modelSha256 !== LSTM_ONNX_SHA256
    || !isScore(agreement.minimumConfidence) || agreement.minimumConfidence > gate.minimumConfidence
    || !isScore(agreement.minimumSecondaryProbability) || !Number.isInteger(agreement.maximumSecondaryRank)
    || agreement.maximumSecondaryRank < 1 || agreement.maximumSecondaryRank > 5)) {
    throw new Error('Ασυμβίβαστος έλεγχος συμφωνίας OpenHands.');
  }
  const temporal = agreement?.temporal;
  if (temporal !== undefined && (!temporal || !isScore(temporal.minimumPrimaryConfidence)
    || !isScore(temporal.minimumPeakConfidence) || !isScore(temporal.minimumConsistencyConfidence)
    || !isScore(temporal.minimumSecondaryProbability)
    || temporal.minimumConsistencyConfidence > temporal.minimumPeakConfidence)) {
    throw new Error('Ασυμβίβαστος έλεγχος ρυθμού OpenHands.');
  }
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = '/vendor/onnxruntime/';
  let primary: Promise<ort.InferenceSession> | undefined;
  let secondary: Promise<ort.InferenceSession> | undefined;
  const roster = new Set(policy.supportedLabels);
  const expansionRoster = new Set(expansion?.labels);
  const classIndex = new Map(config.labels.map((label, index) => [label, index]));
  const versionSuffix = temporal ? '-agreement-temporal-v2' : agreement ? '-agreement-v1' : gate ? '-evidence-v1' : '';
  const expansionSuffix = expansion ? `-roster-${roster.size}` : '';
  const modelVersion = `openhands-gsl-stgcn-${config.model_sha256.slice(0, 8)}${versionSuffix}${expansionSuffix}`;

  async function classify(frames: LandmarkFrame[], allowRecovery = false): Promise<OpenHandsRecognition> {
    const tracking = measureRawTracking(frames);
    const trackingQuality = tracking.eitherHandDetection;
    const skipped = (rejectionCode: 'tracking' | 'no_motion', reason: string): OpenHandsRecognition => ({
      accepted: false, label: null, confidence: null, classificationScore: null, classifierExecuted: false,
      reason, trackingQuality, modelVersion, topCandidates: [], rejectionCode,
    });
    if (frames.length < tracking.minimumFrames || tracking.invalidMetadataFrames > 0
      || !tracking.timestampsStrictlyIncreasing || trackingQuality < MIN_TRACKING_COVERAGE) {
      return skipped('tracking', 'Χρειάζεται επανάληψη: κράτησε το πάνω μέρος του σώματος και το χέρι που νοηματίζει ορατά.');
    }
    const points = prepareOpenHandsPose(frames);
    if (!points) return skipped('tracking', 'Χρειάζεται επανάληψη: δεν εντοπίστηκαν οι ώμοι.');
    const motion = measureQualityMotion(preprocess(frames).x);
    if (motion.motionBodyRms < MIN_BODY_MOTION) {
      return skipped('no_motion', 'Δεν εντοπίστηκε αρκετή κίνηση. Δοκίμασε ξανά.');
    }
    primary ??= checkedModel(modelUrl, config.model_sha256).catch(error => {
      primary = undefined;
      throw error;
    });
    const session = await primary;
    const tensor = new ort.Tensor('float32', points, [1, 2, frames.length, JOINTS.length]);
    const result = await session.run({ landmarks: tensor });
    const logits = result.logits.data as Float32Array;
    const candidates = topCandidates(logits, config.labels);
    const [best, second] = candidates;
    const winningLogit = logits[classIndex.get(best.gloss)!];
    const added = expansionRoster.has(best.gloss);
    const confidenceFloor = (added ? expansion?.minimumConfidence : gate?.minimumConfidence) ?? policy.minConfidence;
    let rejectionCode: OpenHandsRejection | null = null;
    if (!roster.has(best.gloss)) rejectionCode = 'outside_roster';
    else if (best.score < confidenceFloor) rejectionCode = 'low_confidence';
    else if (gate && winningLogit < gate.minimumEvidence) rejectionCode = 'low_evidence';
    else if (!gate && best.score - second.score < policy.minMargin) rejectionCode = 'low_margin';
    let acceptedBy: OpenHandsRecognition['acceptedBy'] = rejectionCode === null ? 'primary' : undefined;
    let recoveryEvidence: RecoveryEvidence | undefined;
    if (allowRecovery && !added && motion.motionLocalRms >= MIN_RECOVERY_HAND_MOTION
      && agreement && gate && secondaryModelUrl && rejectionCode === 'low_confidence'
      && best.score >= agreement.minimumConfidence && winningLogit >= gate.minimumEvidence) {
      // Keep the primary decision if the optional verifier cannot run.
      try {
        secondary ??= checkedModel(secondaryModelUrl, agreement.modelSha256).catch(error => {
          secondary = undefined;
          throw error;
        });
        const verifier = await secondary;
        const checked = await verifier.run({ landmarks: forLstm(points, frames.length) });
        const support = topCandidates(checked.logits.data as Float32Array, config.labels)
          .slice(0, agreement.maximumSecondaryRank).find(candidate => candidate.gloss === best.gloss)?.score ?? 0;
        recoveryEvidence = { primaryScore: best.score, primaryThreshold: confidenceFloor,
          secondarySupport: support, temporalChecks: [] };
        if (support >= agreement.minimumSecondaryProbability) {
          rejectionCode = null;
          acceptedBy = 'agreement';
        } else if (temporal && support >= temporal.minimumSecondaryProbability
          && best.score >= temporal.minimumPrimaryConfidence) {
          for (const view of temporalViews(frames.length)) {
            const { logits: alternative } = await session.run({ landmarks: resampledPose(points, frames.length, view.frames) });
            const scores = alternative.data as Float32Array;
            const [winner] = topCandidates(scores, config.labels);
            const score = winner.gloss === best.gloss && scores[classIndex.get(best.gloss)!] >= gate.minimumEvidence
              ? winner.score : 0;
            recoveryEvidence.temporalChecks.push({ ...view, score });
          }
          const checks = recoveryEvidence.temporalChecks;
          if (checks.length === 2 && checks.every(check => check.score >= temporal.minimumConsistencyConfidence)
            && checks.some(check => check.score >= temporal.minimumPeakConfidence)) {
            rejectionCode = null;
            acceptedBy = 'temporal';
          }
        }
      } catch {
        secondary = undefined;
      }
    }
    const accepted = rejectionCode === null;
    return {
      accepted, label: accepted ? best.gloss : null, confidence: best.score, classificationScore: best.score,
      classifierExecuted: true, trackingQuality, modelVersion, topCandidates: candidates, rejectionCode, acceptedBy,
      ...(recoveryEvidence && { recoveryEvidence }),
      reason: accepted ? 'Πρόβλεψη νοήματος. Μπορείς να την ελέγξεις πριν τη χρησιμοποιήσεις.'
        : 'Δεν αναγνωρίστηκε με αρκετή βεβαιότητα υποστηριζόμενο νόημα.',
    };
  }

  return { labels: [...roster], modelVersion, modelSha256: config.model_sha256, policyVersion: policy.schemaVersion,
    displayLabels: policy.displayLabels ?? {}, rejectionCalibrated: true, classify };
}
