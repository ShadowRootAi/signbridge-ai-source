/** Isolated research comparison. Never changes the product pipeline/model source. */
import { FEATURE_COUNT, FEATURE_SCHEMA, TIMESTEPS, preprocess } from './features';
import type { LandmarkFrame } from './features';
import { measureQualityMotion, scoreOpenSet } from './openset';
import type { OpenSetGateConfig, QualityMotionDiagnostics } from './openset';
import { measureRawTracking, trackingRetryMessage } from './tracking';
import type { TrackingDiagnostics } from './tracking';
import type { ProgressListener } from './pipeline-contract';

export const V8_PILOT_HASHES = Object.freeze({
  model: 'beaed67bc63103370754cdbf25eda012f4464c66af74a11a7923a127254c2a45',
  baselineConfig: '817c389851b13c61b0b7e0feb68e62b848082d049b8abfd7bb5adeb36ab1b7d6',
  v8Config: 'bc0b931319d25986f3c14b9b9aba699ffe448d200f6c147b4b62cf4791f21bc7',
  handModel: 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
});
const LABELS = ['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ'];
const TIMEOUT_MS = 120_000;
// Copied verbatim from the locked V6 rule; the original worker implements it.
const RECOVERY_RULE = Object.freeze({
  minimumPoseVisibility: .5, roiSideShoulderWidths: 1.6,
  roiSideMinimumPixels: 96, roiSideMaximumPixels: 384,
  candidateWristDistanceMaximumShoulderWidths: .45,
  oppositeWristDistanceMinimumShoulderWidths: .5,
  oppositeDistanceMustExceedTargetDistanceBy: 1.5,
  requiredHandedness: 'Right', minimumHandednessScore: .8,
});
interface BaselineConfig extends OpenSetGateConfig {
  modelVersion: string; featureSchema: string; T: number; F: number;
  input: { name: string; shape: number[]; dtype: string };
  output: { name: string; embeddingName: string; labels: string[] };
}
interface PilotConfig extends Omit<BaselineConfig, 'rejection'> {
  featureOrigin: string; researchOnly: boolean;
  rejection: { temperature: number; thresholds: number[] };
}
export interface V8PilotOptions {
  v8ConfigUrl: string;
  handModelUrl: string;
  baselineConfigUrl?: string;
  modelUrl?: string;
}
export interface V8PilotDecision {
  accepted: boolean;
  label: string | null;
  predicted: number | null;
  /** Frozen softmax score, never a correctness probability. Null before inference. */
  classificationScore: number | null;
  threshold: number | null;
  reasons: string[];
}
export interface V8PilotTimings {
  modelLoadMs: number;
  decodeMs: number;
  recoveryMs: number;
  recoveryWorkerMs: number;
  recoveryInitMs: number;
  preprocessMs: number;
  /** Includes worker messaging and its diagnostics; not isolated ONNX execution. */
  inferenceRoundTripMs: number;
  decisionMs: number;
  totalMs: number;
}
export interface V8PilotComparison {
  researchOnly: true;
  source: 'video';
  modelVersion: string;
  baselineModelVersion: string;
  baseline: V8PilotDecision;
  v8: V8PilotDecision;
  /** One inference shared by both decisions; null when raw completeness fails. */
  logits: number[] | null;
  probabilities: number[] | null;
  classifierExecuted: boolean;
  originalTracking: TrackingDiagnostics | null;
  fusedTracking: TrackingDiagnostics | null;
  quality: number | null;
  diagnostics: QualityMotionDiagnostics | null;
  frames: number;
  recoveredFrames: number;
  timings: V8PilotTimings;
  hashes: typeof V8_PILOT_HASHES;
}
interface RecoveryOutput {
  frames: { original: LandmarkFrame; fused: LandmarkFrame; recovered: boolean }[];
  initMs: number; totalMs: number;
}
interface InferenceOutput { logits: number[]; embedding?: number[]; accepted: boolean }
const emptyTimings = (): V8PilotTimings => ({ modelLoadMs: 0, decodeMs: 0, recoveryMs: 0,
  recoveryWorkerMs: 0, recoveryInitMs: 0, preprocessMs: 0, inferenceRoundTripMs: 0, decisionMs: 0, totalMs: 0 });
const abortError = () => new DOMException('Η δοκιμή ακυρώθηκε.', 'AbortError');

function localUrl(value: string): string {
  const url = new URL(value, window.location.href);
  if (url.origin !== window.location.origin || url.username || url.password
    || !['http:', 'https:', 'blob:'].includes(url.protocol)) throw new Error('Η δοκιμή επιτρέπει μόνο τοπικά αρχεία του ίδιου ιστοτόπου.');
  return url.href;
}
async function verifiedBytes(url: string, hash: string, maximum: number, signal: AbortSignal) {
  const response = await fetch(localUrl(url), { cache: 'no-store', signal });
  if (!response.ok) throw new Error(`Δεν φορτώθηκε το αρχείο δοκιμής (${response.status}).`);
  if (response.url) localUrl(response.url);
  if (Number(response.headers.get('content-length')) > maximum) throw new Error('Το αρχείο δοκιμής υπερβαίνει το επιτρεπόμενο μέγεθος.');
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > maximum) throw new Error('Το αρχείο δοκιμής υπερβαίνει το επιτρεπόμενο μέγεθος.');
  const actual = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), n => n.toString(16).padStart(2, '0')).join('');
  signal.throwIfAborted();
  if (actual !== hash) throw new Error('Το αρχείο δεν αντιστοιχεί στην παγωμένη έκδοση της δοκιμής V8.');
  return bytes;
}
function validateConfigs(baseline: BaselineConfig, pilot: PilotConfig) {
  for (const config of [baseline, pilot]) {
    if (config.featureSchema !== FEATURE_SCHEMA || config.T !== TIMESTEPS || config.F !== FEATURE_COUNT
      || JSON.stringify(config.labels) !== JSON.stringify(LABELS) || config.backgroundIndex !== 3
      || config.input.name !== 'landmarks' || config.input.dtype !== 'float32'
      || JSON.stringify(config.input.shape) !== '[1,48,224]' || config.output.name !== 'logits'
      || config.output.embeddingName !== 'embedding') throw new Error('Ασύμβατη διαμόρφωση δοκιμής.');
  }
  if (pilot.researchOnly !== true || pilot.featureOrigin !== 'fused'
    || pilot.rejection.temperature !== baseline.rejection.temperature
    || !Array.isArray(pilot.rejection.thresholds) || pilot.rejection.thresholds.length !== 3
    || pilot.rejection.thresholds.some(t => !Number.isFinite(t) || t < baseline.rejection.confidenceThreshold || t > 1)) {
    throw new Error('Μη έγκυρα όρια απόρριψης V8.');
  }
}
function callWorker<T>(worker: Worker, payload: Record<string, unknown>, signal: AbortSignal): Promise<T> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('Η επεξεργασία ξεπέρασε τα 120 δευτερόλεπτα.')), TIMEOUT_MS);
    const onAbort = () => finish(signal.reason ?? abortError());
    const onError = (event: ErrorEvent) => finish(new Error(event.message || 'Αποτυχία τοπικής επεξεργασίας.'));
    const onMessageError = () => finish(new Error('Μη έγκυρη απάντηση επεξεργασίας.'));
    const onMessage = ({ data }: MessageEvent) => data.error ? finish(new Error(data.error)) : finish(undefined, ('value' in data ? data.value : data) as T);
    function finish(error?: unknown, value?: T) {
      clearTimeout(timer); signal.removeEventListener('abort', onAbort);
      worker.removeEventListener('message', onMessage); worker.removeEventListener('error', onError);
      worker.removeEventListener('messageerror', onMessageError);
      if (error) { worker.terminate(); reject(error); } else resolve(value as T);
    }
    signal.addEventListener('abort', onAbort, { once: true });
    worker.addEventListener('message', onMessage); worker.addEventListener('error', onError);
    worker.addEventListener('messageerror', onMessageError);
    try { worker.postMessage(payload); } catch (error) { finish(error); }
  });
}
async function seekDecoded(video: HTMLVideoElement, t: number, signal: AbortSignal) {
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('Η αποκωδικοποίηση του βίντεο καθυστέρησε.')), 15_000);
    const onSeek = () => finish();
    const onError = () => finish(new Error('Το βίντεο δεν μπορεί να αποκωδικοποιηθεί.'));
    const onAbort = () => finish(signal.reason ?? abortError());
    function finish(error?: unknown) {
      clearTimeout(timer); video.removeEventListener('seeked', onSeek); video.removeEventListener('error', onError);
      signal.removeEventListener('abort', onAbort);
      if (error) reject(error); else resolve();
    }
    video.addEventListener('seeked', onSeek, { once: true }); video.addEventListener('error', onError, { once: true });
    signal.addEventListener('abort', onAbort, { once: true });
    try { video.currentTime = t; } catch (error) { finish(error); }
  });
  if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) throw new Error('Δεν υπάρχει αποκωδικοποιημένο καρέ.');
}

export function createV8Pilot(options: V8PilotOptions) {
  let active: AbortController | undefined, disposed = false, classifier: Worker | undefined;
  let assets: { baseline: BaselineConfig; pilot: PilotConfig; configUrl: string; modelUrl: string; handUrl: string } | undefined;
  const objectUrls = new Set<string>();
  const blobUrl = (bytes: ArrayBuffer, type: string) => {
    const url = URL.createObjectURL(new Blob([bytes], { type })); objectUrls.add(url); return url;
  };
  async function operation<T>(work: (signal: AbortSignal) => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (disposed) throw new Error('Η δοκιμή έχει κλείσει.');
    if (active) throw new Error('Μία δοκιμή εκτελείται ήδη.');
    signal?.throwIfAborted();
    const controller = new AbortController(); active = controller;
    const relay = () => controller.abort(signal?.reason ?? abortError());
    signal?.addEventListener('abort', relay, { once: true });
    const timer = setTimeout(() => controller.abort(new Error('Η δοκιμή ξεπέρασε τα 120 δευτερόλεπτα.')), TIMEOUT_MS);
    try { const result = await work(controller.signal); controller.signal.throwIfAborted(); return result; }
    catch (error) { classifier?.terminate(); classifier = undefined; throw error; }
    finally { clearTimeout(timer); signal?.removeEventListener('abort', relay); active = undefined; }
  }
  async function prepare(signal: AbortSignal) {
    if (!assets) {
      const [baseBytes, pilotBytes, modelBytes, handBytes] = await Promise.all([
        verifiedBytes(options.baselineConfigUrl ?? '/models/model_config.json', V8_PILOT_HASHES.baselineConfig, 65_536, signal),
        verifiedBytes(options.v8ConfigUrl, V8_PILOT_HASHES.v8Config, 65_536, signal),
        verifiedBytes(options.modelUrl ?? '/models/model.onnx', V8_PILOT_HASHES.model, 5_000_000, signal),
        verifiedBytes(options.handModelUrl, V8_PILOT_HASHES.handModel, 20_000_000, signal),
      ]);
      const baseline = JSON.parse(new TextDecoder().decode(baseBytes)) as BaselineConfig;
      const pilot = JSON.parse(new TextDecoder().decode(pilotBytes)) as PilotConfig;
      validateConfigs(baseline, pilot);
      assets = { baseline, pilot, configUrl: blobUrl(baseBytes, 'application/json'),
        modelUrl: blobUrl(modelBytes, 'application/octet-stream'), handUrl: blobUrl(handBytes, 'application/octet-stream') };
    }
    if (!classifier) {
      classifier = new Worker(new URL('./recognizer.worker.ts', import.meta.url), { type: 'module' });
      await callWorker(classifier, { id: 1, kind: 'status', configUrl: assets.configUrl, modelUrl: assets.modelUrl }, signal);
    }
    return assets;
  }
  function initial(source: V8PilotComparison['source'], timings: V8PilotTimings): V8PilotComparison {
    return { researchOnly: true, source, modelVersion: assets!.pilot.modelVersion,
      baselineModelVersion: assets!.baseline.modelVersion,
      baseline: { accepted: false, label: null, predicted: null, classificationScore: null, threshold: null, reasons: [] },
      v8: { accepted: false, label: null, predicted: null, classificationScore: null, threshold: null, reasons: [] },
      logits: null, probabilities: null, classifierExecuted: false, originalTracking: null, fusedTracking: null,
      quality: null, diagnostics: null, frames: 0, recoveredFrames: 0, timings, hashes: V8_PILOT_HASHES };
  }
  async function infer(x: number[][], quality: number, result: V8PilotComparison, signal: AbortSignal) {
    const diagnostics = measureQualityMotion(x);
    if (!Number.isFinite(quality) || quality < 0 || quality > 1) throw new Error('Μη έγκυρη μέτρηση παρουσίας landmarks.');
    const started = performance.now();
    const output = await callWorker<InferenceOutput>(classifier!, { id: 2, kind: 'infer', x, quality,
      configUrl: assets!.configUrl, modelUrl: assets!.modelUrl }, signal);
    result.timings.inferenceRoundTripMs = performance.now() - started;
    const decisionStart = performance.now();
    const score = scoreOpenSet(output.logits, quality, diagnostics.motionBodyRms, assets!.baseline, output.embedding);
    if (score.accepted !== output.accepted) throw new Error('Οι κοινές αποφάσεις του μοντέλου δεν συμφωνούν.');
    const threshold = assets!.pilot.rejection.thresholds[score.predicted];
    const reasons = Object.entries(score.reasons).filter(([, failed]) => failed).map(([name]) => name);
    const accepted = score.accepted && score.confidence >= threshold;
    result.baseline = { accepted: score.accepted, label: score.accepted ? LABELS[score.predicted] : null,
      predicted: score.predicted, classificationScore: score.confidence, threshold: assets!.baseline.rejection.confidenceThreshold, reasons };
    result.v8 = { accepted, label: accepted ? LABELS[score.predicted] : null, predicted: score.predicted,
      classificationScore: score.confidence, threshold, reasons: [...reasons, ...(score.confidence < threshold ? ['v8_confidence_veto'] : [])] };
    result.logits = output.logits; result.probabilities = score.probabilities;
    result.classifierExecuted = true; result.quality = quality; result.diagnostics = diagnostics;
    result.timings.decisionMs = performance.now() - decisionStart;
    return result;
  }
  async function compareFrames(original: LandmarkFrame[], fused: LandmarkFrame[], result: V8PilotComparison, signal: AbortSignal, progress?: ProgressListener) {
    if (!Array.isArray(original) || !Array.isArray(fused) || original.length !== fused.length || fused.length > 48) throw new Error('Μη συμβατά καρέ σύγκρισης.');
    result.frames = fused.length; result.originalTracking = measureRawTracking(original); result.fusedTracking = measureRawTracking(fused);
    if (!result.fusedTracking.usable) {
      const reason = trackingRetryMessage(result.fusedTracking);
      result.baseline.reasons = [reason]; result.v8.reasons = [reason]; return result;
    }
    signal.throwIfAborted(); progress?.({ stage: 'inference', message: 'Κοινή εκτέλεση μοντέλου για V4 και V8…', frames: fused.length });
    const start = performance.now(), features = preprocess(fused); result.timings.preprocessMs = performance.now() - start;
    return infer(features.x, features.quality, result, signal);
  }
  async function recover(paths: string[], timestamps: number[], result: V8PilotComparison, signal: AbortSignal, progress?: ProgressListener) {
    progress?.({ stage: 'tracking', message: 'Ανίχνευση και ελεγχόμενη ανάκτηση δεξιού χεριού…' });
    const started = performance.now();
    const worker = new Worker(new URL('./v6_hand_recovery.worker.ts', import.meta.url), { type: 'module' });
    try {
      const output = await callWorker<RecoveryOutput>(worker, { paths, rule: RECOVERY_RULE, control: 'none', handModelUrl: assets!.handUrl }, signal);
      result.timings.recoveryMs = performance.now() - started;
      result.timings.recoveryWorkerMs = output.totalMs; result.timings.recoveryInitMs = output.initMs;
      if (output.frames.length !== paths.length) throw new Error('Λείπουν καρέ από την ανάκτηση.');
      result.recoveredFrames = output.frames.filter(frame => frame.recovered).length;
      // V6 processes JPEG sequences at30FPS. Restore actual sample times for videos.
      const original = output.frames.map((frame, i) => ({ ...frame.original, t: timestamps[i] }));
      const fused = output.frames.map((frame, i) => ({ ...frame.fused, t: timestamps[i] }));
      return await compareFrames(original, fused, result, signal, progress);
    } finally { worker.terminate(); }
  }
  async function runComparison(source: V8PilotComparison['source'], progress: ProgressListener | undefined, signal: AbortSignal | undefined,
    work: (result: V8PilotComparison, signal: AbortSignal) => Promise<V8PilotComparison>) {
    return operation(async inner => {
      const start = performance.now(), timings = emptyTimings();
      progress?.({ stage: 'loading', message: 'Έλεγχος παγωμένων αρχείων V4 και V8…' });
      await prepare(inner); timings.modelLoadMs = performance.now() - start;
      const result = await work(initial(source, timings), inner); result.timings.totalMs = performance.now() - start; return result;
    }, signal);
  }
  return {
    analyzeVideo: (video: HTMLVideoElement, progress?: ProgressListener, signal?: AbortSignal) => runComparison('video', progress, signal, async (result, inner) => {
      if (!Number.isFinite(video.duration) || video.duration <= 0 || video.duration > 15) throw new Error('Χρειάζεται αναγνώσιμο βίντεο έως 15 δευτερολέπτων.');
      video.pause(); const paths: string[] = [], timestamps: number[] = [];
      const started = performance.now(), sourceCount = Math.max(2, Math.round(video.duration * 30)), count = Math.min(48, sourceCount);
      const canvas = new OffscreenCanvas(video.videoWidth, video.videoHeight);
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Δεν υποστηρίζεται ανάγνωση καρέ.');
      try {
        for (let i = 0; i < count; i++) {
          inner.throwIfAborted();
          const index = Math.round(i * (sourceCount - 1) / (count - 1));
          const t = Math.min(index / 30, Math.max(0, video.duration - .001));
          await seekDecoded(video, t, inner);
          context.drawImage(video, 0, 0);
          paths.push(URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }))); timestamps.push(t * 1000);
          progress?.({ stage: 'decoding', message: `Ανάγνωση καρέ ${i + 1}/${count}`, fraction: (i + 1) / count, frames: i + 1 });
        }
        result.timings.decodeMs = performance.now() - started;
        return await recover(paths, timestamps, result, inner, progress);
      } finally { paths.forEach(url => URL.revokeObjectURL(url)); }
    }),
    dispose: () => {
      disposed = true; active?.abort(abortError()); classifier?.terminate(); classifier = undefined;
      objectUrls.forEach(url => URL.revokeObjectURL(url)); objectUrls.clear(); assets = undefined;
    },
  };
}
