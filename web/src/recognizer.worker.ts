/// <reference lib="webworker" />
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/wasm';
import { FEATURE_COUNT, FEATURE_SCHEMA, TIMESTEPS, preprocess } from './features';
import type { LandmarkFrame } from './features';
import { measureQualityMotion, scoreOpenSet } from './openset';
import type { ClipObservationMetadata, OpenSetGateConfig } from './openset';
import { measureRawTracking, trackingRetryMessage } from './tracking';

let detectorPromise: Promise<HolisticLandmarker> | undefined;
let modelPromise: Promise<{ session: ort.InferenceSession; config: ModelConfig }> | undefined;
let modelSourceKey = '';
interface ModelConfig extends OpenSetGateConfig {
  modelVersion: string; featureSchema: string; T: number; F: number;
  labels: string[]; backgroundIndex: number;
  input: { name: string }; output: { name: string; embeddingName?: string };
  rejection: OpenSetGateConfig['rejection'] & { rejectBackground: boolean; calibrated?: boolean };
}

function detector() {
  detectorPromise ??= (async () => HolisticLandmarker.createFromOptions(
    await FilesetResolver.forVisionTasks('/vendor/mediapipe', true), {
      baseOptions: { modelAssetPath: '/models/holistic_landmarker.task', delegate: 'CPU' },
      // IMAGE alone did not make repeated sequences deterministic in testing.
      // resetDetector closes/recreates the graph at each clip boundary.
      runningMode: 'IMAGE', outputPoseSegmentationMasks: false, outputFaceBlendshapes: false,
      minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
      minHandLandmarksConfidence: 0.5,
    }))().catch(error => { detectorPromise = undefined; throw error; });
  return detectorPromise;
}

function classifier(source?: { configUrl?: string; modelUrl?: string }) {
  const asset = (value: string | undefined, fallback: string) => {
    const url = new URL(value ?? fallback, self.location.origin);
    if (url.origin !== self.location.origin || url.username || url.password) throw new Error('Only same-origin model artifacts are allowed.');
    return url.href;
  };
  const configUrl = asset(source?.configUrl, '/models/model_config.json');
  const modelUrl = asset(source?.modelUrl, '/models/model.onnx');
  const key = `${configUrl}\n${modelUrl}`;
  if (modelPromise && modelSourceKey !== key) throw new Error('Model source changed without an isolated worker.');
  modelSourceKey = key;
  modelPromise ??= (async () => {
    const response = await fetch(configUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error('Το εκπαιδευμένο μοντέλο δεν έχει εγκατασταθεί ακόμη.');
    let config: ModelConfig;
    try { config = await response.json(); }
    catch { throw new Error('Το εκπαιδευμένο μοντέλο δεν έχει εγκατασταθεί ακόμη.'); }
    if (config.featureSchema !== FEATURE_SCHEMA || config.T !== TIMESTEPS || config.F !== FEATURE_COUNT) {
      throw new Error('Ασυμβατότητα μοντέλου και προεπεξεργασίας.');
    }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = '/vendor/onnxruntime/';
    const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    return { session, config };
  })().catch(error => { modelPromise = undefined; throw error; });
  return modelPromise;
}

async function infer(x: number[][], quality: number, source: { configUrl?: string; modelUrl?: string }, observation?: ClipObservationMetadata) {
  const { session, config } = await classifier(source);
  if (x.length !== TIMESTEPS || x.some(row => row.length !== FEATURE_COUNT || row.some(v => !Number.isFinite(v)))) {
    throw new Error('Invalid model input shape or non-finite input');
  }
  const tensor = new ort.Tensor('float32', Float32Array.from(x.flat()), [1, TIMESTEPS, FEATURE_COUNT]);
  const output = await session.run({ [config.input.name]: tensor });
  const logits = Array.from(output[config.output.name].data as Float32Array);
  const embeddingName = config.output.embeddingName ?? config.prototypeGate?.embeddingName;
  const embedding = embeddingName ? Array.from(output[embeddingName].data as Float32Array) : undefined;
  const diagnostics = measureQualityMotion(x);
  const scored = scoreOpenSet(logits, quality, diagnostics.motionBodyRms, config, embedding);
  const reason = scored.reasons.quality
    ? 'Δεν αναγνωρίστηκε: ανεπαρκής ανίχνευση χεριών ή σώματος.'
    : scored.reasons.motion
      ? 'Δεν αναγνωρίστηκε: η καταγεγραμμένη κίνηση είναι μικρότερη από το όριο του πειραματικού μοντέλου.'
    : scored.reasons.prototypeDisagreement || scored.reasons.prototypeDistance
      ? 'Δεν αναγνωρίστηκε: δεν βρέθηκε επαρκώς συμβατό υποστηριζόμενο νόημα.'
    : scored.reasons.background || scored.reasons.confidence
      ? 'Δεν αναγνωρίστηκε: η κίνηση είναι άγνωστη ή η πρόβλεψη αβέβαιη.'
      : 'Πρόβλεψη πειραματικού μοντέλου. Χρειάζεται επιβεβαίωση από τον χρήστη.';
  const observationMetadata = {
    frameCount: Number.isSafeInteger(observation?.frameCount) && observation!.frameCount! >= 0 ? observation!.frameCount : null,
    observedSpanMs: typeof observation?.observedSpanMs === 'number' && Number.isFinite(observation.observedSpanMs) && observation.observedSpanMs >= 0 ? observation.observedSpanMs : null,
  };
  return { ...scored, label: scored.accepted ? config.labels[scored.predicted] : null, reason,
    classificationScore: scored.confidence, classifierExecuted: true,
    trackingQuality: quality, modelVersion: config.modelVersion, logits, embedding, diagnostics, observationMetadata };
}

async function inferClip(frames: LandmarkFrame[], source: { configUrl?: string; modelUrl?: string }) {
  if (!Array.isArray(frames)) throw new Error('Expected raw landmark frames');
  const tracking = measureRawTracking(frames);
  if (!tracking.usable) {
    return { accepted: false, label: null, confidence: null, classificationScore: null, classifierExecuted: false,
      reason: trackingRetryMessage(tracking), tracking, trackingQuality: tracking.requiredDataCompleteness,
      modelVersion: 'not-executed-incomplete-input' };
  }
  const features = preprocess(frames);
  const result = await infer(features.x, features.quality, source,
    { frameCount: frames.length, observedSpanMs: frames[frames.length - 1].t - frames[0].t });
  return { ...result, tracking, trackingQuality: tracking.requiredDataCompleteness };
}

async function handleMessage(event: MessageEvent) {
  const { id, kind } = event.data;
  try {
    let value: unknown;
    if (kind === 'frame') {
      const bitmap: ImageBitmap = event.data.bitmap;
      try {
        const model = await detector();
        const result = model.detect(bitmap);
        const frame: LandmarkFrame = {
          t: event.data.t, width: bitmap.width, height: bitmap.height,
          left: result.leftHandLandmarks[0] ?? [], right: result.rightHandLandmarks[0] ?? [],
          pose: result.poseLandmarks[0] ?? [], face: result.faceLandmarks[0] ?? [],
        };
        value = frame;
      } finally { bitmap.close(); }
    } else if (kind === 'preprocess') {
      value = preprocess(event.data.frames);
    } else if (kind === 'infer') {
      value = await infer(event.data.x, event.data.quality, event.data, event.data.observationMetadata);
    } else if (kind === 'inferClip') {
      value = await inferClip(event.data.frames, event.data);
    } else if (kind === 'resetDetector') {
      const previous = detectorPromise;
      if (previous) {
        try { (await previous).close(); }
        catch { /* A failed previous initialization has no live graph to retain. */ }
      }
      detectorPromise = undefined;
      await detector();
      value = { ready: true };
    } else if (kind === 'warmup') {
      await detector(); value = { ready: true };
    } else if (kind === 'status') {
      const { config } = await classifier(event.data);
      value = { ready: true, message: 'Το εκπαιδευμένο ONNX μοντέλο φορτώθηκε τοπικά.',
        modelVersion: config.modelVersion, labels: config.labels, rejectionCalibrated: config.rejection.calibrated === true };
    } else { throw new Error('Unknown worker request'); }
    self.postMessage({ id, value });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
}

// Serialize graph resets and frames, including when a caller cancels and starts
// another clip before an existing graph initialization has completed.
let visionOperations = Promise.resolve();
self.onmessage = (event: MessageEvent) => {
  if (['frame', 'warmup', 'resetDetector'].includes(event.data.kind)) {
    visionOperations = visionOperations.then(() => handleMessage(event), () => handleMessage(event));
  } else {
    void handleMessage(event);
  }
};
