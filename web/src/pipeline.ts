import { preprocess } from './features';
import type { LandmarkFrame } from './features';
import type { ModelStatus, ProgressListener, RecognitionResult } from './pipeline-contract';
import type { ClipObservationMetadata, OpenSetGateResult, QualityMotionDiagnostics } from './openset';

export interface ModelSource { configUrl: string; modelUrl: string }
let modelSource: ModelSource = { configUrl: '/models/model_config.json', modelUrl: '/models/model.onnx' };

let worker: Worker | undefined;
let requestId = 0;
const REQUEST_TIMEOUT_MS = 120_000;
const pending = new Map<number, { owner: Worker; resolve: (value: unknown) => void; reject: (error: unknown) => void }>();

function failWorker(instance: Worker, error: Error) {
  for (const task of [...pending.values()]) if (task.owner === instance) task.reject(error);
  instance.terminate();
  if (worker === instance) worker = undefined;
}

function request<T>(kind: string, data: Record<string, unknown> = {}, transfer: Transferable[] = [], signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) return Promise.reject(signal.reason);
  if (!worker) {
    const instance = new Worker(new URL('./recognizer.worker.ts', import.meta.url), { type: 'module' });
    worker = instance;
    instance.onmessage = ({ data }) => {
      const task = pending.get(data.id);
      if (!task) return;
      if (data.error) task.reject(new Error(data.error)); else task.resolve(data.value);
    };
    instance.onerror = event => failWorker(instance, new Error(event.message || 'Αποτυχία μηχανής αναγνώρισης.'));
    instance.onmessageerror = () => failWorker(instance, new Error('Αποτυχία επικοινωνίας με τη μηχανή αναγνώρισης.'));
  }
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    const instance = worker!;
    const cleanup = () => {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', onAbort);
      pending.delete(id);
    };
    const onAbort = () => {
      cleanup();
      // In-flight synchronous WASM may finish in the worker. Its late result is
      // ignored, and the worker's frame finally block closes transferred data.
      reject(signal?.reason ?? new DOMException('Cancelled', 'AbortError'));
    };
    const timeout = setTimeout(() => failWorker(instance, new Error('Η επεξεργασία ξεπέρασε τα 120 δευτερόλεπτα. Δοκιμάστε ξανά.')), REQUEST_TIMEOUT_MS);
    pending.set(id, {
      owner: instance,
      resolve: value => { cleanup(); resolve(value as T); },
      reject: error => { cleanup(); reject(error); },
    });
    signal?.addEventListener('abort', onAbort, { once: true });
    try { instance.postMessage({ id, kind, ...data, ...modelSource }, transfer); }
    catch (error) { pending.get(id)?.reject(error); }
  });
}

function aborted(signal?: AbortSignal) { signal?.throwIfAborted(); }
function nextFrame(signal?: AbortSignal) {
  aborted(signal);
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => finish(signal?.reason ?? new DOMException('Cancelled', 'AbortError'));
    const frame = requestAnimationFrame(() => finish());
    const timeout = setTimeout(() => finish(new Error('Ο browser δεν παρείχε νέο καρέ εγκαίρως.')), REQUEST_TIMEOUT_MS);
    function finish(error?: unknown) {
      cancelAnimationFrame(frame);
      clearTimeout(timeout);
      signal?.removeEventListener('abort', onAbort);
      if (error) reject(error); else resolve();
    }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
export async function getModelStatus(): Promise<ModelStatus> {
  try { return await request<ModelStatus>('status'); }
  catch (error) { return { ready: false, message: error instanceof Error ? error.message : String(error) }; }
}
/** Route/unmount cleanup: abort outstanding worker work without leaving a late result. */
export function cancelPipeline() {
  preparedClipToken++;
  if (worker) failWorker(worker, new DOMException('Cancelled', 'AbortError'));
}
/** Scientific QA can select real local artifacts without replacing UI assets. */
export function setModelSource(source?: ModelSource) {
  const next = source ?? { configUrl: '/models/model_config.json', modelUrl: '/models/model.onnx' };
  const validate = (value: string) => {
    const url = new URL(value, window.location.origin);
    if (url.origin !== window.location.origin || url.username || url.password) throw new Error('Only same-origin model artifacts are allowed.');
    return url.href;
  };
  modelSource = { configUrl: validate(next.configUrl), modelUrl: validate(next.modelUrl) };
  if (worker) failWorker(worker, new Error('Άλλαξε το τοπικό μοντέλο δοκιμής.'));
}
export async function warmupLandmarks(signal?: AbortSignal) { return request('warmup', {}, [], signal); }
let preparedClipToken = 0;
export async function beginClip(signal?: AbortSignal): Promise<number> {
  aborted(signal);
  // MediaPipe's ESM factory cannot reliably initialize a second graph after
  // close() in the same realm. A fresh worker isolates graph and module state.
  if (worker) failWorker(worker, new Error('Ξεκίνησε νέα ανεξάρτητη ανάλυση νοήματος.'));
  await request('resetDetector', {}, [], signal);
  aborted(signal);
  return ++preparedClipToken;
}
const videoCanvases = new WeakMap<HTMLVideoElement, OffscreenCanvas>();
async function snapshot(source: ImageBitmapSource): Promise<ImageBitmap> {
  if (!(source instanceof HTMLVideoElement)) return createImageBitmap(source);
  if (source.readyState < 2 || source.videoWidth <= 0 || source.videoHeight <= 0) {
    throw new Error('Το τρέχον καρέ βίντεο δεν είναι έτοιμο για ανάλυση.');
  }
  // Chromium can reject createImageBitmap(video) for a paused, decoded H.264
  // frame even at HAVE_ENOUGH_DATA. Canvas reads the actual decoded pixels and
  // avoids that direct-video snapshot path without changing the frame content.
  let canvas = videoCanvases.get(source);
  if (!canvas) {
    canvas = new OffscreenCanvas(source.videoWidth, source.videoHeight);
    videoCanvases.set(source, canvas);
  }
  if (canvas.width !== source.videoWidth) canvas.width = source.videoWidth;
  if (canvas.height !== source.videoHeight) canvas.height = source.videoHeight;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Δεν είναι διαθέσιμη η ανάγνωση καρέ από τον browser.');
  context.drawImage(source, 0, 0);
  return canvas.transferToImageBitmap();
}
async function detectBitmap(bitmap: ImageBitmap, t: number, signal?: AbortSignal): Promise<LandmarkFrame> {
  try {
    aborted(signal);
    return await request<LandmarkFrame>('frame', { bitmap, t }, [bitmap], signal);
  } finally {
    // close() is safe after transfer; otherwise it releases the local bitmap.
    bitmap.close();
  }
}
async function detect(source: ImageBitmapSource, t: number, signal?: AbortSignal): Promise<LandmarkFrame> {
  aborted(signal);
  const bitmap = await snapshot(source);
  return detectBitmap(bitmap, t, signal);
}
export async function classifyFeatures(x: number[][], quality: number, signal?: AbortSignal, observationMetadata?: ClipObservationMetadata) {
  return request<Omit<RecognitionResult, 'frames' | 'elapsedMs'> & OpenSetGateResult & {
    logits: number[]; embedding?: number[]; diagnostics: QualityMotionDiagnostics; observationMetadata: ClipObservationMetadata;
  }>('infer', { x, quality, observationMetadata }, [], signal);
}
/** Product decision path: worker measures original raw frames and enforces the completeness guard. */
export async function classifyLandmarkFrames(frames: LandmarkFrame[], signal?: AbortSignal) {
  return request<Omit<RecognitionResult, 'frames' | 'elapsedMs'>>('inferClip', { frames }, [], signal);
}
async function finish(frames: LandmarkFrame[], start: number, signal?: AbortSignal): Promise<RecognitionResult> {
  aborted(signal);
  const result = await classifyLandmarkFrames(frames, signal);
  aborted(signal);
  return { ...result, frames: frames.length, elapsedMs: performance.now() - start };
}

async function seek(video: HTMLVideoElement, seconds: number, signal?: AbortSignal) {
  aborted(signal);
  // Always seek, including time zero: HAVE_ENOUGH_DATA alone allowed an empty
  // initial canvas snapshot in Chromium before the first presentation/seek.
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => done(new Error('Η αποκωδικοποίηση του βίντεο καθυστέρησε.')), 15000);
    const onSeek = () => done();
    const onError = () => done(new Error('Το βίντεο δεν μπορεί να αποκωδικοποιηθεί.'));
    const onAbort = () => done(new DOMException('Cancelled', 'AbortError'));
    function done(error?: Error) {
      clearTimeout(timeout);
      video.removeEventListener('seeked', onSeek);
      video.removeEventListener('error', onError);
      signal?.removeEventListener('abort', onAbort);
      if (error) reject(error); else resolve();
    }
    video.addEventListener('seeked', onSeek, { once: true });
    video.addEventListener('error', onError, { once: true });
    signal?.addEventListener('abort', onAbort, { once: true });
    video.currentTime = seconds;
  });
}

export async function collectVideoLandmarks(video: HTMLVideoElement, progress?: ProgressListener, signal?: AbortSignal,
  seekAfterFrameBoundary = false) {
  const start = performance.now();
  if (!Number.isFinite(video.duration) || video.duration <= 0 || video.duration > 15) {
    throw new Error('Χρειάζεται αναγνώσιμο βίντεο διάρκειας έως 15 δευτερολέπτων.');
  }
  progress?.({ stage: 'loading', message: 'Φόρτωση ανιχνευτή landmarks…' });
  await beginClip(signal);
  aborted(signal);
  video.pause();
  // Source clips are encoded at 30 FPS. Match offline endpoint sampling rather
  // than silently halving the number of source frames used during training.
  const sourceFrameCount = Math.max(2, Math.round(video.duration * 30));
  const count = Math.min(48, sourceFrameCount);
  const frames: LandmarkFrame[] = [];
  for (let i = 0; i < count; i++) {
    aborted(signal);
    const sourceIndex = Math.round(i * (sourceFrameCount - 1) / (count - 1));
    const t = Math.min(sourceIndex / 30, Math.max(0, video.duration - 0.001));
    // A rounded seek at exactly a 30 Hz boundary can select the preceding
    // decoded frame. Keep feature timestamps on the model's original grid.
    const target = seekAfterFrameBoundary ? Math.min(t + .002, video.duration - .000001) : t;
    await seek(video, target, signal);
    frames.push(await detect(video, t * 1000, signal));
    progress?.({ stage: 'tracking', message: `Ανάλυση καρέ ${i + 1}/${count}`, fraction: (i + 1) / count, frames: i + 1 });
    await nextFrame(signal);
  }
  aborted(signal);
  return { frames, start };
}

export const CONTINUOUS_MAX_DURATION_SECONDS = 12;
export const CONTINUOUS_SAMPLE_RATE = 15;
export interface ContinuousRange { startMs: number; endMs: number }

/** Read presentation time from the decoded frame, never from the seek target. */
function seekPresentation(video: HTMLVideoElement, target: number, signal?: AbortSignal) {
  aborted(signal);
  return new Promise<number | null>((resolve, reject) => {
    let callback: number;
    let presentationTimer: ReturnType<typeof setTimeout> | undefined;
    let seeked = false;
    let mediaTime: number | undefined;
    const timeout = setTimeout(() => fail(new Error('Η αποκωδικοποίηση του βίντεο καθυστέρησε.')), 15_000);
    const onAbort = () => fail(signal?.reason ?? new DOMException('Cancelled', 'AbortError'));
    const onError = () => fail(new Error('Το βίντεο δεν μπορεί να αποκωδικοποιηθεί.'));
    function clean() {
      clearTimeout(timeout);
      clearTimeout(presentationTimer);
      video.cancelVideoFrameCallback(callback);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      signal?.removeEventListener('abort', onAbort);
    }
    function finish(time: number | null) { clean(); resolve(time); }
    function fail(error: unknown) { clean(); reject(error); }
    function onSeeked() {
      seeked = true;
      if (mediaTime !== undefined) finish(mediaTime);
      else presentationTimer = setTimeout(() => finish(null), 300);
    }
    function onPresentation(_now: number, metadata: VideoFrameCallbackMetadata) {
      if (Math.abs(video.currentTime - target) > 0.006 || !Number.isFinite(metadata.mediaTime)
        || metadata.mediaTime < 0 || metadata.mediaTime > target + 0.006) {
        callback = video.requestVideoFrameCallback(onPresentation);
        return;
      }
      mediaTime = metadata.mediaTime;
      if (seeked) finish(mediaTime);
    }
    callback = video.requestVideoFrameCallback(onPresentation);
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onError, { once: true });
    signal?.addEventListener('abort', onAbort, { once: true });
    try { video.currentTime = target; } catch (error) { fail(error); }
  });
}

/** Keep the time grid used in CTC training and retain absent or duplicate samples. */
export async function collectContinuousVideoLandmarks(video: HTMLVideoElement, progress?: ProgressListener,
  signal?: AbortSignal, range?: ContinuousRange) {
  const sourceMs = video.duration * 1000;
  const startMs = range?.startMs ?? 0;
  const endMs = Math.min(range?.endMs ?? sourceMs, sourceMs);
  const durationMs = endMs - startMs;
  if (!Number.isFinite(sourceMs) || !Number.isFinite(startMs) || !Number.isFinite(endMs)
    || (range && (!Number.isFinite(range.endMs) || range.endMs > sourceMs + 1)) || startMs < 0
    || durationMs <= 0 || durationMs > CONTINUOUS_MAX_DURATION_SECONDS * 1000 + .0001) {
    throw new Error(`Επίλεξε απόσπασμα μέσα στο βίντεο διάρκειας έως ${CONTINUOUS_MAX_DURATION_SECONDS} δευτερολέπτων.`);
  }
  if (typeof video.requestVideoFrameCallback !== 'function') {
    throw new Error('Ο browser δεν παρέχει τους χρόνους των αποκωδικοποιημένων καρέ. Δοκίμασε ενημερωμένο browser.');
  }
  video.pause();
  progress?.({ stage: 'loading', message: 'Φόρτωση ανιχνευτή landmarks…' });
  await beginClip(signal);
  // Media containers round durations to their timestamp precision. Do not add
  // a sample at the end when a rounded duration is exactly on the training grid.
  const count = Math.max(1, Math.ceil(durationMs / 1000 * CONTINUOUS_SAMPLE_RATE - 0.0001));
  const frames: (LandmarkFrame | null)[] = [];
  let previous = -1;
  for (let index = 0; index < count; index++) {
    // Seek just past the grid point to avoid rounded container timestamps
    // selecting the preceding source frame at a 30 FPS boundary.
    const target = Math.min(startMs / 1000 + index / CONTINUOUS_SAMPLE_RATE + 0.002, endMs / 1000 - 0.000001);
    const time = await seekPresentation(video, Math.max(0, target), signal);
    const inRange = time !== null && time > previous && time * 1000 >= startMs - .01 && time * 1000 < endMs;
    const frame = inRange ? await detect(video, Math.max(startMs, time * 1000), signal) : null;
    if (time !== null && frame) previous = time;
    frames.push(frame);
    progress?.({ stage: 'tracking', message: `Ανάλυση καρέ ${index + 1}/${count}`,
      fraction: (index + 1) / count, frames: index + 1 });
  }
  aborted(signal);
  return { frames, startMs, durationMs };
}

export async function analyzeVideo(video: HTMLVideoElement, progress?: ProgressListener, signal?: AbortSignal): Promise<RecognitionResult> {
  const { frames, start } = await collectVideoLandmarks(video, progress, signal);
  progress?.({ stage: 'inference', message: 'Εκτέλεση εκπαιδευμένου μοντέλου…', fraction: 1, frames: frames.length });
  return finish(frames, start, signal);
}

export async function collectCameraLandmarks(video: HTMLVideoElement, durationMs: number, progress?: ProgressListener, signal?: AbortSignal, clipToken?: number) {
  if (video.readyState < 2 || !video.srcObject) throw new Error('Η κάμερα δεν είναι έτοιμη.');
  if (!Number.isFinite(durationMs) || durationMs < 100 || durationMs > 15_000) throw new Error('Μη έγκυρη διάρκεια καταγραφής.');
  progress?.({ stage: 'loading', message: 'Προετοιμασία ανιχνευτή…' });
  // The UI prepares a fresh clip before its countdown. Direct callers get the
  // same isolation here without having to manage a preparation token.
  if (clipToken === undefined || clipToken !== preparedClipToken) await beginClip(signal);
  aborted(signal);
  const start = performance.now();
  const buffer: { bitmap: ImageBitmap; t: number }[] = [];
  // Aim for 30 FPS; longer captures spread the bounded 48-frame budget across
  // the full interval. No landmark inference occurs during capture.
  const targetCount = Math.min(48, Math.max(1, Math.round(durationMs * 30 / 1000)));
  const intervalMs = durationMs / targetCount;
  let nextSampleMs = 0;
  let previous = -1;
  try {
    progress?.({ stage: 'capture', message: 'Νοημάτισε τώρα — καταγραφή κίνησης', fraction: 0, frames: 0 });
    while (performance.now() - start < durationMs) {
      aborted(signal);
      const elapsed = performance.now() - start;
      if (buffer.length < targetCount && elapsed >= nextSampleMs && video.readyState >= 2 && video.currentTime !== previous) {
        previous = video.currentTime;
        const bitmap = await snapshot(video);
        if (signal?.aborted) { bitmap.close(); aborted(signal); }
        buffer.push({ bitmap, t: elapsed });
        // Skip missed slots rather than generating a burst of duplicate frames.
        nextSampleMs = (Math.floor(elapsed / intervalMs) + 1) * intervalMs;
        progress?.({ stage: 'capture', message: 'Νοημάτισε τώρα — καταγραφή κίνησης', fraction: Math.min(elapsed / durationMs, 1), frames: buffer.length });
      }
      await nextFrame(signal);
    }
    aborted(signal);
    if (buffer.length < 4) throw new Error('Καταγράφηκαν πολύ λίγα καρέ. Δοκίμασε ξανά.');
    const frames: LandmarkFrame[] = [];
    for (let i = 0; i < buffer.length; i++) {
      aborted(signal);
      frames.push(await detectBitmap(buffer[i].bitmap, buffer[i].t, signal));
      progress?.({ stage: 'tracking', message: `Η καταγραφή τελείωσε — ανάλυση καρέ ${i + 1}/${buffer.length}`, fraction: (i + 1) / buffer.length, frames: i + 1 });
    }
    return { frames, start };
  } finally {
    // Also covers cancellation while a single transferred frame finishes in
    // the worker. Remaining local buffers are always released immediately.
    for (const item of buffer) item.bitmap.close();
  }
}

export async function analyzeCamera(video: HTMLVideoElement, durationMs: number, progress?: ProgressListener, signal?: AbortSignal, clipToken?: number): Promise<RecognitionResult> {
  const { frames, start } = await collectCameraLandmarks(video, durationMs, progress, signal, clipToken);
  return finish(frames, start, signal);
}

/** Offline scientific extraction uses this exact detector and feature implementation. */
export async function extractImagePaths(paths: string[], fps = 30, includeRaw = false) {
  const start = performance.now();
  await beginClip();
  const count = Math.min(paths.length, 48);
  if (count === 0) throw new Error('Empty source clip');
  const indices = Array.from({ length: count }, (_, i) => count === 1 ? 0 : Math.round(i * (paths.length - 1) / (count - 1)));
  const frames: LandmarkFrame[] = [];
  for (const index of indices) {
    const response = await fetch(paths[index]);
    if (!response.ok) throw new Error(`Frame unavailable: ${response.status}`);
    const image = await createImageBitmap(await response.blob());
    frames.push(await request<LandmarkFrame>('frame', { bitmap: image, t: index / fps * 1000 }, [image]));
  }
  const features = preprocess(frames);
  return { ...features, frames: frames.length, observedSpanMs: frames[frames.length - 1].t - frames[0].t, elapsedMs: performance.now() - start,
    ...(includeRaw ? { raw: frames } : {}) };
}
