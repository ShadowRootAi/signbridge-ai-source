import { FEATURE_COUNT, frameFeatures } from './features';
import { cancelPipeline, collectContinuousVideoLandmarks } from './pipeline';
import type { ContinuousRange } from './pipeline';
import type { ContinuousSegment, SequenceTimeline } from './continuous-inference';
import type { ProgressListener } from './pipeline-contract';

export { CONTINUOUS_MAX_DURATION_SECONDS } from './pipeline';
export type { ContinuousRange } from './pipeline';
export type { ContinuousSegment } from './continuous-inference';
export type { PipelineProgress } from './pipeline-contract';

type Prediction = {
  glosses: string[];
  /** Exact candidate match to an annotated training reference, if present. */
  text: string | null;
  confidence: number;
  segments: ContinuousSegment[];
  inference_ms: number;
};

export type ContinuousVideoResult = Prediction & {
  quality: number;
  frames: number;
  observedFrames: number;
  durationMs: number;
  startMs: number;
  elapsedMs: number;
};

let worker: Worker | undefined;
let active: AbortController | undefined;

function predict(features: number[][], timeline: SequenceTimeline, signal: AbortSignal) {
  signal.throwIfAborted();
  const instance = worker ??= new Worker(new URL('./continuous.worker.ts', import.meta.url), { type: 'module' });
  return new Promise<Prediction>((resolve, reject) => {
    const timeout = setTimeout(() => fail(new Error('Η αναγνώριση ακολουθίας καθυστέρησε.')), 60_000);
    const abort = () => fail(signal.reason);
    function clean() {
      clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
      instance.onmessage = instance.onerror = instance.onmessageerror = null;
    }
    function fail(error: unknown) {
      clean();
      instance.terminate();
      if (worker === instance) worker = undefined;
      reject(error);
    }
    signal.addEventListener('abort', abort, { once: true });
    instance.onmessage = ({ data }) => {
      if (data.error) { fail(new Error(data.error)); return; }
      clean();
      resolve(data.result);
    };
    instance.onerror = event => fail(new Error(event.message || 'Αποτυχία αναγνώρισης ακολουθίας.'));
    instance.onmessageerror = () => fail(new Error('Μη έγκυρη απάντηση αναγνώρισης ακολουθίας.'));
    try { instance.postMessage({ features, timeline }); } catch (error) { fail(error); }
  });
}

export function cancelContinuous() {
  active?.abort();
  worker?.terminate();
  worker = undefined;
}

export async function analyzeContinuousVideo(video: HTMLVideoElement, signal?: AbortSignal,
  progress?: ProgressListener, range?: ContinuousRange): Promise<ContinuousVideoResult> {
  signal?.throwIfAborted();
  if (active) throw new Error('Υπάρχει ήδη ανάλυση σε εξέλιξη.');
  const controller = new AbortController();
  const abort = () => controller.abort(signal?.reason);
  signal?.addEventListener('abort', abort, { once: true });
  active = controller;
  const start = performance.now();
  try {
    const { frames, startMs, durationMs } = await collectContinuousVideoLandmarks(video, progress, controller.signal, range);
    const rows = frames.map(frame => frame ? frameFeatures(frame) : null);
    const tracked = rows.map(row => Boolean(row?.quality));
    const timestampsMs = frames.map(frame => frame ? frame.t - startMs : null);
    const features = rows.map(row => row ? row.values.map(Math.fround) : new Array<number>(FEATURE_COUNT).fill(0));
    progress?.({ stage: 'inference', message: 'Αναγνώριση ακολουθίας…', fraction: 1, frames: frames.length });
    const result = await predict(features, { timestampsMs, tracked, durationMs }, controller.signal);
    controller.signal.throwIfAborted();
    const segments = result.segments.map(segment => ({
      ...segment, startMs: segment.startMs + startMs, endMs: segment.endMs + startMs,
    }));
    return { ...result, segments, quality: tracked.filter(Boolean).length / frames.length, frames: frames.length,
      observedFrames: timestampsMs.filter(time => time !== null).length, startMs, durationMs, elapsedMs: performance.now() - start };
  } finally {
    signal?.removeEventListener('abort', abort);
    cancelPipeline();
    active = undefined;
  }
}
