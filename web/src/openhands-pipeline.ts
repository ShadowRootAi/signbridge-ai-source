import { cancelPipeline, collectCameraLandmarks, collectVideoLandmarks } from './pipeline';
import type { LandmarkFrame } from './features';
import type { ProgressListener } from './pipeline-contract';
import type { OpenHandsRecognition, OpenHandsPolicy } from './openhands-recognizer';

export type OpenHandsResult = OpenHandsRecognition & { frames: number; elapsedMs: number };
export const OPENHANDS_CAPTURE_MS = 1600;
let worker: Worker | undefined;
let requestId = 0;
let policyPromise: Promise<OpenHandsPolicy> | undefined;
let rejectActive: ((reason: unknown) => void) | undefined;

export function getOpenHandsPolicy(): Promise<OpenHandsPolicy> {
  policyPromise ??= fetch('/__signbridge/openhands/policy.json', { cache: 'no-store' }).then(async response => {
    if (!response.ok) throw new Error('Η τοπική ρύθμιση OpenHands δεν είναι διαθέσιμη.');
    return await response.json() as OpenHandsPolicy;
  }).catch(error => { policyPromise = undefined; throw error; });
  return policyPromise;
}

function classify(frames: LandmarkFrame[], allowRecovery: boolean, signal?: AbortSignal): Promise<OpenHandsRecognition> {
  signal?.throwIfAborted();
  if (rejectActive) throw new Error('Υπάρχει ήδη ανάλυση σε εξέλιξη.');
  const instance = worker ??= new Worker(new URL('./openhands-classifier.worker.ts', import.meta.url), { type: 'module' });
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => fail(new Error('Η πρόβλεψη OpenHands καθυστέρησε.')), 60_000);
    const clean = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      instance.onmessage = instance.onerror = instance.onmessageerror = null;
      rejectActive = undefined;
    };
    const fail = (error: unknown) => {
      clean();
      instance.terminate();
      if (worker === instance) worker = undefined;
      reject(error);
    };
    const abort = () => fail(signal?.reason ?? new DOMException('Cancelled', 'AbortError'));
    rejectActive = fail;
    signal?.addEventListener('abort', abort, { once: true });
    instance.onmessage = ({ data }) => {
      if (data.id !== id) return;
      if (data.error) { fail(new Error(data.error)); return; }
      clean();
      resolve(data.result as OpenHandsRecognition);
    };
    instance.onerror = event => fail(new Error(event.message || 'Αποτυχία του OpenHands.'));
    instance.onmessageerror = () => fail(new Error('Μη έγκυρη απάντηση OpenHands.'));
    try { instance.postMessage({ id, frames, allowRecovery }); } catch (error) { fail(error); }
  });
}

export function cancelOpenHands() {
  cancelPipeline();
  rejectActive?.(new DOMException('Cancelled', 'AbortError'));
  worker?.terminate();
  worker = undefined;
}

async function decide(frames: LandmarkFrame[], start: number, allowRecovery: boolean, signal?: AbortSignal, progress?: ProgressListener): Promise<OpenHandsResult> {
  progress?.({ stage: 'inference', message: 'Αναγνώριση νοήματος…', fraction: 1, frames: frames.length });
  const result = await classify(frames, allowRecovery, signal);
  signal?.throwIfAborted();
  return { ...result, frames: frames.length, elapsedMs: performance.now() - start };
}

export async function analyzeOpenHandsVideo(video: HTMLVideoElement, signal?: AbortSignal,
  progress?: ProgressListener, allowRecovery = true, onLandmarks?: (frames: LandmarkFrame[]) => void) {
  const { frames, start } = await collectVideoLandmarks(video, progress, signal,
    (allowRecovery || Boolean(onLandmarks)) && import.meta.env.VITE_SIGNBRIDGE_OPENHANDS_FRAME_OFFSET === '1');
  onLandmarks?.(frames);
  return decide(frames, start, allowRecovery, signal, progress);
}

export async function analyzeOpenHandsCamera(video: HTMLVideoElement, durationMs: number, signal?: AbortSignal, progress?: ProgressListener) {
  const { frames, start } = await collectCameraLandmarks(video, durationMs, progress, signal);
  return decide(frames, start, false, signal, progress);
}
