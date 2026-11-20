import { FEATURE_SCHEMA, frameFeatures } from './features';
import type { LandmarkFrame } from './features';

/** Offline harness: keep the physical time axis for the sequence recognizer. */
export async function extractContinuousFrames(paths: string[], sourceFps: number, stride: number) {
  if (!paths.length || !Number.isFinite(sourceFps) || sourceFps <= 0 || !Number.isInteger(stride) || stride < 1) {
    throw new Error('Invalid continuous RGB input');
  }
  const start = performance.now();
  const worker = new Worker(new URL('./recognizer.worker.ts', import.meta.url), { type: 'module' });
  function detect(bitmap: ImageBitmap, index: number) {
    return new Promise<LandmarkFrame>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error(`Detector timed out at frame ${index}`)), 30_000);
      worker.onmessage = ({ data }) => {
        clearTimeout(timeout);
        if (data.id !== index || data.error) reject(new Error(data.error ?? 'Unexpected detector response'));
        else resolve(data.value);
      };
      worker.onerror = (event) => { clearTimeout(timeout); reject(new Error(event.message || 'Detector worker failed')); };
      try { worker.postMessage({ id: index, kind: 'frame', bitmap, t: index * 1000 / sourceFps }, [bitmap]); }
      catch (error) { clearTimeout(timeout); bitmap.close(); reject(error); }
    });
  }

  const features: number[][] = [];
  let tracked = 0;
  try {
    for (let i = 0; i < paths.length; i += stride) {
      const response = await fetch(paths[i]);
      if (!response.ok) throw new Error(`Source RGB frame ${i}: HTTP ${response.status}`);
      const row = frameFeatures(await detect(await createImageBitmap(await response.blob()), i));
      features.push(row.values.map(Math.fround));
      tracked += row.quality;
    }
  } finally { worker.terminate(); }
  return {
    features, fps: sourceFps / stride, quality: tracked / features.length,
    elapsed_ms: performance.now() - start, feature_schema: FEATURE_SCHEMA,
    source_frames: paths.length, stride,
  };
}
