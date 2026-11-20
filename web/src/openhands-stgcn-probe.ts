/** Local research harness. The public SignBridge app does not import this module. */
import * as ort from 'onnxruntime-web/wasm';
import { extractImagePaths } from './pipeline';
import type { LandmarkFrame, Point } from './features';

const selected = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74];

function coordinates(frame: LandmarkFrame) {
  const part = (points: Point[], count: number) => {
    if (points.length < count || points.slice(0, count).some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
      return Array.from({ length: count }, () => [0, 0]);
    }
    return points.slice(0, count).map(p => [p.x, p.y]);
  };
  return [...part(frame.pose, 33), ...part(frame.left, 21), ...part(frame.right, 21)];
}

export async function loadOpenHandsProbe(modelUrl: string, configUrl: string) {
  const response = await fetch(configUrl);
  if (!response.ok) throw new Error(`OpenHands config HTTP ${response.status}`);
  const { labels }: { labels: string[] } = await response.json();
  if (!Array.isArray(labels) || labels.length !== 310 || new Set(labels).size !== 310) throw new Error('Expected OpenHands GSL class order');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = '/vendor/onnxruntime/';
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  return async (paths: string[], fps: number) => {
    if (!paths.length || paths.length > 48) throw new Error('This smoke harness requires every original frame, at most 48');
    const extraction = await extractImagePaths(paths, fps, true);
    if (!extraction.raw || extraction.raw.length !== paths.length) throw new Error('Missing original raw landmarks');
    const points = extraction.raw.map(coordinates);
    const center = [0, 0];
    let width = 0;
    for (const frame of points) {
      const a = frame[11], b = frame[12];
      for (let c = 0; c < 2; c++) center[c] += (a[c] + b[c]) / 2;
      width += Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
    width /= points.length;
    center.forEach((value, c) => { center[c] = value / points.length; });
    if (!Number.isFinite(width) || width <= 0) return { top5: [], reason: 'no_shoulders', extraction_ms: extraction.elapsedMs };
    const flat = Float32Array.from(center.flatMap((origin, c) =>
      points.flatMap(frame => selected.map(i => (frame[i][c] - origin) / width))));
    const start = performance.now();
    const result = await session.run({ landmarks: new ort.Tensor('float32', flat, [1, 2, points.length, selected.length]) });
    const logits = Array.from(result.logits.data as Float32Array);
    if (logits.length !== 310 || logits.some(value => !Number.isFinite(value))) throw new Error('Invalid OpenHands result');
    const max = Math.max(...logits);
    const normalizer = logits.reduce((sum, x) => sum + Math.exp(x - max), 0);
    const top5 = logits.map((x, i) => ({ gloss: labels[i], score: Math.exp(x - max) / normalizer }))
      .sort((a, b) => b.score - a.score).slice(0, 5);
    return { top5, tracking_quality: extraction.quality, frames: points.length,
      extraction_ms: extraction.elapsedMs, onnx_ms: performance.now() - start };
  };
}
