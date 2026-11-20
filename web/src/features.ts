/** Versioned, deterministic feature contract. No learned statistics live here. */
export const FEATURE_SCHEMA = 'signbridge-landmarks-xy-v1';
export const TIMESTEPS = 48;
export const FEATURE_COUNT = 224;
export const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24];
export const FACE_INDICES = [1, 4, 33, 133, 362, 263, 61, 291, 13, 14, 70, 300, 105, 334, 159, 386, 152];

export interface Point { x: number; y: number; z?: number; visibility?: number }
export interface LandmarkFrame {
  t: number;
  width: number;
  height: number;
  left: Point[];
  right: Point[];
  pose: Point[];
  face: Point[];
}

const clip = (x: number) => Math.max(-8, Math.min(8, Number.isFinite(x) ? x : 0));
const valid = (points: Point[], n: number) => points.length >= n && points.slice(0, n).every(p => Number.isFinite(p.x) && Number.isFinite(p.y));

export function frameFeatures(frame: LandmarkFrame): { values: number[]; quality: number } {
  const aspect = frame.width / frame.height;
  const xy = (p: Point): [number, number] => [p.x * aspect, p.y];
  const distance = (a: Point, b: Point) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);
  const poseValid = valid(frame.pose, 25) && (frame.pose[11].visibility ?? 1) >= 0.5 && (frame.pose[12].visibility ?? 1) >= 0.5;
  const leftValid = valid(frame.left, 21);
  const rightValid = valid(frame.right, 21);
  const faceValid = valid(frame.face, 387);
  const origin: [number, number] = poseValid
    ? [(frame.pose[11].x + frame.pose[12].x) * aspect / 2, (frame.pose[11].y + frame.pose[12].y) / 2]
    : [0, 0];
  const bodyScale = poseValid ? Math.max(0.05, distance(frame.pose[11], frame.pose[12])) : 1;
  const values: number[] = [];
  const append = (point: Point | undefined, center: [number, number], scale: number) => {
    if (!point) values.push(0, 0);
    else { const p = xy(point); values.push(clip((p[0] - center[0]) / scale), clip((p[1] - center[1]) / scale)); }
  };
  for (const [hand, present] of [[frame.left, leftValid], [frame.right, rightValid]] as const) {
    for (let i = 0; i < 21; i++) append(poseValid && present ? hand[i] : undefined, origin, bodyScale);
    const wrist = present ? xy(hand[0]) : [0, 0] as [number, number];
    const palmScale = present ? Math.max(0.01, distance(hand[0], hand[9])) : 1;
    for (let i = 0; i < 21; i++) append(present ? hand[i] : undefined, wrist, palmScale);
  }
  for (const i of POSE_INDICES) append(poseValid ? frame.pose[i] : undefined, origin, bodyScale);
  const faceCenter = faceValid ? xy(frame.face[1]) : [0, 0] as [number, number];
  const faceScale = faceValid ? Math.max(0.01, distance(frame.face[33], frame.face[263])) : 1;
  for (const i of FACE_INDICES) append(faceValid ? frame.face[i] : undefined, faceCenter, faceScale);
  values.push(Number(leftValid), Number(rightValid), Number(poseValid), Number(faceValid));
  if (values.length !== FEATURE_COUNT) throw new Error('Feature schema length mismatch');
  return { values, quality: Number(poseValid && (leftValid || rightValid)) };
}

/** Linear time interpolation. Endpoint/missingness handling is identical offline/live. */
export function preprocess(frames: LandmarkFrame[]): { x: number[][]; quality: number } {
  if (!frames.length) throw new Error('Δεν υπάρχουν καρέ προς ανάλυση.');
  if (frames.some((f, i) => i > 0 && f.t <= frames[i - 1].t)) throw new Error('Non-monotonic frame timestamps');
  const features = frames.map(frameFeatures);
  const start = frames[0].t;
  const span = frames[frames.length - 1].t - start;
  const x: number[][] = [];
  let lo = 0;
  for (let i = 0; i < TIMESTEPS; i++) {
    const t = start + span * i / (TIMESTEPS - 1);
    while (lo + 1 < frames.length && frames[lo + 1].t < t) lo++;
    const hi = Math.min(lo + 1, frames.length - 1);
    const w = hi === lo ? 0 : (t - frames[lo].t) / (frames[hi].t - frames[lo].t);
    x.push(features[lo].values.map((v, j) => Math.fround(v + w * (features[hi].values[j] - v))));
  }
  return { x, quality: features.reduce((sum, f) => sum + f.quality, 0) / features.length };
}
