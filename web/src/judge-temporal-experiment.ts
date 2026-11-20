import { frameFeatures } from './features';
import type { LandmarkFrame } from './features';
import { measureRawTracking } from './tracking';

export const temporalExperiment = Object.freeze({
  seeds: [7, 31, 113],
  minimumPairCoverage: .75,
  minimumHighActivity: .018,
  minimumQuietActivity: .009,
  minimumHotPairs: 3,
  maximumActivityGap: 2,
  minimumRemovedPairs: 3,
  contextMs: 65,
  localShapeWeight: .16,
  minimumWindowFrames: 6,
});

function trackingSignature(frame: LandmarkFrame) {
  const track = measureRawTracking([frame]);
  return [frame.width, frame.height, track.invalidMetadataFrames, track.poseCompleteness, track.leftHandCompleteness,
    track.rightHandCompleteness, track.facePresence].join('/');
}

/** Keep the raw spatial observations, timestamps and per-slot tracking pattern while changing order. */
export function shuffleLandmarkOrder(frames: LandmarkFrame[], seed: number) {
  const groups = new Map<string, number[]>();
  for (const [index, frame] of frames.entries()) {
    const signature = trackingSignature(frame);
    const group = groups.get(signature) ?? [];
    group.push(index);
    groups.set(signature, group);
  }
  const permutation = frames.map((_, index) => index);
  let state = seed >>> 0;
  for (const slots of groups.values()) {
    const values = [...slots];
    for (let index = values.length - 1; index > 0; index--) {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      const swap = Math.floor((state / 2 ** 32) * (index + 1));
      [values[index], values[swap]] = [values[swap], values[index]];
    }
    for (const [index, slot] of slots.entries()) permutation[slot] = values[index];
  }
  return { frames: permutation.map((source, slot) => ({ ...frames[source], t: frames[slot].t })),
    permutation, changedSlots: permutation.filter((source, slot) => source !== slot).length };
}

const fingertips = [4, 8, 12, 16, 20];

/** Motion is measured only where the same hand and both shoulders exist at both ends of the pair. */
function handActivity(previous: number[], current: number[], deltaMs: number) {
  if (previous[222] !== 1 || current[222] !== 1 || !(deltaMs > 0)) return null;
  const movement = [];
  for (const hand of [0, 1]) {
    if (previous[220 + hand] !== 1 || current[220 + hand] !== 1) continue;
    const body = hand * 84;
    const local = body + 42;
    const wrist = Math.hypot(current[body] - previous[body], current[body + 1] - previous[body + 1]);
    const shape = Math.sqrt(fingertips.reduce((sum, tip) => {
      const index = local + 2 * tip;
      return sum + (current[index] - previous[index]) ** 2 + (current[index + 1] - previous[index + 1]) ** 2;
    }, 0) / fingertips.length);
    movement.push(Math.hypot(wrist, shape * temporalExperiment.localShapeWeight));
  }
  return movement.length ? Math.max(...movement) * Math.min(1.5, (1000 / 30) / deltaMs) : null;
}

function quantile(sorted: number[], p: number) {
  return sorted[Math.floor((sorted.length - 1) * p)];
}

export function selectMotionWindow(frames: LandmarkFrame[]) {
  const cfg = temporalExperiment;
  if (frames.length < cfg.minimumWindowFrames + cfg.minimumRemovedPairs) return { reason: 'too_short' as const };
  const values = frames.map(frame => frameFeatures(frame).values);
  const activity = values.slice(1).map((row, i) => handActivity(values[i], row, frames[i + 1].t - frames[i].t));
  const sorted = activity.filter((value): value is number => value !== null).sort((a, b) => a - b);
  const coverage = sorted.length / activity.length;
  if (coverage < cfg.minimumPairCoverage) return { reason: 'insufficient_tracking' as const, coverage };

  const quiet = Math.max(cfg.minimumQuietActivity, 2 * quantile(sorted, .20));
  const high = Math.max(cfg.minimumHighActivity, 2 * quiet, .20 * quantile(sorted, .90));
  const hot = activity.flatMap((value, i) => value !== null && value >= high ? [i] : []);
  const evidence = { coverage, quiet, high, hotPairs: hot.length };
  if (hot.length < cfg.minimumHotPairs) return { reason: 'insufficient_movement' as const, ...evidence };

  const runs: number[][] = [];
  for (const index of hot) {
    const previous = runs.at(-1);
    if (previous && index - previous.at(-1)! <= cfg.maximumActivityGap + 1) previous.push(index);
    else runs.push([index]);
  }
  const coherent = runs.filter(run => run.length >= cfg.minimumHotPairs);
  if (coherent.length !== 1) return { reason: 'ambiguous_movement' as const, ...evidence };

  const run = coherent[0];
  let first = run[0];
  let last = run.at(-1)! + 1;
  while (first > 0 && activity[first - 1] !== null && activity[first - 1]! > quiet) first--;
  while (last < activity.length && activity[last] !== null && activity[last]! > quiet) last++;
  const quietEdge = (pair: number | null) => pair !== null && pair <= quiet;
  if (first < cfg.minimumRemovedPairs || !activity.slice(0, first).every(quietEdge)) first = 0;
  if (activity.length - last < cfg.minimumRemovedPairs || !activity.slice(last).every(quietEdge)) last = frames.length - 1;
  const startMs = frames[first].t - cfg.contextMs;
  const endMs = frames[last].t + cfg.contextMs;
  while (first > 0 && frames[first - 1].t >= startMs) first--;
  while (last + 1 < frames.length && frames[last + 1].t <= endMs) last++;
  const kept = last - first + 1;
  if (frames.length - kept < cfg.minimumRemovedPairs || kept < cfg.minimumWindowFrames) {
    return { reason: 'no_confident_trim' as const, ...evidence };
  }
  return { reason: 'window' as const, first, last, startMs: frames[first].t, endMs: frames[last].t, ...evidence };
}
