/** Raw-frame engineering guard. This is presence/completeness, not recognition accuracy. */
import type { LandmarkFrame, Point } from './features';

export const TRACKING_POLICY = Object.freeze({
  version: 'signbridge-raw-both-hands-v4-20261108',
  minimumFrames: 4,
  minimumRequiredCoverage: 0.90,
  shoulderVisibilityMinimum: 0.5,
  motionThreshold: null,
  interpretation: 'Engineering input-completeness rule, not a fitted or validated accuracy guarantee.',
  limitation: 'Requires both hands; may reject a valid one-handed sign when the other hand is outside the frame.',
});

export type TrackingRejectionReason = 'too_few_frames' | 'invalid_frame_metadata' | 'incomplete_required_landmarks';
export interface TrackingDiagnostics {
  policyVersion: string;
  source: 'raw_landmark_frames';
  frameCount: number;
  poseCompleteness: number;
  leftHandCompleteness: number;
  rightHandCompleteness: number;
  facePresence: number;
  /** Legacy detection criterion: pose and either hand. Never overall completeness. */
  eitherHandDetection: number;
  /** Fraction of raw frames with pose and BOTH hands, before interpolation. */
  requiredDataCompleteness: number;
  allRequiredFramesPresent: boolean;
  usable: boolean;
  reasons: TrackingRejectionReason[];
  minRequiredCoverage: number;
  minimumFrames: number;
  invalidMetadataFrames: number;
  timestampsStrictlyIncreasing: boolean;
  missingFrameCounts: { pose: number; leftHand: number; rightHand: number; required: number };
  longestMissingRuns: { pose: number; leftHand: number; rightHand: number; required: number };
  /** Adjacent raw endpoints with pose and both hands / all adjacent raw pairs. No minimum motion gate. */
  validMotionPairFraction: number;
  validMotionPairs: number;
  totalMotionPairs: number;
}

function pointsValid(points: Point[], count: number): boolean {
  return Array.isArray(points) && points.length >= count
    && points.slice(0, count).every(p => p && Number.isFinite(p.x) && Number.isFinite(p.y));
}
function longestMissing(values: boolean[]): number {
  let longest = 0, current = 0;
  for (const value of values) { current = value ? 0 : current + 1; longest = Math.max(longest, current); }
  return longest;
}

export function measureRawTracking(frames: LandmarkFrame[]): TrackingDiagnostics {
  const n = frames.length;
  const pose: boolean[] = [], left: boolean[] = [], right: boolean[] = [], face: boolean[] = [], required: boolean[] = [], either: boolean[] = [];
  let invalidMetadataFrames = 0, timestampsStrictlyIncreasing = true;
  for (let i = 0; i < n; i++) {
    const frame = frames[i];
    const metadataValid = Number.isFinite(frame.t) && Number.isFinite(frame.width) && frame.width > 0
      && Number.isFinite(frame.height) && frame.height > 0;
    if (!metadataValid) invalidMetadataFrames++;
    if (!Number.isFinite(frame.t) || (i > 0 && !(frame.t > frames[i - 1].t))) timestampsStrictlyIncreasing = false;
    const shoulders = pointsValid(frame.pose, 25)
      && [11, 12].every(j => Number.isFinite(frame.pose[j].visibility ?? 1)
        && (frame.pose[j].visibility ?? 1) >= TRACKING_POLICY.shoulderVisibilityMinimum);
    const p = metadataValid && shoulders;
    const l = metadataValid && pointsValid(frame.left, 21), r = metadataValid && pointsValid(frame.right, 21);
    pose.push(p); left.push(l); right.push(r); face.push(metadataValid && pointsValid(frame.face, 387));
    required.push(p && l && r); either.push(p && (l || r));
  }
  const count = (values: boolean[]) => values.reduce((sum, value) => sum + Number(value), 0);
  const fraction = (values: boolean[]) => n ? count(values) / n : 0;
  const requiredCount = count(required), totalMotionPairs = Math.max(0, n - 1);
  let validMotionPairs = 0;
  for (let i = 1; i < n; i++) if (required[i - 1] && required[i] && frames[i].t > frames[i - 1].t) validMotionPairs++;
  const reasons: TrackingRejectionReason[] = [];
  if (n < TRACKING_POLICY.minimumFrames) reasons.push('too_few_frames');
  if (invalidMetadataFrames > 0 || !timestampsStrictlyIncreasing) reasons.push('invalid_frame_metadata');
  if (!n || requiredCount / n < TRACKING_POLICY.minimumRequiredCoverage) reasons.push('incomplete_required_landmarks');
  return {
    policyVersion: TRACKING_POLICY.version, source: 'raw_landmark_frames', frameCount: n,
    poseCompleteness: fraction(pose), leftHandCompleteness: fraction(left), rightHandCompleteness: fraction(right),
    facePresence: fraction(face), eitherHandDetection: fraction(either), requiredDataCompleteness: n ? requiredCount / n : 0,
    allRequiredFramesPresent: n > 0 && requiredCount === n, usable: reasons.length === 0, reasons,
    minRequiredCoverage: TRACKING_POLICY.minimumRequiredCoverage, minimumFrames: TRACKING_POLICY.minimumFrames,
    invalidMetadataFrames, timestampsStrictlyIncreasing,
    missingFrameCounts: { pose: n - count(pose), leftHand: n - count(left), rightHand: n - count(right), required: n - requiredCount },
    longestMissingRuns: { pose: longestMissing(pose), leftHand: longestMissing(left), rightHand: longestMissing(right), required: longestMissing(required) },
    validMotionPairFraction: totalMotionPairs ? validMotionPairs / totalMotionPairs : 0,
    validMotionPairs, totalMotionPairs,
  };
}

export function trackingRetryMessage(tracking: TrackingDiagnostics): string {
  if (tracking.reasons.includes('too_few_frames')) return 'Χρειάζεται επανάληψη: καταγράφηκαν πολύ λίγα καρέ.';
  if (tracking.reasons.includes('invalid_frame_metadata')) return 'Χρειάζεται επανάληψη: η λήψη έχει μη έγκυρα καρέ ή χρονική σειρά.';
  return 'Χρειάζεται επανάληψη: κράτησε και τα δύο χέρια και το πάνω μέρος του σώματος ορατά σε όλη τη λήψη.';
}
