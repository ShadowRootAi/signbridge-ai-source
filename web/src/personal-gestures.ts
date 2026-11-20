import { FEATURE_COUNT, preprocess, TIMESTEPS } from './features';
import type { LandmarkFrame } from './features';
import { measureQualityMotion } from './openset';
import { measureRawTracking } from './tracking';
import type { ProgressListener } from './pipeline-contract';

export const teachingCounts = { training: 3, checks: 2, negatives: 2 } as const;
const steps = 20;
const handJoints = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const fingertips = [4, 8, 12, 16, 20];
const handWidth = (handJoints.length + fingertips.length) * 2 + 1;
const vectorWidth = handWidth * 2;
const storageKey = 'signbridge-personal-signs-v1';
const maxSigns = 10;
const maxDistance = .30;

export interface TeachingSample { fingerprint: string; sequence: number[][] }
export interface PersonalSign { label: string; examples: TeachingSample[]; threshold: number }

const labelKey = (label: string) => label.normalize('NFC').trim().toLocaleLowerCase('el-GR').replace(/\s+/g, ' ');

function compact(features: number[][]) {
  return Array.from({ length: steps }, (_, index) => {
    const frame = features[Math.round(index * (TIMESTEPS - 1) / (steps - 1))];
    return [0, 1].flatMap(hand => {
      const start = hand * 84;
      const part = [
        ...handJoints.flatMap(joint => frame.slice(start + 2 * joint, start + 2 * joint + 2)),
        ...fingertips.flatMap(joint => frame.slice(start + 42 + 2 * joint, start + 44 + 2 * joint)),
        Number(frame[FEATURE_COUNT - 4 + hand] >= .75),
      ];
      return part.map(value => Math.round(value * 10_000) / 10_000);
    });
  });
}

function checkDuration(duration: number) {
  if (!Number.isFinite(duration) || duration < .2 || duration > 3) {
    throw new Error('Χρησιμοποίησε βίντεο 0,2–3 δευτερολέπτων με ένα νόημα.');
  }
}

function checkedSequence(frames: LandmarkFrame[]) {
  const tracking = measureRawTracking(frames);
  if (frames.length < 6 || tracking.eitherHandDetection < .8
    || tracking.invalidMetadataFrames > 0 || !tracking.timestampsStrictlyIncreasing) {
    throw new Error('Χρειάζεται πιο καθαρή λήψη με ορατά χέρια και ώμους.');
  }
  const { x } = preprocess(frames);
  if (measureQualityMotion(x).motionBodyRms < .004) throw new Error('Χρειάζεται λήψη με ορατή κίνηση του χεριού.');
  return compact(x);
}

const sequenceKey = (sequence: number[][]) => JSON.stringify(sequence);

async function fingerprintSample(sequence: number[][], source?: Blob): Promise<TeachingSample> {
  const bytes = source ? await source.arrayBuffer() : new TextEncoder().encode(sequenceKey(sequence));
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return {
    fingerprint: Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join(''),
    sequence,
  };
}

export async function teachingSampleFromFrames(frames: LandmarkFrame[], duration: number,
  source?: Blob): Promise<TeachingSample | null> {
  let sequence: number[][];
  try {
    checkDuration(duration);
    sequence = checkedSequence(frames);
  } catch {
    return null;
  }
  return fingerprintSample(sequence, source);
}

export async function collectTeachingSample(
  video: HTMLVideoElement, file: File, signal: AbortSignal, progress?: ProgressListener,
): Promise<TeachingSample> {
  checkDuration(video.duration);
  const { collectVideoLandmarks } = await import('./pipeline');
  const { frames } = await collectVideoLandmarks(video, progress, signal,
    import.meta.env.VITE_SIGNBRIDGE_OPENHANDS_FRAME_OFFSET === '1');
  signal.throwIfAborted();
  const sample = await fingerprintSample(checkedSequence(frames), file);
  signal.throwIfAborted();
  return sample;
}

function poseDistance(a: number[], b: number[]) {
  let total = 0, visible = 0;
  for (let start = 0; start < vectorWidth; start += handWidth) {
    const end = start + handWidth - 1;
    if (a[end] !== b[end]) { total += .5; visible++; continue; }
    if (!a[end]) continue;
    let position = 0, shape = 0;
    const jointEnd = start + 2 * handJoints.length;
    for (let index = start; index < jointEnd; index++) position += (a[index] - b[index]) ** 2;
    for (let index = jointEnd; index < end; index++) shape += (a[index] - b[index]) ** 2;
    total += Math.sqrt(position / handJoints.length + .35 * shape / fingertips.length);
    visible++;
  }
  return visible ? total / visible : .5;
}

/** Align similar movement at slightly different speeds, keeping both endpoints. */
function sequenceDistance(a: TeachingSample, b: TeachingSample) {
  let previous = new Float64Array(steps + 1).fill(Infinity);
  previous[0] = 0;
  for (let i = 1; i <= steps; i++) {
    const current = new Float64Array(steps + 1).fill(Infinity);
    for (let j = Math.max(1, i - 4); j <= Math.min(steps, i + 4); j++) {
      const cost = poseDistance(a.sequence[i - 1], b.sequence[j - 1]);
      current[j] = Math.min(previous[j - 1] + 2 * cost, previous[j] + cost, current[j - 1] + cost);
    }
    previous = current;
  }
  return previous[steps] / (steps * 2);
}

export function personalDistance(sample: TeachingSample, examples: TeachingSample[]) {
  const [first, second] = examples.map(example => sequenceDistance(sample, example)).sort((a, b) => a - b);
  return (first + second) / 2;
}

function validSample(value: TeachingSample) {
  return value && /^[0-9a-f]{64}$/.test(value.fingerprint) && Array.isArray(value.sequence)
    && value.sequence.length === steps && value.sequence.every(row => Array.isArray(row) && row.length === vectorWidth
      && row.every(cell => typeof cell === 'number' && Number.isFinite(cell) && Math.abs(cell) <= 8));
}

export function fitPersonalSign(label: string, training: TeachingSample[], checks: TeachingSample[], negatives: TeachingSample[]) {
  const name = label.trim().normalize('NFC');
  if (!name || name.length > 60) throw new Error('Γράψε ένα όνομα έως 60 χαρακτήρες.');
  if (training.length !== teachingCounts.training || checks.length !== teachingCounts.checks || negatives.length !== teachingCounts.negatives) {
    throw new Error('Χρειάζονται 3 λήψεις εκμάθησης, 2 ελέγχου και 2 με διαφορετικά νοήματα.');
  }
  const all = [...training, ...checks, ...negatives];
  if (all.some(sample => !validSample(sample))) throw new Error('Ένα πρότυπο κίνησης δεν είναι έγκυρο.');
  // Compare exact motion sequences as well, even if the input container has changed.
  if (new Set(all.map(sample => sample.fingerprint)).size !== all.length
    || new Set(all.map(sample => sequenceKey(sample.sequence))).size !== all.length) {
    throw new Error('Χρησιμοποίησε διαφορετική λήψη για κάθε παράδειγμα, χωρίς αντίγραφα.');
  }

  const leaveOneOut = training.map((sample, index) => personalDistance(sample, training.filter((_, i) => i !== index)));
  const threshold = Math.min(maxDistance, 1.15 * Math.max(...leaveOneOut));
  const correct = checks.filter(sample => personalDistance(sample, training) <= threshold).length;
  const rejected = negatives.filter(sample => personalDistance(sample, training) > threshold).length;
  const sign: PersonalSign | null = threshold > 0 && correct === checks.length && rejected === negatives.length
    ? { label: name, examples: training, threshold } : null;
  return { sign, correct, rejected };
}

export function recognizePersonalSign(sample: TeachingSample, signs: PersonalSign[]) {
  const matches = signs.filter(sign => personalDistance(sample, sign.examples) <= sign.threshold);
  return matches.length === 1 ? matches[0].label : null;
}

export function readPersonalSigns(): PersonalSign[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(storageKey) ?? '[]');
    if (!Array.isArray(raw) || raw.length > maxSigns) return [];
    return raw.filter((sign: PersonalSign) => sign && typeof sign.label === 'string' && sign.label.trim()
      && sign.label.length <= 60 && sign.threshold > 0 && sign.threshold <= maxDistance
      && Array.isArray(sign.examples) && sign.examples.length === teachingCounts.training && sign.examples.every(validSample));
  } catch { return []; }
}

export function savePersonalSign(sign: PersonalSign) {
  const next = [...readPersonalSigns().filter(item => labelKey(item.label) !== labelKey(sign.label)), sign];
  if (next.length > maxSigns) throw new Error('Χώρος για έως 10 προσωπικά νοήματα. Διάγραψε ένα για να προσθέσεις άλλο.');
  localStorage.setItem(storageKey, JSON.stringify(next));
  return next;
}

export function deletePersonalSign(label: string) {
  const next = readPersonalSigns().filter(item => labelKey(item.label) !== labelKey(label));
  localStorage.setItem(storageKey, JSON.stringify(next));
  return next;
}

