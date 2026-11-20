import { analyzeOpenHandsCamera, cancelOpenHands, getOpenHandsPolicy, OPENHANDS_CAPTURE_MS } from './openhands-pipeline';

type Policy = Awaited<ReturnType<typeof getOpenHandsPolicy>>;
type Prediction = Awaited<ReturnType<typeof analyzeOpenHandsCamera>>;
type Mode = 'full' | 'quick' | 'controls';
type Control = 'idle' | 'ordinary_motion' | 'hands_out_of_frame' | 'no_person';
type Trial =
  | { kind: 'known' | 'held_out_sign'; gloss: string; repeat: number }
  | { kind: 'no_sign'; control: Control; repeat: number };
type Attempt = {
  index: number;
  status: 'scored';
  accepted: boolean;
  predictedGloss: string | null;
  rejectionCode: Prediction['rejectionCode'];
  topCandidates: Prediction['topCandidates'];
  trackingQuality: number | null;
  frames: number;
  elapsedMs: number;
  modelVersion: string;
  camera: { width: number; height: number; frameRate: number | null };
} | { index: number; status: 'skipped' };
type Session = {
  id: string;
  createdAt: string;
  signature: string;
  mode: Mode;
  repeats: number;
  seed: number;
  attempts: Attempt[];
};

const durationMs = OPENHANDS_CAPTURE_MS;
const storageKey = 'signbridge-openhands-webcam-v1';
const quickGlosses = ['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ'];
const controls: Record<Control, { name: string; instruction: string }> = {
  idle: { name: 'Μείνε ακίνητος', instruction: 'Κοίτα την κάμερα με τα χέρια χαμηλά.' },
  ordinary_motion: { name: 'Κάνε μια καθημερινή κίνηση', instruction: 'Τακτοποίησε τα μαλλιά ή τα ρούχα σου χωρίς να κάνεις σκόπιμο νόημα.' },
  hands_out_of_frame: { name: 'Χέρια εκτός κάδρου', instruction: 'Άφησε τα χέρια έξω από το κάδρο και κινήσου λίγο φυσιολογικά.' },
  no_person: { name: 'Άφησε άδειο το πλάνο', instruction: 'Μετακινήσου έξω από την εικόνα μόλις αρχίσει η αντίστροφη μέτρηση.' },
};

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modeInput = el<HTMLSelectElement>('mode');
const repeatsInput = el<HTMLSelectElement>('repeats');
const video = el<HTMLVideoElement>('live');
const progress = el<HTMLProgressElement>('progress');
const cue = el('cue');
const error = el('error');
const button = (id: string) => el<HTMLButtonElement>(id);
const start = button('start');
const reset = button('reset');
const download = button('download');
const openCamera = button('open-camera');
const closeCamera = button('close-camera');
const capture = button('capture');
const skip = button('skip');
const saveTrial = button('save-trial');
const retry = button('retry');

let policy: Policy | undefined;
let signature = '';
let session: Session | undefined;
let plan: Trial[] = [];
let stream: MediaStream | undefined;
let task: AbortController | undefined;
let pending: Extract<Attempt, { status: 'scored' }> | undefined;

function seededShuffle<T>(values: T[], seed: number) {
  const shuffled = values.slice();
  let state = seed >>> 0;
  for (let i = shuffled.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const target = Math.floor((state / 4294967296) * (i + 1));
    [shuffled[i], shuffled[target]] = [shuffled[target], shuffled[i]];
  }
  return shuffled;
}

function createPlan(settings: Pick<Session, 'mode' | 'seed' | 'repeats'>, config: Policy) {
  const { mode, seed, repeats } = settings;
  const known = mode === 'full' ? config.supportedLabels : mode === 'quick' ? quickGlosses : [];
  const heldOut = mode === 'full'
    ? seededShuffle(config.unknownLabels, seed ^ 0x5bf03635).slice(0, 16)
    : mode === 'quick' ? seededShuffle(config.unknownLabels, seed ^ 0x5bf03635).slice(0, 1) : [];
  const controlsPerMode = mode === 'controls' ? 3 : mode === 'full' ? 2 : 1;
  const trials: Trial[] = [];
  for (let repeat = 1; repeat <= repeats; repeat++) {
    for (const gloss of known) trials.push({ kind: 'known', gloss, repeat });
    for (const gloss of heldOut) trials.push({ kind: 'held_out_sign', gloss, repeat });
  }
  for (let repeat = 1; repeat <= controlsPerMode; repeat++) {
    for (const control of Object.keys(controls) as Control[]) trials.push({ kind: 'no_sign', control, repeat });
  }
  return seededShuffle(trials, seed);
}

function outcome(trial: Trial, attempt: Attempt) {
  if (attempt.status === 'skipped') return 'skipped';
  if (trial.kind !== 'known') return attempt.accepted ? 'false_accept' : 'rejected';
  if (!attempt.accepted) return 'abstained';
  return attempt.predictedGloss === trial.gloss ? 'correct' : 'wrong';
}

function summarize(trials: Trial[], attempts: Attempt[]) {
  const byGloss: Record<string, { planned: number; scored: number; correct: number; abstained: number; wrong: number }> = Object.create(null);
  const unknown = {
    held_out_sign: { scored: 0, rejected: 0, false_accept: 0 },
    no_sign: { scored: 0, rejected: 0, false_accept: 0 },
  };
  const known = { scored: 0, correct: 0, abstained: 0, wrong: 0, classified: 0, topCandidateCorrect: 0 };
  for (const trial of trials) {
    if (trial.kind !== 'known') continue;
    byGloss[trial.gloss] ??= { planned: 0, scored: 0, correct: 0, abstained: 0, wrong: 0 };
    byGloss[trial.gloss].planned++;
  }
  let skipped = 0;
  for (const attempt of attempts) {
    const trial = trials[attempt.index];
    const decision = outcome(trial, attempt);
    if (decision === 'skipped') {
      skipped++;
    } else if (trial.kind === 'known' && (decision === 'correct' || decision === 'abstained' || decision === 'wrong')) {
      known.scored++;
      known[decision]++;
      byGloss[trial.gloss].scored++;
      byGloss[trial.gloss][decision]++;
      if (attempt.status === 'scored' && attempt.topCandidates[0]) {
        known.classified++;
        if (attempt.topCandidates[0].gloss === trial.gloss) known.topCandidateCorrect++;
      }
    } else if (trial.kind !== 'known' && (decision === 'rejected' || decision === 'false_accept')) {
      unknown[trial.kind].scored++;
      unknown[trial.kind][decision]++;
    }
  }
  return { known, unknown, byGloss, skipped, planned: trials.length, completed: attempts.length,
    distinctGlossesPlanned: Object.keys(byGloss).length,
    distinctGlossesScored: Object.values(byGloss).filter(entry => entry.scored > 0).length,
    distinctGlossesCorrect: Object.values(byGloss).filter(entry => entry.correct > 0).length };
}

function showSummary() {
  if (!session || !policy) return;
  const data = summarize(plan, session.attempts);
  el('known-summary').textContent = data.known.scored
    ? `Από τις ${data.known.scored} δοκιμές γνωστών νοημάτων, ${data.known.correct} ήταν σωστές, ${data.known.abstained} κατέληξαν σε αποχή και ${data.known.wrong} σε λάθος. Ελέγχθηκαν ${data.distinctGlossesScored}/${data.distinctGlossesPlanned} διαφορετικά νοήματα, με τουλάχιστον μία σωστή πρόβλεψη σε ${data.distinctGlossesCorrect}.`
    : 'Καμία καταγεγραμμένη δοκιμή γνωστού νοήματος.';
  el('candidate-summary').textContent = data.known.classified
    ? `Πριν εφαρμοστούν τα όρια αποδοχής, ο πρώτος υποψήφιος ήταν σωστός σε ${data.known.topCandidateCorrect}/${data.known.classified} δοκιμές γνωστών νοημάτων όπου εκτελέστηκε το μοντέλο.`
    : '';
  const breakdown = [
    ['νοήματα εκτός του καταλόγου της δοκιμής', data.unknown.held_out_sign],
    ['καθημερινές κινήσεις ή άδειο πλάνο', data.unknown.no_sign],
  ] as const;
  el('unknown-summary').textContent = breakdown.map(([name, counts]) =>
    counts.scored
      ? `Για ${name} καταγράφηκαν ${counts.rejected}/${counts.scored} απορρίψεις και ${counts.false_accept} λανθασμένες αποδοχές.`
      : `Δεν καταγράφηκε δοκιμή για ${name}.`,
  ).join(' ');
  el('skip-summary').textContent = `Παραλείφθηκαν ${data.skipped} δοκιμές. Οι παραλείψεις δεν υπολογίζονται στην ακρίβεια.`;
  el('table-area').hidden = !data.distinctGlossesPlanned;
  el('per-label').replaceChildren(...Object.entries(data.byGloss).sort(([a], [b]) => a.localeCompare(b, 'el')).map(([gloss, entry]) => {
    const row = document.createElement('tr');
    for (const value of [policy?.displayLabels?.[gloss] ?? gloss, `${entry.scored}/${entry.planned}`, entry.correct, entry.abstained, entry.wrong]) {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      row.append(cell);
    }
    return row;
  }));
}

function render() {
  const current = session ? plan[session.attempts.length] : undefined;
  const waiting = !!task;
  modeInput.disabled = !policy || !!session || waiting;
  repeatsInput.disabled = modeInput.disabled || modeInput.value !== 'full';
  start.disabled = !policy || !!session || waiting;
  reset.disabled = !session || waiting;
  download.disabled = !session?.attempts.length || waiting;
  openCamera.disabled = !current || !!stream || waiting;
  closeCamera.disabled = !stream && !waiting;
  capture.disabled = !current || !stream || waiting || !!pending;
  skip.disabled = !current || waiting || !!pending;
  saveTrial.hidden = retry.hidden = !pending;
  video.hidden = !stream;
  el('trial-area').hidden = !current;
  el('summary').hidden = !session || !!current;
  if (!session) return;
  if (!current) {
    showSummary();
    return;
  }
  progress.max = plan.length;
  progress.value = session.attempts.length;
  el('progress-label').textContent = `Δοκιμή ${session.attempts.length + 1} από ${plan.length}`;
  if (current.kind === 'no_sign') {
    el('prompt').textContent = controls[current.control].name;
    el('instructions').textContent = controls[current.control].instruction;
  } else {
    el('prompt').textContent = policy?.displayLabels?.[current.gloss] ?? current.gloss;
    el('instructions').textContent = 'Κάνε το νόημα μία φορά όταν εμφανιστεί «Τώρα». Αν δεν γνωρίζεις την εκτέλεσή του, επίλεξε «Παράλειψη».';
  }
}

function persist() {
  if (!session) return;
  try { localStorage.setItem(storageKey, JSON.stringify(session)); }
  catch { el('resume-note').textContent = 'Η αυτόματη αποθήκευση είναι απενεργοποιημένη. Κράτησε το JSON πριν κλείσεις τη σελίδα.'; }
}

function restore(config: Policy) {
  try {
    const stored = localStorage.getItem(storageKey);
    if (!stored) return;
    const saved: Session = JSON.parse(stored);
    if (saved.signature !== signature || typeof saved.id !== 'string' || typeof saved.createdAt !== 'string' ||
        !['full', 'quick', 'controls'].includes(saved.mode) || ![1, 2].includes(saved.repeats) ||
        !Number.isInteger(saved.seed) || !Array.isArray(saved.attempts)) return;
    const previousPlan = createPlan(saved, config);
    if (saved.attempts.length > previousPlan.length || saved.attempts.some((row, index) =>
      row.index !== index || (row.status !== 'scored' && row.status !== 'skipped') ||
      (row.status === 'scored' && (typeof row.accepted !== 'boolean' ||
        !Array.isArray(row.topCandidates) ||
        (row.accepted ? !config.supportedLabels.includes(row.predictedGloss ?? '') : row.predictedGloss !== null))))) return;
    session = saved;
    plan = previousPlan;
    modeInput.value = saved.mode;
    repeatsInput.value = String(saved.repeats);
    el('resume-note').textContent = 'Φορτώθηκε η προηγούμενη τοπική συνεδρία. Μπορείς να συνεχίσεις ή να κατεβάσεις όσα έχεις συγκεντρώσει.';
  } catch {
    el('resume-note').textContent = 'Δεν ήταν δυνατή η ανάκτηση προηγούμενης συνεδρίας.';
  }
}

async function run(action: (signal: AbortSignal) => Promise<void>) {
  if (task) return;
  const controller = new AbortController();
  task = controller;
  error.textContent = '';
  render();
  try { await action(controller.signal); }
  catch (cause) {
    cue.textContent = '';
    if (!controller.signal.aborted) error.textContent = cause instanceof Error ? cause.message : String(cause);
  } finally {
    task = undefined;
    render();
  }
}

function closeStream() {
  task?.abort();
  cancelOpenHands();
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.srcObject = null;
  cue.textContent = '';
}

function wait(ms: number, signal: AbortSignal) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, ms);
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    signal.addEventListener('abort', abort, { once: true });
  });
}

function finishAttempt(attempt: Attempt) {
  if (!session || attempt.index !== session.attempts.length) return;
  session.attempts.push(attempt);
  pending = undefined;
  cue.textContent = 'Η δοκιμή καταχωρίστηκε.';
  persist();
  if (session.attempts.length === plan.length) closeStream();
  render();
}

modeInput.addEventListener('change', render);
start.addEventListener('click', () => {
  if (!policy || session) return;
  const mode = modeInput.value as Mode;
  if (mode === 'quick' && quickGlosses.some(gloss => !policy?.supportedLabels.includes(gloss))) {
    error.textContent = 'Το ενεργό μοντέλο δεν έχει και τα τρία νοήματα της σύντομης δοκιμής.';
    return;
  }
  const seed = mode === 'quick' ? 214937029 : crypto.getRandomValues(new Uint32Array(1))[0];
  session = {
    id: crypto.randomUUID(), createdAt: new Date().toISOString(), signature, mode,
    repeats: mode === 'full' ? Number(repeatsInput.value) : 1, seed, attempts: [],
  };
  plan = createPlan(session, policy);
  el('resume-note').textContent = 'Η πρόοδος αποθηκεύεται σε αυτόν τον browser.';
  error.textContent = '';
  persist();
  render();
});
reset.addEventListener('click', () => {
  if (session?.attempts.length && !window.confirm('Νέα συνεδρία; Για να κρατήσεις την προηγούμενη, κατέβασε πρώτα το JSON.')) return;
  closeStream();
  session = undefined;
  pending = undefined;
  plan = [];
  try { localStorage.removeItem(storageKey); } catch { /* Storage may be disabled. */ }
  el('resume-note').textContent = '';
  error.textContent = '';
  render();
});
openCamera.addEventListener('click', () => void run(async signal => {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Ο browser δεν επιτρέπει πρόσβαση στην κάμερα.');
  cue.textContent = 'Περιμένω άδεια για την κάμερα…';
  const acquired = await navigator.mediaDevices.getUserMedia({
    audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
  });
  try {
    signal.throwIfAborted();
    stream = acquired;
    video.srcObject = acquired;
    video.hidden = false;
    await video.play();
    signal.throwIfAborted();
    cue.textContent = 'Η κάμερα είναι έτοιμη.';
  } catch (cause) {
    acquired.getTracks().forEach(track => track.stop());
    stream = undefined;
    video.srcObject = null;
    throw cause;
  }
}));
closeCamera.addEventListener('click', () => {
  closeStream();
  render();
});
capture.addEventListener('click', () => void run(async signal => {
  if (!session || !stream || pending) return;
  const index = session.attempts.length;
  const frameRate = stream.getVideoTracks()[0]?.getSettings().frameRate ?? null;
  for (const count of [3, 2, 1]) {
    cue.textContent = String(count);
    await wait(600, signal);
  }
  cue.textContent = 'Περιμένω να αρχίσει η λήψη…';
  const result = await analyzeOpenHandsCamera(video, durationMs, signal, update => {
    if (update.stage === 'capture') cue.textContent = 'Τώρα';
    if (update.stage === 'tracking' || update.stage === 'inference') cue.textContent = 'Η λήψη ολοκληρώθηκε. Ανάλυση…';
  });
  signal.throwIfAborted();
  if (result.accepted && result.label === null) throw new Error('Το μοντέλο δεν επέστρεψε έγκυρη πρόβλεψη.');
  pending = {
    index, status: 'scored', accepted: result.accepted, predictedGloss: result.accepted ? result.label : null,
    rejectionCode: result.rejectionCode, topCandidates: result.topCandidates,
    trackingQuality: result.trackingQuality ?? null, frames: result.frames,
    elapsedMs: result.elapsedMs, modelVersion: result.modelVersion,
    camera: { width: video.videoWidth, height: video.videoHeight, frameRate },
  };
  cue.textContent = 'Επιβεβαίωσε μόνο αν εκτέλεσες αυτό που ζητήθηκε.';
}));
saveTrial.addEventListener('click', () => { if (pending) finishAttempt(pending); });
retry.addEventListener('click', () => {
  pending = undefined;
  cue.textContent = 'Έτοιμο για επανάληψη.';
  render();
});
skip.addEventListener('click', () => {
  if (session) finishAttempt({ index: session.attempts.length, status: 'skipped' });
});
download.addEventListener('click', () => {
  if (!session || !policy) return;
  const { id, mode, repeats, seed, createdAt, attempts } = session;
  const report = {
    schema: 'signbridge-openhands-webcam-v1', id, mode, repeats, seed, createdAt,
    exportedAt: new Date().toISOString(), captureDurationMs: durationMs,
    policy, summary: summarize(plan, attempts),
    trials: plan.map((trial, index) => ({ index, ...trial })),
    attempts: attempts.map(attempt => ({ ...attempt, outcome: outcome(plan[attempt.index], attempt) })),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `signbridge-openhands-${id}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener('pagehide', closeStream);

async function init() {
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname)) {
    throw new Error('Άνοιξε τη δοκιμή από τον τοπικό server του SignBridge.');
  }
  const config = await getOpenHandsPolicy();
  if (config.schemaVersion !== 1 || !/^[0-9a-f]{64}$/.test(config.runtimeModelSha256) ||
      config.supportedLabels.length < 101 || config.unknownLabels.length < 16) {
    throw new Error('Λείπουν νοήματα από τον κατάλογο της πλήρους δοκιμής.');
  }
  signature = JSON.stringify([config.schemaVersion, config.sourceModelSha256, config.runtimeModelSha256, config.supportedLabels,
    config.unknownLabels, config.minConfidence, config.minMargin, config.decisionGate ?? null]);
  policy = config;
  el('roster').textContent = `Η πλήρης δοκιμή καλύπτει ${policy.supportedLabels.length} γνωστά νοήματα, 16 νοήματα εκτός καταλόγου και καθημερινές κινήσεις.`;
  restore(policy);
  render();
}
void init().catch((cause: unknown) => {
  el('roster').textContent = 'Η δοκιμή κάμερας δεν είναι διαθέσιμη.';
  error.textContent = cause instanceof Error ? cause.message : String(cause);
});
