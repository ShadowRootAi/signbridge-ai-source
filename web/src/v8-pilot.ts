import pilotConfigUrl from '../../artifacts/v8/browser/model-config.json?url';
import handModelUrl from '../../research/cache/mediapipe/hand_landmarker-float16-1.task?url';
import { createV8Pilot } from './v8-pilot-core';

const debugMode = import.meta.env.DEV && new URLSearchParams(window.location.search).get('debug') === '1';
document.documentElement.dataset.mode = debugMode ? 'debug' : 'app';
const pilotOptions = { v8ConfigUrl: pilotConfigUrl, handModelUrl };
let pilot = createV8Pilot(pilotOptions);
const byId = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fileInput = byId<HTMLInputElement>('file-input');
const consent = byId<HTMLInputElement>('consent');
const mirror = byId<HTMLInputElement>('mirror');
const blind = byId<HTMLInputElement>('blind');
const collectionFields = byId<HTMLFieldSetElement>('collection-fields');
const live = byId<HTMLVideoElement>('live-video');
const preview = byId<HTMLVideoElement>('preview-video');
const openCamera = byId<HTMLButtonElement>('open-camera');
const closeCamera = byId<HTMLButtonElement>('close-camera');
const record = byId<HTMLButtonElement>('record');
const stopRecording = byId<HTMLButtonElement>('stop-recording');
const analyze = byId<HTMLButtonElement>('analyze');
const clear = byId<HTMLButtonElement>('clear');
const results = byId('results');
const publicResultArea = byId('public-result-area');
const stage = byId('capture-stage');
const status = byId('status');
const error = byId('error');
const group = byId<HTMLSelectElement>('trial-group');

type CameraMetadata = {
  preview_mirrored: boolean;
  width: number | null;
  height: number | null;
  frame_rate: number | null;
  facing_mode: string | null;
};
type Clip = {
  blob: Blob;
  source: 'file' | 'webcam';
  camera: CameraMetadata | null;
  extension: 'mp4' | 'webm';
};
type Comparison = Awaited<ReturnType<typeof pilot.analyzeVideo>>;
type Report = {
  schema: 'signbridge-v8-pilot-v1';
  clip_id: string;
  source: Clip['source'];
  media: { sha256: string; mime_type: string; byte_length: number; duration_ms: number; width: number; height: number };
  collection: ReturnType<typeof collectionMetadata>;
  comparison: Comparison;
  camera: CameraMetadata | null;
};

let stream: MediaStream | undefined;
let recorder: MediaRecorder | undefined;
let task: { kind: 'load' | 'open' | 'capture' | 'analyze'; controller: AbortController } | undefined;
let clip: Clip | undefined;
let report: Report | undefined;
let clipUrl: string | undefined;

function update() {
  const busy = task !== undefined;
  const analyzed = !results.hidden;
  consent.disabled = busy;
  fileInput.disabled = busy;
  openCamera.disabled = busy || !consent.checked || !!stream;
  closeCamera.disabled = !stream;
  record.disabled = busy || !stream || !consent.checked || !!clip;
  stopRecording.disabled = recorder?.state !== 'recording';
  mirror.disabled = task?.kind === 'capture';
  analyze.disabled = busy || !clip || analyzed;
  clear.disabled = busy || !clip;
  preview.controls = !busy;
  blind.disabled = !debugMode || busy || analyzed;
  collectionFields.disabled = blind.disabled;
  collectionFields.hidden = !debugMode || !blind.checked;
  byId('camera-area').hidden = !stream;
  byId('stage-empty').hidden = !!stream || !!clip;
  if (task?.kind === 'analyze') stage.dataset.state = 'analyzing';
  else if (task?.kind === 'capture') stage.dataset.state = 'recording';
  else if (analyzed) stage.dataset.state = 'complete';
  else if (clip) stage.dataset.state = 'ready';
  else stage.dataset.state = stream ? 'camera' : 'idle';
  stage.setAttribute('aria-busy', String(task?.kind === 'analyze'));
  byId('known-field').hidden = group.value !== 'known';
  byId('unknown-field').hidden = group.value === 'known';
}

function clearClip() {
  clip = undefined;
  report = undefined;
  results.hidden = true;
  delete results.dataset.state;
  publicResultArea.hidden = true;
  delete publicResultArea.dataset.outcome;
  byId('public-result').textContent = '';
  for (const id of ['baseline-result', 'v8-result']) {
    const output = byId(id);
    output.textContent = '';
    delete output.dataset.accepted;
    delete output.dataset.label;
  }
  byId('tracking-result').textContent = '';
  byId('timing-result').textContent = '';
  byId('preview-area').hidden = true;
  preview.removeAttribute('src');
  preview.load();
  if (clipUrl) URL.revokeObjectURL(clipUrl);
  clipUrl = undefined;
}

async function run(kind: NonNullable<typeof task>['kind'], action: (signal: AbortSignal) => Promise<void>) {
  if (task) return;
  const controller = new AbortController();
  task = { kind, controller };
  if (kind === 'analyze') preview.pause();
  error.textContent = '';
  update();
  try {
    await action(controller.signal);
  } catch (failure) {
    if (controller.signal.aborted) status.textContent = 'Η ενέργεια ακυρώθηκε.';
    else {
      status.textContent = 'Η ενέργεια δεν ολοκληρώθηκε.';
      error.textContent = kind === 'analyze' && !debugMode
        ? 'Δεν ήταν δυνατή η ανάλυση. Δοκιμάστε ξανά.'
        : failure instanceof Error ? failure.message : String(failure);
    }
  } finally {
    task = undefined;
    update();
  }
}

function mediaEvent(video: HTMLVideoElement, event: 'loadeddata' | 'seeked', signal: AbortSignal, start: () => void) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const ready = () => done();
    const failed = () => done(new Error('Ο browser δεν μπορεί να διαβάσει αυτό το βίντεο. Δοκιμάστε MP4 ή WebM.'));
    const cancelled = () => done(signal.reason);
    const timer = setTimeout(() => done(new Error('Η αποκωδικοποίηση του βίντεο καθυστέρησε.')), 15_000);
    function done(failure?: unknown) {
      clearTimeout(timer);
      video.removeEventListener(event, ready);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', cancelled);
      if (failure) reject(failure); else resolve();
    }
    video.addEventListener(event, ready, { once: true });
    video.addEventListener('error', failed, { once: true });
    signal.addEventListener('abort', cancelled, { once: true });
    try { start(); } catch (failure) { done(failure); }
  });
}

async function prepareClip(next: Clip, signal: AbortSignal) {
  clearClip();
  if (next.blob.size === 0 || next.blob.size > 40_000_000) throw new Error('Επιλέξτε βίντεο μεγέθους έως 40 MB.');
  const url = URL.createObjectURL(next.blob);
  clipUrl = url;
  status.textContent = 'Αποκωδικοποίηση βίντεο…';
  try {
    await mediaEvent(preview, 'loadeddata', signal, () => { preview.src = url; preview.load(); });
    // MediaRecorder WebM can omit the duration field. Seeking to its end makes
    // Chromium resolve the real duration of the exact same downloadable bytes.
    if (!Number.isFinite(preview.duration)) {
      await mediaEvent(preview, 'seeked', signal, () => { preview.currentTime = 1e8; });
      await mediaEvent(preview, 'seeked', signal, () => { preview.currentTime = 0; });
    }
    if (!Number.isFinite(preview.duration) || preview.duration <= 0 || preview.duration > 15 || !preview.videoWidth || !preview.videoHeight) {
      throw new Error('Χρειάζεται αναγνώσιμο βίντεο διάρκειας έως 15 δευτερολέπτων.');
    }
    clip = next;
    byId('preview-area').hidden = false;
    if (debugMode) byId('media-info').textContent = `${preview.videoWidth} × ${preview.videoHeight} · ${preview.duration.toFixed(2)} δευτ.`;
    status.textContent = 'Το βίντεο είναι έτοιμο για ανάλυση.';
  } catch (failure) {
    clearClip();
    fileInput.value = '';
    throw failure;
  }
}

function releaseCamera() {
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  live.srcObject = null;
  update();
}

function shutCamera() {
  if (task?.kind === 'capture' || task?.kind === 'open') task.controller.abort();
  releaseCamera();
}

function capture(active: MediaStream, signal: AbortSignal) {
  if (typeof MediaRecorder === 'undefined') throw new Error('Ο browser δεν υποστηρίζει εγγραφή βίντεο.');
  const type = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(mime => MediaRecorder.isTypeSupported(mime));
  if (!type) throw new Error('Δοκιμάστε Chrome για καταγραφή WebM ή MP4.');
  const recording = new MediaRecorder(active, { mimeType: type, videoBitsPerSecond: 2_000_000 });
  recorder = recording;
  return new Promise<Blob>((resolve, reject) => {
    const parts: BlobPart[] = [];
    const stop = () => { if (recording.state !== 'inactive') recording.stop(); };
    const timer = setTimeout(stop, 3000);
    const failTimer = setTimeout(() => finish(new Error('Η κάμερα δεν ολοκλήρωσε την εγγραφή εγκαίρως.')), 8000);
    const abort = () => finish(signal.reason);
    function finish(failure?: unknown) {
      clearTimeout(timer);
      clearTimeout(failTimer);
      signal.removeEventListener('abort', abort);
      recording.ondataavailable = recording.onerror = recording.onstop = null;
      stop();
      recorder = undefined;
      update();
      if (failure) reject(failure);
      else resolve(new Blob(parts, { type: recording.mimeType || type }));
    }
    signal.addEventListener('abort', abort, { once: true });
    recording.ondataavailable = event => { if (event.data.size) parts.push(event.data); };
    recording.onerror = () => finish(new Error('Η κάμερα δεν ολοκλήρωσε την εγγραφή.'));
    recording.onstop = () => finish();
    try { recording.start(); update(); } catch (failure) { finish(failure); }
  });
}

function collectionMetadata() {
  if (!blind.checked) return null;
  const signer = byId<HTMLInputElement>('signer-id').value.trim();
  const session = byId<HTMLInputElement>('session-id').value.trim();
  if (!/^P\d{3,6}$/.test(signer) || !/^[A-Za-z0-9_-]{1,24}$/.test(session)) {
    throw new Error('Συμπληρώστε ψευδώνυμο όπως P001 και κωδικό συνεδρίας όπως S01.');
  }
  if (!byId<HTMLInputElement>('collection-consent').checked) throw new Error('Επιβεβαιώστε τα στοιχεία συμμετοχής πριν από την ανάλυση.');
  const known = byId<HTMLSelectElement>('known-label').value;
  const unknown = byId<HTMLInputElement>('unknown-category').value.trim();
  if (group.value === 'known' && !known) throw new Error('Επιλέξτε το επαληθευμένο νόημα.');
  if (group.value !== 'known' && !/^u_[a-z0-9][a-z0-9_]{0,37}$/.test(unknown)) throw new Error('Χρησιμοποιήστε κωδικό κατηγορίας όπως u_idle.');
  const truth = group.value === 'known' ? { kind: 'known', id: known } : { kind: 'unknown', category_id: unknown };
  return {
    signer_id: signer, session_id: session, condition_id: byId<HTMLSelectElement>('condition-id').value,
    trial_group: group.value, truth, consent_local_ml: true, human_recorded: true, new_participant_declared: true,
  };
}

function showDiagnostics(comparison: Comparison) {
  const before = comparison.originalTracking;
  const after = comparison.fusedTracking;
  if (!before || !after) return;
  const percent = new Intl.NumberFormat('el', { style: 'percent', maximumFractionDigits: 1 });
  const oldCoverage = percent.format(before.requiredDataCompleteness);
  const newCoverage = percent.format(after.requiredDataCompleteness);
  const outcome = comparison.classifierExecuted ? 'Έγινε μία κοινή πρόβλεψη.' : 'Τα σημεία ήταν ανεπαρκή για πρόβλεψη.';
  byId('tracking-result').textContent = `Πληρότητα σημείων από ${oldCoverage} σε ${newCoverage} μετά την ανάκτηση. ${outcome}`;
  byId('timing-result').textContent = `${comparison.frames} καρέ · ${(comparison.timings.totalMs / 1000).toFixed(1)} δευτ. επεξεργασίας`;
}

function render(comparison: Comparison, blindResult: boolean) {
  byId('visible-results').hidden = !debugMode || blindResult;
  byId('blind-result').hidden = !debugMode || !blindResult;
  if (!debugMode) {
    const { accepted, label } = comparison.v8;
    byId('public-result').textContent = accepted && label ? label : 'Δεν αναγνωρίστηκε';
    publicResultArea.dataset.outcome = accepted ? 'accepted' : 'rejected';
    publicResultArea.hidden = false;
  } else if (!blindResult) {
    for (const [id, decision] of [['baseline-result', comparison.baseline], ['v8-result', comparison.v8]] as const) {
      const output = byId(id);
      output.textContent = decision.accepted && decision.label ? decision.label : 'Δεν αναγνωρίστηκε';
      output.dataset.accepted = String(decision.accepted);
      output.dataset.label = decision.label ?? '';
    }
    showDiagnostics(comparison);
  }
  results.hidden = false;
  results.dataset.state = 'ready';
}

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  if (!debugMode && stream) shutCamera();
  const extension = file.name.toLowerCase().endsWith('.webm') ? 'webm' : 'mp4';
  void run('load', signal => prepareClip({ blob: file, source: 'file', camera: null, extension }, signal));
});
preview.addEventListener('play', () => { if (task?.kind === 'analyze') preview.pause(); });
consent.addEventListener('change', () => { if (!consent.checked) shutCamera(); update(); });
blind.addEventListener('change', update);
group.addEventListener('change', update);
mirror.addEventListener('change', () => live.classList.toggle('mirrored', mirror.checked));
openCamera.addEventListener('click', () => void run('open', async signal => {
  if (!consent.checked || !navigator.mediaDevices?.getUserMedia) throw new Error('Η κάμερα χρειάζεται άδεια και ασφαλή τοπική σελίδα.');
  status.textContent = 'Περιμένουμε άδεια κάμερας…';
  const active = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: { ideal: 'user' } }, audio: false,
  }).catch(failure => {
    if (failure instanceof DOMException && failure.name === 'NotAllowedError') {
      throw new Error('Η πρόσβαση στην κάμερα δεν επιτράπηκε. Ελέγξτε την άδεια του browser και δοκιμάστε ξανά.');
    }
    if (!debugMode) throw new Error('Η κάμερα δεν άνοιξε. Ελέγξτε τη σύνδεσή της και δοκιμάστε ξανά.');
    throw failure;
  });
  if (signal.aborted) {
    active.getTracks().forEach(track => track.stop());
    signal.throwIfAborted();
  }
  if (!debugMode) {
    clearClip();
    fileInput.value = '';
  }
  stream = active;
  live.srcObject = active;
  update();
  for (const track of active.getVideoTracks()) track.addEventListener('ended', () => { if (stream === active) shutCamera(); }, { once: true });
  try { await live.play(); } catch (failure) { releaseCamera(); throw failure; }
  signal.throwIfAborted();
  status.textContent = 'Η κάμερα είναι έτοιμη.';
}));
closeCamera.addEventListener('click', () => { shutCamera(); if (!task) status.textContent = 'Η κάμερα έκλεισε.'; });
record.addEventListener('click', () => void run('capture', async signal => {
  const active = stream;
  if (!active || !consent.checked) return;
  const track = active.getVideoTracks()[0];
  const settings = track.getSettings();
  const camera = {
    preview_mirrored: mirror.checked, width: settings.width ?? null, height: settings.height ?? null,
    frame_rate: settings.frameRate ?? null, facing_mode: settings.facingMode ?? null,
  };
  for (let seconds = 3; seconds > 0; seconds--) {
    status.textContent = `Έναρξη σε ${seconds}…`;
    await new Promise(resolve => setTimeout(resolve, 1000));
    signal.throwIfAborted();
  }
  if (track.readyState !== 'live') throw new Error('Η κάμερα έκλεισε πριν από την εγγραφή.');
  status.textContent = 'Καταγραφή τώρα.';
  const blob = await capture(active, signal);
  if (!debugMode) releaseCamera();
  await prepareClip({ blob, source: 'webcam', camera, extension: blob.type.includes('mp4') ? 'mp4' : 'webm' }, signal);
}));
stopRecording.addEventListener('click', () => { if (recorder?.state === 'recording') recorder.stop(); });
clear.addEventListener('click', () => {
  clearClip();
  fileInput.value = '';
  status.textContent = 'Επιλέξτε νέο απόσπασμα.';
  error.textContent = '';
  update();
});
analyze.addEventListener('click', () => void run('analyze', async signal => {
  if (!clip) return;
  const collection = debugMode ? collectionMetadata() : null;
  status.textContent = 'Ανάλυση βίντεο…';
  const comparison = await pilot.analyzeVideo(preview, debugMode ? progress => { status.textContent = progress.message; } : undefined, signal);
  signal.throwIfAborted();
  if (debugMode) {
    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await clip.blob.arrayBuffer())))
      .map(byte => byte.toString(16).padStart(2, '0')).join('');
    signal.throwIfAborted();
    const media = {
      sha256, mime_type: clip.blob.type || `video/${clip.extension}`, byte_length: clip.blob.size,
      duration_ms: preview.duration * 1000, width: preview.videoWidth, height: preview.videoHeight,
    };
    report = { schema: 'signbridge-v8-pilot-v1', clip_id: `V8_${crypto.randomUUID()}`, source: clip.source, media, collection, comparison, camera: clip.camera };
  }
  render(comparison, collection !== null);
  status.textContent = debugMode ? 'Η σύγκριση ολοκληρώθηκε.' : 'Η ανάλυση ολοκληρώθηκε.';
}));
byId('save-report').addEventListener('click', () => {
  if (report) download(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), `${report.clip_id}.json`);
});
byId('save-video').addEventListener('click', () => {
  if (report && clip) download(clip.blob, `${report.clip_id}.${clip.extension}`);
});
window.addEventListener('pagehide', () => {
  task?.controller.abort();
  shutCamera();
  pilot.dispose();
  clearClip();
  fileInput.value = '';
});
window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  pilot = createV8Pilot(pilotOptions);
  status.textContent = 'Επιλέξτε αρχείο ή ανοίξτε την κάμερα.';
  error.textContent = '';
  update();
});
update();
