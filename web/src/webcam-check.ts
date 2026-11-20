const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const consent = byId<HTMLInputElement>('consent');
const open = byId<HTMLButtonElement>('open-camera');
const close = byId<HTMLButtonElement>('close-camera');
const record = byId<HTMLButtonElement>('record');
const discard = byId<HTMLButtonElement>('discard');
const save = byId<HTMLButtonElement>('save');
const kind = byId<HTMLSelectElement>('trial-kind');
const confirmed = byId<HTMLInputElement>('confirm-label');
const live = byId<HTMLVideoElement>('live-video');
const review = byId<HTMLVideoElement>('review-video');
const status = byId('status');
const error = byId('error');
const countdown = byId('countdown');
const directions: Record<string, string> = {
  idle: 'Κρατήστε φυσική στάση και τα χέρια ήρεμα. Μην εκτελέσετε σκόπιμη νοηματική.',
  ordinary_hand_motion: 'Κάντε μια καθημερινή κίνηση, για παράδειγμα διορθώστε απαλά τα ρούχα σας. Μην μιμηθείτε κάποιο νόημα.',
  hands_out_of_view: 'Κρατήστε τα χέρια εκτός κάδρου, με το πρόσωπο και τους ώμους μέσα στο κάδρο.',
  position_or_light_change: 'Μετακινηθείτε λίγο στο κάδρο ή αλλάξτε τον φωτισμό, χωρίς σκόπιμη νοηματική.',
};
let stream: MediaStream | undefined;
let recorder: MediaRecorder | undefined;
let busy = false, endpointReady = false, cancelled = false;
let pending: { blob: Blob; metadata: Record<string, unknown> } | undefined;
let previewUrl: string | undefined;

function update() {
  open.disabled = !consent.checked || !!stream || busy || !endpointReady;
  close.disabled = !stream;
  record.disabled = !stream || !consent.checked || busy || !!pending || !endpointReady;
  discard.disabled = !pending || busy;
  save.disabled = !pending || !confirmed.checked || !consent.checked || busy;
  kind.disabled = busy || !!pending;
  consent.disabled = busy;
  byId('instructions').textContent = directions[kind.value];
}
function clearPending() {
  pending = undefined;
  confirmed.checked = false;
  review.removeAttribute('src');
  review.load();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = undefined;
  byId('review-area').hidden = true;
  byId('confirm-area').hidden = true;
  update();
}
function stopCamera() {
  cancelled = true;
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  live.srcObject = null;
  update();
}
const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const message = (failure: unknown) => failure instanceof Error ? failure.message : String(failure);
consent.addEventListener('change', () => { if (!consent.checked) stopCamera(); update(); });
kind.addEventListener('change', update);
confirmed.addEventListener('change', update);
close.addEventListener('click', stopCamera);
discard.addEventListener('click', () => { clearPending(); status.textContent = 'Η μη αποθηκευμένη καταγραφή απορρίφθηκε.'; });

open.addEventListener('click', async () => {
  if (!consent.checked) return;
  busy = true; update(); error.textContent = '';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' }, audio: false });
    live.srcObject = stream;
    await live.play();
    status.textContent = 'Η κάμερα είναι έτοιμη. Επιλέξτε κατάσταση και πατήστε καταγραφή.';
  } catch (failure) { stopCamera(); error.textContent = `Δεν άνοιξε η κάμερα: ${message(failure)}`; }
  finally { busy = false; update(); }
});

record.addEventListener('click', async () => {
  if (!stream || !consent.checked || busy) return;
  busy = true; cancelled = false; update(); error.textContent = '';
  const trialKind = kind.value;
  try {
    for (let remaining = 3; remaining > 0; remaining--) {
      countdown.textContent = `Έναρξη σε ${remaining}…`;
      await delay(1000);
      if (cancelled || !stream) throw new DOMException('Η καταγραφή ακυρώθηκε.', 'AbortError');
    }
    const activeStream = stream;
    const mimeType = ['video/webm;codecs=vp8', 'video/webm'].find((mime) => MediaRecorder.isTypeSupported(mime));
    if (!mimeType) throw new Error('Χρειάζεται browser που υποστηρίζει καταγραφή WebM, όπως Chrome.');
    const chunks: BlobPart[] = [];
    const frameTimes: number[] = [];
    let firstMediaTime: number | undefined;
    let frameCallback = 0;
    let startTime = 0;
    let startedAt = '';
    let stopTimer: ReturnType<typeof setTimeout> | undefined;
    recorder = new MediaRecorder(activeStream, { mimeType, videoBitsPerSecond: 1_500_000 });
    const settings = activeStream.getVideoTracks()[0].getSettings();
    const trackFrame: VideoFrameRequestCallback = (_now, metadata) => {
      if (recorder?.state !== 'recording') return;
      firstMediaTime ??= metadata.mediaTime;
      frameTimes.push((metadata.mediaTime - firstMediaTime) * 1000);
      frameCallback = live.requestVideoFrameCallback(trackFrame);
    };
    const captured = await new Promise<{ blob: Blob; duration: number; finishedAt: string }>((resolve, reject) => {
      recorder!.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder!.onerror = () => reject(new Error('Αποτυχία εγγραφής βίντεο.'));
      recorder!.onstart = () => {
        startTime = performance.now(); startedAt = new Date().toISOString();
        countdown.textContent = 'Καταγραφή τώρα · 1,5 δευτερόλεπτο';
        if ('requestVideoFrameCallback' in live) frameCallback = live.requestVideoFrameCallback(trackFrame);
        stopTimer = setTimeout(() => { if (recorder?.state === 'recording') recorder.stop(); }, 1500);
      };
      recorder!.onstop = () => {
        clearTimeout(stopTimer);
        if (frameCallback) live.cancelVideoFrameCallback(frameCallback);
        if (cancelled) return reject(new DOMException('Η καταγραφή ακυρώθηκε.', 'AbortError'));
        resolve({ blob: new Blob(chunks, { type: mimeType }), duration: performance.now() - startTime, finishedAt: new Date().toISOString() });
      };
      recorder!.start();
    });
    if (!captured.blob.size) throw new Error('Η καταγραφή δεν περιέχει βίντεο. Δοκιμάστε ξανά.');
    pending = { blob: captured.blob, metadata: {
      trialKind, consentLocalRecording: true, noIntentionalSignLanguage: true,
      requestedDurationMs: 1500, actualDurationMs: captured.duration,
      captureStartedAtUtc: startedAt, captureFinishedAtUtc: captured.finishedAt,
      observedFrameTimestampsMs: frameTimes,
      observedFps: frameTimes.length > 1 && frameTimes.at(-1)! > 0 ? (frameTimes.length - 1) * 1000 / frameTimes.at(-1)! : null,
      trackSettings: { width: settings.width, height: settings.height, frameRate: settings.frameRate, facingMode: settings.facingMode },
      navigator: { userAgent: navigator.userAgent, platform: navigator.platform },
    } };
    previewUrl = URL.createObjectURL(captured.blob);
    review.src = previewUrl;
    byId('review-area').hidden = false;
    byId('confirm-area').hidden = false;
    status.textContent = 'Δείτε την καταγραφή, επιβεβαιώστε την κατάσταση και αποθηκεύστε μόνο αν συμφωνείτε.';
  } catch (failure) { error.textContent = message(failure); }
  finally { busy = false; recorder = undefined; countdown.textContent = ''; update(); }
});

save.addEventListener('click', async () => {
  if (!pending || !confirmed.checked || !consent.checked || busy) return;
  busy = true; update(); error.textContent = '';
  try {
    const captured = pending;
    const videoBase64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(captured.blob);
    });
    const response = await fetch('/__signbridge/webcam-capture', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ metadata: { ...captured.metadata, labelConfirmed: true }, mimeType: captured.blob.type, videoBase64 }) });
    const saved = await response.json();
    if (!response.ok || !saved.saved) throw new Error(saved.error ?? 'Η τοπική αποθήκευση απέτυχε.');
    byId('no-saved')?.remove();
    const item = document.createElement('li');
    item.textContent = `${kind.options[kind.selectedIndex].text}: αποθηκεύτηκε τοπικά (${saved.id}).`;
    byId('saved-list').append(item);
    clearPending();
    status.textContent = 'Αποθηκεύτηκε. Μπορείτε να επιλέξετε την επόμενη κατάσταση ή να κλείσετε την κάμερα.';
  } catch (failure) { error.textContent = message(failure); }
  finally { busy = false; update(); }
});

window.addEventListener('beforeunload', stopCamera);
void fetch('/__signbridge/webcam-capture/status').then(async (response) => {
  const value = await response.json();
  if (!response.ok || value.ready !== true) throw new Error('Η τοπική υπηρεσία αποθήκευσης δεν είναι διαθέσιμη.');
  endpointReady = true;
  status.textContent = 'Έτοιμο για προαιρετική τοπική καταγραφή. Δώστε άδεια μόνο αν θέλετε να συμμετάσχετε.';
}).catch((failure) => { error.textContent = message(failure); }).finally(update);
update();
export {};
