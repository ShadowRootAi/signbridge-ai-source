import { useEffect, useRef, useState } from 'react';
import pilotConfigUrl from '../../../artifacts/v8/browser/model-config.json?url';
import handModelUrl from '../../../research/cache/mediapipe/hand_landmarker-float16-1.task?url';
import { createV8Pilot } from '../v8-pilot-core';
import { judgeDemo, localAssistantEnabled, localOpenHands, personalTeaching, publicOpenHands, sequencePilot, useOpenHands } from '../recognition-release';
import { useOpenHandsVocabularyLabel } from '../use-openhands-vocabulary';
import type { LandmarkFrame } from '../features';
import type { OpenHandsResult } from '../openhands-pipeline';
import type { TeachingSample } from '../personal-gestures';
import type { PipelineProgress } from '../pipeline-contract';
import { DecisionTrace } from './DecisionTrace';
import { Icon } from './icons';
import { displaySign } from './evidence';
import { normalizedPhrase, useCatalog } from './Library';
import { LocalAssistant } from './LocalAssistant';
import './recognition.css';

type Phase = 'opening' | 'countdown' | 'recording' | 'loading' | 'analyzing';
type Clip = { label: string; personalSource?: Blob };
type Prediction = {
  label: string | null;
  needsFraming: boolean;
  source: 'model' | 'personal';
  trace?: OpenHandsResult;
  teachingSample?: TeachingSample;
};
const options = { v8ConfigUrl: pilotConfigUrl, handModelUrl };
const captureDuration = useOpenHands ? 1600 : 3000;
const judgeSamples = [
  { label: 'Εντάξει', url: '/library/entaxei.mp4' },
  { label: 'Παγωμένο καρέ', url: '/__signbridge/judge/frozen.mp4' },
  { label: 'Μαύρη εικόνα', url: '/__signbridge/judge/no-tracking.mp4' },
  { label: 'Εκτός λεξιλογίου', url: '/__signbridge/judge/unsupported.mp4' },
];

function delay(milliseconds: number, signal: AbortSignal) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, milliseconds);
    function abort() {
      window.clearTimeout(timer);
      reject(signal.reason);
    }
    signal.addEventListener('abort', abort, { once: true });
  });
}

function mediaEvent(video: HTMLVideoElement, name: 'loadeddata' | 'seeked', signal: AbortSignal, start: () => void) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const ready = () => finish();
    const failed = () => finish(new Error('Το βίντεο δεν ανοίγει. Δοκίμασε MP4 ή WebM.'));
    const abort = () => finish(signal.reason);
    const timer = window.setTimeout(() => finish(new Error('Η φόρτωση του βίντεο καθυστέρησε. Δοκίμασε ξανά.')), 15000);
    function finish(error?: unknown) {
      window.clearTimeout(timer);
      video.removeEventListener(name, ready);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', abort);
      if (error) reject(error); else resolve();
    }
    video.addEventListener(name, ready, { once: true });
    video.addEventListener('error', failed, { once: true });
    signal.addEventListener('abort', abort, { once: true });
    try { start(); } catch (error) { finish(error); }
  });
}

function recordClip(recorder: MediaRecorder, signal: AbortSignal) {
  signal.throwIfAborted();
  return new Promise<Blob>((resolve, reject) => {
    const parts: Blob[] = [];
    let stopTimer: number | undefined;
    let failTimer: number | undefined;
    const stop = () => { if (recorder.state !== 'inactive') recorder.stop(); };
    const abort = () => finish(signal.reason);
    function finish(error?: unknown) {
      window.clearTimeout(stopTimer);
      window.clearTimeout(failTimer);
      signal.removeEventListener('abort', abort);
      recorder.ondataavailable = recorder.onerror = recorder.onstop = null;
      stop();
      if (error) reject(error); else resolve(new Blob(parts, { type: recorder.mimeType }));
    }
    recorder.ondataavailable = event => { if (event.data.size) parts.push(event.data); };
    recorder.onerror = () => finish(new Error('Η καταγραφή διακόπηκε. Δοκίμασε ξανά.'));
    recorder.onstop = () => finish();
    signal.addEventListener('abort', abort, { once: true });
    try {
      recorder.start();
      stopTimer = window.setTimeout(stop, captureDuration);
      failTimer = window.setTimeout(() => finish(new Error('Η κάμερα δεν ολοκλήρωσε την εγγραφή.')), 8000);
    } catch (error) { finish(error); }
  });
}

function explain(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError') return 'Δεν δόθηκε άδεια κάμερας. Μπορείς να την επιτρέψεις από τις ρυθμίσεις του browser.';
    if (error.name === 'NotFoundError') return 'Δεν βρέθηκε κάμερα. Μπορείς να χρησιμοποιήσεις ένα αρχείο βίντεο.';
    if (error.name === 'NotReadableError') return 'Η κάμερα δεν είναι διαθέσιμη. Έλεγξε μήπως τη χρησιμοποιεί άλλη εφαρμογή.';
    return 'Η ενέργεια δεν ολοκληρώθηκε. Δοκίμασε ξανά.';
  }
  return error instanceof Error ? error.message : 'Η ενέργεια δεν ολοκληρώθηκε. Δοκίμασε ξανά.';
}

function releaseStream(stream: MediaStream | undefined) {
  stream?.getTracks().forEach(track => track.stop());
}

export function RecognitionPage({ mode, onTeach }: { mode: 'camera' | 'file'; onTeach?: (sample: TeachingSample) => void }) {
  const cameraMode = mode === 'camera';
  const { entries, state: catalogState } = useCatalog(!cameraMode);
  const examples = entries.filter(entry => {
    if (entry.contentKind === 'phrase' || typeof entry.annotationLabel !== 'string') return false;
    const label = normalizedPhrase(entry.annotationLabel);
    return label !== '' && label === normalizedPhrase(entry.phrase);
  }).slice(0, 6);
  const [consent, setConsent] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [phase, setPhase] = useState<Phase>();
  const [countdown, setCountdown] = useState<number>();
  const [clip, setClip] = useState<Clip>();
  const [result, setResult] = useState<Prediction>();
  const [analysisProgress, setAnalysisProgress] = useState<PipelineProgress>();
  const [error, setError] = useState('');
  const [copyStatus, setCopyStatus] = useState('');
  const vocabulary = useOpenHandsVocabularyLabel();
  const live = useRef<HTMLVideoElement>(null);
  const preview = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | undefined>(undefined);
  const recorder = useRef<MediaRecorder | undefined>(undefined);
  const objectUrl = useRef<string | undefined>(undefined);
  const task = useRef<AbortController | undefined>(undefined);
  const analyzer = useRef<ReturnType<typeof createV8Pilot> | undefined>(undefined);
  const openHands = useRef<typeof import('../openhands-pipeline') | undefined>(undefined);

  useEffect(() => {
    const current = useOpenHands ? undefined : createV8Pilot(options);
    analyzer.current = current;
    return () => {
      const pending = task.current;
      task.current = undefined;
      pending?.abort();
      releaseStream(stream.current);
      stream.current = undefined;
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = undefined;
      current?.dispose();
      openHands.current?.cancelOpenHands();
      openHands.current = undefined;
      analyzer.current = undefined;
    };
  }, []);

  async function perform(next: Phase, action: (signal: AbortSignal) => Promise<void>) {
    if (task.current) return;
    const controller = new AbortController();
    task.current = controller;
    setPhase(next);
    setError('');
    try {
      await action(controller.signal);
    } catch (failure) {
      if (!controller.signal.aborted) {
        if (localOpenHands) console.error('SignBridge OpenHands', failure);
        setError(next === 'analyzing' ? 'Η ανάλυση δεν ολοκληρώθηκε. Δοκίμασε ξανά.' : explain(failure));
      }
    } finally {
      if (task.current === controller) {
        task.current = undefined;
        setPhase(undefined);
        setCountdown(undefined);
      }
    }
  }

  function clearClip() {
    setClip(undefined);
    setResult(undefined);
    setAnalysisProgress(undefined);
    setCopyStatus('');
    preview.current?.removeAttribute('src');
    preview.current?.load();
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = undefined;
  }

  function closeCamera() {
    releaseStream(stream.current);
    stream.current = undefined;
    if (live.current) live.current.srcObject = null;
    setCameraActive(false);
  }

  async function openCamera(signal: AbortSignal) {
    if (!consent) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Η κάμερα χρειάζεται ασφαλή σύνδεση HTTPS ή localhost.');
    const media = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: { ideal: 'user' } },
      audio: false,
    });
    if (signal.aborted || !live.current) {
      releaseStream(media);
      signal.throwIfAborted();
      return;
    }
    stream.current = media;
    live.current.srcObject = media;
    try {
      await live.current.play();
      signal.throwIfAborted();
      clearClip();
      setCameraActive(true);
      for (const track of media.getVideoTracks()) track.addEventListener('ended', () => {
        if (stream.current !== media) return;
        task.current?.abort();
        closeCamera();
        setError('Η κάμερα έκλεισε. Μπορείς να την ανοίξεις ξανά.');
      }, { once: true });
    } catch (failure) {
      if (stream.current === media) closeCamera();
      throw failure;
    }
  }

  async function prepareClip(source: Blob, label: string, userProvided: boolean, signal: AbortSignal) {
    clearClip();
    const video = preview.current;
    if (!video) throw new Error('Η προεπισκόπηση βίντεο δεν είναι διαθέσιμη.');
    if (!source.size || source.size > 40_000_000) {
      throw new Error('Επίλεξε βίντεο μεγέθους έως 40 MB.');
    }
    const url = URL.createObjectURL(source);
    objectUrl.current = url;
    try {
      await mediaEvent(video, 'loadeddata', signal, () => { video.src = url; video.load(); });
      // Chrome's recorded WebM may omit its duration until the browser seeks to its end.
      if (!Number.isFinite(video.duration)) {
        await mediaEvent(video, 'seeked', signal, () => { video.currentTime = 1e8; });
        await mediaEvent(video, 'seeked', signal, () => { video.currentTime = 0; });
      }
      if (!Number.isFinite(video.duration) || video.duration <= 0 || video.duration > 15 || !video.videoWidth || !video.videoHeight) {
        throw new Error('Επίλεξε αναγνώσιμο βίντεο διάρκειας έως 15 δευτερολέπτων.');
      }
      signal.throwIfAborted();
      setClip({ label, personalSource: personalTeaching && userProvided ? source : undefined });
    } catch (failure) {
      if (!signal.aborted) clearClip();
      throw failure;
    }
  }

  async function loadExample(url: string, label: string, signal: AbortSignal) {
    // Decode a local blob so frame seeks work on hosts without byte-range support.
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error('Το παράδειγμα δεν φορτώθηκε. Δοκίμασε ξανά.');
    const video = await response.blob();
    signal.throwIfAborted();
    await prepareClip(video, `Παράδειγμα · ${label}`, false, signal);
  }

  async function capture(signal: AbortSignal) {
    const current = stream.current;
    if (!current) return;
    clearClip();
    try {
      if (typeof MediaRecorder === 'undefined') throw new Error('Ο browser δεν υποστηρίζει εγγραφή βίντεο.');
      const mimeType = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
      if (!mimeType) throw new Error('Δοκίμασε Chrome ή άλλο browser με υποστήριξη εγγραφής MP4 ή WebM.');
      for (let seconds = 3; seconds > 0; seconds--) {
        setCountdown(seconds);
        await delay(1000, signal);
      }
      signal.throwIfAborted();
      const active = new MediaRecorder(current, { mimeType, videoBitsPerSecond: 2_000_000 });
      recorder.current = active;
      setCountdown(undefined);
      setPhase('recording');
      const captured = await recordClip(active, signal);
      signal.throwIfAborted();
      closeCamera();
      setPhase('loading');
      await prepareClip(captured, 'Καταγραφή κάμερας', true, signal);
    } finally {
      recorder.current = undefined;
      if (stream.current === current) closeCamera();
    }
  }

  async function analyze(signal: AbortSignal) {
    const video = preview.current;
    if (!clip || !video) return;
    setResult(undefined);
    setAnalysisProgress(undefined);
    setCopyStatus('');
    if (useOpenHands) {
      const engine = openHands.current ?? await import('../openhands-pipeline');
      signal.throwIfAborted();
      openHands.current = engine;
      let frames: LandmarkFrame[] | undefined;
      const retainFrames = clip.personalSource
        ? (captured: LandmarkFrame[]) => { frames = captured; } : undefined;
      const [output, policy] = await Promise.all([
        engine.analyzeOpenHandsVideo(video, signal, judgeDemo ? setAnalysisProgress : undefined, !cameraMode, retainFrames),
        engine.getOpenHandsPolicy(),
      ]);
      signal.throwIfAborted();
      const modelLabel = output.label ? (policy.displayLabels?.[output.label] ?? output.label) : null;
      let teachingSample: TeachingSample | undefined;
      let personalLabel: string | null = null;
      if (frames) {
        try {
          const personal = await import('../personal-gestures');
          teachingSample = await personal.teachingSampleFromFrames(frames, video.duration, clip.personalSource) ?? undefined;
          if (teachingSample) personalLabel = personal.recognizePersonalSign(teachingSample, personal.readPersonalSigns());
        } catch {
          // A local vocabulary failure should not interrupt the base model result.
        }
        signal.throwIfAborted();
      }
      const usePersonal = personalLabel !== null;
      setResult({
        label: usePersonal ? personalLabel : modelLabel,
        needsFraming: !usePersonal && output.rejectionCode === 'tracking',
        source: usePersonal ? 'personal' : 'model',
        trace: judgeDemo ? output : undefined,
        teachingSample,
      });
      return;
    }
    if (!analyzer.current) return;
    const output = await analyzer.current.analyzeVideo(video, undefined, signal);
    signal.throwIfAborted();
    setResult({ label: output.v8.accepted ? output.v8.label : null, needsFraming: !output.classifierExecuted, source: 'model' });
  }

  async function copy(label: string) {
    try {
      await navigator.clipboard.writeText(label);
      setCopyStatus('Το κείμενο αντιγράφηκε.');
    } catch {
      setCopyStatus('Μπορείς να επιλέξεις και να αντιγράψεις τη λέξη χειροκίνητα.');
    }
  }

  const busy = phase !== undefined;
  const resultText = result?.label ? (result.source === 'personal' ? result.label : displaySign(result.label)) : null;
  const teachingSample = result?.teachingSample;
  return <div className="recognition-page">
    <div className="page-heading compact">
      <span className="eyebrow">{cameraMode ? 'ΝΟΗΜΑ → ΚΕΙΜΕΝΟ' : 'ΔΟΚΙΜΗ ΜΕ ΒΙΝΤΕΟ'}</span>
      <h1>{cameraMode ? 'Νόημα σε κείμενο' : 'Αναγνώριση από βίντεο'}</h1>
      <p>{cameraMode ? 'Κατάγραψε ένα νόημα και δες την πρόταση του SignBridge.' : 'Άνοιξε δικό σου βίντεο ή διάλεξε ένα παράδειγμα από τη βιβλιοθήκη.'}</p>
    </div>

    <div className="recognition-workspace">
      <section className="surface recognition-capture" aria-labelledby="recognition-source">
        <div className="recognition-card-head">
          <h2 id="recognition-source">{cameraMode ? 'Λήψη κάμερας' : 'Βίντεο'}</h2>
          <span className="recognition-private"><Icon name="shield" size={16} />Τοπική επεξεργασία</span>
        </div>
        <div className="recognition-stage">
          {cameraMode && <video id="live-camera" ref={live} hidden={!cameraActive} playsInline muted aria-label="Ζωντανή κάμερα" />}
          <video id="recognition-video" ref={preview} hidden={!clip} controls={!busy && !!clip} playsInline muted preload="auto"
            onPlay={event => { if (task.current) event.currentTarget.pause(); }} aria-label="Προεπισκόπηση βίντεο" />
          {!cameraActive && !clip && <div className="recognition-placeholder"><Icon name={cameraMode ? 'camera' : 'file'} size={34} /><strong>{phase === 'loading' ? 'Φόρτωση βίντεο…' : cameraMode ? 'Έτοιμο για λήψη' : 'Διάλεξε ένα βίντεο'}</strong></div>}
          {countdown !== undefined && <div className="recognition-countdown" role="status" aria-live="assertive"><strong>{countdown}</strong><span>Ετοιμάσου</span></div>}
          {phase === 'recording' && <span className="recognition-recording" role="status">Γίνεται καταγραφή</span>}
          {phase === 'analyzing' && <span className="recognition-scan" aria-hidden="true" />}
        </div>
        {clip && <p className="recognition-clip-name">{clip.label}</p>}
        <div className="recognition-actions">
          {cameraMode && !cameraActive && <button className="button" type="button" disabled={busy || !consent} onClick={() => void perform('opening', openCamera)}>
            <Icon name="camera" size={18} />{phase === 'opening' ? 'Άνοιγμα…' : 'Άνοιγμα κάμερας'}
          </button>}
          {cameraMode && cameraActive && phase === 'recording' && <button className="button" type="button" onClick={() => {
            if (recorder.current?.state === 'recording') recorder.current.stop();
          }}>Τέλος καταγραφής</button>}
          {cameraMode && cameraActive && phase !== 'recording' && <button className="button" type="button" disabled={busy} onClick={() => void perform('countdown', capture)}>Καταγραφή</button>}
          {cameraMode && cameraActive && <button className="button secondary" type="button" onClick={() => {
            task.current?.abort();
            closeCamera();
          }}>Κλείσιμο κάμερας</button>}
          {!cameraMode && <label className="button secondary recognition-upload">
            <Icon name="upload" size={18} />Επιλογή βίντεο
            <input id="video-file" type="file" accept=".mp4,.webm,video/mp4,video/webm" aria-label="Επιλογή βίντεο" disabled={busy}
              onChange={event => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = '';
                if (file) void perform('loading', signal => prepareClip(file, file.name, true, signal));
              }} />
          </label>}
          {clip && <button className="button" type="button" disabled={busy} onClick={() => void perform('analyzing', analyze)}>Ανάλυση βίντεο</button>}
          {clip && !busy && <button className="button quiet" type="button" onClick={() => { clearClip(); setError(''); }}>Καθαρισμός</button>}
          {phase === 'analyzing' && <button className="button secondary" type="button" onClick={() => task.current?.abort()}>Ακύρωση</button>}
        </div>
        {cameraMode && !cameraActive && !clip && <label className="recognition-consent">
          <input type="checkbox" checked={consent} disabled={busy} onChange={event => setConsent(event.currentTarget.checked)} />
          Δίνω άδεια για καταγραφή εικόνας.
        </label>}
        {cameraMode && cameraActive && <p className="microcopy recognition-hint">Η λήψη διαρκεί {useOpenHands ? '1,6' : 'έως 3'} δευτερόλεπτα. Η κάμερα κλείνει μετά την καταγραφή.</p>}
        {error && <p className="error-message recognition-error" role="alert">{error}</p>}
        {!cameraMode && <div className="recognition-examples">
          <h3>Έτοιμα παραδείγματα</h3>
          {catalogState === 'loading' ? <p className="microcopy" role="status">Φόρτωση παραδειγμάτων…</p> : examples.length ? <div className="recognition-actions">
            {examples.map(entry => <button className="phrase-chip" key={entry.id} type="button" disabled={busy}
              onClick={() => void perform('loading', signal => loadExample(entry.videoUrl, entry.phrase, signal))}>
              <Icon name="play" size={15} />{entry.phrase}</button>)}
          </div> : <p className="microcopy">Μπορείς να συνεχίσεις με δικό σου βίντεο.</p>}
        </div>}
        {judgeDemo && !cameraMode && <div className="recognition-examples" aria-labelledby="judge-samples-title">
          <h3 id="judge-samples-title">Δείγματα επίδειξης</h3>
          <div className="recognition-actions">{judgeSamples.map(sample => <button key={sample.url} type="button"
            className="phrase-chip" disabled={busy} onClick={() => void perform('loading', signal => loadExample(sample.url, sample.label, signal))}>
            <Icon name="play" size={15} />{sample.label}</button>)}</div>
        </div>}
      </section>

      <section className="surface recognition-message" aria-labelledby="recognition-message-title">
        <div className="recognition-card-head"><h2 id="recognition-message-title">Το μήνυμά σου</h2></div>
        <div className="recognition-response" aria-live="polite">
          {result ? <>
            <output id="recognition-result" data-accepted={result.label !== null} data-source={personalTeaching ? result.source : undefined}>{resultText ?? 'Δεν αναγνωρίστηκε'}</output>
            <p>{result.source === 'personal' ? 'Αντιστοιχίστηκε με νόημα από το προσωπικό σου λεξιλόγιο.' : result.label ? 'Έλεγξε αν η πρόταση ταιριάζει με αυτό που ήθελες να πεις.' : result.needsFraming ? 'Κράτησε τα χέρια και τους ώμους ορατά σε όλη τη λήψη.' : 'Δοκίμασε ξανά με ένα από τα διαθέσιμα νοήματα.'}</p>
            {resultText && <button className="button secondary" type="button" onClick={() => void copy(resultText)}>Αντιγραφή κειμένου</button>}
            {onTeach && teachingSample && <button className="button secondary" type="button" disabled={busy || cameraActive}
              onClick={() => onTeach(teachingSample)}>Διόρθωσε στο Teach Me</button>}
            {copyStatus && <p className="microcopy" role="status">{copyStatus}</p>}
          </> : <>
            <span className="recognition-response-icon"><Icon name={phase === 'analyzing' ? 'spark' : 'text'} size={29} /></span>
            <p>{phase === 'analyzing' ? 'Αναλύουμε το βίντεο…' : 'Το αποτέλεσμα θα εμφανιστεί εδώ.'}</p>
          </>}
        </div>
        {judgeDemo && <DecisionTrace result={result?.trace} progress={phase === 'analyzing' ? analysisProgress : undefined} />}
        <p className="recognition-reminder"><Icon name="info" size={16} />Ένα νόημα κάθε φορά, με χέρια και ώμους στο κάδρο.</p>
        {localAssistantEnabled && <LocalAssistant disabled={busy || cameraActive} recognition={result
          ? (resultText ? { accepted: true, label: resultText } : { accepted: false }) : undefined} />}
      </section>
    </div>
    <div className="recognition-support">
      <span>Διαθέσιμα νοήματα</span>
      <ul aria-label="Διαθέσιμα νοήματα">{localOpenHands
        ? <li><a href="/openhands-webcam.html">Δοκιμή 120 νοημάτων</a></li>
        : publicOpenHands ? <li>{vocabulary}</li>
          : <><li>Γεια</li><li>Ευχαριστώ</li><li>Εντάξει</li></>}</ul>
      <a className="text-link" href={cameraMode ? '#/try' : '#/sign'}>{cameraMode ? 'Δοκίμασε με βίντεο' : 'Χρησιμοποίησε κάμερα'}<Icon name="arrow" size={17} /></a>
    </div>
    {sequencePilot && !cameraMode && <div className="recognition-actions">
      <a className="text-link" href="#/sequence">Δοκίμασε συνεχές βίντεο <Icon name="arrow" size={16} /></a>
      <a className="text-link" href="#/teach">Teach Me <Icon name="arrow" size={16} /></a>
    </div>}
  </div>;
}
