import { useEffect, useRef, useState } from 'react';
import type { ContinuousVideoResult } from '../continuous-pipeline';
import { loadLocalVideo, localVideoUrl } from '../local-video';
import { useCatalog } from './Library';
import { Icon } from './icons';
import './recognition.css';
import './sequence.css';

const seconds = (value: number) => (value / 1000).toLocaleString('el-GR', { maximumFractionDigits: 1 });
const maximumVideoSeconds = 90;
const maximumExcerptSeconds = 12;
const phraseExamples = new Set(['kalo-apogevma', 'poio-provlima', 'den-egine-tipota', 'thelo-kataggelia']);
type Turn = { startMs: number; endMs: number; text: string };

export function SequencePage() {
  const { entries } = useCatalog();
  const phrases = entries.filter(entry => phraseExamples.has(entry.id) && entry.contentKind === 'phrase');
  const video = useRef<HTMLVideoElement>(null);
  const objectUrl = useRef<string>(undefined);
  const controller = useRef<AbortController>(undefined);
  const cancelAnalysis = useRef<() => void>(undefined);
  const [clip, setClip] = useState('');
  const [sourceSeconds, setSourceSeconds] = useState(0);
  const [rangeStart, setRangeStart] = useState('0');
  const [rangeEnd, setRangeEnd] = useState('');
  const [progress, setProgress] = useState('');
  const [result, setResult] = useState<ContinuousVideoResult>();
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [transcriptStatus, setTranscriptStatus] = useState('');
  const [error, setError] = useState('');
  const startMs = Number(rangeStart) * 1000;
  const endMs = Number(rangeEnd) * 1000;
  const validRange = rangeStart !== '' && rangeEnd !== '' && startMs >= 0 && endMs <= sourceSeconds * 1000 + 1
    && endMs > startMs && endMs - startMs <= maximumExcerptSeconds * 1000 + .0001;

  useEffect(() => () => {
    controller.current?.abort();
    cancelAnalysis.current?.();
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
  }, []);

  async function run(message: string, action: (signal: AbortSignal) => Promise<void>) {
    if (controller.current) return;
    const task = new AbortController();
    controller.current = task;
    setProgress(message);
    setError('');
    try { await action(task.signal); }
    catch (failure) {
      if (!task.signal.aborted) setError(failure instanceof Error ? failure.message : 'Η ανάλυση διακόπηκε.');
    } finally {
      controller.current = undefined;
      if (!task.signal.aborted) setProgress('');
    }
  }

  async function select(source: (signal: AbortSignal) => Promise<Blob>, title: string) {
    await run('Άνοιγμα βίντεο…', async signal => {
      const player = video.current!;
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      player.removeAttribute('src');
      player.load();
      setClip('');
      setResult(undefined);
      setDraft('');
      setTurns([]);
      setTranscriptStatus('');
      const url = localVideoUrl(await source(signal));
      objectUrl.current = url;
      await loadLocalVideo(player, url, signal);
      if (player.duration > maximumVideoSeconds) {
        throw new Error('Επίλεξε βίντεο έως ' + maximumVideoSeconds + ' δευτερόλεπτα.');
      }
      setSourceSeconds(player.duration);
      setRangeStart('0');
      setRangeEnd(String(Math.min(player.duration, maximumExcerptSeconds)));
      setClip(title);
    });
  }

  function analyze() {
    void run('Προετοιμασία ανάλυσης…', async signal => {
      setResult(undefined);
      setDraft('');
      const { analyzeContinuousVideo, cancelContinuous } = await import('../continuous-pipeline');
      cancelAnalysis.current = cancelContinuous;
      const prediction = await analyzeContinuousVideo(video.current!, signal, value => setProgress(value.message), { startMs, endMs });
      signal.throwIfAborted();
      setResult(prediction);
      setDraft(prediction.text ?? '');
    });
  }

  function changeRange(part: 'start' | 'end', value: string) {
    if (part === 'start') setRangeStart(value);
    else setRangeEnd(value);
    setResult(undefined);
    setDraft('');
  }

  function confirmTurn() {
    if (!result || !draft.trim()) return;
    const turn = { startMs: result.startMs, endMs: result.startMs + result.durationMs, text: draft.trim() };
    const otherTurns = turns.filter(item => item.startMs !== turn.startMs || item.endMs !== turn.endMs);
    if (otherTurns.some(item => item.startMs < turn.endMs - .01 && item.endMs > turn.startMs + .01)) {
      setTranscriptStatus('Το απόσπασμα επικαλύπτει υπάρχουσα ατάκα. Άλλαξε τα όρια ή αφαίρεσε την προηγούμενη.');
      return;
    }
    if (otherTurns.length >= 32) {
      setTranscriptStatus('Μπορείς να κρατήσεις έως 32 ατάκες για αυτό το βίντεο.');
      return;
    }
    setTurns([...otherTurns, turn].sort((a, b) => a.startMs - b.startMs));
    setTranscriptStatus('Η ατάκα προστέθηκε στο κείμενο.');
  }

  async function copyTranscript() {
    try {
      await navigator.clipboard.writeText(turns.map(turn => turn.text).join('\n'));
      setTranscriptStatus('Το κείμενο αντιγράφηκε.');
    } catch {
      setTranscriptStatus('Μπορείς να επιλέξεις και να αντιγράψεις το κείμενο χειροκίνητα.');
    }
  }

  return <div className="recognition-page">
    <header className="page-heading">
      <p className="eyebrow">Νοηματική σε κείμενο · Δοκιμή</p>
      <h1>Συνεχές βίντεο</h1>
      <p>Επίλεξε ένα απόσπασμα, δες τι ξεχώρισε το μοντέλο και κράτησε τις ατάκες που επιβεβαιώνεις.</p>
    </header>
    <div className="recognition-workspace">
      <section className="surface recognition-capture" aria-labelledby="sequence-source">
        <div className="recognition-card-head"><h2 id="sequence-source">Βίντεο</h2><span className="recognition-private"><Icon name="shield" size={14} /> Στη συσκευή σου</span></div>
        <div className="recognition-stage">
          <video id="sequence-video" ref={video} controls={!progress} playsInline muted hidden={!clip}
            onPlay={event => { if (controller.current) event.currentTarget.pause(); }} />
          {!clip && <div className="recognition-placeholder"><Icon name="play" size={32} /><span>Επίλεξε βίντεο έως 90 δευτερόλεπτα</span></div>}
        </div>
        {clip && <p className="recognition-clip-name">{clip}</p>}
        <div className="recognition-actions">
          <label className="button secondary recognition-upload"><Icon name="upload" size={18} />Επιλογή βίντεο
            <input aria-label="Επιλογή συνεχούς βίντεο" type="file" accept="video/mp4,video/webm" disabled={!!progress} onChange={event => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = '';
              if (file) void select(async () => file, file.name);
            }} />
          </label>
          <button className="button" type="button" disabled={!clip || !validRange || !!progress} onClick={analyze}>Ανάλυση φράσης</button>
        </div>
        {clip && <details className="recognition-examples" open={sourceSeconds > maximumExcerptSeconds || undefined}>
          <summary>Χρονικά όρια αποσπάσματος</summary>
          <p className="microcopy">Διάλεξε έως 12 δευτερόλεπτα. Μπορείς να ορίσεις τα όρια από το τρέχον σημείο του βίντεο.</p>
          <div className="sequence-range">
            <label>Αρχή σε δευτερόλεπτα <input type="number" min={0} max={sourceSeconds} step="any" value={rangeStart}
              disabled={!!progress} onChange={event => changeRange('start', event.currentTarget.value)} /></label>
            <label>Τέλος σε δευτερόλεπτα <input type="number" min={0} max={sourceSeconds} step="any" value={rangeEnd}
              disabled={!!progress} onChange={event => changeRange('end', event.currentTarget.value)} /></label>
            <button className="phrase-chip" type="button" disabled={!!progress}
              onClick={() => changeRange('start', String(video.current!.currentTime))}>Αρχή εδώ</button>
            <button className="phrase-chip" type="button" disabled={!!progress}
              onClick={() => changeRange('end', String(video.current!.currentTime))}>Τέλος εδώ</button>
          </div>
          {!validRange && <p className="microcopy">Έλεγξε ότι η αρχή προηγείται του τέλους και η διάρκεια είναι έως 12 δευτερόλεπτα.</p>}
        </details>}
        {progress && <p className="microcopy recognition-hint" role="status">{progress}</p>}
        {error && <p className="error-message recognition-error" role="alert">{error}</p>}
        {!!phrases.length && <div className="recognition-examples"><h3>Παραδείγματα φράσεων</h3><div className="recognition-actions">
          {phrases.map(entry => <button key={entry.id} className="phrase-chip" type="button" disabled={!!progress}
            onClick={() => void select(async signal => {
              const response = await fetch(entry.videoUrl, { signal });
              if (!response.ok) throw new Error('Το παράδειγμα δεν φορτώθηκε.');
              return response.blob();
            }, entry.phrase)}><Icon name="play" size={15} />{entry.phrase}</button>)}
        </div></div>}
      </section>
      <section className="surface recognition-message" aria-labelledby="sequence-result-heading">
        <div className="recognition-card-head"><h2 id="sequence-result-heading">Αποτέλεσμα</h2></div>
        <div className="recognition-response" aria-live="polite">
          {result ? <>
            {result.text ? <><output id="sequence-text">{result.text}</output><p>Αντιστοίχιση σε φράση του συνόλου εκμάθησης. Έλεγξε αν ταιριάζει με τη λήψη.</p></>
              : <><strong id="sequence-review">Η φράση χρειάζεται έλεγχο</strong>
                <p>{result.segments.some(segment => segment.kind === 'gloss')
                  ? 'Δες τα σημεία που ξεχώρισε το μοντέλο μέσα στο βίντεο.'
                  : 'Δοκίμασε μικρότερο απόσπασμα με ορατά χέρια και ώμους.'}</p></>}
            <ul className="sequence-timeline" aria-label="Σημεία του βίντεο">
              {result.segments.filter(segment => segment.kind !== 'blank').map(segment =>
                <li key={segment.startMs}><button className="phrase-chip" type="button"
                  onClick={() => { if (video.current) video.current.currentTime = segment.startMs / 1000; }}>
                  <span>{seconds(segment.startMs)}–{seconds(segment.endMs)}″</span>
                  {segment.kind === 'gloss' ? segment.gloss : 'Χρειάζεται έλεγχος'}
                </button></li>)}
            </ul>
            <div className="sequence-review-input"><label htmlFor="sequence-turn-text">Κείμενο ατάκας</label>
              <textarea id="sequence-turn-text" rows={2} value={draft} maxLength={300} onChange={event => setDraft(event.currentTarget.value)} />
            </div>
            <button className="button secondary" type="button" disabled={!draft.trim()} onClick={confirmTurn}>Επιβεβαίωση ατάκας</button>
          </> : <p>{progress ? 'Η ανάλυση βρίσκεται σε εξέλιξη…' : 'Η φράση και τα σημεία του βίντεο θα εμφανιστούν εδώ.'}</p>}
        </div>
        <p className="recognition-reminder"><Icon name="info" size={16} />Η δοκιμή καλύπτει περιορισμένο σύνολο φράσεων.</p>
      </section>
    </div>
    {turns.length > 0 && <section className="surface sequence-transcript" aria-labelledby="transcript-heading">
      <div className="recognition-card-head"><h2 id="transcript-heading">Κείμενο συνομιλίας</h2>
        <button type="button" className="button secondary" onClick={() => void copyTranscript()}>Αντιγραφή κειμένου</button></div>
      <ol aria-label="Επιβεβαιωμένες ατάκες">{turns.map(turn => <li key={turn.startMs}>
        <button className="phrase-chip" type="button" disabled={!!progress}
          onClick={() => { if (video.current) video.current.currentTime = turn.startMs / 1000; }}>
          {seconds(turn.startMs)}–{seconds(turn.endMs)}″</button>
        <span>{turn.text}</span>
        <button className="phrase-chip" type="button" disabled={!!progress} aria-label={'Αφαίρεση ατάκας από ' + seconds(turn.startMs)}
          onClick={() => { setTurns(turns.filter(item => item !== turn)); setTranscriptStatus('Η ατάκα αφαιρέθηκε.'); }}>Αφαίρεση</button>
      </li>)}</ol>
      <p className="microcopy">Αντέγραψε το κείμενο πριν κλείσεις τη σελίδα.</p>
    </section>}
    {transcriptStatus && <p className="microcopy recognition-hint" role="status">{transcriptStatus}</p>}
    <div className="recognition-actions"><a className="text-link" href="#/try">Δοκιμή μεμονωμένου νοήματος</a>
      <a className="text-link" href="#/teach">Teach Me</a></div>
  </div>;
}

