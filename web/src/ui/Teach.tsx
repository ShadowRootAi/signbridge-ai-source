import { useEffect, useRef, useState } from 'react';
import { loadLocalVideo, localVideoUrl } from '../local-video';
import { collectTeachingSample, deletePersonalSign, fitPersonalSign, readPersonalSigns, recognizePersonalSign, savePersonalSign, teachingCounts } from '../personal-gestures';
import type { TeachingSample } from '../personal-gestures';
import { personalTeaching, sequencePilot } from '../recognition-release';
import { Icon } from './icons';
import './recognition.css';
import './sequence.css';

const groups = [
  { key: 'training', title: '3 λήψεις του νοήματος για εκμάθηση' },
  { key: 'checks', title: '2 νέες λήψεις του ίδιου νοήματος για έλεγχο' },
  { key: 'negatives', title: '2 λήψεις διαφορετικών νοημάτων για έλεγχο' },
] as const;
const emptyFiles = () => ({ training: [] as File[], checks: [] as File[], negatives: [] as File[] });

type TeachProps = { initialSample?: TeachingSample; returnTo?: string; onClearSample: () => void };

export function TeachPage({ initialSample, returnTo = '#/try', onClearSample }: TeachProps) {
  const [signs, setSigns] = useState(readPersonalSigns);
  const [name, setName] = useState('');
  const [files, setFiles] = useState(emptyFiles);
  const [progress, setProgress] = useState('');
  const [feedback, setFeedback] = useState('');
  const [error, setError] = useState('');
  const [match, setMatch] = useState<string | null>();
  const task = useRef<AbortController>(undefined);
  const video = useRef<HTMLVideoElement>(null);
  const form = useRef<HTMLFormElement>(null);
  const ready = name.trim() && groups.every(group => {
    const carried = group.key === 'training' && initialSample ? 1 : 0;
    return files[group.key].length + carried === teachingCounts[group.key];
  });

  useEffect(() => () => task.current?.abort(), []);

  async function run(action: (signal: AbortSignal) => Promise<void>) {
    if (task.current) return;
    const controller = new AbortController();
    task.current = controller;
    setError('');
    setFeedback('');
    setMatch(undefined);
    setProgress('Άνοιγμα λήψης…');
    try { await action(controller.signal); }
    catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Ο έλεγχος διακόπηκε.');
    } finally {
      task.current = undefined;
      if (!controller.signal.aborted) setProgress('');
    }
  }

  async function sample(file: File, signal: AbortSignal): Promise<TeachingSample> {
    const url = localVideoUrl(file);
    const player = video.current!;
    try {
      await loadLocalVideo(player, url, signal);
      return await collectTeachingSample(player, file, signal);
    } catch (failure) {
      signal.throwIfAborted();
      throw new Error(file.name + ' · ' + (failure instanceof Error ? failure.message : 'Το βίντεο δεν άνοιξε.'));
    } finally {
      player.removeAttribute('src');
      player.load();
      URL.revokeObjectURL(url);
    }
  }

  function teach() {
    void run(async signal => {
      const samples = { training: initialSample ? [initialSample] : [], checks: [] as TeachingSample[], negatives: [] as TeachingSample[] };
      let processed = samples.training.length;
      const total = teachingCounts.training + teachingCounts.checks + teachingCounts.negatives;
      for (const group of groups) {
        for (const file of files[group.key]) {
          setProgress(`Ανάλυση λήψης ${++processed} από ${total}…`);
          samples[group.key].push(await sample(file, signal));
        }
      }
      signal.throwIfAborted();
      const result = fitPersonalSign(name, samples.training, samples.checks, samples.negatives);
      if (!result.sign) {
        setFeedback('Το νόημα χρειάζεται νέα παραδείγματα. Σωστές αντιστοιχίσεις ' + result.correct
          + '/2 και σωστές απορρίψεις ' + result.rejected + '/2 στις λήψεις ελέγχου.');
        return;
      }
      setSigns(savePersonalSign(result.sign));
      setFeedback('Το «' + result.sign.label + '» αποθηκεύτηκε αφού πέρασε και τους τέσσερις ελέγχους.');
      setFiles(emptyFiles());
      setName('');
      onClearSample();
      form.current?.reset();
    });
  }

  return <div className="recognition-page">
    <header className="page-heading">
      <p className="eyebrow">Προσωπικό λεξιλόγιο · Δοκιμή</p><h1>Teach Me</h1>
      <p>Δίδαξε ένα νόημα από δικές σου λήψεις. Η εφαρμογή το ενεργοποιεί αφού περάσει τον έλεγχο σε άλλα βίντεο.</p>
    </header>
    <div className="recognition-workspace">
      <section className="surface" aria-labelledby="teach-enroll">
        <div className="recognition-card-head"><h2 id="teach-enroll">Πρόσθεσε νόημα</h2><span className="recognition-private"><Icon name="shield" size={14} /> Στη συσκευή σου</span></div>
        <form ref={form} className="teach-form" onSubmit={event => { event.preventDefault(); if (ready) teach(); }}>
          <label>Σημασία του νοήματος <input value={name} onChange={event => setName(event.currentTarget.value)} maxLength={60} disabled={!!progress} required /></label>
          <p className="microcopy">Ένα νόημα ανά βίντεο, έως 3 δευτερόλεπτα. Χρησιμοποίησε διαφορετικές λήψεις από γνώστη ΕΝΓ, με χέρια και ώμους ορατά.</p>
          {initialSample && <div className="recognition-actions" data-testid="teach-carried-sample">
            <span className="microcopy">Η λήψη που ανέλυσες μετρά ως το πρώτο παράδειγμα.</span>
            <button className="button quiet" type="button" disabled={!!progress} onClick={onClearSample}>Αφαίρεση αρχικής λήψης</button>
          </div>}
          {groups.map(group => <label key={group.key}>{group.key === 'training' && initialSample ? '2 ακόμη λήψεις του νοήματος για εκμάθηση' : group.title}
            <input type="file" name={group.key} accept="video/mp4,video/webm" multiple disabled={!!progress}
              onChange={event => setFiles({ ...files, [group.key]: Array.from(event.currentTarget.files ?? []) })} />
          </label>)}
          <button className="button" type="submit" disabled={!ready || !!progress}>Έλεγχος και αποθήκευση</button>
        </form>
        <video ref={video} playsInline muted hidden />
        {progress && <p className="microcopy recognition-hint" role="status">{progress}</p>}
        {feedback && <p className="recognition-hint" role="status">{feedback}</p>}
        {error && <p className="error-message recognition-error" role="alert">{error}</p>}
      </section>
      <section className="surface" aria-labelledby="teach-words">
        <div className="recognition-card-head"><h2 id="teach-words">Τα νοήματά σου</h2></div>
        {signs.length ? <ul className="teach-saved">{signs.map(sign => <li key={sign.label}><span>{sign.label}</span>
          <button className="phrase-chip" type="button" disabled={!!progress} onClick={() => {
            try { setSigns(deletePersonalSign(sign.label)); setMatch(undefined); }
            catch { setError('Η διαγραφή δεν ολοκληρώθηκε. Δοκίμασε ξανά.'); }
          }} aria-label={'Διαγραφή ' + sign.label}>Διαγραφή</button></li>)}</ul>
          : <p className="microcopy">Τα νοήματα που περνούν τον έλεγχο εμφανίζονται εδώ.</p>}
        <div className="recognition-actions">
          <label className="button secondary recognition-upload"><Icon name="upload" size={18} /> Δοκίμασε νέο βίντεο
            <input aria-label="Δοκίμασε νέο προσωπικό νόημα" type="file" accept="video/mp4,video/webm" disabled={!!progress || !signs.length}
              onChange={event => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = '';
                if (file) void run(async signal => setMatch(recognizePersonalSign(await sample(file, signal), signs)));
              }} />
          </label>
        </div>
        {match !== undefined && <output className="teach-result" aria-live="polite">{match ?? 'Χρειάζεται νέο παράδειγμα ή πιο καθαρή λήψη.'}</output>}
        <p className="recognition-reminder"><Icon name="info" size={16} />Αποθηκεύονται μόνο τα πρότυπα κίνησης στον browser. Δοκίμασε με τον ίδιο νοηματιστή.</p>
      </section>
    </div>
    <div className="recognition-actions">
      {personalTeaching && <a className="text-link" href={returnTo}>Επιστροφή στην αναγνώριση</a>}
      {sequencePilot && <a className="text-link" href="#/sequence">Συνεχές βίντεο</a>}
    </div>
  </div>;
}

