import { useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { Icon } from './icons';
import './library.css';

export interface LibraryEntry {
  id: string;
  phrase: string;
  aliases: string[];
  contentKind?: 'word' | 'phrase';
  annotationLabel?: string;
  videoUrl: string;
  posterUrl?: string;
  sourceUrl: string;
  attribution: string;
  modifications?: string;
  license: string | { id: string; name: string; url: string };
  verification: { status: 'dataset-annotated' | 'expert-reviewed'; basis: string };
  rights?: { redistributionConfirmed: boolean };
}

const contexts: Record<string, { category: string; description: string }> = {
  geia: { category: 'Χαιρετισμός', description: 'Για να χαιρετήσεις κάποιον.' },
  efcharisto: { category: 'Ευγνωμοσύνη', description: 'Για να εκφράσεις ευχαριστίες.' },
  entaxei: { category: 'Επιβεβαίωση', description: 'Για να επιβεβαιώσεις ότι κάτι είναι εντάξει.' },
};

export function normalizedPhrase(text: string) {
  return text.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase('el-GR').replace(/ς/g, 'σ')
    .trim().replace(/[.!?;;·…]+$/u, '').trim().replace(/\s+/g, ' ');
}

function displayPhrase(phrase: string) {
  return phrase.charAt(0).toLocaleUpperCase('el-GR') + phrase.slice(1);
}

function isLocalLibraryAsset(value: string) {
  try {
    const url = new URL(value, window.location.origin);
    return url.origin === window.location.origin && /^https?:$/.test(url.protocol) && url.pathname.startsWith('/library/');
  } catch {
    return false;
  }
}

function verifiedEntry(value: unknown): value is LibraryEntry {
  if (!value || typeof value !== 'object') return false;
  const entry = value as Partial<LibraryEntry>;
  return typeof entry.id === 'string' && typeof entry.phrase === 'string'
    && Array.isArray(entry.aliases) && entry.aliases.every(alias => typeof alias === 'string')
    && typeof entry.videoUrl === 'string' && isLocalLibraryAsset(entry.videoUrl)
    && (entry.posterUrl === undefined || typeof entry.posterUrl === 'string' && isLocalLibraryAsset(entry.posterUrl))
    && typeof entry.sourceUrl === 'string' && typeof entry.attribution === 'string'
    && (typeof entry.license === 'string' || typeof entry.license?.name === 'string' && typeof entry.license.url === 'string')
    && typeof entry.verification?.basis === 'string'
    && ['dataset-annotated', 'expert-reviewed'].includes(entry.verification.status)
    && entry.rights?.redistributionConfirmed === true;
}

export function useCatalog(enabled = true) {
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [state, setState] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    fetch('/library/catalog.json', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Catalog unavailable');
      const catalog = await response.json();
      const items: unknown = Array.isArray(catalog) ? catalog : catalog?.entries;
      if (!Array.isArray(items)) throw new Error('Invalid catalog');
      const verified = items.filter(verifiedEntry);
      if (controller.signal.aborted) return;
      setEntries(verified);
      setState(verified.length ? 'ready' : 'unavailable');
    }).catch(() => {
      if (!controller.signal.aborted) setState('unavailable');
    });
    return () => controller.abort();
  }, [attempt, enabled]);

  function retry() {
    setState('loading');
    setAttempt(value => value + 1);
  }

  return { entries, state, retry };
}

function LibraryPlayer({ entry, heading }: { entry: LibraryEntry; heading: RefObject<HTMLHeadingElement | null> }) {
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState('');
  const [canPlay, setCanPlay] = useState(false);
  const license = typeof entry.license === 'object' ? entry.license : undefined;

  async function replay() {
    if (!video.current) return;
    setError('');
    try {
      video.current.currentTime = 0;
      await video.current.play();
    } catch {
      setError('Η αναπαραγωγή δεν ξεκίνησε. Δοκίμασε από τα χειριστήρια του βίντεο.');
    }
  }

  return <>
    <div className="library-player-title">
      <div><p className="eyebrow">Βίντεο ΕΝΓ</p><h2 ref={heading} tabIndex={-1} id="library-player-heading">{displayPhrase(entry.phrase)}</h2></div>
      <span className="library-category">{contexts[entry.id]?.category ?? 'Διαθέσιμη απόδοση'}</span>
    </div>
    <div className="library-screen">
      <video id="library-video" ref={video} controls playsInline preload="metadata" poster={entry.posterUrl} src={entry.videoUrl}
        onLoadedMetadata={() => setCanPlay(true)} onError={() => { setCanPlay(false); setError('Το βίντεο δεν φορτώθηκε. Δοκίμασε ξανά.'); }}
        aria-label={`Απόδοση της φράσης ${entry.phrase} στην Ελληνική Νοηματική`} />
    </div>
    <div className="library-player-actions">
      <button className="button secondary" type="button" onClick={() => void replay()} disabled={!canPlay}><Icon name="refresh" size={17} />Αναπαραγωγή από την αρχή</button>
      <p className="microcopy">Ρύθμισε την ταχύτητα από τα χειριστήρια του βίντεο.</p>
    </div>
    {error && <div className="library-error" role="alert"><p>{error}</p>{!canPlay && <button className="button secondary" type="button" onClick={() => { setError(''); video.current?.load(); }}>Νέα προσπάθεια</button>}</div>}
    <details className="library-source">
      <summary>Πηγή και άδεια</summary>
      <p><strong>{entry.verification.status === 'expert-reviewed' ? 'Έλεγχος από ειδικό' : 'Επισήμανση του αρχικού συνόλου δεδομένων'}</strong></p>
      <p>{entry.verification.basis}</p>
      <p>{entry.attribution}</p>
      {entry.modifications && <p>{entry.modifications}</p>}
      <div className="library-source-links">
        {/^https?:\/\//.test(entry.sourceUrl) && <a href={entry.sourceUrl} target="_blank" rel="noreferrer">Αρχική πηγή <Icon name="arrow" size={15} /></a>}
        {license && /^https?:\/\//.test(license.url) ? <a href={license.url} target="_blank" rel="noreferrer">{license.name} <Icon name="arrow" size={15} /></a>
          : typeof entry.license === 'string' && <span>{entry.license}</span>}
      </div>
    </details>
  </>;
}

type LibrarySelection = { kind: 'default' } | { kind: 'entry'; id: string } | { kind: 'missing'; query: string };

const libraryMilestone = 3_000;
const libraryNumber = new Intl.NumberFormat('el-GR');
const reducedMotionQuery = '(prefers-reduced-motion: reduce)';

function LibraryCount({ total, filtered }: { total: number; filtered?: number }) {
  const reachedMilestone = total >= libraryMilestone;
  const [count, setCount] = useState(() => window.matchMedia(reducedMotionQuery).matches ? libraryMilestone : 0);

  useEffect(() => {
    if (!reachedMilestone) return;
    const preference = window.matchMedia(reducedMotionQuery);
    if (preference.matches) {
      setCount(libraryMilestone);
      return;
    }

    const started = performance.now();
    let frame = requestAnimationFrame(update);
    function update(now: number) {
      const progress = Math.min(1, Math.max(0, now - started) / 950);
      setCount(Math.floor(libraryMilestone * (1 - (1 - progress) ** 3)));
      if (progress < 1) frame = requestAnimationFrame(update);
    }
    function onPreferenceChange(event: MediaQueryListEvent) {
      if (!event.matches) return;
      cancelAnimationFrame(frame);
      setCount(libraryMilestone);
    }
    preference.addEventListener('change', onPreferenceChange);
    return () => {
      cancelAnimationFrame(frame);
      preference.removeEventListener('change', onPreferenceChange);
    };
  }, [reachedMilestone]);

  const exact = filtered === undefined ? `${libraryNumber.format(total)} βίντεο`
    : `${libraryNumber.format(filtered)} από ${libraryNumber.format(total)} βίντεο`;
  const visible = filtered !== undefined ? `${filtered} από ${total}`
    : reachedMilestone ? `${libraryNumber.format(count)}${count === libraryMilestone ? '+' : ''}` : total;
  return <>
    <span aria-hidden="true" title={exact}>{visible}</span>
    <div className="sr-only" role="status">{exact}</div>
  </>;
}

export function LibraryPage() {
  const { entries, state, retry } = useCatalog();
  const [query, setQuery] = useState('');
  const [editingQuery, setEditingQuery] = useState(false);
  const [selection, setSelection] = useState<LibrarySelection>({ kind: 'default' });
  const heading = useRef<HTMLHeadingElement>(null);
  const index = useMemo(() => entries.map(entry => ({
    entry,
    terms: [entry.phrase, ...entry.aliases].map(normalizedPhrase),
  })), [entries]);
  const filterAsYouType = entries.length > 24;
  const normalizedQuery = normalizedPhrase(query);
  const filtering = filterAsYouType && editingQuery && normalizedQuery !== '';
  const visibleEntries = useMemo(() => filtering
    ? index.filter(item => item.terms.some(term => term.includes(normalizedQuery))).map(item => item.entry)
    : entries, [entries, filtering, index, normalizedQuery]);
  const match = selection.kind === 'default' ? entries[0] : selection.kind === 'entry'
    ? entries.find(entry => entry.id === selection.id) : undefined;

  useEffect(() => {
    if (selection.kind !== 'default') heading.current?.focus();
  }, [selection]);

  function choose(entry: LibraryEntry) {
    setQuery(entry.phrase);
    setEditingQuery(false);
    setSelection({ kind: 'entry', id: entry.id });
  }

  function search() {
    if (!normalizedQuery || state !== 'ready') return;
    setEditingQuery(false);
    const result = index.find(item => item.terms.includes(normalizedQuery))?.entry;
    if (result) choose(result);
    else setSelection({ kind: 'missing', query: query.trim() });
  }

  return <>
    <header className="page-heading library-page-heading">
      <span className="eyebrow">Κείμενο σε νόημα</span>
      <h1>Βιβλιοθήκη νοημάτων</h1>
      <p>Γράψε μια λέξη ή σύντομη φράση και δες τη διαθέσιμη απόδοση σε βίντεο Ελληνικής Νοηματικής.</p>
    </header>
    <div className="library-workbench">
      <section className="surface library-browser-card" aria-labelledby="library-browser-heading">
        <h2 id="library-browser-heading">Βρες μια λέξη</h2>
        <form className="library-query" role="search" aria-label="Αναζήτηση στη βιβλιοθήκη" onSubmit={event => { event.preventDefault(); search(); }}>
          <label className="sr-only" htmlFor="phrase">Ελληνική λέξη ή φράση</label>
          <div className="library-query-row">
            <input id="phrase" type="search" value={query} onChange={event => {
              setQuery(event.target.value);
              setEditingQuery(true);
            }} placeholder="π.χ. ευχαριστώ" autoComplete="off" maxLength={180} aria-describedby="library-query-hint" />
            <button className="button" type="submit" disabled={!query.trim() || state !== 'ready'}>Αναζήτηση</button>
          </div>
          <p id="library-query-hint" className="microcopy">{filterAsYouType && 'Η λίστα φιλτράρεται καθώς γράφεις. '}Η αναζήτηση ταιριάζει ολόκληρη τη διαθέσιμη λέξη ή φράση.</p>
        </form>
        <div className="library-list-heading"><h3>Βίντεο στη βιβλιοθήκη</h3>
          {state === 'ready' && <LibraryCount total={entries.length} filtered={filtering ? visibleEntries.length : undefined} />}
        </div>
        {state === 'loading' && <p className="library-state" role="status">Φόρτωση βιβλιοθήκης…</p>}
        {state === 'unavailable' && <div className="library-state"><p role="status">Η βιβλιοθήκη δεν είναι διαθέσιμη αυτή τη στιγμή.</p><button className="button secondary" type="button" onClick={retry}>Δοκίμασε ξανά</button></div>}
        {filtering && !visibleEntries.length && <p className="library-state">Η λίστα δεν έχει καταχωρίσεις που να περιέχουν αυτό το κείμενο.</p>}
        {visibleEntries.length > 0 && <ul className="library-entry-list">{visibleEntries.map(entry => <li key={entry.id}>
          <button className="library-entry" type="button" data-library-entry={entry.id} aria-pressed={match?.id === entry.id} onClick={() => choose(entry)}>
            <span className="library-entry-icon"><Icon name="play" size={19} /></span>
            <span><strong>{displayPhrase(entry.phrase)}</strong><small>{contexts[entry.id]?.description ?? 'Δες την απόδοση σε βίντεο.'}</small></span>
            <Icon name="arrow" size={17} className="library-entry-arrow" />
          </button>
        </li>)}</ul>}
      </section>
      <section className="surface library-viewer-card" aria-labelledby="library-player-heading">
        {match ? <LibraryPlayer key={match.id} entry={match} heading={heading} /> : <div className="library-placeholder">
          <span className="library-entry-icon"><Icon name={selection.kind === 'missing' ? 'info' : 'hand'} size={25} /></span>
          <h2 ref={heading} tabIndex={-1} id="library-player-heading">{selection.kind === 'missing' ? 'Δεν βρέθηκε απόδοση' : 'Βίντεο νοηματικής'}</h2>
          <p>{selection.kind === 'missing' ? <>Η βιβλιοθήκη δεν έχει βίντεο για «{selection.query}». Επίλεξε ένα από τα διαθέσιμα νοήματα.</>
            : 'Οι διαθέσιμες αποδόσεις θα εμφανιστούν εδώ.'}</p>
        </div>}
      </section>
    </div>
    <aside className="library-note" aria-label="Σχετικά με τη βιβλιοθήκη">
      <Icon name="info" size={19} /><p>Η αναζήτηση χρησιμοποιεί το καταγεγραμμένο κείμενο κάθε βίντεο. Η συλλογή περιλαμβάνει και αποσπάσματα σχολικών κειμένων.</p>
      <a className="text-link" href="#/about">Πώς λειτουργεί <Icon name="arrow" size={16} /></a>
    </aside>
  </>;
}
