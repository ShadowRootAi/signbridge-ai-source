import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import brandLogo from './assets/signbridge-logo.png';
import { Icon } from './ui/icons';
import type { IconName } from './ui/icons';
import { RecognitionPage } from './ui/Recognition';
import { LibraryPage } from './ui/Library';
import { personalTeaching, sequencePilot, useOpenHands } from './recognition-release';
import type { TeachingSample } from './personal-gestures';
import { useOpenHandsVocabularyLabel } from './use-openhands-vocabulary';

const navigation = [
  { route: 'home', href: '#/', label: 'Αρχική' },
  { route: 'sign', href: '#/sign', label: 'Κάμερα' },
  { route: 'library', href: '#/library', label: 'Βιβλιοθήκη' },
  { route: 'try', href: '#/try', label: 'Δοκιμή βίντεο' },
  { route: 'about', href: '#/about', label: 'Βοήθεια' },
] as const;
const experiments = [
  { route: 'sequence', href: '#/sequence', label: 'Συνεχές βίντεο' },
  { route: 'teach', href: '#/teach', label: 'Teach Me' },
] as const;
const routes = [...navigation, ...(sequencePilot ? experiments : personalTeaching ? [experiments[1]] : [])];
type Route = typeof routes[number]['route'];
type TeachingDraft = { sample?: TeachingSample; returnTo: '#/sign' | '#/try' };
const SequencePage = lazy(() => import('./ui/Sequence').then(module => ({ default: module.SequencePage })));
const TeachPage = lazy(() => import('./ui/Teach').then(module => ({ default: module.TeachPage })));

const supported = ['Γεια', 'Ευχαριστώ', 'Εντάξει'];
const shortcuts: { href: string; icon: IconName; title: string; body: string; action: string }[] = [
  { href: '#/sign', icon: 'camera', title: 'Νόημα σε κείμενο', body: 'Κάνε ένα νόημα στην κάμερα και διάβασε το αποτέλεσμα.', action: 'Άνοιξε την κάμερα' },
  { href: '#/library', icon: 'hand', title: 'Κείμενο σε νόημα', body: 'Επίλεξε μια λέξη και δες πώς αποδίδεται σε βίντεο ΕΝΓ.', action: 'Μπες στη βιβλιοθήκη' },
  { href: '#/try', icon: 'play', title: 'Δοκιμή με βίντεο', body: 'Διάλεξε ένα έτοιμο παράδειγμα ή ανάλυσε δικό σου αρχείο.', action: 'Δοκίμασε ένα βίντεο' },
];
const questions = [
  { title: 'Πώς κάνω μια καθαρή λήψη;', answer: 'Στάσου με φως μπροστά σου. Κράτα τους ώμους και τα δύο χέρια στο κάδρο σε όλη τη διάρκεια του νοήματος.' },
  { title: 'Τι κάνω αν δεν ανοίγει η κάμερα;', answer: 'Έλεγξε την άδεια κάμερας στις ρυθμίσεις του browser. Μπορείς επίσης να χρησιμοποιήσεις τη δοκιμή βίντεο, που λειτουργεί χωρίς κάμερα.', href: '#/try', action: 'Πήγαινε στη δοκιμή βίντεο' },
  { title: 'Τι σημαίνει «Δεν αναγνωρίστηκε»;', answer: useOpenHands
    ? 'Το σύστημα μπορεί να μην έχει αρκετά καθαρή λήψη ή να μην αναγνωρίζει το νόημα. Κράτα τα χέρια ορατά και κάνε μία νέα λήψη με ένα νόημα τη φορά.'
    : 'Η λήψη μπορεί να ήταν ασαφής ή το νόημα να είναι εκτός λεξιλογίου. Αν δοκιμάζεις μία από τις τρεις διαθέσιμες λέξεις, κράτα τα χέρια ορατά και κάνε μια νέα λήψη.' },
  { title: 'Πού πηγαίνει η λήψη μου;', answer: 'Η αναγνώριση γίνεται στον browser σου. Η εφαρμογή χρησιμοποιεί την κάμερα με την άδειά σου, καταγράφει μόνο εικόνα και δεν ανεβάζει τη λήψη.' },
  { title: 'Από πού προέρχονται τα βίντεο της βιβλιοθήκης;', answer: 'Προέρχονται από τα σύνολα δεδομένων Greek Sign Language και Elementary23. Κάθε βίντεο αναφέρει την πηγή, την άδεια και τις αλλαγές που έγιναν στο αρχείο.', href: '#/library', action: 'Δες τη βιβλιοθήκη' },
];

function currentRoute(): Route {
  return routes.find(item => item.href === window.location.hash)?.route ?? 'home';
}

function Brand() {
  return (
    <a className="brand" href="#/" aria-label="SignBridge AI, αρχική">
      <img className="brand-mark" src={brandLogo} alt="" width={38} height={38} />
      <span>SignBridge <span className="brand-ai">AI</span></span>
    </a>
  );
}

function SignList() {
  const vocabulary = useOpenHandsVocabularyLabel();
  return <ul className="sign-list" aria-label="Υποστηριζόμενα νοήματα">{useOpenHands
    ? <li>{vocabulary}</li>
    : supported.map(sign => <li key={sign}>{sign}</li>)}</ul>;
}

function Home() {
  return (
    <>
      <section className="home-hero" aria-labelledby="home-heading">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-dot" aria-hidden="true" /> Ελληνική Νοηματική Γλώσσα</p>
          <h1 id="home-heading">Η επικοινωνία έχει <em>πολλούς τρόπους.</em></h1>
          <p className="hero-description">Δείξε ένα νόημα στην κάμερα ή βρες την απόδοσή του σε βίντεο. Με το SignBridge AI έχεις και τις δύο επιλογές σε ένα μέρος.</p>
          <div className="hero-actions">
            <a className="button" href="#/sign"><Icon name="camera" size={19} /> Άνοιξε την κάμερα</a>
            <a className="button secondary" href="#/library">Δες τη βιβλιοθήκη <Icon name="arrow" size={18} /></a>
          </div>
          <p className="hero-privacy"><Icon name="shield" size={18} /> Η ανάλυση γίνεται στη συσκευή σου.</p>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="hero-visual-header"><span><Icon name="camera" size={18} /> Από τη λήψη</span><span className="visual-dots"><i /><i /><i /></span></div>
          <div className="visual-stage">
            <div className="visual-frame"><i /><i /><i /><i /></div>
            <span className="visual-hand"><Icon name="hand" size={44} /></span>
          </div>
          <div className="visual-message"><span className="visual-message-icon"><Icon name="text" size={24} /></span><div><strong>Στο κείμενο</strong><span>Ένα νόημα κάθε φορά</span></div><Icon name="arrow" size={19} /></div>
        </div>
      </section>

      <div className="home-scope"><span><Icon name="info" size={18} /> Διαθέσιμα νοήματα</span><SignList /><a className="text-link" href="#/about">Πώς λειτουργεί <Icon name="arrow" size={16} /></a></div>

      <section className="home-section" aria-labelledby="shortcuts-heading">
        <div className="section-heading"><h2 id="shortcuts-heading">Τι θέλεις να κάνεις;</h2><p>Διάλεξε τον τρόπο που σε εξυπηρετεί.</p></div>
        <div className="shortcut-grid">
          {shortcuts.map(shortcut => (
            <a key={shortcut.href} href={shortcut.href} className="surface shortcut-card">
              <span className="shortcut-icon"><Icon name={shortcut.icon} size={24} /></span>
              <h3>{shortcut.title}</h3>
              <p>{shortcut.body}</p>
              <span className="text-link">{shortcut.action} <Icon name="arrow" size={17} /></span>
            </a>
          ))}
        </div>
      </section>
    </>
  );
}

function About() {
  return (
    <>
      <div className="page-heading">
        <p className="eyebrow">Βοήθεια</p>
        <h1>Πώς χρησιμοποιείς το SignBridge</h1>
        <p>Ξεκίνα με μια σύντομη λήψη ή επίλεξε ένα από τα διαθέσιμα βίντεο της βιβλιοθήκης.</p>
      </div>

      <div className="help-grid">
        <section className="surface help-panel" aria-labelledby="help-camera">
          <span className="shortcut-icon"><Icon name="camera" size={24} /></span>
          <h2 id="help-camera">Από νόημα σε κείμενο</h2>
          <ol><li>Άνοιξε την κάμερα και δώσε άδεια.</li><li>Κάνε ένα από τα διαθέσιμα νοήματα στη λήψη.</li><li>Διάβασε το αποτέλεσμα και επιβεβαίωσε ότι ταιριάζει.</li></ol>
          <a className="text-link" href="#/sign">Άνοιξε την κάμερα <Icon name="arrow" size={16} /></a>
        </section>
        <section className="surface help-panel" aria-labelledby="help-library">
          <span className="shortcut-icon"><Icon name="hand" size={24} /></span>
          <h2 id="help-library">Από κείμενο σε νόημα</h2>
          <ol><li>Γράψε μια λέξη ή επίλεξέ την από τη λίστα.</li><li>Άνοιξε το βίντεο της αντίστοιχης απόδοσης.</li><li>Χρησιμοποίησε τα χειριστήρια για παύση ή επανάληψη.</li></ol>
          <a className="text-link" href="#/library">Άνοιξε τη βιβλιοθήκη <Icon name="arrow" size={16} /></a>
        </section>
      </div>

      <section className="help-scope" aria-labelledby="help-scope-heading">
        <Icon name="info" size={23} />
        <div><h2 id="help-scope-heading">{sequencePilot ? 'Νοήματα και σύντομες φράσεις' : useOpenHands ? 'Ένα νόημα κάθε φορά' : 'Τρία νοήματα, ένα κάθε φορά'}</h2>
          <p>{sequencePilot ? 'Η δοκιμή συνεχούς βίντεο ψάχνει γνωστές φράσεις και δείχνει ποια σημεία χρειάζονται έλεγχο. Στο Teach Me μπορείς να προσθέτεις προσωπικά νοήματα με δικές σου λήψεις.' : 'Αυτή η έκδοση δουλεύει με μεμονωμένα νοήματα. Μια λήψη μπορεί να απορριφθεί ή το αποτέλεσμα να χρειάζεται διόρθωση.'}</p>
          {sequencePilot && <div className="recognition-actions">{experiments.map(item => <a key={item.route} className="text-link" href={item.href}>{item.label}</a>)}</div>}
          <SignList />
        </div>
      </section>

      <section className="faq" aria-labelledby="faq-heading">
        <div className="section-heading"><h2 id="faq-heading">Συχνές ερωτήσεις</h2></div>
        <div className="surface faq-list">
          {questions.map(item => (
            <details key={item.title}>
              <summary>{item.title}<span aria-hidden="true" /></summary>
              <p>{item.answer}</p>
              {item.href && <a className="text-link" href={item.href}>{item.action} <Icon name="arrow" size={16} /></a>}
            </details>
          ))}
        </div>
        <a className="text-link" href="/NOTICE.txt">Πηγές και άδειες <Icon name="arrow" size={16} /></a>
      </section>
    </>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute);
  const [menuOpen, setMenuOpen] = useState(false);
  const [teachingDraft, setTeachingDraft] = useState<TeachingDraft>();
  const content = useRef<HTMLElement>(null);
  const menuToggle = useRef<HTMLButtonElement>(null);
  const previousRoute = useRef(route);

  useEffect(() => {
    function updateRoute() {
      const next = currentRoute();
      setRoute(next);
      if (next !== 'teach') setTeachingDraft(undefined);
      setMenuOpen(false);
    }
    window.addEventListener('hashchange', updateRoute);
    return () => window.removeEventListener('hashchange', updateRoute);
  }, []);

  useEffect(() => {
    document.title = `${routes.find(item => item.route === route)?.label} · SignBridge AI`;
    if (previousRoute.current !== route) {
      previousRoute.current = route;
      window.scrollTo(0, 0);
      content.current?.focus({ preventScroll: true });
    }
  }, [route]);

  function teachFromRecognition(sample: TeachingSample) {
    setTeachingDraft({ sample, returnTo: route === 'sign' ? '#/sign' : '#/try' });
    window.location.hash = '#/teach';
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content" onClick={event => { event.preventDefault(); content.current?.focus(); }}>Μετάβαση στο περιεχόμενο</a>
      <header className="site-header" onKeyDown={event => {
        if (event.key === 'Escape' && menuOpen) {
          setMenuOpen(false);
          menuToggle.current?.focus();
        }
      }}>
        <div className="header-inner">
          <Brand />
          <button ref={menuToggle} className="menu-toggle" type="button" aria-expanded={menuOpen} aria-controls="main-navigation" aria-label={menuOpen ? 'Κλείσιμο μενού' : 'Άνοιγμα μενού'} onClick={() => setMenuOpen(open => !open)}>
            <Icon name={menuOpen ? 'close' : 'menu'} size={23} />
          </button>
          <nav id="main-navigation" className={menuOpen ? 'navigation open' : 'navigation'} aria-label="Κύρια πλοήγηση">
            {navigation.map(item => <a key={item.route} href={item.href} aria-current={route === item.route ? 'page' : undefined} onClick={() => setMenuOpen(false)}>{item.label}</a>)}
          </nav>
          <span className="header-edition">Δοκιμαστική έκδοση</span>
        </div>
      </header>

      <main id="main-content" ref={content} tabIndex={-1} className={`main-content route-${route}`}>
        {route === 'home' && <Home />}
        {route === 'sign' && <RecognitionPage key="camera" mode="camera" onTeach={personalTeaching ? teachFromRecognition : undefined} />}
        {route === 'try' && <RecognitionPage key="file" mode="file" onTeach={personalTeaching ? teachFromRecognition : undefined} />}
        {route === 'library' && <LibraryPage />}
        {route === 'about' && <About />}
        {(route === 'sequence' || route === 'teach') && <Suspense fallback={<p role="status">Φόρτωση δοκιμής…</p>}>
          {route === 'sequence' ? <SequencePage /> : <TeachPage initialSample={teachingDraft?.sample} returnTo={teachingDraft?.returnTo}
            onClearSample={() => setTeachingDraft(draft => draft && { returnTo: draft.returnTo })} />}
        </Suspense>}
      </main>

      <footer className="site-footer">
        <div className="footer-inner"><Brand /><span>Ελληνική Νοηματική Γλώσσα</span><div className="footer-links"><a href="#/about">Βοήθεια</a><a href="/NOTICE.txt">Πηγές και άδειες</a></div></div>
      </footer>
    </div>
  );
}
