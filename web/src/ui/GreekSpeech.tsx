import { useEffect, useState } from 'react';
import { Icon } from './icons';
export function GreekSpeech({ text }: { text: string }) {
  const [voice, setVoice] = useState<SpeechSynthesisVoice>();
  const [checked, setChecked] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!('speechSynthesis' in window)) { setChecked(true); return; }
    const synth = window.speechSynthesis;
    const update = () => {
      const greek = synth.getVoices().filter(item => /^el(?:[-_]|$)/i.test(item.lang));
      setVoice(greek.find(item => item.localService) ?? greek[0]);
      if (greek.length || synth.getVoices().length) setChecked(true);
    };
    update(); synth.addEventListener('voiceschanged', update);
    const timeout = window.setTimeout(() => { update(); setChecked(true); }, 1800);
    return () => { clearTimeout(timeout); synth.removeEventListener('voiceschanged', update); synth.cancel(); };
  }, []);
  useEffect(() => { window.speechSynthesis?.cancel(); setSpeaking(false); setError(''); }, [text]);
  function speak() {
    if (!voice || !text.trim()) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = voice.lang; utterance.voice = voice; utterance.rate = .94;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = event => { setSpeaking(false); if (event.error !== 'canceled' && event.error !== 'interrupted') setError('Η εκφώνηση δεν ολοκληρώθηκε. Το κείμενο παραμένει διαθέσιμο.'); };
    setError(''); window.speechSynthesis.cancel(); window.speechSynthesis.speak(utterance);
  }
  return <div className="speech-control">
    <p className="microcopy"><Icon name="sound" size={16} />{!checked ? 'Έλεγχος ελληνικής φωνής…' : voice ? 'Ελληνική φωνή διαθέσιμη' : 'Δεν βρέθηκε ελληνική φωνή σε αυτόν τον browser.'}</p>
    {voice && <p className="microcopy">{voice.localService ? 'Εκφώνηση με φωνή της συσκευής σου.' : 'Αυτή η φωνή μπορεί να στείλει το κείμενο σε υπηρεσία φωνής του browser ή της συσκευής.'} Επιβεβαίωσε πρώτα ότι το μήνυμα είναι σωστό.</p>}
    {speaking ? <button className="button secondary full" onClick={() => { window.speechSynthesis.cancel(); setSpeaking(false); }}><Icon name="pause" />Διακοπή εκφώνησης</button>
      : <button className="button secondary full" disabled={!voice} onClick={speak}><Icon name="sound" />Επιβεβαίωση και εκφώνηση</button>}
    {!voice && checked && <p className="microcopy">Μπορείς να δείξεις το κείμενο. Δεν χρησιμοποιούμε αγγλική φωνή ως υποκατάστατο.</p>}
    {error && <p className="inline-error" role="alert">{error}</p>}
  </div>;
}
