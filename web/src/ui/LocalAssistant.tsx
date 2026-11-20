import { useEffect, useRef, useState } from 'react';
import './local-assistant.css';

type Correction = { from: string; to: string };
type Recognition = { accepted: true; label: string } | { accepted: false; label?: never };
type Request = { message: string; recognition?: Recognition; correction?: Correction } | { memory: 'clear' };
type Reply = { reply: string; remembered?: boolean; appliedCorrections?: Correction[]; error?: string };

const fallbackError = 'Ο τοπικός βοηθός δεν απάντησε. Δοκίμασε ξανά.';

export function LocalAssistant({ recognition, disabled }: { recognition?: Recognition; disabled: boolean }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const openButton = useRef<HTMLButtonElement>(null);
  const messageField = useRef<HTMLTextAreaElement>(null);
  const pending = useRef<AbortController>(null);
  const [message, setMessage] = useState('');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [busy, setBusy] = useState(false);
  const [reply, setReply] = useState<Reply>();
  const [error, setError] = useState('');
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => () => pending.current?.abort(), []);

  function open() {
    setFrom(recognition?.label ?? '');
    dialog.current?.showModal();
    messageField.current?.focus();
  }

  function close() {
    pending.current?.abort();
    pending.current = null;
    setBusy(false);
    setReply(undefined);
    setError('');
    setMessage('');
    setFrom('');
    setTo('');
    setConfirmClear(false);
    openButton.current?.focus();
  }

  async function request(payload: Request) {
    if (pending.current) return;
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setReply(undefined);
    setError('');
    try {
      const response = await fetch('/__signbridge/local-assistant/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        cache: 'no-store', redirect: 'error', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(125_000)]),
      });
      const data = await response.json() as Reply;
      if (!response.ok || typeof data.reply !== 'string') {
        throw new Error(typeof data.error === 'string' ? data.error : fallbackError);
      }
      if (controller.signal.aborted) return;
      setReply(data);
      if (data.remembered) { setFrom(''); setTo(''); }
      setConfirmClear(false);
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof Error && !(failure instanceof TypeError || failure instanceof SyntaxError || failure instanceof DOMException)
        ? failure.message : fallbackError);
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setBusy(false);
      }
    }
  }

  return <>
    <button ref={openButton} type="button" className="button quiet local-assistant-open" disabled={disabled}
      aria-haspopup="dialog" onClick={open}>Τοπικός βοηθός</button>
    <dialog ref={dialog} className="surface local-assistant" aria-labelledby="local-assistant-title" onClose={close}>
      <div className="recognition-card-head">
        <h2 id="local-assistant-title">Τοπικός βοηθός</h2>
        <button type="button" className="button quiet" onClick={() => dialog.current?.close()}>Κλείσιμο</button>
      </div>
      {recognition && <p className="microcopy">
        {recognition.accepted ? <>Αποτέλεσμα αναγνώρισης <strong>{recognition.label}</strong></> : 'Η τελευταία λήψη δεν αναγνωρίστηκε.'}
      </p>}
      <form onSubmit={event => {
        event.preventDefault();
        if (message.trim()) void request({ message: message.trim(), recognition });
      }}>
        <label>Μήνυμα<textarea ref={messageField} name="message" rows={3} maxLength={1500} required value={message}
          onChange={event => setMessage(event.target.value)} /></label>
        <button type="submit" className="button" disabled={busy || !message.trim()}>Αποστολή</button>
      </form>
      <details>
        <summary>Αποθήκευση διόρθωσης</summary>
        <form onSubmit={event => {
          event.preventDefault();
          if (from.trim() && to.trim()) void request({ message: 'Αποθήκευσε αυτή τη διόρθωση.', correction: { from: from.trim(), to: to.trim() } });
        }}>
          <label>Αρχική φράση<input name="correction-from" required maxLength={120} value={from} onChange={event => setFrom(event.target.value)} /></label>
          <label>Σωστή φράση<input name="correction-to" required maxLength={120} value={to} onChange={event => setTo(event.target.value)} /></label>
          <p className="microcopy">Οι διορθώσεις αποθηκεύονται στη μνήμη κειμένου του βοηθού.</p>
          <button type="submit" className="button secondary" disabled={busy || !from.trim() || !to.trim()}>Αποθήκευση</button>
        </form>
      </details>
      <div className="local-assistant-result" aria-live="polite">
        {busy && <p>Επεξεργασία…</p>}
        {error && <p className="error-message" role="alert">{error}</p>}
        {reply && <>
          <p data-local-assistant-reply>{reply.reply}</p>
          {!!reply.appliedCorrections?.length && <ul aria-label="Διορθώσεις που χρησιμοποιήθηκαν">
            {reply.appliedCorrections.map(item => <li key={item.from}>«{item.from}» → «{item.to}»</li>)}
          </ul>}
        </>}
      </div>
      {confirmClear ? <div className="local-assistant-clear">
        <p>Θέλεις να διαγράψεις όλες τις αποθηκευμένες διορθώσεις;</p>
        <button type="button" className="button secondary" disabled={busy} onClick={() => void request({ memory: 'clear' })}>Διαγραφή όλων</button>
        <button type="button" className="button quiet" disabled={busy} onClick={() => setConfirmClear(false)}>Ακύρωση</button>
      </div> : <button type="button" className="button quiet" disabled={busy} onClick={() => setConfirmClear(true)}>Εκκαθάριση μνήμης</button>}
    </dialog>
  </>;
}
