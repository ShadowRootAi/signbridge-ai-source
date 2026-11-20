import type { OpenHandsResult } from '../openhands-pipeline';
import type { PipelineProgress } from '../pipeline-contract';

const outcomes = {
  tracking: 'Ανεπαρκής ανίχνευση χεριών ή σώματος',
  no_motion: 'Ανεπαρκής κίνηση',
  outside_roster: 'Πρώτη πρόταση εκτός ενεργού λεξιλογίου',
  low_confidence: 'Χαμηλή βαθμολογία μοντέλου',
  low_evidence: 'Ανεπαρκής ένδειξη από το μοντέλο',
  low_margin: 'Μικρή διαφορά των δύο πρώτων προτάσεων',
  primary: 'Αποδοχή από το ST-GCN',
  agreement: 'Αποδοχή με επιβεβαίωση LSTM',
  temporal: 'Αποδοχή με LSTM και δύο χρονικές επαναδειγματοληψίες',
};

const trackingFormat = new Intl.NumberFormat('el-GR', { style: 'percent', maximumFractionDigits: 1 });
const scoreFormat = new Intl.NumberFormat('el-GR', { minimumFractionDigits: 6, maximumFractionDigits: 6 });
const timeFormat = new Intl.NumberFormat('el-GR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export function DecisionTrace({ result, progress }: { result?: OpenHandsResult; progress?: PipelineProgress }) {
  const gate = result?.rejectionCode ?? result?.acceptedBy;
  return <section className="judge-trace" aria-labelledby="judge-trace-title">
    <h3 id="judge-trace-title">Ίχνος απόφασης AI</h3>
    {result ? <>
      <dl className="judge-stages">
        <div><dt>Ανίχνευση</dt><dd>{result.frames} καρέ{typeof result.trackingQuality === 'number'
          && <> · {trackingFormat.format(result.trackingQuality)} με ώμους και τουλάχιστον ένα χέρι</>}</dd></div>
        <div><dt>Ταξινόμηση</dt><dd data-classifier-executed={!!result.classifierExecuted}>
          {result.classifierExecuted ? 'Εκτελέστηκε το ST-GCN' : 'Ο ταξινομητής δεν εκτελέστηκε'}</dd></div>
        <div><dt>Κανόνας αποδοχής</dt><dd data-decision-gate={gate}>{gate ? outcomes[gate] : result.reason}</dd></div>
      </dl>
      {result.recoveryEvidence && (gate === 'agreement' || gate === 'temporal') && <details className="judge-scores">
        <summary>Έλεγχος δεύτερου μοντέλου</summary>
        <ul>
          <li><span>Πρώτη βαθμολογία / όριο</span><span>{scoreFormat.format(result.recoveryEvidence.primaryScore)} / {scoreFormat.format(result.recoveryEvidence.primaryThreshold)}</span></li>
          <li><span>Στήριξη LSTM</span><span>{scoreFormat.format(result.recoveryEvidence.secondarySupport)}</span></li>
          {result.recoveryEvidence.temporalChecks.map(check => <li key={check.factor}>
            <span>ST-GCN · {trackingFormat.format(check.factor)} · {check.frames} καρέ</span><span>{scoreFormat.format(check.score)}</span>
          </li>)}
        </ul>
      </details>}
      {result.classifierExecuted && result.topCandidates.length > 0 && <details className="judge-scores">
        <summary>Βαθμολογίες ST-GCN (softmax)</summary>
        <ul>{result.topCandidates.slice(0, 3).map(candidate => <li key={candidate.gloss}>
          <span>{candidate.gloss}</span><span>{scoreFormat.format(candidate.score)}</span>
        </li>)}</ul>
      </details>}
      <p className="judge-time">Συνολική ανάλυση · {timeFormat.format(result.elapsedMs / 1000)} δευτ.</p>
    </> : progress ? <>
      <p role="status">{progress.message}</p>
      {progress.stage === 'tracking' && typeof progress.fraction === 'number'
        && <progress aria-label="Πρόοδος ανάλυσης καρέ" value={progress.fraction} max={1} />}
    </> : <p>Ανάλυσε ένα βίντεο για να δεις ποιον έλεγχο πέρασε η λήψη.</p>}
  </section>;
}
