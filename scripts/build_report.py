"""Assemble the measured experiment; never fill missing checks with successes."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))

def pct(value):
    return '—' if value is None else f'{value * 100:.2f}%'

def main():
    training = read('artifacts/report.json')
    manifest = read('data/manifest.json')
    verify = read('reports/dataset_verification.json')
    parity = read('data/features-preprocessing-parity.json')
    failure = read('reports/failure_analysis.json')
    browser_path = ROOT / 'web/qa-artifacts/real-model-qa.json'
    browser = json.loads(browser_path.read_text(encoding='utf-8')) if browser_path.exists() else {'status': 'not_tested'}
    audit_path = ROOT / 'reports/independent_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {'status': 'not_saved'}
    version = training['modelVersion']
    model_hash = hashlib.sha256((ROOT / 'artifacts/model.onnx').read_bytes()).hexdigest()
    if training['export']['onnx_sha256'] != model_hash:
        raise ValueError('Training report does not describe the current model bytes.')
    if browser.get('modelVersion') != version or browser.get('modelSha256') != model_hash:
        raise ValueError('Run browser QA on the current trained model before building the report.')
    if audit.get('modelVersion') != version or audit.get('modelSha256') != model_hash:
        raise ValueError('Independent audit must match the current trained model.')
    if failure.get('model_version') != version or failure['artifact_sha256']['model.onnx'] != model_hash:
        raise ValueError('Failure diagnosis must match the current trained model.')
    extraction = [read(f'reports/extraction-{shard}.json') for shard in range(3)]
    pipeline_hash = hashlib.sha256()
    for name in ['src/features.ts', 'src/pipeline.ts', 'src/recognizer.worker.ts']:
        pipeline_hash.update(name.encode())
        pipeline_hash.update((ROOT / 'web' / name).read_bytes())
    if any(item['pipeline_sha256'] != pipeline_hash.hexdigest() for item in extraction):
        raise ValueError('Browser inference source differs from the recorded extractor.')
    production_path = ROOT / 'web/qa-artifacts/production-qa.json'
    production = json.loads(production_path.read_text(encoding='utf-8')) if production_path.exists() else {'status': 'not_yet_tested'}
    if production.get('modelVersion') and (production['modelVersion'] != version or production['modelSha256'] != model_hash):
        raise ValueError('Production QA describes a different model.')
    known = training['metrics']['test']
    baseline = training['baseline']['metrics']['test']
    correct = sum(row[i] for i, row in enumerate(known['closed_set_confusion_matrix']))
    baseline_correct = sum(row[i] for i, row in enumerate(baseline['closed_set_confusion_matrix']))
    accepted_correct = round((known['accepted_known_accuracy'] or 0) * known['accepted_known_n'])
    baseline_accepted_correct = round((baseline['accepted_known_accuracy'] or 0) * baseline['accepted_known_n'])
    videos = browser.get('videos', [])
    video_known = [v for v in videos if v['is_known']]
    video_unknown = [v for v in videos if not v['is_known']]
    video_correct = sum(v.get('accepted_and_correct') is True for v in video_known)
    video_false = sum(v.get('unknown_false_accept') is True for v in video_unknown)
    video_times = [v['elapsedMs'] / 1000 for v in videos if v.get('elapsedMs')]
    video_time_text = (f'{min(video_times):.2f}–{max(video_times):.2f} δευτερόλεπτα, '
                       f'διάμεσος {statistics.median(video_times):.2f}') if video_times else 'δείτε τα αναλυτικά χρονόμετρα του browser QA'
    repeated_path = ROOT / 'web/qa-artifacts/repeatability-v2-reset-probe.json'
    repeatability = json.loads(repeated_path.read_text(encoding='utf-8')) if repeated_path.exists() else {'status': 'not_saved'}
    rejection = training['rejection']
    with (ROOT / 'artifacts/predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    unknown_rows = [r for r in predictions if r['split'] == 'test' and r['true_y'] == '-1']
    unknown_score_min = min(float(r['confidence']) for r in unknown_rows)
    unknown_score_max = max(float(r['confidence']) for r in unknown_rows)
    unknown_quality_min = min(float(r['quality']) for r in unknown_rows)
    unknown_labels = ', '.join(sorted({r['top_known_label'] for r in unknown_rows}))
    known_error_labels = ', '.join(sorted({r['raw_label'] for r in predictions if r['split'] == 'test' and r['true_y'] != '-1' and r['true_y'] != r['top_known_y']})) or 'καμία'
    report = {
        'date': '2026-11-08', 'project': 'SignBridge AI', 'modelVersion': training['modelVersion'],
        'modelSha256': model_hash,
        'evaluationContext': training.get('evaluationContext'),
        'decision': {
            'scope': 'local scientific proof of concept for three annotated GSL glosses',
            'knownSignClassification': f'{correct}/{known["known_n"]} correct top-known predictions on two held-out research-dataset signers',
            'unknownRejection': f'FAILED: {known["unknown_false_acceptance_n"]}/{known["unknown_n"]} held-out unknown examples accepted',
            'publicCommunicationReadiness': False,
            'expandTo15SignsNow': False,
            'liveHumanWebcamAccuracy': 'NOT TESTED: user confirmed no fluent GSL participant is currently available',
        },
        'dataset': training['dataset'], 'datasetVerification': verify,
        'metrics': training['metrics'], 'baseline': training['baseline'],
        'rejection': training['rejection'], 'export': training['export'],
        'preprocessingParity': {k: parity[k] for k in ['status', 'max_abs_error', 'atol', 'rtol']},
        'preprocessingFixtureCount': len(parity['examples']),
        'browserVerification': browser, 'independentAudit': audit, 'clipIsolationCheck': repeatability,
        'productionVerification': production, 'extractionProvenance': extraction,
        'experimentHistory': 'v1 is preserved. The corrected run isolates each clip and fixes a transparent first video frame. Same model protocol and signer split; this is a repeated holdout after a preprocessing bug fix, not a new untouched test.',
        'failureAnalysis': failure,
        'limitations': training['limitations'] + [
            'Only one unknown semantic category per partition; 5 validation and 9 test negative examples.',
            'Official single-gloss utterances, not the official isolated benchmark; lexical annotation verified, no new expert review.',
            'Three known classes and two test people from the same capture environment do not establish deployment generalization.',
            'No idle/everyday-motion negative collection or physical webcam recognition evaluation was performed.',
            'No model or threshold changes were made after inspecting final test results.',
            'Future model selection needs new untouched test examples; these test outcomes are now known.',
        ],
    }
    (ROOT / 'reports/technical-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    target = ROOT / 'web/public/reports/technical-report.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    split_rows = []
    for split, title in [('train', 'Εκπαίδευση'), ('val', 'Validation'), ('test', 'Ανεξάρτητο test')]:
        info = training['dataset']['splits'][split]
        metric = training['metrics'][split]
        split_rows.append(f"| {title} | {', '.join(info['signers'])} | {metric['known_n']} | {metric['unknown_n']} | {pct(metric['closed_set_accuracy'])} |")
    confusion = '\n'.join(f"| {label} | {' | '.join(map(str, row))} |" for label, row in zip(manifest['classes'], known['closed_set_confusion_matrix']))
    browser_status = browser.get('status', browser.get('passed', 'see JSON'))
    text = f'''# SignBridge AI — πραγματική τεχνική αναφορά

Ημερομηνία: 8 Νοεμβρίου 2026. Μοντέλο: `{training['modelVersion']}`.

## Απόφαση

Ο πραγματικός μηχανισμός εκπαιδεύτηκε και εξήχθη σε ONNX. Στους χωριστούς
νοηματιστές του test, η επιλογή μεταξύ των τριών γνωστών ετικετών ήταν σωστή σε
{correct}/{known['known_n']} δείγματα. Η τελική πύλη αποδοχής εμφάνισε πρόβλεψη μόνο σε
{known['accepted_known_n']}/{known['known_n']} γνωστά δείγματα. **Η απόρριψη αγνώστων απέτυχε: {known['unknown_false_acceptance_n']}/{known['unknown_n']} άγνωστα
δείγματα έγιναν λανθασμένα αποδεκτά.** Δεν τεκμηριώνεται ακόμη αξιόπιστη
επικοινωνία σε ανοιχτό λεξιλόγιο, ούτε ετοιμότητα για δημόσια χρήση. Δεν προτείνεται
επέκταση σε 10–15 νοήματα πριν αντιμετωπιστεί αυτό το πρόβλημα.

## Πραγματικά δεδομένα

- 243 πλήρεις ακολουθίες RGB, 4.203 πρωτότυπα JPEG, 648×480. Καμία συνθετική κίνηση
  ή προκαθορισμένη απάντηση στην εκπαίδευση ή στις μετρήσεις.
- 209 γνωστά δείγματα: ΓΕΙΑ 69, ΕΥΧΑΡΙΣΤΩ 70, ΕΝΤΑΞΕΙ 70. Επιπλέον 34 δείγματα
  διαφορετικών νοημάτων χρησιμοποιήθηκαν ως αρνητικά.
- Τα δεδομένα είναι δικό μας υποσύνολο πλήρων εκφωνήσεων με **ακριβώς μία επίσημη
  ετικέτα νοήματος**, από το continuous GSL. Δεν είναι το επίσημο isolated benchmark.
- Λήψη μόνο συγκεκριμένων RGB καρέ μέσω HTTP byte ranges. Δεν κατέβηκε ολόκληρη
  η συλλογή 134,8 GB ή κάποιο αρχείο βάθους.
- Έγινε αποκωδικοποίηση όλων των εικόνων και έλεγχος hashes, συνεχειών καρέ και
  αντιστοίχισης ετικέτας–νοηματιστή. Δεν βρέθηκαν ίδια καρέ μεταξύ των συνόλων.
- Άδεια CC BY 4.0. Πλήρης αναφορά δημιουργών: `data/ATTRIBUTION.md`.

Οι επίσημες αντιστοιχίσεις κειμένου είναι ΓΕΙΑ → «γεια σας», ΕΥΧΑΡΙΣΤΩ →
«ευχαριστώ», ΕΝΤΑΞΕΙ → «εντάξει». Τα επιλεγμένα ΓΕΙΑ προέρχονται από το κλείσιμο
διαλόγου. Έγινε έλεγχος των επισημάνσεων· δεν έχει γίνει νέα γλωσσική πιστοποίηση
από συνεργάτη ΕΝΓ.

| Σύνολο | Νοηματιστές | Γνωστά | Άγνωστα | Top-known accuracy |
|---|---|---:|---:|---:|
{chr(10).join(split_rows)}

Αρνητικές κατηγορίες: ΦΥΣΙΚΑ στην εκπαίδευση, ΟΧΙ στο validation,
ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ στο test. Οι κατηγορίες και οι νοηματιστές είναι χωριστοί.
Συνολικά: τέσσερα άτομα στην εκπαίδευση, ένα στο validation, δύο στο test.

## Μηχανισμός

Πραγματικά καρέ → MediaPipe Holistic 1.2.2 στον browser → 224 χαρακτηριστικά
χεριών/άνω σώματος/επιλεγμένων σημείων προσώπου → 48 χρονικά δείγματα → Temporal
CNN PyTorch → ONNX Runtime Web/WASM.

Το ίδιο MediaPipe και το ίδιο TypeScript preprocessing χρησιμοποιούνται για την
εξαγωγή του dataset και τη web εφαρμογή. Η ανεξάρτητη Python υλοποίηση έδωσε
**ακριβώς ίδια αποτελέσματα σε {len(parity['examples'])} πραγματικές ακολουθίες**.
Τα x/y διατηρούν τη θέση των χεριών ως προς το σώμα και το σχήμα τους τοπικά.
Απουσίες landmarks έχουν ρητές σημαίες. Δεν συγχωνεύονται ασύμβατες συντεταγμένες z.

Η κανονικοποίηση μαθαίνεται αποκλειστικά από το training και περιέχεται στο ONNX.
Για τη συγκεκριμένη έκδοση έγινε μία εκπαίδευση με seed {training['seed']}, επιλογή checkpoint στο epoch
{training['training']['best_epoch']} μέσω validation και πρόωρη διακοπή μετά από
{len(training['training']['history'])} epochs. Δεν έγινε αλλαγή μοντέλου ή ορίων
με βάση το τελικό test. Διατηρείται η αρχική v1 εκτέλεση στα `artifacts/v1` και
`reports/v1`. Η διορθωμένη επανάληψη απομονώνει τον ανιχνευτή ανά βίντεο και
διορθώνει το κενό αρχικό καρέ του browser. Χρησιμοποιεί το ίδιο πρωτόκολλο και
τους ίδιους χωριστούς νοηματιστές. **Είναι επανάληψη του holdout μετά από διόρθωση
του pipeline, όχι νέο ανέγγιχτο test.**

## Ανεξάρτητα αποτελέσματα

| Μετρική test | Temporal CNN | Απλή baseline nearest-centroid |
|---|---:|---:|
| Ακρίβεια γνωστών πριν την απόρριψη | {pct(known['closed_set_accuracy'])} ({correct}/{known['known_n']}) | {pct(baseline['closed_set_accuracy'])} ({baseline_correct}/{baseline['known_n']}) |
| Macro F1 γνωστών | {known['closed_set_macro_f1']:.4f} | {baseline['closed_set_macro_f1']:.4f} |
| Αποδεκτά γνωστά | {known['accepted_known_n']}/{known['known_n']} ({pct(known['known_coverage'])}) | {baseline['accepted_known_n']}/{baseline['known_n']} ({pct(baseline['known_coverage'])}) |
| Σωστά μεταξύ αποδεκτών γνωστών | {accepted_correct}/{known['accepted_known_n']} ({pct(known['accepted_known_accuracy'])}) | {baseline_accepted_correct}/{baseline['accepted_known_n']} ({pct(baseline['accepted_known_accuracy'])}) |
| Λανθασμένες αποδοχές άγνωστης κατηγορίας | {known['unknown_false_acceptance_n']}/{known['unknown_n']} ({pct(known['unknown_false_acceptance_rate'])}) | {baseline['unknown_false_acceptance_n']}/{baseline['unknown_n']} ({pct(baseline['unknown_false_acceptance_rate'])}) |
| Σωστά μεταξύ ΟΛΩΝ των αποδεκτών | {accepted_correct}/{known['all_accepted_n']} ({pct(known['all_accepted_accuracy'])}) | {baseline_accepted_correct}/{baseline['all_accepted_n']} ({pct(baseline['all_accepted_accuracy'])}) |

Η ακρίβεια {pct(known['closed_set_accuracy'])} περιλαμβάνει όλα τα γνωστά δείγματα, ακόμη και εκείνα που
αργότερα απορρίπτονται. Εξαιρεί τα άγνωστα και αγνοεί την πύλη αποδοχής.
Δεν είναι ακρίβεια ολόκληρης της εφαρμογής σε αυθαίρετες κινήσεις.
Η σύγκριση με την απλή baseline πρέπει να διατηρείται σε κάθε επόμενο πείραμα.
Η παρούσα μέτρηση δεν τεκμηριώνει πλεονέκτημα του CNN έναντι της baseline.

Πίνακας σύγχυσης γνωστών: γραμμές πραγματικό νόημα, στήλες top-known πρόβλεψη,
πριν την απόρριψη.

| Πραγματικό / πρόβλεψη | ΓΕΙΑ | ΕΥΧΑΡΙΣΤΩ | ΕΝΤΑΞΕΙ |
|---|---:|---:|---:|
{confusion}

## Τι απέτυχε και τι γνωρίζουμε

Οι top-known προβλέψεις των αγνώστων ήταν {unknown_labels}, με βαθμολογίες
{unknown_score_min:.4f}–{unknown_score_max:.4f}, και ελάχιστη ποιότητα tracking
{unknown_quality_min:.4f}. Οι λανθασμένες αποδοχές δεν εξηγούνται αυτόματα ως
απώλεια landmarks. Το validation περιείχε μόνο
πέντε αρνητικά από μία διαφορετική κατηγορία, στην οποία η απόρριψη πέτυχε.

Η εκμάθηση μιας μόνο αρνητικής κατηγορίας και η βαθμονόμηση σε άλλη μία είναι
εύλογη εξήγηση της ανεπαρκούς κάλυψης άγνωστων κινήσεων. Δεν αποτελεί πειραματικά
απομονωμένη αιτία. Οι αστοχίες στα γνωστά αφορούν: {known_error_labels}. Παρατηρήθηκαν
επίσης χαμένα χέρια σε ορισμένα training δείγματα. Η αναλυτική διάγνωση διατηρείται
στο `reports/failure_analysis.json`.

Η πύλη αποδοχής έχει temperature {rejection['temperature']}, confidence threshold
{rejection['confidence_threshold']:.10f} και ελάχιστη ποιότητα {rejection['min_quality']}.
Ρυθμίστηκε μόνο στο validation. Το ελάχιστο ποιότητας είναι
τεχνικός κανόνας εγκυρότητας, όχι εγγύηση αναγνώρισης αγνώστων.

## Μεταφορά στον browser

- ONNX: {training['export']['onnx_bytes']:,} bytes.
- PyTorch / ONNX: 12 πραγματικά fixtures, μέγιστη απόλυτη διαφορά logits
  {training['export']['max_abs_error']:.3g}, ίδιες προβλέψεις.
- Κατάσταση καταγεγραμμένου browser QA: `{browser_status}`. Δείτε
  `web/qa-artifacts/real-model-qa.json` για κάθε πραγματικό MP4, αριθμητικό έλεγχο
  και τυχόν σφάλμα. Η επιτυχία εκτέλεσης του pipeline δεν σημαίνει σωστή μετάφραση.
- Υπάρχουν 8 MP4 που προήλθαν από πραγματικά test καρέ, με πλήρη αντιστοίχιση
  στο `data/clips/mapping.json`. Η μετακωδικοποίηση δηλώνεται ρητά.
- Στα ξεχωριστά browser παραδείγματα: {video_correct}/{len(video_known)} γνωστά
  έγιναν σωστά αποδεκτά και {video_false}/{len(video_unknown)} άγνωστα έγιναν
  λανθασμένα αποδεκτά. Δεν αντικαθιστούν το test των 60 γνωστών ακολουθιών.
- Το UI επιτρέπει upload, κάμερα, αντίστροφη μέτρηση, επιλογή σύντομης διάρκειας,
  ακύρωση, εμφάνιση πραγματικής πρόβλεψης ή απόρριψης.
- Χρόνοι των καταγεγραμμένων αναλύσεων MP4: {video_time_text}.
  Πρόκειται για ανάλυση σύντομου αποσπάσματος, όχι συνεχή μετάφραση πραγματικού χρόνου.
- Οι λήψεις κάμερας συλλέγονται πρώτα και αναλύονται μετά σε worker, ώστε η CPU
  αναγνώριση να μην καθορίζει τον ρυθμό λήψης. Δεν έχει μετρηθεί πραγματική κάμερα.

Ο χρόνος ONNX μόνος του (median {training['export']['python_onnx_cpu_classifier_ms']['median']:.3f} ms)
δεν περιλαμβάνει landmarks, αποκωδικοποίηση, λήψη ή UI. Δεν αποτελεί χρόνο απόκρισης
ολόκληρης της εφαρμογής. Οι πραγματικοί browser χρόνοι καταγράφονται χωριστά.
Κατάσταση ελέγχου του χτισμένου UI, ανεξάρτητα από τον development server:
`{production['status']}`. Η αντίστοιχη τεκμηρίωση βρίσκεται στο
`web/qa-artifacts/production-qa.json` όταν ο έλεγχος έχει εκτελεστεί.

## Εκκρεμότητες πριν από επέκταση

1. Περισσότερες αρνητικές κατηγορίες και πραγματικές καθημερινές κινήσεις, με νέο
   ανέγγιχτο test. Τα αποτελέσματα του σημερινού test είναι πλέον γνωστά και δεν
   πρέπει να χρησιμοποιηθούν για επιλογή ορίων και να παρουσιαστούν ξανά ως ανεξάρτητα.
2. Έλεγχος δυσκολότερων οπτικά όμοιων νοημάτων και εναλλακτικών μηχανισμών απόρριψης,
   κρατώντας την απλή baseline ως υποχρεωτική σύγκριση.
3. Δοκιμές με άτομα που γνωρίζουν ΕΝΓ σε νέες κάμερες, φωτισμό και περιβάλλον.
   Ο χρήστης επιβεβαίωσε ότι δεν υπάρχει ακόμη διαθέσιμος νοηματιστής.
4. Γλωσσική επαλήθευση από συνεργάτη ΕΝΓ και μεγαλύτερη ποικιλία πραγματικών χρηστών.

**Η ακρίβεια με πραγματική webcam παραμένει μη δοκιμασμένη.** Δεν εκτελέστηκε
προσομοιωμένη webcam για να παρουσιαστεί ως άνθρωπος. Δεν έγινε δημοσίευση
παραγωγικής εφαρμογής ή επέκταση του λεξιλογίου μετά την αποτυχία απόρριψης.
'''
    (ROOT / 'reports/TECHNICAL_REPORT.md').write_text(text, encoding='utf-8')
    print(json.dumps({'report': 'reports/TECHNICAL_REPORT.md', 'browser_status': browser_status, 'known_test_accuracy': known['closed_set_accuracy'], 'unknown_false_accepts': known['unknown_false_acceptance_n']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
