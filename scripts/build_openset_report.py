"""Report frozen open-set measurements without converting missing checks to passes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/open-set-v2'
OUT = ROOT / 'reports/open-set-v2'
NAMES = {
    'original_frozen_tcn': 'Προηγούμενο μοντέλο, αμετάβλητο',
    'A_frozen_tcn_recalibrated': 'A — νέα βαθμονόμηση',
    'B_tcn_diverse_other': 'B — εκπαίδευση με περισσότερα OTHER',
    'C_frozen_embedding_prototype': 'C — έλεγχος απόστασης embeddings',
    'nearest_centroid': 'Baseline nearest-centroid',
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def optional(path):
    path = Path(path)
    return read(path) if path.exists() else {'status': 'not_yet_tested'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pc(value):
    return '—' if value is None else f'{100 * value:.2f}%'

def check_status(value):
    if 'status' in value:
        return value['status']
    return 'passed' if value.get('passed') is True else 'failed' if value.get('passed') is False else 'in_progress'

def main():
    selection = read(ART / 'selection.locked.json')
    final = read(ART / 'final-evaluation.json')
    selected = selection['selected_candidate']
    if not selected:
        raise ValueError('No eligible selected candidate; do not silently install a fallback.')
    assert final['selected_before_final_test'] == selected
    assert final['selection_sha256'] == sha(ART / 'selection.locked.json')
    assert final['selection_changed_after_test'] is False
    config = read(ART / selected / 'config.json')
    export = selection['candidates'][selected]['export']
    assert export['model_sha256'] == sha(ART / selected / 'model.onnx')
    assert export['config_sha256'] == sha(ART / selected / 'config.json')
    verification = read(OUT / 'dataset_verification.json')
    packing = read(ROOT / 'data/open-set-v2/packing-audit.json')
    manifest = read(ROOT / 'data/open-set-v2/manifest.json')
    diagnosis = read(OUT / 'diagnosis-frozen-poc.json')
    failure = optional(OUT / 'final-failure-analysis.json')
    audit = optional(OUT / 'final_independent_audit.json')
    browser = optional(ROOT / 'web/qa-artifacts/open-set-v2/browser-comparison.json')
    stress = optional(ROOT / 'web/qa-artifacts/open-set-v2/stress-checks.json')
    production = optional(ROOT / 'web/qa-artifacts/open-set-v2/production-qa.json')
    if browser.get('selectionSha256') and browser['selectionSha256'] != sha(ART / 'selection.locked.json'):
        raise ValueError('Browser QA describes a different selection.')
    chosen = final['candidates'][selected]
    previous = final['references']['original_frozen_tcn']
    baseline = final['references']['nearest_centroid']
    known_correct = sum(row[i] for i, row in enumerate(chosen['closed_set_confusion_matrix']))
    accepted_correct = round(chosen['accepted_known_n'] * chosen['accepted_known_accuracy'])
    rows = []
    all_methods = {**final['references'], **final['candidates']}
    for name in NAMES:
        m = all_methods[name]
        rows.append(f"| {NAMES[name]} | {pc(m['closed_set_accuracy'])} | {m['closed_set_macro_f1']:.4f} | {m['accepted_known_n']}/{m['known_n']} ({pc(m['known_coverage'])}) | {pc(m['unknown_rejection_rate'])} | {m['unknown_false_acceptance_n']}/{m['unknown_n']} | {pc(m['all_accepted_accuracy'])} |")
    validation_rows = []
    for name, candidate in selection['candidates'].items():
        m = candidate['validation']
        validation_rows.append(f"| {NAMES[name]} | {m['accepted_known_n']}/{m['known_n']} | {pc(m['accepted_known_accuracy'])} | {m['unknown_false_acceptance_n']}/{m['unknown_n']} |")
    unknown_rows = []
    for label, m in chosen['per_unknown_category'].items():
        unknown_rows.append(f"| {label} | {m['n']} | {m['false_acceptance_n']} | {m['n'] - m['false_acceptance_n']} | {json.dumps(m['accepted_as'], ensure_ascii=False)} |")
    confusion = '\n'.join(f"| {label} | {' | '.join(map(str, row))} |" for label, row in zip(config['labels'], chosen['closed_set_confusion_matrix']))
    browser_rows = []
    for name, m in browser.get('summary', {}).items():
        if m.get('total'):
            browser_rows.append(f"| {NAMES[name]} | {m['acceptedKnownCorrect']}/{m['known']} | {m['falselyAcceptedUnknown']}/{m['unknown']} | {m['medianElapsedMs']/1000:.2f} s |")
    browser_table = ('| Browser σε ίδια MP4 | Σωστές αποδοχές γνωστών | Λάθος αποδοχές αγνώστων | Διάμεσος συνολικός χρόνος |\n'
                     '|---|---:|---:|---:|\n' + '\n'.join(browser_rows)) if browser_rows else 'Οι πραγματικές μετρήσεις browser δεν έχουν ολοκληρωθεί ακόμη.'
    originals = {row['id']: row for row in stress.get('rows', []) if row['transformation'] == 'original'}
    stress_summary = {}
    for transformation in ['brightness_0.6', 'horizontal_flip', 'translate_right_0.1_width_black']:
        variants = [row for row in stress.get('rows', []) if row['transformation'] == transformation]
        if not variants:
            continue
        changes = 0
        confidence_changes = []
        for row in variants:
            before = next(p for p in originals[row['id']]['predictions'] if p['candidate'] == selected)
            after = next(p for p in row['predictions'] if p['candidate'] == selected)
            changes += (before['accepted'], before['label']) != (after['accepted'], after['label'])
            confidence_changes.append(abs(before['confidence'] - after['confidence']))
        stress_summary[transformation] = {'samples': len(variants), 'changedDecisions': changes,
                                          'maximumAbsoluteConfidenceChange': max(confidence_changes),
                                          'minimumLandmarkAvailability': min(row['quality'] for row in variants)}
    stress_table = '\n'.join(f"| {name} | {item['changedDecisions']}/{item['samples']} | {item['maximumAbsoluteConfidenceChange']:.4f} |" for name, item in stress_summary.items())
    parity = packing['raw_preprocessing_parity']
    report = {
        'project': 'SignBridge AI — Open-Set Recognition V2', 'date': '2026-11-08',
        'modelVersion': config['modelVersion'], 'modelSha256': export['model_sha256'],
        'selectedCandidate': selected, 'selectedUsingValidationOnly': True,
        'metrics': {'test': chosen}, 'allMethodsFinal': all_methods,
        'selection': selection, 'finalEvaluation': final,
        'datasetVerification': verification, 'packingAudit': packing,
        'oldFailureDiagnosis': diagnosis, 'finalFailureAnalysis': failure,
        'independentAudit': audit, 'browserVerification': browser,
        'transformedImageStressChecks': stress, 'productionVerification': production,
        'selectedStressSummary': stress_summary,
        'humanWebcam': {'status': 'not_tested', 'savedRecordings': 0,
                       'reason': 'User could not complete recordings now; helper remains available.',
                       'knownSignWebcamAccuracy': 'not_tested', 'negativeWebcamFalseAcceptances': None},
        'decision': {'expandTo15SignsNow': False, 'publicCommunicationReady': False,
                     'reason': f"Selected method falsely accepts {chosen['unknown_false_acceptance_n']}/{chosen['unknown_n']} unseen-category examples; the simple baseline performs better but also fails open-set reliability.",
                     'nextExperiment': 'Compare a full-feature distance/centroid approach and a representation trained explicitly for separation on a new locked evaluation; add real ordinary-motion negatives and fluent-human tests.'},
        'limits': [
            'Final examples are new; the two signer identities were previously viewed. They remain absent from training and validation.',
            'Only three held-out unknown categories and nine final GEIA examples; not universal open-set evidence.',
            'All clips are whole utterances with exactly one official gloss, not the official isolated benchmark.',
            'Landmark availability is not a measurement of landmark accuracy.',
            'The selected motion threshold is recorded explicitly; a disabled motion gate does not prove idle-motion rejection.',
            'Brightness/mirror/translation variants are derived stress tests, not actual webcam users or lighting conditions.',
            'The better final baseline was not silently substituted for the validation-selected model.',
        ],
    }
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (OUT / 'technical-report.json').write_text(json_text, encoding='utf-8')
    public = ROOT / 'web/public/reports/technical-report.json'
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json_text, encoding='utf-8')
    text = f'''# SignBridge AI — Open-Set Recognition V2

8 Νοεμβρίου 2026 · `{config['modelVersion']}`

## Απόφαση

**Δεν προχωράμε ακόμη στα 10–15 νοήματα.** Η πραγματική νέα έκδοση βελτιώνει την
απόρριψη, αλλά παραμένει ανεπαρκής: δέχθηκε λανθασμένα
**{chosen['unknown_false_acceptance_n']}/{chosen['unknown_n']} άγνωστα δείγματα**.
Το αμετάβλητο προηγούμενο μοντέλο δέχθηκε {previous['unknown_false_acceptance_n']}/{previous['unknown_n']}
στα **ίδια νέα δεδομένα**. Η σύγκριση δεν συγχέει τα παλιά 9 άγνωστα με τα νέα 50.

Η απλή baseline αποδίδει καλύτερα στο τελικό test, αλλά και αυτή δέχθηκε
{baseline['unknown_false_acceptance_n']}/{baseline['unknown_n']} άγνωστα. Χρειάζεται νέο πείραμα με διαφορετικό
μηχανισμό διάκρισης και περισσότερη ποικιλία δεδομένων. Δεν αποδεικνύεται
αξιοπιστία για καθημερινή επικοινωνία ή χρήση χωρίς επίβλεψη.

## Τι βρέθηκε στο προηγούμενο μοντέλο

Τα 9 παλιά άγνωστα έγιναν δεκτά ως ΓΕΙΑ με confidence 0.6694–0.9952. Όλα είχαν
επαρκή παρουσία landmarks σε κάθε καρέ. Αυτό δεν μετρά γεωμετρική ακρίβεια tracking,
αλλά δεν υποστηρίζει την εξήγηση ότι η αποτυχία ήταν απλώς έλλειψη χεριών.
Επτά embeddings είχαν πλησιέστερο γνωστό prototype το ΓΕΙΑ, δύο το ΕΥΧΑΡΙΣΤΩ.
Οι πλήρεις αποστάσεις, logits και περιθώρια βρίσκονται στο `diagnosis-frozen-poc.json`.

Στα ήδη γνωστά διαγνωστικά δεδομένα, η CNN έμεινε σωστή σε 60/60 γνωστά μετά από
αντιστροφή χρόνου, τρεις σταθερές ανακατατάξεις και αντικατάσταση με τη μέση στάση.
Η baseline με τοπικό σχήμα χεριών πέτυχε 59/60, με πρόσωπο μόνο 41/60 και με
σημαίες παρουσίας 34/60. Ένας ξεχωριστός έλεγχος που χρησιμοποιεί μόνο το σενάριο
λήψης πέτυχε 30/60· το σενάριο **δεν αποτελεί είσοδο του πραγματικού μοντέλου**.

Αυτά δείχνουν ότι το μικρό αρχικό λεξιλόγιο δεν απαιτούσε αποδεδειγμένα χρονική
σειρά και εξηγούν την ισχυρή baseline. Το σχήμα χεριών είναι νόμιμη πληροφορία
νοηματισμού· η καλή απόδοσή του δεν αποδεικνύει άσχετη συντόμευση. Οι αλλοιώσεις
εισόδου είναι διαγνωστικές μεταβολές κατανομής, όχι απομονωμένη αιτιώδης απόδειξη.

## Νέα δεδομένα και ανεξαρτησία

478 πραγματικές ακολουθίες, 8.251 αυθεντικά RGB JPEG: 304 νέες ακολουθίες και
174 ρητά επαναχρησιμοποιημένες από training/validation. Επιλεκτική λήψη RGB,
χωρίς ολόκληρα αρχεία βάθους ή την πλήρη συλλογή. Όλα αποκωδικοποιήθηκαν και
ελέγχθηκαν με hashes/CRC. Δεν υπάρχουν ίδια καρέ μεταξύ συνόλων ή αντιστοιχίες
νέων καρέ με τα προηγούμενα 243 δείγματα.

| Σύνολο | Νοηματιστές | Γνωστά | Άγνωστα |
|---|---|---:|---:|
| Training | 1, 2, 4, 5 | 178 | 100 |
| Validation | 6 | 41 | 30 |
| Νέο τελικό test | 3, 7 | 79 | 50 |

Τα 129 τελικά παραδείγματα είναι αχρησιμοποίητα και οι ταυτότητες session διαφέρουν
από το προηγούμενο test. **Οι δύο άνθρωποι είχαν εμφανιστεί στις προηγούμενες
αξιολογήσεις**· δεν ισχυριζόμαστε νέες ταυτότητες ή επαληθευμένες διαφορετικές ημέρες
λήψης. Παραμένουν εκτός training/validation. Τα γνωστά του test είναι άνισα:
ΓΕΙΑ 9, ΕΥΧΑΡΙΣΤΩ 30, ΕΝΤΑΞΕΙ 40.

Άγνωστες κατηγορίες training: ΦΥΣΙΚΑ, ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ, ΟΚ, ΤΕΛΕΙΑ.
Validation: ΟΧΙ, ΚΑΛΟ. Τελικό test: ΝΑΙ, ΚΑΤΑΛΑΒΑΙΝΩ, ΗΜΕΡΑ_ΤΡΕΙΣ.
Οι τρεις τελικές κατηγορίες δεν χρησιμοποιήθηκαν στην εκπαίδευση ή στα όρια.
Η επίσημη αντιστοίχιση ΜΕΡΕΣ_ΤΡΕΙΣ → ΗΜΕΡΑ_ΤΡΕΙΣ διατηρεί και τις αρχικές ετικέτες.
Το ΟΚ παραμένει χωριστή επίσημη ετικέτα από το ΕΝΤΑΞΕΙ· η γλωσσική διάκρισή τους
χρειάζεται έλεγχο από ειδικό ΕΝΓ. Ισχύει CC BY 4.0, με πλήρη αναφορά στο
`data/open-set-v2/ATTRIBUTION.md`.

## Σύγκριση και προεπιλογή

A: αμετάβλητα βάρη της προηγούμενης CNN, νέα βαθμονόμηση/όρια.
B: ίδια αρχιτεκτονική, νέα πραγματική εκπαίδευση με τέσσερις κατηγορίες OTHER.
C: παλιά CNN και έλεγχος απόστασης των embeddings 48 διαστάσεων από γνωστά prototypes.
Οι μέσοι όροι, κλίμακες και όρια αποστάσεων του C προκύπτουν από training.
Και στις τρεις μεθόδους ελέγχονται παρουσία landmarks και μετρήσεις κίνησης.

Πριν ανοιχτεί το test, απαιτήθηκαν κάλυψη γνωστών ≥60% και ακρίβεια αποδεκτών
γνωστών ≥90%. Μεταξύ επιλέξιμων ρυθμίσεων προτιμήθηκαν οι λιγότερες ψευδείς
αποδοχές αγνώστων και μετά οι περισσότερες σωστές αποδοχές γνωστών.

| Validation | Αποδεκτά γνωστά | Ακρίβεια αποδεκτών γνωστών | Ψευδείς αποδοχές αγνώστων |
|---|---:|---:|---:|
{chr(10).join(validation_rows)}

Επιλέχθηκε η **{NAMES[selected]}**, πριν την τελική αξιολόγηση. Η επιλογή, τα μοντέλα,
τα configs και τα δεδομένα σφραγίστηκαν με hashes. Η καλύτερη baseline στο τελικό
test δεν αντικατέστησε εκ των υστέρων αυτή την επιλογή. Αποτελεί σημαντική ένδειξη
για το επόμενο πείραμα, που θα χρειαστεί νέο test.

## Ίδιο τελικό test για όλες τις μεθόδους

| Μέθοδος | Accuracy γνωστών | Macro F1 | Αποδεκτά γνωστά | Απόρριψη αγνώστων | Ψευδείς αποδοχές αγνώστων | Σωστά μεταξύ όλων των αποδεκτών |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Το accuracy και το Macro F1 βαθμολογούν την επιλογή μεταξύ των τριών γνωστών,
πριν από την απόρριψη. Δεν είναι accuracy σε αυθαίρετες κινήσεις. Στην B,
{known_correct}/79 γνωστά είχαν σωστή top-known ετικέτα, {accepted_correct}/{chosen['accepted_known_n']}
αποδεκτά γνωστά ήταν σωστά, αλλά {chosen['known_n'] - chosen['accepted_known_n']} γνωστά απορρίφθηκαν.
Οι {chosen['unknown_false_acceptance_n']} άγνωστες λανθασμένες αποδοχές μειώνουν τη συνολική ορθότητα
των εμφανιζόμενων απαντήσεων σε {accepted_correct}/{chosen['all_accepted_n']} ({pc(chosen['all_accepted_accuracy'])}).
Αυτό αφορά τη συγκεκριμένη αναλογία γνωστών/αγνώστων του test.

| Πραγματικό / πρόβλεψη B | ΓΕΙΑ | ΕΥΧΑΡΙΣΤΩ | ΕΝΤΑΞΕΙ |
|---|---:|---:|---:|
{confusion}

| Άγνωστη κατηγορία | Δείγματα | Λάθος αποδοχές B | Απορρίψεις B | Αποδεκτές ετικέτες |
|---|---:|---:|---:|---|
{chr(10).join(unknown_rows)}

Και τα 10 ΚΑΤΑΛΑΒΑΙΝΩ έγιναν δεκτά ως ΓΕΙΑ, με confidence περίπου
0.99917–0.999995 και επαρκή παρουσία landmarks σε όλα τα καρέ. Η υψηλή βαθμολογία
δεν είναι πιθανότητα ορθότητας σε άγνωστη κατηγορία. Οι 20 απορρίψεις γνωστών
αφορούν το ΕΝΤΑΞΕΙ. Αναλυτικά δεδομένα: `final-failure-analysis.json`.

## Όρια και κίνηση

Η B χρησιμοποιεί temperature {config['rejection']['temperature']}, confidence ≥
{config['rejection']['confidenceThreshold']}, ποιότητα ≥{config['rejection']['minQuality']} και απόρριψη
όταν κερδίζει η κατηγορία OTHER. Η επιλογή ελάχιστης κίνησης είναι
`{config['rejection']['motionSelection']}`: υπολογίζεται πραγματική μεταβολή των έγκυρων landmarks,
αλλά το validation δεν επέλεξε πρόσθετο κατώφλι κίνησης. Δεν ισχυριζόμαστε ότι
λύθηκε η απόρριψη ακινησίας. Ένα στατικό έγκυρο νόημα δεν πρέπει να βαφτίζεται
αυτόματα άγνωστο. Η μεγαλύτερη softmax τιμή δεν θεωρείται απόδειξη εγκυρότητας.

## Browser, πραγματική webcam και έλεγχοι

Η προεπεξεργασία Python/TypeScript συμφωνεί ακριβώς σε {len(parity)} πραγματικές
ακολουθίες. Κάθε μοντέλο εξήχθη πραγματικά σε ONNX με logits και embeddings.
Η αριθμητική συμφωνία, οι αποφάσεις απόρριψης και οι χρόνοι συγκρίνονται χωριστά
στον browser, σε 12 ίδια πραγματικά MP4 επιλεγμένα πριν εμφανιστούν αποτελέσματα.

- Browser σύγκριση: `{check_status(browser)}`.
- Παράγωγες δοκιμές φωτεινότητας/μετατόπισης/κατοπτρισμού: `{check_status(stress)}`.
- Τελικό χτισμένο UI: `{check_status(production)}`.
- Ανεξάρτητος αριθμητικός έλεγχος αποτελεσμάτων: `{check_status(audit)}`.

{browser_table}

Οι χρόνοι MP4 περιλαμβάνουν αποκωδικοποίηση, landmarks και ταξινόμηση στο
συγκεκριμένο μηχάνημα. Δεν είναι μόνο χρόνος ONNX και δεν αποδεικνύουν συνεχή
αναγνώριση πραγματικού χρόνου. Υπάρχουν επίσης 30 επιτυχείς έλεγχοι λογισμικού
στα `ml-test-summary.json`· τα συνθετικά inputs αυτών των tests δεν προσμετρώνται
στις μετρήσεις αναγνώρισης.

Οι μετασχηματισμοί πραγματικών καρέ δεν είναι δοκιμές πραγματικού φωτισμού ή
ανθρώπινης webcam. Ο κατοπτρισμός δεν θεωρείται αυτομάτως γλωσσικά ισοδύναμος.
Το UI και η είσοδος κάμερας παραμένουν μη κατοπτρισμένα.

| Παράγωγη δοκιμή B | Αλλαγές αποδοχής ή ετικέτας έναντι αρχικού | Μέγιστη απόλυτη μεταβολή confidence |
|---|---:|---:|
{stress_table}

Οι αλλαγές αυτές μετρούν ευαισθησία στο συγκεκριμένο μετασχηματισμό, όχι ακρίβεια
σε νέους ανθρώπους. Οι αρχικές και μετασχηματισμένες προβλέψεις διατηρούνται όλες.

**Πραγματική ανθρώπινη webcam: δεν δοκιμάστηκε.** Ο χρήστης δεν μπόρεσε να
ολοκληρώσει καταγραφές τώρα. Το `Start-Webcam-Check.cmd` παραμένει διαθέσιμο για
προαιρετικές τοπικές καταγραφές ακινησίας, καθημερινών κινήσεων, χεριών εκτός
κάδρου και αλλαγής θέσης/φωτισμού. Δεν δημιουργήθηκαν ψεύτικες ανθρώπινες λήψεις.
Δεν έχουμε πραγματική μέτρηση αναγνώρισης γνωστών νοημάτων από νέο χρήστη ΕΝΓ.

## Επόμενη απόφαση

Κρατάμε το λειτουργικό pipeline και αλλάζουμε την επόμενη πειραματική σύγκριση:
η πλήρης χρονική αναπαράσταση της απλής baseline αξίζει προτεραιότητα απέναντι
στο συμπιεσμένο embedding της CNN. Χρειάζονται νέα πραγματικά αρνητικά και
νέα ανέγγιχτη αξιολόγηση, καθώς τα σημερινά αποτελέσματα είναι πλέον γνωστά.
Οι υποθέσεις για τη συμπίεση, τη στατική στάση και τις λίγες αρνητικές κατηγορίες
δεν έχουν απομονωθεί ως μοναδική αιτία. **Η αύξηση του λεξιλογίου δεν αποτελεί
λύση για τις σημερινές ψευδείς αποδοχές.**

Αναπαραγωγή: `OPEN_SET_V2.md`. Αρχική εργασία και αρχειοθέτηση: `artifacts/poc-v2`,
`reports/poc-v2`. Δεν δημοσιεύτηκε παραγωγική εφαρμογή.
'''
    (OUT / 'TECHNICAL_REPORT.md').write_text(text, encoding='utf-8')
    print(json.dumps({'selected': selected, 'unknown_false_accepts': chosen['unknown_false_acceptance_n'],
                      'browser': check_status(browser), 'production': check_status(production)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
