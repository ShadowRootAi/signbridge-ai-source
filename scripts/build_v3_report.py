"""Build the V3 report from actual saved measurements; never fit a model."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/v3'
NAMES = {'frozen_reference': 'Παγωμένη baseline V2', 'recalibrated_original_pool': 'Baseline με νέα βαθμονόμηση',
         'augmented_pool': 'Περισσότερα αρνητικά, ενιαία OTHER', 'separate_negative_prototypes': 'Χωριστά αρνητικά πρότυπα',
         'CNN_B': 'CNN V2 (B)', 'full_feature_centroid': 'Παγωμένη baseline V2', 'frozen_C': 'Embedding gate C'}


def read(path): return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))
def pct(value): return '—' if value is None else f'{100*value:.1f}%'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def selective(rows):
    classes = ['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ']; confusion = np.zeros((4, 4), dtype=int)
    for row in rows:
        truth = classes.index(row['label']) if row['label'] in classes else 3
        label = row.get('predicted', row.get('predicted_known'))
        predicted = classes.index(label) if row['accepted'] else 3
        confusion[truth, predicted] += 1
    f1 = []
    for c in range(3):
        denominator = confusion[c, :].sum() + confusion[:, c].sum()
        f1.append(float(2*confusion[c, c]/denominator) if denominator else 0.)
    return {'known_macro_f1_with_rejections_and_unknown_false_acceptances': float(np.mean(f1)),
            'per_known_class_f1_with_rejections': f1, 'open_set_confusion_matrix': confusion.tolist(),
            'matrix_order': classes + ['UNKNOWN / rejection']}


def metric_table(items):
    lines = ['| Μέθοδος | Σωστή αποδοχή γνωστών | Ψευδής αποδοχή αγνώστων | Precision αποδεκτών | Macro F1 κλειστού συνόλου | Macro F1 των 3 γνωστών με απορρίψεις/άγνωστα |',
             '|---|---:|---:|---:|---:|---:|']
    for name, item in items.items():
        m = item['metrics']; s = selective(item['predictions'])
        correct = sum(row.get('known', row['label'] in ['ΓΕΙΑ','ΕΥΧΑΡΙΣΤΩ','ΕΝΤΑΞΕΙ']) and row['accepted'] and row['label'] == row.get('predicted',row.get('predicted_known')) for row in item['predictions'])
        lines.append(f"| {NAMES.get(name,name)} | {correct}/{m['known_n']} ({pct(m['correct_known_acceptance_rate'])}) | {m['unknown_false_acceptance_n']}/{m['unknown_n']} ({pct(m['unknown_false_acceptance_rate'])}) | {pct(m['all_accepted_accuracy'])} | {m['closed_set_macro_f1']:.3f} | {s['known_macro_f1_with_rejections_and_unknown_false_acceptances']:.3f} |")
    return '\n'.join(lines)


def build():
    dev = read('reports/v3/centroid-development-results.json')
    reg = read('reports/v3/centroid-exposed-regression.json')
    temporal = read('reports/v3/all-exposed-temporal-ablations.json')
    numerical = read('reports/v3/numerical-diagnosis.json')
    packing = read('data/v3/packing-audit.json')
    browser_path = ROOT / 'reports/v3/browser-comparison.json'
    browser = read('reports/v3/browser-comparison.json') if browser_path.exists() else None
    audit_path = ROOT / 'reports/v3/centroid-independent-audit.json'
    audit = read('reports/v3/centroid-independent-audit.json') if audit_path.exists() else None
    browser_lines = ['**Έλεγχος νέων ONNX:** αναμένεται ολοκληρωμένο browser report.']
    if browser and browser.get('passed'):
        browser_lines = ['**Ολοκληρωμένοι έλεγχοι:**344 validation περιπτώσεις (86×4 μέθοδοι) με αριθμητική συμφωνία NumPy/PyTorch/desktop ONNX/browser ONNX και ακριβώς ίδιες αποφάσεις απόρριψης. Επιπλέον48 εκτελέσεις πραγματικών MP4 (12 ίδια αρχεία×4 μέθοδοι), χωρίς προσομοιωμένη κάμερα.', '',
                         '| Μέθοδος | MP4 γνωστά σωστά αποδεκτά /6 | MP4 άγνωστα ψευδώς αποδεκτά /6 | Warm classifier median / p95 | Πλήρης αλυσίδα median |',
                         '|---|---:|---:|---:|---:|']
        for name, item in browser['summary'].items():
            timing=item['warmClassifierTiming']; full=item['fullPipelineTiming']
            browser_lines.append(f"| {NAMES.get(name, 'Παγωμένη baseline V2')} | {item['mp4AcceptedKnownCorrect']} | {item['mp4FalseAcceptedUnknown']} | {timing['medianMs']:.3f} / {timing['p95Ms']:.3f} ms | {full['medianMs']/1000:.3f} s |")
        browser_lines += ['', 'Το warm timing είναι50 επαναλήψεις ενός validation fixture μετά από προθέρμανση. Η πλήρης αλυσίδα περιλαμβάνει νέο worker, φόρτωση/αποκωδικοποίηση, MediaPipe και απόφαση, με ήδη τοπικά διαθέσιμα assets. Οι λίγες δεκάδες ms διαφοράς μεταξύ μεθόδων δεν τεκμηριώνουν πλεονέκτημα ταχύτητας· το μεγαλύτερο κόστος είναι η εξαγωγή landmarks. Δεν μετρήθηκε δημόσιο δίκτυο, κινητό ή latency ζωντανής ανθρώπινης εκτέλεσης.']
    validation = {'frozen_reference': dev['frozen_reference_validation'], **{name: v['validation'] for name,v in dev['methods'].items()}}
    b_rows = [row for row in temporal['predictions'] if row['model']=='CNN_B' and row['variant']=='normal']
    comparison = {'CNN_B': {'metrics': temporal['results']['normal']['CNN_B'], 'predictions': b_rows},
                  **{name: reg['methods'][name] for name in ('frozen_reference','recalibrated_original_pool','augmented_pool','separate_negative_prototypes')}}
    summary = {
        'version': 'SignBridge V3 diagnosis and adaptive development, 2026-11-08',
        'fresh_final_evaluation_completed': False, 'human_webcam_tested': False, 'publicly_published': False,
        'new_CNN_training': False, 'production_model_changed': False,
        'development_selected_candidate': dev['fit']['selected_development_candidate'],
        'production_recommendation': 'No validated replacement. Augmented pooled centroid is a candidate for independent testing, not a certified deployment choice; validation rejects all11 GEIA.',
        'expand_to_10_15_signs': False,
        'validation': {name: {'metrics': item['metrics'], **selective(item['predictions'])} for name,item in validation.items()},
        'exposed_regression': {name: {'metrics': item['metrics'], **selective(item['predictions'])} for name,item in comparison.items()},
        'data': packing['outputs'],
        'browser_evidence': browser,
        'independent_audit': audit,
        'evidence_hashes': {str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob('*.json')) if p.name != 'technical-report.json'},
        'metrics_notes': {
            'known_recall': 'Correct accepted known / all known. Rejecting a known sign is a miss.',
            'accepted_precision': 'Correct accepted known / all accepted, including false accepted unknowns in denominator.',
            'closed_set_macro_f1': 'Forced best-known prediction; ignores rejection and all unknowns, so1.0 does not establish usability.',
            'selective_known_macro_f1': 'Macro over3 known classes; rejection is a false negative, accepted unknown is a false positive.',
            'regression': 'Same129 already exposed V2final clips. Category-driven changes make this adaptive regression, not independent confirmation.',
        },
    }
    (OUT/'technical-report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = [
        '# SignBridge AI — V3: διάγνωση, μικρό πείραμα και προετοιμασία δημοσίευσης',
        '', 'Ημερομηνία: 8 Νοεμβρίου 2026.', '',
        '**Απόφαση:** δεν επεκτείνουμε ακόμη στα 10–15 νοήματα. Η ενισχυμένη ενιαία OTHER είναι υποψήφια για νέα ανεξάρτητη δοκιμή, αλλά δεν αποτελεί επαληθευμένη αντικατάσταση της εφαρμογής. Απορρίπτει όλα τα «ΓΕΙΑ» του validation. Δεν έγινε νέα μεγάλη εκπαίδευση CNN, δημοσίευση ή δοκιμή ανθρώπου με webcam.',
        '', '## 1. Τι έδειξε η διάγνωση των 10 λαθών', '',
        'Τα αρχικά annotations συμφωνούν: και τα10 αποσπάσματα είναι **ΚΑΤΑΛΑΒΑΙΝΩ**, με ελληνικό κείμενο **«κατάλαβα»**, χωρίς άρνηση. Το ίδιο ισχύει και για τα25 αχρησιμοποίητα αποσπάσματα αυτής της ετικέτας. Δεν αλλάξαμε καμία ετικέτα.', '',
        'Ελέγχθηκαν και τα547 ήδη αποκτημένα αποσπάσματα ως προς τα επίσημα frame intervals· τα πλήθη καρέ συμφωνούν. Στις10 περιπτώσεις αποτυχίας υπάρχουν12–18 καρέ στα30fps. Σε μία περίπτωση το αρχικό timestamp interval διαφέρει κατά ένα καρέ από το αρχικό frame interval. Δεν πρόκειται για τοπική απώλεια και δεν εξηγεί10/10 συστηματικά λάθη. Τα χέρια βρίσκονται ήδη κοντά στο κεφάλι στην αρχή/λήξη: η γλωσσική πληρότητα των ορίων χρειάζεται ειδικό ΕΝΓ.', '',
        'Στις πραγματικές εικόνες το «ΚΑΤΑΛΑΒΑΙΝΩ» έχει έντονα εκτεταμένο δείκτη με τα υπόλοιπα δάχτυλα περισσότερο κλειστά. Τα συγκρίσιμα «ΓΕΙΑ» έχουν γενικά πιο επίπεδη παλάμη και κίνηση από το μέτωπο προς τα έξω. Αυτή είναι οπτική περιγραφή, όχι νέα γλωσσική επιβεβαίωση. Οι δύο κατηγορίες μοιράζονται τη γενική θέση του χεριού κοντά στο πρόσωπο.', '',
        '**Βρέθηκε ουσιαστικό κενό στον έλεγχο tracking:** και οι19 ακολουθίες έχουν quality=1, αλλά αυτό σημαίνει pose + τουλάχιστον ένα χέρι, όχι πλήρη καταγραφή του ενεργού χεριού. Σε8/19 υπάρχει απώλεια του δεξιού χεριού (5 ΓΕΙΑ,3 ΚΑΤΑΛΑΒΑΙΝΩ), ενώ το αριστερό χέρι σε ηρεμία κρατά το quality υψηλό. Στο πραγματικό raw ΚΑΤΑΛΑΒΑΙΝΩ ατόμου3/επανάληψη2 λείπει το ενεργό χέρι σε6/15 καρέ· η επανάληψη3 χάνει1/13. Οι καρποί αντιστοιχούν στις αναμενόμενες πλευρές όταν ανιχνεύονται. Δεν υπάρχουν χειροκίνητες ground-truth συντεταγμένες για ακρίβεια εντόπισης. Τα7 υπόλοιπα ΚΑΤΑΛΑΒΑΙΝΩ αποτυγχάνουν παρότι το δεξί χέρι έχει πλήρη παρουσία, συνεπώς το dropout δεν εξηγεί όλα τα10 λάθη.', '',
        '![Πραγματικά καρέ και υπάρχοντα hand landmarks](tracking-overlay.png)', '',
        'Η τοπική προεπεξεργασία δεν καθρεφτίζει εικόνες ή ταξινομεί χέρια βάσει πλευράς οθόνης. Διατηρεί χωριστά left/right κανάλια. Δεν τεκμηριώθηκε ότι κάθε αρχική καταγραφή του dataset είναι ανεστραμμένη ή μη ανεστραμμένη. Η παλιότερη δοκιμή ψηφιακού flip έδειξε ευαισθησία και δεν ισοδυναμεί με ανθρώπινη δοκιμή κάμερας.', '',
        'Οι διαφορές σχήματος **παραμένουν στα υπάρχοντα XY χαρακτηριστικά**. Για το ενεργό δεξί χέρι, οι μέσες αποστάσεις δείκτη/μέσου/παράμεσου/μικρού από τον καρπό είναι περίπου1,76/1,82/1,79/1,64 μονάδες παλάμης για «ΓΕΙΑ» και1,86/0,96/0,87/0,80 για «ΚΑΤΑΛΑΒΑΙΝΩ». Άρα δεν δικαιολογείται το συμπέρασμα ότι η προεπεξεργασία έχει καταστρέψει όλη τη διακριτική πληροφορία.', '',
        '![Πραγματικά χαρακτηριστικά σχήματος χεριού](local-hand-shape-proxies.png)', '',
        'Το CNN συμπιέζει10.752 εισόδους σε48 μέσες χρονικές ενεργοποιήσεις. Στον χώρο αυτό τα λάθος δείγματα πλησιάζουν την περιοχή «ΓΕΙΑ» και λαμβάνουν confidence0,99917–0,999995. Η πλήρης baseline απορρίπτει5/10, όλα από το άτομο7, αλλά δέχεται5/5 από το άτομο3. Αυτό δείχνει ανεπαρκή αξιοποίηση/οριοθέτηση των διαθέσιμων χαρακτηριστικών και μεγάλη εξάρτηση από τον άνθρωπο· δεν απομονώνει αιτιωδώς ένα μόνο πρόβλημα αρχιτεκτονικής.', '',
        'Το παγωμένο B εφαρμόζει temperature=2, confidence≥0,90, quality≥0,25 και απόρριψη όταν κερδίζει το OTHER. Το motion gate είχε επιλεγεί ως ανενεργό. Όλα τα10 λάθη περνούν κατά πολύ τα αριθμητικά όρια· η υψηλή softmax πιθανότητα δεν αποτελεί εκτίμηση ότι το δείγμα ανήκει οπωσδήποτε στο γνωστό λεξιλόγιο.', '',
        '![Embeddings και αποστάσεις πραγματικών ακολουθιών](embedding-and-prototype-distances.png)', '',
        'Δεν εισέρχονται στον classifier pixels, ονόματα αρχείων, ετικέτες, ρούχα ή background. Ωστόσο landmarks μπορούν να μεταφέρουν στάση/ιδιαιτερότητες ανθρώπου και σφάλματα ανιχνευτή. Το «ΚΑΤΑΛΑΒΑΙΝΩ» και το «ΓΕΙΑ» προέρχονται από διαφορετικά σενάρια, οπότε τα διαθέσιμα δεδομένα δεν αποκλείουν έμμεσες συσχετίσεις. Η baseline δεν αποδεικνύει γενική κατανόηση της ΕΝΓ.', '',
        'Τεκμήρια: [annotations/όρια](annotation_boundary_audit.json), [οπτικές παρατηρήσεις](visual_observations.json), [αριθμητικός έλεγχος](numerical-diagnosis.json), [αρχικά καρέ ΚΑΤΑΛΑΒΑΙΝΩ](contact-sheets/katalavaino_signer3.png), [συγκρίσιμα ΓΕΙΑ](contact-sheets/geia_farewell_matched_signer3.png).', '',
        '## 2. Χρονικά ablations χωρίς επανεκπαίδευση', '',
        'Στα19 εστιασμένα παραδείγματα, η κανονική/αντίστροφη/ανακατεμένη/στατική είσοδος διατηρεί όλες τις αποφάσεις: B10/10 ψευδείς αποδοχές «ΚΑΤΑΛΑΒΑΙΝΩ», baseline5/10, C5/10. Όλα δέχονται9/9 «ΓΕΙΑ».', '',
        'Σε ολόκληρο το ήδη εκτεθειμένο σύνολο129 παραδειγμάτων:', '',
        '| Είσοδος | B σωστή top-1 γνωστών | B αποδεκτά γνωστά (σωστά) | B ψευδείς αποδοχές /50 | Baseline αποδεκτά γνωστά /79 |',
        '|---|---:|---:|---:|---:|',
    ]
    for variant, methods in temporal['results'].items():
        b=methods['CNN_B']; c=methods['full_feature_centroid']
        correct_top = sum(b['closed_set_confusion_matrix'][i][i] for i in range(3))
        accepted_correct = round(b['correct_known_acceptance_rate']*79)
        lines.append(f"| {variant} | {correct_top}/79 | {b['accepted_known_n']} ({accepted_correct}) | {b['unknown_false_acceptance_n']} | {c['accepted_known_n']} |")
    lines += ['', 'Το C διατηρεί τις αποφάσεις του και στα129. Το B επηρεάζεται στις αποδοχές, αλλά πολύ λιγότερο στην αναγκαστική top-1 ταξινόμηση. Η baseline επίσης είναι ευαίσθητη στη χρονική μεταβολή. **Δεν συμπεραίνουμε ότι κανένα μοντέλο δεν χρησιμοποιεί καθόλου κίνηση.** Αυτές είναι ελεγχόμενες αλλοιώσεις εισόδου και όχι αιτιώδης απόδειξη σημασίας χαρακτηριστικών. Σε όλες τις παραλλαγές, το B δέχεται λανθασμένα10/10 ΚΑΤΑΛΑΒΑΙΝΩ, ενώ baseline/C δέχονται5/10.', '',
        'Η αναπαράσταση κρατά χρονική σειρά, XY μορφή δαχτύλων, χωριστά χέρια και σχετική θέση χεριών/μύτης του pose. Δεν κρατά z, πραγματική διάρκεια ή ταχύτητα σε δευτερόλεπτα. Διπλασιασμός timestamps αφήνει τα features ακριβώς ίδια. Η κατεύθυνση ανά κανονικοποιημένο βήμα μπορεί ήδη να παραχθεί από τις48 θέσεις. Δεν αφαιρούμε face-local συντεταγμένες από body-local χέρια: για κοινή γεωμετρία χρησιμοποιούμε τη μύτη του pose.', '',
        'Τα6 διαθέσιμα raw fixtures περνούν ακριβώς τους ελέγχους μετάθεσης/κλίμακας, ανάκλασης συντεταγμένων, αντιστροφής και αλλαγής καναλιών. Τα raw fixtures αφορούν το άτομο3· δεν καλύπτουν ανεξάρτητα την εντόπιση του ατόμου7. [Όλες οι μετρήσεις και2322 προβλέψεις](all-exposed-temporal-ablations.json).', '',
        '## 3. Το μικρό στοχευμένο πείραμα', '',
        'Αποκτήθηκαν75 πραγματικά, προηγουμένως αχρησιμοποίητα αποσπάσματα:25 ΚΑΤΑΛΑΒΑΙΝΩ,25 ΝΑΙ,25 ΗΜΕΡΑ_ΤΡΕΙΣ. Μόνο1.364 απαραίτητα RGB καρέ, χωρίς depth ή πλήρη archives. Για κάθε ετικέτα χρησιμοποιήθηκαν20train από άτομα1/2/4/5 και5validation από άτομο6. Τα75 αποσπάσματα και όλα τα καρέ τους δεν ταυτίζονται με τα547 παλιά. Οι κατηγορίες όμως επιλέχθηκαν αφού είδαμε τη V2: πρόκειται για **προσαρμοστική ανάπτυξη**.', '',
        '| Ρόλος | Άτομα | Γνωστά | Αρνητικά | Σύνολο |', '|---|---|---:|---:|---:|',
        '| Training | 1,2,4,5 | 178 | 160 | 338 |', '| Validation/thresholds | 6 | 41 | 45 | 86 |',
        '| Επαναληπτικός έλεγχος V2 | 3,7 | 79 | 50 | 129 |', '',
        'Χρησιμοποιήθηκε το ίδιο πραγματικό MediaPipe Web pipeline. Τα18 νέα raw fixtures έδωσαν μέγιστη απόκλιση Python/TypeScript **0**. Τα νέα δεδομένα διατηρούνται χωριστά και τα παλιά artifacts δεν αλλάζουν.', '',
        'Κρατήσαμε ακριβώς τα ίδια normalization statistics και τα τρία γνωστά centroids της baseline. Συγκρίναμε: (α) μόνο νέα βαθμονόμηση της παλιάς baseline, (β) προσθήκη των60 νέων αρνητικών στο ενιαίο OTHER centroid, (γ) ένα centroid ανά μία από τις7 αρνητικές κατηγορίες. Στη(γ), το πλησιέστερο αρνητικό δίνει ένα ενιαίο OTHER logit, ώστε να μην ευνοείται τεχνητά η απόρριψη από περισσότερους όρους softmax. Πρόκειται για εκμάθηση προτύπων από δεδομένα, όχι για νέο πολυκατηγορικό CNN.', '',
        'Πρότυπα μόνο από train· θερμοκρασία και όρια μόνο από validation· κλείδωμα επιλογής πριν από ξεχωριστή εντολή επαναληπτικής αξιολόγησης. Η ίδια προκαθορισμένη πολιτική V2 απαιτεί συνολική αποδοχή γνωστών≥60% και ακρίβεια αποδεκτών γνωστών≥90%, κατόπιν ελαχιστοποιεί unknown FAR. Δεν αλλάξαμε thresholds μετά το regression.', '',
        'Τα κλειδωμένα νέα όρια είναι temperature=0,5 και quality≥0,25 σε όλες τις μεθόδους. Confidence:0,55 στην επαναβαθμονομημένη baseline,0,65 στην ενισχυμένη OTHER,0,45 στα χωριστά αρνητικά πρότυπα. Όλες απορρίπτουν όταν το πλησιέστερο/pooled αρνητικό κερδίζει και καμία δεν επέλεξε motion threshold. Τα confidence scores επομένως δεν είναι ο μοναδικός έλεγχος και δεν είναι συγκρίσιμα με ποσοστά πραγματικής αξιοπιστίας.', '',
        '**Validation — χρησιμοποιήθηκε για επιλογή, δεν είναι τελική αξιολόγηση:**', '', metric_table(validation), '',
        'Η ενισχυμένη ενιαία OTHER κέρδισε τη συγκεκριμένη πολιτική validation. Όμως και οι τρεις ρυθμισμένες μέθοδοι δέχονται **0/11 ΓΕΙΑ** του ατόμου6. Τα συνολικά όρια επιλεξιμότητας επέτρεψαν να χαθεί μία ολόκληρη κλάση. Αυτό αποτελεί αποτυχία λειτουργίας, ακόμη κι αν το συνολικό FAR είναι0%. Σε επόμενο πρωτόκολλο χρειάζονται ελάχιστα όρια ανά υποστηριζόμενο νόημα· δεν προσθέσαμε εκ των υστέρων διαφορετικό κανόνα για να διαλέξουμε άλλο νικητή.', '',
        '**Ίδια129 παλιά δείγματα — αποκλειστικά regression, όχι νέο ανεξάρτητο test:**', '', metric_table(comparison), '',
        'Το closed-set Macro F1=1,000 των centroids μετρά τη σωστή επιλογή ανάμεσα στα τρία γνωστά ακόμη κι όταν απορρίπτονται. Γι’ αυτό δίνουμε επιπλέον F1 όπου η απόρριψη γνωστού είναι false negative και η αποδοχή αγνώστου false positive. Το0/50 είναι περιγραφικό αποτέλεσμα αυτού του προσαρμοστικού ελέγχου και δεν δίνει εκτίμηση μηδενικού σφάλματος στον πραγματικό κόσμο.', '',
        '| Μέθοδος | Άτομο3: σωστή αποδοχή /40 | Άτομο3: ψευδής αποδοχή /25 | Άτομο7: σωστή αποδοχή /39 | Άτομο7: ψευδής αποδοχή /25 |',
        '|---|---:|---:|---:|---:|',
    ]
    for name,item in comparison.items():
        people=item['metrics']['per_signer']; a,b=people['3'],people['7']
        lines.append(f"| {NAMES[name]} | {round(a['correct_known_acceptance_rate']*40)} | {a['unknown_false_acceptance_n']} | {round(b['correct_known_acceptance_rate']*39)} | {b['unknown_false_acceptance_n']} |")
    lines += ['', '| Μέθοδος | ΚΑΤΑΛΑΒΑΙΝΩ ψευδείς /10 | ΝΑΙ ψευδείς /30 | ΗΜΕΡΑ_ΤΡΕΙΣ ψευδείς /10 |', '|---|---:|---:|---:|']
    for name,item in comparison.items():
        c=item['metrics']['per_unknown_category']
        lines.append(f"| {NAMES[name]} | {c['ΚΑΤΑΛΑΒΑΙΝΩ']['false_acceptance_n']} | {c['ΝΑΙ']['false_acceptance_n']} | {c['ΗΜΕΡΑ_ΤΡΕΙΣ']['false_acceptance_n']} |")
    lines += ['', 'Η ενισχυμένη OTHER απορρίπτει8 γνωστά:4 ΓΕΙΑ του ατόμου7 και4 ΕΝΤΑΞΕΙ του ίδιου ατόμου. Τα χωριστά αρνητικά πρότυπα απορρίπτουν30 γνωστά, από τα οποία28 ΕΝΤΑΞΕΙ. Δεν αποδείχθηκαν καλύτερα από την ενιαία OTHER. Η ομοιότητα ΟΚ/ΕΝΤΑΞΕΙ των αρχικών αρνητικών παραμένει γλωσσική εκκρεμότητα, όχι αφορμή για αυθαίρετη αλλαγή labels.', '',
        'Οι πλήρεις μετρήσεις, confusion matrices, αποτυχίες, in-sample training ανά άτομο και validation ανά αρνητική κατηγορία βρίσκονται στα [development αποτελέσματα](centroid-development-results.json), [regression αποτελέσματα](centroid-exposed-regression.json) και [συγκεντρωτικό JSON](technical-report.json). Τα in-sample training αποτελέσματα δεν παρουσιάζονται ως γενίκευση.', '',
        '## 4. Γιατί δεν υπάρχει νέο πλήρες ανεξάρτητο test', '',
        'Στην επαληθευμένη δεξαμενή ολόκληρων utterances με μία μόνο γλωσσική ετικέτα υπάρχουν1.838 επιλέξιμα αποσπάσματα. Έχουν ήδη εκτεθεί και τα98 ΓΕΙΑ. Τα άτομα3/7 δεν έχουν επιπλέον ΕΥΧΑΡΙΣΤΩ ή ΚΑΤΑΛΑΒΑΙΝΩ. Τα μόνα εντελώς ανεκδήλωτα labels αυτής της δεξαμενής είναι ΣΥΝΕΧΕΙΑ και ΣΦΡΑΓΙΔΑ με ένα δείγμα το καθένα. Δεν είναι σοβαρό νέο benchmark.', '',
        'Δεν τεμαχίσαμε αυθαίρετα πολυνοηματικές προτάσεις ούτε ονομάσαμε το regression «φρέσκο test». Η αξιολόγηση σε πλήρως νέους χρήστες/συνθήκες παραμένει ανολοκλήρωτη. Χρειάζονται νέες καταγραφές όλων των γνωστών και αρνητικών κινήσεων, με άτομα/συνεδρίες που δεν θα μπουν σε fitting ή thresholds. [Απογραφή δεδομένων](unused_data_inventory.json).', '',
        '## 5. Browser και πακέτο δημοσίευσης', '',
        'Η υπάρχουσα εφαρμογή έχει πραγματικό upload και webcam με προθέρμανση detector, αντίστροφη μέτρηση3s, αυτόματη λήξη καταγραφής και κοινή προεπεξεργασία. Η απόρριψη ζητά χειροκίνητη επανάληψη. Δεν υπάρχει αυτόματη ομιλία. Δεν αλλάξαμε την κύρια εφαρμογή ή το δημοσιεύσιμο μοντέλο B βάσει του ήδη γνωστού test.', '',
        'Το `signbridge-publication-review.zip` περιέχει τοπικά ελεγμένο static build, μοντέλο/ρυθμίσεις, αναφορές χωρίς απόλυτα προσωπικά paths, attribution/άδειες και headers. Δεν περιέχει raw videos, φωτογραφίες dataset ή dev capture endpoints. Πραγματικό MP4 πέρασε την τοπική δοκιμή CSP/WASM. Οι launchers ελέγχουν τα ακριβή hashes HTML/config/μοντέλου και απορρίπτουν παλιό ή development server χωρίς να τον τερματίσουν.', '',
        '**Δεν υπάρχει ακόμη δημόσιο URL. Δεν έγινε ανθρώπινη webcam δοκιμή.** Το πρόγραμμα κάμερας και η ανάλυση MP4 δεν υποκαθιστούν αυτή τη δοκιμή. Ο χρήστης δήλωσε ότι δεν μπορεί να την ολοκληρώσει τώρα. Το πακέτο είναι προετοιμασία δημοσίευσης ερευνητικού πρωτοτύπου, με εμφανή τα παλιά24/50 λάθη του B.', '',
        'Τα νέα centroid μοντέλα ελέγχονται σε χωριστά ONNX artifacts. Τα τελικά browser timings, parity και όρια τους βρίσκονται στο [browser-comparison.json](browser-comparison.json). Χρόνος features→απόφαση (μαζί με serialization, worker, diagnostics και gate) και συνολικός χρόνος decode→MediaPipe→απόφαση αναφέρονται χωριστά. Τα timings τοπικού Chrome δεν είναι μέτρηση δημόσιου δικτύου/κινητού.', '',
        *browser_lines, '',
        'Ανεξάρτητη επανυλοποίηση επαλήθευσε1.874 αποθηκευμένες προβλέψεις και όλες τις μετρικές/επιλογές/πρότυπα, χωρίς εισαγωγή των fitting/scoring modules. Χωριστός έλεγχος επαλήθευσε ότι τα ONNX initializers είναι ακριβώς τα παγωμένα πρότυπα και ότι η μόνη διαφορά του browser config είναι η παράλειψη ανύπαρκτου embedding output. [Audit μετρήσεων](centroid-independent-audit.json), [audit export](centroid-export-source-audit.json).', '',
        '## 6. Τεχνική απόφαση', '',
        'Υπάρχει μετρήσιμη βελτίωση στην ήδη εκτεθειμένη επαναληπτική δοκιμή: η ενισχυμένη baseline μειώνει τα ψευδώς αποδεκτά άγνωστα12→0, με σωστή αποδοχή γνωστών77→71. Υπάρχει όμως σαφής αποτυχία validation στο ΓΕΙΑ και δεν υπάρχει νέο πλήρες τελικό σύνολο. **Δεν τεκμηριώνεται ακόμη βελτίωση γενίκευσης.**', '',
        'Κρατάμε τη baseline ως κύρια ερευνητική κατεύθυνση και την ενισχυμένη OTHER ως υποψήφια για νέα ανεξάρτητη δοκιμή. Τα χωριστά αρνητικά πρότυπα δεν κερδίζουν αυτό το πείραμα. Δεν δικαιολογείται μεγαλύτερο CNN πριν λυθεί η διατήρηση των γνωστών ανά άνθρωπο. Το B παραμένει μόνο το παγωμένο μοντέλο αναφοράς της έτοιμης εφαρμογής, όχι σύσταση αξιοπιστίας για δημόσια επικοινωνία.', '',
        'Επόμενη στοχευμένη αλλαγή προς έλεγχο: επάρκεια tracking του ενεργού χεριού, αντί μόνο τουλάχιστον ενός χεριού, και αναπαράσταση/απόσταση που δίνει ρητή βαρύτητα στη μορφή του και συγκρίνει σωστά χέρι–πρόσωπο. Η πραγματική διάρκεια/τοπικό z μπορούν να εξεταστούν χωριστά. Τα δεδομένα δείχνουν ήδη υπάρχον διακριτικό XY σχήμα, επομένως η αλλαγή δεν πρέπει να δικαιολογηθεί ως «λείπει όλη η πληροφορία». Δεν εγκαταστήσαμε αυθαίρετο threshold ενεργού χεριού πάνω στο παρατηρημένο test· οι νέες επιλογές χρειάζονται κλείδωμα πριν από νέα αξιολόγηση.', '',
        'Η υλοποίηση και το πακέτο web είναι συγκεκριμένα παραδοτέα. Η ανεξάρτητη τελική αξιολόγηση, η ανθρώπινη webcam και η δημόσια δημοσίευση παραμένουν σαφώς εκκρεμή.', '',
        '## 7. Αναπαραγωγή και πηγές', '',
        'Οδηγίες εντολών: [V3_DIAGNOSIS.md](../../V3_DIAGNOSIS.md). Locks και ακριβή hashes διατηρούνται στο `artifacts/v3-centroid`, οι raw μετρήσεις στο `reports/v3`. Το ανεξάρτητο audit δεν αλλάζει τα μοντέλα ή τα thresholds.', '',
        'Η επίσημη περιγραφή αφορά επτά νοηματιστές και διαφορετικά continuous/isolated splits. Η δική μας επιλογή ολόκληρων utterances με ένα gloss δεν είναι το επίσημο isolated benchmark: [GSL / ΕΚΕΤΑ](https://vcl.iti.gr/dataset/gsl/), [αρχικά αρχεία Zenodo](https://zenodo.org/records/4756317). Για τα χωριστά landmark systems: [MediaPipe Holistic](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker?tab=Web). Οι αριθμοί της αναφοράς προέρχονται από τα δικά μας αποθηκευμένα πειράματα, όχι από αυτές τις εξωτερικές πηγές.',
    ]
    prose='\n'.join(lines)+'\n'
    greek='Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋΐΰ'
    prose=re.sub(f'([{greek}])([0-9])', r'\1 \2', prose)
    prose=re.sub(f'([0-9])([{greek}])', r'\1 \2', prose)
    prose=prose.replace('**Ολοκληρωμένοι έλεγχοι:**344', '**Ολοκληρωμένοι έλεγχοι:** 344')
    prose=prose.replace('ανεκδήλωτα labels', 'κατηγορίες που δεν είχαν ακόμη χρησιμοποιηθεί')
    (OUT/'TECHNICAL_REPORT.md').write_text(prose,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'TECHNICAL_REPORT.md'),'browser_report_present':browser is not None,'independent_audit_present':audit is not None},ensure_ascii=False))


if __name__ == '__main__': build()
