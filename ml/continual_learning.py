"""Local continuous recognition retraining and paired checkpoint validation."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import torch

from continual_review import approve, load_approved
from continuous_selective import PERTURBATIONS, guard_rows, summarize_guard
from train_continuous import reference_lookup, replay, summarize
from train_continuous_contrastive import real_data
from train_continuous_hard import datasets, digest, read_selected
from train_continuous_hard_isolated import CHECKPOINT, FREEZE
from train_continuous_multitask import OrderAwareCTC, SEED, train_candidate


ROOT = Path(__file__).resolve().parents[1]
HARD = Path('data/continuous-hard-split')
CONTEXT = Path('data/continuous-context')
DEFAULT_STATE = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share')) / 'SignBridge' / 'continual'
POLICY = {'minimum_hand_motion_rms': .008, 'minimum_forward_order_gain': .025, 'require_plain_ctc_agreement': False}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def open_registry(state):
    state.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(state / 'registry.sqlite3', timeout=2)
    db.execute('CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, report TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.commit()
    return db


def setting(db, name, default=None):
    row = db.execute('SELECT value FROM settings WHERE name = ?', (name,)).fetchone()
    return json.loads(row[0]) if row else default


def set_setting(db, name, value):
    db.execute('INSERT OR REPLACE INTO settings VALUES (?, ?)', (name, json.dumps(value, ensure_ascii=False)))


def base_model():
    return {'path': str(CHECKPOINT.resolve()), 'sha256': read_json(FREEZE)['checkpoint_sha256']}


def load_model(ref):
    path = Path(ref['path'])
    if digest(path) != ref['sha256']:
        raise ValueError(f'Model digest mismatch: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    weights, labels = checkpoint['state_dict'], checkpoint['labels']
    model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    model.load_state_dict(weights)
    return model.eval(), labels


def corpus():
    _, training, hard_dev, provenance = datasets(HARD / 'manifest.json', HARD / 'features.jsonl')
    frozen = read_json(FREEZE)
    if provenance != frozen['provenance'] or [row.id for row in hard_dev] != frozen['development_ids']:
        raise ValueError('The initial training or development split has changed')
    prior, archival, _ = real_data()
    hard_ids = {row.id for row in hard_dev}
    familiar = [row for row in prior['val'] if row.id not in hard_ids]
    protected = hard_dev + familiar + [row for rows in archival.values() for row in rows]
    for folder in (HARD, CONTEXT):
        held, _ = read_selected(folder / 'manifest.json', folder / 'features.jsonl', ('test',))
        protected.extend(held['test'])
    return training, hard_dev, familiar, protected, provenance


def sign_metrics(model, labels, rows):
    result = summarize(replay(model, rows, labels, {}))
    keys = ('utterances', 'reference_gloss_count', 'substitutions', 'deletions', 'insertions', 'gloss_wer', 'gloss_sequence_exact')
    counts = {key: result[key] for key in keys}
    counts['edits'] = sum(counts[key] for key in ('substitutions', 'deletions', 'insertions'))
    return counts


def measure(model, labels, hard, familiar, lookup):
    clips = hard + familiar
    guarded = guard_rows(model, None, clips, labels, lookup, .60)
    controls = {}
    for mode in PERTURBATIONS:
        rows = guard_rows(model, None, clips, labels, lookup, .60, mode, False)
        controls[mode] = summarize_guard(rows, POLICY, False)['wrong_or_out_of_scope_text']
    result = summarize_guard(guarded, POLICY, True)
    return {'hard': sign_metrics(model, labels, hard), 'familiar': sign_metrics(model, labels, familiar),
            'correct_texts': result['correct_text'], 'wrong_texts': result['wrong_or_out_of_scope_text'],
            'false_texts_on_controls': controls, 'controls_per_type': len(clips)}


def rejection_reasons(current, trial):
    hard, old_hard = trial['hard'], current['hard']
    familiar, old_familiar = trial['familiar'], current['familiar']
    checks = {
        'no net improvement on hard phrases': hard['edits'] >= old_hard['edits'],
        'more missing or inserted signs': hard['deletions'] > old_hard['deletions'] or hard['insertions'] > old_hard['insertions'],
        'fewer exact hard phrases': hard['gloss_sequence_exact'] < old_hard['gloss_sequence_exact'],
        'regression on familiar phrases': familiar['edits'] > old_familiar['edits'] or familiar['gloss_sequence_exact'] < old_familiar['gloss_sequence_exact'],
        'correct Greek phrase coverage below floor': trial['correct_texts'] < max(1, current['correct_texts']),
        'incorrect Greek text on validation or controls': trial['wrong_texts'] > 0 or any(trial['false_texts_on_controls'].values()),
    }
    return [reason for reason, failed in checks.items() if failed]


def save_candidate(state, key, model, labels):
    output = state / 'models'
    output.mkdir(exist_ok=True)
    target = output / f'{key}.pt'
    with tempfile.NamedTemporaryFile(dir=output, suffix='.pt', delete=False) as stream:
        staging = Path(stream.name)
    try:
        torch.save({'state_dict': model.state_dict(), 'labels': labels}, staging)
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    return {'path': str(target), 'sha256': digest(target)}


def cycle(state, submitted=None):
    with closing(open_registry(state)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        champion = setting(db, 'champion') or base_model()
        model, labels = load_model(champion)
        base, hard, familiar, protected, provenance = corpus()
        reviewed, approved = load_approved(state, base, protected, labels)
        recipe = {name: digest(Path(__file__).with_name(name)) for name in ('continual_learning.py', 'continual_review.py',
                                                                           'continuous_selective.py', 'train_continuous_multitask.py')}
        inputs = {'provenance': provenance, 'approved': approved, 'recipe': recipe,
                  'submission': submitted['sha256'] if submitted else None}
        key = sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        if db.execute('SELECT 1 FROM runs WHERE key = ?', (key,)).fetchone():
            result = {'status': 'already_evaluated' if submitted else 'waiting_for_reviewed_examples',
                      'checked_at': now(), 'reviewed_clips': len(reviewed)}
            set_setting(db, 'last_check', result)
            return result
        lookup = reference_lookup(base)
        current = measure(model, labels, hard, familiar, lookup)
        if submitted:
            trial, trial_labels = load_model(submitted)
            if trial_labels != labels:
                raise ValueError('Candidate and incumbent gloss vocabularies differ')
            training = {'mode': 'frozen_experiment'}
        else:
            torch.manual_seed(SEED)
            vocabulary = {gloss: index for index, gloss in enumerate(labels) if index}
            trial, _, epoch = train_candidate({'train': base + reviewed, 'val': hard}, labels, vocabulary, lookup,
                                              initial_model=model, epochs=5, learning_rate=.0003)
            training = {'mode': 'continue_training', 'train_clips': len(base) + len(reviewed), 'selected_epoch': epoch}
        artifact = save_candidate(state, key, trial, labels)
        candidate = measure(trial, labels, hard, familiar, lookup)
        reasons = rejection_reasons(current, candidate)
        result = {'status': 'kept_incumbent' if reasons else 'promoted_local_candidate', 'checked_at': now(),
                  'key': key, 'training': training, 'reviewed_clips': len(reviewed), 'approved_batches': approved,
                  'validation_scope': 'Repeated development regression. Held test features are only used for duplicate checks.',
                  'incumbent': champion, 'candidate': artifact, 'incumbent_metrics': current,
                  'candidate_metrics': candidate, 'rejection_reasons': reasons}
        db.execute('INSERT INTO runs VALUES (?, ?)', (key, json.dumps(result, ensure_ascii=False)))
        set_setting(db, 'champion', champion if reasons else artifact)
        set_setting(db, 'last_check', {'status': result['status'], 'checked_at': result['checked_at'], 'key': key})
        return result


def status(state):
    with closing(open_registry(state)) as db:
        row = db.execute('SELECT report FROM runs ORDER BY rowid DESC LIMIT 1').fetchone()
        return {'state_dir': str(state), 'champion': setting(db, 'champion', base_model()),
                'last_check': setting(db, 'last_check'), 'last_run': json.loads(row[0]) if row else None}


def run_check(state, submitted=None):
    try:
        return cycle(state, submitted)
    except Exception as error:
        with closing(open_registry(state)) as db, db:
            set_setting(db, 'last_check', {'status': 'error', 'checked_at': now(), 'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest='command', required=True)
    once = commands.add_parser('once')
    once.add_argument('--candidate', type=Path)
    once.add_argument('--candidate-sha256')
    watch = commands.add_parser('watch')
    watch.add_argument('--interval-seconds', type=int, default=3600)
    commands.add_parser('status')
    review = commands.add_parser('approve')
    review.add_argument('--manifest', type=Path, required=True)
    review.add_argument('--features', type=Path, required=True)
    review.add_argument('--reviewer', required=True)
    args = parser.parse_args()
    state = args.state_dir.resolve()
    if args.command == 'approve':
        args.manifest, args.features = args.manifest.resolve(), args.features.resolve()
    if args.command == 'once' and bool(args.candidate) != bool(args.candidate_sha256):
        parser.error('--candidate and --candidate-sha256 must be used together')
    if args.command == 'watch' and args.interval_seconds < 1:
        parser.error('--interval-seconds must be positive')
    submitted = None
    if args.command == 'once' and args.candidate:
        submitted = {'path': str(args.candidate.resolve()), 'sha256': args.candidate_sha256}
    os.chdir(ROOT)
    torch.set_num_threads(2)
    if args.command == 'status':
        result = status(state)
    elif args.command == 'watch':
        while True:
            started = time.monotonic()
            try:
                print(json.dumps(run_check(state), ensure_ascii=False), flush=True)
            except Exception as error:
                print(json.dumps({'status': 'error', 'error': str(error)}, ensure_ascii=False), flush=True)
            time.sleep(max(1, args.interval_seconds - (time.monotonic() - started)))
    elif args.command == 'approve':
        _, labels = load_model(base_model())
        base, _, _, protected, _ = corpus()
        with closing(open_registry(state)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            result = approve(state, args.manifest, args.features, args.reviewer, base, protected, labels)
    else:
        result = run_check(state, submitted)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
