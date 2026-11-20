"""Measure visual-only training and isolated-sign rehearsal on the existing development clips."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from train_continuous import replay, summarize
from train_continuous_library import PRETRAIN_ARTIFACT, PRETRAIN_FREEZE, isolated_rows, transfer_state
from train_continuous_multitask import OrderAwareCTC, train_candidate
from train_scaled_phrases import ARTIFACTS, REPORT, SEED, cohort_digest, digest, read_selected, static_videos, write


TRANCHE = Path('data/continuous-scale-tranche1')
RESULT = REPORT / 'visual-development.json'


def rehearsal_bank(source):
    full, _, audit = isolated_rows()
    if audit != source:
        raise ValueError('The isolated training membership differs from the frozen initialization')
    by_gloss = defaultdict(list)
    for row in full:
        by_gloss[row.glosses[0]].append(row)
    rng = random.Random(SEED)
    return [replace(row, frames=row.frames.clone())
            for _, items in sorted(by_gloss.items())
            for row in rng.sample(items, min(len(items), 4))]


def visual_metrics(model, rows, labels, train_glosses):
    predictions = replay(model, rows, labels, {})
    controls = replay(model, static_videos(rows), labels, {})
    # This lexical count ignores order; the sequence scores below do not.
    support = {'continuous': [0, 0], 'isolated_only': [0, 0], 'outside_model': [0, 0]}
    vocabulary = set(labels[1:])
    for row in predictions:
        actual = Counter(row['reference_glosses'])
        hits = actual & Counter(row['predicted_glosses'])
        for gloss, count in actual.items():
            group = 'continuous' if gloss in train_glosses else 'isolated_only' if gloss in vocabulary else 'outside_model'
            support[group][0] += hits[gloss]
            support[group][1] += count
    return {'ordered': summarize(predictions), 'lexical_hits_ignoring_order': support,
            'static_gloss_emissions': sum(bool(row['predicted_glosses']) for row in controls),
            'predictions': predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=26)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('Positive epoch and thread counts are required')
    if RESULT.exists():
        raise ValueError(f'Previous development experiment already exists: {RESULT}')
    torch.set_num_threads(args.threads)
    manifest, features = TRANCHE / 'manifest.json', TRANCHE / 'features.fit.jsonl'
    roles, feature_hash = read_selected(manifest, features, ('train', 'val'))
    train_glosses = {gloss for row in roles['train'] for gloss in row.glosses}
    freeze = json.loads((REPORT / 'baseline-freeze.json').read_bytes())
    baseline_path = ARTIFACTS / 'baseline.pt'
    if (digest(manifest) != freeze['manifest_sha256'] or digest(baseline_path) != freeze['checkpoint_sha256']
            or feature_hash != freeze['train_dev_features_sha256']
            or cohort_digest(roles['val']) != freeze['dev_cohort_sha256']):
        raise ValueError('Development inputs differ from the frozen baseline')
    init = json.loads(PRETRAIN_FREEZE.read_bytes())
    if digest(PRETRAIN_ARTIFACT) != init['checkpoint_sha256']:
        raise ValueError('Isolated initialization differs from its freeze')
    baseline = torch.load(baseline_path, weights_only=True, map_location='cpu')
    pretraining = torch.load(PRETRAIN_ARTIFACT, weights_only=True, map_location='cpu')
    labels = baseline['labels']
    indexes = {gloss: i for i, gloss in enumerate(labels) if i}
    weights = {key.removeprefix('visual.'): value for key, value in baseline['state_dict'].items()
               if key.startswith('visual.')}
    old = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    old.load_state_dict(weights)
    outcomes = {'joint_baseline': visual_metrics(old, roles['val'], labels, train_glosses)}
    print('Baseline development', outcomes['joint_baseline']['ordered'], flush=True)
    bank = rehearsal_bank(init['source'])
    if {row.signer for row in bank} - {'1', '2', '5'}:
        raise ValueError('Isolated rehearsal contains an unexpected signer')
    for name, rehearsal in (('visual_only', ()), ('visual_with_rehearsal', bank)):
        torch.manual_seed(SEED)
        state = pretraining['state_dict']
        model = OrderAwareCTC(len(labels), state['mean'], state['scale'])
        transfer_state(model, labels, pretraining)
        print(name, len(roles['train']), 'phrase clips', len(rehearsal), 'isolated clips', flush=True)
        model, history, selected = train_candidate(roles, labels, indexes, {}, initial_model=model,
                                                   epochs=args.epochs, rehearsal=rehearsal, learning_rate=.0012)
        path = ARTIFACTS / f'{name}.pt'
        pending = path.with_suffix('.tmp')
        torch.save({'state_dict': model.state_dict(), 'labels': labels}, pending)
        pending.replace(path)
        outcomes[name] = {**visual_metrics(model, roles['val'], labels, train_glosses),
                          'checkpoint_sha256': digest(path), 'selected_epoch': selected, 'history': history}
        print(name, outcomes[name]['ordered'], flush=True)
    write(RESULT, {'scope': 'Existing signer-3 development. Scores are gloss recognition with no independent test inference.',
                   'train_dev_features_sha256': feature_hash, 'baseline_checkpoint_sha256': digest(baseline_path),
                   'initialization_sha256': init['checkpoint_sha256'], 'implementation_sha256': digest(Path(__file__)),
                   'rehearsal_clips': len(bank), 'rehearsal_glosses': len({row.glosses[0] for row in bank}),
                   'rehearsal_ids_sha256': sha256('\n'.join(row.id for row in bank).encode()).hexdigest(),
                   'matched_epochs': args.epochs, 'outcomes': outcomes})


if __name__ == '__main__':
    main()
