"""Bounded local-only reranker; fit on signer 4, replay exposed signer 3 once.

No CNN fitting, pretrained OpenHands output, production writes, or fresh-test claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_isolated_exemplar_retrieval import (
    annotations, open_source, read_bank, descriptors, unit, align_scores,
    probabilities, mix, select_gate, summarize, sha256, write_json,
)

CACHE = Path('research/cache/openhands-gsl')
SOURCE = Path('reports/isolated-exemplar-retrieval-expanded')
CLASSIFIER = Path('reports/isolated-large-vocabulary')
OUTPUT = Path('reports/isolated-balanced-prototypes')
WEIGHTS = (0., .5, 1., 2., 4.)
METHODS = ('maximum_signer_center', 'top_two_signer_centers')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identities():
    paths = [Path(__file__), Path('ml/evaluate_isolated_exemplar_retrieval.py'),
             Path('ml/evaluate_synthetic_ctc_transfer.py'),
             SOURCE / 'freeze.json', SOURCE / 'fusion-expanded-freeze.json',
             CLASSIFIER / 'expanded-freeze.json', CLASSIFIER / 'expanded-dev-scores.npz',
             Path('artifacts/isolated-large-vocabulary/expanded.pt'),
             CACHE / 'gsl_metadata.zip', CACHE / 'official-train-prototypes-poses.npz']
    return {p.as_posix(): sha256(p) for p in paths}


def load_sources():
    old = read_json(SOURCE / 'fusion-expanded-freeze.json')
    source = read_json(SOURCE / 'freeze.json')
    model = read_json(CLASSIFIER / 'expanded-freeze.json')
    if (sha256(SOURCE / 'freeze.json') != old['exemplar_freeze_sha256']
            or sha256(CLASSIFIER / 'expanded-freeze.json') != old['classifier_freeze_sha256']
            or sha256(CLASSIFIER / 'expanded-dev-scores.npz') != old['classifier_dev_scores_sha256']
            or sha256(Path('artifacts/isolated-large-vocabulary/expanded.pt')) != model['checkpoint_sha256']
            or model['train_signers'] != [1, 2, 5]
            or model['development_signers'] != [4] or model['test_signers'] != [3]):
        raise ValueError('Frozen local classifier/exemplar provenance mismatch')
    bank = read_bank(source['variants']['all_verified_labels'])
    if set(bank['labels']) != set(model['labels']):
        raise ValueError('Local classifier and bank labels differ')
    return old, source, bank


def prototypes(bank):
    """Each source signer gets at most one center per class, regardless of count."""
    labels, vectors, center_labels, center_signers = bank['labels'], [], [], []
    for label_index, label in enumerate(labels):
        for signer in (1, 2, 5):
            mask = (bank['glosses'] == label) & (bank['signers'] == signer)
            if mask.any():
                center = unit(bank['vectors'][mask].mean(axis=0, keepdims=True))[0]
                if not np.isfinite(center).all() or np.linalg.norm(center) < .99:
                    raise ValueError('Invalid source-only prototype')
                vectors.append(center)
                center_labels.append(label_index)
                center_signers.append(signer)
    groups = np.asarray(center_labels)
    if set(groups) != set(range(len(labels))):
        raise ValueError('A vocabulary class has no training prototype')
    return np.asarray(vectors, dtype=np.float32), groups, np.asarray(center_signers)


def reduce_centers(similarity, groups, label_count):
    outputs = {method: np.empty((len(similarity), label_count), dtype=np.float32) for method in METHODS}
    for label in range(label_count):
        values = np.sort(similarity[:, groups == label], axis=1)
        if values.shape[1] == 0:
            raise ValueError('Missing class prototypes')
        outputs[METHODS[0]][:, label] = values[:, -1]
        outputs[METHODS[1]][:, label] = values[:, -min(2, values.shape[1]):].mean(axis=1)
    return outputs


def score_prototypes(bank, sequences):
    centers, groups, signers = prototypes(bank)
    values = descriptors(sequences, bank['location'], bank['scale'], bank['motion_scale'])
    matrix = (torch.from_numpy(values) @ torch.from_numpy(centers).T).numpy()
    return reduce_centers(matrix, groups, len(bank['labels'])), {
        'centers': len(centers), 'labels': len(bank['labels']),
        'centers_per_label': dict(Counter(Counter(groups.tolist()).values())),
        'signers': sorted(set(signers.tolist())),
    }


def summary(matrix, labels, references, policy):
    result, ranks, accepted = summarize(matrix, labels, references, policy)
    result['top1'] = result['top1_correct'] / result['known_clips']
    result['top5'] = result['top5_correct'] / result['known_clips']
    result['correct_acceptance'] = result['accepted_correct'] / result['known_clips']
    accepted_count = result['accepted_correct'] + result['accepted_wrong_known'] + result['accepted_outside']
    result['accepted_precision'] = result['accepted_correct'] / accepted_count if accepted_count else None
    result['coverage'] = accepted_count / result['clips']
    return result, ranks, accepted


def fit():
    target = OUTPUT / 'freeze.json'
    if target.exists():
        raise ValueError('This bounded experiment has already been frozen')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    old, source, bank = load_sources()
    official = annotations(CACHE / 'gsl_metadata.zip')
    path = CACHE / 'official-train-prototypes-poses.npz'
    if sha256(path) != source['source_train_pose_sha256']:
        raise ValueError('Development pose source changed')
    seq, ids, refs, signers = open_source(path, official['train'], role='train')
    chosen = signers == 4
    seq, ids, refs = seq[chosen], ids[chosen], refs[chosen]
    if set(ids) & set(bank['ids']):
        raise ValueError('Development identity entered prototype bank')
    logits = align_scores(CLASSIFIER / 'expanded-dev-scores.npz', ids, refs, bank['labels'])
    matrices, counts = score_prototypes(bank, seq)
    grid = []
    for method in METHODS:
        for weight in WEIGHTS:
            values = mix(logits, matrices[method], weight)
            correct = bank['labels'][values.argmax(1)] == refs
            macro = np.mean([correct[refs == label].mean() for label in np.unique(refs)])
            grid.append({'method': method, 'weight': weight, 'top1_correct': int(correct.sum()), 'macro_top1': float(macro)})
    # Stable, prespecified tie-break: fewer mixture weight, then method order.
    selected = max(grid, key=lambda row: (row['top1_correct'], row['macro_top1'], -row['weight'], -METHODS.index(row['method'])))
    probs = probabilities(mix(logits, matrices[selected['method']], selected['weight']))
    policy = select_gate(probs, bank['labels'], refs)
    measured = summary(probs, bank['labels'], refs, policy)[0]
    result = {
        'scope': 'Adaptive development experiment; source-only prototypes and frozen local CNN. No upstream pretrained output.',
        'protocol': {'fit_signers': [1, 2, 5], 'calibration_signers': [4], 'retrospective_signers': [3],
                     'methods': METHODS, 'weights': WEIGHTS, 'selection': 'Development raw top1, macro recall, lower weight, method order',
                     'gate': 'Maximize correct development accepts under <=1% wrong-known over all known; unknown support absent',
                     'prototype': 'Unit mean descriptor within each (gloss, training signer); equal maximum one center per signer',
                     'fresh_test': False},
        'sources': identities(), 'bank_sha256': source['variants']['all_verified_labels']['bank_sha256'],
        'prototype_counts': counts, 'development_grid': grid, 'selected': selected,
        'policy': policy, 'development': measured,
        'existing_fusion_development': old['development']['weighted'],
        'new_raw_top1_improves_existing_on_development': measured['top1_correct'] > old['development']['weighted']['top1_correct'],
        'development_ids': ids.tolist(), 'elapsed_seconds': time.perf_counter() - started,
        'limits': ['Signer4 has already been used for many adaptive choices.',
                   'Signer3 results were previously exposed and will be a repeated diagnostic, never a fresh benchmark.',
                   'This all-label comparison has no unseen-sign or everyday-gesture negatives.',
                   'Native pre-cut pose clips do not validate browser tracking or continuous signing.',
                   'The per-signer prototype does not infer or normalize a new user identity.'],
    }
    write_json(target, result)
    print(json.dumps({'phase': 'frozen', 'selected': selected, 'development': measured}, ensure_ascii=False), flush=True)


def evaluate():
    started = time.perf_counter()
    path = OUTPUT / 'benchmark.json'
    if path.exists():
        raise ValueError('The retrospective diagnostic has already been evaluated')
    frozen = read_json(OUTPUT / 'freeze.json')
    if identities() != frozen['sources']:
        raise ValueError('Code/source changed after calibration freeze')
    old, source, bank = load_sources()
    if source['variants']['all_verified_labels']['bank_sha256'] != frozen['bank_sha256']:
        raise ValueError('Prototype bank changed')
    prior = read_json(SOURCE / 'fusion-expanded-benchmark.json')
    if prior['freeze_sha256'] != sha256(SOURCE / 'fusion-expanded-freeze.json'):
        raise ValueError('Prior fusion result does not match its frozen policy')
    held_path = CACHE / 'official-test-poses.npz'
    scores_path = CLASSIFIER / 'expanded-scores.npz'
    if sha256(held_path) != prior['held_pose_sha256'] or sha256(scores_path) != prior['held_classifier_scores_sha256']:
        raise ValueError('Prior and current diagnostic inputs differ')
    official = annotations(CACHE / 'gsl_metadata.zip')
    seq, ids, refs, signers = open_source(held_path, official['test'], role='test')
    if set(ids) & (set(bank['ids']) | set(frozen['development_ids'])) or set(signers) != {3}:
        raise ValueError('Physical source or signer overlap')
    logits = align_scores(scores_path, ids, refs, bank['labels'])
    matrices, _ = score_prototypes(bank, seq)
    choice = frozen['selected']
    probs = probabilities(mix(logits, matrices[choice['method']], choice['weight']))
    measured, ranks, accepted = summary(probs, bank['labels'], refs, frozen['policy'])
    previous = {row['id']: row for row in prior['rows']}
    if len(previous) != len(ids) or set(previous) != set(ids) or any(previous[i]['reference_gloss'] != r for i, r in zip(ids, refs, strict=True)):
        raise ValueError('Historical comparison row identity mismatch')
    rows, changes = [], Counter()
    for i, identity in enumerate(ids):
        guess, reference = str(bank['labels'][ranks[i, 0]]), str(refs[i])
        earlier = previous[str(identity)]
        old_correct = earlier['weighted_gloss'] == reference
        changes['both_correct' if old_correct and guess == reference else 'new_only_correct' if guess == reference else 'old_only_correct' if old_correct else 'both_wrong'] += 1
        rows.append({'id': str(identity), 'reference_gloss': reference, 'prediction': guess, 'accepted': bool(accepted[i]),
                     'existing_fusion_prediction': earlier['weighted_gloss'], 'existing_fusion_accepted': earlier['weighted_accepted']})
    # Reproduce the historical summary from its exact saved per-clip outcomes.
    old_correct = sum(row['existing_fusion_prediction'] == row['reference_gloss'] for row in rows)
    old_accepted_correct = sum(row['existing_fusion_accepted'] and row['existing_fusion_prediction'] == row['reference_gloss'] for row in rows)
    old_wrong = sum(row['existing_fusion_accepted'] and row['existing_fusion_prediction'] != row['reference_gloss'] for row in rows)
    if (old_correct, old_accepted_correct, old_wrong) != tuple(prior['held']['weighted'][key] for key in ('top1_correct', 'accepted_correct', 'accepted_wrong_known')):
        raise ValueError('Historical summary disagrees with saved outcomes')
    write_json(path, {'scope': 'Repeated already-exposed signer3 diagnostic, not independent final evidence',
        'freeze_sha256': sha256(OUTPUT / 'freeze.json'), 'prior_fusion_report_sha256': sha256(SOURCE / 'fusion-expanded-benchmark.json'),
        'held_pose_sha256': sha256(held_path), 'held_score_sha256': sha256(scores_path),
        'selected': choice, 'new_candidate': measured, 'existing_classifier': prior['held']['classifier'],
        'existing_weighted_fusion': prior['held']['weighted'], 'paired_correctness_vs_existing_fusion': dict(changes),
        'elapsed_seconds': time.perf_counter() - started, 'rows': rows, 'limits': frozen['limits']})
    print(json.dumps({'phase': 'evaluated', 'new_candidate': measured, 'paired': dict(changes)}, ensure_ascii=False), flush=True)


def self_test():
    bank = {'labels': np.array(['A', 'B']), 'glosses': np.array(['A', 'A', 'A', 'B']),
            'signers': np.array([1, 1, 2, 5]), 'vectors': np.array([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)}
    centers, groups, signers = prototypes(bank)
    assert centers.shape == (3, 2) and groups.tolist() == [0, 0, 1] and signers.tolist() == [1, 2, 5]
    result = reduce_centers(np.array([[.9, .3, .2]], dtype=np.float32), groups, 2)
    np.testing.assert_allclose(result[METHODS[0]], [[.9, .2]])
    np.testing.assert_allclose(result[METHODS[1]], [[.6, .2]])
    duplicated = {key: value.copy() for key, value in bank.items()}
    for key in ('glosses', 'signers', 'vectors'):
        duplicated[key] = np.concatenate((bank[key], bank[key][:1]), axis=0)
    np.testing.assert_array_equal(prototypes(duplicated)[0], centers)
    try:
        reduce_centers(np.ones((1, 1)), np.array([0]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing class did not fail closed')
    print('Four bounded prototype invariants passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate', 'self-test'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {'fit': fit, 'evaluate': evaluate, 'self-test': self_test}[args.phase]()
