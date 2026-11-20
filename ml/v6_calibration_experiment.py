"""Compare single-person and two-person rejection calibration on identical CNNs."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import torch

from common import fit_temperature, softmax
from train import fit_normalization
from v5_grouped_experiment import (
    CALIBRATION, CHANNELS, DATA, FLOORS, LABELS, NOVEL, PEOPLE, ROOT, SCORE_GRID,
    SEED, counts, digest, outputs, read_data, sample_weights, train_tcn, write_json,
)


REPORT = ROOT / 'reports/v6'
ARTIFACTS = ROOT / 'artifacts/v6-calibration'
LOCK = ARTIFACTS / 'protocol.locked.json'
PARTNER = {'1': '4', '2': '5', '4': '6', '5': '6', '6': '2'}
METHODS = ('single_a', 'single_b', 'pooled_global', 'worst_person_global', 'worst_person_per_sign')


def protocol():
    sources = [Path(__file__).resolve(), DATA / 'train.npz', DATA / 'val.npz']
    sources.extend(ROOT / 'ml' / name for name in (
        'common.py', 'train.py', 'preprocess.py', 'openset_train.py', 'v5_grouped_experiment.py',
    ))
    return {
        'id': 'signbridge-v6-matched-calibration-v1',
        'files': {str(path.relative_to(ROOT)): digest(path) for path in sources},
        'people': PEOPLE, 'labels': LABELS, 'methods': METHODS,
        'primary_calibration_pairs': {person: [CALIBRATION[person], PARTNER[person]] for person in PEOPLE},
        'assessment_only_unknown_labels': NOVEL,
        'assessment': 'One disjoint person; fit on two and calibrate on the remaining two. Run all six calibration pairs per assessment person.',
        'primary': 'Predeclared balanced pairs. Exactly one assessment decision per clip per method.',
        'sensitivity': 'All six pairs per assessment person; split outcomes share assessment clips and must not be pooled as independent examples.',
        'model': 'Same dropout-trained CNN and logits for all gates within a split. V5 training schedule and weighting.',
        'seed': 'V5 SEED + 6 * assessment index + pair index in sorted combinations.',
        'thresholds': SCORE_GRID, 'targets': FLOORS,
        'temperature': 'One scalar fit on the selected calibration data only, shared by all two-person variants of a given CNN.',
        'gate': 'Top known probability strictly exceeds OTHER probability and meets its selected threshold.',
        'pool_objective': 'Eligible first, then minimum total target violation, minimum false acceptance, maximum macro and minimum sign recall.',
        'robust_objective': 'Every calibration person eligible first, then minimum worst and mean target violation, minimum worst false acceptance, maximum worst sign recall and pooled macro recall.',
        'fallback': 'Keep the ranked diagnostic decision rule and record ineligible when no thresholds pass the requirements.',
        'release': 'Development comparison using previously exposed people. Old 129, raw tracking and human webcam are outside this classifier experiment.',
    }


def check_protocol(mode):
    current = json.loads(json.dumps(protocol()))
    if mode == 'lock' and not LOCK.exists():
        write_json(LOCK, current)
    if not LOCK.is_file() or json.loads(LOCK.read_text(encoding='utf-8')) != current:
        raise RuntimeError('Missing or changed V6 protocol lock')


def eligible(metrics):
    return (
        metrics['minimum_sign_recall'] >= FLOORS['per_sign_correct_acceptance']
        and metrics['accepted_precision'] >= FLOORS['accepted_precision']
        and metrics['unknown_false_acceptance'] <= FLOORS['unknown_false_acceptance_max']
    )


def violation(metrics):
    sign_shortfall = sum(
        max(0., FLOORS['per_sign_correct_acceptance'] - sign['correct_accepted'] / sign['n'])
        for sign in metrics['per_sign'].values()
    )
    return (sign_shortfall + max(0., FLOORS['accepted_precision'] - metrics['accepted_precision'])
            + max(0., metrics['unknown_false_acceptance'] - FLOORS['unknown_false_acceptance_max']))


def evidence(logits, temperature):
    probability = softmax(logits, temperature)
    predicted = probability[:, :3].argmax(axis=1)
    confidence = probability[np.arange(len(predicted)), predicted]
    return predicted, confidence, probability[:, 3]


def select_gate(logits, labels, people, robust, per_sign):
    temperature = fit_temperature(logits, np.where(labels < 0, 3, labels))
    predicted, confidence, other = evidence(logits, temperature)
    groups = {person: people == person for person in np.unique(people)}
    threshold_grid = product(SCORE_GRID, repeat=3) if per_sign else ((value,) * 3 for value in SCORE_GRID)
    best = None
    for thresholds in threshold_grid:
        accepted = (confidence > other) & (confidence >= np.asarray(thresholds)[predicted])
        pooled = counts(labels, predicted, accepted)
        by_person = {person: counts(labels[mask], predicted[mask], accepted[mask]) for person, mask in groups.items()}
        passed_all = all(eligible(value) for value in by_person.values())
        if robust:
            penalties = [violation(value) for value in by_person.values()]
            rank = (passed_all, -max(penalties), -float(np.mean(penalties)),
                    -max(value['unknown_false_acceptance'] for value in by_person.values()),
                    min(value['minimum_sign_recall'] for value in by_person.values()), pooled['macro_sign_recall'])
        else:
            rank = (eligible(pooled), -violation(pooled), -pooled['unknown_false_acceptance'],
                    pooled['macro_sign_recall'], pooled['minimum_sign_recall'])
        if best is None or rank > best[0]:
            best = (rank, {'temperature': temperature, 'thresholds': thresholds, 'pooled': pooled,
                           'per_person': by_person, 'pooled_eligible': eligible(pooled),
                           'every_calibration_person_eligible': passed_all})
    return best[1]


def train_split(person, pair, seed, x, y, people, raw_label):
    assessment = people == person
    novel = (y < 0) & np.isin(raw_label, NOVEL[person])
    calibration = np.isin(people, pair) & ~novel
    fitting = ~np.isin(people, (*pair, person)) & ~novel
    if set(people[fitting]) & set(people[calibration | assessment]):
        raise ValueError('Signer overlap')
    if len(set(people[fitting])) != 2 or len(set(people[calibration])) != 2 or np.any(novel & (fitting | calibration)):
        raise ValueError('Unexpected fit/calibration partition')
    for mask in [fitting, assessment, *(calibration & (people == p) for p in pair)]:
        if set(y[mask]) != {-1, 0, 1, 2}:
            raise ValueError('Partition lacks an expected class')
    normalization = fit_normalization(x[fitting], 'signbridge-landmarks-xy-v1')[:3]
    weights = sample_weights(y[fitting], people[fitting], raw_label[fitting])
    model, seconds = train_tcn(x[fitting], y[fitting], weights, normalization, True, seed)
    return model, seconds, fitting, calibration, assessment, novel


def run():
    x, y, people, raw_label, source = read_data()
    rows = {method: [] for method in METHODS}
    splits = []
    for person_index, person in enumerate(PEOPLE):
        remaining = [p for p in PEOPLE if p != person]
        primary_pair = (CALIBRATION[person], PARTNER[person])
        print(f'Assessment person {person}: six matched fit/calibration splits', flush=True)
        for pair_index, sorted_pair in enumerate(combinations(remaining, 2)):
            is_primary = set(sorted_pair) == set(primary_pair)
            pair = primary_pair if is_primary else sorted_pair
            seed = SEED + 6 * person_index + pair_index
            model, seconds, fit, calibration, assessment, novel = train_split(person, pair, seed, x, y, people, raw_label)
            logits, _ = outputs(model, x)
            configurations = {
                'single_a': (calibration & (people == pair[0]), False, False),
                'single_b': (calibration & (people == pair[1]), False, False),
                'pooled_global': (calibration, False, False),
                'worst_person_global': (calibration, True, False),
                'worst_person_per_sign': (calibration, True, True),
            }
            split = {'assessment_person': person, 'calibration_people': pair, 'fit_people': sorted(set(people[fit])),
                     'seed': seed, 'fit_n': int(fit.sum()), 'calibration_n': int(calibration.sum()),
                     'assessment_n': int(assessment.sum()), 'novel_labels': NOVEL[person], 'primary': is_primary,
                     'fit_label_counts': dict(Counter(raw_label[fit])),
                     'calibration_label_counts': dict(Counter(raw_label[calibration])),
                     'train_seconds': seconds, 'methods': {}}
            for method, (cal, robust, per_sign) in configurations.items():
                gate = select_gate(logits[cal], y[cal], people[cal], robust, per_sign)
                predicted, confidence, other = evidence(logits[assessment], gate['temperature'])
                accepted = (confidence > other) & (confidence >= np.asarray(gate['thresholds'])[predicted])
                metrics = counts(y[assessment], predicted, accepted)
                metrics['novel_n'] = int(novel[assessment].sum())
                metrics['novel_false_accepted'] = int((novel[assessment] & accepted).sum())
                split['methods'][method] = {'gate': gate, 'assessment': metrics, 'assessment_eligible': eligible(metrics)}
                if is_primary:
                    for i, row in enumerate(np.flatnonzero(assessment)):
                        rows[method].append({'source': str(source[row]), 'person': person, 'label': str(raw_label[row]),
                                             'y': int(y[row]), 'predicted': int(predicted[i]), 'accepted': bool(accepted[i]),
                                             'score': float(confidence[i]), 'other_score': float(other[i]),
                                             'novel_unknown': bool(novel[row])})
            if is_primary:
                checkpoint = ARTIFACTS / f'cnn-fold-{person}.pt'
                torch.save({'state_dict': model.state_dict(), 'channels': CHANNELS, 'labels': LABELS,
                            'fit_people': split['fit_people'], 'assessment_person': person}, checkpoint)
                split['checkpoint'] = {'name': checkpoint.name, 'sha256': digest(checkpoint)}
            splits.append(split)
    primary_splits = [split for split in splits if split['primary']]
    aggregate = {}
    for name, predictions in rows.items():
        if len(predictions) != len(y) or len({row['source'] for row in predictions}) != len(y):
            raise ValueError('Primary predictions must cover every clip exactly once')
        summary = counts(*(np.asarray([row[key] for row in predictions]) for key in ('y', 'predicted', 'accepted')))
        summary['novel_n'] = sum(row['novel_unknown'] for row in predictions)
        summary['novel_false_accepted'] = sum(row['novel_unknown'] and row['accepted'] for row in predictions)
        for title, subset in (('primary', primary_splits), ('all_splits', splits)):
            assessments = [split['methods'][name] for split in subset]
            summary[title] = {
                'n': len(assessments),
                'calibration_pass': sum(item['gate']['every_calibration_person_eligible'] for item in assessments),
                'assessment_pass': sum(item['assessment_eligible'] for item in assessments),
                'both_pass': sum(item['gate']['every_calibration_person_eligible'] and item['assessment_eligible'] for item in assessments),
                'mean_sign_recall': float(np.mean([item['assessment']['macro_sign_recall'] for item in assessments])),
                'mean_unknown_false_acceptance': float(np.mean([item['assessment']['unknown_false_acceptance'] for item in assessments])),
            }
        aggregate[name] = summary
    write_json(REPORT / 'calibration-experiment.json', {
        'protocol_sha256': digest(LOCK), 'scope': 'Previously exposed development people only',
        'old_129_read': False, 'raw_guard_applied': False, 'release_approved': False,
        'aggregate': aggregate, 'splits': splits, 'primary_predictions': rows,
    })
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'run'))
    args = parser.parse_args()
    check_protocol(args.mode)
    if args.mode == 'run':
        run()
    else:
        print(f'Locked V6 calibration protocol: {digest(LOCK)}')
