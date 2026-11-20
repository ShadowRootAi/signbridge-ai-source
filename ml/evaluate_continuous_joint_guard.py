"""Replay a development-only joint rejection gate for an existing continuous checkpoint."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import torch

from continuous_selective import PERTURBATIONS, guard_rows, summarize_guard
from evaluate_continuous_transfer import load_inputs
from train_continuous import read_utterances, reference_lookup, train_model
from train_continuous_multitask import BATCH_SIZE, EPOCHS, SEED, OrderAwareCTC, digest


ROOT = Path('data/continuous-fresh-compact')
ARCHIVED = Path('reports/continuous-fresh-compact/multitask-benchmark.json')
REPORT = Path('reports/continuous-fresh-compact/joint-guard-diagnostic.json')
CONFIDENCE = (0., .60, .75, .85, .90, .95, .98, .99, .995, .998, 1.000001)
ORDER = (0., .025, .05, .10, .20, .40, 1., 10.)
DURATION = (None, 2., 1.5, 1.3, 1.15, 1.)


def tag_candidates(rows, lengths):
    for row in rows:
        limit = lengths.get(row['predicted_glosses'])
        row['train_max_frame_ratio'] = row['frames'] / limit if limit else None


def as_base_guard(rows, policy):
    return [{**row, 'eligible_text': row['eligible_text'] and row['confidence'] >= policy['minimum_confidence']
             and (policy['max_train_frame_ratio'] is None or row['train_max_frame_ratio'] <= policy['max_train_frame_ratio'])}
            for row in rows]


def fit(rows, agreement, duration):
    best = None
    for confidence, order, ratio in product(CONFIDENCE, ORDER, DURATION if duration else (None,)):
        policy = {'minimum_confidence': confidence, 'minimum_hand_motion_rms': .004,
                  'minimum_forward_order_gain': order, 'require_plain_ctc_agreement': agreement,
                  'max_train_frame_ratio': ratio}
        accepted = [row for row in rows if row['eligible_text'] and row['confidence'] >= confidence
                    and row['tracked_hand_motion_rms'] >= .004 and row['forward_order_gain'] > order
                    and (ratio is None or row['train_max_frame_ratio'] <= ratio)
                    and (not agreement or row['same_order_plain_ctc'])]
        correct = sum(row['legitimate_text'] for row in accepted)
        key = (len(accepted) - correct, -correct, ratio is not None, confidence, order, -(ratio or 0.))
        if best is None or key < best[0]:
            best = key, policy
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    archived = json.loads(ARCHIVED.read_text(encoding='utf-8'))
    for name, expected in archived['sources'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Archived source changed: {name}')
    checkpoint = Path(archived['candidate_artifact']['file'])
    if digest(checkpoint) != archived['candidate_artifact']['sha256']:
        raise ValueError('Candidate artifact differs from archived benchmark')
    frozen = torch.load(checkpoint, map_location='cpu', weights_only=True)
    labels, lookup, state = (frozen[key] for key in ('labels', 'training_greek_references', 'state_dict'))
    vocabulary = {label: i for i, label in enumerate(labels) if i}
    candidate = OrderAwareCTC(len(labels), state['mean'], state['scale'])
    candidate.load_state_dict(state)
    expanded, composition = (load_inputs(name) for name in ('expanded', 'composition'))
    fresh, _ = read_utterances(ROOT / 'manifest.json', [ROOT / 'features.jsonl'], False)
    roles = {role: expanded[role] + composition[role] + fresh[role] for role in ('train', 'val')}
    if lookup != reference_lookup(roles['train']):
        raise ValueError('The replay training references differ from the frozen checkpoint')
    torch.manual_seed(SEED)
    plain, _, epoch = train_model(roles, labels, vocabulary, lookup, EPOCHS, BATCH_SIZE, SEED)
    if epoch != archived['outcomes']['plain_ctc']['selected_epoch']:
        raise ValueError('Plain CTC validation selection did not reproduce')
    lengths = defaultdict(int)
    for row in roles['train']:
        lengths[row.glosses] = max(lengths[row.glosses], len(row.frames))
    dev = {'development': guard_rows(candidate, plain, roles['val'], labels, lookup, 0.)}
    for mode in PERTURBATIONS:
        dev[f'dev_{mode}'] = guard_rows(candidate, plain, roles['val'], labels, lookup, 0., mode, False)
    dev_rows = [row for rows in dev.values() for row in rows]
    tag_candidates(dev_rows, lengths)
    configurations = {'motion_order': (False, False), 'plus_agreement': (True, False),
                      'with_duration': (False, True), 'duration_and_agreement': (True, True)}
    selected = {name: fit(dev_rows, *options) for name, options in configurations.items()}
    winner = min(selected, key=lambda name: (*selected[name][0][:2], sum(configurations[name])))

    # Replaying the now-observed held cohort is explicitly diagnostic.
    known = [item for item in fresh['test'] if item.glosses in lengths]
    novel = [item for item in fresh['test'] if item.glosses not in lengths]
    controls = load_inputs('old_unknowns', ('test',))['test']
    tests = {'known_fresh': guard_rows(candidate, plain, known, labels, lookup, 0.),
             'unseen_composition': guard_rows(candidate, plain, novel, labels, lookup, 0., genuine=False),
             'genuine_out_of_scope': guard_rows(candidate, plain, controls, labels, lookup, 0., genuine=False)}
    for mode in PERTURBATIONS:
        tests[mode] = guard_rows(candidate, plain, fresh['test'], labels, lookup, 0., mode, False)
    tag_candidates([row for rows in tests.values() for row in rows], lengths)
    results = {}
    for name, (_, policy) in selected.items():
        results[name] = {'policy': policy,
                         'development': {group: summarize_guard(as_base_guard(rows, policy), policy, group == 'development')
                                         for group, rows in dev.items()},
                         'test_diagnostic': {group: summarize_guard(as_base_guard(rows, policy), policy, group not in PERTURBATIONS)
                                             for group, rows in tests.items()}}
    report = {'scope': 'Posthoc replay of a fixed checkpoint. Only development examples and transformed development '
                       'controls choose the threshold. The test cohorts are already observed and remain diagnostic.',
              'selection': 'Jointly minimize false development emissions, then maximize correct development outputs. '
                           'A trained complete gloss sequence is still required. Duration uses only the longest '
                           'recorded training example of that exact template. Simpler configuration wins ties.',
              'archived_report_sha256': digest(ARCHIVED), 'candidate_artifact_sha256': digest(checkpoint),
              'threshold_grid': {'confidence': CONFIDENCE, 'order': ORDER, 'duration': DURATION},
              'selected_on_development': winner, 'variants': results, 'development_rows': dev, 'test_rows': tests}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, variant in results.items():
        print(name, variant['policy'], 'dev', variant['development']['development']['correct_text'],
              {group: (metric['correct_text'], metric['wrong_or_out_of_scope_text'])
               for group, metric in variant['test_diagnostic'].items()}, flush=True)


if __name__ == '__main__':
    main()
