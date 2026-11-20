"""Development-fitted vetoes for full text emitted by a continuous CTC model."""

from itertools import product
import math

import torch
from torch.nn import functional as F

from train_continuous import altered_frames, ctc_decode, edit_counts, replay


MOTION_FLOORS = (.004, .008, .016, .032)
ORDER_FLOORS = (0., .01, .025, .05, .10, .20, .40, 1.0, 10.0)
PERTURBATIONS = ('reversed', 'shuffled', 'static_midpoint')


def tracked_hand_motion(frames):
    present = frames[:, 220:223] >= .75
    pose = present[:-1, 2] & present[1:, 2]
    squared, pairs = 0., 0
    for hand, start in ((0, 0), (1, 84)):
        valid = pose & present[:-1, hand] & present[1:, hand]
        motion = torch.diff(frames[:, start:start + 42], dim=0)
        squared += float(motion[valid].square().sum())
        pairs += int(valid.sum())
    return math.sqrt(squared / (pairs * 21)) if pairs else 0.


def sequence_probability(scores, target):
    return -float(F.ctc_loss(scores[:, None, :], target, [len(scores)], [len(target)], reduction='sum'))


@torch.inference_mode()
def guard_rows(model, secondary, items, labels, lookup, threshold, mode='original', genuine=True):
    rows = replay(model, items, labels, lookup, threshold, mode)
    token_ids = {label: index for index, label in enumerate(labels) if index}
    if secondary is not None:
        secondary.eval()
    for row, item in zip(rows, items, strict=True):
        row['eligible_text'] = row['predicted_text'] is not None
        row['legitimate_text'] = genuine and item.glosses in lookup and row['greek_reference_exact']
        row['tracked_hand_motion_rms'] = None
        row['forward_order_gain'] = None
        row['same_order_plain_ctc'] = False
        if row['eligible_text']:
            frames = altered_frames(item, mode)
            length = [len(frames)]
            target = torch.tensor([token_ids[label] for label in row['predicted_glosses']], dtype=torch.long)
            original_scores = model(frames[None], length)[0]
            reversed_scores = model(frames.flip(0)[None], length)[0]
            row['tracked_hand_motion_rms'] = tracked_hand_motion(frames)
            row['forward_order_gain'] = (sequence_probability(original_scores, target)
                                         - sequence_probability(reversed_scores, target)) / len(frames)
            if secondary is not None:
                alternate = ctc_decode(secondary(frames[None], length)[0], labels)[0]
                row['same_order_plain_ctc'] = alternate == row['predicted_glosses']
    return rows


def accepts(row, policy):
    return (row['eligible_text'] and row['tracked_hand_motion_rms'] >= policy['minimum_hand_motion_rms']
            and row['forward_order_gain'] > policy['minimum_forward_order_gain']
            and (not policy['require_plain_ctc_agreement'] or row['same_order_plain_ctc']))


def summarize_guard(rows, policy, genuine):
    accepted = [row for row in rows if accepts(row, policy)]
    correct = sum(row['legitimate_text'] for row in accepted)
    raw_edits = sum(row['substitutions'] + row['deletions'] + row['insertions'] for row in rows)
    nref = sum(len(row['reference_glosses']) for row in rows)
    selected_edits = sum(sum(edit_counts(row['reference_glosses'], row['predicted_glosses'] if accepts(row, policy) else ()))
                         for row in rows)
    return {'clips': len(rows), 'raw_gloss_wer': raw_edits / nref if genuine else None,
            'selective_wer_counting_abstentions_as_deletions': selected_edits / nref if genuine else None,
            'text_accepted': len(accepted), 'text_coverage': len(accepted) / len(rows), 'correct_text': correct,
            'wrong_or_out_of_scope_text': len(accepted) - correct,
            'accepted_ids': [row['id'] for row in accepted]}


def choose_guard(development, require_agreement):
    trials = []
    for motion, order in product(MOTION_FLOORS, ORDER_FLOORS):
        policy = {'minimum_hand_motion_rms': motion, 'minimum_forward_order_gain': order,
                  'require_plain_ctc_agreement': require_agreement}
        accepted = [row for row in development if accepts(row, policy)]
        correct = sum(row['legitimate_text'] for row in accepted)
        trials.append(((len(accepted) - correct, -correct, motion, order), policy))
    return min(trials, key=lambda trial: trial[0])


def evaluate_selective(model, secondary, validation, test_cohorts, labels, lookup, threshold):
    development = {'development': guard_rows(model, secondary, validation, labels, lookup, threshold)}
    for mode in PERTURBATIONS:
        development[f'dev_{mode}'] = guard_rows(model, secondary, validation, labels, lookup, threshold, mode, False)
    rows = [row for cohort in development.values() for row in cohort]
    policies = {name: choose_guard(rows, agreement) for name, agreement in (('motion_and_order', False),
                                                                           ('plus_plain_ctc_agreement', True))}
    selected = min(policies, key=lambda name: (*policies[name][0][:2], policies[name][1]['require_plain_ctc_agreement']))
    # Test inputs are scored only after these frozen development choices.
    tests = {}
    for name, items in test_cohorts.items():
        mode = name if name in PERTURBATIONS else 'original'
        tests[name] = guard_rows(model, secondary, items, labels, lookup, threshold, mode, name.startswith('known_'))
    return {'selection': 'Dev-only minimum false emissions, then maximum correct text, then smaller floors. '
                         'Select agreement variant on development only, preferring the simpler rule on ties.',
            'feature_definition': 'Tracked-hand RMS uses valid consecutive hand/pose points at native continuous feature cadence. '
                                  'Order is log P of the same complete predicted token sequence on original minus reversed frames, per frame.',
            'threshold_grid': {'hand_motion_rms': MOTION_FLOORS, 'forward_order_gain': ORDER_FLOORS},
            'selected_on_development': selected,
            'variants': {name: {'policy': policy,
                               'development': {key: summarize_guard(cohort, policy, key == 'development')
                                               for key, cohort in development.items()},
                               'test': {key: summarize_guard(cohort, policy, key not in PERTURBATIONS)
                                        for key, cohort in tests.items()}}
                         for name, (_, policy) in policies.items()},
            'development_rows': development, 'test_rows': tests}
