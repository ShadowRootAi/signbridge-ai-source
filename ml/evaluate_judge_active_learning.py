"""Measure an equal-budget offline review policy on signer-disjoint native poses."""

from collections import Counter
from pathlib import Path
import json
import re
import time

import numpy as np

from train_judge_open_set import (
    ROOT, CACHE, CALIBRATION, HELD, POLICY, TRAIN, authority, digest, load_native,
    prepare_stgcn_pose, resize, write_json,
)


REPORT = ROOT / 'reports/judge-ai-development/active-learning.json'
FREEZE = REPORT.with_name('active-learning-freeze.json')
HELD_PROTOCOL = ROOT / 'reports/judge-final/held-native.json'
BUDGET = 120
RANDOM_SEEDS = (91, 216, 341, 467, 568)
STEPS = 16
FALSE_ACCEPT_LIMIT = .05
WRONG_KNOWN_LIMIT = .01
SIMILARITIES = (0., .2, .35, .45, .52, .58, .63, .68, .74, .8, 1.01)
MARGINS = (0., .02, .04, .06, .09, .12)
OTHER_GAPS = (0., .025, .05, .075, .10)


def unit(values):
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


def describe(poses):
    fixed = np.stack([resize(pose, STEPS) for pose in poses])
    speed = np.diff(fixed, axis=2, prepend=fixed[:, :, :1])
    hand_shape = np.concatenate((fixed[:, :, :, 7:17] - fixed[:, :, :, 7:8],
                                 fixed[:, :, :, 17:27] - fixed[:, :, :, 17:18]), axis=3)
    return np.concatenate((fixed.clip(-8, 8), (2 * speed).clip(-4, 4), hand_shape.clip(-4, 4)), axis=3).reshape(len(poses), -1)


def unlabeled_pool(path):
    # No `glosses` member is read in the pool until the query list is frozen.
    with np.load(path, allow_pickle=False) as archive:
        ids, positions, offsets = (archive[key] for key in ('ids', 'keypoints', 'offsets'))
    if (offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(positions)
            or np.any(np.diff(offsets) <= 0) or not np.isfinite(positions).all()):
        raise ValueError('Invalid unlabeled pose offsets or coordinates')
    selected = np.flatnonzero([bool(re.search(r'_signer5_', identity)) for identity in ids])
    poses = [prepare_stgcn_pose(positions[offsets[i]:offsets[i + 1]]).numpy()[0] for i in selected]
    return ids[selected], poses


def reveal(path, chosen_ids, published):
    wanted = set(chosen_ids)
    with np.load(path, allow_pickle=False) as archive:
        ids, glosses = archive['ids'], archive['glosses']
    revealed = {identity: gloss for identity, gloss in zip(ids, glosses, strict=True) if identity in wanted}
    if set(revealed) != wanted or any(published.get(identity) != gloss for identity, gloss in revealed.items()):
        raise ValueError('Queried IDs must have matching source annotations')
    return np.array([revealed[identity] for identity in chosen_ids])


def prototypes(vectors, labels, signers):
    groups = sorted(set(zip(labels, signers, strict=True)))
    centers = np.stack([vectors[(labels == label) & (signers == signer)].mean(axis=0) for label, signer in groups])
    return unit(centers).astype(np.float32), np.asarray([label for label, _ in groups])


def class_scores(vectors, centers, labels, classes):
    similarities = vectors @ centers.T
    return np.stack([similarities[:, labels == label].max(axis=1) for label in classes], axis=1)


def acquisition(scores, vectors):
    first, second = np.partition(scores, -2, axis=1)[:, -2:].T[::-1]
    # Percentile rank gives a comparable scale even when cosine gaps cluster.
    uncertainty = np.argsort(np.argsort(first - second, kind='stable'), kind='stable')
    utility = 1 - uncertainty / max(1, len(uncertainty) - 1)
    distance = np.ones(len(vectors), dtype=np.float32)
    selected = []
    for _ in range(BUDGET):
        priority = .65 * utility + .35 * distance
        priority[selected] = -np.inf
        index = int(priority.argmax())
        selected.append(index)
        np.minimum(distance, ((1 - vectors @ vectors[index]) / 2).clip(0, 1), out=distance)
    return np.asarray(selected)


def decision(scores, score_labels, active):
    mask = np.isin(score_labels, active)
    known = scores[:, mask]
    candidate = known.argmax(axis=1)
    strongest, runner_up = np.partition(known, -2, axis=1)[:, -2:].T[::-1]
    return (score_labels[mask][candidate], strongest, strongest - runner_up,
            strongest - scores[:, ~mask].max(axis=1))


def summarize(outputs, labels, active, gate):
    predicted, similarity, margin, other_gap = outputs
    known = np.isin(labels, active)
    correct = predicted == labels
    accepted = (similarity >= gate['minimum_similarity']) & (margin >= gate['minimum_margin']) & (other_gap >= gate['minimum_other_gap'])
    return {'known': int(known.sum()), 'outside': int((~known).sum()), 'known_top1_correct': int((known & correct).sum()),
            'known_macro_top1': float(np.mean([correct[labels == label].mean() for label in np.unique(labels[known])])),
            'correct_accepted': int((known & correct & accepted).sum()),
            'wrong_known_accepted': int((known & ~correct & accepted).sum()),
            'outside_accepted': int((~known & accepted).sum()), 'abstained': int((~accepted).sum())}


def calibrate(outputs, labels, active):
    _, similarity, margin, other_gap = outputs
    known = np.isin(labels, active)
    correct = (outputs[0] == labels) & known
    options = []
    for minimum_similarity in SIMILARITIES:
        for minimum_margin in MARGINS:
            for minimum_other_gap in OTHER_GAPS:
                accepted = (similarity >= minimum_similarity) & (margin >= minimum_margin) & (other_gap >= minimum_other_gap)
                wrong, outside = int((accepted & known & ~correct).sum()), int((accepted & ~known).sum())
                if wrong <= WRONG_KNOWN_LIMIT * known.sum() and outside <= FALSE_ACCEPT_LIMIT * (~known).sum():
                    rank = int((accepted & correct).sum()), -outside, -wrong, minimum_similarity, minimum_margin, minimum_other_gap
                    options.append((rank, {'minimum_similarity': minimum_similarity, 'minimum_margin': minimum_margin,
                                           'minimum_other_gap': minimum_other_gap}))
    return max(options, key=lambda pair: pair[0])[1]


def frozen_write(document):
    if FREEZE.exists():
        if json.loads(FREEZE.read_bytes()) != document:
            raise ValueError('The previously frozen queries, gates or source fingerprints differ')
    else:
        write_json(FREEZE, document)


def main():
    started = time.perf_counter()
    source = authority()
    audit = json.loads(HELD_PROTOCOL.read_bytes())
    if audit['npz_sha256'] != digest(HELD) or not audit['complete']:
        raise ValueError('Held acquisition provenance does not match the NPZ')
    active = np.asarray(json.loads(POLICY.read_bytes())['supportedLabels'])
    fit_ids, fit_labels, fit_signers, fit_poses = load_native(TRAIN, (1, 2), source)
    cal_ids, cal_labels, _, cal_poses = load_native(CALIBRATION, (4,), source)
    pool_ids, pool_poses = unlabeled_pool(TRAIN)
    if set(fit_ids) & set(pool_ids) or set(cal_ids) & (set(fit_ids) | set(pool_ids)):
        raise ValueError('A signer-disjoint development ID is overlapping')
    raw = describe(fit_poses)
    location, scale = raw.mean(axis=0), raw.std(axis=0).clip(min=.12)

    def normalize(array):
        return unit(((array - location) / scale).clip(-8, 8)).astype(np.float32)

    fit_x, cal_x, pool_x = normalize(raw), normalize(describe(cal_poses)), normalize(describe(pool_poses))
    centers, tags = prototypes(fit_x, fit_labels, fit_signers)
    classes = np.unique(fit_labels)
    if not set(active) <= set(classes):
        raise ValueError('A 200-roster sign has no existing fit examples')
    query = {'active': acquisition(class_scores(pool_x, centers, tags, classes), pool_x)}
    query.update({f'random-{seed}': np.random.default_rng(seed).choice(len(pool_x), BUDGET, replace=False)
                  for seed in RANDOM_SEEDS})
    frozen = {'scope': 'Native pose experiment. Held actors were upstream OpenHands training signers and have been '
                       'seen in previous local head assessment. This prototype itself consumes no pretrained logits.',
              'baseline_fit_signers': [1, 2], 'unlabeled_pool_signers': [5], 'calibration_signers': [4], 'held_signers': [6, 7],
              'settings': {'labeled_budget': BUDGET, 'pose_steps': STEPS, 'random_seeds': list(RANDOM_SEEDS),
                           'active_query': '65% low top-two class cosine-gap rank, 35% greedily uncovered cosine distance',
                           'prototype': 'One cluster mean per source signer and gloss. Known labels compete with individual OTHER subclasses.',
                           'outside_false_accept_limit': FALSE_ACCEPT_LIMIT, 'wrong_known_limit': WRONG_KNOWN_LIMIT,
                           'similarities': list(SIMILARITIES), 'margins': list(MARGINS), 'other_gaps': list(OTHER_GAPS)},
              'sources': {p.relative_to(ROOT).as_posix(): digest(p) for p in
                          (Path(__file__).resolve(), TRAIN, CALIBRATION, POLICY, HELD_PROTOCOL, CACHE / 'gsl_metadata.zip')},
              'choices_before_label_lookup': {key: pool_ids[x].tolist() for key, x in query.items()}}
    methods = {'baseline': (centers, tags)}
    annotations = {}
    for name, rows in query.items():
        labels = reveal(TRAIN, pool_ids[rows], source)
        methods[name] = prototypes(np.concatenate((fit_x, pool_x[rows])), np.concatenate((fit_labels, labels)),
                                   np.concatenate((fit_signers, np.full(len(rows), 5))))
        annotations[name] = {'queried': len(rows), 'known': int(np.isin(labels, active).sum()),
                             'outside': int((~np.isin(labels, active)).sum()), 'distinct_glosses': len(set(labels)),
                             'gloss_counts': dict(Counter(labels))}
    frozen['gates'] = {}
    calibration = {}
    for name, (vectors, tags) in methods.items():
        labels = np.unique(tags)
        output = decision(class_scores(cal_x, vectors, tags, labels), labels, active)
        frozen['gates'][name] = gate = calibrate(output, cal_labels, active)
        calibration[name] = summarize(output, cal_labels, active, gate)
    frozen_write(frozen)
    print(f'Frozen {len(query)} label queries before held pose inference', flush=True)

    held_ids, held_labels, held_signers, held_poses = load_native(HELD, (6, 7), source)
    if set(held_ids) & (set(fit_ids) | set(cal_ids) | set(pool_ids)):
        raise ValueError('Held pose IDs overlapped fit, pool or calibration')
    held_x = normalize(describe(held_poses))
    held = {}
    for name, (vectors, tags) in methods.items():
        labels = np.unique(tags)
        output = decision(class_scores(held_x, vectors, tags, labels), labels, active)
        held[name] = summarize(output, held_labels, active, frozen['gates'][name])
        held[name]['by_signer'] = {str(signer): summarize(tuple(value[held_signers == signer] for value in output),
            held_labels[held_signers == signer], active, frozen['gates'][name]) for signer in (6, 7)}
    random = {key: float(np.median([held[f'random-{seed}'][key] for seed in RANDOM_SEEDS])) for key in
              ('known_top1_correct', 'correct_accepted', 'wrong_known_accepted', 'outside_accepted')}
    active_result = held['active']
    wins = (active_result['correct_accepted'] > max(random['correct_accepted'], held['baseline']['correct_accepted'])
            and all(active_result[k] <= min(random[k], held['baseline'][k]) for k in ('wrong_known_accepted', 'outside_accepted')))
    write_json(REPORT, {'schema_version': 1, 'status': 'measured', 'scope': frozen['scope'], 'freeze_sha256': digest(FREEZE),
             'query_labels_revealed_from': 'Published source glosses after feature-only selection. No new human annotation.',
             'scope_sizes': {'fit': len(fit_ids), 'pool': len(pool_ids), 'calibration': len(cal_ids), 'held': len(held_ids)},
             'queried_cohorts': annotations, 'calibration': calibration, 'held': held, 'random_held_median': random,
             'active_improved_at_equal_budget_without_new_errors': wins, 'elapsed_seconds': time.perf_counter() - started})
    print(json.dumps({'status': 'measured', 'held': {key: held[key] for key in ('baseline', 'active')},
                       'random_held_median': random, 'active_improved': wins}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
