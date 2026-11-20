"""Freeze an expanded isolated-sign policy and audit additional decisions on native poses."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from evaluate_openhands_verifier import MODEL_SHA256, sha256, write_json


CACHE = Path('research/cache/openhands-gsl')
REPORT = Path('reports/pretrained-gsl')
BASE = REPORT / 'stgcn-gate-agreement-candidate.json'
POLICY = REPORT / 'stgcn-gate-roster-200.json'
AUDIT = REPORT / 'roster-200-native.json'
FLOOR = .9999


def evidence(role, labels):
    pose = CACHE / f'official-{role}-poses.npz'
    predictions = pose.with_name(pose.stem + '-verifier-scores.npz')
    index = {label: i for i, label in enumerate(labels)}
    with np.load(pose, allow_pickle=False) as data, np.load(predictions, allow_pickle=False) as cache:
        ids, glosses = data['ids'], data['glosses']
        logits = cache['logits'].astype(np.float64)
        if cache['sourceHash'].item() != sha256(pose) or cache['modelHash'].item() != MODEL_SHA256:
            raise ValueError(f'Native evidence is not bound to its pose and model: {role}')
    if logits.shape != (len(ids), len(labels)) or len(set(ids)) != len(ids) or not np.isfinite(logits).all():
        raise ValueError(f'Unexpected native score matrix: {role}')
    winners = logits.argmax(1)
    maximum = logits[np.arange(len(ids)), winners]
    confidence = 1 / np.exp(logits - maximum[:, None]).sum(1)
    return ids, np.array([index[label] for label in glosses]), winners, confidence, maximum, {
        'pose_sha256': sha256(pose), 'scores_sha256': sha256(predictions),
    }


def summarize(ids, reference, winner, added, old_labels, new_labels, labels, include):
    correct = added & (reference == winner)
    wrong = added & ~correct
    new_refs = new_labels[reference]
    partitions = {'original_labels': old_labels[reference], 'added_labels': new_refs,
                  'still_outside': ~old_labels[reference] & ~new_refs}
    return {
        'clips': int(include.sum()), 'added_correct': int(correct.sum()), 'added_wrong': int(wrong.sum()),
        'added_reference_labels_correct': len(set(reference[correct])),
        'partitions': {name: {'clips': int((include & mask).sum()),
                              'added_correct': int((mask & correct).sum()),
                              'added_wrong': int((mask & wrong).sum())} for name, mask in partitions.items()},
        'wrong_rows': [{'id': str(ids[i]), 'reference': labels[reference[i]], 'prediction': labels[winner[i]]}
                       for i in np.flatnonzero(wrong)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'evaluate'))
    args = parser.parse_args()
    baseline = json.loads(BASE.read_bytes())
    config = json.loads((CACHE / 'stgcn-onnx.json').read_bytes())
    labels = config['labels']
    if (len(labels) != 310 or len(set(labels)) != 310
            or baseline['runtimeModelSha256'] != MODEL_SHA256 or config['model_sha256'] != MODEL_SHA256
            or sha256(CACHE / 'stgcn.onnx') != MODEL_SHA256):
        raise ValueError('Unexpected OpenHands model')
    old = np.isin(labels, baseline['supportedLabels'])
    ids, ref, win, conf, logits, provenance = evidence('val', labels)
    calibration = np.array(['_signer3_rep1_glosses/' in item for item in ids])
    comparison = np.array(['_signer3_rep2_glosses/' in item for item in ids])
    if not (calibration ^ comparison).all():
        raise ValueError('Expected just two native validation repetitions')
    evidence_floor = baseline['decisionGate']['minimumEvidence']
    eligible = (conf >= FLOOR) & (logits >= evidence_floor) & ~old[win]
    positive = set(win[calibration & eligible & (ref == win)])
    negative = set(win[calibration & eligible & (ref != win)])
    new = np.isin(np.arange(len(labels)), sorted(positive - negative))
    additions = [label for label, keep in zip(labels, new, strict=True) if keep]
    if int(old.sum()) != 120 or len(additions) != 80:
        raise ValueError('The fixed expansion no longer yields 80 additional categories')
    candidate = deepcopy(baseline)
    candidate['supportedLabels'].extend(additions)
    candidate['unknownLabels'] = [label for label in labels if label not in candidate['supportedLabels']]
    candidate['expansionGate'] = {'labels': additions, 'minimumConfidence': FLOOR}
    candidate['decisionRule'] += (' Additional roster entries require primary confidence at least 0.9999 and '
                                  'the same logit floor. Recovery is reserved for the original roster.')
    if args.phase == 'freeze':
        if POLICY.exists() and json.loads(POLICY.read_bytes()) != candidate:
            raise ValueError('A different policy was already frozen')
        if not POLICY.exists():
            write_json(POLICY, candidate)
        print('Frozen', len(candidate['supportedLabels']), 'categories:', ', '.join(additions), flush=True)
        return
    if json.loads(POLICY.read_bytes()) != candidate:
        raise ValueError('The frozen expansion does not match calibration')
    stats = {}
    for name, mask in (('calibration_rep1', calibration), ('observed_rep2', comparison)):
        stats[name] = summarize(ids, ref, win, mask & eligible & new[win], old, new, labels, mask)
    ids, ref, win, conf, logits, test_source = evidence('test', labels)
    added = (conf >= FLOOR) & (logits >= evidence_floor) & new[win]
    stats['upstream_test_diagnostic'] = summarize(ids, ref, win, added, old, new, labels, np.ones(len(ids), dtype=bool))
    write_json(AUDIT, {'scope': 'Native MediaPipe pose replays from the same corpus and signer used upstream. '
                              'These are diagnostics, not independent human or browser-video accuracy.',
                       'selection': 'Rep1 correct observed category, zero wrong accepts for that category, fixed 0.9999 floor.',
                       'base_policy_sha256': sha256(BASE), 'candidate_policy_sha256': sha256(POLICY),
                       'validation_source': provenance, 'upstream_test_source': test_source,
                       'added_categories': additions, 'results': stats})
    for name, result in stats.items():
        print(name, result['added_correct'], 'correct additional decisions,', result['added_wrong'], 'incorrect',
              'across', result['added_reference_labels_correct'], 'categories', flush=True)


if __name__ == '__main__':
    main()
