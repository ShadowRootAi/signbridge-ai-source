"""Compare a template veto with a matched confidence veto on GSL development clips."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cdist

from common import softmax
from openset_core import motion_metrics, score_gate
from v5_grouped_experiment import LABELS, ROOT, digest, write_json
from v7_train_core import DATA, PEOPLE, SEED, development_fold, fit_candidate, logits_for, metrics, predictions, read, read_split


OUT = ROOT / 'reports/v8'
LOCK = OUT / 'verifier-protocol.locked.json'
METHODS = ('base', 'confidence_veto', 'template_veto')
STEPS = 16
BAND = 4
MIN_CORRECT_CALIBRATION = 3


def protocol():
    paths = [Path(__file__).resolve(), DATA / 'extraction-protocol.json', DATA / 'train.npz', DATA / 'val.npz',
             ROOT / 'web/public/models/model.onnx', ROOT / 'web/public/models/model_config.json']
    paths.extend(ROOT / 'ml' / filename for filename in ('v7_train_core.py', 'v7_model.py', 'common.py', 'openset_core.py', 'v5_grouped_experiment.py'))
    return {
        'id': 'signbridge-v8-paired-verifier-v1',
        'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        'grouped_development': 'V7 recovered TCN, re-fit in each of four disjoint person folds using the same seeds. Novel labels excluded from fit and calibration.',
        'frozen_product_diagnostic': 'V4 with V6 fused features and raw guard; template fitting people 2/4/5, calibration person 1, reused diagnostic person 6. V4 historically used this person.',
        'methods': list(METHODS), 'paired_calibration_budget': 'Zero lost correct accepted samples per sign. A veto must remove at least one wrong acceptance to be enabled.',
        'minimum_correct_calibration_per_sign': MIN_CORRECT_CALIBRATION,
        'template': {'steps': STEPS, 'alignment_band': BAND,
                     'features': 'Both wrist positions relative to nose and both palm-normalized hand shapes. Fit-only scaling. Presence masks and a missingness cost.',
                     'classes': 'Known signs plus each real training unknown label separately. Per-class distance averages each fitting person’s nearest usable example.',
                     'risk': 'Distance to predicted class and ratio to nearest alternative class, including distinct training negatives.'},
        'product_status': 'No release decision on these previously exposed people. No cloud requests or old129 access.',
    }


def verify_or_lock(mode):
    current = protocol()
    if mode == 'lock' and not LOCK.exists():
        write_json(LOCK, current)
    if not LOCK.is_file() or read(LOCK) != current:
        raise RuntimeError('Missing or changed V8 verifier protocol')


def components(x):
    sampled = x[:, np.linspace(0, 47, STEPS).round().astype(int)]
    left = sampled[:, :, 220] >= .999
    right = sampled[:, :, 221] >= .999
    pose = sampled[:, :, 222] >= .999
    nose = sampled[:, :, 168:170]
    return (
        (sampled[:, :, :2] - nose, left & pose, .20),
        (sampled[:, :, 84:86] - nose, right & pose, .20),
        (sampled[:, :, 42:84], left, .30),
        (sampled[:, :, 126:168], right, .30),
    )


@dataclass
class Templates:
    features: tuple
    scales: tuple
    labels: np.ndarray
    people: np.ndarray


def fit_templates(x, y, raw_label, people, guard):
    if set(y[guard]) != {-1, 0, 1, 2}:
        raise ValueError('Usable training templates need all known classes and unknowns')
    features = components(x[guard])
    scales = tuple(np.maximum(values[valid].std(axis=0), .08) for values, valid, _ in features)
    if not all(np.isfinite(scale).all() for scale in scales):
        raise ValueError('Invalid template scaling')
    names = np.array([LABELS[int(label)] if label >= 0 else 'unknown:' + raw
                      for label, raw in zip(y[guard], raw_label[guard])])
    return Templates(features, scales, names, people[guard])


def template_distances(x, bank):
    query_n = len(x)
    ref_n = len(bank.labels)
    cost = np.zeros((query_n, STEPS, ref_n, STEPS), dtype=np.float32)
    available = np.zeros_like(cost)
    for (query, query_mask, weight), (reference, ref_mask, _), scale in zip(components(x), bank.features, bank.scales):
        q = (query / scale).reshape(query_n * STEPS, -1)
        r = (reference / scale).reshape(ref_n * STEPS, -1)
        error = cdist(q, r, metric='sqeuclidean').reshape(cost.shape) / q.shape[-1]
        valid = query_mask[:, :, None, None] & ref_mask[None, None, :, :]
        cost += weight * np.minimum(error, 16.) * valid
        available += weight * valid
    cost = cost / np.maximum(available, .01) + 2. * (1. - available)
    # Dynamic alignment works over all query/reference pairs in parallel.
    previous = np.full((STEPS + 1, query_n, ref_n), np.inf, dtype=np.float32)
    previous[0] = 0.
    for i in range(1, STEPS + 1):
        current = np.full_like(previous, np.inf)
        for j in range(max(1, i - BAND), min(STEPS, i + BAND) + 1):
            current[j] = cost[:, i - 1, :, j - 1] + np.minimum(np.minimum(previous[j], current[j - 1]), previous[j - 1])
        previous = current
    return previous[STEPS] / STEPS


def template_risk(x, predicted, bank):
    # Chunking keeps memory bounded for larger template banks.
    distances = np.concatenate([template_distances(chunk, bank) for chunk in np.array_split(x, max(1, (len(x) + 23) // 24))])
    labels = [*LABELS, *sorted(set(bank.labels) - set(LABELS))]
    grouped = []
    for label in labels:
        candidates = bank.labels == label
        by_person = [distances[:, candidates & (bank.people == p)].min(axis=1) for p in sorted(set(bank.people[candidates]))]
        grouped.append(np.stack(by_person).mean(axis=0))
    by_class = np.stack(grouped, axis=1)
    own = by_class[np.arange(len(x)), predicted]
    alternatives = by_class.copy()
    alternatives[np.arange(len(x)), predicted] = np.inf
    closest = alternatives.argmin(axis=1)
    ratio = own / np.maximum(alternatives[np.arange(len(x)), closest], 1e-6)
    return np.column_stack((own, ratio)), np.array(labels)[closest]


def calibrate_veto(y, predicted, accepted, risk):
    gates = []
    for index, label in enumerate(LABELS):
        current = accepted & (predicted == index)
        correct = current & (y == index)
        wrong = current & (y != index)
        cutoff = None
        removed = 0
        if correct.sum() >= MIN_CORRECT_CALIBRATION:
            candidate = risk[correct].max(axis=0)
            removed = int((wrong & np.any(risk > candidate, axis=1)).sum())
            if removed:
                cutoff = candidate.tolist()
        gates.append({'label': label, 'correct_calibration_support': int(correct.sum()), 'removed_calibration_errors': removed, 'cutoff': cutoff})
    return gates


def apply_veto(predicted, accepted, risk, gates):
    final = accepted.copy()
    for index, gate in enumerate(gates):
        cutoff = gate['cutoff']
        if cutoff is not None:
            final &= (predicted != index) | np.all(risk <= np.asarray(cutoff), axis=1)
    return final


def compare(calibration, assessment, bank):
    cal_x, cal_y, _, cal_pred, cal_base, cal_score = calibration
    x, y, guard, predicted, base, score = assessment
    template_cal, _ = template_risk(cal_x, cal_pred, bank)
    template_eval, nearest = template_risk(x, predicted, bank)
    data = {'base': {'assessment': metrics(y, guard, predicted, base)}}
    final_decisions = {'base': base}
    for name, cal_risk, eval_risk in (
        ('confidence_veto', (1. - cal_score)[:, None], (1. - score)[:, None]),
        ('template_veto', template_cal, template_eval),
    ):
        gates = calibrate_veto(cal_y, cal_pred, cal_base, cal_risk)
        adjusted_cal = apply_veto(cal_pred, cal_base, cal_risk, gates)
        if np.any(cal_base & (cal_y == cal_pred) & ~adjusted_cal):
            raise ValueError('Calibration veto violated the common zero-loss budget')
        final = apply_veto(predicted, base, eval_risk, gates)
        final_decisions[name] = final
        data[name] = {'gates': gates, 'assessment': metrics(y, guard, predicted, final),
                      'known_correct_lost': int(((y == predicted) & base & ~final).sum()),
                      'false_unknown_removed': int(((y < 0) & base & ~final).sum())}
    return data, final_decisions, template_eval, nearest


def evaluate_model(model, data, mask, gate):
    x = data['x_fused'][mask]
    guard = data['guard_fused'][mask]
    predicted, accepted, score, _ = predictions(logits_for(model, x), guard, gate['temperature'], gate['threshold'])
    return x, data['y'][mask], guard, predicted, accepted, score


def grouped(train):
    folds = []
    for index, person in enumerate(PEOPLE):
        fit, cal, assessment, _ = development_fold(train, person)
        model, gate, _ = fit_candidate(train, 'recovered_tcn', fit, cal, SEED + index)
        bank = fit_templates(train['x_fused'][fit], train['y'][fit], train['raw_label'][fit], train['signer'][fit], train['guard_fused'][fit])
        result, _, _, _ = compare(evaluate_model(model, train, cal, gate), evaluate_model(model, train, assessment, gate), bank)
        folds.append({'assessment_person': person, 'fit_people': sorted(set(train['signer'][fit])), 'calibration_people': sorted(set(train['signer'][cal])), 'methods': result})
        print(f'Completed grouped person {person}', flush=True)
    fields = ('known_n', 'correct_known_accepted', 'unknown_n', 'unknown_false_accepted', 'wrong_known_accepted')
    aggregate = {method: {key: sum(fold['methods'][method]['assessment'][key] for fold in folds) for key in fields} for method in METHODS}
    return {'aggregate': aggregate, 'folds': folds}


def product_diagnostic(train):
    val = read_split('val')
    config = read(ROOT / 'web/public/models/model_config.json')
    session = ort.InferenceSession(str(ROOT / 'web/public/models/model.onnx'), providers=['CPUExecutionProvider'])

    def run(data, mask):
        x = data['x_fused'][mask]
        guard = data['guard_fused'][mask]
        outputs = [session.run(None, {config['input']['name']: row[None]}) for row in x]
        logits, embedding = (np.concatenate(parts) for parts in zip(*outputs))
        scored = score_gate(logits, data['quality_fused'][mask], motion_metrics(x)['motionBodyRms'], config, embedding)
        return x, data['y'][mask], guard, scored['predicted'], guard & scored['accepted'], scored['confidence']

    fit = np.isin(train['signer'], ('2', '4', '5'))
    cal = train['signer'] == '1'
    bank = fit_templates(train['x_fused'][fit], train['y'][fit], train['raw_label'][fit], train['signer'][fit], train['guard_fused'][fit])
    measured = run(val, np.ones(len(val['y']), dtype=bool))
    result, decisions, risk, nearest = compare(run(train, cal), measured, bank)
    _, _, guard, predicted, _, score = measured
    cases = [{'source': str(source), 'label': str(label), 'guard_pass': bool(guard[i]), 'predicted': LABELS[int(predicted[i])],
              'score': float(score[i]) if guard[i] else None, 'template_risk': risk[i].tolist() if guard[i] else None,
              'nearest_alternative': str(nearest[i]) if guard[i] else None,
              'accepted': {method: bool(decisions[method][i]) for method in METHODS}}
             for i, (source, label) in enumerate(zip(val['source_id'], val['raw_label']))]
    return {'methods': result, 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'run'))
    args = parser.parse_args()
    verify_or_lock(args.mode)
    if args.mode == 'lock':
        print(f'Locked V8 verifier: {digest(LOCK)}')
    else:
        train = read_split('train')
        development = grouped(train)
        write_json(OUT / 'grouped-verifier.json', {'protocol_sha256': digest(LOCK), **development})
        diagnostic = product_diagnostic(train)
        write_json(OUT / 'product-verifier-diagnostic.json', {'protocol_sha256': digest(LOCK), **diagnostic})
        print(json.dumps({'grouped': development['aggregate'], 'product': {name: row['assessment'] for name, row in diagnostic['methods'].items()}}, ensure_ascii=False, indent=2))
