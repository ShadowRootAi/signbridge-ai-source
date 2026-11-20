"""Fit and assess SignBridge candidates on previously seen development data only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import fit_temperature, softmax
from openset_train import load_split
from train import TemporalClassifier, fit_normalization, normalize_numpy, seed_everything


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/v3'
REPORT = ROOT / 'reports/v5'
ARTIFACTS = ROOT / 'artifacts/v5-grouped'
LOCK = ARTIFACTS / 'protocol.locked.json'
PEOPLE = ('1', '2', '4', '5', '6')
LABELS = ('ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ')
CALIBRATION = {'1': '2', '2': '4', '4': '5', '5': '1', '6': '1'}
NOVEL = {
    '1': ('ΚΑΤΑΛΑΒΑΙΝΩ',), '2': ('ΝΑΙ',), '4': ('ΗΜΕΡΑ_ΤΡΕΙΣ',),
    '5': ('ΤΕΛΕΙΑ',), '6': ('ΟΧΙ', 'ΚΑΛΟ'),
}
METHODS = ('centroid', 'tcn', 'tcn_distance', 'tcn_dropout_distance')
SCORE_GRID = (.25, .35, .45, .55, .65, .75, .85, .90, .95, .98, .995)
DISTANCE_GRID = (None, .90, .97, 1.0)
FLOORS = {'per_sign_correct_acceptance': .60, 'accepted_precision': .80, 'unknown_false_acceptance_max': .20}
SEED = 5206
EPOCHS = 32
CHANNELS = 32


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def protocol():
    sources = [Path(__file__).resolve(), DATA / 'train.npz', DATA / 'val.npz']
    sources.extend(ROOT / 'ml' / name for name in ('common.py', 'train.py', 'preprocess.py', 'openset_train.py'))
    return {
        'id': 'signbridge-v5-grouped-development-v1',
        'purpose': 'Adaptive research on existing development signers. No independent evaluation or browser deployment.',
        'files': {str(path.relative_to(ROOT)): digest(path) for path in sources},
        'people': PEOPLE, 'labels': LABELS, 'calibration_person': CALIBRATION,
        'assessment_only_unknown_labels': NOVEL, 'methods': METHODS,
        'score_grid': SCORE_GRID, 'distance_training_quantile_grid': DISTANCE_GRID,
        'calibration_targets': FLOORS,
        'calibration_fallback': 'If no gate satisfies all targets, report minimum total violation and mark ineligible.',
        'training': {
            'seed': SEED, 'epochs': EPOCHS, 'channels': CHANNELS, 'batch_size': 32,
            'optimizer': 'AdamW, lr=.001, weight_decay=.001',
            'weighting': 'Equal known class and pooled OTHER mass, equal negative subclass mass within OTHER, equal signers within each subclass.',
            'dropout': 'Probability .40 per training clip: zero both coordinate blocks and presence of one hand in a contiguous 8 to 32 of 48 frame window. Same transform for known and unknown.',
            'distance': 'Squared distance to predicted-class embedding mean using pooled within-known-class diagonal variance shrunk 25% toward median variance. Scale by per-class training quantile.',
        },
        'limitations': [
            'All five source people were already seen in earlier work. The 129 exposed regression samples are not loaded.',
            'Saved normalized-length features cannot reproduce the raw V4 tracking guard. This is a classifier comparison.',
            'No human webcam data or independently verified new signer is available.',
        ],
    }


def locked_protocol(mode):
    current = json.loads(json.dumps(protocol()))
    if mode == 'lock' and not LOCK.exists():
        write_json(LOCK, current)
    if not LOCK.is_file() or json.loads(LOCK.read_text(encoding='utf-8')) != current:
        raise RuntimeError('Missing or changed protocol lock. Use a new experiment version for changes.')
    return current


def read_data():
    splits = [load_split(DATA / f'{name}.npz', name) for name in ('train', 'val')]
    if any(tuple(part.classes) != LABELS for part in splits):
        raise ValueError('Unexpected class order')
    x, y, people, label, source = (
        np.concatenate([getattr(part, field) for part in splits])
        for field in ('x', 'y', 'signer', 'raw_label', 'source_id')
    )
    if len(y) != 424 or len(set(source)) != len(y) or set(people) != set(PEOPLE):
        raise ValueError('Unexpected development sample set')
    return x, y, people, label, source


def get_fold(person, y, people, label):
    assessment = people == person
    calibration = people == CALIBRATION[person]
    novel = (y < 0) & np.isin(label, NOVEL[person])
    fit = ~(assessment | calibration | novel)
    calibration &= ~novel
    if set(people[fit]) & set(people[assessment | calibration]):
        raise ValueError('Signer overlap')
    if np.any(novel & (fit | calibration)) or not np.any(assessment & novel):
        raise ValueError('Assessment-only negative category overlap or absence')
    for mask in (fit, calibration, assessment):
        if set(y[mask]) != {-1, 0, 1, 2}:
            raise ValueError('Missing known class or OTHER in split')
    return fit, calibration, assessment, novel


def sample_weights(y, people, raw_label):
    weights = np.zeros(len(y), dtype=np.float32)
    for target in (-1, 0, 1, 2):
        category = y == target
        subclasses = np.unique(raw_label[category])
        for label in subclasses:
            subgroup = category & (raw_label == label)
            signers = np.unique(people[subgroup])
            for person in signers:
                rows = subgroup & (people == person)
                weights[rows] = 1 / (4 * len(subclasses) * len(signers) * rows.sum())
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)
    return weights * len(y)


def drop_one_hand(batch, rng):
    batch = batch.clone()
    for index in np.flatnonzero(rng.random(len(batch)) < .40):
        hand = int(rng.integers(2))
        length = int(rng.integers(8, 33))
        start = int(rng.integers(0, 49 - length))
        batch[index, start:start + length, hand * 84:(hand + 1) * 84] = 0
        batch[index, start:start + length, 220 + hand] = 0
    return batch


def train_tcn(x, y, weights, normalization, dropout, seed):
    seed_everything(seed)
    mean, std, keep = normalization
    model = TemporalClassifier(mean, std, 4, CHANNELS, keep)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    samples = torch.from_numpy(x)
    targets = torch.from_numpy(np.where(y < 0, 3, y))
    weights = torch.from_numpy(weights)
    order = torch.Generator().manual_seed(seed)
    augmentation = np.random.default_rng(seed + 1)
    started = time.perf_counter()
    for _ in range(EPOCHS):
        for indices in torch.randperm(len(x), generator=order).split(32):
            batch = samples[indices]
            if dropout:
                batch = drop_one_hand(batch, augmentation)
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(model(batch), targets[indices], reduction='none') * weights[indices]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    return model.eval(), time.perf_counter() - started


def outputs(model, x):
    embeddings = []
    handle = model.classifier.register_forward_pre_hook(lambda _, args: embeddings.append(args[0].numpy().copy()))
    try:
        with torch.inference_mode():
            logits = np.concatenate([model(torch.from_numpy(batch)).numpy() for batch in np.array_split(x, 6) if len(batch)])
    finally:
        handle.remove()
    return logits, np.concatenate(embeddings)


def class_distances(train_embedding, y, query_embedding):
    centers = np.stack([train_embedding[y == k].mean(axis=0) for k in range(3)])
    residual = train_embedding[y >= 0] - centers[y[y >= 0]]
    variance = np.square(residual).mean(axis=0)
    variance = np.maximum(.75 * variance + .25 * np.median(variance), 1e-5)
    training_distances = np.mean(np.square(residual) / variance, axis=1)
    distances = np.mean(np.square(query_embedding[:, None, :] - centers) / variance, axis=2)
    caps = {q: np.maximum([np.quantile(training_distances[y[y >= 0] == k], q) for k in range(3)], 1e-6)
            for q in DISTANCE_GRID if q is not None}
    return distances, caps


def score(logits, temperature, threshold, distances=None, caps=None):
    probability = softmax(logits, temperature)
    predicted = probability[:, :3].argmax(axis=1)
    confidence = probability[np.arange(len(predicted)), predicted]
    accepted = (confidence > probability[:, 3]) & (confidence >= threshold)
    if caps is not None:
        accepted &= distances[np.arange(len(predicted)), predicted] <= caps[predicted]
    return predicted, accepted, confidence


def counts(y, predicted, accepted):
    known = y >= 0
    correct = known & accepted & (y == predicted)
    by_sign = {label: {'n': int((y == i).sum()), 'correct_accepted': int((correct & (y == i)).sum())}
               for i, label in enumerate(LABELS)}
    per_sign = [row['correct_accepted'] / row['n'] for row in by_sign.values()]
    unknown_n = int((~known).sum())
    unknown_false = int((~known & accepted).sum())
    accepted_n = int(accepted.sum())
    return {
        'per_sign': by_sign, 'minimum_sign_recall': min(per_sign), 'macro_sign_recall': float(np.mean(per_sign)),
        'known_n': int(known.sum()), 'correct_known_accepted': int(correct.sum()),
        'wrong_known_accepted': int((known & accepted & (y != predicted)).sum()),
        'known_rejected': int((known & ~accepted).sum()),
        'unknown_n': unknown_n, 'unknown_false_accepted': unknown_false,
        'unknown_false_acceptance': unknown_false / unknown_n,
        'accepted_n': accepted_n, 'accepted_precision': int(correct.sum()) / accepted_n if accepted_n else 0.,
    }


def calibrate(logits, y, distances=None, caps_by_quantile=None):
    temperature = fit_temperature(logits, np.where(y < 0, 3, y))
    grid = []
    for quantile in DISTANCE_GRID if caps_by_quantile is not None else (None,):
        caps = None if quantile is None else caps_by_quantile[quantile]
        for threshold in SCORE_GRID:
            predicted, accepted, _ = score(logits, temperature, threshold, distances, caps)
            metrics = counts(y, predicted, accepted)
            violation = (sum(max(0., FLOORS['per_sign_correct_acceptance'] - row['correct_accepted'] / row['n'])
                             for row in metrics['per_sign'].values())
                         + max(0., FLOORS['accepted_precision'] - metrics['accepted_precision'])
                         + max(0., metrics['unknown_false_acceptance'] - FLOORS['unknown_false_acceptance_max']))
            rank = (violation == 0, -violation, -metrics['unknown_false_acceptance'], metrics['macro_sign_recall'],
                    metrics['minimum_sign_recall'], quantile is None)
            grid.append((rank, {'threshold': threshold, 'distance_quantile': quantile,
                                'eligible': violation == 0, 'violation': violation, 'metrics': metrics}))
    best = max(grid, key=lambda item: item[0])[1]
    best['temperature'] = temperature
    return best


def run():
    x, y, people, label, source = read_data()
    results = {}
    predictions = {method: [] for method in METHODS}
    for fold_number, person in enumerate(PEOPLE):
        fit, calibration, assessment, novel = get_fold(person, y, people, label)
        print(f'Fold {person}: fit={fit.sum()}, calibration={calibration.sum()}, assessment={assessment.sum()}', flush=True)
        mean, std, keep, _ = fit_normalization(x[fit], 'signbridge-landmarks-xy-v1')
        normal = normalize_numpy(x, mean, std, keep).reshape(len(x), -1)
        centers = np.stack([normal[fit & (y == k)].mean(axis=0) for k in (0, 1, 2, -1)])
        centroid_logits = -np.stack([np.square(normal - center).mean(axis=1) for center in centers], axis=1)
        weights = sample_weights(y[fit], people[fit], label[fit])
        standard, standard_time = train_tcn(x[fit], y[fit], weights, (mean, std, keep), False, SEED + fold_number)
        regularized, regularized_time = train_tcn(x[fit], y[fit], weights, (mean, std, keep), True, SEED + fold_number)
        standard_logits, standard_embedding = outputs(standard, x)
        regularized_logits, regularized_embedding = outputs(regularized, x)
        base_dist, base_caps = class_distances(standard_embedding[fit], y[fit], standard_embedding)
        regularized_dist, regularized_caps = class_distances(regularized_embedding[fit], y[fit], regularized_embedding)
        models = {
            'centroid': (centroid_logits, None, None), 'tcn': (standard_logits, None, None),
            'tcn_distance': (standard_logits, base_dist, base_caps),
            'tcn_dropout_distance': (regularized_logits, regularized_dist, regularized_caps),
        }
        fold_result = {
            'fit_people': sorted(set(people[fit])), 'fit_n': int(fit.sum()),
            'calibration_person': CALIBRATION[person], 'calibration_n': int(calibration.sum()),
            'assessment_person': person, 'assessment_n': int(assessment.sum()),
            'assessment_only_unknown_labels': NOVEL[person],
            'fit_label_counts': dict(Counter(label[fit])), 'calibration_label_counts': dict(Counter(label[calibration])),
            'assessment_label_counts': dict(Counter(label[assessment])),
            'training_seconds': {'tcn': standard_time, 'tcn_dropout': regularized_time}, 'methods': {},
        }
        for name, (logits, distances, caps) in models.items():
            gate = calibrate(logits[calibration], y[calibration], None if distances is None else distances[calibration], caps)
            quantile = gate['distance_quantile']
            if quantile is not None:
                gate['distance_caps'] = caps[quantile].tolist()
            predicted, accepted, confidence = score(logits[assessment], gate['temperature'], gate['threshold'],
                                                   None if distances is None else distances[assessment],
                                                   None if quantile is None else caps[quantile])
            metrics = counts(y[assessment], predicted, accepted)
            metrics['novel_unknown_n'] = int(novel[assessment].sum())
            metrics['novel_unknown_false_accepted'] = int((novel[assessment] & accepted).sum())
            fold_result['methods'][name] = {'gate': gate, 'assessment': metrics}
            for i, row in enumerate(np.flatnonzero(assessment)):
                predictions[name].append({'source': str(source[row]), 'person': person, 'label': str(label[row]),
                                          'y': int(y[row]), 'predicted': int(predicted[i]),
                                          'accepted': bool(accepted[i]), 'score': float(confidence[i]),
                                          'novel_unknown': bool(novel[row])})
        for model_name, model in (('tcn', standard), ('tcn_dropout', regularized)):
            torch.save({'state_dict': model.state_dict(), 'channels': CHANNELS, 'labels': LABELS,
                        'assessment_person': person, 'model_name': model_name}, ARTIFACTS / f'{model_name}-fold-{person}.pt')
        results[person] = fold_result
    aggregate = {}
    for name, rows in predictions.items():
        if len({row['source'] for row in rows}) != len(y):
            raise ValueError('Assessment does not cover each clip exactly once')
        aggregate[name] = counts(*(np.asarray([row[key] for row in rows]) for key in ('y', 'predicted', 'accepted')))
        aggregate[name].update({
            'novel_unknown_n': sum(row['novel_unknown'] for row in rows),
            'novel_unknown_false_accepted': sum(row['novel_unknown'] and row['accepted'] for row in rows),
            'eligible_calibration_folds': sum(fold['methods'][name]['gate']['eligible'] for fold in results.values()),
            'assessment_folds_meeting_targets': sum(
                (m['minimum_sign_recall'] >= FLOORS['per_sign_correct_acceptance']
                 and m['accepted_precision'] >= FLOORS['accepted_precision']
                 and m['unknown_false_acceptance'] <= FLOORS['unknown_false_acceptance_max'])
                for m in (fold['methods'][name]['assessment'] for fold in results.values())
            ),
        })
    document = {'protocol_sha256': digest(LOCK), 'scope': 'Development-only grouped classifier assessment',
                'exposed_129_loaded': False, 'human_webcam': False, 'raw_tracking_guard_applied': False,
                'release_candidate_approved': False, 'aggregate': aggregate, 'folds': results,
                'checkpoint_hashes': {p.name: digest(p) for p in sorted(ARTIFACTS.glob('*.pt'))},
                'predictions': predictions}
    write_json(REPORT / 'grouped-model-experiment.json', document)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'run'))
    args = parser.parse_args()
    locked_protocol(args.mode)
    if args.mode == 'run':
        run()
    else:
        print(f'Locked protocol SHA-256: {digest(LOCK)}')
