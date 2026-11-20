"""Train, select, and freeze a browser-compatible research candidate on recovered GSL."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F

from common import fit_temperature, softmax
from openset_core import motion_metrics, score_gate
from train import seed_everything
from v5_grouped_experiment import CALIBRATION, FLOORS, LABELS, NOVEL, ROOT, SCORE_GRID, counts, digest, sample_weights, write_json
from v6_calibration_experiment import eligible, violation
from v7_model import make_model


DATA = ROOT / 'data/v7-recovery'
ARTIFACTS = ROOT / 'artifacts/v7'
REPORT = ROOT / 'reports/v7'
LOCK = ARTIFACTS / 'training-protocol.locked.json'
SELECTION = ARTIFACTS / 'selection.locked.json'
PEOPLE = ('1', '2', '4', '5')
METHODS = {
    'original_tcn': ('original', 'tcn'),
    'recovered_tcn': ('fused', 'tcn'),
    'recovered_motion': ('fused', 'masked_motion'),
}
SEED = 7319
EPOCHS = 40
QUALITY_WEIGHT = .35
OOD_LOSS_WEIGHT = .10


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def protocol():
    sources = [Path(__file__).resolve(), DATA / 'extraction-protocol.json',
               ROOT / 'web/public/models/model.onnx', ROOT / 'web/public/models/model_config.json']
    sources.extend(ROOT / 'ml' / name for name in (
        'v7_model.py', 'train.py', 'common.py', 'openset_core.py', 'v5_grouped_experiment.py', 'v6_calibration_experiment.py',
    ))
    return {
        'id': 'signbridge-v7-common-landmark-core-v1',
        'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        'methods': METHODS, 'development_people': PEOPLE,
        'calibration_person_by_development_fold': {person: CALIBRATION[person] for person in PEOPLE},
        'excluded_negative_labels_by_fold': NOVEL,
        'train': {'seed': SEED, 'epochs': EPOCHS, 'batch_size': 32, 'optimizer': 'AdamW lr=.001 weight_decay=.001',
                  'weights': 'V5 class/gloss/person balance, multiplied by .35 when the raw guard fails',
                  'loss': 'Weighted four-way cross entropy plus .10 uniform cross entropy over known logits for OTHER rows.'},
        'gate': {'threshold_grid': SCORE_GRID, 'targets': FLOORS,
                 'temperature': 'fit on raw-guard-passing calibration rows only; otherwise T=1 if fewer than 4 rows',
                 'decision': 'Raw guard, then top known strictly exceeds OTHER and meets the global threshold.'},
        'selection': 'Inside train338 only. Max joint eligible folds, then min worst and mean assessment target violation, then max mean sign recall and min mean unknown FAR.',
        'frozen_diagnostic': {'fit_people': ['2', '4', '5'], 'calibration_people': ['1'], 'assessment_people': ['6'],
                              'restriction': 'The already exposed person 6 set is opened only after candidate checkpoint and gate are frozen. Research only.'},
        'scope': 'Development RGB records only, no old 129, fresh people or human webcam. Never automatically replace the shipped model.',
    }


def verify_or_lock(mode):
    current = json.loads(json.dumps(protocol()))
    if mode == 'lock' and not LOCK.exists():
        write_json(LOCK, current)
    if not LOCK.is_file() or read(LOCK) != current:
        raise RuntimeError('Missing or changed V7 training protocol')


def read_split(split):
    pack = read(DATA / 'pack-report.json')
    plan = read(DATA / 'extraction-protocol.json')
    path = DATA / f'{split}.npz'
    expected = {row['source_id']: row for row in plan['samples'] if row['split'] == split}
    if pack['protocolSha256'] != digest(DATA / 'extraction-protocol.json') or digest(path) != pack['splits'][split]['npzSha256']:
        raise ValueError('Packed split and extraction protocol do not match')
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files if key != 'metadata_json'}
        metadata = json.loads(str(data['metadata_json'].item()))
    if metadata['protocolSha256'] != pack['protocolSha256'] or metadata['pipelineSha256'] != plan['pipelineSha256']:
        raise ValueError('Feature provenance mismatch')
    n = len(values['y'])
    if n != (338 if split == 'train' else 86) or tuple(values['classes']) != LABELS or set(values['split']) != {split}:
        raise ValueError('Unexpected classes or split size')
    if len(set(values['source_id'])) != n or set(values['source_id']) != set(expected):
        raise ValueError('Missing or duplicate clip')
    if len({row['sourceContentSha256'] for row in plan['samples']}) != 424:
        raise ValueError('Duplicate RGB sequence in locked corpus')
    for index, source in enumerate(values['source_id']):
        row = expected[source]
        if (row['y'], row['signer'], row['raw_label']) != (values['y'][index], values['signer'][index], values['raw_label'][index]):
            raise ValueError('Source label/signer mismatch')
    values['signer'] = values['signer'].astype(str)
    if set(values['signer']) != (set(PEOPLE) if split == 'train' else {'6'}):
        raise ValueError('Forbidden person in split')
    for stream in ('original', 'fused'):
        x = values['x_' + stream]
        raw_guard = values['guard_' + stream]
        quality = values['quality_' + stream]
        if (x.shape != (n, 48, 224) or x.dtype != np.float32 or not np.isfinite(x).all()
                or raw_guard.shape != (n,) or raw_guard.dtype != np.bool_
                or quality.shape != (n,) or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1))):
            raise ValueError('Invalid browser feature or guard array')
    if np.any(values['guard_original'] & ~values['guard_fused']):
        raise ValueError('Recovery removed a previously usable clip')
    return values


def logits_for(model, x):
    with torch.inference_mode():
        return torch.cat([model(batch) for batch in torch.from_numpy(x).split(32)]).numpy()


def fit_model(architecture, x, y, person, raw_label, raw_guard, seed):
    seed_everything(seed)
    model = make_model(architecture, x)
    weights = sample_weights(y, person, raw_label) * np.where(raw_guard, 1., QUALITY_WEIGHT)
    weights = torch.from_numpy((weights / weights.mean()).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    samples = torch.from_numpy(x)
    targets = torch.from_numpy(np.where(y < 0, 3, y))
    generator = torch.Generator().manual_seed(seed)
    losses = []
    started = time.perf_counter()
    for _ in range(EPOCHS):
        epoch_loss = 0.
        for batch in torch.randperm(len(x), generator=generator).split(32):
            optimizer.zero_grad(set_to_none=True)
            logits = model(samples[batch])
            cross_entropy = F.cross_entropy(logits, targets[batch], reduction='none')
            known_logp = F.log_softmax(logits[:, :3], dim=1)
            uncertainty = -known_logp.mean(dim=1) - np.log(3.)
            loss = ((cross_entropy + OOD_LOSS_WEIGHT * uncertainty * (targets[batch] == 3)) * weights[batch]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        losses.append(epoch_loss / len(x))
    return model.eval(), {'seconds': time.perf_counter() - started, 'loss': losses, 'fit_n': len(y), 'fit_guard_pass': int(raw_guard.sum())}


def predictions(logits, raw_guard, temperature, threshold):
    probs = softmax(logits, temperature)
    predicted = probs[:, :3].argmax(axis=1)
    score = probs[np.arange(len(logits)), predicted]
    accepted = raw_guard & (score > probs[:, 3]) & (score >= threshold)
    return predicted, accepted, score, probs[:, 3]


def metrics(y, raw_guard, predicted, accepted):
    value = counts(y, predicted, accepted)
    value.update(raw_guard_pass=int(raw_guard.sum()), raw_guard_rejected=int((~raw_guard).sum()),
                 known_raw_guard_rejected=int(((y >= 0) & ~raw_guard).sum()), eligible=eligible(value))
    return value


def calibrate(logits, y, guard):
    temperature = fit_temperature(logits[guard], np.where(y[guard] < 0, 3, y[guard])) if guard.sum() >= 4 else 1.
    options = []
    for threshold in SCORE_GRID:
        predicted, accepted, _, _ = predictions(logits, guard, temperature, threshold)
        result = metrics(y, guard, predicted, accepted)
        rank = (result['eligible'], -violation(result), -result['unknown_false_acceptance'], result['macro_sign_recall'])
        options.append((rank, {'temperature': temperature, 'threshold': threshold, 'calibration': result}))
    return max(options, key=lambda option: option[0])[1]


def development_fold(data, person):
    y = data['y']
    people = data['signer']
    novel = (y < 0) & np.isin(data['raw_label'], NOVEL[person])
    assessment = people == person
    calibration = (people == CALIBRATION[person]) & ~novel
    fit = ~np.isin(people, (person, CALIBRATION[person])) & ~novel
    if any(set(y[mask]) != {-1, 0, 1, 2} for mask in (fit, calibration, assessment)) or not np.any(assessment & novel):
        raise ValueError('Insufficient label support for grouped fold')
    return fit, calibration, assessment, novel


def fit_candidate(data, method, fit, calibration, seed):
    stream, architecture = METHODS[method]
    x = data['x_' + stream]
    y = data['y']
    guard = data['guard_' + stream]
    model, history = fit_model(architecture, x[fit], y[fit], data['signer'][fit], data['raw_label'][fit], guard[fit], seed)
    gate = calibrate(logits_for(model, x[calibration]), y[calibration], guard[calibration])
    return model, gate, history


def run_development():
    if SELECTION.exists():
        raise RuntimeError('A candidate is already frozen. Run diagnostic with that artifact.')
    data = read_split('train')
    results = {name: [] for name in METHODS}
    for index, person in enumerate(PEOPLE):
        fit, calibration, assessment, novel = development_fold(data, person)
        for name, (stream, _) in METHODS.items():
            model, gate, history = fit_candidate(data, name, fit, calibration, SEED + index)
            guard = data['guard_' + stream][assessment]
            predicted, accepted, _, _ = predictions(logits_for(model, data['x_' + stream][assessment]), guard, gate['temperature'], gate['threshold'])
            result = metrics(data['y'][assessment], guard, predicted, accepted)
            result.update(novel_n=int(novel[assessment].sum()), novel_false_accepted=int((novel[assessment] & accepted).sum()))
            results[name].append({'assessment_person': person, 'calibration_person': CALIBRATION[person], 'fit_people': sorted(set(data['signer'][fit])),
                                  'fit_label_counts': dict(Counter(data['raw_label'][fit])), 'gate': gate, 'assessment': result, 'fit_history': history})
            print(f'{person} {name}: {result["correct_known_accepted"]}/{result["known_n"]} known, '
                  f'{result["unknown_false_accepted"]}/{result["unknown_n"]} false unknown', flush=True)
    aggregate = {}
    for method, rows in results.items():
        measured = [row['assessment'] for row in rows]
        penalties = [violation(row) for row in measured]
        aggregate[method] = {
            'joint_eligible_folds': sum(row['gate']['calibration']['eligible'] and row['assessment']['eligible'] for row in rows),
            'worst_violation': max(penalties), 'mean_violation': float(np.mean(penalties)),
            'mean_sign_recall': float(np.mean([row['macro_sign_recall'] for row in measured])),
            'mean_unknown_false_acceptance': float(np.mean([row['unknown_false_acceptance'] for row in measured])),
            'correct_known_accepted': sum(row['correct_known_accepted'] for row in measured),
            'known_n': sum(row['known_n'] for row in measured),
            'unknown_false_accepted': sum(row['unknown_false_accepted'] for row in measured),
            'unknown_n': sum(row['unknown_n'] for row in measured),
        }
    selected = max(aggregate, key=lambda name: (
        aggregate[name]['joint_eligible_folds'], -aggregate[name]['worst_violation'], -aggregate[name]['mean_violation'],
        aggregate[name]['mean_sign_recall'], -aggregate[name]['mean_unknown_false_acceptance'],
    ))
    fit = np.isin(data['signer'], ('2', '4', '5'))
    calibration = data['signer'] == '1'
    if np.any((data['y'] < 0) & np.isin(data['raw_label'], NOVEL['6'])):
        raise ValueError('Person-6 novel labels appeared in model selection inputs')
    model, gate, history = fit_candidate(data, selected, fit, calibration, SEED + 10)
    checkpoint = ARTIFACTS / 'candidate.pt'
    torch.save({'state_dict': model.state_dict(), 'method': selected}, checkpoint)
    document = {'protocol_sha256': digest(LOCK), 'train_sha256': digest(DATA / 'train.npz'),
                'aggregate': aggregate, 'folds': results, 'selected': selected, 'gate': gate, 'refit': history,
                'checkpoint_sha256': digest(checkpoint), 'person6_tensors_opened': False, 'production_approved': False}
    write_json(SELECTION, document)
    print(f'Frozen candidate: {selected}, selection SHA-256 {digest(SELECTION)}')


def load_candidate():
    selection = read(SELECTION)
    checkpoint = ARTIFACTS / 'candidate.pt'
    if (selection['protocol_sha256'] != digest(LOCK) or selection['checkpoint_sha256'] != digest(checkpoint)
            or selection['train_sha256'] != digest(DATA / 'train.npz')):
        raise ValueError('Frozen candidate or training artifact changed')
    chosen = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if chosen['method'] != selection['selected']:
        raise ValueError('Wrong candidate checkpoint')
    _, architecture = METHODS[selection['selected']]
    model = make_model(architecture, np.zeros((1, 48, 224), dtype=np.float32))
    model.load_state_dict(chosen['state_dict'])
    return model.eval(), selection


def diagnostic():
    torch.set_num_threads(2)
    model, selection = load_candidate()
    data = read_split('val')
    config = read(ROOT / 'web/public/models/model_config.json')
    session = ort.InferenceSession(str(ROOT / 'web/public/models/model.onnx'), providers=['CPUExecutionProvider'])
    y = data['y']
    result = {}
    for stream in ('original', 'fused'):
        x = data['x_' + stream]
        guard = data['guard_' + stream]
        outputs = [session.run(None, {config['input']['name']: row[None]}) for row in x]
        logits, embedding = (np.concatenate(parts) for parts in zip(*outputs))
        scoring = score_gate(logits, data['quality_' + stream], motion_metrics(x)['motionBodyRms'], config, embedding)
        accepted = guard & scoring['accepted']
        result['v4_' + stream] = metrics(y, guard, scoring['predicted'], accepted)
        result['v4_' + stream]['unknown_by_label'] = unknown_by_label(y, data['raw_label'], accepted)
    stream, _ = METHODS[selection['selected']]
    guard = data['guard_' + stream]
    gate = selection['gate']
    predicted, accepted, score, other = predictions(logits_for(model, data['x_' + stream]), guard, gate['temperature'], gate['threshold'])
    result['v7_candidate'] = metrics(y, guard, predicted, accepted)
    result['v7_candidate']['unknown_by_label'] = unknown_by_label(y, data['raw_label'], accepted)
    decisions = [{'source_id': str(source), 'label': str(label), 'guard_pass': bool(usable), 'predicted': LABELS[int(index)],
                  'accepted': bool(decision), 'score': float(value) if usable else None, 'other_score': float(bg) if usable else None}
                 for source, label, usable, index, decision, value, bg in zip(data['source_id'], data['raw_label'], guard, predicted, accepted, score, other)]
    write_json(REPORT / 'candidate-diagnostic.json', {'selection_sha256': digest(SELECTION), 'val_sha256': digest(DATA / 'val.npz'),
                                                    'scope': 'Previously exposed signer 6, offline end-to-end guard and classification replay',
                                                    'candidate': selection['selected'], 'metrics': result, 'candidate_decisions': decisions})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def unknown_by_label(y, labels, accepted):
    return {str(label): {'n': int((labels == label).sum()), 'false_accepted': int(((labels == label) & accepted).sum())}
            for label in sorted(set(labels[y < 0]))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'train', 'diagnostic'))
    args = parser.parse_args()
    verify_or_lock(args.mode)
    if args.mode == 'train':
        run_development()
    elif args.mode == 'diagnostic':
        diagnostic()
    else:
        print(f'Locked training plan: {digest(LOCK)}')
