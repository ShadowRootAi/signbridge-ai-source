"""Test whether frozen hand recovery admits more unsupported real signs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from v6_replay_recovered import (
    HAND_PROTOCOL, REPORT, V4_CONFIG, V4_MODEL, V5_MODEL, V5_REPORT,
    classify_v4, classify_v5, read,
)
from train import TemporalClassifier
from v5_grouped_experiment import ROOT, digest, write_json


PROTOCOL = REPORT / 'hand-recovery-unknown-protocol.json'
TRACKING = REPORT / 'hand-recovery-unknown-results.json'
LOCK = REPORT / 'unknown-replay-v2.locked.json'
LABEL_COUNTS = {'ΟΧΙ': 15, 'ΚΑΛΟ': 15, 'ΚΑΤΑΛΑΒΑΙΝΩ': 5, 'ΝΑΙ': 5, 'ΗΜΕΡΑ_ΤΡΕΙΣ': 5}


def verify_or_lock(mode):
    paths = [Path(__file__).resolve(), PROTOCOL, HAND_PROTOCOL, V4_CONFIG, V4_MODEL, V5_REPORT, V5_MODEL,
             ROOT / 'ml/v6_replay_recovered.py', ROOT / 'data/v3/val.npz']
    document = {
        'id': 'signbridge-v6-unknown-transfer-replay-v2',
        'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        'labels': LABEL_COUNTS,
        'scope': 'All 45 previously exposed development unknown clips of signer 6. Same recovery rule; no fit or calibration.',
        'label_provenance': 'Join raw_label from the locked validation NPZ by source_id. The tracking manifest may use the shared __unknown__ label.',
        'rules': 'Score only streams passing the unchanged V4 raw guard. Compare paired false acceptances before and after recovery.',
    }
    if mode == 'lock' and not LOCK.exists():
        write_json(LOCK, document)
    if not LOCK.is_file() or read(LOCK) != document:
        raise RuntimeError('Missing or changed unknown replay lock')


def run():
    data = read(TRACKING)
    if not data.get('completed') or data['protocolSha256'] != digest(PROTOCOL):
        raise ValueError('Transfer run does not match the completed protocol')
    plan = read(PROTOCOL)
    parent = read(HAND_PROTOCOL)
    if plan['parentProtocolSha256'] != digest(HAND_PROTOCOL) or any(plan[key] != parent[key] for key in ('rule', 'model')):
        raise ValueError('Unknown transfer changed the locked recovery rule or model')
    samples = {item['id']: item for item in plan['samples']}
    rows = data['rows']
    if len(rows) != 45 or len(samples) != 45 or {row['id'] for row in rows} != set(samples):
        raise ValueError('Expected all 45 unique unknown clips')
    with np.load(ROOT / 'data/v3/val.npz', allow_pickle=False) as validation:
        reference_labels = {str(source): str(label) for source, label, y, person in
                            zip(validation['source_id'], validation['raw_label'], validation['y'], validation['signer'])
                            if y < 0 and str(person) == '6'}
    if set(reference_labels) != {row['source_id'] for row in rows} or Counter(reference_labels.values()) != LABEL_COUNTS:
        raise ValueError('Unexpected unknown label distribution')
    config = read(V4_CONFIG)
    session = ort.InferenceSession(str(V4_MODEL), providers=['CPUExecutionProvider'])
    v5 = read(V5_REPORT)
    gate = v5['folds']['6']['methods']['tcn_dropout_distance']['gate']
    if gate['distance_quantile'] is not None or digest(V5_MODEL) != v5['checkpoint_hashes'][V5_MODEL.name]:
        raise ValueError('Unexpected V5 checkpoint or gate')
    torch.set_num_threads(2)
    checkpoint = torch.load(V5_MODEL, map_location='cpu', weights_only=True)
    model = TemporalClassifier(np.zeros(224, dtype=np.float32), np.ones(224, dtype=np.float32), 4, checkpoint['channels'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    results = []
    for row in rows:
        if row['source_id'] != samples[row['id']]['source_id']:
            raise ValueError('Wrong clip source')
        result = {'id': row['id'], 'label': reference_labels[row['source_id']], 'streams': {}}
        for stream in ('original', 'fused'):
            guard = row[stream + 'Guard']
            features = row[stream + 'Features']
            if guard['usable'] != (features is not None):
                raise ValueError('Features must be present exactly when raw guard passes')
            decisions = None
            if features is not None:
                x = np.asarray(features['x'], dtype=np.float32)
                quality = float(features['quality'])
                if x.shape != (48, 224) or not np.isfinite(x).all() or not 0 <= quality <= 1:
                    raise ValueError('Invalid browser feature array or quality')
                decisions = {'v4': classify_v4(session, config, x, quality), 'v5': classify_v5(model, gate, x)}
            result['streams'][stream] = {'guard_pass': guard['usable'], 'classifiers': decisions}
        results.append(result)
    summary = {'guard_pass': {s: sum(row['streams'][s]['guard_pass'] for row in results) for s in ('original', 'fused')}}
    for name in ('v4', 'v5'):
        aggregate = Counter()
        per_label = {label: Counter() for label in LABEL_COUNTS}
        for row in results:
            accepted = {}
            for stream in ('original', 'fused'):
                scores = row['streams'][stream]['classifiers']
                accepted[stream] = scores is not None and scores[name]['accepted']
            metrics = {'original_false_accepted': accepted['original'], 'fused_false_accepted': accepted['fused'],
                       'new_false_accepted': accepted['fused'] and not accepted['original'],
                       'removed_false_accepted': accepted['original'] and not accepted['fused']}
            aggregate.update(metrics)
            per_label[row['label']].update(metrics)
        summary[name] = {'total': dict(aggregate), 'per_label': {label: dict(values) for label, values in per_label.items()}}
    write_json(REPORT / 'unknown-replay.json', {'lock_sha256': digest(LOCK), 'tracking_sha256': digest(TRACKING),
                                               'summary': summary, 'rows': results})
    print(json.dumps({name: summary[name]['total'] for name in ('v4', 'v5')} | {'guard_pass': summary['guard_pass']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'run'))
    args = parser.parse_args()
    verify_or_lock(args.mode)
    if args.mode == 'run':
        run()
    else:
        print(f'Locked unknown replay: {digest(LOCK)}')
