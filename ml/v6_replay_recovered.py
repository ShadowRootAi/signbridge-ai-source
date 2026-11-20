"""Replay locked classifiers on original and recovered V6 landmark streams."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import softmax
from openset_core import motion_metrics, score_gate
from preprocess import preprocess
from train import TemporalClassifier
from v4_tracking import measure_raw_tracking
from v5_grouped_experiment import LABELS, ROOT, digest, outputs, write_json


REPORT = ROOT / 'reports/v6'
LOCK = REPORT / 'recovery-replay.locked.json'
HAND_PROTOCOL = REPORT / 'hand-recovery-protocol.json'
V4_MODEL = ROOT / 'web/public/models/model.onnx'
V4_CONFIG = ROOT / 'web/public/models/model_config.json'
V5_REPORT = ROOT / 'reports/v5/grouped-model-experiment.json'
V5_MODEL = ROOT / 'artifacts/v5-grouped/tcn_dropout-fold-6.pt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def protocol():
    paths = [Path(__file__).resolve(), HAND_PROTOCOL, V4_MODEL, V4_CONFIG, V5_REPORT, V5_MODEL]
    paths.extend(ROOT / 'ml' / name for name in ('preprocess.py', 'v4_tracking.py', 'train.py', 'openset_core.py', 'common.py'))
    return {
        'id': 'signbridge-v6-recovery-replay-v1',
        'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        'scope': 'Previously seen signer 6 Gamma clips and derived controls, offline. Recovery inputs only; two frozen classifiers.',
        'classifier_a': 'V4 shipped ONNX with frozen configuration, historically calibrated using person 6.',
        'classifier_b': 'V5 dropout person-6 holdout checkpoint and its pre-existing calibration gate.',
        'rules': 'Apply exact V4 raw guard to each stream; execute classifier only when it passes. Controls are artificial occlusion/blank, not novel semantic signs.',
    }


def verify_or_lock(mode):
    current = protocol()
    if mode == 'lock' and not LOCK.exists():
        write_json(LOCK, current)
    if not LOCK.is_file() or read(LOCK) != current:
        raise RuntimeError('Missing or changed replay lock')


def classify_v4(session, config, x, quality):
    values = session.run(None, {config['input']['name']: x[None]})
    scored = score_gate(values[0], np.array([quality]), motion_metrics(x[None])['motionBodyRms'], config, values[1])
    return {'accepted': bool(scored['accepted'][0]), 'top_sign': LABELS[int(scored['predicted'][0])],
            'score': float(scored['confidence'][0]), 'rejection_reasons': [key for key, flag in scored['reasons'].items() if flag[0]]}


def classify_v5(model, gate, x):
    logits, _ = outputs(model, x[None])
    probabilities = softmax(logits, gate['temperature'])[0]
    predicted = int(probabilities[:3].argmax())
    score = float(probabilities[predicted])
    reasons = []
    if score <= probabilities[3]:
        reasons.append('background')
    if score < gate['threshold']:
        reasons.append('confidence')
    return {'accepted': not reasons, 'top_sign': LABELS[predicted], 'score': score, 'rejection_reasons': reasons}


def run():
    tracking = read(REPORT / 'hand-recovery-results.json')
    planned = read(HAND_PROTOCOL)
    if tracking['protocolSha256'] != digest(HAND_PROTOCOL) or not tracking.get('completed'):
        raise ValueError('Tracking results must match the completed locked protocol')
    plan = {sample['id']: sample for sample in planned['samples']}
    person6 = {sample['id'] for sample in planned['samples'] if sample['split'] == 'val' and sample['signer'] == 6}
    expected = {(identifier, control) for identifier in person6 for control in planned['conditions']}
    relevant = [row for row in tracking['rows'] if row['id'] in person6]
    if len(person6) != 11 or len(relevant) != len(expected) or {(r['id'], r['control']) for r in relevant} != expected:
        raise ValueError('Unexpected person-6 replay sample set')
    config = read(V4_CONFIG)
    session = ort.InferenceSession(str(V4_MODEL), providers=['CPUExecutionProvider'])
    v5 = read(V5_REPORT)
    gate = v5['folds']['6']['methods']['tcn_dropout_distance']['gate']
    if gate['distance_quantile'] is not None or digest(V5_MODEL) != v5['checkpoint_hashes'][V5_MODEL.name]:
        raise ValueError('Unexpected frozen V5 scorer')
    torch.set_num_threads(2)
    checkpoint = torch.load(V5_MODEL, map_location='cpu', weights_only=True)
    if checkpoint['assessment_person'] != '6':
        raise ValueError('Wrong person-6 checkpoint')
    model = TemporalClassifier(np.zeros(224, dtype=np.float32), np.ones(224, dtype=np.float32), 4, checkpoint['channels'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    summary = Counter()
    rows = []
    for row in relevant:
        if len(row['frames']) != len(plan[row['id']]['paths']):
            raise ValueError('Wrong source frame count')
        if any({k: v for k, v in frame['original'].items() if k != 'right'}
               != {k: v for k, v in frame['fused'].items() if k != 'right'} for frame in row['frames']):
            raise ValueError('Recovery modified fields beyond the missing right hand')
        if any(len(frame['original']['right']) >= 21 and frame['original']['right'] != frame['fused']['right']
               for frame in row['frames']):
            raise ValueError('Recovery modified an existing right hand')
        result = {'id': row['id'], 'control': row['control'], 'streams': {}}
        for stream in ('original', 'fused'):
            frames = [frame[stream] for frame in row['frames']]
            guard = measure_raw_tracking(frames)
            browser_guard = row[stream + 'Guard']
            if guard['usable'] != browser_guard['usable'] or guard['requiredDataCompleteness'] != browser_guard['requiredDataCompleteness']:
                raise ValueError('Python/browser raw guard mismatch')
            details = {'guard': guard, 'classifiers': None}
            if guard['usable']:
                x, quality = preprocess(frames)
                details['classifiers'] = {'v4': classify_v4(session, config, x, quality), 'v5': classify_v5(model, gate, x)}
                summary[f'{row["control"]}:{stream}:guard_pass'] += 1
                for name, decision in details['classifiers'].items():
                    if decision['accepted']:
                        outcome = 'gamma' if decision['top_sign'] == LABELS[0] else 'other_sign'
                        summary[f'{row["control"]}:{stream}:{name}:accepted_{outcome}'] += 1
            result['streams'][stream] = details
        rows.append(result)
    write_json(REPORT / 'recovery-replay.json', {
        'lock_sha256': digest(LOCK), 'tracking_sha256': digest(REPORT / 'hand-recovery-results.json'),
        'scope': 'Offline replay on 11 previously exposed Gamma clips and their controls. No raw-guard bypass, retraining or release.',
        'counts': dict(summary), 'rows': rows,
    })
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lock', 'run'))
    args = parser.parse_args()
    verify_or_lock(args.mode)
    if args.mode == 'run':
        run()
    else:
        print(f'Locked recovery replay: {digest(LOCK)}')
