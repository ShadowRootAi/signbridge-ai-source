"""Export the frozen V7 model and verify ONNX before emitting browser fixtures."""
from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import softmax
from preprocess import preprocess
from v4_tracking import measure_raw_tracking
from v5_grouped_experiment import LABELS, ROOT, digest, write_json
from v7_train_core import ARTIFACTS, DATA, LOCK, SELECTION, load_candidate, logits_for, read, read_split, verify_or_lock


BROWSER = ARTIFACTS / 'browser'


def reference(logits, gate):
    probabilities = softmax(np.asarray(logits)[None], gate['temperature'])[0]
    index = int(probabilities[:3].argmax())
    score = float(probabilities[index])
    reasons = []
    if score <= probabilities[3]:
        reasons.append('background')
    if score < gate['threshold']:
        reasons.append('confidence')
    return {'predictedIndex': index, 'accepted': not reasons, 'classificationScore': score, 'reasons': reasons}


def run():
    verify_or_lock('export')
    torch.set_num_threads(2)
    model, selection = load_candidate()
    BROWSER.mkdir(parents=True, exist_ok=True)
    filename = BROWSER / 'model.onnx'
    temporary = BROWSER / 'model.tmp.onnx'
    torch.onnx.export(model, torch.zeros(1, 48, 224), str(temporary), input_names=['landmarks'], output_names=['logits'],
                      opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(temporary))
    session = ort.InferenceSession(str(temporary), providers=['CPUExecutionProvider'])
    data = read_split('val')
    gate = selection['gate']
    cases = []
    maximum = 0.
    for stream in ('original', 'fused'):
        x = data['x_' + stream]
        torch_output = logits_for(model, x)
        ort_output = np.concatenate([session.run(['logits'], {'landmarks': row[None]})[0] for row in x])
        np.testing.assert_allclose(torch_output, ort_output, atol=1e-5, rtol=1e-4)
        maximum = max(maximum, float(np.abs(torch_output - ort_output).max()))
        for expected, actual in zip(torch_output, ort_output):
            a, b = reference(expected, gate), reference(actual, gate)
            if (a['predictedIndex'], a['accepted'], a['reasons']) != (b['predictedIndex'], b['accepted'], b['reasons']):
                raise ValueError('PyTorch/ONNX decision mismatch')
        for label in sorted(set(data['raw_label'])):
            for index in np.flatnonzero(data['raw_label'] == label)[:2]:
                source = str(data['source_id'][index])
                cases.append({'id': f'{stream}:{source}', 'sourceId': source, 'split': 'development-person-6', 'origin': stream,
                              'trueLabel': str(label), 'classId': int(data['y'][index]), 'x': x[index].tolist(),
                              'expectedLogits': torch_output[index].tolist(), 'expected': reference(torch_output[index], gate)})
    raw_source = ROOT / 'reports/v6/hand-recovery-results.json'
    recovery = read(raw_source)
    if not recovery['completed']:
        raise ValueError('Incomplete tracking fixture source')
    anchor = next(row for row in recovery['rows'] if row['split'] == 'val' and row['control'] == 'none'
                  and not row['originalGuard']['usable'] and row['fusedGuard']['usable'])
    occluded = next(row for row in recovery['rows'] if row['id'] == anchor['id'] and row['control'] == 'target_wrist_occlusion')
    raw_cases = []
    for row in (anchor, occluded):
        for stream in ('original', 'fused'):
            frames = [pair[stream] for pair in row['frames']]
            tracking = measure_raw_tracking(frames)
            item = {'id': f'raw:{row["id"]}:{row["control"]}:{stream}', 'sourceId': row['id'],
                    'split': 'development-person-6-tracking', 'origin': stream,
                    'classId': 0 if row['control'] == 'none' else -1, 'rawFrames': frames, 'expectedTracking': tracking}
            if tracking['usable']:
                x, _ = preprocess(frames)
                logits = logits_for(model, x[None])[0]
                item.update(x=x.tolist(), expectedLogits=logits.tolist(), expected=reference(logits, gate))
            raw_cases.append(item)
    temporary.replace(filename)
    config = {
        'modelVersion': 'signbridge-v7-research-' + selection['selected'], 'featureSchema': 'signbridge-landmarks-xy-v1',
        'T': 48, 'F': 224, 'labels': LABELS, 'backgroundIndex': 3, 'featureOrigin': selection['selected'].split('_')[0].replace('recovered', 'fused'),
        'input': {'name': 'landmarks', 'shape': [1, 48, 224], 'dtype': 'float32'},
        'output': {'name': 'logits'}, 'rejection': {'temperature': gate['temperature'], 'thresholds': [gate['threshold']] * 3},
        'protocolSha256': digest(LOCK), 'selectionSha256': digest(SELECTION), 'researchOnly': True,
    }
    config_file = BROWSER / 'model-config.json'
    write_json(config_file, config)
    write_json(BROWSER / 'parity-fixtures.json', {
        'modelSha256': digest(filename), 'configSha256': digest(config_file), 'featureSchema': config['featureSchema'],
        'T': 48, 'F': 224, 'atol': 1e-5, 'rtol': 1e-4, 'cases': cases, 'rawCases': raw_cases,
        'sourceValSha256': digest(DATA / 'val.npz'), 'rawSourceSha256': digest(raw_source),
    })
    audit = {'model_sha256': digest(filename), 'config_sha256': digest(config_file), 'parity_cases': len(cases),
             'raw_cases': len(raw_cases), 'cpu_onnx_checked_tensors': 2 * len(data['y']), 'max_cpu_logit_error': maximum,
             'browser_qa_required': True, 'production_approved': False}
    write_json(BROWSER / 'export-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
