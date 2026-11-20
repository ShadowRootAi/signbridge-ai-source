"""Package and verify the simple V8 veto for isolated browser replay."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from preprocess import preprocess
from v4_tracking import measure_raw_tracking
from v5_grouped_experiment import LABELS, ROOT, digest, write_json
from v7_export_core import reference
from v7_train_core import read, read_split
from v8_verifier import LOCK, OUT, verify_or_lock


def main():
    verify_or_lock('verify')
    diagnostic_path = OUT / 'product-verifier-diagnostic.json'
    report = read(diagnostic_path)
    if report['protocol_sha256'] != digest(LOCK):
        raise ValueError('Diagnostic results do not match the locked protocol')
    frozen = ROOT / 'web/public/models/model.onnx'
    current = read(ROOT / 'web/public/models/model_config.json')
    base_threshold = current['rejection']['confidenceThreshold']
    gates = report['methods']['confidence_veto']['gates']
    thresholds = [max(base_threshold, 1. - gate['cutoff'][0]) if gate['cutoff'] is not None else base_threshold for gate in gates]
    temperature = current['rejection']['temperature']
    config = {
        'modelVersion': 'signbridge-v8-v4-confidence-pilot', 'featureSchema': 'signbridge-landmarks-xy-v1',
        'T': 48, 'F': 224, 'labels': LABELS, 'backgroundIndex': 3, 'featureOrigin': 'fused',
        'input': current['input'], 'output': current['output'], 'rejection': {'temperature': temperature, 'thresholds': thresholds},
        'protocolSha256': digest(LOCK), 'diagnosticSha256': digest(diagnostic_path), 'researchOnly': True,
    }
    session = ort.InferenceSession(str(frozen), providers=['CPUExecutionProvider'])

    def expected(x):
        logits = session.run(['logits'], {'landmarks': x[None]})[0][0]
        prediction = int(logits[:3].argmax())
        gate = {'temperature': temperature, 'threshold': thresholds[prediction]}
        return {'expectedLogits': logits.tolist(), 'expected': reference(logits, gate)}

    data = read_split('val')
    sources = {row['source']: row for row in report['cases']}
    cases = []
    for i, source in enumerate(data['source_id']):
        row = sources[source]
        x = data['x_fused'][i]
        result = expected(x)
        decision = bool(data['guard_fused'][i]) and result['expected']['accepted']
        if decision != row['accepted']['confidence_veto']:
            raise ValueError('Exported thresholds changed the frozen decisions')
        cases.append({'id': str(source), 'sourceId': str(source), 'split': 'development-person-6',
                      'origin': 'fused', 'classId': int(data['y'][i]), 'x': x.tolist(), **result})
    raw_source = ROOT / 'reports/v6/hand-recovery-results.json'
    recovery = read(raw_source)
    if not recovery['completed']:
        raise ValueError('Incomplete hand recovery results')
    anchor = next(row for row in recovery['rows'] if row['split'] == 'val' and row['control'] == 'none'
                  and not row['originalGuard']['usable'] and row['fusedGuard']['usable'])
    occluded = next(row for row in recovery['rows'] if row['id'] == anchor['id'] and row['control'] == 'target_wrist_occlusion')
    raw_cases = []
    for row in (anchor, occluded):
        for stream in ('original', 'fused'):
            frames = [pair[stream] for pair in row['frames']]
            guard = measure_raw_tracking(frames)
            item = {'id': f'{row["id"]}:{row["control"]}:{stream}', 'sourceId': row['id'],
                    'split': 'development-person-6-tracking', 'origin': stream,
                    'classId': 0 if row['control'] == 'none' else -1, 'rawFrames': frames, 'expectedTracking': guard}
            if guard['usable']:
                x, _ = preprocess(frames)
                item.update(x=x.tolist(), **expected(x))
            raw_cases.append(item)
    target = ROOT / 'artifacts/v8/browser'
    config_path = target / 'model-config.json'
    write_json(config_path, config)
    write_json(target / 'parity-fixtures.json', {
        'modelSha256': digest(frozen), 'configSha256': digest(config_path), 'featureSchema': config['featureSchema'],
        'T': 48, 'F': 224, 'atol': 1e-5, 'rtol': 1e-4, 'cases': cases, 'rawCases': raw_cases,
        'sourceValSha256': digest(ROOT / 'data/v7-recovery/val.npz'), 'rawSourceSha256': digest(raw_source),
    })
    print(f'Prepared {len(cases)} paired tensor fixtures and {len(raw_cases)} raw guard fixtures')


if __name__ == '__main__':
    main()
