"""Train and freeze an open-set sign recognizer before evaluating fresh clips."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import classification_metrics, decisions, fit_temperature, softmax, tune_rejection
from evaluate_recognition_scale_pilot import active_classes, arrays, digest, reference, text_key, write_json
from train import TemporalClassifier, predict, seed_everything, train_model


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'artifacts/recognition-scale-pilot'
PRIOR_MANIFEST = ROOT / '.runtime-tmp/recognition-scale-pilot/manifest.json'
MIN_TRACKING = .6
HANDS = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
COORDS = tuple(axis for joint in HANDS for axis in (2 * joint, 2 * joint + 1))


def load_protocol(manifest, previous):
    payload = json.loads(manifest.read_bytes())
    if not payload['complete']:
        raise ValueError('Acquisition has not completed')
    samples = payload['samples']
    for key in ('id', 'sha256', 'original_sha256'):
        if len({row[key] for row in samples}) != len(samples):
            raise ValueError(f'Duplicate recording ({key})')
    prior = json.loads(previous.read_bytes())['samples']
    by_role = {role: [row for row in samples if row['role'] == role] for role in ('train', 'validation', 'test', 'diagnostic')}
    if sum(map(len, by_role.values())) != len(samples) or any(not by_role[role] for role in ('train', 'validation', 'test')):
        raise ValueError('Missing or unknown split')
    used = set()
    for rows in (by_role['train'], by_role['validation'], by_role['test'] + by_role['diagnostic']):
        current = {row['signer'] for row in rows}
        if used & current:
            raise ValueError('Signer overlaps fit, calibration or held data')
        used |= current
    known = {role: {row['label'] for row in rows if row['is_known']} for role, rows in by_role.items()}
    if len(known['train']) < 2 or known['validation'] != known['train'] or not known['test'] or known['test'] - known['train']:
        raise ValueError('Missing or invalid known labels')
    if known['diagnostic'] - known['train']:
        raise ValueError('Diagnostic target absent from training plan')
    used = set(map(text_key, known['train']))
    for role in ('train', 'validation', 'test'):
        keys = {text_key(reference(row)) for row in by_role[role] if not row['is_known']}
        if not keys or used & keys:
            raise ValueError(f'Overlapping or missing OTHER texts in {role}')
        used |= keys
    for key in ('id', 'sha256', 'original_sha256'):
        if {row[key] for row in prior} & {row[key] for row in by_role['test']}:
            raise ValueError(f'Prior clip entered the fresh test ({key})')
    return samples, sorted(known['train'])


def landmark_descriptor(x):
    """Keep handshape, position and ordered changes, with missing landmarks masked."""
    features = []
    pose = x[:, :, 222] >= .75
    for hand in range(2):
        present = x[:, :, 220 + hand] >= .75
        for offset, visible in ((0, present & pose), (42, present)):
            values = x[:, :, [hand * 84 + offset + column for column in COORDS]].reshape(-1, 4, 12, len(COORDS))
            mask = visible.reshape(-1, 4, 12, 1).to(x.dtype)
            count = mask.sum(dim=2)
            time = (values * mask).sum(dim=2) / count.clamp_min(1)
            delta = (time[:, 1:] - time[:, :-1]) * ((count[:, 1:] > 0) & (count[:, :-1] > 0))
            features.append(F.normalize(time.flatten(1), dim=1))
            features.append(.8 * F.normalize(delta.flatten(1), dim=1))
    return F.normalize(torch.cat(features, dim=1), dim=1)


class MotionPrototypes(nn.Module):
    def __init__(self, centers, unknown):
        super().__init__()
        self.register_buffer('centers', centers)
        self.register_buffer('unknown', unknown)

    def forward(self, x):
        z = landmark_descriptor(x)
        flat = self.centers.flatten(0, 1)
        known = (z @ flat.T).reshape(-1, self.centers.shape[0], self.centers.shape[1]).mean(dim=2)
        other = (z @ self.unknown.T).topk(k=2, dim=1).values.mean(dim=1, keepdim=True)
        return 12 * torch.cat([known, other], dim=1)


def fit_prototypes(data, good):
    z = landmark_descriptor(torch.from_numpy(data.x[good]))
    y, signers = data.y[good], data.signer[good]
    centers = torch.stack([torch.stack([F.normalize(z[(y == i) & (signers == signer)].mean(dim=0), dim=0)
                                       for signer in sorted(set(data.signer[good & (data.y == i)]))])
                           for i in range(len(data.classes))])
    return MotionPrototypes(centers, z[y < 0])


class Combined(nn.Module):
    def __init__(self, temporal, prototype, weight):
        super().__init__()
        self.temporal, self.prototype, self.weight = temporal, prototype, weight

    def forward(self, x):
        visual = self.temporal(x)
        motion = self.prototype(x)
        combined = (1 - self.weight) * visual + self.weight * motion
        return combined, visual, motion


def guarded_quality(quality, logits, count, independent):
    usable = quality >= MIN_TRACKING
    if independent is not None:
        expected = logits[:, :count].argmax(axis=1)
        for head in independent:
            usable &= expected == head[:, :count].argmax(axis=1)
    return np.where(usable, quality, 0)


def validation_policy(logits, y, quality, count, independent=None):
    q = guarded_quality(quality, logits, count, independent)
    mapped = np.where(y < 0, count, y)
    temp = fit_temperature(logits, mapped)
    policy = tune_rejection(softmax(logits, temp), y, q, count, target_far=0, target_accepted_accuracy=1)
    return {**policy, 'temperature': temp, 'motion_agreement': independent is not None}


def model_logits(temporal, prototype, x):
    return predict(temporal, x), predict(prototype, x)


def fit(manifest, features, directory, rows, planned):
    data = arrays(rows, planned, features, manifest, {'train', 'validation'})
    return fit_arrays(data, directory, {'manifest_sha256': digest(manifest)})


def fit_arrays(data, directory, provenance):
    planned = data.classes
    usable = data.mask('train') & (data.quality >= MIN_TRACKING)
    fit_signers = sorted(set(data.signer[usable & (data.y >= 0)]))
    active = [label for label in planned if set(data.signer[usable & (data.raw_label == label)]) == set(fit_signers)]
    if len(fit_signers) < 2 or len(active) < 2:
        raise ValueError('Insufficient independent training recordings')
    data = active_classes(data, active)
    good = data.mask('train') & (data.quality >= MIN_TRACKING)
    if len(set(data.raw_label[good & (data.y < 0)])) < 8:
        raise ValueError('Insufficient usable varied OTHER words')
    data.split[data.mask('train') & ~good] = 'unusable'
    val = data.mask('val')
    count = len(active)
    seed_everything(42)
    args = SimpleNamespace(seed=42, channels=48, learning_rate=.001, epochs=90, patience=16, batch_size=8)
    temporal, training, _, _, _ = train_model(data, args)
    prototype = fit_prototypes(data, good)
    cnn, motion = model_logits(temporal, prototype, data.x[val])
    candidates = [('temporal_other', 0., False), ('motion_prototypes', 1., False),
                  ('agreement_25', .25, True), ('agreement_50', .5, True), ('agreement_75', .75, True)]
    policies = {}
    for name, weight, agree in candidates:
        logits = (1 - weight) * cnn + weight * motion
        policies[name] = {**validation_policy(logits, data.y[val], data.quality[val], count, (cnn, motion) if agree else None),
                          'weight': weight}
    selected = max(policies, key=lambda name: (policies[name]['validation_metrics']['correct_known_acceptance_rate'],
                                               policies[name]['validation_metrics']['closed_set_macro_f1']))
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({'temporal': temporal.state_dict(), 'prototype': prototype.state_dict()}, directory / 'model.pt')
    # Preserve the same weights and tensor interface for direct ONNX replay.
    module = Combined(temporal, prototype, policies[selected]['weight']).eval()
    torch.onnx.export(module, (torch.from_numpy(data.x[val][:1]),), str(directory / 'model.onnx'),
                      input_names=['landmarks'], output_names=['logits', 'temporal', 'motion'],
                      dynamic_axes={'landmarks': {0: 'batch'}, **{name: {0: 'batch'} for name in ('logits', 'temporal', 'motion')}},
                      opset_version=17, dynamo=False)
    state = {**provenance, 'extractor_sha256': data.metadata['extractor_sha256'],
             'classes': active, 'excluded_by_fit_coverage': sorted(set(planned) - set(active)),
             'model_sha256': {name: digest(directory / name) for name in ('model.pt', 'model.onnx')},
             'fit_usable': int(good.sum()), 'fit_other_usable': int((good & (data.y < 0)).sum()),
             'fit_signers': fit_signers, 'calibration_signers': sorted(set(data.signer[val])),
             'calibration_clips': int(val.sum()), 'selected': selected, 'policies': policies, 'training': training}
    write_json(directory / 'freeze.json', state)
    print(json.dumps({'classes': active, 'selected': selected, 'fit_clips': state['fit_usable'],
                      'fit_other': state['fit_other_usable'], 'validation': policies[selected]['validation_metrics']}, ensure_ascii=False))


def restore(directory):
    state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
    buffers = state['temporal']
    temporal = TemporalClassifier(buffers['feature_mean'].numpy().ravel(), buffers['feature_std'].numpy().ravel(),
                                  buffers['classifier.bias'].numel(), keep=buffers['feature_keep'].numpy().ravel())
    temporal.load_state_dict(buffers)
    return temporal, MotionPrototypes(state['prototype']['centers'], state['prototype']['unknown'])


def summarize(data, logits, policy, independent=None):
    count = len(data.classes)
    quality = guarded_quality(data.quality, logits, count, independent)
    guess, accepted, confidences = decisions(softmax(logits, policy['temperature']), quality, count,
                                            policy['confidence_threshold'], policy['min_quality'])
    correct = guess == data.y
    known = data.y >= 0
    metrics = classification_metrics(data.y, guess, accepted, count)
    present_classes = np.unique(data.y[known])
    if len(present_classes):
        metrics['observed_known_class_macro_f1'] = float(np.asarray(metrics['per_class_f1'])[present_classes].mean())
    return {'metrics': metrics,
            'counts': {'correct_top1': int((correct & known).sum()), 'correct_accepted': int((correct & accepted).sum()),
                       'wrong_known_accepted': int((known & ~correct & accepted).sum()),
                       'unknown_accepted': int((~known & accepted).sum())},
            'rows': [{'id': str(identity), 'signer': str(signer), 'reference': str(ref), 'known': bool(is_known),
                      'prediction': data.classes[index], 'accepted': bool(take), 'score': float(confidence)}
                     for identity, signer, ref, is_known, index, take, confidence in
                     zip(data.source_id, data.signer, data.raw_label, known, guess, accepted, confidences, strict=True)]}


def evaluate_arms(data, temporal, prototype, frozen):
    a, b = model_logits(temporal, prototype, data.x)
    return {name: summarize(data, (1 - policy['weight']) * a + policy['weight'] * b, policy,
                            (a, b) if policy['motion_agreement'] else None) for name, policy in frozen['policies'].items()}


def compare_v1(full, arms, v1_dir):
    v1 = json.loads((v1_dir / 'freeze.json').read_bytes())
    if digest(v1_dir / 'temporal.pt') != v1['model_sha256']['temporal.pt']:
        raise ValueError('V1 checkpoint hash changed')
    old_labels = v1['classes']
    # Keep the complete V1 roster at inference while restricting the footage to common known labels.
    common = sorted(set(full.classes) & set(old_labels))
    baseline_data = active_classes(full, common)
    baseline_data = active_classes(baseline_data, old_labels)
    with np.load(v1_dir / 'baseline.npz', allow_pickle=False) as baseline:
        baseline_model = TemporalClassifier(baseline['mean'], baseline['std'], len(old_labels), keep=baseline['keep'])
    baseline_model.load_state_dict(torch.load(v1_dir / 'temporal.pt', map_location='cpu', weights_only=True))
    baseline = summarize(baseline_data, predict(baseline_model, baseline_data.x), v1['policies']['temporal'])
    ids = {row['id'] for row in baseline['rows']}
    comparison = {'v1': baseline['counts'], 'common_fresh_known': baseline['metrics']['known_n'],
                  'common_fresh_unknown': baseline['metrics']['unknown_n']}
    for name, result in arms.items():
        rows = [row for row in result['rows'] if row['id'] in ids]
        comparison[name] = {key: sum(predicate(row) for row in rows) for key, predicate in {
            'correct_top1': lambda row: row['known'] and row['prediction'] == row['reference'],
            'correct_accepted': lambda row: row['known'] and row['accepted'] and row['prediction'] == row['reference'],
            'wrong_known_accepted': lambda row: row['known'] and row['accepted'] and row['prediction'] != row['reference'],
            'unknown_accepted': lambda row: not row['known'] and row['accepted'],
        }.items()}
    return comparison


def evaluate(manifest, features, directory, report, fixture, rows, planned, previous):
    import onnxruntime

    frozen = json.loads((directory / 'freeze.json').read_bytes())
    if frozen['manifest_sha256'] != digest(manifest):
        raise ValueError('Cohort changed after training')
    for name, expected in frozen['model_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError('Model changed after training')
    held = arrays(rows, planned, features, manifest, {'test'})
    if held.metadata['extractor_sha256'] != frozen['extractor_sha256']:
        raise ValueError('Held feature extractor differs from training')
    # A proposed label without two usable fit signers is tested as unknown.
    held.y[(held.y >= 0) & ~np.isin(held.raw_label, frozen['classes'])] = -1
    held = active_classes(held, frozen['classes'])
    temporal, prototype = restore(directory)
    arms = evaluate_arms(held, temporal, prototype, frozen)
    selected = frozen['selected']
    cnn, motion = model_logits(temporal, prototype, held.x)
    policy = frozen['policies'][selected]
    ort = onnxruntime.InferenceSession(str(directory / 'model.onnx'), providers=['CPUExecutionProvider'])
    outputs = ort.run(None, {'landmarks': held.x})
    predicted_logits = (1 - policy['weight']) * cnn + policy['weight'] * motion
    parity = max(float(np.max(np.abs(x - y))) for x, y in zip(outputs, [predicted_logits, cnn, motion], strict=True))
    if parity > .0002:
        raise ValueError(f'ONNX logits differ from PyTorch: {parity}')
    old_signers = {row['signer'] for row in json.loads(previous.read_bytes())['samples']}
    report_data = {'scope': 'New Elementary23 MP4 identities, held signer IDs disjoint from fit and calibration. '
                            'Some held signer IDs were observed during the previous experiment. One-word whole-clip '
                            'transcripts are weak supervision. No human webcam trial.',
                   'frozen_sha256': digest(directory / 'freeze.json'), 'features_sha256': sha256(features.read_bytes()).hexdigest(),
                   'manifest_sha256': frozen['manifest_sha256'], 'classes': frozen['classes'], 'selected_before_test': selected,
                   'excluded_by_fit_coverage': frozen['excluded_by_fit_coverage'],
                   'fit_clips': frozen['fit_usable'], 'fit_other_clips': frozen['fit_other_usable'],
                   'calibration_clips': frozen['calibration_clips'], 'train_signers': frozen['fit_signers'],
                   'validation_signers': frozen['calibration_signers'], 'held_signers': sorted(set(held.signer)),
                   'held_signers_never_in_v1': sorted(set(held.signer) - old_signers),
                   'onnx_python_max_abs_logit_error': parity, 'held': arms}
    report_data['matched_fresh_comparison'] = compare_v1(held, arms, V1)
    from dataclasses import replace

    shuffled = replace(held, x=held.x[:, np.random.default_rng(7).permutation(held.x.shape[1])].copy())
    still = replace(held, x=np.repeat(held.x[:, 24:25], held.x.shape[1], axis=1), quality=np.zeros_like(held.quality))
    report_data['internal_motion_controls'] = {
        'scope': 'Controls inherit the original clip references; they are not new recordings.',
        'time_shuffled': evaluate_arms(shuffled, temporal, prototype, frozen)[selected]['counts'],
        'single_frame_repeated_with_no_motion_gate': evaluate_arms(still, temporal, prototype, frozen)[selected]['counts'],
    }
    write_json(report, report_data)
    expected_rows = arms[selected]['rows']
    write_json(fixture, {'classes': held.classes, 'policy': policy,
                         'rows': [{'id': expected['id'], 'landmarks': x.tolist(), 'logits': a.tolist(), 'temporal': b.tolist(),
                                   'motion': c.tolist(), 'quality': float(quality),
                                   'expectedLabel': expected['prediction'] if expected['accepted'] else None}
                                  for x, a, b, c, quality, expected in zip(held.x, predicted_logits, cnn, motion,
                                                                          held.quality, expected_rows, strict=True)]})
    print(json.dumps({'selected': selected, 'result': arms[selected]['counts'],
                      'comparison': report_data['matched_fresh_comparison'], 'onnx_max_abs_error': parity}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('features', type=Path)
    parser.add_argument('--prior-manifest', type=Path, default=PRIOR_MANIFEST)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/recognition-scale-v2')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/recognition-scale-v2/benchmark.json')
    parser.add_argument('--fixture', type=Path, default=ROOT / '.runtime-tmp/recognition-scale-v2/browser-fixture.json')
    args = parser.parse_args()
    rows, labels = load_protocol(args.manifest, args.prior_manifest)
    if args.phase == 'fit':
        fit(args.manifest, args.features, args.artifacts, rows, labels)
    else:
        evaluate(args.manifest, args.features, args.artifacts, args.report, args.fixture, rows, labels, args.prior_manifest)
