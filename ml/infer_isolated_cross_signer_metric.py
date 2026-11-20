"""Offline research inference for full native MediaPipe75 landmarks.

Input: float32[T,75,3], normalized image xy, pose33,left21,right21;
native848x480 aspect ratio. This is not the reduced OpenHands29 tensor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_isolated_exemplar_retrieval import (
    sample_sequence, scores, mix, probabilities, sha256, write_json,
)
from evaluate_isolated_balanced_prototypes import load_sources
from evaluate_synthetic_ctc_transfer import frame_feature_array, verify_feature_math
from train_isolated_large_vocabulary import load_checkpoint, resample_features
from train_isolated_cross_signer_metric import (
    ARTIFACT, REPORT, CACHE, CLASSIFIER, MetricNetwork, convert, read, source_hashes,
)


class FrozenCrossSignerRecognizer:
    """Single-clip inference, sharing the exact frozen research feature math."""
    def __init__(self):
        self.freeze = read(REPORT / 'freeze.json')
        self.model_path = ARTIFACT / 'model.pt'
        if (sha256(self.model_path) != self.freeze['checkpoint_sha256']
                or source_hashes() != self.freeze['source_hashes']
                or sha256(REPORT / 'protocol.json') != self.freeze['protocol_sha256']):
            raise ValueError('Frozen trained metric/source changed')
        self.original, self.retrieval_freeze, self.bank = load_sources()
        self.classifier, labels, self.classifier_freeze = load_checkpoint('expanded')
        self.classifier.eval()
        lookup = {label: i for i, label in enumerate(labels)}
        self.columns = [lookup[label] for label in self.bank['labels']]
        state = torch.load(self.model_path, weights_only=True, map_location='cpu')
        if state['labels'] != self.bank['labels'].tolist():
            raise ValueError('Metric class order mismatch')
        self.metric = MetricNetwork(state['input_size'], len(state['labels']))
        self.metric.load_state_dict(state['state_dict'])
        self.metric.eval()
        self.norm = {key: value.numpy() for key, value in state['normalizer'].items()}
        self.model_version = 'isolated-local-cross-signer-metric-' + sha256(self.model_path)[:12]

    @torch.inference_mode()
    def predict(self, raw, *, return_arrays=False):
        started = time.perf_counter()
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[1:] != (75, 3) or len(raw) < 2 or not np.isfinite(raw).all():
            raise ValueError('Expected finite normalized native[T>=2,75,3]; OpenHands29 cannot recover missing hand joints')
        present = {key: np.any(raw[:, start:end, :2] != 0., axis=(1, 2))
                   for key, start, end in (('pose', 0, 33), ('left', 33, 54), ('right', 54, 75))}
        tracking = {key + '_presence': float(value.mean()) for key, value in present.items()}
        enough = bool((present['pose'] & (present['left'] | present['right'])).any())
        if not enough:
            return {'research_only': True, 'model_version': self.model_version,
                    'classifier_executed': False, 'accepted': False, 'label': None,
                    'reason': 'No frame has both pose and at least one hand', 'tracking': tracking,
                    'frames': len(raw), 'score': None, 'elapsed_ms': (time.perf_counter() - started) * 1000}
        frames = frame_feature_array(raw)
        verify_feature_math(raw, frames)
        classifier_input = resample_features(torch.from_numpy(frames))[None]
        logits = self.classifier(classifier_input).numpy()[:, self.columns].astype(np.float64)
        seq = sample_sequence(frames, 0, len(frames))[None]
        retrieval_method = self.retrieval_freeze['variants']['all_verified_labels']['selected']
        retrieval = scores(self.bank, seq)[0][retrieval_method]
        anchor = mix(logits, retrieval, self.original['weight'])
        metric = self.metric(torch.from_numpy(convert(seq, self.norm)))[0].numpy()
        values = probabilities(mix(anchor, metric, self.freeze['selected']['weight']))[0]
        ranks = np.argsort(-values, kind='stable')
        confidence, margin = float(values[ranks[0]]), float(values[ranks[0]] - values[ranks[1]])
        policy = self.freeze['selected']['policy']
        accepted = confidence >= policy['minimum_similarity'] and margin >= policy['minimum_margin']
        result = {'research_only': True, 'model_version': self.model_version, 'classifier_executed': True,
                  'accepted': bool(accepted), 'label': str(self.bank['labels'][ranks[0]]) if accepted else None,
                  'candidate': str(self.bank['labels'][ranks[0]]), 'score': confidence, 'margin': margin,
                  'score_is_probability_of_correctness': False, 'frames': len(raw), 'tracking': tracking,
                  'tracking_limit': 'Presence is a research diagnostic; no production completeness/motion guard was ported.',
                  'scope': 'Native848x480 isolated pose; not independently evaluated webcam/unknown-sign recognition',
                  'top5': [{'gloss': str(self.bank['labels'][i]), 'score': float(values[i])} for i in ranks[:5]],
                  'elapsed_ms': (time.perf_counter() - started) * 1000}
        if return_arrays:
            return result, {'logits': logits[0], 'metric_cosines': metric[0], 'probabilities': values}
        return result


def load_input(path, clip_id=None):
    if path.suffix.lower() == '.json':
        value = json.loads(path.read_text(encoding='utf-8'))
        return np.asarray(value['keypoints'] if isinstance(value, dict) else value, dtype=np.float32)
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as values:
            raw = values['keypoints']
            if 'offsets' in values.files:
                if not clip_id:
                    raise ValueError('--id is required for a multi-clip NPZ')
                matches = np.flatnonzero(values['ids'] == clip_id)
                if len(matches) != 1:
                    raise ValueError('Expected exactly one matching source ID')
                i = int(matches[0])
                offsets = values['offsets']
                return raw[offsets[i]:offsets[i + 1]]
            return raw
    raise ValueError('Use .npz, .npy, or .json native pose input')


def verify():
    started = time.perf_counter()
    recognizer = FrozenCrossSignerRecognizer()
    with np.load(REPORT / 'development-scores.npz', allow_pickle=False) as values:
        ids, refs, labels, metric, anchor = (values[key] for key in ('ids', 'glosses', 'labels', 'metric_cosines', 'anchor_logits'))
    with np.load(CLASSIFIER / 'expanded-dev-scores.npz', allow_pickle=False) as values:
        index = {str(i): n for n, i in enumerate(values['ids'])}
        columns = {str(label): n for n, label in enumerate(values['labels'])}
        cnn = values['logits'][:, [columns[str(label)] for label in labels]]
    source = CACHE / 'official-train-prototypes-poses.npz'
    with np.load(source, allow_pickle=False) as values:
        source_ids, offsets, poses = (values[key] for key in ('ids', 'offsets', 'keypoints'))
    source_index = {str(i): n for n, i in enumerate(source_ids)}
    expected = probabilities(mix(anchor, metric, recognizer.freeze['selected']['weight']))
    policy = recognizer.freeze['selected']['policy']
    selected = np.unique(np.linspace(0, len(ids) - 1, 12, dtype=int))
    errors = {'classifier_logits': 0., 'metric_cosines': 0., 'probabilities': 0.}
    rows = []
    for number in selected:
        identity = str(ids[number])
        i = source_index[identity]
        raw = poses[offsets[i]:offsets[i + 1]]
        prediction, details = recognizer.predict(raw, return_arrays=True)
        for key, a, b, tolerance in (
                ('classifier_logits', details['logits'], cnn[index[identity]], 1e-4),
                ('metric_cosines', details['metric_cosines'], metric[number], 2e-6),
                ('probabilities', details['probabilities'], expected[number], 2e-5)):
            errors[key] = max(errors[key], float(np.abs(a - b).max()))
            np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)
        order = np.argsort(-expected[number], kind='stable')
        confidence = expected[number, order[0]]
        accepted = confidence >= policy['minimum_similarity'] and confidence - expected[number, order[1]] >= policy['minimum_margin']
        if prediction['accepted'] != bool(accepted) or prediction['candidate'] != str(labels[order[0]]):
            raise ValueError('Single-clip outcome differs from frozen development batch')
        rows.append({'id': identity, 'reference': str(refs[number]), **prediction})
    invalid = recognizer.predict(np.zeros((8, 75, 3), dtype=np.float32))
    assert not invalid['accepted'] and not invalid['classifier_executed'] and invalid['score'] is None
    try:
        recognizer.predict(np.zeros((8, 29, 3), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError('Reduced29-joint input must fail closed')
    sample_id = str(ids[0])
    i = source_index[sample_id]
    example = ARTIFACT / 'example-native.npz'
    np.savez_compressed(example, keypoints=poses[offsets[i]:offsets[i + 1]], source_id=sample_id, reference=refs[0])
    sample_prediction = recognizer.predict(poses[offsets[i]:offsets[i + 1]])
    report = {'passed': True, 'native_cases': len(rows), 'maximum_absolute_error': errors,
              'scope': 'Real source pose inference vs cached batch computation; not ONNX/browser parity',
              'checkpoint_sha256': sha256(ARTIFACT / 'model.pt'), 'freeze_sha256': sha256(REPORT / 'freeze.json'),
              'source_native_sha256': sha256(source), 'inference_source_sha256': sha256(Path(__file__)),
              'example_path': example.as_posix(), 'example_sha256': sha256(example), 'example_id': sample_id,
              'example_prediction': sample_prediction, 'blank_rejected_before_model': True,
              'reduced29_joint_input_rejected': True, 'rows': rows,
              'elapsed_seconds': time.perf_counter() - started}
    write_json(REPORT / 'native-smoke.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('rows', 'example_prediction')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', type=Path)
    parser.add_argument('--id')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.verify:
        verify()
    else:
        if args.pose is None:
            parser.error('--pose is required without --verify')
        outcome = FrozenCrossSignerRecognizer().predict(load_input(args.pose, args.id))
        print(json.dumps(outcome[0] if isinstance(outcome, tuple) else outcome, ensure_ascii=False, indent=2))
