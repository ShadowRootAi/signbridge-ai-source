"""Export only the frozen learned metric evidence, not the full recognizer."""
from pathlib import Path
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from train_isolated_cross_signer_metric import ARTIFACT, REPORT, CACHE, MetricNetwork, read, source_hashes
from evaluate_isolated_balanced_prototypes import load_sources
from evaluate_isolated_exemplar_retrieval import annotations, open_source, sha256, mix, probabilities, write_json


class MetricEvidence(nn.Module):
    def __init__(self, state):
        super().__init__()
        for key, value in state['normalizer'].items():
            self.register_buffer(key, value)
        self.network = MetricNetwork(state['input_size'], len(state['labels']))
        self.network.load_state_dict(state['state_dict'])

    def forward(self, sequence):
        pose = ((sequence - self.location) / self.scale).clamp(-8, 8).flatten(1)
        delta = torch.cat((sequence[:, :1] * 0., sequence[:, 1:] - sequence[:, :-1]), dim=1)
        velocity = (delta / self.motion_scale).clamp(-8, 8).flatten(1)
        descriptor = F.normalize(torch.cat((pose, .6 * velocity), dim=1), dim=1)
        return self.network(descriptor)[0]


def main():
    torch.set_num_threads(2)
    output = ARTIFACT / 'metric-evidence.onnx'
    if output.exists():
        raise ValueError('Frozen export already exists')
    frozen = read(REPORT / 'freeze.json')
    checkpoint = ARTIFACT / 'model.pt'
    if sha256(checkpoint) != frozen['checkpoint_sha256'] or source_hashes() != frozen['source_hashes']:
        raise ValueError('Frozen experiment identity changed')
    original, retrieval, bank = load_sources()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['labels'] != bank['labels'].tolist():
        raise ValueError('Export class order changed')
    module = MetricEvidence(state).eval()
    official = annotations(CACHE / 'gsl_metadata.zip')
    sequences, ids, references, signers = open_source(CACHE / 'official-train-prototypes-poses.npz', official['train'], role='train')
    mask = signers == 4
    sequences, ids, references = sequences[mask], ids[mask], references[mask]
    with np.load(REPORT / 'development-scores.npz', allow_pickle=False) as saved:
        if not np.array_equal(ids, saved['ids']) or not np.array_equal(references, saved['glosses']):
            raise ValueError('Parity population differs from development snapshot')
        expected = saved['metric_cosines']
        anchor = saved['anchor_logits']
    with torch.no_grad():
        eager = torch.cat([module(torch.from_numpy(chunk)) for chunk in np.array_split(sequences, 9)]).numpy()
    np.testing.assert_allclose(eager, expected, atol=2e-6, rtol=2e-5)
    torch.onnx.export(module, torch.from_numpy(sequences[:1]), output, input_names=['sequence'],
                      output_names=['metric_cosines'], dynamic_axes={'sequence': {0: 'batch'}, 'metric_cosines': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    actual = np.concatenate([runtime.run(['metric_cosines'], {'sequence': chunk})[0]
                             for chunk in np.array_split(sequences, 9)])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    weight = frozen['selected']['weight']
    expected_probability = probabilities(mix(anchor, expected, weight))
    actual_probability = probabilities(mix(anchor, actual, weight))
    np.testing.assert_allclose(actual_probability, expected_probability, atol=3e-6, rtol=2e-5)
    policy = frozen['selected']['policy']
    def decisions(probability):
        order = np.argsort(-probability, axis=1, kind='stable')
        values = np.take_along_axis(probability, order[:, :2], axis=1)
        accepted = (values[:, 0] >= policy['minimum_similarity']) & (values[:, 0] - values[:, 1] >= policy['minimum_margin'])
        return order[:, 0], accepted
    for first, second in zip(decisions(actual_probability), decisions(expected_probability), strict=True):
        np.testing.assert_array_equal(first, second)
    config = {'schema': 'isolated-cross-signer-metric-evidence-v1', 'research_only': True,
              'input': {'name': 'sequence', 'dtype': 'float32', 'shape': ['batch', 16, 189],
                        'meaning': '16 endpoint-inclusive linear time samples of224-feature columns0:186 plus220:223'},
              'output': {'name': 'metric_cosines', 'shape': ['batch', len(state['labels'])],
                         'meaning': 'Normalized learned embedding dot normalized trained class proxies; not probability'},
              'labels': state['labels'], 'model_sha256': sha256(output), 'checkpoint_sha256': sha256(checkpoint),
              'freeze_sha256': sha256(REPORT / 'freeze.json'), 'mixture_weight': weight, 'policy': policy,
              'normalization_embedded': 'Training-only per-channel location/scale and motion scale,clip8,velocityweight.6,unitL2',
              'remaining_adapter': ['Raw MediaPipe75 XY to native224feature math including848/480 aspect and zero face channels',
                                    'Exact16-step linear sampling including linear presence masks',
                                    'Existing local CNN32×224 branch and source-only exemplar/centroid fusion',
                                    'Class-aligned addition of normalized metric cosine scores to existing fusion logits',
                                    'Softmax and frozen score/margin gate, plus separately verified production input safeguards'],
              'limitations': ['This ONNX is only one evidence branch, not standalone end-to-end sign recognition.',
                              'Full fusion needs the existing compressed212MB training exemplar bank.',
                              'CPU ONNX tested; browser/WASM and RGB extraction-domain parity not established.']}
    write_json(ARTIFACT / 'metric-evidence-config.json', config)
    fixtures = []
    for i in np.unique(np.linspace(0, len(ids) - 1, 12, dtype=int)):
        fixtures.append({'id': str(ids[i]), 'reference': str(references[i]), 'sequence': sequences[i].tolist(),
                         'metric_cosines': expected[i].tolist(), 'anchor_logits': anchor[i].tolist(),
                         'fused_probabilities': expected_probability[i].tolist()})
    write_json(ARTIFACT / 'metric-evidence-fixtures.json', {'cases': fixtures, 'atol': 3e-6})
    report = {'passed': True, 'cases': len(ids), 'scope': 'All1084 development clips; no fresh final evaluation',
              'onnx_bytes': output.stat().st_size, 'onnx_sha256': sha256(output), 'checkpoint_sha256': sha256(checkpoint),
              'export_source_sha256': sha256(Path(__file__)),
              'maximum_absolute_error': {'torch_vs_numpy_descriptor_path': float(np.abs(eager - expected).max()),
                                         'onnx_vs_frozen_torch_cosines': float(np.abs(actual - expected).max()),
                                         'resulting_fused_probability': float(np.abs(actual_probability - expected_probability).max())},
              'exact_winners_and_acceptance': True, 'fixtures': len(fixtures),
              'remaining_adapter': config['remaining_adapter'], 'limitations': config['limitations']}
    write_json(REPORT / 'onnx-parity.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
