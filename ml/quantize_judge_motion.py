"""Quantize the research motion classifier's linear layers and compare on real RGB poses."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from common import softmax
from evaluate_openhands_scope_veto import session
from evaluate_recognition_scale_pilot import write_json
from train_judge_open_set import MODEL_DIR, ROOT, STEPS, digest, hybrid_accept, resize, summarize


OUTPUT = ROOT / 'artifacts/judge-ai-development/motion-int8-linear.onnx'
REPORT = ROOT / 'reports/judge-ai-development/quantization-native.json'
RGB = ROOT / '.runtime-tmp/judge-final-rgb/poses.jsonl'


def main():
    source = MODEL_DIR / 'motion.onnx'
    freeze_path = MODEL_DIR / 'freeze.json'
    state = json.loads(freeze_path.read_bytes())
    source_hash = digest(source)
    if source_hash != state['files']['motion.onnx']:
        raise ValueError('Reference graph bytes disagree with the current frozen candidate')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pending = OUTPUT.with_suffix('.pending.onnx')
    try:
        fp_model = onnx.shape_inference.infer_shapes(onnx.load(source, load_external_data=False))
        quantize_dynamic(fp_model, pending, op_types_to_quantize=['Gemm', 'MatMul'],
                         per_channel=True, weight_type=QuantType.QInt8, use_external_data_format=False)
        graph = onnx.load(pending, load_external_data=False)
        onnx.checker.check_model(graph)
        integer_ops = sorted({node.op_type for node in graph.graph.node if node.op_type in ('MatMulInteger', 'QLinearMatMul')})
        if not integer_ops or any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
            raise ValueError('The candidate graph has no executable integer linear operations or needs sidecar weights')
        graph_hash = digest(pending)
        rows = [json.loads(line) for line in RGB.read_text(encoding='utf-8').splitlines()]
        if not rows or len({row['id'] for row in rows}) != len(rows) or any(
                row['policySha256'] != state['baseline_policy_sha256'] or row['seekOffsetMs'] != 2 for row in rows):
            raise ValueError('RGB evidence differs from the frozen recognition pipeline')
        rows = [row for row in rows if row['pose'] is not None and row['base']['classifierExecuted']]
        sample = np.stack([resize(np.asarray(row['pose'], dtype=np.float32).reshape(2, row['frames'], 27), STEPS) for row in rows])
        def infer(path, sha):
            return session(path, sha).run(['logits'], {'landmarks': sample})[0]

        normal, quantized = infer(source, source_hash), infer(pending, graph_hash)
        if normal.shape != quantized.shape or not np.isfinite(quantized).all():
            raise ValueError('Quantization produced an incompatible result')
        normal_first, int8_first = normal.argmax(axis=1), quantized.argmax(axis=1)
        classes = state['classes']
        references = np.asarray([row['gloss'] for row in rows])
        winning = np.asarray([row['base']['topCandidates'][0]['gloss'] for row in rows])
        confidence = np.asarray([row['base']['confidence'] for row in rows])
        accepted = np.asarray([row['base']['accepted'] for row in rows])
        eligible = np.isin(winning, classes) & np.asarray([row['base']['rejectionCode'] != 'low_evidence' for row in rows])
        base = (winning, confidence, accepted, eligible)
        full_decisions = hybrid_accept(base, softmax(normal, state['temperature']), classes, state['hybrid'])
        int8_decisions = hybrid_accept(base, softmax(quantized, state['temperature']), classes, state['hybrid'])
        result = {'scope': 'Prepared landmarks from previously evaluated real MP4s, with the same frozen thresholds. '
                           'Only linear-layer weights and operations are quantized. Convolutions remain float.',
                  'source_sha256': source_hash, 'freeze_sha256': digest(freeze_path), 'rgb_sha256': digest(RGB),
                  'int8_sha256': graph_hash, 'integer_ops': integer_ops,
                  'float_bytes': source.stat().st_size, 'int8_bytes': pending.stat().st_size,
                  'tested_inputs': len(rows), 'head_top1_agreements': int((normal_first == int8_first).sum()),
                  'hybrid_decision_agreements': int((full_decisions == int8_decisions).sum()),
                  'float_hybrid': summarize(references, winning, full_decisions, classes),
                  'int8_hybrid': summarize(references, winning, int8_decisions, classes),
                  'max_abs_logit_difference': float(np.abs(normal - quantized).max())}
        pending.replace(OUTPUT)
        write_json(REPORT, result)
        print(json.dumps({key: result[key] for key in ('float_bytes', 'int8_bytes', 'tested_inputs',
                                                       'head_top1_agreements', 'hybrid_decision_agreements')}, ensure_ascii=False))
    finally:
        pending.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
