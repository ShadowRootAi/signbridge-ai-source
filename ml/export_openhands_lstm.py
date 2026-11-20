"""Export and verify the second official OpenHands Greek checkpoint."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from openhands_gsl import LSTM_SHA256, load_openhands_gsl, prepare_pose


def main():
    cache = Path("research/cache/openhands-gsl")
    torch.set_num_threads(1)
    model, labels = load_openhands_gsl(cache)
    with np.load(cache / "official-val-poses.npz", allow_pickle=False) as poses:
        points, offsets = poses["keypoints"], poses["offsets"]
    lengths = np.diff(offsets)
    chosen = [np.flatnonzero(lengths >= target)[0] for target in (6, 16, 32, 48)]
    inputs = [prepare_pose(points[offsets[i]:offsets[i + 1]])[:target].unsqueeze(0)
              for i, target in zip(chosen, (6, 16, 32, 48), strict=True)]
    output = cache / "lstm.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (inputs[1],), output, input_names=["landmarks"], output_names=["logits"],
                          dynamic_axes={"landmarks": {1: "time"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    errors = []
    for value in inputs:
        with torch.inference_mode():
            expected = model(value).numpy()
        actual, = session.run(["logits"], {"landmarks": value.numpy()})
        assert actual.shape == (1, len(labels)) and actual.argmax() == expected.argmax()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
        errors.append(float(np.abs(actual - expected).max()))
    with output.open("rb") as handle:
        checksum = file_digest(handle, "sha256").hexdigest()
    report = {"sourceCheckpointSha256": LSTM_SHA256, "modelSha256": checksum,
              "bytes": output.stat().st_size, "inputShape": [1, "time", 54], "parityLengths": [v.shape[1] for v in inputs],
              "maxAbsLogitError": max(errors)}
    (Path("reports/pretrained-gsl") / "lstm-onnx-export.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

