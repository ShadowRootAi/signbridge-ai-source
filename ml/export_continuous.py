"""Export one complete utterance at a time, preserving its variable time axis."""

import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort
import numpy as np
import torch

from train_continuous import ContinuousCTC, ctc_decode


class OneUtterance(torch.nn.Module):
    def __init__(self, trained):
        super().__init__()
        self.trained = trained

    def forward(self, frames):
        model = self.trained
        features = model.projection((frames - model.mean) / model.scale)
        encoded, _ = model.temporal(features)
        return model.classifier(encoded).log_softmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/continuous-v1/ctc.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/continuous-v1/ctc.json"))
    parser.add_argument("--features", type=Path, nargs="+", default=[Path("data/continuous-v1/features.jsonl")])
    parser.add_argument("--output", type=Path, default=Path("artifacts/continuous-v1/browser"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    labels = checkpoint["labels"]
    model = ContinuousCTC(len(labels), state["mean"], state["scale"]).eval()
    model.load_state_dict(state)
    fixed = OneUtterance(model).eval()
    evaluation = json.loads(args.report.read_text(encoding="utf8"))
    test = {entry["id"]: entry for entry in evaluation["test_predictions"]}
    features = {}
    for path in args.features:
        for line in path.read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            if row["id"] in test:
                if row["id"] in features:
                    raise ValueError(f'Duplicate test features: {row["id"]}')
                features[row["id"]] = torch.tensor(row["features"], dtype=torch.float32).unsqueeze(0)
    if features.keys() != test.keys():
        raise ValueError("Need the real extracted features for every held-out utterance")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "sequence.onnx"
    with torch.no_grad():
        torch.onnx.export(
            fixed, (next(iter(features.values())),), destination,
            input_names=["frames"], output_names=["log_probabilities"],
            dynamic_axes={"frames": {1: "time"}, "log_probabilities": {1: "time"}},
            opset_version=17, dynamo=False,
        )
    onnx.checker.check_model(destination)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    max_error = 0.0
    for identity, frames in features.items():
        with torch.no_grad():
            expected = model(frames, [frames.shape[1]])[0].numpy()
        actual = session.run(None, {"frames": frames.numpy()})[0][0]
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=2e-5)
        max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        decoded, confidence = ctc_decode(torch.from_numpy(actual), labels)
        expected_row = test[identity]
        if list(decoded) != expected_row["predicted_glosses"] or abs(confidence - expected_row["confidence"]) > 2e-5:
            raise ValueError(f"Sequence parity failed for {identity}")
    output = {
        "feature_schema": checkpoint["feature_schema"],
        "labels": labels,
        "confidence_threshold": checkpoint["confidence_threshold"],
        "training_greek_references": [
            {"glosses": glosses, "text": text}
            for glosses, text in checkpoint["training_greek_references"].items()
        ],
        "parity": {
            "pytorch_vs_onnx_cpu_clips": len(features),
            "distinct_lengths": sorted({frames.shape[1] for frames in features.values()}),
            "max_absolute_log_probability_error": max_error,
        },
    }
    (args.output / "sequence-config.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(output["parity"]))


if __name__ == "__main__":
    main()
