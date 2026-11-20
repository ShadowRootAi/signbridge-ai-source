"""Train, calibrate, evaluate and export a small landmark-sequence classifier.

Run: python signbridge-poc/ml/train.py --features ...npz --output .../artifacts
No data downloads, partition changes, or webcam claims are performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from common import (QUALITY_FLOOR, classification_metrics, dataset_summary, decisions, fit_temperature,
                    load_dataset, softmax, tune_rejection)
from preprocess import FEATURE_SCHEMA


def fit_normalization(train_x: np.ndarray, feature_schema: str | None = None):
    """Fit only training data; keep binary presence flags bounded in their native scale."""
    mean = train_x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    raw_std = train_x.std(axis=(0, 1), dtype=np.float64)
    std = np.maximum(raw_std, 0.01).astype(np.float32)
    keep = (raw_std >= 1e-6).astype(np.float32)
    flag_indices = []
    if feature_schema == FEATURE_SCHEMA:
        flag_indices = list(range(train_x.shape[2] - 4, train_x.shape[2]))
        mean[flag_indices] = 0
        std[flag_indices] = 1
        keep[flag_indices] = 1
    info = {"fitted_on": "train", "std_floor": 0.01, "invariant_std_cutoff": 1e-6,
            "standardized_clip": [-8, 8], "native_scale_flag_indices": flag_indices,
            "ignored_training_invariant_indices": np.flatnonzero(keep == 0).tolist(),
            "train_examples": int(len(train_x))}
    return mean, std, keep, info


def normalize_numpy(x: np.ndarray, mean: np.ndarray, std: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.clip((x - mean[None, None, :]) / std[None, None, :], -8, 8) * keep[None, None, :]


class TemporalClassifier(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, outputs: int, channels: int = 48,
                 keep: np.ndarray | None = None):
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32)[None, None, :])
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32)[None, None, :])
        self.register_buffer("feature_keep", torch.as_tensor(np.ones_like(mean) if keep is None else keep,
                                                             dtype=torch.float32)[None, None, :])
        self.temporal = nn.Sequential(
            nn.Conv1d(len(mean), channels, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.15),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.15),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Linear(channels, outputs)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((landmarks - self.feature_mean) / self.feature_std, -8, 8) * self.feature_keep
        features = self.temporal(normalized.transpose(1, 2)).mean(dim=2)
        return self.classifier(features)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(x)).numpy()


def train_model(data, args):
    train = data.mask("train")
    val = data.mask("val")
    k = len(data.classes)
    background = bool(np.any(data.y[train] == -1))
    outputs = k + int(background)
    # All fitted feature statistics come exclusively from training examples.
    mean, std, keep, normalization = fit_normalization(data.x[train], data.metadata.get("featureSchema"))
    model = TemporalClassifier(mean, std, outputs, args.channels, keep)
    train_y = np.where(data.y[train] < 0, k, data.y[train])
    train_x = torch.from_numpy(data.x[train])
    train_targets = torch.from_numpy(train_y)
    val_scored = val if background else val & (data.y >= 0)
    val_x = torch.from_numpy(data.x[val_scored])
    val_targets = torch.from_numpy(np.where(data.y[val_scored] < 0, k, data.y[val_scored]))
    counts = np.bincount(train_y, minlength=outputs)
    weights = len(train_y) / (outputs * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        loss_sum = 0.0
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[batch]), train_targets[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(val_x), val_targets))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_x), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch, "selection_metric": "validation_cross_entropy",
                   "training_seconds": time.perf_counter() - started,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "background_output": background, "normalization": normalization}, mean, std, keep


def evaluate_logits(data, logits: np.ndarray, background: bool, args) -> tuple[dict, dict, np.ndarray]:
    k = len(data.classes)
    val = data.mask("val")
    calibration = val if background else val & (data.y >= 0)
    targets = np.where(data.y[calibration] < 0, k, data.y[calibration])
    temperature = fit_temperature(logits[calibration], targets)
    probs = softmax(logits, temperature)
    rejection = tune_rejection(probs[val], data.y[val], data.quality[val], k,
                               args.target_far, args.target_accepted_accuracy)
    rejection["temperature"] = temperature
    rejection.update({"calibrated": True, "validationOnly": True,
                      "calibrationCount": int(val.sum()),
                      "calibration_provenance": {"split": "val", "threshold_examples": int(val.sum()),
                                                 "temperature_examples": int(calibration.sum()),
                                                 "known_examples": int((val & (data.y >= 0)).sum()),
                                                 "unknown_examples": int((val & (data.y == -1)).sum()),
                                                 "signers": sorted(set(data.signer[val].tolist())),
                                                 "method": "validation NLL temperature grid; validation-only confidence/quality threshold search"}})
    predicted, accepted, _ = decisions(probs, data.quality, k, rejection["confidence_threshold"], rejection["min_quality"])
    report = {}
    for split in ("train", "val", "test"):
        mask = data.mask(split)
        metric = classification_metrics(data.y[mask], predicted[mask], accepted[mask], k)
        metric["per_signer"] = {}
        for signer in sorted(set(data.signer[mask].tolist())):
            signer_mask = mask & (data.signer == signer)
            metric["per_signer"][signer] = classification_metrics(data.y[signer_mask], predicted[signer_mask], accepted[signer_mask], k)
        report[split] = metric
    return report, rejection, probs


def centroid_logits(data, mean: np.ndarray, std: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, bool]:
    """Non-neural baseline using the same complete, time-ordered input sequences."""
    train = data.mask("train")
    background = bool(np.any(data.y[train] == -1))
    k = len(data.classes)
    normalized = normalize_numpy(data.x, mean, std, keep).reshape(len(data.y), -1)
    targets = np.where(data.y < 0, k, data.y)
    centroids = np.stack([normalized[train & (targets == i)].mean(axis=0) for i in range(k + int(background))])
    logits = -np.stack([np.square(normalized - c).mean(axis=1) for c in centroids], axis=1)
    return logits, background


def export_and_check(model, data, output: Path) -> dict:
    import onnx
    import onnxruntime as ort

    model.eval()
    sample = torch.from_numpy(data.x[:1])
    model_path = output / "model.onnx"
    torch.onnx.export(model, (sample,), str(model_path), input_names=["landmarks"], output_names=["logits"],
                      dynamo=True, verify=True, external_data=False, opset_version=18, verbose=False)
    onnx.checker.check_model(str(model_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    # Keep a small set from each split; parity checks do not tune model parameters.
    indices = np.asarray([
        int(candidates[0])
        for split in ("train", "val", "test")
        for label in list(range(len(data.classes))) + [-1]
        if len(candidates := np.flatnonzero(data.mask(split) & (data.y == label)))
    ], dtype=np.int64)
    fixture_x = data.x[indices]
    reference = predict(model, fixture_x)
    actual = np.concatenate([session.run(["logits"], {"landmarks": x[None]})[0] for x in fixture_x], axis=0)
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    if not np.array_equal(actual.argmax(axis=1), reference.argmax(axis=1)):
        raise AssertionError("ONNX and PyTorch predicted different classes in parity fixtures")
    np.savez_compressed(output / "parity-fixtures.npz", x=fixture_x, pytorch_logits=reference, onnx_logits=actual,
                        source_id=data.source_id[indices], split=data.split[indices])
    (output / "parity-fixtures.json").write_text(json.dumps({
        "shape": list(fixture_x.shape), "x": fixture_x.tolist(),
        "pytorch_logits": reference.tolist(), "onnx_logits": actual.tolist(),
        "source_id": data.source_id[indices].tolist(), "split": data.split[indices].tolist(),
        "atol": 1e-5, "rtol": 1e-4,
    }, ensure_ascii=False), encoding="utf-8")
    # Exclude warmup. These numbers measure the classifier only, not camera/CV latency.
    for _ in range(5):
        session.run(None, {"landmarks": data.x[:1]})
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, {"landmarks": data.x[:1]})
        timings.append((time.perf_counter() - start) * 1000)
    return {"checked_examples": len(indices), "max_abs_error": float(np.abs(actual - reference).max()),
            "class_predictions_identical": True, "atol": 1e-5, "rtol": 1e-4,
            "onnx_bytes": model_path.stat().st_size,
            "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "python_onnx_cpu_classifier_ms": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)), "runs": len(timings)},
            "browser_parity": "not_yet_tested", "live_webcam_accuracy": "not_tested"}


def write_predictions(output: Path, data, probs: np.ndarray, rejection: dict) -> None:
    k = len(data.classes)
    predicted, accepted, confidence = decisions(probs, data.quality, k, rejection["confidence_threshold"], rejection["min_quality"])
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "signer", "split", "true_y", "raw_label", "top_known_y", "top_known_label", "accepted", "confidence", "quality"])
        for i in range(len(data.y)):
            writer.writerow([data.source_id[i], data.signer[i], data.split[i], int(data.y[i]), data.raw_label[i],
                             int(predicted[i]), data.classes[predicted[i]], bool(accepted[i]), float(confidence[i]), float(data.quality[i])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.85)
    parser.add_argument("--model-version", default="signbridge-poc-tcn-v1")
    parser.add_argument("--evaluation-context", default="initial_signer_heldout_evaluation",
                        help="Reporting provenance only; never changes training or rejection")
    args = parser.parse_args()
    if not 0 <= args.target_far <= 1 or not 0 <= args.target_accepted_accuracy <= 1:
        parser.error("rejection targets must be within [0,1]")
    seed_everything(args.seed)
    data = load_dataset(args.features)
    args.output.mkdir(parents=True, exist_ok=True)
    model, training, mean, std, keep = train_model(data, args)
    logits = predict(model, data.x)
    metrics, rejection, probs = evaluate_logits(data, logits, training["background_output"], args)
    baseline_logits, baseline_background = centroid_logits(data, mean, std, keep)
    baseline_metrics, baseline_rejection, _ = evaluate_logits(data, baseline_logits, baseline_background, args)
    parity = export_and_check(model, data, args.output)
    config = {
        "modelVersion": args.model_version,
        "evaluationContext": args.evaluation_context,
        "featureSchema": data.metadata.get("featureSchema", "unspecified"),
        "T": data.x.shape[1], "F": data.x.shape[2],
        "input": {"name": "landmarks", "dtype": "float32", "shape": [1, data.x.shape[1], data.x.shape[2]]},
        "output": {"name": "logits", "labels": data.classes + (["__background__"] if training["background_output"] else [])},
        "labels": data.classes,
        "backgroundIndex": len(data.classes) if training["background_output"] else None,
        "normalization": {**training["normalization"], "embeddedInOnnx": True,
                          "inputRule": "supply unstandardized preprocessed features"},
        "rejection": {"temperature": rejection["temperature"], "confidenceThreshold": rejection["confidence_threshold"],
                      "minQuality": rejection["min_quality"], "rejectBackground": True,
                      "qualityFloor": QUALITY_FLOOR,
                      "calibrated": rejection["calibrated"], "validationOnly": rejection["validationOnly"],
                      "calibrationCount": rejection["calibrationCount"],
                      "provenance": rejection["calibration_provenance"],
                      "rule": "softmax(logits / temperature); reject background winner, low known confidence, or low quality"},
        "metadata": data.metadata,
    }
    report = {"modelVersion": args.model_version, "evaluationContext": args.evaluation_context,
              "seed": args.seed, "dataset": dataset_summary(data),
              "training": training, "metrics": metrics, "rejection": rejection,
              "baseline": {"method": "nearest centroid on normalized full temporal sequences", "metrics": baseline_metrics, "rejection": baseline_rejection},
              "export": parity,
              "preprocessing_parity": data.metadata.get("preprocessing_parity", {
                  "fixture_count": data.metadata.get("preprocessing_parity_examples", 0),
                  "status": "not_tested_no_raw_fixtures"}),
              "limitations": ["Small research-data proof of concept; not evidence of live-webcam accuracy.",
                              "Validation thresholds do not guarantee rejection of every unseen sign.",
                              "Closed-set metrics exclude unknown examples and ignore the rejection gate.",
                              "Report accepted accuracy together with coverage and unknown false acceptance.",
                              "Python classifier latency excludes video decoding, tracking, sampling and UI."]}
    for filename, value in (("config.json", config), ("classes.json", data.classes), ("report.json", report)):
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "config": config, "channels": args.channels}, args.output / "model.pt")
    write_predictions(args.output, data, probs, rejection)
    print(json.dumps({"artifacts": str(args.output.resolve()), "test": metrics["test"], "parity": parity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
