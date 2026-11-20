"""Data contracts and honest selective-classification metrics for the POC."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

QUALITY_FLOOR = 0.25  # Engineering validity policy, not an empirical accuracy guarantee.


@dataclass
class FeatureDataset:
    x: np.ndarray
    y: np.ndarray
    signer: np.ndarray
    source_id: np.ndarray
    split: np.ndarray
    classes: list[str]
    raw_label: np.ndarray
    quality: np.ndarray
    metadata: dict[str, Any]

    def mask(self, split: str) -> np.ndarray:
        return self.split == split


def load_dataset(path: str | Path) -> FeatureDataset:
    """Load without pickle and fail on leakage rather than silently repartitioning."""
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "signer", "source_id", "split", "classes"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ is missing required fields: {sorted(missing)}")
        x = np.asarray(data["x"], dtype=np.float32)
        y_raw = np.asarray(data["y"])
        if not np.issubdtype(y_raw.dtype, np.integer):
            raise ValueError("y must use an integer dtype")
        y = y_raw.astype(np.int64)
        n = len(y)
        if np.any(y == -1) and "raw_label" not in data:
            raise ValueError("raw_label is required to audit unknown-category leakage")
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        result = FeatureDataset(
            x=x, y=y,
            signer=np.asarray(data["signer"]).astype(str),
            source_id=np.asarray(data["source_id"]).astype(str),
            split=np.asarray(data["split"]).astype(str),
            classes=np.asarray(data["classes"]).astype(str).tolist(),
            raw_label=np.asarray(data["raw_label"]).astype(str) if "raw_label" in data else y.astype(str),
            quality=np.asarray(data["quality"], dtype=np.float32) if "quality" in data else np.ones(n, dtype=np.float32),
            metadata=metadata,
        )
    validate_dataset(result)
    return result


def validate_dataset(data: FeatureDataset) -> None:
    n = len(data.y)
    if data.x.ndim != 3 or data.x.shape[0] != n or min(data.x.shape) == 0:
        raise ValueError("x must be nonempty [N,T,F] matching y")
    if not np.isfinite(data.x).all():
        raise ValueError("x contains NaN/Inf; missing landmarks need a mask and finite values")
    for name in ("y", "signer", "source_id", "split", "raw_label", "quality"):
        if getattr(data, name).shape != (n,):
            raise ValueError(f"{name} must have shape [N]")
    if len(data.classes) < 2 or len(set(data.classes)) != len(data.classes):
        raise ValueError("classes must contain at least two unique labels")
    if np.any((data.y < -1) | (data.y >= len(data.classes))):
        raise ValueError("y must be -1 (unknown) or a valid known class index")
    if not np.isfinite(data.quality).all() or np.any((data.quality < 0) | (data.quality > 1)):
        raise ValueError("quality must be finite and within [0,1]")
    if set(data.split.tolist()) != {"train", "val", "test"}:
        raise ValueError("split must contain exactly train, val, and test")
    for name in ("signer", "source_id", "raw_label"):
        if any(not value.strip() for value in getattr(data, name)):
            raise ValueError(f"{name} contains an empty identity")
    for split in ("train", "val", "test"):
        present = set(data.y[data.mask(split) & (data.y >= 0)].tolist())
        if present != set(range(len(data.classes))):
            raise ValueError(f"{split} does not contain every known class")
    # A signer/source may have many examples, but only within one split.
    for field in ("signer", "source_id"):
        identities = getattr(data, field)
        for value in np.unique(identities):
            splits = set(data.split[identities == value].tolist())
            if len(splits) > 1:
                raise ValueError(f"{field} leakage: {value!r} appears in {sorted(splits)}")
    # Unknown labels are semantic categories, not file names.
    for label in np.unique(data.raw_label[data.y == -1]):
        splits = set(data.split[(data.y == -1) & (data.raw_label == label)].tolist())
        if len(splits) > 1:
            raise ValueError(f"Unknown category leakage: {label!r} appears in {sorted(splits)}")
    for label in np.unique(data.raw_label):
        if len(set(data.y[data.raw_label == label].tolist())) > 1:
            raise ValueError(f"raw_label {label!r} maps to inconsistent targets")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def decisions(probs: np.ndarray, quality: np.ndarray, known_count: int,
              confidence_threshold: float, min_quality: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top known prediction plus an explicit accept/reject decision."""
    predicted = probs[:, :known_count].argmax(axis=1)
    confidence = probs[np.arange(len(probs)), predicted]
    background_wins = probs.argmax(axis=1) >= known_count
    accepted = (confidence >= confidence_threshold) & (quality >= max(QUALITY_FLOOR, min_quality)) & ~background_wins
    return predicted, accepted, confidence


def classification_metrics(y: np.ndarray, predicted: np.ndarray, accepted: np.ndarray,
                           known_count: int) -> dict[str, Any]:
    """Unknowns never enter closed-set accuracy; abstentions are always disclosed."""
    known = y >= 0
    unknown = ~known
    matrix = np.zeros((known_count, known_count), dtype=np.int64)
    for truth, prediction in zip(y[known], predicted[known]):
        matrix[int(truth), int(prediction)] += 1
    tp = np.diag(matrix).astype(float)
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    accepted_known = accepted & known
    correct = predicted == y
    ratio = lambda a, b: float(a / b) if b else None
    return {
        "n": int(len(y)), "known_n": int(known.sum()), "unknown_n": int(unknown.sum()),
        "closed_set_accuracy": ratio((correct & known).sum(), known.sum()),
        "closed_set_macro_f1": float(f1.mean()) if known.any() else None,
        "closed_set_confusion_matrix": matrix.tolist(),
        "per_class_f1": f1.tolist(),
        "accepted_known_n": int(accepted_known.sum()),
        "accepted_known_accuracy": ratio((correct & accepted_known).sum(), accepted_known.sum()),
        "known_coverage": ratio(accepted_known.sum(), known.sum()),
        "known_rejection_rate": ratio((known & ~accepted).sum(), known.sum()),
        "correct_known_acceptance_rate": ratio((correct & accepted_known).sum(), known.sum()),
        "unknown_false_acceptance_n": int((unknown & accepted).sum()),
        "unknown_false_acceptance_rate": ratio((unknown & accepted).sum(), unknown.sum()),
        "unknown_rejection_rate": ratio((unknown & ~accepted).sum(), unknown.sum()),
        "all_accepted_n": int(accepted.sum()),
        "all_accepted_accuracy": ratio((correct & accepted_known).sum(), accepted.sum()),
    }


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Small, fixed search on validation examples only, not an accuracy claim."""
    options = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    losses = [float(-np.log(np.clip(softmax(logits, t)[np.arange(len(targets)), targets], 1e-12, 1)).mean())
              for t in options]
    return float(options[int(np.argmin(losses))])


def tune_rejection(probs: np.ndarray, y: np.ndarray, quality: np.ndarray, known_count: int,
                   target_far: float = 0.05, target_accepted_accuracy: float = 0.85) -> dict[str, Any]:
    """Tune ONLY the supplied validation set; retain reject-all when constraints fail."""
    if not np.any(y >= 0):
        raise ValueError("Validation needs known examples")
    confidence = probs[:, :known_count].max(axis=1)
    thresholds = np.unique(np.concatenate(([0.0, 1.000001], confidence, confidence + 1e-7)))
    quality_thresholds = np.unique(np.concatenate(([QUALITY_FLOOR], quality[quality >= QUALITY_FLOOR], [1.000001])))
    best: tuple[Any, ...] | None = None
    chosen = None
    for threshold in thresholds:
        for q in quality_thresholds:
            predicted, accepted, _ = decisions(probs, quality, known_count, float(threshold), float(q))
            metric = classification_metrics(y, predicted, accepted, known_count)
            far = metric["unknown_false_acceptance_rate"]
            accuracy = metric["accepted_known_accuracy"]
            if far is not None and far > target_far + 1e-12:
                continue
            if accuracy is not None and accuracy + 1e-12 < target_accepted_accuracy:
                continue
            score = (metric["correct_known_acceptance_rate"], accuracy or 0.0,
                     -(far or 0.0), metric["known_coverage"], -float(q), -float(threshold))
            if best is None or score > best:
                best = score
                chosen = {"confidence_threshold": float(threshold), "min_quality": float(q),
                          "engineering_quality_floor": QUALITY_FLOOR,
                          "target_validation_unknown_far": target_far,
                          "target_validation_accepted_known_accuracy": target_accepted_accuracy,
                          "validation_metrics": metric,
                          "unknown_constraint_evaluated": bool(np.any(y == -1))}
    assert chosen is not None  # Reject-all is always feasible, with accuracy=null.
    return chosen


def dataset_summary(data: FeatureDataset) -> dict[str, Any]:
    return {
        "shape": list(data.x.shape), "classes": data.classes, "metadata": data.metadata,
        "splits": {split: {
            "n": int(data.mask(split).sum()),
            "signers": sorted(set(data.signer[data.mask(split)].tolist())),
            "known_counts": {label: int((data.mask(split) & (data.y == i)).sum()) for i, label in enumerate(data.classes)},
            "unknown_n": int((data.mask(split) & (data.y == -1)).sum()),
            "unknown_categories": sorted(set(data.raw_label[data.mask(split) & (data.y == -1)].tolist())),
        } for split in ("train", "val", "test")},
    }
