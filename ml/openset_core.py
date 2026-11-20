"""Float64 gate mathematics; shared contract with independent browser implementation."""
from __future__ import annotations

from collections import Counter
import numpy as np

from common import classification_metrics, softmax
from openset_protocol import PROTOCOL


def motion_metrics(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (48, 224) or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,48,224] float32-origin feature sequences")
    body_square = np.zeros(len(x)); local_square = np.zeros(len(x))
    body_pairs = np.zeros(len(x)); local_pairs = np.zeros(len(x))
    present = x[:, :, 220:224] >= 1 - 1e-6
    for hand, body_start, local_start in ((0, 0, 42), (1, 84, 126)):
        hand_pairs = present[:, :-1, hand] & present[:, 1:, hand]
        both = hand_pairs & present[:, :-1, 2] & present[:, 1:, 2]
        body_delta = np.diff(x[:, :, body_start:body_start + 42].reshape(len(x), 48, 21, 2), axis=1)
        local_delta = np.diff(x[:, :, local_start:local_start + 42].reshape(len(x), 48, 21, 2), axis=1)
        body_square += np.sum(body_delta ** 2 * both[:, :, None, None], axis=(1, 2, 3))
        local_square += np.sum(local_delta ** 2 * hand_pairs[:, :, None, None], axis=(1, 2, 3))
        body_pairs += both.sum(axis=1)
        local_pairs += hand_pairs.sum(axis=1)
    return {"motionBodyRms": np.sqrt(np.divide(body_square, body_pairs * 21, out=np.zeros(len(x)), where=body_pairs > 0)),
            "motionLocalRms": np.sqrt(np.divide(local_square, local_pairs * 21, out=np.zeros(len(x)), where=local_pairs > 0)),
            "validTrackedPairFraction": body_pairs / 94,
            "meanPresence": x[:, :, 220:224].mean(axis=1)}


def fit_prototypes(embeddings: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    known = y >= 0
    e = np.asarray(embeddings[known], dtype=np.float64)
    targets = y[known]
    mean, raw_std = e.mean(axis=0), e.std(axis=0, ddof=0)
    scale = np.maximum(raw_std, 0.05)
    active = raw_std >= 1e-6
    if not active.any():
        raise ValueError("Prototype candidate has no active embedding dimensions")
    z = (e - mean) / scale * active
    centroids, loo_distances = [], []
    for label in range(len(classes)):
        cluster = z[targets == label]
        if len(cluster) < 2:
            raise ValueError("Prototype fitting needs at least two training examples per class")
        center = cluster.mean(axis=0)
        centroids.append(center)
        leave_one_out = (cluster.sum(axis=0) - cluster) / (len(cluster) - 1)
        loo_distances.append(np.sqrt(np.sum((cluster - leave_one_out) ** 2, axis=1) / active.sum()).tolist())
    return {"classOrder": classes, "embeddingName": "embedding", "embeddingSize": e.shape[1],
            "mean": mean.tolist(), "scale": scale.tolist(), "active": active.astype(int).tolist(),
            "activeCount": int(active.sum()), "centroids": np.asarray(centroids).tolist(),
            "trainLeaveOneOutDistances": loo_distances,
            "distance": "rms_euclidean_active_dimensions", "ddof": 0, "stdFloor": 0.05,
            "inactiveStdCutoff": 1e-6, "firstIndexWinsTies": True}


def prototype_distances(embeddings: np.ndarray, prototype: dict) -> np.ndarray:
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != prototype["embeddingSize"] or not np.isfinite(e).all():
        raise ValueError("Invalid prototype embedding; must not produce an accepted prediction")
    active = np.asarray(prototype["active"], dtype=np.float64)
    if not active.sum():
        raise ValueError("No active embedding dimensions")
    z = (e - np.asarray(prototype["mean"])) / np.asarray(prototype["scale"]) * active
    centers = np.asarray(prototype["centroids"], dtype=np.float64)
    return np.sqrt(np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2) / active.sum())


def score_gate(logits: np.ndarray, quality: np.ndarray, motion: np.ndarray, config: dict,
               embeddings: np.ndarray | None = None) -> dict:
    k = len(config["labels"])
    if logits.ndim != 2 or logits.shape[1] != k + 1 or not np.isfinite(logits).all():
        raise ValueError("Invalid classifier logits")
    if quality.shape != (len(logits),) or motion.shape != (len(logits),) or not np.isfinite(quality).all() or not np.isfinite(motion).all():
        raise ValueError("Invalid quality/motion metadata")
    if np.any((quality < 0) | (quality > 1)) or np.any(motion < 0):
        raise ValueError("Out-of-range quality/motion metadata")
    gate = config["rejection"]
    probabilities = softmax(logits, gate["temperature"])
    predicted = probabilities[:, :k].argmax(axis=1)
    confidence = probabilities[np.arange(len(predicted)), predicted]
    reasons = {"quality": quality < max(0.25, gate["minQuality"]),
               "motion": motion < gate.get("minMotionBodyRms", 0.0),
               "confidence": confidence < gate["confidenceThreshold"],
               "background": probabilities.argmax(axis=1) >= k}
    result = {"predicted": predicted, "confidence": confidence, "probabilities": probabilities}
    prototype = config.get("prototypeGate")
    if prototype is not None:
        if embeddings is None:
            raise ValueError("Prototype gate requires embeddings")
        distances = prototype_distances(embeddings, prototype)
        nearest = distances.argmin(axis=1)
        predicted_distance = distances[np.arange(len(predicted)), predicted]
        reasons["prototypeDisagreement"] = nearest != predicted
        cutoffs = prototype.get("classDistanceCutoffs")
        reasons["prototypeDistance"] = (predicted_distance > np.asarray(cutoffs)[predicted]
                                          if cutoffs is not None else np.zeros(len(predicted), dtype=bool))
        result.update({"prototypeDistances": distances, "nearestPrototype": nearest,
                       "predictedPrototypeDistance": predicted_distance})
    result["reasons"] = reasons
    result["accepted"] = ~np.logical_or.reduce(list(reasons.values()))
    return result


def metrics(y: np.ndarray, scored: dict, classes: list[str], signer=None, raw_label=None) -> dict:
    predicted, accepted = scored["predicted"], scored["accepted"]
    report = classification_metrics(y, predicted, accepted, len(classes))
    report["gate_reasons_nonexclusive"] = {name: int(mask.sum()) for name, mask in scored["reasons"].items()}
    report["per_known_sign"] = {label: {"n": int((y == i).sum()),
                                         "top1_correct_n": int(((y == i) & (predicted == y)).sum()),
                                         "accepted_n": int(((y == i) & accepted).sum()),
                                         "accepted_correct_n": int(((y == i) & accepted & (predicted == y)).sum()),
                                         "rejected_n": int(((y == i) & ~accepted).sum()),
                                         "rejection_reasons_nonexclusive": {reason: int(((y == i) & mask).sum()) for reason, mask in scored["reasons"].items()}}
                                for i, label in enumerate(classes)}
    if signer is not None:
        report["per_signer"] = {str(s): classification_metrics(y[signer == s], predicted[signer == s], accepted[signer == s], len(classes))
                                for s in np.unique(signer)}
    if raw_label is not None:
        report["per_unknown_category"] = {}
        for label in np.unique(raw_label[y < 0]):
            mask = (y < 0) & (raw_label == label)
            n, false_n = int(mask.sum()), int((mask & accepted).sum())
            report["per_unknown_category"][str(label)] = {"n": n, "false_acceptance_n": false_n,
                                                           "false_acceptance_rate": false_n / n,
                                                           "accepted_as": dict(Counter(classes[int(p)] for p in predicted[mask & accepted]))}
    return report


def eligibility_score(report: dict):
    policy = PROTOCOL["selection"]
    accuracy = report["accepted_known_accuracy"]
    if report["known_coverage"] < policy["minimum_known_coverage"] or accuracy is None or accuracy < policy["minimum_accepted_known_accuracy"]:
        return None
    if report["unknown_false_acceptance_rate"] is None:
        raise ValueError("Validation must contain unsupported examples")
    return (-report["unknown_false_acceptance_rate"], report["correct_known_acceptance_rate"],
            accuracy, report["known_coverage"])
