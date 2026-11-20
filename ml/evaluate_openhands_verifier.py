"""Compare frozen OpenHands confidence and training-pose verification gates."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from calibrate_openhands_gsl import MAX_ERROR_RATE, read_metadata
from openhands_gsl_stgcn import prepare_stgcn_pose


EMBEDDING_WIDTH = 256
MODEL_SHA256 = "8ace9360ed396ecd6df2670a9c34d1bab271d6384808f2972456355c577fbd79"


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_native_poses(path):
    with np.load(path, allow_pickle=False) as dataset:
        keypoints, offsets, ids, glosses = (dataset[key] for key in ("keypoints", "offsets", "ids", "glosses"))
    if (len(ids) != len(glosses) or len(set(ids)) != len(ids) or len(offsets) != len(ids) + 1
            or offsets[0] != 0 or offsets[-1] != len(keypoints) or np.any(np.diff(offsets) <= 0)):
        raise ValueError(f"Invalid pose archive index: {path.name}")
    return keypoints, offsets, ids, glosses


def make_session(model_path, export_path):
    if sha256(model_path) != MODEL_SHA256:
        raise ValueError("Unexpected source ST-GCN model bytes")
    model = onnx.load(model_path)
    head = model.graph.node[-1]
    if head.op_type != "Gemm" or list(head.output) != ["logits"]:
        raise ValueError("Expected the frozen ST-GCN classifier as final graph node")
    weights, bias = (numpy_helper.to_array(next(value for value in model.graph.initializer if value.name == name))
                     for name in head.input[1:])
    if weights.shape != (310, EMBEDDING_WIDTH) or bias.shape != (310,):
        raise ValueError("Unexpected ST-GCN classifier shape")
    embedding = "embedding"
    model.graph.node.append(helper.make_node("Identity", [head.input[0]], [embedding]))
    model.graph.output.append(helper.make_tensor_value_info(embedding, TensorProto.FLOAT, [1, EMBEDDING_WIDTH]))
    onnx.checker.check_model(model)
    model_bytes = model.SerializeToString()
    if export_path is not None:
        export_path.write_bytes(model_bytes)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
    return session, bias.astype(np.float64), np.linalg.norm(weights.astype(np.float64), axis=1)


def infer(session, clips):
    logits, embeddings = [], []
    for points in clips:
        scores, vector = session.run(None, {"landmarks": prepare_stgcn_pose(points).numpy()})
        logits.append(scores[0])
        embeddings.append(vector[0])
    scores, vectors = np.stack(logits), np.stack(embeddings)
    if not np.isfinite(scores).all() or not np.isfinite(vectors).all():
        raise ValueError("Model returned non-finite verifier inputs")
    return scores, vectors


def infer_native(session, path, cache):
    keypoints, offsets, ids, glosses = read_native_poses(path)
    cache_path = cache / f"{path.stem}-verifier-scores.npz"
    source_hash = sha256(path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as dataset:
            if dataset["sourceHash"].item() == source_hash and dataset["modelHash"].item() == MODEL_SHA256:
                return ids, glosses, dataset["logits"], dataset["vectors"]
    slices = (keypoints[offsets[index]:offsets[index + 1]] for index in range(len(ids)))
    logits, vectors = infer(session, slices)
    np.savez_compressed(cache_path, sourceHash=source_hash, modelHash=MODEL_SHA256, logits=logits, vectors=vectors)
    return ids, glosses, logits, vectors


def normalize(vectors):
    vectors = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def train_centroids(ids, classes, vectors, val_ids):
    signers = {int(re.search(r"_signer(\d+)_rep1_glosses/", item).group(1)) for item in ids}
    if not signers or not signers <= {1, 2, 4} or set(ids) & set(val_ids):
        raise ValueError("Prototype fitting requires source-training clips from signers 1, 2 and 4 only")
    vectors = normalize(vectors)
    centers = np.zeros((310, EMBEDDING_WIDTH), dtype=np.float64)
    counts = np.bincount(classes, minlength=310)
    np.add.at(centers, classes, vectors)
    return normalize(centers), counts


def evidence(logits, vectors, bias, weights, centers, counts):
    logits = logits.astype(np.float64)
    order = np.argsort(-logits, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(logits))
    winner, runner_up = order.T
    winning_logit = logits[rows, winner]
    exps = np.exp(logits - winning_logit[:, None])
    denominator = exps.sum(axis=1)
    confidence = 1. / denominator
    margin = (1. - exps[rows, runner_up]) / denominator
    adjusted = (logits - bias) / weights
    chosen_adjusted = adjusted[rows, winner].copy()
    adjusted[rows, winner] = -np.inf
    signals = {
        "winningLogit": winning_logit,
        "classWeightMargin": chosen_adjusted - adjusted.max(axis=1),
    }
    if centers is not None:
        similarity = normalize(vectors) @ centers.T
        similarity[:, counts == 0] = -np.inf
        prototype_similarity = similarity[rows, winner].copy()
        similarity[rows, winner] = -np.inf
        signals["prototypeSimilarity"] = prototype_similarity
        signals["prototypeMargin"] = prototype_similarity - similarity.max(axis=1)
    return winner, confidence, margin, signals


def counts(accepted, reference, winners, roster):
    known = roster[reference]
    right = accepted & (reference == winners)
    wrong = accepted & known & ~right
    outside = accepted & ~known
    return {
        "knownClips": int(known.sum()), "correct": int(right.sum()), "wrong": int(wrong.sum()),
        "abstentions": int((known & ~accepted).sum()), "outsideClips": int((~known).sum()),
        "outsideFalseAccepts": int(outside.sum()),
        "correctLabels": int(np.unique(reference[right]).size),
    }


def fit_signal(confidence, signal, in_roster, reference, winners, roster, *, max_wrong=None, max_outside=None):
    known = roster[reference]
    right = known & (reference == winners)
    wrong = known & ~right
    outside = ~known
    # Deterministic quantiles depend only on the calibration repetition.
    probabilities = np.unique(np.quantile(confidence[in_roster], np.linspace(0., 1., 101), method="nearest"))
    finite_signal = signal[in_roster & np.isfinite(signal)]
    thresholds = np.unique(np.r_[np.quantile(finite_signal, np.linspace(0., 1., 101), method="nearest"),
                                np.nextafter(finite_signal.max(), np.inf)])
    wrong_limit = MAX_ERROR_RATE * known.sum() if max_wrong is None else max_wrong
    outside_limit = MAX_ERROR_RATE * outside.sum() if max_outside is None else max_outside
    signal_masks = signal >= thresholds[:, None]
    best = None
    for minimum_confidence in probabilities:
        active = signal_masks & in_roster & (confidence >= minimum_confidence)
        wrong_count = np.count_nonzero(active & wrong, axis=1)
        outside_count = np.count_nonzero(active & outside, axis=1)
        right_count = np.count_nonzero(active & right, axis=1)
        valid = (wrong_count <= wrong_limit) & (outside_count <= outside_limit)
        for index in np.flatnonzero(valid):
            key = (int(right_count[index]), -int(outside_count[index]), -int(wrong_count[index]),
                   float(minimum_confidence), float(thresholds[index]))
            if best is None or key > best[0]:
                best = (key, {"minimumConfidence": float(minimum_confidence), "minimumEvidence": float(thresholds[index])})
    if best is None:
        raise ValueError("No feasible signal thresholds")
    return best


def cached_rgb_poses(data_dir, labels):
    manifest = {}
    files = [data_dir / "manifest.json", data_dir / "v3/manifest.json", data_dir / "open-set-v2/manifest.json"]
    for path in files:
        for row in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            manifest[row["id"]] = row["raw_label"]
    clips = {}
    for path in sorted((data_dir / "features").glob("extracted*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            identity, raw = row["id"], row.get("raw")
            if not raw or identity in clips or manifest.get(identity) not in labels or "_signer3_" in identity:
                continue
            points = np.zeros((len(raw), 75, 3), dtype=np.float32)
            for index, frame in enumerate(raw):
                for key, start, length in (("pose", 0, 33), ("left", 33, 21), ("right", 54, 21)):
                    if len(frame[key]) >= length:
                        points[index, start:start + length, :2] = [(joint["x"], joint["y"]) for joint in frame[key][:length]]
            clips[identity] = (manifest[identity], points)
    return clips


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/cache/openhands-gsl"))
    parser.add_argument("--reports", type=Path, default=Path("reports/pretrained-gsl"))
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    report_path = args.reports / ("stgcn-geometry-experiment.json" if args.geometry_only else "stgcn-verifier-experiment.json")
    old_policy = json.loads((args.reports / "stgcn-gate-config.json").read_text(encoding="utf-8"))
    labels, supported, val_references = read_metadata(args.cache / "gsl_metadata.zip")
    if supported != old_policy["supportedLabels"] or old_policy["runtimeModelSha256"] != MODEL_SHA256:
        raise ValueError("Existing policy does not match source model and training roster")
    index = {label: position for position, label in enumerate(labels)}
    roster = np.isin(labels, supported)
    verifier_path = args.cache / "stgcn-embedding-experiment.onnx"
    session, bias, weights = make_session(args.cache / "stgcn.onnx", None if args.geometry_only else verifier_path)
    ids, glosses, logits, vectors = infer_native(session, args.cache / "official-val-poses.npz", args.cache)
    if dict(zip(ids, glosses)) != val_references:
        raise ValueError("Validation archive IDs or references do not match official metadata")
    reference = np.asarray([index[label] for label in glosses])
    reps = np.asarray([int(re.search(r"_signer3_rep(\d+)_glosses/", item).group(1)) for item in ids])
    calibration, evaluation = reps == 1, reps == 2
    centers = train_counts = None
    train_ids = ()
    if not args.geometry_only:
        train_ids, train_glosses, _, train_vectors = infer_native(
            session, args.cache / "official-train-prototypes-poses.npz", args.cache)
        centers, train_counts = train_centroids(train_ids, [index[label] for label in train_glosses], train_vectors, ids)
    winners, confidence, margin, signals = evidence(logits, vectors, bias, weights, centers, train_counts)
    in_roster = roster[winners]
    old_accept = in_roster & (confidence >= old_policy["minConfidence"]) & (margin >= old_policy["minMargin"])
    old_metrics = counts(old_accept[evaluation], reference[evaluation], winners[evaluation], roster)
    baseline_calibration = counts(old_accept[calibration], reference[calibration], winners[calibration], roster)
    print(json.dumps({"baseline": old_metrics}), flush=True)

    candidates = {}
    strict_candidates = {}
    for name, signal in signals.items():
        values = (confidence[calibration], signal[calibration], in_roster[calibration],
                  reference[calibration], winners[calibration], roster)
        candidates[name] = fit_signal(*values)
        strict_candidates[name] = fit_signal(*values, max_wrong=baseline_calibration["wrong"], max_outside=0)
    # Candidate and thresholds are both selected before scoring repetition 2.
    method = max(candidates, key=lambda name: candidates[name][0][:3])
    _, thresholds = candidates[method]
    strict_method = max(strict_candidates, key=lambda name: strict_candidates[name][0][:3])
    strict_key, strict_thresholds = strict_candidates[strict_method]

    def accept(eligible, scores, signal):
        return eligible & (scores >= thresholds["minimumConfidence"]) & (signal >= thresholds["minimumEvidence"])

    selected = accept(in_roster, confidence, signals[method])
    strict_selected = (in_roster & (confidence >= strict_thresholds["minimumConfidence"])
                       & (signals[strict_method] >= strict_thresholds["minimumEvidence"]))
    proposed_metrics = counts(selected[evaluation], reference[evaluation], winners[evaluation], roster)
    strict_metrics = counts(strict_selected[evaluation], reference[evaluation], winners[evaluation], roster)
    def diagnostic_outcomes(fitted):
        outcomes = {}
        for name, (key, values) in fitted.items():
            accepted = (in_roster & (confidence >= values["minimumConfidence"])
                        & (signals[name] >= values["minimumEvidence"]))
            outcomes[name] = {"calibration": {"correct": key[0], "outsideFalseAccepts": -key[1], "wrong": -key[2]},
                              "thresholds": values,
                              "heldRepetition": counts(accepted[evaluation], reference[evaluation], winners[evaluation], roster)}
        return outcomes
    report = {
        "scope": ["Calibration on signer 3 repetition 1 only; evaluation on repetition 2, same signer and mostly repeated prompts.",
                  "Prototype classes fitted only on official TRAIN signer 1, 2 and 4 repetition 1.",
                  "Cached MediaPipe RGB replay uses other signers and the browser's landmark family, but shared upstream corpus and possible training overlap.",
                  "A different signer, a person at a new webcam, and true out-of-distribution movements remain untested."],
        "method": method, "thresholds": thresholds, "trainClips": len(train_ids),
        "trainClassCoverage": int(np.count_nonzero(train_counts)) if train_counts is not None else 0,
        "calibration": {name: {"correct": key[0], "outsideFalseAccepts": -key[1], "wrong": -key[2], **values}
                        for name, (key, values) in candidates.items()},
        "heldRepetition": {"baseline": old_metrics, "proposed": proposed_metrics},
        "diagnosticUnselectedCandidates": {"onePercentCalibration": diagnostic_outcomes(candidates),
                                           "strictCalibration": diagnostic_outcomes(strict_candidates)},
        "strictVariant": {"method": strict_method, "thresholds": strict_thresholds,
                          "calibrationConstraint": {"outsideFalseAccepts": 0, "wrongAtMost": baseline_calibration["wrong"]},
                          "calibration": {"correct": strict_key[0], "outsideFalseAccepts": -strict_key[1], "wrong": -strict_key[2]},
                          "heldRepetition": strict_metrics},
        "heldByGloss": {label: {
            "clips": int((evaluation & (reference == index[label])).sum()),
            "baselineCorrect": int((evaluation & (reference == index[label]) & (winners == reference) & old_accept).sum()),
            "proposedCorrect": int((evaluation & (reference == index[label]) & (winners == reference) & selected).sum()),
        } for label in supported},
    }
    rgb = cached_rgb_poses(Path("data"), index)
    if rgb:
        rgb_labels, rgb_poses = zip(*rgb.values())
        rgb_reference = np.asarray([index[label] for label in rgb_labels])
        rgb_winners, rgb_conf, rgb_margin, rgb_signals = evidence(*infer(session, rgb_poses), bias, weights, centers, train_counts)
        rgb_eligible = roster[rgb_winners]
        rgb_old = rgb_eligible & (rgb_conf >= old_policy["minConfidence"]) & (rgb_margin >= old_policy["minMargin"])
        rgb_new = accept(rgb_eligible, rgb_conf, rgb_signals[method])
        report["cachedRgb"] = {"baseline": counts(rgb_old, rgb_reference, rgb_winners, roster),
                               "proposed": counts(rgb_new, rgb_reference, rgb_winners, roster),
                               "rows": [{"id": identity, "reference": expected, "winner": labels[prediction],
                                         "baselineAccepted": bool(old), "proposedAccepted": bool(new)}
                                        for identity, expected, prediction, old, new in
                                        zip(rgb, rgb_labels, rgb_winners, rgb_old, rgb_new)]}
    write_json(report_path, report)
    policy = {"method": method, "runtimeModelSha256": MODEL_SHA256, **thresholds}
    if method == "classWeightMargin":
        policy.update({"classifierBias": bias.tolist(), "classifierWeightNorm": weights.tolist()})
    if method.startswith("prototype"):
        policy.update({"runtimeModelSha256": sha256(verifier_path), "embeddingWidth": EMBEDDING_WIDTH,
                       "classCentroids": centers.tolist(), "classCentroidCounts": train_counts.tolist()})
    policy_path = report_path.with_name(report_path.stem + "-candidate.json")
    write_json(policy_path, policy)
    if args.geometry_only:
        full_policy = {**old_policy, "decisionGate": {key: value for key, value in policy.items()
                                                      if key != "runtimeModelSha256"}}
        write_json(args.reports / "stgcn-gate-expanded-candidate.json", full_policy)
    print(json.dumps({"candidate": policy_path.name, "method": method, "evaluation": proposed_metrics,
                      "strictVariant": report["strictVariant"],
                      "cachedRgb": {key: value for key, value in report.get("cachedRgb", {}).items() if key != "rows"}}), flush=True)


if __name__ == "__main__":
    main()
