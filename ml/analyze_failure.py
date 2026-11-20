"""Post-test diagnosis of frozen artifacts; never fit or change deployment thresholds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import decisions, load_dataset, softmax
from train import TemporalClassifier, predict


def summary(values: np.ndarray) -> dict:
    return {"n": int(len(values)), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max())} if len(values) else {"n": 0}


def analyze(features: Path, artifacts: Path, output: Path) -> dict:
    data = load_dataset(features)
    report = json.loads((artifacts / "report.json").read_text(encoding="utf-8"))
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = TemporalClassifier(state["feature_mean"].numpy().reshape(-1), state["feature_std"].numpy().reshape(-1),
                               len(config["output"]["labels"]), checkpoint["channels"], state["feature_keep"].numpy().reshape(-1))
    model.load_state_dict(state)
    probs = softmax(predict(model, data.x), config["rejection"]["temperature"])
    k = len(data.classes)
    predicted, accepted, confidence = decisions(probs, data.quality, k,
                                               config["rejection"]["confidenceThreshold"], config["rejection"]["minQuality"])
    with (artifacts / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        saved = {row["source_id"]: row for row in csv.DictReader(stream)}
    for i, identity in enumerate(data.source_id):
        row = saved[identity]
        if int(row["top_known_y"]) != int(predicted[i]) or (row["accepted"] == "True") != bool(accepted[i]):
            raise AssertionError(f"Frozen diagnostic predictions differ from original run: {identity}")
        if abs(float(row["confidence"]) - confidence[i]) > 1e-6:
            raise AssertionError(f"Frozen diagnostic confidence differs from original run: {identity}")
    competing = probs.copy()
    competing[np.arange(len(probs)), predicted] = -np.inf
    margin = confidence - competing.max(axis=1)
    background_index = config["backgroundIndex"]
    background = probs[:, background_index] if background_index is not None else np.zeros(len(probs))
    output_labels = config["output"]["labels"]

    def record(i):
        return {"source_id": str(data.source_id[i]), "signer": str(data.signer[i]), "split": str(data.split[i]),
                "true_label": str(data.raw_label[i]), "top_known_label": data.classes[int(predicted[i])],
                "global_top_label": output_labels[int(probs[i].argmax())],
                "accepted": bool(accepted[i]), "known_confidence": float(confidence[i]),
                "background_probability": float(background[i]), "top_known_minus_best_competitor_margin": float(margin[i]),
                "probabilities": {label: float(probs[i, j]) for j, label in enumerate(output_labels)},
                "tracking_quality": float(data.quality[i]),
                "resampled_presence_means_left_right_pose_face": data.x[i, :, -4:].mean(axis=0).tolist()}

    unknown = {}
    for split in ("train", "val", "test"):
        mask = data.mask(split) & (data.y == -1)
        unknown[split] = {"semantic_categories": sorted(set(data.raw_label[mask].tolist())),
                          "false_acceptance_n": int(accepted[mask].sum()), "examples_n": int(mask.sum()),
                          "known_confidence": summary(confidence[mask]), "margin": summary(margin[mask]),
                          "background_probability": summary(background[mask]),
                          "quality": summary(data.quality[mask]),
                          "predicted_known_counts": {label: int((predicted[mask] == index).sum()) for index, label in enumerate(data.classes)},
                          "records": [record(int(i)) for i in np.flatnonzero(mask)]}
    known_test = data.mask("test") & (data.y >= 0)
    rejected = known_test & ~accepted
    wrong = known_test & (predicted != data.y)
    quality_fail = data.quality < config["rejection"]["minQuality"]
    confidence_fail = confidence < config["rejection"]["confidenceThreshold"]
    background_fail = probs.argmax(axis=1) >= k
    metric_keys = ["known_n", "unknown_n", "closed_set_accuracy", "closed_set_macro_f1",
                   "accepted_known_n", "accepted_known_accuracy", "known_coverage", "known_rejection_rate",
                   "unknown_false_acceptance_n", "unknown_false_acceptance_rate", "all_accepted_n", "all_accepted_accuracy"]
    tcn_test = report["metrics"]["test"]
    centroid_test = report["baseline"]["metrics"]["test"]
    known_correct_n = int(np.trace(np.asarray(tcn_test["closed_set_confusion_matrix"])))
    centroid_correct_n = int(np.trace(np.asarray(centroid_test["closed_set_confusion_matrix"])))
    accepted_correct_n = int((known_test & accepted & (predicted == data.y)).sum())
    accepted_unknown = data.mask("test") & (data.y == -1) & accepted
    accepted_unknown_labels = {label: int((accepted_unknown & (predicted == index)).sum())
                               for index, label in enumerate(data.classes)
                               if np.any(accepted_unknown & (predicted == index))}
    cnn_comparison = ("higher" if tcn_test["closed_set_accuracy"] > centroid_test["closed_set_accuracy"]
                      else "lower" if tcn_test["closed_set_accuracy"] < centroid_test["closed_set_accuracy"] else "equal")
    result = {
        "analysis_type": "post-test descriptive diagnosis of frozen single default run; no parameter or threshold changes",
        "evaluation_context": report.get("evaluationContext", "initial_signer_heldout_evaluation"),
        "seed": report["seed"], "model_version": report["modelVersion"],
        "artifact_sha256": {name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
                            for name in ("model.onnx", "config.json", "report.json", "predictions.csv")},
        "frozen_predictions_match_original": True,
        "frozen_rejection_config": config["rejection"],
        "unknown_comparison": unknown,
        "known_test_failures": {
            "misclassified_n": int(wrong.sum()), "accepted_wrong_n": int((wrong & accepted).sum()),
            "misclassified_records": [record(int(i)) for i in np.flatnonzero(wrong)],
            "rejected_n": int(rejected.sum()),
            "rejection_reason_counts_nonexclusive": {
                "insufficient_tracking": int((rejected & quality_fail).sum()),
                "known_confidence_below_threshold": int((rejected & confidence_fail).sum()),
                "background_output_wins": int((rejected & background_fail).sum())},
            "accepted_quality": summary(data.quality[known_test & accepted]),
            "rejected_quality": summary(data.quality[rejected]),
        },
        "baseline_comparison": {model_name: {split: {key: metrics[split][key] for key in metric_keys}
                                             for split in ("train", "val", "test")}
                                for model_name, metrics in (("temporal_cnn", report["metrics"]),
                                                            ("nearest_centroid", report["baseline"]["metrics"]))},
        "observations": [
            f"The frozen rejection rule falsely accepted {tcn_test['unknown_false_acceptance_n']}/{tcn_test['unknown_n']} test examples from {len(unknown['test']['semantic_categories'])} unsupported semantic categories.",
            f"Accepted unknown prediction counts are {accepted_unknown_labels}; all of these are errors, regardless of the model score.",
            f"TCN known-only accuracy is {known_correct_n}/{tcn_test['known_n']}; accepted accuracy over both known and unknown inputs is {accepted_correct_n}/{tcn_test['all_accepted_n']}.",
            f"The centroid baseline recognizes {centroid_correct_n}/{centroid_test['known_n']} known test examples and falsely accepts {centroid_test['unknown_false_acceptance_n']}/{centroid_test['unknown_n']} unknown examples.",
            f"On this small test set, TCN closed-set accuracy is {cnn_comparison} than the baseline; this is not a general superiority claim.",
            f"Training background covered {len(unknown['train']['semantic_categories'])} negative semantic categories and validation covered {len(unknown['val']['semantic_categories'])}; successful validation rejection is not evidence of universal background recognition.",
            "No live-webcam evaluation with a new fluent GSL signer has been performed.",
        ],
        "hypotheses_not_verified_causes": [
            "The training background category may not cover the feature patterns of other unsupported signs.",
            "Some unsupported motions may share captured geometric or temporal features with ΓΕΙΑ.",
            "The small scripted subset may allow a simple distance baseline to separate known classes without needing the CNN's additional capacity.",
            "Tracking loss may contribute to some errors or abstentions; the recorded quality alone does not establish causation.",
        ],
        "next_experiment_requirements": [
            "Retain this initial report unchanged and preregister new training/validation categories before changing the method.",
            "Any future method informed by these failures needs a new untouched unknown test set; the inspected test clips are no longer fresh selection evidence.",
            "Evaluate more unsupported categories, non-sign motions, and genuinely new webcam signers, reporting counts and coverage.",
            "Do not change the deployment threshold based on this test result or present the current system as reliably rejecting unsupported signs.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"report": str(output.resolve()),
            "validation_unknown": {key: unknown["val"][key] for key in ("known_confidence", "margin", "background_probability")},
            "test_unknown": {key: unknown["test"][key] for key in ("known_confidence", "margin", "background_probability")},
            "known_failure_counts": {"misclassified": int(wrong.sum()), "accepted_wrong": int((wrong & accepted).sum()),
                                     "rejected": int(rejected.sum()), "reason_counts": result["known_test_failures"]["rejection_reason_counts_nonexclusive"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.features, args.artifacts, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
