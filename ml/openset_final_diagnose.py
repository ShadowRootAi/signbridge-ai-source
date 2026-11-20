"""Bounded post-evaluation description; frozen B, no fitting or selection changes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openset_core import metrics, motion_metrics, score_gate
from openset_diagnose import infer_embeddings, load_frozen
from openset_train import load_split


def stats(values):
    values = np.asarray(values)
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max())} if len(values) else {"n": 0}


def run(data_dir, artifact_dir, output):
    final = json.loads((artifact_dir / "final-evaluation.json").read_text(encoding="utf-8"))
    selection = json.loads((artifact_dir / "selection.locked.json").read_text(encoding="utf-8"))
    b_dir = artifact_dir / "B_tcn_diverse_other"
    saved = json.loads((b_dir / "final-predictions.json").read_text(encoding="utf-8"))
    data = load_split(data_dir / "final_test.npz", "test")
    if saved["source_id"] != data.source_id.tolist():
        raise ValueError("Saved final prediction identities do not match the sealed tensors")
    predicted = np.asarray(saved["predicted"]); accepted = np.asarray(saved["accepted"], dtype=bool)
    confidence = np.asarray(saved["confidence"])
    motion = motion_metrics(data.x)
    category = {}
    for label in np.unique(data.raw_label[data.y < 0]):
        mask = (data.y < 0) & (data.raw_label == label)
        category[str(label)] = {"n": int(mask.sum()), "false_acceptance_n": int((mask & accepted).sum()),
                                "confidence_all": stats(confidence[mask]), "confidence_false_accepts": stats(confidence[mask & accepted]),
                                "tracking_quality": stats(data.quality[mask]),
                                "motion_body_rms": stats(motion["motionBodyRms"][mask]),
                                "motion_local_rms": stats(motion["motionLocalRms"][mask]),
                                "false_label_counts": {class_name: int((mask & accepted & (predicted == i)).sum())
                                                       for i, class_name in enumerate(data.classes)}}
    known = data.y >= 0
    model, config = load_frozen(b_dir)
    development_diagnostics = {}
    permutation = np.random.default_rng(20261108).permutation(48)
    for split in ("train", "val"):
        development = load_split(data_dir / f"{split}.npz", split)
        measured_motion = motion_metrics(development.x)["motionBodyRms"]
        changes = {"identity": development.x,
                   "static_temporal_mean": np.repeat(development.x.mean(axis=1, keepdims=True), 48, axis=1),
                   "shuffle_time_seed_20261108": development.x[:, permutation].copy()}
        development_diagnostics[split] = {}
        for label, x in changes.items():
            logits, embedding = infer_embeddings(model, x)
            scored = score_gate(logits, development.quality, measured_motion, config, embedding)
            development_diagnostics[split][label] = metrics(development.y, scored, development.classes,
                                                           development.signer, development.raw_label)
    b = final["candidates"]["B_tcn_diverse_other"]
    result = {
        "scope": "post-test description using saved final predictions; temporal ablations use development data only",
        "no_refitting_or_threshold_change": True, "selected_candidate_unchanged": selection["selected_candidate"],
        "final_evaluation_sha256": hashlib.sha256((artifact_dir / "final-evaluation.json").read_bytes()).hexdigest(),
        "unknown_categories": category,
        "known_errors": {"misclassified_n": int((known & (predicted != data.y)).sum()),
                         "accepted_wrong_n": int((known & accepted & (predicted != data.y)).sum()),
                         "rejected_n": int((known & ~accepted).sum()), "per_sign": b["per_known_sign"]},
        "selected_B_final": b, "strong_full_feature_baseline_final": final["references"]["nearest_centroid"],
        "development_only_temporal_diagnostics": development_diagnostics,
        "observed_comparison": {
            "same_final_set_original_frozen_unknown_false_acceptance": final["references"]["original_frozen_tcn"]["unknown_false_acceptance_rate"],
            "same_final_set_B_unknown_false_acceptance": b["unknown_false_acceptance_rate"],
            "baseline_unknown_false_acceptance": final["references"]["nearest_centroid"]["unknown_false_acceptance_rate"],
            "baseline_selected_post_test": False},
        "hypotheses_not_causal_proof": [
            "A single OTHER output still covers only four training semantic categories; its learned boundary may not encompass unseen signs.",
            "Unknown signs accepted as ΓΕΙΑ may share captured hand shape, location or pose statistics with that class; confidence does not establish semantic correctness.",
            "The full-feature centroid retains the complete standardized temporal vector, whereas CNN pooling compresses it into48values; information loss or fitting bias could explain some baseline advantage, but was not isolated experimentally.",
            "Additional known examples, broader negative data and recalibration changed together, so improvement over the original frozen POC cannot be attributed solely to OTHER training.",
            "Scripted scenarios, body geometry and tracking patterns may remain correlated with labels despite using no RGB classifier inputs. These correlations are not established as causal explanations.",
            "Static and shuffled-input diagnostics alter the distribution and do not prove causal reliance or nonreliance on temporal order; static hand shape may carry legitimate semantics."],
        "limits_and_next_work": [
            "B was selected among the three preregistered candidates using validation only. The centroid was an explicitly excluded reference, so its stronger final result cannot silently replace the winner.",
            "The full-feature baseline is a serious next-method candidate requiring a new untouched evaluation protocol.",
            "All tested methods still falsely accept unsupported signs; this is experimental progress, not reliable open-set translation.",
            "Final clips are new, but people3and7 had been seen in earlier diagnostics; identities are not newly unseen.",
            "No human webcam negative or fluent-GSL recording was supplied; human webcam performance remains untested."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"report": str(output.resolve()), "development_known_accuracy": {
        split: {name: report["closed_set_accuracy"] for name, report in rows.items()}
        for split, rows in development_diagnostics.items()}, "unknown_categories": category}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.artifacts, args.output), ensure_ascii=False, indent=2))
