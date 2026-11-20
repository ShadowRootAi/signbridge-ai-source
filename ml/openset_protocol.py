"""Prespecified open-set methods. Frozen before new feature fitting or final evaluation."""
import hashlib
import json
from pathlib import Path

PROTOCOL = {
    "id": "signbridge-open-set-v2-20261108", "seed": 20261108,
    "candidates": ["A_frozen_tcn_recalibrated", "B_tcn_diverse_other", "C_frozen_embedding_prototype"],
    "references_not_selected": ["original_frozen_tcn_config", "nearest_centroid"],
    "selection": {"minimum_known_coverage": 0.60, "minimum_accepted_known_accuracy": 0.90,
                  "priority": ["minimize_unknown_false_acceptance", "maximize_correct_known_acceptance",
                               "maximize_accepted_known_accuracy", "maximize_known_coverage"],
                  "no_eligible_setting": "report_failure_without_relaxing_constraints",
                  "candidate_tie_order": ["A_frozen_tcn_recalibrated", "B_tcn_diverse_other", "C_frozen_embedding_prototype"]},
    "temperature_grid": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "temperature_objective": "validation cross-entropy; unknown target is background output",
    "confidence_grid": [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 0.975, 0.99],
    "quality_grid": [0.25, 0.5, 0.75],
    "motion_grid": ["disabled", "train_known_q01", "train_known_q05"],
    "motion_metric": "motionBodyRms; sqrt(sum(dx^2+dy^2)/valid_joint_pairs), original preprocessed XY units",
    "motion_validity": "adjacent endpoint hand and pose masks >= 1-1e-6; absent pairs excluded; no pairs gives zero",
    "prototype": {"embedding": "frozen TCN final temporal ReLU output averaged over time; 48 dimensions",
                  "statistics": "training known examples only; mean/std ddof0; std floor0.05; inactive std<1e-6",
                  "distance": "RMS Euclidean over active standardized embedding dimensions",
                  "tie_rule": "first class index", "require_tcn_nearest_prototype_agreement": True,
                  "threshold_quantiles_percent": [50, 75, 90, 95, 97.5, 99, 100, "disabled"],
                  "threshold_source": "per-class TRAIN leave-one-out prototype distances; numpy linear quantile",
                  "no_active_dimensions_or_fewer_than_two_examples_per_class": "candidate_invalid"},
    "training_B": {"architecture": "unchanged 3-layer TemporalClassifier, channels48, outputs4",
                   "epochs": 250, "patience": 35, "batch_size": 32, "learning_rate": 0.001,
                   "weight_decay": 0.001, "checkpoint": "validation_cross_entropy",
                   "normalization": "unchanged training-only normalization from POC v2",
                   "augmentation": "none"},
    "test_policy": "fit separate train/validation NPZ only; freeze selection hash before separate final-test NPZ evaluation; evaluate all candidates once",
    "human_negative_policy": "separate observational holdout; never fit thresholds on those recordings",
    "old_test_policy": "old signer3/7 test clips remain diagnostic only; never training or calibration",
    "diagnostic_ablation_policy": "frozen already-seen POC data only; altered inputs are distribution shifts, not causal proof",
}


def freeze(output: Path, plan_lock: Path | None = None) -> dict:
    document = {"protocol": PROTOCOL}
    if plan_lock is not None:
        document["dataset_plan_sha256"] = hashlib.sha256(plan_lock.read_bytes()).hexdigest()
    encoded = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if output.exists() and output.read_bytes() != encoded:
        raise ValueError("Refusing to overwrite a different frozen algorithm protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return {"path": str(output.resolve()), "sha256": hashlib.sha256(encoded).hexdigest()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-lock", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.plan_lock), indent=2))
