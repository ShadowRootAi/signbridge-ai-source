"""Bounded descriptive controls on existing development data; never opens final data.

Run from the project directory:
  .venv/Scripts/python.exe -X utf8 ml/v5_diagnostics.py

No model selection, acceptance calibration, production edits, or CNN fitting.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from openset_core import metrics, motion_metrics, score_gate


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261109
BLOCKS = {"left_body": (0, 42), "left_shape": (42, 84),
          "right_body": (84, 126), "right_shape": (126, 168),
          "pose": (168, 186), "face": (186, 220), "presence": (220, 224)}
INPUTS = ["data/v3/train.npz", "data/v3/val.npz",
          "artifacts/open-set-v2/reference-centroid.npz",
          "artifacts/open-set-v2/reference-centroid-config.json",
          "artifacts/v3-centroid/augmented_pool/centroids.npz",
          "artifacts/v3-centroid/augmented_pool/config.json",
          "reports/v4/greeting-validation-diagnosis.json"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_development(role):
    if role not in ("train", "val"):
        raise ValueError("This diagnostic only accepts train or val")
    with np.load(ROOT / f"data/v3/{role}.npz", allow_pickle=False) as f:
        d = {k: f[k].copy() for k in ("x", "y", "signer", "source_id", "split", "raw_label", "quality", "classes")}
    assert set(d["split"]) == {role}
    assert set(d["signer"]) == ({"1", "2", "4", "5"} if role == "train" else {"6"})
    assert d["x"].shape == (len(d["y"]), 48, 224)
    assert np.isfinite(d["x"]).all()
    assert set(d["y"]) == {-1, 0, 1, 2}
    d["scenario"] = np.array([re.fullmatch(r"(.+)_signer\d+_rep\d+_sentences/sentences\d+", s).group(1)
                               for s in d["source_id"]])
    return d


def load_reference(npz, config):
    with np.load(ROOT / npz, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in ("mean", "std", "keep", "centroids")}
    arrays["config"] = json.loads((ROOT / config).read_text(encoding="utf-8"))
    return arrays


def normalize(x, model):
    # Match the frozen Float32 reference arithmetic exactly.
    return np.clip((x - model["mean"][None, None, :]) / model["std"][None, None, :], -8, 8) * model["keep"][None, None, :]


def logits_for(x, model):
    z = normalize(x, model).reshape(len(x), -1)
    return -np.mean((z[:, None, :] - model["centroids"][None, :, :]) ** 2, axis=2)


def score(x, d, model):
    return score_gate(logits_for(x, model), d["quality"], motion_metrics(x)["motionBodyRms"], model["config"])


def description(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def counts(d):
    return {"n": len(d["y"]), "known_n": int((d["y"] >= 0).sum()), "unknown_n": int((d["y"] < 0).sum()),
            "per_signer": dict(Counter(d["signer"].tolist())), "per_label": dict(Counter(d["raw_label"].tolist()))}


def report_score(d, scored):
    return metrics(d["y"], scored, d["classes"].tolist(), d["signer"], d["raw_label"])


def frozen_ablations(d, model):
    baseline = score(d["x"], d, model)
    result = {}
    for name in ("normal", "reverse", "shuffle_0", "shuffle_1", "shuffle_2", "static_mean",
                 "drop_reported_left", "drop_reported_right"):
        x = d["x"].copy()
        if name == "reverse":
            x = x[:, ::-1].copy()
        elif name.startswith("shuffle_"):
            rng = np.random.default_rng(SEED + int(name[-1]))
            x = np.stack([row[rng.permutation(48)] for row in x])
        elif name == "static_mean":
            x = np.repeat(x.mean(axis=1, keepdims=True), 48, axis=1)
        elif name.startswith("drop_reported_"):
            left = name.endswith("left")
            x[:, :, 0 if left else 84:84 if left else 168] = 0
            x[:, :, 220 if left else 221] = 0
        s = score(x, d, model)
        result[name] = {"metrics": report_score(d, s),
                        "best_known_class_changed_all_rows_n": int((s["predicted"] != baseline["predicted"]).sum()),
                        "best_known_class_changed_known_rows_n": int(((d["y"] >= 0) & (s["predicted"] != baseline["predicted"])).sum()),
                        "acceptance_changed_n": int((s["accepted"] != baseline["accepted"]).sum()),
                        "accepted_newly_n": int((s["accepted"] & ~baseline["accepted"]).sum())}
    return result


def forced_report(y, distances, classes):
    predicted = distances.argmin(axis=1)
    target = np.where(y < 0, 3, y)
    conf = np.zeros((4, 4), dtype=int)
    np.add.at(conf, (target, predicted), 1)
    known = y >= 0
    forced_known = distances[:, :3].argmin(axis=1)
    return {"four_way_confusion_true_rows": conf.tolist(), "class_order": classes + ["OTHER"],
            "four_way_correct_n": int((target == predicted).sum()),
            "forced_known_top1_correct_n": int((known & (forced_known == y)).sum()),
            "known_n": int(known.sum()), "unknown_n": int((~known).sum()),
            "unknown_winning_known_centroid_n": int(((~known) & (predicted < 3)).sum()),
            "per_known_sign": {label: {"n": int((y == i).sum()),
                                        "forced_known_correct_n": int(((y == i) & (forced_known == i)).sum()),
                                        "four_way_correct_n": int(((y == i) & (predicted == i)).sum())}
                               for i, label in enumerate(classes)}}


def feature_controls(train, val, frozen):
    # Existing train-only scaling is frozen; fit only class means to development-train.
    ztrain, zval = normalize(train["x"], frozen), normalize(val["x"], frozen)
    cols = {"full_ordered": np.arange(224), "all_static_mean": np.arange(224),
            "hand_shape_static": np.r_[42:84, 126:168], "hand_body_location_static": np.r_[0:42, 84:126],
            "pose_static": np.arange(168, 186), "face_static": np.arange(186, 220),
            "presence_static": np.arange(220, 224)}
    target = np.where(train["y"] < 0, 3, train["y"])
    results = {}
    for name, indices in cols.items():
        a, b = ztrain[:, :, indices], zval[:, :, indices]
        a, b = (a.reshape(len(a), -1), b.reshape(len(b), -1)) if name == "full_ordered" else (a.mean(1), b.mean(1))
        centers = np.stack([a[target == label].mean(0) for label in range(4)])
        distance = np.mean((b[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        results[name] = {"dimension_n": a.shape[1], "validation": forced_report(val["y"], distance, val["classes"].tolist())}
    return results


def scenario_control(train, val):
    known = train["y"] >= 0
    fallback = int(np.bincount(train["y"][known], minlength=3).argmax())
    table = {s: np.bincount(train["y"][known & (train["scenario"] == s)], minlength=3).tolist()
             for s in sorted(set(train["scenario"][known]))}
    predictions = np.array([int(np.argmax(table[s])) if s in table else fallback for s in val["scenario"]])
    vk = val["y"] >= 0
    return {"training_known_label_counts_by_scenario": table, "first_index_wins_ties": True,
            "unseen_scenario_fallback_label": train["classes"][fallback],
            "known_val_n": int(vk.sum()), "known_val_correct_n": int((vk & (predictions == val["y"])).sum()),
            "val_unseen_scenario_n": int(sum(s not in table for s in val["scenario"])),
            "known_val_unseen_scenario_n": int(sum(s not in table for s in val["scenario"][vk])),
            "per_known_sign": {label: {"n": int((val["y"] == i).sum()),
                                        "correct_n": int(((val["y"] == i) & (predictions == i)).sum())}
                               for i, label in enumerate(val["classes"])},
            "interpretation": "Source scenario metadata is a diagnostic-only control, never classifier input; no RGB/background causal conclusion."}


def paired_greetings(train, val, models, prior):
    prior_rows = {r["source_path"]: r for r in prior["greeting_validation_rows"]}
    tridx, vidx = np.flatnonzero(train["y"] == 0), np.flatnonzero(val["y"] == 0)
    assert len(vidx) == 11
    frozen = models["frozen_v2_centroid"]
    ztrain, zval = normalize(train["x"], frozen), normalize(val["x"], frozen)
    scored = {name: score(val["x"], val, model) for name, model in models.items()}
    rows = []
    max_sum_error = 0.0
    max_prior_error = 0.0
    for i in vidx:
        source = str(val["source_id"][i]); previous = prior_rows[source]
        candidates = tridx[train["scenario"][tridx] == val["scenario"][i]]
        assert len(candidates), "Every greeting must have a same-scenario training comparator"
        pair_distances = np.mean((ztrain[candidates] - zval[i]) ** 2, axis=(1, 2))
        pair = int(candidates[pair_distances.argmin()])
        record = {"source_id": source, "scenario": str(val["scenario"][i]),
                  "legacy_quality": float(val["quality"][i]),
                  "mean_resampled_presence_left_right_pose_face": val["x"][i, :, 220:224].mean(0).tolist(),
                  "raw_tracking_from_existing_verified_report": previous.get("raw_tracking"),
                  "closest_same_scenario_train_greeting": {"source_id": str(train["source_id"][pair]),
                      "signer": str(train["signer"][pair]), "candidate_n": len(candidates),
                      "normalized_mse": float(pair_distances.min()),
                      "mean_resampled_presence_left_right_pose_face": train["x"][pair, :, 220:224].mean(0).tolist()}, "models": {}}
        for name, model in models.items():
            z = normalize(val["x"][i:i + 1], model)[0].astype(np.float64)
            centers = model["centroids"].reshape(4, 48, 224).astype(np.float64)
            geia_error, other_error = (z - centers[0]) ** 2, (z - centers[3]) ** 2
            block = {b: {"greeting_distance_contribution": float(geia_error[:, start:end].sum() / (48 * 224)),
                          "greeting_minus_other_logit_margin_contribution": float((other_error - geia_error)[:, start:end].sum() / (48 * 224))}
                     for b, (start, end) in BLOCKS.items()}
            margin = float((other_error - geia_error).mean())
            max_sum_error = max(max_sum_error, abs(sum(x["greeting_minus_other_logit_margin_contribution"] for x in block.values()) - margin))
            s = scored[name]
            max_prior_error = max(max_prior_error, abs(float(s["confidence"][i]) - previous["models"][name]["confidence"]))
            record["models"][name] = {"predicted_known_label": str(val["classes"][s["predicted"][i]]),
                "accepted": bool(s["accepted"][i]), "classification_score": float(s["confidence"][i]),
                "confidence_threshold": model["config"]["rejection"]["confidenceThreshold"],
                "rejection_reasons": [r for r, mask in s["reasons"].items() if mask[i]],
                "greeting_minus_other_logit_margin": margin, "block_contributions": block}
        rows.append(record)
    assert max_sum_error < 1e-12 and max_prior_error < 1e-6
    by_signer = {}
    for name, d in (("train", train), ("val", val)):
        for signer in np.unique(d["signer"]):
            mask = (d["signer"] == signer) & (d["y"] == 0)
            presence = d["x"][mask, :, 220:224].mean(1)
            by_signer[str(signer)] = {"role": name, "greeting_n": int(mask.sum()),
                "left_mean_presence": description(presence[:, 0]), "right_mean_presence": description(presence[:, 1]),
                "legacy_quality": description(d["quality"][mask])}
    summary = {name: {"greeting_n": len(rows),
        "known_argmax_correct_n": sum(r["models"][name]["predicted_known_label"] == val["classes"][0] for r in rows),
        "accepted_n": sum(r["models"][name]["accepted"] for r in rows),
        "classification_score": description([r["models"][name]["classification_score"] for r in rows]),
        "mean_greeting_vs_other_block_margin": {
            b: float(np.mean([r["models"][name]["block_contributions"][b]["greeting_minus_other_logit_margin_contribution"] for r in rows]))
            for b in BLOCKS},
        "mean_greeting_distance_block_contribution": {
            b: float(np.mean([r["models"][name]["block_contributions"][b]["greeting_distance_contribution"] for r in rows]))
            for b in BLOCKS}}
        for name in models}
    return {"rows": rows, "summary": summary, "presence_by_signer": by_signer,
            "checks": {"block_sum_max_error": max_sum_error, "existing_v4_score_max_error": max_prior_error},
            "pairing_limit": "Nearest same-label/same-scenario training row is a descriptive comparator selected with labels; no restoration, training, causal estimate, or deployable correction."}


def main():
    started = time.perf_counter()
    before = {p: sha(ROOT / p) for p in INPUTS}
    train, val = load_development("train"), load_development("val")
    assert len(train["y"]) == 338 and len(val["y"]) == 86
    assert not set(train["source_id"]) & set(val["source_id"])
    models = {"frozen_v2_centroid": load_reference(INPUTS[2], INPUTS[3]),
              "augmented_pool": load_reference(INPUTS[4], INPUTS[5])}
    for key in ("mean", "std", "keep"):
        assert np.array_equal(models["frozen_v2_centroid"][key], models["augmented_pool"][key])
    assert np.array_equal(models["frozen_v2_centroid"]["centroids"][:3], models["augmented_pool"]["centroids"][:3])
    prior = json.loads((ROOT / INPUTS[-1]).read_text(encoding="utf-8"))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "role": "Descriptive diagnostic on previously exposed development data only; no selection or release decision",
        "data": {"train": counts(train), "validation": counts(val), "final_test_read_or_inferred": False},
        "frozen_centroid_validation_ablations": frozen_ablations(val, models["frozen_v2_centroid"]),
        "static_feature_controls": feature_controls(train, val, models["frozen_v2_centroid"]),
        "scenario_metadata_control": scenario_control(train, val),
        "paired_validation_greeting_failures": paired_greetings(train, val, models, prior),
        "method_notes": [
            "Only the seven explicitly hashed inputs are read. No final tensor, final outcomes report, or RGB frames are opened.",
            "Ablations use frozen original centroid parameters and thresholds. Dropout zeros all84 hand coordinates and its presence flag. Legacy quality is retained; this is scientific scoring, not the V4 raw-guard product path.",
            "Reverse/shuffle/static/dropout perturbations are distribution shifts. Sensitivity or insensitivity is not causal proof of a linguistic shortcut.",
            "Feature-view controls refit only four means from338 training rows, using frozen train-only normalization. Reduced views use forced nearest-centroid choices without acceptance calibration, so their errors are not deployed false-acceptance rates.",
            "Static means of zero-filled local hand features also encode missingness. Body/pose/face views can contain genuine linguistic information; they are not pure nuisance variables.",
            "Per-person presence contrasts measure a signer-associated tracking difference. No signer-ID probe or causal identity-dependence estimate was run.",
            "Classifier inputs contain no RGB background. Scenario-only label association cannot establish a background shortcut; tracking may still depend indirectly on capture conditions.",
            "Raw missing counts are inherited from the verified V4 development-only report. A resampled mean presence fraction is not raw frame completeness or anatomical tracking accuracy.",
            "Only one validation person is present; all development data and the added negative categories have been examined before. These are not new generalization results.",
            "No thresholds, weights, frozen feature schema, historical reports, or production files are changed."],
        "input_sha256": before, "script_sha256": sha(__file__)}
    after = {p: sha(ROOT / p) for p in INPUTS}
    assert before == after
    report["all_inputs_unchanged"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    output = ROOT / "reports/v5/diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output),
                      "elapsed_seconds": report["elapsed_seconds"], "train_n": 338, "val_n": 86,
                      "greeting_n": 11, "final_test_read_or_inferred": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
