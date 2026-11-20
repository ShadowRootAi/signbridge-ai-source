"""Small adaptive-development experiment: pooled vs separate negative prototypes.

No CNN training. No new final-test claim. Each command has a separate phase:
lock -> pack -> fit (train/validation only) -> regression (exposed V2 test).
The existing known prototypes and normalization stay byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import FeatureDataset
from openset_core import metrics, motion_metrics, score_gate
from openset_train import load_split, tune_candidate
from preprocess import preprocess
from train import normalize_numpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v3-centroid"
DATA = ROOT / "data/v3"
REPORT = ROOT / "reports/v3"
V2 = ROOT / "artifacts/open-set-v2"
FIELDS = ("x", "y", "signer", "source_id", "split", "raw_label", "quality")
PROTOCOL = {
    "id": "signbridge-v3-adaptive-negative-prototypes-20261108",
    "status": "development_experiment_not_independent_confirmation",
    "no_large_training": True,
    "methods": ["recalibrated_original_pool", "augmented_pool", "separate_negative_prototypes"],
    "reference": "frozen_V2_nearest_centroid_and_config",
    "features": "unchanged 48x224 XY representation; frozen V2 train normalization and known centroids",
    "targeted_additions": "75 unused clips:20train+5val each KATALAVAINO,NAI,HMERA_TREIS; labels chosen after observing V2 final errors",
    "train_people": [1, 2, 4, 5], "validation_people": [6],
    "original_negative_categories": "retain all original V2 development negatives",
    "negative_centers": "augmented_pool uses one sample-weighted mean; separate uses one mean per training negative label",
    "distance": "mean squared Euclidean distance across the complete normalized 48x224 sequence",
    "separate_collapse": "background logit is minus minimum distance to any negative prototype; known logits unchanged; no summing softmax mass across additional classes",
    "selection": "same V2 validation-only temperature/confidence/quality/motion grid and eligibility; no test fitting; no automatic deployment",
    "regression": "V2 final129 previously exposed clips, separately invoked only after fit is locked; NOT a fresh final set",
    "unknown_generalization_limit": "added categories may appear across train/val on separate people; original val-only OXI/KALO remain separate and reported; no fresh unknown-class final claim",
    "webcam": "no human webcam recordings available",
    "no_eligible_candidate": "report failure, do not relax constraints",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def lock():
    files = [Path(__file__), ROOT / "ml/openset_train.py", ROOT / "ml/openset_core.py", ROOT / "ml/openset_protocol.py",
             ROOT / "ml/train.py", ROOT / "ml/common.py", ROOT / "ml/preprocess.py",
             ROOT / "data/open-set-v2/train.npz", ROOT / "data/open-set-v2/val.npz",
             DATA / "targeted-development.plan.locked.json", DATA / "manifest.json",
             V2 / "reference-centroid.npz", V2 / "reference-centroid-config.json"]
    document = {"protocol": PROTOCOL, "file_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files}}
    path = OUT / "protocol.locked.json"
    if path.exists():
        if read(path) != document:
            raise ValueError("Refusing to replace a locked experiment protocol")
    else:
        write(path, document)
    return {"protocol_sha256": sha(path), "no_fitting_performed": True}


def check_lock():
    doc = read(OUT / "protocol.locked.json")
    if doc["protocol"] != PROTOCOL:
        raise ValueError("Protocol changed after lock")
    for name, value in doc["file_hashes"].items():
        if sha(ROOT / name) != value:
            raise ValueError(f"Locked code or source artifact changed: {name}")


def combine(base, extra, role):
    values = {name: np.concatenate([getattr(base, name), extra[name]]) for name in FIELDS}
    if len(set(values["source_id"])) != len(values["source_id"]):
        raise ValueError("Duplicate clip in development split")
    if set(values["split"]) != {role}:
        raise ValueError("Split mismatch")
    return FeatureDataset(**values, classes=base.classes, metadata={**base.metadata, "v3_adaptive_development": True})


def pack(args):
    check_lock()
    if args.manifest.resolve() != (DATA / "manifest.json").resolve() or args.plan.resolve() != (DATA / "targeted-development.plan.locked.json").resolve():
        raise ValueError("Only the data plan and manifest bound into the algorithm lock may be packed")
    manifest = read(args.manifest); plan = read(args.plan)
    samples = manifest["samples"]; planned = {s["id"]: s for s in plan["samples"]}
    if len(samples) != 75 or len(planned) != 75 or {s["id"] for s in samples} != set(planned):
        raise ValueError("Expected exactly75 locked additional development clips")
    old_samples = read(ROOT / "data/manifest.json")["samples"] + read(ROOT / "data/open-set-v2/manifest.json")["samples"]
    old_sources = {s["source_path"] for s in old_samples}
    old_hashes = {h for s in old_samples for h in s["frame_sha256"]}
    feature_hash = sha(ROOT / "web/src/features.ts")
    records = {}
    for path in args.jsonl:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            row = json.loads(line)
            if row["id"] not in planned:
                raise ValueError("Extraction contains an unplanned clip")
            if row["id"] in records or row["feature_sha256"] != feature_hash:
                raise ValueError("Duplicate extraction or changed feature representation")
            records[row["id"]] = row
    if set(records) != set(planned):
        raise ValueError("Not all locked clips were extracted")
    parity = []; frame_splits = {}; values = {name: [] for name in FIELDS}; pipelines = set()
    for sample in samples:
        for key in ("source_path", "class_id", "signer_id", "split", "raw_label"):
            if sample[key] != planned[sample["id"]][key]:
                raise ValueError(f"Locked metadata changed: {key}")
        if sample["source_path"] in old_sources or sample["class_id"] != -1:
            raise ValueError("Extra clip is already exposed or unexpectedly known")
        role = sample["split"]; person = int(sample["signer_id"])
        if role not in ("train", "val") or person not in ({1, 2, 4, 5} if role == "train" else {6}):
            raise ValueError("Forbidden person or split")
        row = records[sample["id"]]; x = np.asarray(row["x"], dtype=np.float32)
        if x.shape != (48, 224) or not np.isfinite(x).all():
            raise ValueError("Invalid feature tensor")
        if row["frames"] != len(sample["frame_paths"]) or len(sample["frame_sha256"]) != row["frames"]:
            raise ValueError("Source frame count mismatch")
        for filename, expected in zip(sample["frame_paths"], sample["frame_sha256"]):
            if sha(ROOT / filename) != expected or expected in old_hashes:
                raise ValueError("Source image changed or duplicates exposed data")
            if frame_splits.setdefault(expected, role) != role:
                raise ValueError("Identical source frame crosses splits")
        if row.get("raw"):
            actual, quality = preprocess(row["raw"])
            np.testing.assert_allclose(actual, x, atol=1e-6, rtol=1e-6)
            if abs(quality - row["quality"]) > 1e-10:
                raise ValueError("Python/browser tracking quality mismatch")
            parity.append({"id": sample["id"], "max_abs_error": float(np.abs(actual-x).max())})
        pipelines.add(row["pipeline_sha256"])
        for key, value in {"x": x, "y": -1, "signer": str(person), "source_id": sample["source_path"],
                           "split": role, "raw_label": sample["raw_label"], "quality": row["quality"]}.items():
            values[key].append(value)
    values = {k: np.asarray(v, dtype=np.float32 if k in ("x", "quality") else np.int64 if k == "y" else str) for k, v in values.items()}
    if Counter(values["split"]) != {"train": 60, "val": 15}:
        raise ValueError("Unexpected addition split counts")
    outputs = {}
    for role in ("train", "val"):
        base = load_split(ROOT / f"data/open-set-v2/{role}.npz", role)
        mask = values["split"] == role
        data = combine(base, {key: value[mask] for key, value in values.items()}, role)
        path = DATA / f"{role}.npz"
        if path.exists():
            raise ValueError("Refusing to overwrite packed V3 features")
        np.savez_compressed(path, **{name: getattr(data, name) for name in FIELDS}, classes=np.asarray(data.classes),
                            metadata_json=np.asarray(json.dumps(data.metadata, ensure_ascii=False)))
        outputs[role] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "n": len(data.y),
                         "known_n": int((data.y>=0).sum()), "unknown_n": int((data.y<0).sum()),
                         "people": sorted(set(data.signer)), "category_counts": dict(Counter(data.raw_label))}
    audit = {"outputs": outputs, "source_manifest_sha256": sha(args.manifest), "data_plan_sha256": sha(args.plan),
             "feature_sha256": feature_hash, "pipeline_hashes": sorted(pipelines), "preprocessing_parity": parity,
             "all75_sources_and_frames_unused": True, "cross_split_frame_duplicates": 0,
             "new_final_test_created": False, "adaptive_categories_exposed_in_V2": True}
    write(DATA / "packing-audit.json", audit)
    return audit


def load_centers(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name].copy() for name in f.files}


def logits_for(x, model):
    z = normalize_numpy(x, model["mean"], model["std"], model["keep"]).reshape(len(x), -1)
    distances = np.stack([((z-center)**2).mean(axis=1) for center in model["centroids"]], axis=1)
    return np.concatenate([-distances[:, :3], -distances[:, 3:].min(axis=1, keepdims=True)], axis=1)


def evaluate(data, model, config):
    started = time.perf_counter()
    logits = logits_for(data.x, model)
    scored = score_gate(logits, data.quality, motion_metrics(data.x)["motionBodyRms"], config)
    elapsed = time.perf_counter() - started
    result = metrics(data.y, scored, data.classes, data.signer, data.raw_label)
    result["cpu_batch_classification_ms_per_clip"] = elapsed * 1000 / len(data.y)
    result["latency_scope"] = "Python features-to-decision only; excludes extraction; not browser latency"
    result["open_set_decision_accuracy"] = float(((data.y>=0) & scored["accepted"] & (data.y==scored["predicted"]) | ((data.y<0) & ~scored["accepted"])).mean())
    rows = [{"id": str(data.source_id[i]), "person": str(data.signer[i]), "label": str(data.raw_label[i]),
             "known": bool(data.y[i]>=0), "predicted": data.classes[int(scored["predicted"][i])],
             "accepted": bool(scored["accepted"][i]), "confidence": float(scored["confidence"][i])} for i in range(len(data.y))]
    return {"metrics": result, "predictions": rows}


def fit():
    check_lock()
    if (OUT / "fit.locked.json").exists():
        raise ValueError("Development fit already locked; no silent refit")
    audit = read(DATA / "packing-audit.json")
    for role in ("train", "val"):
        if sha(DATA / f"{role}.npz") != audit["outputs"][role]["sha256"]:
            raise ValueError("Packed split changed")
    train, val = (load_split(DATA / f"{role}.npz", role) for role in ("train", "val"))
    if set(train.source_id) & set(val.source_id) or set(train.signer) & set(val.signer):
        raise ValueError("Development source/person leakage")
    original = load_centers(V2 / "reference-centroid.npz")
    z = normalize_numpy(train.x, original["mean"], original["std"], original["keep"]).reshape(len(train.y), -1)
    labels = sorted(set(train.raw_label[train.y<0]))
    model_pool = {**original, "centroids": np.concatenate([original["centroids"][:3], z[train.y<0].mean(axis=0, keepdims=True)])}
    model_separate = {**original, "centroids": np.concatenate([original["centroids"][:3], np.stack([z[(train.y<0)&(train.raw_label==label)].mean(axis=0) for label in labels])])}
    models = {"recalibrated_original_pool": original, "augmented_pool": model_pool, "separate_negative_prototypes": model_separate}
    results = {}; locked = {}; selected = None; best = None
    train_motion = motion_metrics(train.x)["motionBodyRms"]
    for name in PROTOCOL["methods"]:
        model = models[name]
        config, validation, score, search = tune_candidate(name, logits_for(val.x, model), None, train, val, train_motion)
        config["modelVersion"] = "signbridge-v3-development-" + name
        config["developmentOnly"] = True
        path = OUT / name; path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "centroids.npz", **model)
        write(path / "config.json", config)
        write(path / "validation-grid.json", search)
        results[name] = {"validation": evaluate(val, model, config), "training_in_sample_only": evaluate(train, model, config),
                         "negative_labels": labels if name == "separate_negative_prototypes" else ["pooled_OTHER"],
                         "eligible": score is not None}
        locked[name] = {"model_sha256": sha(path / "centroids.npz"), "config_sha256": sha(path / "config.json")}
        if score is not None and (best is None or score > best):
            best, selected = score, name
    reference = evaluate(val, original, read(V2 / "reference-centroid-config.json"))
    report = {"protocol_sha256": sha(OUT / "protocol.locked.json"), "packing_audit_sha256": sha(DATA / "packing-audit.json"),
              "selection_uses_old_test": False, "selection_is_independent_of_old_test_design": False,
              "interpretation": "Categories and method chosen adaptively after V2 test; thresholds fitted only on person6; old test was not opened by this fit command.",
              "selected_development_candidate": selected, "candidates": locked, "deployment_changed": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write(REPORT / "centroid-development-results.json", {"fit": report, "methods": results, "frozen_reference_validation": reference})
    write(OUT / "fit.locked.json", report)
    return {"selected_development_candidate": selected, "validation": {name: item["validation"]["metrics"] for name, item in results.items()}}


def regression():
    check_lock(); fitted = read(OUT / "fit.locked.json")
    path = REPORT / "centroid-exposed-regression.json"
    if path.exists():
        raise ValueError("Regression already saved; refusing silent repeated evaluation")
    data = load_split(ROOT / "data/open-set-v2/final_test.npz", "test")
    results = {}
    for name, hashes in fitted["candidates"].items():
        directory = OUT / name
        if sha(directory / "centroids.npz") != hashes["model_sha256"] or sha(directory / "config.json") != hashes["config_sha256"]:
            raise ValueError("Candidate changed after fit lock")
        results[name] = evaluate(data, load_centers(directory / "centroids.npz"), read(directory / "config.json"))
    results["frozen_reference"] = evaluate(data, load_centers(V2 / "reference-centroid.npz"), read(V2 / "reference-centroid-config.json"))
    report = {"evaluation_role": "exposed_V2_regression_NOT_fresh_final_test", "sample_n": len(data.y),
              "fit_lock_sha256": sha(OUT / "fit.locked.json"), "data_sha256": sha(ROOT / "data/open-set-v2/final_test.npz"),
              "deployment_changed": False, "methods": results,
              "warning": "Any apparent gain is adaptive, descriptive evidence. It cannot confirm improvement on new data or choose a final production model."}
    write(path, report)
    return {name: value["metrics"] for name, value in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("lock", "fit", "regression"):
        commands.add_parser(name)
    p = commands.add_parser("pack"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--jsonl", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    result = pack(args) if args.command == "pack" else {"lock": lock, "fit": fit, "regression": regression}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
