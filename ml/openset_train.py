"""Fit on separate development files; freeze selection before explicit final evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from common import FeatureDataset, classification_metrics, fit_temperature
from openset_core import eligibility_score, fit_prototypes, metrics, motion_metrics, score_gate
from openset_diagnose import infer_embeddings, load_frozen
from openset_protocol import PROTOCOL
from train import centroid_logits, normalize_numpy, seed_everything, train_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_split(path: Path, role: str) -> FeatureDataset:
    with np.load(path, allow_pickle=False) as f:
        data = FeatureDataset(np.asarray(f["x"], dtype=np.float32), np.asarray(f["y"], dtype=np.int64),
                              f["signer"].astype(str), f["source_id"].astype(str), f["split"].astype(str),
                              f["classes"].astype(str).tolist(), f["raw_label"].astype(str),
                              np.asarray(f["quality"], dtype=np.float32), json.loads(str(f["metadata_json"].item())))
    if set(data.split) != {role}:
        raise ValueError(f"Fitting/evaluation expected a separate {role} NPZ, got {set(data.split)}")
    expected_signers = {"train": {"1", "2", "4", "5"}, "val": {"6"}, "test": {"3", "7"}}[role]
    if not set(data.signer).issubset(expected_signers):
        raise ValueError(f"Forbidden signer in {role}")
    if data.x.shape != (len(data.y), 48, 224) or not np.isfinite(data.x).all():
        raise ValueError("Invalid feature tensors")
    if not np.isfinite(data.quality).all() or np.any((data.quality < 0) | (data.quality > 1)):
        raise ValueError("Invalid tracking quality")
    if set(data.y[data.y >= 0]) != set(range(len(data.classes))) or not np.any(data.y == -1):
        raise ValueError("Every split must contain all known classes and unsupported examples")
    return data


def development_union(train, val):
    if train.classes != val.classes:
        raise ValueError("Class order differs")
    if set(train.source_id) & set(val.source_id) or set(train.signer) & set(val.signer):
        raise ValueError("Development source/signer leakage")
    if set(train.raw_label[train.y < 0]) & set(val.raw_label[val.y < 0]):
        raise ValueError("Unknown semantic categories cross development splits")
    return FeatureDataset(*(np.concatenate([getattr(train, field), getattr(val, field)]) for field in ("x", "y", "signer", "source_id", "split")),
                          train.classes, np.concatenate([train.raw_label, val.raw_label]),
                          np.concatenate([train.quality, val.quality]), train.metadata)


def tune_candidate(name, logits, embeddings, train, val, training_motion, prototype=None):
    k = len(val.classes)
    temperature = fit_temperature(logits, np.where(val.y < 0, k, val.y))
    val_motion = motion_metrics(val.x)["motionBodyRms"]
    motion_choices = [("disabled", 0.0), ("train_known_q01", float(np.quantile(training_motion[train.y >= 0], .01))),
                      ("train_known_q05", float(np.quantile(training_motion[train.y >= 0], .05)))]
    distance_choices = PROTOCOL["prototype"]["threshold_quantiles_percent"] if prototype else ["disabled"]
    best, best_config, best_metrics = None, None, None
    search = []
    for confidence in PROTOCOL["confidence_grid"]:
        for quality in PROTOCOL["quality_grid"]:
            for motion_name, motion_cutoff in motion_choices:
                for quantile in distance_choices:
                    config = {"modelVersion": f"signbridge-open-set-v2-{name}", "candidate": name,
                              "labels": val.classes, "backgroundIndex": k,
                              "featureSchema": train.metadata["featureSchema"], "T": 48, "F": 224,
                              "input": {"name": "landmarks", "shape": [1, 48, 224], "dtype": "float32"},
                              "output": {"name": "logits", "embeddingName": "embedding", "labels": val.classes + ["__background__"]},
                              "rejection": {"temperature": temperature, "confidenceThreshold": confidence,
                                            "minQuality": quality, "qualityFloor": .25, "minMotionBodyRms": motion_cutoff,
                                            "motionSelection": motion_name, "calibrated": True, "validationOnly": True,
                                            "calibrationCount": len(val.y), "rejectBackground": True}}
                    if prototype:
                        config["prototypeGate"] = {key: value for key, value in prototype.items() if key != "trainLeaveOneOutDistances"}
                        config["prototypeGate"]["classDistanceCutoffs"] = None if quantile == "disabled" else [float(np.quantile(d, quantile / 100)) for d in prototype["trainLeaveOneOutDistances"]]
                        config["prototypeGate"]["trainingDistanceQuantile"] = quantile
                    scored = score_gate(logits, val.quality, val_motion, config, embeddings)
                    report = classification_metrics(val.y, scored["predicted"], scored["accepted"], k)
                    score = eligibility_score(report)
                    search.append({"confidence": confidence, "quality": quality, "motion": motion_name, "distanceQuantile": quantile,
                                   "eligible": score is not None, "unknownFAR": report["unknown_false_acceptance_rate"],
                                   "knownCoverage": report["known_coverage"], "acceptedKnownAccuracy": report["accepted_known_accuracy"],
                                   "correctKnownAcceptance": report["correct_known_acceptance_rate"]})
                    if score is not None and (best is None or score > best):
                        best, best_config = score, config
                        best_metrics = metrics(val.y, scored, val.classes, val.signer, val.raw_label)
    if best_config is None:
        # A predetermined, least restrictive grid setting is descriptive only and cannot be selected.
        best_config = {"modelVersion": f"signbridge-open-set-v2-{name}", "candidate": name, "labels": val.classes,
                       "backgroundIndex": k, "featureSchema": train.metadata["featureSchema"], "T": 48, "F": 224,
                       "input": {"name": "landmarks", "shape": [1, 48, 224], "dtype": "float32"},
                       "output": {"name": "logits", "embeddingName": "embedding", "labels": val.classes + ["__background__"]},
                       "rejection": {"temperature": temperature, "confidenceThreshold": 0.0, "minQuality": .25,
                                     "qualityFloor": .25, "minMotionBodyRms": 0.0, "calibrated": True, "validationOnly": True,
                                     "calibrationCount": len(val.y), "rejectBackground": True},
                       "notEligibleForSelection": True, "diagnosticFallback": "least restrictive prespecified gate; no constraint relaxation for selection"}
        if prototype:
            best_config["prototypeGate"] = {key: value for key, value in prototype.items() if key != "trainLeaveOneOutDistances"}
            best_config["prototypeGate"]["classDistanceCutoffs"] = None
        best_metrics = metrics(val.y, score_gate(logits, val.quality, val_motion, best_config, embeddings), val.classes, val.signer, val.raw_label)
    return best_config, best_metrics, best, search


class DualOutput(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, landmarks):
        m = self.model
        normalized = torch.clamp((landmarks - m.feature_mean) / m.feature_std, -8, 8) * m.feature_keep
        embedding = m.temporal(normalized.transpose(1, 2)).mean(dim=2)
        return m.classifier(embedding), embedding


def export_candidate(model, config, val, directory):
    import onnx
    import onnxruntime as ort
    directory.mkdir(parents=True, exist_ok=True)
    model.eval(); wrapped = DualOutput(model).eval()
    path = directory / "model.onnx"
    torch.onnx.export(wrapped, (torch.from_numpy(val.x[:1]),), str(path), input_names=["landmarks"],
                      output_names=["logits", "embedding"], dynamo=True, verify=True, external_data=False,
                      opset_version=18, verbose=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    indices = np.asarray([int(np.flatnonzero(val.raw_label == label)[0]) for label in np.unique(val.raw_label)])
    reference_logits, reference_embeddings = infer_embeddings(model, val.x[indices])
    runs = [session.run(None, {"landmarks": val.x[i:i + 1]}) for i in indices]
    actual_logits = np.concatenate([run[0] for run in runs]); actual_embeddings = np.concatenate([run[1] for run in runs])
    np.testing.assert_allclose(actual_logits, reference_logits, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(actual_embeddings, reference_embeddings, atol=1e-5, rtol=1e-4)
    motion = motion_metrics(val.x[indices])
    scored = score_gate(actual_logits, val.quality[indices], motion["motionBodyRms"], config, actual_embeddings)
    fixtures = []
    for j, i in enumerate(indices):
        fixtures.append({"id": str(val.source_id[i]), "x": val.x[i].tolist(), "quality": float(val.quality[i]),
                         "logits": actual_logits[j].tolist(), "embedding": actual_embeddings[j].tolist(),
                         "diagnostics": {key: value[j].tolist() for key, value in motion.items()},
                         "accepted": bool(scored["accepted"][j]), "predicted": int(scored["predicted"][j]),
                         "confidence": float(scored["confidence"][j]),
                         "reasons": [reason for reason, mask in scored["reasons"].items() if mask[j]]})
    write_json(directory / "config.json", config)
    write_json(directory / "gate-parity-fixtures.json", {"cases": fixtures, "numericAtol": 1e-6, "numericRtol": 1e-5,
                                                         "decisionAgreementRequired": True, "source": "validation only"})
    torch.save({"state_dict": model.state_dict(), "config": config, "channels": 48}, directory / "model.pt")
    return {"model_sha256": digest(path), "config_sha256": digest(directory / "config.json"),
            "onnx_bytes": path.stat().st_size, "parity_cases": len(indices),
            "max_logit_error": float(np.abs(actual_logits - reference_logits).max()),
            "max_embedding_error": float(np.abs(actual_embeddings - reference_embeddings).max())}


def fit(args):
    output = args.output
    if (output / "selection.locked.json").exists():
        raise ValueError("Selection is already locked; refusing to overwrite")
    protocol = json.loads((output / "algorithm-protocol.json").read_text(encoding="utf-8"))
    if protocol["protocol"] != PROTOCOL:
        raise ValueError("Code protocol differs from frozen algorithm protocol")
    train, val = load_split(args.train, "train"), load_split(args.validation, "val")
    if train.metadata["dataset_plan_sha256"] != protocol["dataset_plan_sha256"] or val.metadata["dataset_plan_sha256"] != protocol["dataset_plan_sha256"]:
        raise ValueError("Development features do not match frozen plan")
    development = development_union(train, val)
    old_model, old_config = load_frozen(args.frozen)
    seed_everything(PROTOCOL["seed"])
    parameters = PROTOCOL["training_B"]
    training_args = SimpleNamespace(seed=PROTOCOL["seed"], epochs=parameters["epochs"], patience=parameters["patience"],
                                   batch_size=parameters["batch_size"], channels=48, learning_rate=parameters["learning_rate"])
    new_model, training_report, mean, std, keep = train_model(development, training_args)
    old_train_logits, old_train_embeddings = infer_embeddings(old_model, train.x)
    old_val_logits, old_val_embeddings = infer_embeddings(old_model, val.x)
    new_val_logits, new_val_embeddings = infer_embeddings(new_model, val.x)
    prototype_error = None
    try:
        prototype = fit_prototypes(old_train_embeddings, train.y, train.classes)
    except ValueError as error:
        prototype, prototype_error = None, str(error)
    train_motion = motion_metrics(train.x)["motionBodyRms"]
    candidates = [(PROTOCOL["candidates"][0], old_model, old_val_logits, old_val_embeddings, None),
                  (PROTOCOL["candidates"][1], new_model, new_val_logits, new_val_embeddings, None),
                  (PROTOCOL["candidates"][2], old_model, old_val_logits, old_val_embeddings, prototype)]
    results, winner, winner_score = {}, None, None
    for name, model, logits, embeddings, proto in candidates:
        if name == "C_frozen_embedding_prototype" and prototype_error is not None:
            results[name] = {"eligible": False, "invalid_reason": prototype_error}
            continue
        config, validation_report, score, search = tune_candidate(name, logits, embeddings, train, val, train_motion, proto)
        config["protocolId"] = PROTOCOL["id"]
        config["trainingPlanSha256"] = protocol["dataset_plan_sha256"]
        config["normalization"] = "embedded in ONNX; supply raw preprocessed features"
        folder = output / name
        exported = export_candidate(model, config, val, folder)
        write_json(folder / "validation-grid.json", search)
        write_json(folder / "validation-report.json", validation_report)
        results[name] = {"eligible": score is not None, "selection_score": list(score) if score else None,
                         "validation": validation_report, "export": exported,
                         "config_path": str((folder / "config.json").resolve())}
        if score is not None and (winner_score is None or score > winner_score):
            winner, winner_score = name, score
    # Reference models are reported but cannot win selection.
    old_reference = metrics(val.y, score_gate(old_val_logits, val.quality, motion_metrics(val.x)["motionBodyRms"], old_config), val.classes, val.signer, val.raw_label)
    normalized = normalize_numpy(development.x, mean, std, keep).reshape(len(development.y), -1)
    tmask = development.mask("train"); targets = np.where(development.y < 0, len(train.classes), development.y)
    centers = np.stack([normalized[tmask & (targets == i)].mean(axis=0) for i in range(len(train.classes) + 1)])
    baseline_logits = -np.stack([((normalized[~tmask] - center) ** 2).mean(axis=1) for center in centers], axis=1)
    base_config, base_validation, _, base_grid = tune_candidate("reference_nearest_centroid", baseline_logits, None, train, val, train_motion)
    np.savez_compressed(output / "reference-centroid.npz", mean=mean, std=std, keep=keep, centroids=centers)
    write_json(output / "reference-centroid-config.json", base_config)
    write_json(output / "training-B.json", training_report)
    write_json(output / "reference-validation.json", {"original_frozen_tcn": old_reference, "nearest_centroid": base_validation})
    packing = json.loads(args.packing_audit.read_text(encoding="utf-8"))
    selection = {"protocolId": PROTOCOL["id"], "algorithm_protocol_sha256": digest(output / "algorithm-protocol.json"),
                 "gate_code_sha256": digest(Path(__file__).with_name("openset_core.py")),
                 "dataset_plan_sha256": protocol["dataset_plan_sha256"],
                 "development_hashes": {"train": digest(args.train), "validation": digest(args.validation)},
                 "sealed_final_test_sha256": packing["outputs"]["final_test"]["sha256"],
                 "selected_candidate": winner, "status": "eligible_candidate_selected" if winner else "no_candidate_met_predeclared_constraints",
                 "selection_uses_final_test": False, "candidates": results,
                 "references_not_selected": {"original_frozen_tcn_validation": old_reference, "nearest_centroid_validation": base_validation}}
    write_json(output / "selection.locked.json", selection)
    print(json.dumps({"selection": str((output / "selection.locked.json").resolve()), "selection_sha256": digest(output / "selection.locked.json"),
                      "winner": winner, "validation": {name: ({"eligible": row["eligible"], "knownCoverage": row["validation"]["known_coverage"],
                                                               "acceptedKnownAccuracy": row["validation"]["accepted_known_accuracy"],
                                                               "unknownFAR": row["validation"]["unknown_false_acceptance_rate"]} if "validation" in row else row)
                                                         for name, row in results.items()}}, ensure_ascii=False, indent=2))


def evaluate(args):
    import onnxruntime as ort
    output = args.output
    if (output / "final-evaluation.json").exists():
        raise ValueError("Final evaluation already exists; refusing an accidental repeated look")
    selection_path = output / "selection.locked.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if digest(output / "algorithm-protocol.json") != selection["algorithm_protocol_sha256"]:
        raise ValueError("Frozen algorithm protocol changed after selection")
    if digest(Path(__file__).with_name("openset_core.py")) != selection["gate_code_sha256"]:
        raise ValueError("Gate mathematics changed after selection")
    if digest(args.test) != selection["sealed_final_test_sha256"]:
        raise ValueError("Final tensors changed since validation selection")
    test = load_split(args.test, "test")
    motion = motion_metrics(test.x)["motionBodyRms"]
    results = {}
    for name in PROTOCOL["candidates"]:
        if "invalid_reason" in selection["candidates"][name]:
            results[name] = {"status": "candidate_invalid_before_final_evaluation", "reason": selection["candidates"][name]["invalid_reason"]}
            continue
        directory = output / name
        if digest(directory / "model.onnx") != selection["candidates"][name]["export"]["model_sha256"] or digest(directory / "config.json") != selection["candidates"][name]["export"]["config_sha256"]:
            raise ValueError("Candidate changed after validation selection")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
        session.run(None, {"landmarks": np.zeros((1, 48, 224), dtype=np.float32)})  # Runtime warmup only; not an accuracy sample.
        runs, timings = [], []
        for sample in test.x:
            start = time.perf_counter()
            runs.append(session.run(["logits", "embedding"], {"landmarks": sample[None]}))
            timings.append((time.perf_counter() - start) * 1000)
        logits, embeddings = np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs])
        scored = score_gate(logits, test.quality, motion, config, embeddings)
        results[name] = metrics(test.y, scored, test.classes, test.signer, test.raw_label)
        results[name]["classifier_cpu_latency_ms"] = {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)),
                                                       "n": len(timings), "excludes": "video decoding, landmark extraction, UI, gate calculation"}
        write_json(directory / "final-predictions.json", {"source_id": test.source_id.tolist(), "y": test.y.tolist(),
                                                          "raw_label": test.raw_label.tolist(), "predicted": scored["predicted"].tolist(),
                                                          "accepted": scored["accepted"].tolist(), "confidence": scored["confidence"].tolist(),
                                                          "reasons": {key: value.tolist() for key, value in scored["reasons"].items()}})
    old_model, old_config = load_frozen(args.frozen)
    old_logits, _ = infer_embeddings(old_model, test.x)
    reference_old = metrics(test.y, score_gate(old_logits, test.quality, motion, old_config), test.classes, test.signer, test.raw_label)
    with np.load(output / "reference-centroid.npz", allow_pickle=False) as f:
        normalized = normalize_numpy(test.x, f["mean"], f["std"], f["keep"]).reshape(len(test.y), -1)
        base_logits = -np.stack([((normalized - center) ** 2).mean(axis=1) for center in f["centroids"]], axis=1)
    base_config = json.loads((output / "reference-centroid-config.json").read_text(encoding="utf-8"))
    reference_baseline = metrics(test.y, score_gate(base_logits, test.quality, motion, base_config), test.classes, test.signer, test.raw_label)
    report = {"selection_sha256": digest(selection_path), "selected_before_final_test": selection["selected_candidate"],
              "final_features_sha256": digest(args.test), "selection_changed_after_test": False,
              "candidates": results, "references": {"original_frozen_tcn": reference_old, "nearest_centroid": reference_baseline},
              "identity_caveat": test.metadata["final_identity_caveat"],
              "human_webcam_performance": "separate observational recordings; not established by this corpus test"}
    write_json(output / "final-evaluation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--train", type=Path, required=True); p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--packing-audit", type=Path, required=True); p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--test", type=Path, required=True); p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(args) if args.command == "fit" else evaluate(args)
