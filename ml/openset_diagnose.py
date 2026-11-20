"""Descriptive diagnosis on already-seen POC data; never new final-test inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import classification_metrics, load_dataset, softmax
from openset_core import fit_prototypes, metrics, motion_metrics, prototype_distances, score_gate
from train import TemporalClassifier, normalize_numpy


def load_frozen(directory: Path):
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = TemporalClassifier(state["feature_mean"].numpy().ravel(), state["feature_std"].numpy().ravel(),
                               len(config["output"]["labels"]), checkpoint["channels"], state["feature_keep"].numpy().ravel())
    model.load_state_dict(state); model.eval()
    torch.set_num_threads(2)
    return model, config


def infer_embeddings(model, x):
    captured = []
    hook = model.temporal.register_forward_hook(lambda module, args, result: captured.append(result.detach().mean(dim=2).numpy()))
    try:
        with torch.inference_mode():
            logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()
    finally:
        hook.remove()
    return logits, captured[0]


def diagnose(frozen: Path, output: Path):
    data = load_dataset(frozen / "features.npz")
    model, config = load_frozen(frozen)
    x, y = data.x, data.y
    logits, embeddings = infer_embeddings(model, x)
    motion = motion_metrics(x)
    scored = score_gate(logits, data.quality, motion["motionBodyRms"], config)
    train, val, old_test = (data.mask(split) for split in ("train", "val", "test"))
    prototype = fit_prototypes(embeddings[train], y[train], data.classes)
    distances = prototype_distances(embeddings, prototype)
    probabilities = scored["probabilities"]
    rows = []
    raw_centers = np.stack([embeddings[train & (y == i)].mean(axis=0) for i in range(len(data.classes))]).astype(np.float64)
    cosine = (embeddings.astype(np.float64) @ raw_centers.T) / np.maximum(
        np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(raw_centers, axis=1)[None, :], 1e-12)
    for i in np.flatnonzero(old_test & (y == -1)):
        top = int(scored["predicted"][i]); other = probabilities[i].copy(); other[top] = -np.inf
        rows.append({"id": str(data.source_id[i]), "signer": str(data.signer[i]), "unsupported_label": str(data.raw_label[i]),
                     "predicted": data.classes[top], "accepted": bool(scored["accepted"][i]),
                     "logits": logits[i].tolist(), "temperature": config["rejection"]["temperature"],
                     "probabilities": probabilities[i].tolist(), "margin": float(probabilities[i, top] - other.max()),
                     "quality": float(data.quality[i]), "motionBodyRms": float(motion["motionBodyRms"][i]),
                     "motionLocalRms": float(motion["motionLocalRms"][i]),
                     "prototype_distances": dict(zip(data.classes, distances[i].tolist())),
                     "nearest_prototype": data.classes[int(distances[i].argmin())],
                     "raw_embedding_cosine_to_class": dict(zip(data.classes, cosine[i].tolist()))})
    variants = {"identity": x, "reverse_time": x[:, ::-1].copy(),
                "static_temporal_mean": np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1)}
    for seed in (20261108, 20261109, 20261110):
        variants[f"shuffle_time_seed_{seed}"] = x[:, np.random.default_rng(seed).permutation(48)].copy()
    # Neutral masking uses TRAIN means, not zeros with extreme standardized meaning.
    mean = model.feature_mean.numpy().ravel()
    for name, columns in (("mask_hands", list(range(168)) + [220, 221]),
                          ("mask_pose_face", list(range(168, 220)) + [222, 223]),
                          ("mask_face_only", list(range(186, 220)) + [223])):
        changed = x.copy(); changed[:, :, columns] = mean[columns]
        variants[name] = changed
    ablations = {}
    for name, changed in variants.items():
        changed_logits, _ = infer_embeddings(model, changed)
        # Original measured quality retained; these are model-input interventions, not new video captures.
        changed_scores = score_gate(changed_logits, data.quality, motion["motionBodyRms"], config)
        ablations[name] = {split: metrics(y[mask], {key: ({reason: flags[mask] for reason, flags in value.items()}
                                                if key == "reasons" else value[mask])
                                             for key, value in changed_scores.items()}, data.classes)
                           for split, mask in (("train", train), ("val", val), ("old_test_diagnostic", old_test))}
    baseline_views = {"all_time_ordered_features": x.reshape(len(x), -1),
                      "static_temporal_mean_only": x.mean(axis=1),
                      "temporal_std_only": x.std(axis=1),
                      "hands_only": x[:, :, :168].reshape(len(x), -1),
                      "pose_face_only_includes_pose_wrists": x[:, :, 168:220].reshape(len(x), -1),
                      "face_only": x[:, :, 186:220].reshape(len(x), -1),
                      "presence_flags_only": x[:, :, 220:224].reshape(len(x), -1),
                      "local_hand_shape_only": np.concatenate([x[:, :, 42:84], x[:, :, 126:168]], axis=2).reshape(len(x), -1)}
    baselines = {}
    for name, values in baseline_views.items():
        fit_values = values[train & (y >= 0)]
        center = fit_values.mean(axis=0); std = np.maximum(fit_values.std(axis=0), 0.01)
        active = fit_values.std(axis=0) >= 1e-6
        z = np.clip((values - center) / std, -8, 8) * active
        centers = np.stack([z[train & (y == i)].mean(axis=0) for i in range(len(data.classes))])
        predictions = np.argmin(np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        baselines[name] = {split: classification_metrics(y[mask & (y >= 0)], predictions[mask & (y >= 0)],
                                                        np.ones(int((mask & (y >= 0)).sum()), dtype=bool), len(data.classes))
                           for split, mask in (("train", train), ("val", val), ("old_test_diagnostic", old_test))}
    scenarios = np.asarray([source.split("_signer")[0] for source in data.source_id])
    majority = int(np.bincount(y[train & (y >= 0)], minlength=len(data.classes)).argmax())
    scenario_labels = {scenario: int(np.bincount(y[train & (y >= 0) & (scenarios == scenario)], minlength=len(data.classes)).argmax())
                       for scenario in np.unique(scenarios[train & (y >= 0)])}
    scenario_predictions = np.asarray([scenario_labels.get(scenario, majority) for scenario in scenarios])
    scenario_control = {"not_deployed_input": True, "rule": "training known majority label per source scenario; first-index ties, global-majority fallback",
                        "mapping": {scenario: data.classes[index] for scenario, index in scenario_labels.items()},
                        "old_test_known": classification_metrics(y[old_test & (y >= 0)], scenario_predictions[old_test & (y >= 0)],
                                                                  np.ones(int((old_test & (y >= 0)).sum()), dtype=bool), len(data.classes))}
    # Compare old unknown distances with TRAIN-only known distributions; not a new gate.
    distance_summary = {}
    for c, label in enumerate(data.classes):
        own_train = distances[train & (y == c), c]
        distance_summary[label] = {"train_known_distance_quantiles": np.quantile(own_train, [0, .5, .9, .95, 1]).tolist(),
                                   "train_leave_one_out_quantiles": np.quantile(prototype["trainLeaveOneOutDistances"][c], [0, .5, .9, .95, 1]).tolist(),
                                   "old_unknown_distance_quantiles": np.quantile(distances[old_test & (y == -1), c], [0, .5, 1]).tolist()}
    result = {"scope": "already-seen corrected POC data only; diagnostic, not fresh heldout performance",
              "frozen_model_sha256": hashlib.sha256((frozen / "model.onnx").read_bytes()).hexdigest(),
              "frozen_feature_sha256": hashlib.sha256((frozen / "features.npz").read_bytes()).hexdigest(),
              "class_order": data.classes, "embedding_definition": "hook on final temporal output, temporal mean, 48D",
              "old_unknown_cases": rows, "prototype_distance_summary": distance_summary,
              "frozen_tcn_input_ablations": ablations, "train_only_feature_baselines": baselines,
              "scenario_metadata_only_control": scenario_control,
              "caveats": ["Temporal reversal, shuffle, masking and static means change the input distribution; results do not prove causal feature importance.",
                          "Quality is held at its original measured value during model-input ablations; no new camera observation is implied.",
                          "Pose features include wrist coordinates, so pose/face-only is not devoid of hand-location information.",
                          "No RGB pixels enter this classifier, but landmark geometry, missingness, body pose and face shape may preserve signer/session/scenario information.",
                          "High performance from static or simple baselines does not establish that semantic sign motion is learned, nor prove a specific confound.",
                          "Static hand shape can carry legitimate sign information; strong hand-shape accuracy is not automatically evidence of a spurious shortcut.",
                          "Scenario-only labels are an explicit metadata control, never an input to deployed TCN or prototype models.",
                          "These old test outcomes may guide hypotheses only; new candidate settings are selected on the locked validation set, never new final results."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"report": str(output.resolve()),
            "ablations_old_known_accuracy": {name: results["old_test_diagnostic"]["closed_set_accuracy"] for name, results in ablations.items()},
            "baselines_old_known_accuracy": {name: results["old_test_diagnostic"]["closed_set_accuracy"] for name, results in baselines.items()},
            "scenario_control_old_known_accuracy": scenario_control["old_test_known"]["closed_set_accuracy"],
            "old_unknown_nearest_prototypes": [row["nearest_prototype"] for row in rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.frozen, args.output), ensure_ascii=False, indent=2))
