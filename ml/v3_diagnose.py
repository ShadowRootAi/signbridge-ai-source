"""Already-seen GSL failure investigation. No new model, gate or benchmark selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import softmax
from openset_core import fit_prototypes, metrics, motion_metrics, prototype_distances, score_gate
from openset_diagnose import infer_embeddings, load_frozen
from openset_train import load_split
from preprocess import frame_features, preprocess
from train import normalize_numpy

GREET, UNDERSTAND = "ΓΕΙΑ", "ΚΑΤΑΛΑΒΑΙΝΩ"


def summarize(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return {"n": int(len(values)), "mean": float(values.mean()), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max())} if len(values) else {"n": 0}


def trace_features(x, span_ms):
    """Coordinates share body origin/scale; local face coordinates are never subtracted."""
    x = np.asarray(x, dtype=float)
    nose = x[:, 168:170]  # Pose index0 in the same body-relative XY frame as the hands.
    result = {}
    for hand, start, local_start, flag, pose_wrist in (("left", 0, 42, 220, 178), ("right", 84, 126, 221, 180)):
        valid = (x[:, flag] >= 1 - 1e-6) & (x[:, 222] >= 1 - 1e-6)
        local_valid = x[:, flag] >= 1 - 1e-6
        body = x[:, start:start + 42].reshape(48, 21, 2)
        local = x[:, local_start:local_start + 42].reshape(48, 21, 2)
        wrist = body[:, 0] - nose
        tip = body[:, 8] - nose
        pair_valid = valid[:-1] & valid[1:]
        delta = np.diff(wrist, axis=0)
        dt = span_ms / 1000 / 47 if span_ms > 0 else None
        speed_step = np.linalg.norm(delta[pair_valid], axis=1)
        own = np.linalg.norm(body[:, 0] - x[:, pose_wrist:pose_wrist + 2], axis=1)
        opposite_start = 180 if hand == "left" else 178
        opposite = np.linalg.norm(body[:, 0] - x[:, opposite_start:opposite_start + 2], axis=1)
        tips = np.linalg.norm(local[:, [4, 8, 12, 16, 20]], axis=2)
        angles = []
        for mcp, pip, distal in ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
            a, b = local[:, mcp] - local[:, pip], local[:, distal] - local[:, pip]
            denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            cosine = np.divide((a * b).sum(axis=1), denominator, out=np.full(48, np.nan), where=denominator > 1e-12)
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        result[hand] = {"valid_frame_fraction": float(valid.mean()), "valid_pair_count": int(pair_valid.sum()),
                        "wrist_to_pose_nose_distance": summarize(np.linalg.norm(wrist[valid], axis=1)),
                        "index_tip_to_pose_nose_distance": summarize(np.linalg.norm(tip[valid], axis=1)),
                        "mean_wrist_relative_nose_xy": np.mean(wrist[valid], axis=0).tolist() if valid.any() else None,
                        "mean_signed_delta_per_normalized_step_xy": np.mean(delta[pair_valid], axis=0).tolist() if pair_valid.any() else None,
                        "normalized_step_speed": summarize(speed_step),
                        "resampled_seconds_velocity_xy": np.mean(delta[pair_valid] / dt, axis=0).tolist() if pair_valid.any() and dt else None,
                        "resampled_seconds_speed": summarize(speed_step / dt) if dt else {"n": 0},
                        "seconds_velocity_note": "derived from interpolated48-step sequence and original observed duration; not raw-frame velocity",
                        "net_wrist_displacement_xy": (wrist[np.flatnonzero(valid)[-1]] - wrist[np.flatnonzero(valid)[0]]).tolist() if valid.any() else None,
                        "own_pose_wrist_distance": summarize(own[valid]), "opposite_pose_wrist_distance": summarize(opposite[valid]),
                        "closer_to_opposite_pose_wrist_fraction": float((opposite[valid] < own[valid]).mean()) if valid.any() else None,
                        "finger_tip_wrist_distance_mean_palm_units": np.mean(tips[local_valid], axis=0).tolist() if local_valid.any() else None,
                        "finger_bend_angle_mean_degrees_2d": [summarize(v[local_valid]).get("mean") for v in angles]}
    return result


def raw_kinematics(frames):
    """Actual sampled raw-frame timestamps; positions still relative to shoulder/nose."""
    times = np.asarray([frame["t"] for frame in frames], dtype=float) / 1000
    features = np.asarray([frame_features(frame)[0] for frame in frames])
    result = {"raw_frame_count": len(frames), "observed_span_ms": float((times[-1] - times[0]) * 1000),
              "timestamp_steps_seconds": summarize(np.diff(times)), "units": "shoulder-widths/second in body-relative2D coordinates"}
    for hand, start, flag, pose_index in (("left", 0, 220, 15), ("right", 84, 221, 16)):
        valid = (features[:, flag] == 1) & (features[:, 222] == 1)
        pair = valid[:-1] & valid[1:]
        position = features[:, start:start + 2] - features[:, 168:170]
        velocity = np.diff(position, axis=0) / np.diff(times)[:, None]
        result[hand] = {"valid_raw_pairs": int(pair.sum()),
                        "signed_velocity_xy_mean": velocity[pair].mean(axis=0).tolist() if pair.any() else None,
                        "speed": summarize(np.linalg.norm(velocity[pair], axis=1)),
                        "pose_wrist_visibility": summarize([frame["pose"][pose_index].get("visibility", 1)
                                                            for frame in frames if len(frame["pose"]) > pose_index])}
    return result


def invariance_audit(frames):
    baseline, quality = preprocess(frames)
    audits = {}
    for name, operation in (("global_translation", lambda p: p.update(x=p["x"] + .07, y=p["y"] - .04)),
                            ("uniform_scale_1p25", lambda p: p.update(x=.5 + 1.25 * (p["x"] - .5), y=.5 + 1.25 * (p["y"] - .5))),
                            ("horizontal_coordinate_reflection_without_channel_swap", lambda p: p.update(x=1 - p["x"]))):
        changed = deepcopy(frames)
        for frame in changed:
            for field in ("left", "right", "pose", "face"):
                for point in frame[field]:
                    operation(point)
        actual, q = preprocess(changed)
        expected = baseline.copy()
        if "reflection" in name:
            expected[:, :220:2] *= -1
        audits[name] = {"max_abs_error_from_expected": float(np.abs(actual - expected).max()),
                        "max_abs_change_from_original": float(np.abs(actual - baseline).max()), "quality_unchanged": q == quality}
    slowed = deepcopy(frames)
    for frame in slowed: frame["t"] *= 2
    slow_x, _ = preprocess(slowed)
    audits["duration_doubled_same_geometry"] = {"feature_max_abs_difference": float(np.abs(slow_x - baseline).max()),
                                               "interpretation": "absolute duration/speed is absent from model input by construction"}
    reversed_frames = deepcopy(frames[::-1])
    for i, frame in enumerate(reversed_frames): frame["t"] = frames[i]["t"]
    reverse_x, _ = preprocess(reversed_frames)
    audits["reversed_raw_order"] = {"error_against_reversed_feature_order": float(np.abs(reverse_x - baseline[::-1]).max()),
                                    "interpretation": "time order is retained by preprocessing rather than averaged away"}
    swapped = deepcopy(frames)
    for frame in swapped: frame["left"], frame["right"] = frame["right"], frame["left"]
    swapped_x, _ = preprocess(swapped)
    expected = baseline.copy(); expected[:, :84] = baseline[:, 84:168]; expected[:, 84:168] = baseline[:, :84]
    expected[:, 220:222] = baseline[:, [221, 220]]
    audits["left_right_channel_swap_only"] = {"max_abs_error_from_exact_channel_swap": float(np.abs(swapped_x - expected).max()),
                                             "interpretation": "hand channels are distinct; not silently sorted by screen location"}
    return audits


def load_raw(project, ids):
    found = {}
    for path in sorted((project / "data/open-set-v2/features").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["id"] in ids and record.get("raw"):
                    found[record["id"]] = record["raw"]
    return found


def describe_outcomes(y, labels, people, classes, scored):
    result = {}
    for label in np.unique(labels):
        result[str(label)] = {}
        for person in np.unique(people[labels == label]):
            mask = (labels == label) & (people == person)
            result[str(label)][str(person)] = {"n": int(mask.sum()), "accepted_n": int((mask & scored["accepted"]).sum()),
                                                "predicted_known_counts": {name: int((mask & (scored["predicted"] == k)).sum()) for k, name in enumerate(classes)},
                                                "accepted_as": {name: int((mask & scored["accepted"] & (scored["predicted"] == k)).sum()) for k, name in enumerate(classes)},
                                                "confidence": summarize(scored["confidence"][mask])}
    return result


def temporal_variants(x):
    result = {"normal": x, "reverse": x[:, ::-1].copy()}
    for seed in (20261108, 20261109, 20261110):
        result[f"shuffle_seed_{seed}"] = x[:, np.random.default_rng(seed).permutation(48)].copy()
    result["static_mean"] = np.repeat(x.mean(axis=1, keepdims=True), 48, axis=1)
    return result


def all_exposed_ablations(final, b_model, b_config, c_model, c_config, baseline, base_config, output):
    """Repeated inference on the exposed129, with all weights and gates unchanged."""
    report = {"scope": "All129 already-exposed V2 final clips; diagnostics only, no fitting or selection",
              "perturbation_contract": "One deterministic common48-frame permutation per shuffle seed; static mean repeats the per-clip channel means. Original measured quality is retained; derived motion is recomputed. These are distribution shifts, not causal proofs.",
              "results": {}, "predictions": []}
    normal = {}
    for variant, changed in temporal_variants(final.x).items():
        motion = motion_metrics(changed)["motionBodyRms"]
        bl, be = infer_embeddings(b_model, changed); cl, ce = infer_embeddings(c_model, changed); dl, _ = baseline(changed)
        scored_models = {"CNN_B": score_gate(bl, final.quality, motion, b_config, be),
                         "full_feature_centroid": score_gate(dl, final.quality, motion, base_config),
                         "frozen_C": score_gate(cl, final.quality, motion, c_config, ce)}
        report["results"][variant] = {}
        for name, scored in scored_models.items():
            if variant == "normal": normal[name] = scored
            result = metrics(final.y, scored, final.classes, final.signer, final.raw_label)
            result["per_category_per_person"] = describe_outcomes(final.y, final.raw_label, final.signer, final.classes, scored)
            result["prediction_changes_from_normal_n"] = int((scored["predicted"] != normal[name]["predicted"]).sum())
            result["acceptance_changes_from_normal_n"] = int((scored["accepted"] != normal[name]["accepted"]).sum())
            result["confidence_change_from_normal"] = summarize(scored["confidence"] - normal[name]["confidence"])
            report["results"][variant][name] = result
            for i, source in enumerate(final.source_id):
                report["predictions"].append({"source_id": str(source), "label": str(final.raw_label[i]), "person": str(final.signer[i]),
                                              "variant": variant, "model": name, "predicted_known": final.classes[int(scored["predicted"][i])],
                                              "accepted": bool(scored["accepted"][i]), "confidence": float(scored["confidence"][i]),
                                              "reasons": [key for key, flags in scored["reasons"].items() if flags[i]]})
    path = output / "all-exposed-temporal-ablations.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    variants = list(report["results"])
    names = ["Normal", "Reverse", "Shuffle1", "Shuffle2", "Shuffle3", "Static mean"]
    for row, model in enumerate(("CNN_B", "full_feature_centroid", "frozen_C")):
        correct, wrong, false = [], [], []
        for variant in variants:
            m = report["results"][variant][model]
            n = sum(v["accepted_correct_n"] for v in m["per_known_sign"].values())
            correct.append(n); wrong.append(m["accepted_known_n"] - n)
            false.append(sum(v["false_acceptance_n"] for v in m["per_unknown_category"].values()))
        axes[row, 0].bar(names, correct, color="#1463a2", label="Accepted correct")
        axes[row, 0].bar(names, wrong, bottom=correct, color="#edaa35", label="Accepted wrong")
        axes[row, 1].bar(names, false, color="#c43d2c")
        for ax, values, total in ((axes[row, 0], np.asarray(correct) + wrong, 79), (axes[row, 1], false, 50)):
            for column, value in enumerate(values): ax.text(column, value + 1, str(value), ha="center", fontsize=9)
            ax.set_ylim(0, total); ax.set_ylabel(f"Clips / {total}"); ax.tick_params(axis="x", labelrotation=25)
            ax.grid(axis="y", alpha=.15)
        axes[row, 0].set_title(f"{model}: known acceptance"); axes[row, 1].set_title(f"{model}: unknown false acceptance")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("All129 exposed clips: frozen temporal perturbations, no refitting", fontsize=14)
    fig.tight_layout(); fig.savefig(output / "all-exposed-temporal-ablations.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    return path.name


def run(project, output):
    artifact = project / "artifacts/open-set-v2"; data_dir = project / "data/open-set-v2"
    final = load_split(data_dir / "final_test.npz", "test"); train = load_split(data_dir / "train.npz", "train")
    selected = np.isin(final.raw_label, [GREET, UNDERSTAND]); ix = np.flatnonzero(selected)
    x, y, labels, people = final.x[selected], final.y[selected], final.raw_label[selected], final.signer[selected]
    sources, quality = final.source_id[selected], final.quality[selected]
    with np.load(data_dir / "final_test.npz", allow_pickle=False) as f:
        spans = f["observed_span_ms"][selected]; frame_counts = f["frame_count"][selected]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    by_source = {s["source_path"]: s for s in manifest["samples"]}
    raw = load_raw(project, {by_source[source]["id"] for source in sources})
    b_model, b_config = load_frozen(artifact / "B_tcn_diverse_other")
    c_model, c_config = load_frozen(artifact / "C_frozen_embedding_prototype")
    b_train_logits, b_train_embed = infer_embeddings(b_model, train.x)
    b_logits, b_embed = infer_embeddings(b_model, x)
    b_proto = fit_prototypes(b_train_embed, train.y, train.classes)  # Descriptive TRAIN-only centroids, not a new classifier/gate.
    b_distances = prototype_distances(b_embed, b_proto)
    with np.load(artifact / "reference-centroid.npz", allow_pickle=False) as f:
        base_parameters = {key: f[key].copy() for key in f.files}
    base_config = json.loads((artifact / "reference-centroid-config.json").read_text(encoding="utf-8"))

    def baseline(input_x):
        z = normalize_numpy(input_x, base_parameters["mean"], base_parameters["std"], base_parameters["keep"]).reshape(len(input_x), -1)
        distances = np.stack([np.sqrt(((z - center) ** 2).mean(axis=1)) for center in base_parameters["centroids"]], axis=1)
        return -distances ** 2, distances

    base_logits, base_distances = baseline(x)
    variants = temporal_variants(x)
    outcomes, scored_normal, row_records = {}, {}, []
    for variant, changed in variants.items():
        changed_motion = motion_metrics(changed)["motionBodyRms"]
        bl, be = infer_embeddings(b_model, changed); cl, ce = infer_embeddings(c_model, changed); dl, dd = baseline(changed)
        scored_models = {"CNN_B": score_gate(bl, quality, changed_motion, b_config, be),
                         "full_feature_centroid": score_gate(dl, quality, changed_motion, base_config),
                         "frozen_C": score_gate(cl, quality, changed_motion, c_config, ce)}
        outcomes[variant] = {name: describe_outcomes(y, labels, people, final.classes, scores) for name, scores in scored_models.items()}
        if variant == "normal": scored_normal = scored_models
        for i, source in enumerate(sources):
            row_records.append({"source_id": str(source), "label": str(labels[i]), "person": str(people[i]), "variant": variant,
                                "models": {name: {"predicted_known": final.classes[int(scores["predicted"][i])],
                                                  "accepted": bool(scores["accepted"][i]), "confidence": float(scores["confidence"][i]),
                                                  "reasons": [key for key, flags in scores["reasons"].items() if flags[i]]}
                                           for name, scores in scored_models.items()}})
    clipping = np.clip((x - b_model.feature_mean.numpy()) / b_model.feature_std.numpy(), -8, 8) * b_model.feature_keep.numpy()
    geometry, invariant, raw_details = [], {}, {}
    for i, source in enumerate(sources):
        sample = by_source[source]; raw_frames = raw.get(sample["id"])
        item = {"source_id": str(source), "id": sample["id"], "label": str(labels[i]), "person": str(people[i]),
                "scenario": sample["scenario"], "source_session": sample["session"], "fps": sample["fps"],
                "frame_count": int(frame_counts[i]), "observed_span_ms": float(spans[i]), "quality": float(quality[i]),
                "mean_presence_left_right_pose_face": x[i, :, 220:224].mean(axis=0).tolist(),
                "interpolated_partial_hand_mask_fraction": float(((x[i, :, 220:222] > 0) & (x[i, :, 220:222] < 1)).mean()),
                "preprocessing_saturation_fraction": float((np.abs(x[i, :, :220]) >= 8).mean()),
                "B_standardized_saturation_fraction": float((np.abs(clipping[i, :, :220]) >= 8).mean()),
                "geometry": trace_features(x[i], spans[i]), "B_logits": b_logits[i].tolist(),
                "B_confidence": float(scored_normal["CNN_B"]["confidence"][i]),
                "B_embedding_known_prototype_distances": dict(zip(final.classes, b_distances[i].tolist())),
                "full_feature_centroid_distances": dict(zip(final.classes + ["OTHER"], base_distances[i].tolist())),
                "raw_landmark_fixture_available": raw_frames is not None}
        if raw_frames:
            px, _ = preprocess(raw_frames)
            item["raw_preprocessing_max_error"] = float(np.abs(px - x[i]).max())
            raw_details[sample["id"]] = raw_kinematics(raw_frames)
            invariant[sample["id"]] = invariance_audit(raw_frames)
        geometry.append(item)
    output.mkdir(parents=True, exist_ok=True)
    all_ablation_path = all_exposed_ablations(final, b_model, b_config, c_model, c_config, baseline, base_config, output)
    plots = make_plots(output, x, labels, people, sources, geometry, raw_details, train, b_train_embed,
                       b_embed, b_distances, base_distances, outcomes)
    report = {"scope": "19 already-seen final clips: ten failedΚΑΤΑΛΑΒΑΙΝΩ and nine comparableΓΕΙΑ; no new fit/selection/test claim",
              "matplotlib_version": matplotlib.__version__,
              "frozen_files": {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (artifact / "B_tcn_diverse_other/model.onnx", artifact / "B_tcn_diverse_other/config.json",
                                            artifact / "B_tcn_diverse_other/model.pt", artifact / "C_frozen_embedding_prototype/model.pt",
                                            artifact / "C_frozen_embedding_prototype/model.onnx", artifact / "C_frozen_embedding_prototype/config.json",
                                            artifact / "reference-centroid.npz", artifact / "reference-centroid-config.json",
                                            data_dir / "train.npz", data_dir / "final_test.npz",
                                            project / "web/src/features.ts", project / "ml/preprocess.py")},
              "raw_fixture_count": len(raw_details), "clip_geometry": geometry, "raw_frame_kinematics": raw_details,
              "tracking_summary": {
                  "quality_definition_limit": "quality counts pose plus either hand; a tracked resting hand can hide active-hand dropout",
                  "all19_original_quality": sorted(set(float(q) for q in quality)),
                  "right_hand_dropout_clip_counts": {label: int(sum(g["label"] == label and g["mean_presence_left_right_pose_face"][1] < 1 for g in geometry)) for label in (GREET, UNDERSTAND)},
                  "right_hand_dropout_does_not_explain_all_failures": "Seven of ten ΚΑΤΑΛΑΒΑΙΝΩ have right-hand presence1.0 and are still falsely accepted",
                  "hand_to_pose_wrist_consistency_scope": "Only fully tracked hand+pose frames enter the own-versus-opposite comparison; missing/interpolated partial hand frames are excluded",
                  "mask_interpolation": "Linear time resampling creates fractional masks and coordinates between present/absent frames; geometry diagnostics exclude partial endpoint pairs, while the unchanged classifier consumes the stored tensor"},
              "preprocessing_invariance_audit": invariant, "temporal_ablation_outcomes": outcomes,
              "temporal_ablation_records": row_records, "plots": plots,
              "all129_already_exposed_temporal_report": all_ablation_path,
              "representation_audit": {
                  "absent_from_model_input": ["hand/pose/face z", "absolute observed duration", "absolute source FPS", "explicit seconds-based velocity channels"],
                  "derivable_from_existing_input": ["time order across48samples", "signed per-normalized-step XY displacement", "hand-to-pose-nose location", "2D finger/local hand shape", "presence/missingness"],
                  "hand_face_coordinate_rule": "hand-body coordinates minus pose-nose body coordinates; never subtract face-local normalized coordinates",
                  "handedness_code_audit": "Worker directly maps Holistic leftHandLandmarks to left and rightHandLandmarks to right. No explicit mirroring/screen-side sorting appears in reviewed capture/features code. Detector semantic correctness is not proven by code inspection.",
                  "normalization": "body-relative hand positions remain separate from wrist/palm-local shape; scale/translation removal intentionally loses global image position/scale",
                  "CNN_pooling": "temporal convolutions followed by mean pooling produce48D embedding; full-feature baseline retains224x48=10752coordinates"},
              "interpretation_limits": ["No new model, prototype gate, threshold or deployment artifact was fitted or modified. B class centroids/PCA are descriptive training-only geometry summaries.",
                                        "Temporal ablations and channel manipulations are distribution shifts, not causal importance proofs.",
                                        "Static hand shape can encode legitimate sign semantics. Temporal-order reliance is not established by high known accuracy alone.",
                                        "Raw-frame derivatives are available only for retained raw fixtures. Other seconds-based derivatives are from interpolated48-step trajectories with recorded duration.",
                                        "Anatomic left/right labels need visual or expert confirmation; nearest-pose-wrist consistency is a geometric diagnostic, not definitive handedness truth.",
                                        "2D finger angles and hand-to-nose distances are geometric proxies, not expert linguistic sign annotations.",
                                        "The ten unsupported examples were already exposed in prior evaluations; this is not a fresh benchmark."],
              "one_targeted_comparator_proposal": {"method": "full-feature nearest centroids with separate training-negative category prototypes versus a pooled OTHER centroid",
                                                    "scope": "same fixed normalized224x48representation; no neural architecture/HPO expansion",
                                                    "fit": "training split only; keep per-negative-category label only for negative prototypes",
                                                    "select": "validation-only same prespecified coverage/accuracy/FAR priorities",
                                                    "evaluate_condition": "requires unused positiveΓΕΙΑ and unused unknown categories/clips for a fresh claim; otherwise explicitly exploratory seen-known evaluation"}}
    (output / "numerical-diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"report": str((output / "numerical-diagnosis.json").resolve()), "raw_fixtures": len(raw_details),
            "plots": plots, "outcomes": outcomes}


def make_plots(output, x, labels, people, sources, geometry, raw_details, train, train_embed, embed, embed_dist, base_dist, outcomes):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 130})
    colors = {GREET: "#1463a2", UNDERSTAND: "#c43d2c"}; paths = []
    def save(fig, name):
        fig.tight_layout(rect=(0, .07 if fig.legends else 0, 1, 1)); fig.savefig(output / name, bbox_inches="tight"); plt.close(fig); paths.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for row, person in enumerate(("3", "7")):
        for col, (hand, start, flag) in enumerate((("Left", 0, 220), ("Right", 84, 221))):
            ax = axes[row, col]
            for i in np.flatnonzero(people == person):
                valid = (x[i, :, flag] >= 1 - 1e-6) & (x[i, :, 222] >= 1 - 1e-6)
                p = x[i, :, start + 16:start + 18] - x[i, :, 168:170]
                p[~valid] = np.nan
                ax.plot(p[:, 0], p[:, 1], color=colors[labels[i]], alpha=.45, linewidth=1.4)
                good = np.flatnonzero(valid)
                if len(good):
                    ax.scatter(*p[good[0]], color=colors[labels[i]], marker="o", s=15)
                    ax.scatter(*p[good[-1]], color=colors[labels[i]], marker="^", s=25)
            ax.scatter(0, 0, color="black", marker="+", label="Pose nose")
            ax.set_title(f"Person {person}: {hand} index tip relative to nose")
            ax.set_xlabel("Horizontal shoulder-width units"); ax.set_ylabel("Vertical shoulder-width units")
            ax.grid(alpha=.2)
    axes[0, 0].invert_yaxis()  # One inversion propagates across the shared axes.
    handles = [plt.Line2D([0], [0], color=color, label=label) for label, color in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle("Hand-to-face trajectories: circle=start, triangle=end; already-seen clips", y=1.02)
    save(fig, "hand-to-face-trajectories.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    known = train.y >= 0; centered = train_embed[known].astype(float)
    mean = centered.mean(axis=0); _, singular, vh = np.linalg.svd(centered - mean, full_matrices=False)
    axes2 = vh[:2].T; ztrain = (centered - mean) @ axes2; z = (embed - mean) @ axes2
    for k, label in enumerate(train.classes):
        axes[0].scatter(ztrain[train.y[known] == k, 0], ztrain[train.y[known] == k, 1], s=13, alpha=.22, label=f"Train {label}")
    for label in (GREET, UNDERSTAND):
        for person, marker in (("3", "o"), ("7", "^")):
            mask = (labels == label) & (people == person)
            axes[0].scatter(z[mask, 0], z[mask, 1], c=colors[label], marker=marker, s=65, edgecolors="black", linewidths=.5, label=f"{label} / person{person}")
    explained = (singular[:2] ** 2).sum() / (singular ** 2).sum()
    axes[0].set_title(f"B48D embedding: train-fit PCA ({explained:.1%} variance)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=7)
    for label in (GREET, UNDERSTAND):
        mask = labels == label
        axes[1].scatter(base_dist[mask, 0] - base_dist[mask, 3], embed_dist[mask, 0], c=colors[label], label=label, s=55)
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].set_xlabel("Full-feature distance to ΓΕΙΑ minus distance to OTHER")
    axes[1].set_ylabel("B embedding distance to train ΓΕΙΑ prototype")
    axes[1].set_title("Different representations: distances are not interchangeable")
    axes[1].legend(); save(fig, "embedding-and-prototype-distances.png")
    row_names = [(model, variant) for model in ("CNN_B", "full_feature_centroid", "frozen_C") for variant in outcomes]
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    for ax, label, cmap in zip(axes, (GREET, UNDERSTAND), ("Blues", "Reds")):
        cells = np.asarray([[outcomes[variant][model][label][person]["accepted_n"] / outcomes[variant][model][label][person]["n"] for person in ("3", "7")] for model, variant in row_names])
        ax.imshow(cells, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for r, (model, variant) in enumerate(row_names):
            for c, person in enumerate(("3", "7")):
                item = outcomes[variant][model][label][person]
                ax.text(c, r, f"{item['accepted_n']}/{item['n']}", ha="center", va="center", color="white" if cells[r, c] > .55 else "black")
        ax.set_xticks([0, 1], ["Person3", "Person7"]); ax.set_yticks(range(len(row_names)), [f"{m}: {v}" for m, v in row_names], fontsize=8)
        ax.set_title(f"{label}: {'known acceptance' if label == GREET else 'false acceptance'}")
    save(fig, "temporal-ablation-decisions.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for label in (GREET, UNDERSTAND):
        for person, marker in (("3", "o"), ("7", "^")):
            indices = np.flatnonzero((labels == label) & (people == person))
            for i in indices:
                item = geometry[i]
                # For this head/hand comparison display the tracked hand nearest the nose;
                # both hands remain available in the numerical report.
                hand = min((h for h in ("left", "right") if item["geometry"][h]["valid_pair_count"]),
                           key=lambda h: item["geometry"][h]["wrist_to_pose_nose_distance"]["mean"])
                g = item["geometry"][hand]
                axes[0].scatter(item["observed_span_ms"] / 1000, g["normalized_step_speed"].get("mean", np.nan), color=colors[label], marker=marker)
                if g["resampled_seconds_velocity_xy"]:
                    axes[1].scatter(*g["resampled_seconds_velocity_xy"], color=colors[label], marker=marker)
                raw = raw_details.get(item["id"])
                if raw:
                    axes[2].scatter(g["resampled_seconds_speed"].get("mean", np.nan), raw[hand]["speed"].get("mean", np.nan), color=colors[label], marker=marker)
    axes[0].set_xlabel("Actual observed span (seconds)"); axes[0].set_ylabel("Mean wrist speed per normalized48-step interval")
    axes[0].set_title("Hand nearest nose: timing and motion")
    axes[1].set_xlabel("Mean signed resampled velocity X (shoulder widths/s)"); axes[1].set_ylabel("Y (shoulder widths/s)"); axes[1].set_title("Direction; interpolated trajectories")
    axes[2].set_xlabel("Resampled seconds-based mean speed"); axes[2].set_ylabel("Actual raw-frame mean speed"); axes[2].set_title("Raw fixtures only; different sampling")
    for ax in axes: ax.grid(alpha=.2)
    fig.legend(handles=handles, loc="lower center", ncol=2); save(fig, "timing-and-signed-velocity.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, hand in zip(axes, ("left", "right")):
        for label in (GREET, UNDERSTAND):
            values = [item["geometry"][hand]["finger_tip_wrist_distance_mean_palm_units"] for item in geometry
                      if item["label"] == label and item["geometry"][hand]["finger_tip_wrist_distance_mean_palm_units"] is not None]
            array = np.asarray(values)
            if len(array): ax.errorbar(np.arange(5), array.mean(axis=0), yerr=array.std(axis=0), color=colors[label], marker="o", label=label)
        ax.set_xticks(range(5), ["Thumb", "Index", "Middle", "Ring", "Little"]); ax.set_title(f"{hand.title()} hand:2D shape proxies")
        ax.set_ylabel("Fingertip-to-wrist distance / palm scale"); ax.legend(); ax.grid(alpha=.2)
    save(fig, "local-hand-shape-proxies.png")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/v3"))
    args = parser.parse_args()
    print(json.dumps(run(args.project.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
