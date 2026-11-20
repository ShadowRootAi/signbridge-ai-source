"""Measure whether a fixed three-speed OpenHands ensemble improves selective recall."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort

from calibrate_openhands_gsl import CONFIDENCE_GRID, MARGIN_GRID, fit_gate, read_metadata
from evaluate_openhands_verifier import MODEL_SHA256, cached_rgb_poses, counts, read_native_poses, sha256, write_json
from openhands_gsl_stgcn import prepare_stgcn_pose


SPEED_LENGTHS = (0.8, 1.2)


def resample(points, factor):
    count = len(points)
    sample_count = max(1, int(np.floor(count * factor + 0.5)))
    times = np.linspace(0, count - 1, sample_count)
    left = times.astype(np.int64)
    right = np.minimum(left + 1, count - 1)
    mix = (times - left).astype(np.float32).reshape(-1, 1, 1)
    return points[left] + (points[right] - points[left]) * mix


def predict(session, points):
    tensor = prepare_stgcn_pose(points).numpy()
    return session.run(["logits"], {"landmarks": tensor})[0][0]


def ensemble(session, points, primary):
    total = primary.astype(np.float64)
    for factor in SPEED_LENGTHS:
        total += predict(session, resample(points, factor))
    return total / (len(SPEED_LENGTHS) + 1)


def scores(logits):
    logits = logits.astype(np.float64)
    winner, runner_up = np.argsort(-logits, axis=1, kind="stable")[:, :2].T
    rows = np.arange(len(logits))
    exps = np.exp(logits - logits[rows, winner, None])
    denominator = exps.sum(axis=1)
    return winner, 1. / denominator, (1. - exps[rows, runner_up]) / denominator


def strict_fit(reference, winners, confidence, margin, roster, known_wrong_limit):
    known = roster[reference]
    known_correct = known & (reference == winners)
    known_wrong = known & ~known_correct
    outside = ~known
    roster_match = roster[winners]
    # Include an all-reject fallback for calibration sets with no feasible selective fit.
    confidence_grid = (*CONFIDENCE_GRID, 1.)
    margins = np.asarray((*MARGIN_GRID, 1.))
    margin_masks = margin >= margins[:, None]
    best = None
    for minimum_confidence in confidence_grid:
        accept = margin_masks & roster_match & (confidence >= minimum_confidence)
        wrong = (accept & known_wrong).sum(axis=1)
        valid = (wrong <= known_wrong_limit) & ~(accept & outside).any(axis=1)
        correct = (accept & known_correct).sum(axis=1)
        for index in np.flatnonzero(valid):
            key = (int(correct[index]), -int(wrong[index]), minimum_confidence + margins[index], minimum_confidence)
            if best is None or key > best[0]:
                best = (key, minimum_confidence, margins[index])
    if best is None:
        raise ValueError("The strict fixed confidence grid had no feasible gate")
    return float(best[1]), float(best[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/cache/openhands-gsl"))
    parser.add_argument("--reports", type=Path, default=Path("reports/pretrained-gsl"))
    args = parser.parse_args()
    policy = json.loads((args.reports / "stgcn-gate-config.json").read_text(encoding="utf-8"))
    labels, supported, references = read_metadata(args.cache / "gsl_metadata.zip")
    label_indices = {label: index for index, label in enumerate(labels)}
    label_array = np.asarray(labels)
    roster = np.isin(labels, supported)
    if supported != policy["supportedLabels"] or sha256(args.cache / "stgcn.onnx") != MODEL_SHA256:
        raise ValueError("Unexpected OpenHands release model or roster")
    path = args.cache / "official-val-poses.npz"
    points, offsets, ids, glosses = read_native_poses(path)
    if dict(zip(ids, glosses)) != references:
        raise ValueError("Unexpected validation ID or gloss mapping")
    reference = np.asarray([label_indices[label] for label in glosses])
    reps = np.asarray([int(re.search(r"_signer3_rep(\d+)_glosses/", item).group(1)) for item in ids])
    calib, held = reps == 1, reps == 2
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.cache / "stgcn.onnx"), options, providers=["CPUExecutionProvider"])

    cache_path = args.cache / "official-val-poses-verifier-scores.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        if cache["sourceHash"].item() != sha256(path) or cache["modelHash"].item() != MODEL_SHA256:
            raise ValueError("Unexpected cached scores")
        base = cache["logits"]
    averaged = np.empty_like(base, dtype=np.float64)
    for index, primary in enumerate(base):
        averaged[index] = ensemble(session, points[offsets[index]:offsets[index + 1]], primary)
        if (index + 1) % 600 == 0:
            print(f"Ensemble scored {index + 1}/{len(base)}", flush=True)

    single_scores = scores(base)
    ensemble_scores = scores(averaged)
    baseline_thresholds = policy["minConfidence"], policy["minMargin"]

    def accepts(values, thresholds):
        winners, conf, margin = values
        return roster[winners] & (conf >= thresholds[0]) & (margin >= thresholds[1])

    base_accept = accepts(single_scores, baseline_thresholds)
    winner, conf, margin = ensemble_scores
    pair = (reference[calib], winner[calib], conf[calib], margin[calib])
    thresholds = fit_gate(supported, (label_array[pair[0]], label_array[pair[1]], pair[2], pair[3]),
                          CONFIDENCE_GRID, MARGIN_GRID)
    base_calibration_wrong = counts(base_accept[calib], reference[calib], single_scores[0][calib], roster)["wrong"]
    strict_thresholds = strict_fit(*pair, roster, base_calibration_wrong)
    report = {
        "modelSha256": MODEL_SHA256, "temporalLengthFactors": [1., *SPEED_LENGTHS],
        "method": "Average full-model logits from original and two time-resampled clips; no weight changes",
        "limit": "Rep1 fits gates, rep2 is diagnostic only. Both have signer 3 and mostly repeated prompts. Cached RGB uses other source-corpus signers, not a new person or camera.",
        "baseline": counts(base_accept[held], reference[held], single_scores[0][held], roster),
        "candidate": {},
    }
    for key, selected_thresholds in (("onePercentCalibration", thresholds), ("strictCalibration", strict_thresholds)):
        accepted = accepts(ensemble_scores, selected_thresholds)
        report["candidate"][key] = {
            "minConfidence": selected_thresholds[0], "minMargin": selected_thresholds[1],
            "calibration": counts(accepted[calib], reference[calib], winner[calib], roster),
            "heldRepetition": counts(accepted[held], reference[held], winner[held], roster),
        }
    clips = cached_rgb_poses(Path("data"), label_indices)
    original = np.stack([predict(session, frames) for _, frames in clips.values()])
    alternate = np.stack([ensemble(session, frames, raw) for (_, frames), raw in zip(clips.values(), original)])
    original, alternate = scores(original), scores(alternate)
    rgb_reference = np.asarray([label_indices[label] for label, _ in clips.values()])
    report["cachedRgb"] = {
        "baseline": counts(accepts(original, baseline_thresholds), rgb_reference, original[0], roster),
        "onePercentCalibration": counts(accepts(alternate, thresholds), rgb_reference, alternate[0], roster),
        "strictCalibration": counts(accepts(alternate, strict_thresholds), rgb_reference, alternate[0], roster),
    }
    report_path = args.reports / "stgcn-temporal-experiment.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
