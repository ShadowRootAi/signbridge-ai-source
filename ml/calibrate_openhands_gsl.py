"""Fit a 120-gloss scope gate on the ONNX model's published native GSL inputs."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort

from openhands_gsl_stgcn import METADATA_ARCHIVE_SHA256, STGCN_ARCHIVE_SHA256, prepare_stgcn_pose


ROSTER_SIZE = 120
MAX_ERROR_RATE = 0.01
COARSE_CONFIDENCE = (0., .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99, .995, 1.)
COARSE_MARGIN = (0., .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95, .975, .99, 1.)
NEAR_ONE = tuple(1 - digit * 10. ** -exponent for exponent in range(3, 13) for digit in (5, 2, 1))
CONFIDENCE_GRID = tuple(sorted(set(COARSE_CONFIDENCE[:-1] + NEAR_ONE)))
MARGIN_GRID = tuple(sorted(set(COARSE_MARGIN[:-1] + NEAR_ONE)))
CALIBRATION_REPETITION = 1
EVALUATION_REPETITION = 2


def read_metadata(path):
    prefix = "GSL_split/GSL_isolated/"
    if sha256(path.read_bytes()).hexdigest() != METADATA_ARCHIVE_SHA256:
        raise ValueError("OpenHands metadata checksum mismatch")
    with ZipFile(path) as archive:
        classes = archive.read(prefix + "iso_classes.csv").decode("utf-8").splitlines()
        train_rows = archive.read(prefix + "train_greek_iso.csv").decode("utf-8").splitlines()
        validation_rows = archive.read(prefix + "val_greek_iso.csv").decode("utf-8").splitlines()
    classes = [name.strip() for name in classes if name.strip()]
    references = dict(line.split("|", 1) for line in validation_rows if line.strip())
    counts = Counter(line.split("|", 1)[1] for line in train_rows if line.strip())
    if len(classes) != 310 or len(set(classes)) != 310 or set(counts) != set(classes):
        raise ValueError("Expected 310 labels matching the source training metadata")
    # Stable sorting preserves the official class-file order when frequencies tie.
    roster = sorted(classes, key=lambda gloss: -counts[gloss])[:ROSTER_SIZE]
    return classes, roster, references


def read_predictions(report, classes, references):
    if report["model"] != "stgcn" or report["checkpoint_sha256"] != STGCN_ARCHIVE_SHA256:
        raise ValueError("Expected original ST-GCN checkpoint predictions")
    if report["source_split"] != "val" or report["source_signers"] != ["3"]:
        raise ValueError("Expected the published val repetition pair from signer 3")
    rows = report["predictions"]
    if len(rows) != len(references) or {row["id"] for row in rows} != set(references):
        raise ValueError("Expected the complete published val split, with unique sample IDs")
    class_set = set(classes)
    for row in rows:
        if references[row["id"]] != row["reference_gloss"]:
            raise ValueError(f"Reference mismatch for {row['id']}")
        if len(set(row["top5"])) != 5 or not set(row["top5"]) <= class_set:
            raise ValueError(f"Invalid prediction glosses for {row['id']}")
    scores = np.asarray([row["top5_scores"] for row in rows], dtype=np.float64)
    if (scores.shape != (len(rows), 5) or not np.isfinite(scores).all()
            or np.any((scores < 0.) | (scores > 1.)) or np.any(np.diff(scores, axis=1) > 0.)):
        raise ValueError("Expected five finite scores in descending order per clip")
    ids = [row["id"] for row in rows]
    reps = np.asarray([int(re.search(r"_signer3_rep(\d+)_glosses/", item).group(1)) for item in ids])
    if set(reps.tolist()) != {CALIBRATION_REPETITION, EVALUATION_REPETITION}:
        raise ValueError("Unexpected validation repetition numbers")
    return (np.asarray([row["reference_gloss"] for row in rows]),
            np.asarray([row["top5"][0] for row in rows]), scores[:, 0], scores[:, 0] - scores[:, 1], reps)


def replay_onnx(cache, classes, predictions):
    with np.load(cache / "official-val-poses.npz", allow_pickle=False) as dataset:
        poses, offsets, ids = (dataset[name] for name in ("keypoints", "offsets", "ids"))
    if ids.tolist() != [row["id"] for row in predictions]:
        raise ValueError("Pose order differs from audited native predictions")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model_path = cache / "stgcn.onnx"
    model_hash = sha256(model_path.read_bytes()).hexdigest()
    model_config = json.loads((cache / "stgcn-onnx.json").read_text(encoding="utf-8"))
    if model_config["labels"] != classes or model_config["model_sha256"] != model_hash:
        raise ValueError("ONNX model is not bound to the expected class order and checksum")
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    scores = np.empty((len(ids), 2), dtype=np.float64)
    guesses = []
    for index, _ in enumerate(ids):
        prepared = prepare_stgcn_pose(poses[offsets[index]:offsets[index + 1]])
        logits = session.run(None, {input_name: prepared.numpy()})[0][0].astype(np.float64)
        if logits.shape != (310,) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid ONNX logits for {ids[index]}")
        top = np.argsort(-logits, kind="stable")[:2]
        # Use double precision throughout the same max-subtracted softmax as the browser.
        exp = np.exp(logits - logits[top[0]])
        scores[index] = exp[top] / exp.sum()
        guesses.append(classes[top[0]])
    return np.asarray(guesses), scores[:, 0], scores[:, 0] - scores[:, 1], model_hash


def metrics(supported, reference, guess, confidence, margin, min_confidence, min_margin, per_gloss=True):
    in_scope = np.isin(reference, supported)
    accepted = np.isin(guess, supported) & (confidence >= min_confidence) & (margin >= min_margin)
    correct = accepted & (reference == guess)
    wrong_in_scope = accepted & in_scope & ~correct
    false_outside = accepted & ~in_scope
    in_scope_count = int(in_scope.sum())
    outside_count = len(reference) - in_scope_count
    result = {
        "clips": len(reference),
        "inScopeClips": in_scope_count,
        "correctlyAccepted": int(correct.sum()),
        "inScopeWrongAccepts": int(wrong_in_scope.sum()),
        "inScopeAbstentions": int((in_scope & ~accepted).sum()),
        "outOfScopeClips": outside_count,
        "outOfScopeFalseAccepts": int(false_outside.sum()),
        "outOfScopeRejections": int((~in_scope & ~accepted).sum()),
        "inScopeCorrectAcceptanceRate": float(correct.sum() / in_scope_count),
        "inScopeWrongAcceptanceRate": float(wrong_in_scope.sum() / in_scope_count),
        "outOfScopeFalseAcceptanceRate": float(false_outside.sum() / outside_count),
    }
    if per_gloss:
        result["perGloss"] = {
            gloss: {"clips": int((reference == gloss).sum()),
                    "correctlyAccepted": int((correct & (reference == gloss)).sum()),
                    "wrongAccepts": int((wrong_in_scope & (reference == gloss)).sum())}
            for gloss in supported
        }
    return result


def fit_gate(supported, calibration, confidence_grid, margin_grid):
    feasible = []
    for min_confidence, min_margin in product(confidence_grid, margin_grid):
        result = metrics(supported, *calibration, min_confidence, min_margin, per_gloss=False)
        if (result["inScopeWrongAcceptanceRate"] <= MAX_ERROR_RATE
                and result["outOfScopeFalseAcceptanceRate"] <= MAX_ERROR_RATE):
            key = (result["correctlyAccepted"], -result["outOfScopeFalseAccepts"],
                   -result["inScopeWrongAccepts"], min_confidence + min_margin, min_confidence)
            feasible.append((key, min_confidence, min_margin))
    if not feasible:
        raise ValueError("No calibration threshold pair satisfied the stated error limits")
    _, min_confidence, min_margin = max(feasible)
    return min_confidence, min_margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/cache/openhands-gsl"))
    parser.add_argument("--reports", type=Path, default=Path("reports/pretrained-gsl"))
    args = parser.parse_args()
    classes, supported, references = read_metadata(args.cache / "gsl_metadata.zip")
    raw_report = (args.reports / "stgcn-native-poses.json").read_bytes()
    native_report = json.loads(raw_report)
    reference, native_guess, native_confidence, native_margin, reps = read_predictions(native_report, classes, references)
    guess, confidence, margin, onnx_sha256 = replay_onnx(args.cache, classes, native_report["predictions"])
    calibration_mask = reps == CALIBRATION_REPETITION
    evaluation_mask = reps == EVALUATION_REPETITION
    calib = (reference[calibration_mask], guess[calibration_mask], confidence[calibration_mask], margin[calibration_mask])
    evaluation = (reference[evaluation_mask], guess[evaluation_mask], confidence[evaluation_mask], margin[evaluation_mask])

    min_confidence, min_margin = fit_gate(supported, calib, CONFIDENCE_GRID, MARGIN_GRID)
    calibration_results = metrics(supported, *calib, min_confidence, min_margin)
    evaluation_results = metrics(supported, *evaluation, min_confidence, min_margin)
    membership_results = metrics(supported, *evaluation, 0., 0.)
    outside = [label for label in classes if label not in supported]
    policy = {
        "schemaVersion": 1,
        "sourceModelSha256": STGCN_ARCHIVE_SHA256,
        "runtimeModelSha256": onnx_sha256,
        "supportedLabels": supported,
        "unknownLabels": outside,
        "displayLabels": {label: label.replace("_", " ") for label in classes},
        "minConfidence": min_confidence,
        "minMargin": min_margin,
        "score": "Full 310-class float64 softmax top-1 probability from float32 ONNX logits",
        "margin": "Full 310-class top-1 probability minus top-2 probability",
        "decisionRule": "Accept only if the full-model winner is in supportedLabels and both minimums pass",
    }
    native_scores = (reference, native_guess, native_confidence, native_margin)
    coarse_calib = tuple(values[calibration_mask] for values in native_scores)
    coarse_evaluation = tuple(values[evaluation_mask] for values in native_scores)
    coarse_confidence, coarse_margin = fit_gate(supported, coarse_calib, COARSE_CONFIDENCE, COARSE_MARGIN)
    summary = {
        "nativePredictionReportSha256": sha256(raw_report).hexdigest(),
        "rosterSelection": "Top 120 by OpenHands source training frequency, ties by official class-file order",
        "rawGlossDisplay": "Preserve exact raw IDs, replacing underscores with spaces for display only. Glosses are labels, not a linguistic translation or a verified signed-language definition.",
        "calibrationSplit": "Published GSL val, signer 3, repetition 1 only",
        "evaluationSplit": "Published GSL val, signer 3, repetition 2 only",
        "limits": [
            "The split is disjoint by recording ID, with the same signer and largely repeated scripts.",
            "The released checkpoint config uses signer 3 repetitions 3–5 for author checkpoint selection.",
            "Other authentic GSL glosses outside the 120-label product roster remain known to the full 310-label model. These are out-of-scope positives, not genuinely unseen motions.",
            "No live webcam, free movement, no-sign, or additional signer examples were used to calibrate or evaluate this gate.",
            "Repetition 2 aggregate metrics were previously observed with a coarse float32 policy. The refined float64 gate was still fitted to repetition 1 alone, but repetition 2 results are a repeated diagnostic.",
        ],
        "calibrationGrid": {"minConfidence": CONFIDENCE_GRID, "minMargin": MARGIN_GRID},
        "maximumCalibrationWrongAcceptRate": MAX_ERROR_RATE,
        "selection": "Maximize correctly accepted known clips under both error limits. Tie-break by fewer false accepts, fewer wrong known accepts, then stronger thresholds.",
        "policy": policy,
        "calibration": calibration_results,
        "heldRepetition": evaluation_results,
        "heldRepetitionRosterOnly": membership_results,
        "earlierCoarseFloat32Gate": {
            "minConfidence": coarse_confidence,
            "minMargin": coarse_margin,
            "heldRepetition": metrics(supported, *coarse_evaluation, coarse_confidence, coarse_margin, per_gloss=False),
        },
        "onnxAndNativeTop1Agreement": int((guess == native_guess).sum()),
    }
    for name, contents in (("stgcn-gate-config.json", policy), ("stgcn-gate-calibration.json", summary)):
        (args.reports / name).write_text(json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    small_result = lambda counts: {key: value for key, value in counts.items() if key != "perGloss"}
    print(json.dumps({"minConfidence": min_confidence, "minMargin": min_margin,
                      "calibration": small_result(calibration_results),
                      "heldRepetition": small_result(evaluation_results),
                      "heldRepetitionRosterOnly": small_result(membership_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
