"""Freeze a strictly checked recovery candidate for the 80 expansion labels."""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path
import json
import math

import numpy as np
import torch

from evaluate_openhands_agreement import probabilities
from evaluate_openhands_scope_veto import CACHE, FACTORS, POLICY, ROOT, digest, evidence, read, session, summarize
from evaluate_openhands_temporal_recovery import resample
from evaluate_openhands_verifier import write_json
from openhands_gsl_stgcn import prepare_stgcn_pose


REPORT_DIR = ROOT / "reports/pretrained-gsl"
FREEZE = REPORT_DIR / "expansion-recovery-freeze.json"
REPORT = REPORT_DIR / "expansion-recovery-experiment.json"
DEVELOPMENT = ("official-train-prototypes-poses", 4)
REPLAY = ("official-test-poses", 3)
GRID = {
    "minimumPrimary": (0.99, 0.995, 0.999, 0.9995, 0.9998),
    "minimumSecondary": (0.03, 0.05, 0.1, 0.25, 0.4, 0.6),
    "minimumShortest": (0.995, 0.999, 0.9995, 0.9999),
    "minimumPeak": (0.9995, 0.9998, 0.9999, 0.99999),
}


def build_evidence(stem, signer, labels, policy, primary, secondary):
    ids, refs, logits, poses, hashes = evidence(stem, signer, labels)
    roster = np.isin(labels, policy["supportedLabels"])
    added = np.isin(labels, policy["expansionGate"]["labels"])
    gate = policy["decisionGate"]
    agreement = policy["agreementGate"]
    temporal_policy = agreement["temporal"]
    winners = logits.argmax(1)
    rows = np.arange(len(ids))
    confidence = probabilities(logits)[rows, winners]
    valid = roster[winners] & (logits[rows, winners] >= gate["minimumEvidence"])
    primary_floor = np.where(added[winners], policy["expansionGate"]["minimumConfidence"], gate["minimumConfidence"])
    accepted = valid & (confidence >= primary_floor)
    original = valid & ~accepted & ~added[winners] & (confidence >= agreement["minimumConfidence"])
    targets = valid & ~accepted & added[winners] & (confidence >= min(GRID["minimumPrimary"]))
    expanded_scores = np.zeros((len(ids), 4), dtype=np.float64)
    for i, raw in enumerate(poses):
        if not (original[i] or targets[i]) or len(raw) < 4:
            continue
        pose = prepare_stgcn_pose(raw).numpy()
        if not np.any(pose):
            continue
        lstm_pose = pose.transpose(0, 2, 1, 3).reshape(1, len(raw), -1).copy()
        second = secondary.run(["logits"], {"landmarks": lstm_pose})[0]
        winner = winners[i]
        within_ranks = (second[0] > second[0, winner]).sum() < agreement["maximumSecondaryRank"]
        support = float(probabilities(second)[0, winner]) if within_ranks else 0.
        if original[i] and support >= agreement["minimumSecondaryProbability"]:
            accepted[i] = True
            continue
        minimum_secondary = temporal_policy["minimumSecondaryProbability"] if original[i] else min(GRID["minimumSecondary"])
        minimum_primary = temporal_policy["minimumPrimaryConfidence"] if original[i] else min(GRID["minimumPrimary"])
        if support < minimum_secondary or confidence[i] < minimum_primary:
            continue
        lengths = tuple(max(4, math.floor(len(raw) * factor + .5)) for factor in FACTORS)
        if len(set(lengths)) != len(lengths) or max(lengths) >= len(raw):
            continue
        outputs = np.stack([primary.run(["logits"], {"landmarks": resample(pose, n)})[0][0] for n in lengths])
        if not (np.all(outputs.argmax(1) == winner) and np.all(outputs[:, winner] >= gate["minimumEvidence"])):
            continue
        scores = probabilities(outputs)[:, winner]
        if original[i]:
            accepted[i] = (scores.min() >= temporal_policy["minimumConsistencyConfidence"]
                           and scores.max() >= temporal_policy["minimumPeakConfidence"])
        else:
            expanded_scores[i] = confidence[i], support, scores.min(), scores.max()
    return ids, refs, winners, accepted, roster, targets, expanded_scores, hashes


def assess(data, thresholds, labels):
    ids, refs, winners, accepted, roster, targets, values, hashes = data
    recovered = targets & np.all(values >= thresholds, axis=1)
    errors = recovered & (refs != winners)
    return {
        "evidenceSha256": hashes, "baseline": summarize(accepted, refs, winners, roster),
        "candidate": summarize(accepted | recovered, refs, winners, roster),
        "targetCandidatesEvaluated": int(targets.sum()),
        "viableSequences": int((values[:, 0] > 0).sum()),
        "additionalCorrect": int((recovered & (refs == winners)).sum()),
        "additionalWrongKnown": int((errors & roster[refs]).sum()),
        "additionalWrongOutside": int((errors & ~roster[refs]).sum()),
        "recoveriesPerLabel": dict(Counter(labels[ix] for ix in winners[recovered])),
        "errorRows": [{"id": str(ids[i]), "reference": labels[refs[i]], "winner": labels[winners[i]]}
                      for i in np.flatnonzero(errors)],
    }


def freeze(data, policy, labels):
    _, refs, winners, _, _, targets, values, hashes = data
    correct = targets & (refs == winners)
    wrong = targets & (refs != winners)
    candidates = []
    for thresholds in product(*GRID.values()):
        if thresholds[2] > thresholds[3]:
            continue
        recovered = targets & np.all(values >= thresholds, axis=1)
        if not (recovered & wrong).any():
            candidates.append((int((recovered & correct).sum()), len(set(winners[recovered])), thresholds))
    best = max(candidates, default=(0, 0, (1., 1., 1., 1.)))
    if best[0] == 0:
        best = (0, 0, (1., 1., 1., 1.))
    chosen = {
        "selection": "Highest new correct count, then label coverage and stricter thresholds, with zero new errors on signer 4.",
        "selectionSigner": DEVELOPMENT[1], "testSigner": REPLAY[1],
        "presetGrid": GRID, "selectedThresholds": dict(zip(GRID, best[2], strict=True)),
        "eligible": best[0] > 0, "selectedCorrect": best[0],
        "policySha256": digest(POLICY), "sourceSha256": digest(Path(__file__)),
        "selectionEvidenceSha256": hashes, "development": assess(data, best[2], labels),
    }
    if FREEZE.exists() and read(FREEZE) != json.loads(json.dumps(chosen)):
        raise ValueError("A different recovery specification is already frozen")
    write_json(FREEZE, chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "replay"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = read(POLICY)
    config = read(CACHE / "stgcn-onnx.json")
    labels = config["labels"]
    if config["model_sha256"] != policy["runtimeModelSha256"] or len(policy["expansionGate"]["labels"]) != 80:
        raise ValueError("Expected the frozen 200-label policy and its 80-label expansion")
    primary = session(CACHE / "stgcn.onnx", policy["runtimeModelSha256"])
    secondary = session(CACHE / "lstm.onnx", policy["agreementGate"]["modelSha256"])
    if args.phase == "freeze":
        selection = build_evidence(*DEVELOPMENT, labels, policy, primary, secondary)
        result = freeze(selection, policy, labels)
    else:
        selection = read(FREEZE)
        if selection["policySha256"] != digest(POLICY) or selection["sourceSha256"] != digest(Path(__file__)):
            raise ValueError("The candidate was frozen for different code or policy")
        if not selection["eligible"]:
            raise ValueError("No prespecified candidate improved development without new errors")
        data = build_evidence(*REPLAY, labels, policy, primary, secondary)
        thresholds = tuple(selection["selectedThresholds"][key] for key in GRID)
        result = {
            "scope": [
                "Native post-landmark replay, with browser tracking/motion guards unobserved.",
                "Selection inputs were seen by the upstream models. Replay signer 3 was not used for upstream training and its examples were previously inspected in this project.",
                "Out-of-roster means outside the 200 deployed labels. The underlying models were trained with these remaining labels.",
                "RGB upload, new users and actual out-of-domain motion require a separate study.",
            ],
            "freezeSha256": digest(FREEZE), "selectedThresholds": selection["selectedThresholds"],
            "retroSigner3": assess(data, thresholds, labels),
        }
        write_json(REPORT, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
