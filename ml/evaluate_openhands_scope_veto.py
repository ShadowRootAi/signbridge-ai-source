"""Replay a late OpenHands disagreement veto on signer-disjoint native poses."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import onnxruntime as ort
import torch

from evaluate_openhands_agreement import probabilities
from evaluate_openhands_recalibration import cached_scores
from evaluate_openhands_temporal_recovery import resample
from evaluate_openhands_verifier import counts, write_json
from openhands_gsl_stgcn import prepare_stgcn_pose


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "research/cache/openhands-gsl"
POLICY = ROOT / "reports/pretrained-gsl/stgcn-gate-roster-200.json"
REPORT = ROOT / "reports/pretrained-gsl/signer-scope-veto.json"
# Prespecified and unchanged across both signer cohorts. This experiment asks
# whether only very strong LSTM disagreements can remove risky primary picks.
VETO_MIN_CHALLENGER = 0.995
VETO_MAX_WINNER_SUPPORT = 0.0001
FACTORS = (0.65, 0.8)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def session(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError(f"Unexpected ONNX model: {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def evidence(stem, signer, labels):
    ids, glosses, logits, _, evidence_hashes = cached_scores(CACHE, stem, len(labels))
    signers = np.array([int(re.search(r"_signer(\d+)_", identity)[1]) for identity in ids])
    keep = signers == signer
    if not np.any(keep):
        raise ValueError(f"Missing signer {signer} in {stem}")
    index = {label: i for i, label in enumerate(labels)}
    with np.load(CACHE / f"{stem}.npz", allow_pickle=False) as data:
        offsets, keypoints = data["offsets"], data["keypoints"]
    source_rows = np.flatnonzero(keep)
    return ids[keep], np.array([index[gloss] for gloss in glosses[keep]]), logits[keep], (
        keypoints[offsets[i]:offsets[i + 1]] for i in source_rows
    ), evidence_hashes


def summarize(decisions, refs, winners, roster):
    metrics = counts(decisions, refs, winners, roster)
    accepted = int(decisions.sum())
    metrics["precisionAmongAccepted"] = metrics["correct"] / accepted if accepted else None
    return metrics


def evaluate(stem, signer, labels, policy, primary, secondary):
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
    recovery = valid & ~accepted & ~added[winners] & (confidence >= agreement["minimumConfidence"])
    veto = np.zeros(len(ids), dtype=bool)
    duplicate_lengths = np.zeros(len(ids), dtype=bool)
    duplicate_accepted = np.zeros(len(ids), dtype=bool)
    branches = np.where(accepted, "primary", "abstain").astype("U12")
    secondary_times = []
    touched = []
    newly_rescued = 0
    for i, raw in enumerate(poses):
        if not (accepted[i] or recovery[i]) or len(raw) < 4:
            continue
        input_pose = prepare_stgcn_pose(raw).numpy()
        if not np.any(input_pose):
            continue
        lstm_pose = input_pose.transpose(0, 2, 1, 3).reshape(1, len(raw), -1).copy()
        started = time.perf_counter()
        second = secondary.run(["logits"], {"landmarks": lstm_pose})[0]
        if accepted[i]:
            secondary_times.append((time.perf_counter() - started) * 1000)
        alternate = probabilities(second)[0]
        winner = winners[i]
        alternate_winner = alternate.argmax()
        if accepted[i]:
            if (alternate_winner != winner and alternate[alternate_winner] >= VETO_MIN_CHALLENGER
                    and alternate[winner] <= VETO_MAX_WINNER_SUPPORT):
                veto[i] = True
                touched.append({"id": str(ids[i]), "reference": labels[refs[i]], "primary": labels[winner],
                                "secondary": labels[alternate_winner], "secondaryBest": float(alternate[alternate_winner]),
                                "secondaryPrimarySupport": float(alternate[winner])})
            continue
        within_ranks = (second[0] > second[0, winner]).sum() < agreement["maximumSecondaryRank"]
        support = float(alternate[winner]) if within_ranks else 0.
        if support >= agreement["minimumSecondaryProbability"]:
            accepted[i] = True
            branches[i] = "agreement"
        elif support >= temporal_policy["minimumSecondaryProbability"] and confidence[i] >= temporal_policy["minimumPrimaryConfidence"]:
            lengths = [max(4, math.floor(len(raw) * factor + .5)) for factor in FACTORS]
            duplicate_lengths[i] = len(set(lengths)) != len(lengths) or max(lengths) >= len(raw)
            outputs = np.stack([primary.run(["logits"], {"landmarks": resample(input_pose, n)})[0][0] for n in lengths])
            agrees = np.all(outputs.argmax(1) == winner) and np.all(outputs[:, winner] >= gate["minimumEvidence"])
            scores = probabilities(outputs)[:, winner]
            if agrees and scores.min() >= temporal_policy["minimumConsistencyConfidence"] and scores.max() >= temporal_policy["minimumPeakConfidence"]:
                accepted[i] = True
                branches[i] = "temporal"
                newly_rescued += 1
                duplicate_accepted[i] = duplicate_lengths[i]
    metrics = summarize(accepted, refs, winners, roster)
    rejected = summarize(accepted & ~veto, refs, winners, roster)
    corrected = summarize(accepted & ~duplicate_accepted, refs, winners, roster)
    return {
        "signer": signer,
        "evidenceSha256": hashes,
        "baseline": metrics,
        "disagreementVeto": rejected,
        "distinctLengths": corrected,
        "extraLstmEvaluations": len(secondary_times),
        "extraLstmCpuMedianMs": float(np.median(secondary_times)) if secondary_times else None,
        "branches": dict(Counter(branches.tolist())),
        "vetoRows": touched,
        "resampledRecoveries": newly_rescued,
        "nonDistinctOrFullLengthCandidates": int(duplicate_lengths.sum()),
        "nonDistinctOrFullLengthAcceptances": int(duplicate_accepted.sum()),
        "nonDistinctOrFullLengthCorrectAcceptances": int((duplicate_accepted & (refs == winners)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Replay the fixed candidate on the previously observed signer 3 test cohort.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = read(POLICY)
    config = read(CACHE / "stgcn-onnx.json")
    labels = config["labels"]
    if config["model_sha256"] != policy["runtimeModelSha256"] or len(policy["supportedLabels"]) != 200:
        raise ValueError("Expected the frozen 200-label policy and its model")
    primary = session(CACHE / "stgcn.onnx", policy["runtimeModelSha256"])
    secondary = session(CACHE / "lstm.onnx", policy["agreementGate"]["modelSha256"])
    report = {
        "scope": [
            "Native-pose diagnostic; replay of the deployed post-landmark decision, without the browser tracking and motion gates.",
            "Signer 4 was seen upstream in model training. Signer 3 was absent upstream in training and has been used in earlier project evaluations.",
            "Out-of-roster references are classes excluded from the app's 200-label policy. They are all represented in both pretrained models' 310 classes.",
            "Prespecified veto thresholds, no fitting. No new-signer webcam or truly unknown movement claim.",
        ],
        "policySha256": digest(POLICY), "sourceSha256": digest(Path(__file__)),
        "selection": {"minimumChallengerScore": VETO_MIN_CHALLENGER, "maximumWinnerSupport": VETO_MAX_WINNER_SUPPORT},
        "cohorts": {},
    }
    cohorts = [("signer4BaselineDiagnostic", "official-train-prototypes-poses", 4)]
    if args.test:
        cohorts.append(("signer3RetrospectiveDiagnostic", "official-test-poses", 3))
    for key, stem, signer in cohorts:
        report["cohorts"][key] = result = evaluate(stem, signer, labels, policy, primary, secondary)
        print(json.dumps({key: {k: v for k, v in result.items() if k != "vetoRows"}}, ensure_ascii=False), flush=True)
    write_json(REPORT, report)


if __name__ == "__main__":
    main()
