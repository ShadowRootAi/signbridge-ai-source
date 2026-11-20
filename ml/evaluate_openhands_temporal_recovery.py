"""Calibrate a bounded temporal consistency check for primary OpenHands abstentions."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from evaluate_openhands_agreement import CACHE, REPORTS, probabilities
from evaluate_openhands_verifier import counts, write_json
from openhands_gsl import load_openhands_gsl, prepare_pose
from openhands_gsl_stgcn import prepare_stgcn_pose


FACTORS = (0.65, 0.8)


def resample(tensor, count):
    times = np.arange(count, dtype=np.float64) * (tensor.shape[2] - 1) / (count - 1)
    before = times.astype(np.int64)
    after = np.minimum(before + 1, tensor.shape[2] - 1)
    return (tensor[:, :, before] + (tensor[:, :, after] - tensor[:, :, before])
            * (times - before).reshape(1, 1, -1, 1)).astype(np.float32)


def evidence(cache_path, labels, policy, session, lstm):
    index = {name: i for i, name in enumerate(labels)}
    supported = np.isin(labels, policy["supportedLabels"])
    gate = policy["decisionGate"]
    agreement = policy["agreementGate"]
    with np.load(cache_path, allow_pickle=False) as raw, np.load(
        cache_path.with_name(cache_path.stem + "-verifier-scores.npz"), allow_pickle=False
    ) as scores:
        points, offsets, ids, glosses = [raw[key] for key in ("keypoints", "offsets", "ids", "glosses")]
        logits = scores["logits"]
    reference = np.array([index[gloss] for gloss in glosses])
    winners = logits.argmax(axis=1)
    rows = np.arange(len(winners))
    confidence = probabilities(logits)[rows, winners]
    winning_logit = logits[rows, winners]
    strong = supported[winners] & (winning_logit >= gate["minimumEvidence"])
    primary = strong & (confidence >= gate["minimumConfidence"])
    second = primary.copy()
    eligible = strong & ~primary & (confidence >= agreement["minimumConfidence"])
    fastest = np.zeros(len(winners))
    slowest = np.zeros(len(winners))
    support = np.zeros(len(winners))
    for i in np.flatnonzero(eligible):
        seq = points[offsets[i]:offsets[i + 1]]
        n = len(seq)
        if n < 4:
            continue
        with torch.inference_mode():
            score = lstm(prepare_pose(seq).unsqueeze(0)).numpy()
        if int((score > score[0, winners[i]]).sum()) < agreement["maximumSecondaryRank"]:
            support[i] = probabilities(score)[0, winners[i]]
        if support[i] >= agreement["minimumSecondaryProbability"]:
            second[i] = True
            continue
        tensor = prepare_stgcn_pose(seq).numpy()
        outputs = np.stack([
            session.run(["logits"], {"landmarks": resample(tensor, max(4, int(np.floor(n * factor + .5))))})[0][0]
            for factor in FACTORS
        ])
        if (outputs.argmax(axis=1) == winners[i]).all() and (outputs[:, winners[i]] >= gate["minimumEvidence"]).all():
            values = probabilities(outputs)[:, winners[i]]
            fastest[i], slowest[i] = values.max(), values.min()
    return ids, reference, winners, confidence, fastest, slowest, support, primary, second


def main():
    torch.set_num_threads(1)
    model, labels = load_openhands_gsl(CACHE)
    policy = json.loads((REPORTS / "stgcn-gate-agreement-candidate.json").read_text(encoding="utf-8"))
    roster = np.isin(labels, policy["supportedLabels"])
    entaxei = labels.index("ΕΝΤΑΞΕΙ")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(CACHE / "stgcn.onnx"), options, providers=["CPUExecutionProvider"])
    val = evidence(CACHE / "official-val-poses.npz", labels, policy, session, model)
    ids, ref, winners, conf, fast, slow, support, _, existing = val
    fit = np.array(["_rep1_" in item for item in ids])
    right = roster[ref] & (winners == ref)
    wrong = roster[ref] & (winners != ref)
    outside = ~roster[ref]
    recoveries = [(0, 0, 0, (1., 1., 1., 1.))]
    for floor in (.99, .995, .999, .9995, .9998, .9999, .99995, .99998):
        for peak in (.999, .9995, .9998, .9999, .99998, .99999, .999995, .999999):
            for consistency in (.98, .99, .995, .999, .9999, .99995):
                for secondary in (.03, .05, .08, .1, .15, .2):
                    added = fit & ~existing & (conf >= floor) & (fast >= peak) & (slow >= consistency) & (support >= secondary)
                    recoveries.append((int((added & right).sum()), int((added & wrong).sum()),
                                       int((added & outside).sum()), (floor, peak, consistency, secondary)))

    candidates = {}
    for name, budget in (("strict", 0), ("bounded", 1)):
        eligible = [row for row in recoveries if row[1] <= budget and row[2] <= budget]
        candidates[name] = max(eligible, key=lambda row: (row[0], -row[2], -row[1], row[3]))[3]

    def describe(values, mask):
        ids, ref, winners, conf, fast, slow, support, primary, second = values
        ent = mask & (ref == entaxei)
        result = {}
        for name, accepted in (("baseline", primary), ("agreement", second), *(
            (name, second | ((conf >= floor) & (fast >= peak) & (slow >= consistency) & (support >= secondary)))
            for name, (floor, peak, consistency, secondary) in candidates.items()
        )):
            result[name] = {**counts(accepted[mask], ref[mask], winners[mask], roster),
                            "entaxei": counts(accepted[ent], ref[ent], winners[ent], roster)}
        return result

    held = np.array(["_rep2_" in item for item in ids])
    report = {
        "scope": "Additional temporal confirmation fitted on upstream val repetition 1, with rep2 and original test diagnostic only. "
                 "Both alternate sequences must rank the original primary candidate first and pass the existing logit floor.",
        "lengthFactors": FACTORS,
        "thresholdKeys": ["minimumPrimaryConfidence", "minimumPeakConfidence", "minimumConsistencyConfidence", "minimumSecondaryProbability"],
        "thresholds": candidates,
        "calibration": describe(val, fit),
        "heldRepetition": describe(val, held),
    }
    print(json.dumps(report, ensure_ascii=True), flush=True)
    test = evidence(CACHE / "official-test-poses.npz", labels, policy, session, model)
    report["officialTestDiagnostic"] = describe(test, np.ones(len(test[0]), dtype=bool))
    print(json.dumps(report["officialTestDiagnostic"]), flush=True)
    write_json(REPORTS / "stgcn-temporal-recovery.json", report)


if __name__ == "__main__":
    main()

