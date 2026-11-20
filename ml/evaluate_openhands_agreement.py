"""Check whether the second OpenHands checkpoint can safely rescue rejected GSL clips."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort
import torch

from evaluate_openhands_verifier import counts, infer_native, make_session, write_json
from openhands_gsl import load_openhands_gsl, prepare_pose


CACHE = Path("research/cache/openhands-gsl")
REPORTS = Path("reports/pretrained-gsl")


def probabilities(logits):
    exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def load_native(path, session, lstm):
    ids, glosses, logits, _ = infer_native(session, path, CACHE)
    with np.load(path, allow_pickle=False) as dataset:
        keypoints, offsets = dataset["keypoints"], dataset["offsets"]
    secondary = []
    with torch.inference_mode():
        for start, end in zip(offsets, offsets[1:]):
            secondary.append(lstm(prepare_pose(keypoints[start:end]).unsqueeze(0))[0].numpy())
    return ids, glosses, logits, np.stack(secondary)


def load_rgb(session, lstm, index):
    manifests = (Path("data/manifest.json"), Path("data/open-set-v2/manifest.json"), Path("data/v3/manifest.json"))
    refs = {}
    for path in manifests:
        for entry in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if entry["annotation_raw_glosses"] == entry["raw_label"] and entry["raw_label"] in index:
                refs[entry["id"]] = entry["raw_label"]
    paths = sorted({path for pattern in ("data/features/*.jsonl", "data/open-set-v2/features/*.jsonl", "data/v3/features/*.jsonl")
                    for path in Path(".").glob(pattern)})
    data = {}
    from openhands_gsl_stgcn import prepare_stgcn_pose
    with torch.inference_mode():
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    identity, frames = row["id"], row.get("raw")
                    if not frames or identity in data or identity not in refs:
                        continue
                    points = np.zeros((len(frames), 75, 3), dtype=np.float32)
                    for n, frame in enumerate(frames):
                        for key, start, length in (("pose", 0, 33), ("left", 33, 21), ("right", 54, 21)):
                            if len(frame[key]) >= length:
                                points[n, start:start + length, :2] = [(p["x"], p["y"]) for p in frame[key][:length]]
                    primary = session.run(["logits"], {"landmarks": prepare_stgcn_pose(points).numpy()})[0][0]
                    secondary = lstm(prepare_pose(points).unsqueeze(0))[0].numpy()
                    coverage = sum(len(f["pose"]) >= 25 and min(f["pose"][i].get("visibility", 1) for i in (11, 12)) >= .5
                                   and (len(f["left"]) >= 21 or len(f["right"]) >= 21) for f in frames) / len(frames)
                    data[identity] = refs[identity], primary, secondary, coverage
    ids = np.array(list(data))
    refs, primary, secondary, coverage = zip(*data.values())
    return ids, refs, np.stack(primary), np.stack(secondary), np.array(coverage) >= .6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Run a final diagnostic on the upstream OpenHands test split.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    lstm, classes = load_openhands_gsl(CACHE)
    index = {label: i for i, label in enumerate(classes)}
    policy = json.loads((REPORTS / "stgcn-gate-expanded-candidate.json").read_text(encoding="utf-8"))
    roster = np.isin(classes, policy["supportedLabels"])
    gate = policy["decisionGate"]
    session, _, _ = make_session(CACHE / "stgcn.onnx", None)
    ids, glosses, primary, secondary = load_native(CACHE / "official-val-poses.npz", session, lstm)

    def signals(logits, alt):
        winners = logits.argmax(axis=1)
        rows = np.arange(len(winners))
        alternate = probabilities(alt)
        ranking = (alt > alt[rows, winners, None]).sum(axis=1) + 1
        return winners, probabilities(logits)[rows, winners], logits[rows, winners], alternate[rows, winners], ranking

    def decisions(logits, alt, thresholds=None, valid=True):
        winner, confidence, evidence, secondary_score, rank = signals(logits, alt)
        baseline = roster[winner] & (confidence >= gate["minimumConfidence"]) & (evidence >= gate["minimumEvidence"]) & valid
        if thresholds is None:
            return winner, baseline, baseline
        c, e, s, r = thresholds
        rescue = roster[winner] & (confidence >= c) & (evidence >= e) & (secondary_score >= s) & (rank <= r) & valid
        return winner, baseline, baseline | rescue

    calibration = np.array(["_rep1_" in clip for clip in ids])
    comparison = np.array(["_rep2_" in clip for clip in ids])
    y = np.array([index[label] for label in glosses])
    z, baseline, _ = decisions(primary, secondary)
    win, conf, evidence, score, rank = signals(primary[calibration], secondary[calibration])
    unseen = ~baseline[calibration] & roster[win]
    ref = y[calibration]
    right, wrong, unknown = (win == ref), (win != ref) & roster[ref], ~roster[ref]
    results = []
    for c in (.95, .98, .99, .995, .998, .999, .9995, .9999, .99999):
        for e in (-2, 0, .8783411979675293, 2, 4, 6, 8):
            for s in (.03, .08, .15, .25, .40, .55, .70):
                for r in (1, 2, 3):
                    rescue = unseen & (conf >= c) & (evidence >= e) & (score >= s) & (rank <= r)
                    results.append((int((rescue & right).sum()), int((rescue & wrong).sum()), int((rescue & unknown).sum()),
                                    (c, e, s, r)))

    chosen = {}
    for name, max_wrong, max_unknown in (("strict", 0, 0), ("bounded", 1, 1), ("exploratory", 2, 2)):
        eligible = [value for value in results if value[1] <= max_wrong and value[2] <= max_unknown]
        candidate = max(eligible, key=lambda value: (value[0], -value[2], -value[1], value[3]))
        chosen[name] = candidate[3]

    def describe(clip_ids, gold, logits, alt, mask=True):
        reference = np.array([index[label] for label in gold])
        report = {}
        for name, thresholds in [("baseline", None), *chosen.items()]:
            winner, _, accepted = decisions(logits, alt, thresholds, mask)
            report[name] = {
                **counts(accepted, reference, winner, roster),
                "entaxei": counts(accepted[reference == index["ΕΝΤΑΞΕΙ"]], reference[reference == index["ΕΝΤΑΞΕΙ"]],
                                  winner[reference == index["ΕΝΤΑΞΕΙ"]], roster),
            }
        w, c, e, s, r = signals(logits, alt)
        report["entaxeiRows"] = [
            {"id": str(clip), "top1": classes[w[i]], "confidence": float(c[i]), "logit": float(e[i]),
             "secondaryScore": float(s[i]), "secondaryRank": int(r[i])}
            for i, clip in enumerate(clip_ids) if reference[i] == index["ΕΝΤΑΞΕΙ"]
        ]
        return report

    rgb_ids, rgb_gold, rgb_primary, rgb_secondary, rgb_valid = load_rgb(session, lstm, index)
    report = {
        "scope": "Frozen OpenHands ST-GCN and LSTM. Fit rescue thresholds on upstream validation signer 3 repetition 1 only. "
                 "Other partitions are diagnostics and share the upstream GSL corpus. RGB is cached MediaPipe and gates "
                 "pose/either-hand coverage, without replaying the full browser motion guard.",
        "thresholdKeys": ["minConfidence", "minWinningLogit", "minSecondaryProbabilityForPrimaryWinner", "maxSecondaryRank"],
        "thresholds": chosen,
        "calibration": describe(ids[calibration], np.asarray(glosses)[calibration], primary[calibration], secondary[calibration]),
        "heldRepetition": describe(ids[comparison], np.asarray(glosses)[comparison], primary[comparison], secondary[comparison]),
        "cachedRgb": describe(rgb_ids, rgb_gold, rgb_primary, rgb_secondary, rgb_valid),
        "cachedRgbBySigner": {},
    }
    signers = np.array([re.search(r"_signer(\d+)_", clip).group(1) for clip in rgb_ids])
    for signer in sorted(set(signers)):
        match = signers == signer
        report["cachedRgbBySigner"][signer] = describe(
            rgb_ids[match], np.asarray(rgb_gold)[match], rgb_primary[match], rgb_secondary[match], rgb_valid[match])
    print(json.dumps({key: {k: v for k, v in report[key].items() if k != "entaxeiRows"}
                      for key in ("thresholds", "calibration", "heldRepetition", "cachedRgb")}), flush=True)
    if args.test:
        test_ids, test_glosses, test_primary, test_secondary = load_native(CACHE / "official-test-poses.npz", session, lstm)
        report["officialTestDiagnostic"] = describe(test_ids, test_glosses, test_primary, test_secondary)
        print(json.dumps({k: v for k, v in report["officialTestDiagnostic"].items() if k != "entaxeiRows"}), flush=True)
    write_json(REPORTS / "stgcn-lstm-agreement.json", report)


if __name__ == "__main__":
    main()

