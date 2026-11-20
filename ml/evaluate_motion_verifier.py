"""Check whether independently matched motion can verify CTC phrase recoveries.

The six-phrase checkpoint remains frozen. Fit distance settings and acceptance
floors on signer 3 only; score signers 6/7 and out-of-scope utterances once.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import file_digest
from itertools import product
import json
from math import ceil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from evaluate_continuous_transfer import CHECKPOINT, SOURCES, exact_template_scores
from train_continuous import ContinuousCTC, altered_frames, ctc_decode, normalize_text, read_utterances


STEPS = 28
BAND = 7
REPORT = Path("reports/continuous-v1-expanded/motion-agreement.json")
# The masks after the 220 coordinates are ordered left, right, pose, face.
GROUPS = (
    (slice(0, 42), (0, 2), 1.0),
    (slice(42, 84), (0,), .45),
    (slice(84, 126), (1, 2), 1.0),
    (slice(126, 168), (1,), .45),
    (slice(168, 186), (2,), .4),
    (slice(186, 220), (3,), .15),
)


def sequence(frames):
    values = frames.numpy()
    times = np.linspace(0, len(values) - 1, STEPS)
    lower = times.astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (times - lower)[:, None]
    values = (values[lower] + (values[upper] - values[lower]) * weight).astype(np.float32)
    mask = values[:, 220:] >= .75
    parts = []
    for channel, needed, weight in GROUPS:
        points = values[:, channel]
        velocity = np.diff(points, axis=0, prepend=points[:1])
        valid = mask[:, needed].all(axis=1)
        motion_valid = valid & np.r_[False, valid[:-1]]
        parts.append((points, velocity, valid, motion_valid, weight))
    return parts


def geometry(left, right):
    shape = np.zeros((STEPS, STEPS))
    motion = np.zeros_like(shape)
    shape_weights = np.zeros_like(shape)
    motion_weights = np.zeros_like(shape)
    for (a, da, amask, av, weight), (b, db, bmask, bv, _) in zip(left, right, strict=True):
        both = amask[:, None] & bmask[None, :]
        mismatch = amask[:, None] ^ bmask[None, :]
        active = av[:, None] & bv[None, :]
        shape += weight * (np.minimum(cdist(a, b, "sqeuclidean") / a.shape[1], 4.) * both + .10 * mismatch)
        shape_weights += weight * (both | mismatch)
        motion += weight * np.minimum(cdist(da, db, "sqeuclidean") / da.shape[1], 1.) * active
        motion_weights += weight * active
    shape = np.divide(shape, shape_weights, out=np.full_like(shape, .2), where=shape_weights > 0)
    motion = np.divide(motion, motion_weights, out=np.zeros_like(motion), where=motion_weights > 0)
    return np.sqrt(shape), np.sqrt(motion)


def align(cost):
    row = np.full(STEPS + 1, np.inf)
    row[0] = 0
    for i in range(1, STEPS + 1):
        next_row = np.full(STEPS + 1, np.inf)
        for j in range(max(1, i - BAND), min(STEPS, i + BAND) + 1):
            distance = cost[i - 1, j - 1]
            next_row[j] = min(row[j - 1] + 2 * distance, row[j] + distance, next_row[j - 1] + distance)
        row = next_row
    return float(row[-1] / (2 * STEPS))


def ordered_subsequence(tokens, target):
    if len(tokens) < max(1, ceil(len(target) * .5)):
        return False
    positions = iter(target)
    return all(any(label == expected for expected in positions) for label in tokens)


def load_roles(name, required=("train", "val", "test")):
    manifest, *features = SOURCES[name]
    return read_utterances(manifest, features, False, required)[0]


def file_sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    roles = load_roles("expanded")
    outside = load_roles("composition")
    holdout = load_roles("old_unknowns", ("test",))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    labels, lookup, state = (checkpoint[key] for key in ("labels", "training_greek_references", "state_dict"))
    templates = list(lookup)
    model = ContinuousCTC(len(labels), state["mean"], state["scale"])
    model.load_state_dict(state)
    model.eval()
    assert len(templates) == 6
    assert set(templates) == {item.glosses for item in roles["train"]}
    assert {item.signer for item in roles["train"]} == {"1", "2", "4", "5"}
    assert {item.signer for item in roles["val"]} == {"3"}
    assert {item.signer for item in roles["test"]} == {"6", "7"}
    label_ids = {label: index for index, label in enumerate(labels)}
    encoded = [[label_ids[label] for label in template] for template in templates]
    targets = torch.tensor([label for template in encoded for label in template])
    lengths = [len(template) for template in encoded]
    prototypes = [(item.signer, templates.index(item.glosses), sequence(item.frames)) for item in roles["train"]]
    variants = {"pose": 0., "pose_motion": .8, "motion": 2.0}

    def evaluate(items, transform="original"):
        rows = []
        for item in items:
            frames = altered_frames(item, transform)
            n = len(frames)
            logits = model(frames[None], [n])[0]
            ctc, confidence = ctc_decode(logits, labels)
            baseline = lookup.get(ctc) if confidence >= checkpoint["confidence_threshold"] else None
            likelihood = exact_template_scores(logits, targets, lengths)
            candidate = int(likelihood.argmax())
            template = templates[candidate]
            text = lookup[template]
            # Every gloss must have visual support, including any gloss CTC omitted.
            token_peak = float(logits[:, encoded[candidate]].amax(dim=0).min().exp())
            # The decoder must prefer the original order over reversed frames.
            reversed_logits = model(frames.flip(0)[None], [n])[0]
            reverse = exact_template_scores(reversed_logits, targets, lengths)
            order_gain = float((likelihood[candidate] - reverse[candidate]) / n)
            query = sequence(frames)
            groups = {name: defaultdict(dict) for name in variants}
            for signer, label, prototype in prototypes:
                shape, motion = geometry(query, prototype)
                for name, weight in variants.items():
                    distance = align(shape + weight * motion)
                    group = groups[name][label]
                    group[signer] = min(distance, group.get(signer, np.inf))
            retrieval = {}
            for name, group in groups.items():
                # The two nearest references must come from different people.
                distances = [sum(sorted(group[index].values())[:2]) / 2 for index in range(len(templates))]
                ranked = np.argsort(distances)
                retrieval[name] = {
                    "winner": int(ranked[0]),
                    "distance": distances[candidate],
                    "margin": float(1 - distances[ranked[0]] / max(1e-8, distances[ranked[1]])),
                }
            real = transform == "original"
            correct = real and item.glosses == template and normalize_text(text) == normalize_text(item.text)
            rows.append({
                "id": item.id, "signer": item.signer, "transform": transform,
                "reference": item.text, "referenceGlosses": item.glosses,
                "ctc": ctc, "baseline": baseline, "baselineCorrect": bool(real and baseline
                    and ctc == item.glosses and normalize_text(baseline) == normalize_text(item.text)),
                "candidate": candidate, "text": text, "candidateGlosses": template, "correct": correct,
                "peak": token_peak, "orderGain": order_gain, "consistent": ordered_subsequence(ctc, template),
                "retrieval": retrieval,
            })
        return rows

    def recovers(row, gate):
        candidate = row["retrieval"][gate["variant"]]
        return (
            not row["baseline"] and row["consistent"] and row["orderGain"] >= gate["orderGain"]
            and row["peak"] >= gate["peak"] and candidate["winner"] == row["candidate"]
            and candidate["distance"] <= gate["distance"] and candidate["margin"] >= gate["margin"]
        )

    def count(rows, gate):
        recovery = [row for row in rows if gate and recovers(row, gate)]
        return {
            "clips": len(rows), "baselineCorrect": sum(row["baselineCorrect"] for row in rows),
            "baselineWrong": sum(bool(row["baseline"] and not row["baselineCorrect"]) for row in rows),
            "recoveredCorrect": sum(row["correct"] for row in recovery),
            "recoveredWrong": sum(not row["correct"] for row in recovery),
            "recoveredIds": [row["id"] for row in recovery],
            "retrievalWinnerCorrect": {name: sum(
                row["transform"] == "original" and row["referenceGlosses"] == templates[row["retrieval"][name]["winner"]]
                for row in rows) for name in variants},
        }

    dev_known = evaluate(roles["val"])
    dev_outside = evaluate([item for item in outside["val"] if item.glosses not in lookup])
    dev_controls = [row for mode in ("reversed", "shuffled", "static_midpoint") for row in evaluate(roles["val"], mode)]
    dev = [*dev_known, *dev_outside, *dev_controls]
    correct_retrievals = [row for row in dev_known if row["correct"] and not row["baseline"] and row["consistent"]]
    best = None
    for variant in variants:
        valid = [row for row in correct_retrievals if row["retrieval"][variant]["winner"] == row["candidate"]]
        distances = sorted({row["retrieval"][variant]["distance"] for row in valid})
        margins = sorted({0., .025, .05, .10, *(row["retrieval"][variant]["margin"] for row in valid)})
        peaks = sorted({.08, .15, .25, *(row["peak"] for row in valid)})
        orders = sorted({0., .01, .03, *(max(0., row["orderGain"]) for row in valid)})
        for distance, margin, peak, order in product(distances, margins, peaks, orders):
            gate = {"variant": variant, "distance": distance, "margin": margin, "peak": peak, "orderGain": order}
            accepted = [row for row in dev if recovers(row, gate)]
            if not accepted or any(not row["correct"] for row in accepted):
                continue
            key = (len(accepted), margin, peak, order, -distance)
            if best is None or key > best[0]:
                best = key, gate
    gate = best[1] if best else None
    print(json.dumps({"gate": gate, "development": count(dev_known, gate), "controls": count(dev_controls, gate)}), flush=True)
    known = evaluate(roles["test"])
    outside_test = evaluate([item for item in [*holdout["test"], *outside["test"]] if item.glosses not in lookup])
    controls = [row for mode in ("reversed", "shuffled", "static_midpoint") for row in evaluate(roles["test"], mode)]
    result = {
        "scope": "Diagnostic on archived utterances with supplied clip boundaries. All parameter selection uses signer 3 "
                 "only. Test signers 6 and 7 were held out of the frozen model training and this tuning. This does not "
                 "measure spontaneous dialogue or camera performance. Perturbations are synthetic negatives.",
        "method": "Full-template CTC likelihood, independent mask-aware DTW over pose and velocity, two source signers "
                  "per distance, and original temporal order plus retained compatible gloss evidence.",
        "checkpointSha256": file_sha(CHECKPOINT), "steps": STEPS, "band": BAND,
        "prototypeCount": len(prototypes), "prototypesPerTemplate": dict(zip(
            [lookup[template] for template in templates], [sum(item.glosses == template for item in roles["train"])
                                                         for template in templates])),
        "frozenGate": gate,
        "development": {"known": count(dev_known, gate), "outside": count(dev_outside, gate), "controls": count(dev_controls, gate)},
        "diagnostic": {"known": count(known, gate), "outside": count(outside_test, gate), "controls": count(controls, gate)},
        "rows": {"developmentKnown": dev_known, "known": known, "outside": outside_test, "controls": controls},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["diagnostic"]), flush=True)


if __name__ == "__main__":
    main()

