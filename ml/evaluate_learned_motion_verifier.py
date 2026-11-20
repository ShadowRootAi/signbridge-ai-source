"""Train a signer-excluded motion verifier for the frozen six-phrase CTC model."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from evaluate_continuous_transfer import CHECKPOINT, exact_template_scores
from evaluate_motion_verifier import align, geometry, load_roles, sequence
from train_continuous import ContinuousCTC, altered_frames, ctc_decode, normalize_text


REPORT = Path("reports/continuous-v1-expanded/learned-motion-verifier.json")
CONTROLS = ("reversed", "shuffled", "static_midpoint")
WEIGHTS = (0., .8, 2.)


def distances(query, prototypes, signer, classes):
    by_candidate = [defaultdict(lambda: np.full(len(WEIGHTS), np.inf)) for _ in range(classes)]
    for ref_signer, candidate, ref in prototypes:
        if signer == ref_signer:
            continue
        position, velocity = geometry(query, ref)
        current = by_candidate[candidate][ref_signer]
        for i, weight in enumerate(WEIGHTS):
            current[i] = min(current[i], align(position + weight * velocity))
    data = np.stack([np.sort(np.stack(list(group.values())), axis=0)[:2].mean(axis=0) for group in by_candidate])
    rows = []
    for candidate, values in enumerate(data):
        alternative = np.delete(data, candidate, axis=0).min(axis=0)
        # Absolute distance and relative separation across three motion metrics.
        rows.append(np.r_[values, values - alternative])
    return np.stack(rows)


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    originals, others = load_roles("expanded"), load_roles("composition")
    holdout = load_roles("old_unknowns", ("test",))["test"]
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"]
    labels, lookup = ckpt["labels"], ckpt["training_greek_references"]
    templates = list(lookup)
    baseline = ContinuousCTC(len(labels), state["mean"], state["scale"])
    baseline.load_state_dict(state)
    baseline.eval()
    prototypes = [(item.signer, templates.index(item.glosses), sequence(item.frames)) for item in originals["train"]]
    labels_encoded = [[labels.index(gloss) for gloss in template] for template in templates]
    targets = torch.tensor([gloss for phrase in labels_encoded for gloss in phrase])
    lengths = [len(phrase) for phrase in templates]

    x, y = [], []
    train = [*originals["train"], *(item for item in others["train"] if item.glosses not in lookup)]
    for item in train:
        for transform in ("original", "reversed"):
            features = distances(sequence(altered_frames(item, transform)), prototypes, item.signer, len(templates))
            x.extend(features)
            y.extend(int(transform == "original" and item.glosses == template) for template in templates)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=41))
    classifier.fit(np.asarray(x), np.asarray(y))
    print(f"Fitted on {len(x)} motion comparisons excluding each source signer", flush=True)

    def evaluate(items, mode="original"):
        rows = []
        for item in items:
            frames = altered_frames(item, mode)
            count = len(frames)
            logits = baseline(frames[None], [count])[0]
            decoded, confidence = ctc_decode(logits, labels)
            first = lookup.get(decoded) if confidence >= ckpt["confidence_threshold"] else None
            likely = exact_template_scores(logits, targets, lengths)
            candidate = int(likely.argmax())
            target = templates[candidate]
            text = lookup[target]
            backwards = baseline(frames.flip(0)[None], [count])[0]
            reverse_likelihood = exact_template_scores(backwards, targets, lengths)
            score = classifier.predict_proba(distances(sequence(frames), prototypes, None, len(templates)))[:, 1]
            real = mode == "original"
            rows.append({
                "id": item.id, "signer": item.signer, "mode": mode, "reference": item.text,
                "candidateText": text, "motionProbability": float(score[candidate]),
                "motionAgrees": int(score.argmax()) == candidate,
                "motionRankCorrect": bool(real and templates[int(score.argmax())] == item.glosses),
                "orderGain": float((likely[candidate] - reverse_likelihood[candidate]) / count),
                "minimumPeak": float(logits[:, labels_encoded[candidate]].amax(dim=0).min().exp()),
                "baselineText": first,
                "baselineCorrect": bool(real and first and decoded == item.glosses
                    and normalize_text(first) == normalize_text(item.text)),
                "candidateCorrect": bool(real and target == item.glosses and normalize_text(text) == normalize_text(item.text)),
            })
        return rows

    dev = {
        "known": evaluate(originals["val"]),
        "outside": evaluate([item for item in others["val"] if item.glosses not in lookup]),
        "controls": [row for mode in CONTROLS for row in evaluate(originals["val"], mode)],
    }

    def eligible(row):
        return not row["baselineText"] and row["motionAgrees"] and row["orderGain"] > 0

    def accepts(row, policy):
        return eligible(row) and row["motionProbability"] >= policy["minimumMotionProbability"] \
            and row["minimumPeak"] >= policy["minimumTokenPeak"]

    possible = []
    for peak in (.0, .005, .02, .05, .10):
        all_rows = [row for rows in dev.values() for row in rows if eligible(row) and row["minimumPeak"] >= peak]
        wrong = max((row["motionProbability"] for row in all_rows if not row["candidateCorrect"]), default=0.)
        minimum = max(.5, float(np.nextafter(wrong, np.inf)))
        gate = {"minimumMotionProbability": minimum, "minimumTokenPeak": peak}
        hits = sum(accepts(row, gate) and row["candidateCorrect"] for row in dev["known"])
        possible.append((hits, peak, minimum, gate))
    chosen = max(possible, key=lambda item: item[:3])[3]

    test = {
        "known": evaluate(originals["test"]),
        "outside": evaluate([item for item in [*holdout, *others["test"]] if item.glosses not in lookup]),
        "controls": [row for mode in CONTROLS for row in evaluate(originals["test"], mode)],
    }

    def summarize(groups):
        summary = {}
        for name, rows in groups.items():
            restored = [row for row in rows if accepts(row, chosen)]
            summary[name] = {
                "clips": len(rows), "baselineCorrect": sum(row["baselineCorrect"] for row in rows),
                "baselineWrong": sum(bool(row["baselineText"] and not row["baselineCorrect"]) for row in rows),
                "motionRankCorrect": sum(row["motionRankCorrect"] for row in rows),
                "recoveredCorrect": sum(row["candidateCorrect"] for row in restored),
                "recoveredWrong": sum(not row["candidateCorrect"] for row in restored),
                "recoveredIds": [row["id"] for row in restored],
            }
        return summary

    result = {
        "scope": "Archived diagnostic. Logistic motion verifier fit on source training signers 1/2/4/5 using "
                 "other-signer references only. Recovery floors chosen on signer 3. Signers 6/7 are replayed "
                 "once after calibration. Real off-scope and artificial order controls are distinct.",
        "trainUtterances": len(train), "motionTrainingComparisons": len(x), "motionPositiveComparisons": sum(y),
        "frozenGate": chosen, "development": summarize(dev), "diagnostic": summarize(test), "rows": test,
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("frozenGate", "development", "diagnostic")}), flush=True)


if __name__ == "__main__":
    main()

