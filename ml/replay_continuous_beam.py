"""Replay a fixed continuous CTC checkpoint with unconstrained prefix beam search."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from train_continuous import ContinuousCTC, ctc_decode, edit_counts, normalize_text, read_utterances


BEAM_WIDTH = 12
BLANK_OFFSETS = (0.0, -0.5, -1.0, -1.5, -2.0)
NEG_INF = -math.inf


def log_add(left, right):
    if left < right:
        left, right = right, left
    return left if right == NEG_INF else left + math.log1p(math.exp(right - left))


def prefix_beam(scores, width, blank_offset):
    # Maintain disjoint probability mass ending in blank or nonblank for each
    # collapsed prefix. Subtracting each new frame's softmax denominator would
    # add the same constant to all path scores, so ranking needs no renormalization.
    active = {(): (0.0, NEG_INF)}
    for frame in scores:
        next_beams = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (blank, nonblank) in active.items():
            total = log_add(blank, nonblank)
            same = next_beams[prefix]
            same[0] = log_add(same[0], total + frame[0] + blank_offset)
            previous = prefix[-1] if prefix else 0
            for index, value in enumerate(frame[1:], 1):
                if index == previous:
                    same[1] = log_add(same[1], nonblank + value)
                    if blank == NEG_INF:
                        continue
                    origin = blank
                else:
                    origin = total
                extended = next_beams[prefix + (index,)]
                extended[1] = log_add(extended[1], origin + value)
        active = dict(sorted(next_beams.items(), key=lambda item: log_add(*item[1]), reverse=True)[:width])
    return next(iter(active))


def check_beam_repeats():
    probabilities = ((0.03, 0.91, 0.06), (0.94, 0.04, 0.02), (0.05, 0.88, 0.07))
    scores = [[math.log(value) for value in frame] for frame in probabilities]
    for blank_offset in BLANK_OFFSETS:
        totals = defaultdict(float)
        for path in itertools.product(range(3), repeat=3):
            collapsed = tuple(index for index, _ in itertools.groupby(path) if index)
            totals[collapsed] += math.prod(probabilities[t][index] for t, index in enumerate(path)) * math.exp(
                blank_offset * path.count(0)
            )
        assert max(totals, key=totals.get) == prefix_beam(scores, 32, blank_offset)
    assert prefix_beam(scores, 32, 0.0) == (1, 1)


@torch.inference_mode()
def collect_logits(model, items):
    model.eval()
    return [model(item.frames.unsqueeze(0), [len(item.frames)])[0] for item in items]


def replay(items, scores, labels, lookup, blank_offset=None):
    predictions = []
    for item, clip_scores in zip(items, scores, strict=True):
        if blank_offset is None:
            tokens, _ = ctc_decode(clip_scores, labels)
        else:
            tokens = tuple(labels[index] for index in prefix_beam(clip_scores.tolist(), BEAM_WIDTH, blank_offset))
        text = lookup.get(tokens)
        substitutions, deletions, insertions = edit_counts(item.glosses, tokens)
        predictions.append({
            "id": item.id, "signer": item.signer, "reference_glosses": item.glosses, "predicted_glosses": tokens,
            "reference_text": item.text, "predicted_text": text, "gloss_sequence_exact": item.glosses == tokens,
            "greek_reference_exact": bool(text and normalize_text(text) == normalize_text(item.text)),
            "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        })
    count = len(predictions)
    edits = {key: sum(row[key] for row in predictions) for key in ("substitutions", "deletions", "insertions")}
    gloss_count = sum(len(row["reference_glosses"]) for row in predictions)
    accepted = sum(row["predicted_text"] is not None for row in predictions)
    correct = sum(row["greek_reference_exact"] for row in predictions)
    return {
        "utterances": count, "reference_glosses": gloss_count, **edits, "gloss_wer": sum(edits.values()) / gloss_count,
        "exact_sequences": sum(row["gloss_sequence_exact"] for row in predictions),
        "accepted_texts": accepted, "abstentions": count - accepted, "correct_texts": correct,
        "wrong_texts": accepted - correct, "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/continuous-v1-expanded/manifest.json"))
    parser.add_argument("--features", type=Path, nargs="+", default=[
        Path("data/continuous-v1/features.jsonl"), Path("data/continuous-v1-extra/features.jsonl"),
    ])
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/continuous-v1-expanded/ctc.pt"))
    parser.add_argument("--holdout-manifest", type=Path, default=Path("data/continuous-v1-holdout/manifest.json"))
    parser.add_argument("--holdout-features", type=Path, default=Path("data/continuous-v1-holdout/features.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/continuous-v1-expanded/beam-posthoc.json"))
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite the previous replay report")

    torch.set_num_threads(2)
    check_beam_repeats()
    roles, _ = read_utterances(args.manifest, args.features, False)
    holdout, _ = read_utterances(args.holdout_manifest, [args.holdout_features], False, ("test",))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["confidence_threshold"] != 0:
        raise ValueError("Greek lookup parity requires the original zero threshold for this replay")
    if {item.id for items in roles.values() for item in items} != set(checkpoint["prior_ids"]):
        raise ValueError("The checkpoint is not from the requested cohort")
    labels, lookup, state = (checkpoint[key] for key in ("labels", "training_greek_references", "state_dict"))
    model = ContinuousCTC(len(labels), state["mean"], state["scale"])
    model.load_state_dict(state)

    validation_scores = collect_logits(model, roles["val"])
    validation = {offset: replay(roles["val"], validation_scores, labels, lookup, offset) for offset in BLANK_OFFSETS}
    selected = min(BLANK_OFFSETS, key=lambda offset: (validation[offset]["gloss_wer"],
                                                     -validation[offset]["exact_sequences"], abs(offset)))
    fixed_conditions = {"greedy": None, "beam_unbiased": 0.0, "beam_validation_selected": selected}
    outcomes = {}
    for name, items in (("expanded_test_reused", roles["test"]), ("genuine_negatives_reused", holdout["test"])):
        scores = collect_logits(model, items)
        outcomes[name] = {method: replay(items, scores, labels, lookup, offset) for method, offset in fixed_conditions.items()}
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "status": "posthoc experiment on previously evaluated test sets; no deployment changes",
        "beam_width": BEAM_WIDTH, "blank_logit_offsets": BLANK_OFFSETS,
        "checkpoint_confidence_threshold": checkpoint["confidence_threshold"],
        "selection": "validation gloss WER, then exact sequences, then smallest absolute offset",
        "selected_offset": selected, "repeated_token_exhaustive_check": "passed",
        "validation_greedy": replay(roles["val"], validation_scores, labels, lookup),
        "validation_beam": validation, **outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_offset": selected, **{
        cohort: {method: {k: v for k, v in result.items() if k != "predictions"} for method, result in methods.items()}
        for cohort, methods in outcomes.items()
    }}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
