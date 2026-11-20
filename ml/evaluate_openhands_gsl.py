"""Measure a released OpenHands model on its native Greek pose inputs, one clip at a time."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/cache/openhands-gsl"))
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--model", choices=("lstm", "stgcn"), default="lstm")
    parser.add_argument("--limit", type=int, default=0, help="Only for smoke checks")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.poses = args.poses or args.cache / "official-val-poses.npz"
    output = args.output or Path(f"reports/pretrained-gsl/{args.model}-native-poses.json")
    if output.exists() or args.limit < 0:
        parser.error("Choose a new report path and a nonnegative limit")
    torch.set_num_threads(1)
    if args.model == "lstm":
        from openhands_gsl import load_openhands_gsl as load, prepare_pose as prepare
        checkpoint = args.cache / "gsl_lstm.zip"
    else:
        from openhands_gsl_stgcn import load_openhands_gsl_stgcn as load, prepare_stgcn_pose as prepare
        checkpoint = args.cache / "gsl_stgcn.zip"
    model, classes = load(args.cache)
    class_index = {label: i for i, label in enumerate(classes)}
    if len(classes) != 310 or len(class_index) != 310:
        raise ValueError("Expected all 310 unique source labels")
    with np.load(args.poses, allow_pickle=False) as data:
        poses, offsets, ids, glosses = [data[key] for key in ("keypoints", "offsets", "ids", "glosses")]
        source_split = str(data["split"].item())
    if (poses.ndim != 3 or poses.shape[1:] != (75, 3) or len(ids) != len(glosses)
            or len(set(ids)) != len(ids) or len(offsets) != len(ids) + 1
            or offsets[0] != 0 or offsets[-1] != len(poses) or np.any(np.diff(offsets) <= 0)
            or not set(glosses) <= set(classes)):
        raise ValueError("Bad native pose index or labels")
    signers = sorted({re.search(r"_signer(\d+)_", identity).group(1) for identity in ids})
    counts = defaultdict(lambda: [0, 0, 0])
    predictions, times = [], []
    count = min(args.limit or len(ids), len(ids))
    with torch.inference_mode():
        for i in range(count):
            started = time.perf_counter()
            x = prepare(poses[offsets[i]:offsets[i+1]])
            logits = model(x.unsqueeze(0) if args.model == "lstm" else x)
            if logits.shape != (1, 310) or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid logits for {ids[i]}")
            scores, top = torch.softmax(logits[0], dim=0).topk(5)
            times.append(1000 * (time.perf_counter() - started))
            guess = [classes[j] for j in top.tolist()]
            target = str(glosses[i])
            counts[target][0] += 1
            counts[target][1] += guess[0] == target
            counts[target][2] += target in guess
            predictions.append({
                "id": str(ids[i]), "reference_gloss": target, "top5": guess, "top5_scores": scores.tolist(),
            })
            if (i + 1) % 500 == 0:
                print(f"Scored {i + 1}/{count}", flush=True)
    totals = np.sum(list(counts.values()), axis=0)
    summary = {
        "clips": count, "output_classes": len(classes), "observed_reference_classes": len(counts),
        "top1_correct": int(totals[1]), "top5_correct": int(totals[2]),
        "top1_accuracy": float(totals[1] / count), "top5_recall": float(totals[2] / count),
        "macro_top1_recall": float(np.mean([correct / n for n, correct, _ in counts.values()])),
        "classes_with_top1_hit": sum(correct > 0 for _, correct, _ in counts.values()),
        "classes_with_at_least_5_examples_and_80_percent_top1": sum(n >= 5 and correct / n >= 0.8 for n, correct, _ in counts.values()),
        "median_cpu_preprocessing_and_prediction_ms": float(np.median(times)),
    }
    with checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "model": args.model, "checkpoint_sha256": checkpoint_sha256,
        "source_split": source_split, "source_signers": signers, "evaluated_clips": count,
        "scope": "Original published Greek landmark clips, known vocabulary, isolated and already segmented, single-clip inference; no webcam, unknown negatives or continuous translation.",
        "summary": summary,
        "per_gloss": {name: {"clips": n, "top1_correct": c1, "top5_correct": c5} for name, (n, c1, c5) in sorted(counts.items())},
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
