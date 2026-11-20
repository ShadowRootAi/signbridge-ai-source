"""Inspect tracking only on a bounded training subset; never inspect test results."""
import argparse
import json
from pathlib import Path

import numpy as np

from preprocess import frame_features, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--label", help="Optional semantic label filter within training only")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    training = {str(sample["id"]): sample for sample in manifest["samples"]
                if sample["split"] == "train" and (args.label is None or sample["raw_label"] == args.label)}
    found = 0
    with args.jsonl.open(encoding="utf-8-sig") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # Extraction may be appending its final line concurrently.
            sample = training.get(str(record["id"]))
            if sample is None:
                continue
            x = np.asarray(record["x"], dtype=np.float32)
            result = {"id": record["id"], "label": sample["raw_label"], "signer": sample["signer_id"],
                      "quality": record["quality"], "resampled_presence_mean_L_R_pose_face": x[:, -4:].mean(axis=0).tolist()}
            raw = record.get("raw")
            if raw:
                frames = raw["frames"] if isinstance(raw, dict) else raw
                per_frame = [frame_features(frame) for frame in frames]
                flags = np.asarray([features[-4:] for features, _ in per_frame])
                result["raw_frame_n"] = len(frames)
                result["raw_presence_counts_L_R_pose_face"] = flags.sum(axis=0).astype(int).tolist()
                result["raw_array_lengths"] = {key: sorted({len(frame[key]) for frame in frames}) for key in ("left", "right", "pose", "face")}
                shoulder_visibility = [[frame["pose"][i].get("visibility", 1.0) for i in (11, 12)] for frame in frames if len(frame["pose"]) >= 13]
                result["shoulder_visibility_min_max"] = [float(np.min(shoulder_visibility)), float(np.max(shoulder_visibility))] if shoulder_visibility else None
                result["valid_hands_with_invalid_pose_n"] = int((((flags[:, 0] + flags[:, 1]) > 0) & (flags[:, 2] == 0)).sum())
                result["valid_pose_without_hands_n"] = int(((flags[:, 2] == 1) & ((flags[:, 0] + flags[:, 1]) == 0)).sum())
                python_x, python_quality = preprocess(frames)
                result["python_browser_max_abs_error"] = float(np.max(np.abs(python_x - x)))
                result["python_quality"] = python_quality
            else:
                result["raw_diagnosis"] = "unavailable_no_raw_frames"
            print(json.dumps(result, ensure_ascii=False))
            found += 1
            if found >= args.limit:
                break


if __name__ == "__main__":
    main()
