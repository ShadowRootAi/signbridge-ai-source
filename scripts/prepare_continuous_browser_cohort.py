"""Prepare a bounded continuous-GSL browser replay cohort from cached RGB."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg

from continuous_dataset import ROOT, read_annotation_maps


MANIFESTS = {
    "known_templates": "data/continuous-v1-expanded/manifest.json",
    "unseen_templates": "data/continuous-v1-holdout/manifest.json",
    "composition": "data/continuous-composition-pilot/manifest.json",
}
REFERENCE_CONFLICT = "police1_signer6_rep1_sentences__sentences0003"


def select_samples():
    selected = []
    excluded = []
    annotation_maps = {scenario: read_annotation_maps(scenario) for scenario in ("police1", "police2")}
    for corpus, path in MANIFESTS.items():
        manifest = json.loads((ROOT / path).read_text(encoding="utf-8"))
        if not manifest["complete"]:
            raise ValueError(f"Incomplete RGB cache: {path}")
        for row in manifest["samples"]:
            if row["id"] == REFERENCE_CONFLICT:
                excluded.append({"id": row["id"], "referenceText": row["text"],
                                 "referenceGlosses": row["glosses"],
                                 "reason": "Greek reference conflicts with three same-template repetitions."})
                continue
            repetition = int(re.search(r"_rep(\d+)_sentences__", row["id"])[1])
            if row["signer"] == 3 and repetition == 1 and row["split"] == "val":
                group = "calibration"
            elif row["signer"] in (6, 7) and row["split"] == "test":
                if corpus == "composition" and row.get("composition") == "unseen_template":
                    group = "unseen_gloss_order"
                elif repetition == 2:
                    group = "composition_known_templates" if corpus == "composition" else corpus
                else:
                    continue
            else:
                continue

            scenario, source = row["source_path"].split("/", 1)
            glosses, texts = annotation_maps[scenario]
            if row["glosses"] != glosses[row["source_path"]] or row["text"] != texts[source]:
                raise ValueError(f"Cached reference differs from official source: {row['id']}")
            selected.append((group, repetition, row))

    expected_counts = {"calibration": 9, "known_templates": 12, "unseen_templates": 4,
                       "composition_known_templates": 6, "unseen_gloss_order": 4}
    if Counter(group for group, _, _ in selected) != expected_counts or len(excluded) != 1:
        raise ValueError("Unexpected replay cohort or reference conflict coverage")
    if len({row["id"] for _, _, row in selected}) != len(selected):
        raise ValueError("Duplicate recordings in replay cohort")
    return sorted(selected, key=lambda sample: (sample[0], sample[2]["id"])), excluded


def encode(group, repetition, row, destination):
    frames = [ROOT / path for path in row["frame_paths"]]
    if not frames or len(frames) != row["frame_count"] or row["fps"] != 30:
        raise ValueError(f"Invalid frame metadata: {row['id']}")
    source_sha = sha256()
    source_images = []
    for frame in frames:
        image = frame.read_bytes()
        source_images.append(image)
        source_sha.update(sha256(image).digest())
    if source_sha.hexdigest() != row["sha256"]:
        raise ValueError(f"Cached source frame sequence differs: {row['id']}")

    target = destination / (row["id"] + ".mp4")
    pending = target.with_suffix(".pending.mp4")
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-xerror", "-y",
        "-threads", "1", "-f", "image2pipe", "-framerate", "30", "-c:v", "mjpeg", "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1", "-an",
        "-movflags", "+faststart", str(pending),
    ], input=b"".join(source_images), check=True, capture_output=True)
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(pending))
    if count != len(frames) or abs(duration - count / 30) > 0.011:
        raise ValueError(f"Frame count or duration changed: {row['id']}")
    pending.replace(target)
    return {
        "id": row["id"], "group": group, "signer": row["signer"], "repetition": repetition,
        "sourcePath": row["source_path"], "referenceText": row["text"], "referenceGlosses": row["glosses"],
        "path": target.name, "framePaths": row["frame_paths"], "frames": count, "fps": row["fps"],
        "sourceRgbSha256": row["sha256"], "mp4Sha256": sha256(target.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime-tmp/continuous-rgb-audit")
    args = parser.parse_args()
    selected, excluded = select_samples()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        clips = list(executor.map(lambda sample: encode(*sample, args.output), selected))
    counts = dict(sorted(Counter(clip["group"] for clip in clips).items()))
    manifest = {
        "schemaVersion": 1,
        "source": "https://zenodo.org/records/4756317",
        "license": "CC-BY-4.0",
        "scope": "Previously evaluated CERTH studio recordings. Clips have original utterance boundaries. Diagnostic browser replay only.",
        "selection": "Calibration signer 3 rep1. Diagnostic signers 6/7 rep2, plus both reps of the unseen composition template.",
        "sourceManifestSha256": {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in MANIFESTS.values()},
        "excludedReferences": excluded,
        "counts": counts,
        "clips": clips,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
