"""Prepare real cached single-gloss MP4s for the larger browser roster audit."""

from collections import Counter
import re
import subprocess
from zipfile import ZipFile

import imageio_ffmpeg

from acquire_fresh_continuous import DATA, ROOT, SOURCE, SPLIT, SUPPLEMENTARY, digest, read_json, write_manifest
from plan_continuous_composition import read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "isolated-expansion-rgb"
AUDIT = ROOT / "reports/pretrained-gsl/rgb-audit-plan.json"
PRIOR = ROOT / "reports/pretrained-gsl/rgb-final-browser.json"
SOURCE_MANIFESTS = (DATA / "manifest.json", DATA / "open-set-v2/manifest.json", DATA / "v3/manifest.json")
PICKS = {
    "ΟΚ": (
        "kep5_signer1_rep3_sentences/sentences0011",
        "kep5_signer1_rep4_sentences/sentences0011",
        "kep5_signer2_rep3_sentences/sentences0011",
        "kep5_signer4_rep4_sentences/sentences0011",
        "kep5_signer5_rep3_sentences/sentences0011",
        "kep5_signer5_rep5_sentences/sentences0011",
    ),
    "ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ": (
        "police2_signer1_rep2_sentences/sentences0003",
        "police2_signer2_rep3_sentences/sentences0003",
        "police2_signer3_rep5_sentences/sentences0003",
        "police2_signer4_rep2_sentences/sentences0003",
        "police2_signer5_rep3_sentences/sentences0003",
        "police2_signer5_rep5_sentences/sentences0003",
    ),
}


def build_plan():
    official, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timing, _ = annotation_timestamps(archive)
    audit = {row["source_path"]: row for row in read_json(AUDIT)["certh_cached"]}
    source = {
        row["source_path"]: row
        for path in SOURCE_MANIFESTS for row in read_json(path)["samples"]
    }
    evaluated = {row["id"] for row in read_json(PRIOR)["rows"]}
    rows = []
    for label, picks in PICKS.items():
        for identity in picks:
            row, raw, annotation = audit[identity], source[identity], timing[identity]
            if (row["id"] in evaluated or row["on_roster"]
                    or row["gloss"] != label or row["annotation_raw_glosses"] != label
                    or official[identity] != (label,) or annotation["raw_gloss"] != label
                    or greek[identity] != annotation["translation"]):
                raise ValueError(f"Prior browser exposure or reference mismatch: {identity}")
            frames = [ROOT / name for name in row["frame_paths"]]
            hashes = [digest(path) for path in frames]
            if (hashes != raw["frame_sha256"] or len(frames) != annotation["implied_inclusive_frame_count"]
                    or any(not re.fullmatch(rf"frame_{index:04d}\.jpg", frame.name)
                           for index, frame in enumerate(frames))):
                raise ValueError(f"Frame provenance mismatch: {identity}")
            target = OUTPUT / (row["id"] + ".mp4")
            rows.append({
                **row, "on_roster": True, "previous_on_roster": False,
                "group": "cached_new_label_rgb_diagnostic", "text": greek[identity],
                "previous_project_rgb_exposure": True, "previous_96_video_browser_exposure": False,
                "frames": len(frames), "frame_sha256": hashes,
                "path": target.name, "video_path": target.relative_to(ROOT).as_posix(),
            })
    return {
        "schema_version": 1, "source": SOURCE, "license": "CC-BY-4.0", "complete": False,
        "scope": "Twelve previously cached complete CERTH utterance clips with exactly one official gloss. "
                 "Two labels are newly allowed. Source RGB appeared in earlier project work but these IDs "
                 "were absent from the earlier 96-clip browser benchmark. Model corpus identity is shared.",
        "selection": "Fixed signer and source ID selection without inspecting model scores on these RGB inputs.",
        "inputs_sha256": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (AUDIT, PRIOR, SPLIT, SUPPLEMENTARY, *SOURCE_MANIFESTS)
        },
        "rows": rows,
    }


def encode(row):
    target = OUTPUT / row["path"]
    first_frame = ROOT / row["frame_paths"][0]
    if first_frame.name != "frame_0000.jpg":
        raise ValueError(f"Unexpected input frame sequence: {row['id']}")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
        "-framerate", str(row["fps"]), "-i", str(first_frame.parent / "frame_%04d.jpg"),
        "-frames:v", str(row["frames"]), "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    subprocess.run(command, check=True, timeout=30)
    decoded = imageio_ffmpeg.read_frames(target, input_params=["-threads", "1"])
    try:
        metadata = next(decoded)
        if metadata["fps"] != row["fps"] or sum(1 for _ in decoded) != row["frames"]:
            raise ValueError(f"Encoded MP4 differs from the source timeline: {row['id']}")
    finally:
        decoded.close()
    video_hash = digest(target)
    return {**row, "sha256": video_hash, "video_sha256": video_hash, "bytes": target.stat().st_size}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    target = OUTPUT / "manifest.planned.json"
    if target.exists() and read_json(target) != plan:
        raise ValueError("The frozen isolated RGB plan changed")
    if not target.exists():
        write_manifest(target, plan)
    rows = [encode(row) for row in plan["rows"]]
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Two staged MP4s have identical content")
    write_manifest(OUTPUT / "manifest.json", {
        **plan, "complete": True, "plan_sha256": digest(target), "rows": rows, "samples": rows,
    })
    print(f"Prepared {len(rows)} source-derived MP4s with counts {Counter(row['gloss'] for row in rows)}")


if __name__ == "__main__":
    main()
