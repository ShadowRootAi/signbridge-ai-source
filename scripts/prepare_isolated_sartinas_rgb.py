"""Stage exact author-timed Sartinas sign intervals for a browser diagnostic."""

from collections import Counter
import subprocess

import imageio_ffmpeg

from acquire_fresh_continuous import DATA, ROOT, digest, read_json, write_manifest
from prepare_sartinas_appointment_holdout import PAGE, PAPER, RECORDINGS
from prepare_sartinas_dialogue_fit import annotation_tiers


OUTPUT = DATA / "isolated-expansion-sartinas"
FPS = 30
SELECTION = {
    "n1": ("a13", "a57", "a129"),
    "n6": ("a2", "a221", "a224"),
    "n7": ("a49", "a106", "a37"),
    "n8": ("a162", "a51", "a340"),
}


def plan():
    selected = []
    sources = {}
    for signer, ids in SELECTION.items():
        source = RECORDINGS[signer]
        annotation = source.media / f"{source.name}.eaf"
        provenance = source.data / "video-manifest.json"
        [video] = read_json(provenance)["samples"]
        if video["id"] != source.id or video["fps"] != FPS or video["elan_sha256"] != digest(annotation):
            raise ValueError(f"Annotation and decoded source do not match: {source.name}")
        _, _, _, intervals = annotation_tiers(annotation.read_bytes(), source.name)
        sources.update({path.relative_to(ROOT).as_posix(): digest(path) for path in (annotation, provenance)})
        frames = video["frame_paths"]
        for index, interval in enumerate(intervals):
            annotation_id = interval["elan_annotation_id"]
            if annotation_id not in ids:
                continue
            start, end = interval["start_ms"], interval["end_ms"]
            if ((index and intervals[index - 1]["end_ms"] > start)
                    or (index + 1 < len(intervals) and intervals[index + 1]["start_ms"] < end)):
                raise ValueError(f"Overlapping neighbor sign intervals: {source.name}/{annotation_id}")
            first, stop = (start * FPS + 999) // 1000, (end * FPS + 999) // 1000
            clip = frames[first:stop]
            if len(clip) != stop - first or len(clip) < 10:
                raise ValueError(f"Incomplete clip interval: {source.name}/{annotation_id}")
            if any(not (ROOT / frame).is_file() for frame in clip):
                raise ValueError(f"Missing cached RGB frames: {source.name}/{annotation_id}")
            identity = f"{source.id}-{annotation_id}"
            filename = identity + ".mp4"
            selected.append({
                "id": identity, "path": filename, "video_path": (OUTPUT / filename).relative_to(ROOT).as_posix(),
                "group": "external_timed_gloss_diagnostic", "gloss": interval["label"], "on_roster": True,
                "signer": f"sartinas:{signer}", "source_path": source.name + ".avi",
                "source_rgb_sha256": video["source_video_sha256"],
                "annotation_id": annotation_id, "elan_sha256": video["elan_sha256"],
                "start_ms": start, "end_ms": end, "first_frame_zero_based": first,
                "fps": FPS, "frames": len(clip), "frame_paths": clip,
                "frame_sha256": [digest(ROOT / frame) for frame in clip],
            })
    if (Counter(row["signer"] for row in selected) != {f"sartinas:{s}": 3 for s in SELECTION}
            or len({row["gloss"] for row in selected}) != 7):
        raise ValueError("The fixed 12-clip/seven-gloss selection changed")
    return {
        "schema_version": 1, "source": PAGE, "paper": PAPER, "complete": False,
        "scope": "External diagnostic on 12 author-timed signs embedded in four scripted Sartinas S10 takes. "
                 "These continuous recordings were used in other project evaluations. "
                 "Clip duration and contextual transitions can differ from standalone signs.",
        "selection": "Seven exact official Sartinas gloss spellings matching expansion labels. "
                     "Three predetermined, nonoverlapping ELAN annotations per signer. "
                     "No model output used to choose the clips.",
        "timing": "Use original decoded 30 Hz frame timestamps t where start_ms <= t < end_ms. No added frames.",
        "source_hashes": sources, "rows": selected,
    }


def encode(row):
    directory = (ROOT / row["frame_paths"][0]).parent
    target = OUTPUT / row["path"]
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
        "-framerate", str(FPS), "-start_number", str(row["first_frame_zero_based"] + 1),
        "-i", str(directory / "frame-%06d.jpg"), "-frames:v", str(row["frames"]),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ], check=True, timeout=30)
    decoded = imageio_ffmpeg.read_frames(target, input_params=["-threads", "1"])
    try:
        metadata = next(decoded)
        if metadata["fps"] != FPS or sum(1 for _ in decoded) != row["frames"]:
            raise ValueError(f"Encoded MP4 differs from the annotation timeline: {row['id']}")
    finally:
        decoded.close()
    return {**row, "sha256": digest(target), "bytes": target.stat().st_size}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = plan()
    frozen = OUTPUT / "manifest.planned.json"
    if frozen.exists() and read_json(frozen) != manifest:
        raise ValueError("The frozen author-interval plan changed")
    if not frozen.exists():
        write_manifest(frozen, manifest)
    rows = [encode(row) for row in manifest["rows"]]
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate staged MP4s")
    write_manifest(OUTPUT / "manifest.json", {
        **manifest, "complete": True, "plan_sha256": digest(frozen), "rows": rows, "samples": rows,
    })
    print(f"Prepared {len(rows)} Sartinas MP4s with {len({row['gloss'] for row in rows})} distinct glosses")


if __name__ == "__main__":
    main()
