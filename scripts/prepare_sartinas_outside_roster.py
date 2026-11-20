"""Freeze 24 author-timed Sartinas glosses outside OpenHands' 310-label map."""

from collections import Counter
import re
import unicodedata

from acquire_fresh_continuous import DATA, ROOT, digest, read_json, write_manifest
from prepare_isolated_sartinas_rgb import FPS, encode
from prepare_sartinas_appointment_holdout import PAGE, PAPER, RECORDINGS
from prepare_sartinas_dialogue_fit import annotation_tiers


OUTPUT = DATA / "isolated-sartinas-outside-roster"
CONFIG = ROOT / "publication-recognition-200/dist/__signbridge/openhands/stgcn-onnx.json"
CONFIG_SHA256 = "2d87351438ca308f72291bd35287749c209c7a33fa9baa2f8501749860409596"
DISTINCT_PER_SIGNER = 6


def greek_key(label):
    return "".join(c for c in unicodedata.normalize("NFD", label.casefold())
                   if c.isalpha() and not unicodedata.combining(c))


def within_one_edit(a, b):
    if len(a) == len(b):
        return sum(left != right for left, right in zip(a, b)) <= 1
    short, long = sorted((a, b), key=len)
    return len(long) == len(short) + 1 and any(short == long[:i] + long[i + 1:] for i in range(len(long)))


def plan():
    if digest(CONFIG) != CONFIG_SHA256:
        raise ValueError("Frozen OpenHands class-map changed")
    labels = read_json(CONFIG)["labels"]
    if len(set(labels)) != 310:
        raise ValueError("Expected 310 classes")
    excluded = {greek_key(part) for label in labels for part in (label, *label.split("_"))}
    source_hashes = {CONFIG.relative_to(ROOT).as_posix(): CONFIG_SHA256}
    rows = []
    for signer, recording in RECORDINGS.items():
        annotations = recording.media / f"{recording.name}.eaf"
        provenance = recording.data / "video-manifest.json"
        [video] = read_json(provenance)["samples"]
        if (video["id"] != recording.id or video["fps"] != FPS
                or digest(annotations) != video["elan_sha256"] or video["elan_sha256"] != recording.eaf_sha256):
            raise ValueError(f"Unverified recording {recording.name}")
        source_hashes.update({path.relative_to(ROOT).as_posix(): digest(path) for path in (annotations, provenance)})
        _, _, _, intervals = annotation_tiers(annotations.read_bytes(), recording.name)
        count = 0
        for position, interval in enumerate(intervals):
            label = interval["label"]
            if len(label) < 3 or not re.fullmatch(r"[Α-ΩΆΈΉΊΌΎΏΪΫ]+", label):
                continue
            key = greek_key(label)
            if any(within_one_edit(key, other) for other in excluded):
                continue
            start, end = interval["start_ms"], interval["end_ms"]
            if ((position and intervals[position - 1]["end_ms"] > start)
                    or (position + 1 < len(intervals) and intervals[position + 1]["start_ms"] < end)):
                continue
            first, stop = (start * FPS + 999) // 1000, (end * FPS + 999) // 1000
            clip = video["frame_paths"][first:stop]
            if len(clip) != stop - first or len(clip) < 10 or any(not (ROOT / f).is_file() for f in clip):
                continue
            annotation_id = interval["elan_annotation_id"]
            identity = f"{recording.id}-{annotation_id}"
            rows.append({
                "id": identity, "path": identity + ".mp4", "group": "external_outside_label_diagnostic",
                "gloss": label, "on_roster": False, "signer": f"sartinas:{signer}",
                "source_path": recording.name + ".avi", "source_rgb_sha256": video["source_video_sha256"],
                "annotation_id": annotation_id, "elan_sha256": video["elan_sha256"],
                "start_ms": start, "end_ms": end, "first_frame_zero_based": first,
                "fps": FPS, "frames": len(clip), "frame_paths": clip,
                "frame_sha256": [digest(ROOT / f) for f in clip],
            })
            excluded.add(key)
            count += 1
            if count == DISTINCT_PER_SIGNER:
                break
    if Counter(row["signer"] for row in rows) != {f"sartinas:{s}": DISTINCT_PER_SIGNER for s in RECORDINGS}:
        raise ValueError("Expected six distinct negative annotations from each of four signers")
    return {
        "scope": "Author-timed contextual Sartinas signs with a written label absent from OpenHands' complete 310-label map. "
                 "Continuous parent recordings were used in other project probes. No output used for this cohort's selection. "
                 "Lexical and possible semantic equivalence are unverified, and isolated recognizer behavior on these clips is diagnostic.",
        "selection": "Source signer order n1/n6/n7/n8 then CHRONOLOGICAL within recording; first six nonoverlapping Greek one-word labels "
                     "at least 10 frames. A one-edit exclusion against OpenHands whole labels and underscore components and all previously "
                     "selected labels reduces common spelling variants. No time padding.",
        "source": PAGE, "paper": PAPER, "source_hashes": source_hashes, "clips": rows,
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    selection = plan()
    frozen = OUTPUT / "manifest.planned.json"
    if frozen.exists() and read_json(frozen) != selection:
        raise ValueError("Frozen selection changed")
    if not frozen.exists():
        write_manifest(frozen, selection)
    rows = [encode({**row, "path": (OUTPUT / row["path"]).as_posix()}) for row in selection["clips"]]
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate staged videos")
    rows = [{**row, "path": (OUTPUT / row["path"]).name} for row in rows]
    write_manifest(OUTPUT / "manifest.json", {
        **selection, "clips": rows, "plan_sha256": digest(frozen), "complete": True,
    })
    print(f"Encoded {len(rows)} fixed negatives from author-matched video and gloss timelines", flush=True)


if __name__ == "__main__":
    main()
