"""Add two author-paired n2 recordings without replacing the original experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import isfinite
from prepare_sartinas_dialogue_fit import (
    BLOCK, DATA as BASE, FPS, ROOT, annotation_tiers, digest, prepare_video, request, save, source_key,
)


DATA = ROOT / "data/sartinas-dialogue-fit-expanded"
MEDIA = ROOT / ".runtime-tmp/sartinas-dialogue-fit-expanded"
SOURCES = (
    {"name": "n2_s12_t1", "scenario": "sartinas-s12", "id": "sartinas-n2-s12-t1",
     "avi_id": "1k173ilov0fdR4IbcUIjK_Wd5R2fKKHxn", "avi_bytes": 238_822_896,
     "eaf_id": "15FMhnif20XbFKmEBbDTc7ScRzrhSd2Sq", "eaf_bytes": 95_957,
     "eaf_sha256": "6ddea3ab0ebfcd40518dab7b318bcb8bbedc68cc9348426d0739c69453009e38"},
    {"name": "n2_s18_t2", "scenario": "sartinas-s18", "id": "sartinas-n2-s18-t2",
     "avi_id": "1Iq6mYIHuatR0gJF8kIUgKDLEZoKIUOFI", "avi_bytes": 340_002_286,
     "eaf_id": "1487g_Rl2kgL_TDORqjc5LurGtyuCpJO_", "eaf_bytes": 136_306,
     "eaf_sha256": "474eaca5aa73f047fce5535ab5fe0ab7246f51b35a900bc648ae25db48016a66"},
)
BASE_MANIFEST = "b9ef16576e66bd3f2b3b9997113dcdea83a90f3c9bfde8daefd46c9019edff57"
BASE_FEATURES = "33c81e95d6a76f9a49a3c21a48e5b4c60fc4755d8634deec4e04d21fcd1bcd46"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def base_inputs():
    manifest_path, feature_path = BASE / "manifest.json", BASE / "features.jsonl"
    if digest(manifest_path) != BASE_MANIFEST or digest(feature_path) != BASE_FEATURES:
        raise ValueError("Original frozen native fit inputs changed")
    manifest, video = load(manifest_path), load(BASE / "video-manifest.json")
    if not manifest["complete"] or video["source_manifest_sha256"] != BASE_MANIFEST:
        raise ValueError("Original video source binding differs")
    return manifest, video["samples"]


def annotations(source):
    name = source["name"]
    path = MEDIA / f"{name}.eaf"
    cached = path.is_file()
    if cached:
        payload = path.read_bytes()
    else:
        with request("https://drive.google.com/uc?export=download&id=" + source["eaf_id"]) as incoming:
            payload = incoming.read(source["eaf_bytes"] + 1)
    if len(payload) != source["eaf_bytes"] or sha256(payload).hexdigest() != source["eaf_sha256"]:
        raise ValueError(f"Author EAF bytes changed: {name}")
    greek_tier, gloss_tier, greek_rows, gloss_rows = annotation_tiers(payload, name)
    if not cached:
        path.write_bytes(payload)
    dialogue = {**source, "signer": "sartinas-n2", "split": "train", "fps": FPS,
                "greek_tier": greek_tier, "gloss_tier": gloss_tier, "gloss_intervals": gloss_rows}
    turns = [{**row, "id": f"{source['id']}__turn{index:03d}", "dialogue_id": source["id"],
              "signer": dialogue["signer"], "split": "train", "glosses": [
                  item["label"] for item in gloss_rows
                  if row["start_ms"] <= item["start_ms"] < item["end_ms"] <= row["end_ms"]]}
             for index, row in enumerate(greek_rows, 1)]
    return {**dialogue, "turn_ids": [row["id"] for row in turns]}, turns


def freeze():
    base, _ = base_inputs()
    DATA.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    if sum(r["avi_bytes"] + r["eaf_bytes"] for r in SOURCES) > 700_000_000:
        raise ValueError("Expanded source budget exceeded")
    extras, extra_turns = [], []
    for source in SOURCES:
        dialogue, turns = annotations(source)
        extras.append(dialogue)
        extra_turns.extend(turns)
    plan = {**base, "complete": False, "protocol": "n2 S12/S18 added as train. n4 development remains independent "
             "by signer and shares S12 with training. S6, S10 and held signers n1/n6/n7 remain excluded.",
            "baseline_manifest_sha256": BASE_MANIFEST, "baseline_features_sha256": BASE_FEATURES,
            "planned_avi_bytes": base["planned_avi_bytes"] + sum(row["avi_bytes"] for row in extras),
            "dialogues": [*base["dialogues"], *extras], "samples": [*base["samples"], *extra_turns]}
    path = DATA / "manifest.planned.json"
    if path.is_file() and load(path) != plan:
        raise ValueError("Expanded EAF plan changed after freezing")
    if not path.is_file():
        save(path, plan)
    print(json.dumps({"added_train_turns": len(extra_turns), "total_turns": len(plan["samples"])}), flush=True)
    return plan, extras


def prepare():
    plan, extras = freeze()
    _, original_videos = base_inputs()
    with ThreadPoolExecutor(max_workers=2) as workers:
        new = list(workers.map(lambda row: prepare_video(row, media=MEDIA), extras))
    by_id = {row["id"]: row for row in new}
    for row in plan["samples"]:
        video = by_id.get(row["dialogue_id"])
        if video and row["end_ms"] > video["frame_count"] * 1000 / FPS:
            raise ValueError(f"Greek time extends beyond the AVI: {row['id']}")
    dialogues = [{**row, **{k: by_id[row["id"]][k] for k in ("source_video", "source_video_sha256", "frame_count")}}
                 if row["id"] in by_id else row for row in plan["dialogues"]]
    source = DATA / "manifest.json"
    save(source, {**plan, "complete": True, "dialogues": dialogues})
    save(DATA / "video-manifest.json", {"scope": plan["scope"], "source_manifest_sha256": digest(source),
                                        "samples": [*original_videos, *new]})


def blocks():
    source = DATA / "video-manifest.json"
    videos = load(source)["samples"]
    new_ids = {row["id"] for row in SOURCES}
    added = []
    for video in videos:
        if video["id"] not in new_ids:
            continue
        paths = video["frame_paths"]
        added.append([{"id": f"{video['id']}-block{start // BLOCK:03d}", "fps": FPS, "stream_id": video["id"],
                       "start_frame": start, "frame_paths": paths[start:start + BLOCK]}
                      for start in range(0, len(paths), BLOCK)])
    queue = [items[i] for i in range(max(map(len, added))) for items in added if i < len(items)]
    return {"scope": "Fixed original-time blocks for new source extraction. Detector restarts are independent of annotation.",
            "video_manifest_sha256": digest(source), "samples": queue}


def merge():
    _, original_videos = base_inputs()
    plan = blocks()
    if load(DATA / "extraction-blocks.json") != plan:
        raise ValueError("Expanded extraction block plan changed")
    raw = DATA / "features-blocks.jsonl"
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {row["id"]: row for row in records}
    if len(records) != len(seen) or set(seen) != {block["id"] for block in plan["samples"]}:
        raise ValueError("Missing or duplicate new extraction blocks")
    prior = (BASE / "features.jsonl").read_text(encoding="utf-8").splitlines()
    old = {row["id"]: row for row in map(json.loads, prior)}
    for video in original_videos:
        if old[video["id"]]["source_key"] != source_key(video):
            raise ValueError("Baseline feature source binding changed")
    extractor, = {row["extractor_sha256"] for row in [*old.values(), *records]}
    schema, = {row["feature_schema"] for row in [*old.values(), *records]}
    if schema != "signbridge-landmarks-xy-v1":
        raise ValueError("Unexpected expanded feature schema")
    rows, audit = [], []
    for video in load(DATA / "video-manifest.json")["samples"]:
        if video["id"] in old:
            continue
        features, resets, tracking, elapsed = [], [], 0., 0.
        for block in plan["samples"]:
            if block["stream_id"] != video["id"]:
                continue
            r = seen[block["id"]]
            frames = r["features"]
            if (r["source_key"] != source_key(block) or r["stride"] != 1 or r["fps"] != FPS
                    or block["start_frame"] != len(features) or len(frames) != len(block["frame_paths"])
                    or any(len(frame) != 224 or not all(map(isfinite, frame)) for frame in frames)):
                raise ValueError(f"Bad source binding for {r['id']}")
            resets.append(len(features))
            features.extend(frames)
            tracking += r["quality"] * len(frames)
            elapsed += r["elapsed_ms"]
        if len(features) != video["frame_count"]:
            raise ValueError("Expanded features do not cover the entire time axis")
        item = {"id": video["id"], "fps": FPS, "stride": 1, "source_frames": len(features),
                "feature_schema": schema, "extractor_sha256": extractor, "source_key": source_key(video),
                "quality": tracking / len(features), "elapsed_ms": elapsed}
        rows.append(json.dumps({**item, "features": features}, separators=(",", ":")))
        audit.append({**item, "detector_restart_sampled_frames": resets})
    final = DATA / "features.jsonl"
    pending = final.with_suffix(".tmp")
    with pending.open("wb") as target:
        with (BASE / "features.jsonl").open("rb") as baseline:
            while chunk := baseline.read(1024 ** 2):
                target.write(chunk)
        target.write(("\n".join(rows) + "\n").encode("utf-8"))
    pending.replace(final)
    save(DATA / "feature-audit.json", {"scope": plan["scope"], "baseline_features_sha256": BASE_FEATURES,
         "video_manifest_sha256": plan["video_manifest_sha256"], "features_sha256": digest(final),
         "blocks_sha256": digest(raw),
         "added_streams": audit})
    print(json.dumps({"total_streams": len(prior) + len(rows), "new_frames": sum(r["source_frames"] for r in audit)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "prepare", "blocks", "merge"), nargs="?", default="prepare")
    phase = parser.parse_args().phase
    if phase == "plan":
        freeze()
    elif phase == "prepare":
        prepare()
    elif phase == "blocks":
        save(DATA / "extraction-blocks.json", blocks())
    else:
        merge()
