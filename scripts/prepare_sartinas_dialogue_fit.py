"""Prepare author-paired, uncut Sartinas train and development recordings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import isfinite
import re
import subprocess
import xml.etree.ElementTree as ET

import imageio_ffmpeg

from prepare_sartinas_appointment_holdout import DOWNLOADS, PAGE, PAPER, ROOT, digest, request, save, source_key


DATA = ROOT / "data/sartinas-dialogue-fit"
HELD = ROOT / "data/sartinas-s6-n7-holdout"
MEDIA = ROOT / ".runtime-tmp/sartinas-dialogue-fit"
FPS = 10
FIT_SELECTION = (("n3_s18_t1", "train"), ("n5_s12_t1", "train"), ("n4_s12_t2", "val"))
HELD_SELECTION = (("n7_s6_t1", "test"),)
FIT_BYTES = 896_728_936
HELD_BYTES = 399_537_288
BLOCK = 400
SOURCES = {
    "n3_s18_t1": {
        "avi_id": "14hx-dYO1H5NiA0pJpQOYViZn7-ZTWbib", "avi_bytes": 338_487_058,
        "eaf_id": "1KFdPl64DLZWpJSV8Hc5NPaV7gH5jYnWB",
        "eaf_sha256": "6fff3fc441649bd50672ae7d28fea434a46d2eb3299b1dc624c02f2dbe823a0f",
    },
    "n5_s12_t1": {
        "avi_id": "1zX73Vqexc_GdukEbxwXjsAMRA4Y5_EBS", "avi_bytes": 294_101_824,
        "eaf_id": "16Z1E79Shj9cIElmrcx1JuRXe_Kn4icg6",
        "eaf_sha256": "ec704d3b7fbb055232fe09237dfd1089d03ff1e1fc2501179168a2c3926880bf",
    },
    "n4_s12_t2": {
        "avi_id": "1NkwL4Alpi14ihp0_tW6m0YLShh1YxIVB", "avi_bytes": 264_140_054,
        "eaf_id": "1g-5WJSwm8mAUn9EPmeu9szzeFCmRGJmD",
        "eaf_sha256": "302e394d8b4ead83eb5e7e3240bb29948795f923dd5bcaf9927b169a0724f94b",
    },
    "n7_s6_t1": {
        "avi_id": "1Gg020w9GBzqBKV5i33WsYfaFPzuEuvO-", "avi_bytes": 399_537_288,
        "eaf_id": "1fuhckmW-8F4ICiTfK4qnjmwX2ex9qBCZ",
        "eaf_sha256": "e7017e69e4d3bdc1c086c29f4756bd4809537b7b078238f0b0f39b9e07722f31",
    },
}


def annotation_tiers(payload, name):
    doc = ET.fromstring(payload)
    media, = doc.findall("./HEADER/MEDIA_DESCRIPTOR")
    if (media.attrib.get("TIME_ORIGIN", "0") != "0"
            or any(media.attrib[key].replace("\\", "/").rsplit("/", 1)[-1] != f"{name}.avi"
                   for key in ("MEDIA_URL", "RELATIVE_MEDIA_URL"))):
        raise ValueError(f"EAF media provenance changed: {name}")
    times = {slot.attrib["TIME_SLOT_ID"]: int(slot.attrib["TIME_VALUE"])
             for slot in doc.findall("./TIME_ORDER/TIME_SLOT") if "TIME_VALUE" in slot.attrib}
    greek, = [tier for tier in doc.findall("./TIER")
              if tier.attrib["TIER_ID"].casefold() in {"protaseis", "προτασεισ"}]
    gloss, = [tier for tier in doc.findall("./TIER") if tier.attrib["TIER_ID"].casefold() == "glosses"]

    def intervals(tier, key):
        result = []
        for unit in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start, end = [times[unit.attrib[f"TIME_SLOT_REF{i}"]] for i in (1, 2)]
            label = (unit.findtext("ANNOTATION_VALUE") or "").strip()
            if not label or not 0 <= start < end:
                raise ValueError(f"Invalid author interval in {name}: {unit.attrib}")
            result.append({key: label, "start_ms": start, "end_ms": end,
                           "elan_annotation_id": unit.attrib["ANNOTATION_ID"]})
        return sorted(result, key=lambda item: (item["start_ms"], item["end_ms"]))

    texts, glosses = intervals(greek, "text"), intervals(gloss, "label")
    if not texts or not glosses or any(a["end_ms"] > b["start_ms"] for a, b in zip(texts, texts[1:])):
        raise ValueError(f"Missing or overlapping Greek reference intervals: {name}")
    return greek.attrib["TIER_ID"], gloss.attrib["TIER_ID"], texts, glosses


def manifest(held=False):
    MEDIA.mkdir(parents=True, exist_ok=True)
    turns, dialogues = [], []
    for name, role in (HELD_SELECTION if held else FIT_SELECTION):
        row = SOURCES[name]
        match = re.fullmatch(r"(n[0-9]+)_(s[0-9]+)_(t[0-9]+)", name)
        actor, script, take = match.groups()
        if actor in {"n1", "n2", "n6"} or script == "s10":
            raise ValueError("A reserved signer or script entered this corpus")
        path = MEDIA / f"{name}.eaf"
        cached = path.is_file()
        if cached:
            payload = path.read_bytes()
        else:
            with request("https://drive.google.com/uc?export=download&id=" + row["eaf_id"]) as response:
                payload = response.read(524_289)
        if len(payload) > 524_288 or sha256(payload).hexdigest() != row["eaf_sha256"]:
            raise ValueError(f"Author EAF bytes changed: {name}")
        greek_tier, gloss_tier, texts, glosses = annotation_tiers(payload, name)
        if not cached:
            path.write_bytes(payload)
        identity, signer = f"sartinas-{actor}-{script}-{take}", f"sartinas-{actor}"
        scenario = f"sartinas-{script}"
        ids = []
        for index, item in enumerate(texts, 1):
            sample_id = f"{identity}__turn{index:03d}"
            ids.append(sample_id)
            contained = [g["label"] for g in glosses
                         if item["start_ms"] <= g["start_ms"] < g["end_ms"] <= item["end_ms"]]
            turns.append({**item, "id": sample_id, "dialogue_id": identity, "signer": signer,
                          "split": role, "glosses": contained})
        dialogues.append({key: row[key] for key in ("avi_id", "eaf_id", "avi_bytes", "eaf_sha256")} | {
            "id": identity, "name": name, "scenario": scenario, "signer": signer, "split": role,
            "turn_ids": ids, "gloss_intervals": glosses, "fps": FPS,
            "greek_tier": greek_tier, "gloss_tier": gloss_tier,
        })
    expected_bytes = HELD_BYTES if held else FIT_BYTES
    if sum(d["avi_bytes"] for d in dialogues) != expected_bytes:
        raise ValueError("Author AVI byte budget changed")
    if len({d["signer"] for d in dialogues}) != len(dialogues):
        raise ValueError("Fit and development signers must remain disjoint")
    if not held and any(d["scenario"] in {"sartinas-s6", "sartinas-s10"} or d["signer"] == "sartinas-n7" for d in dialogues):
        raise ValueError("A held scenario or signer entered fit")
    protocol = ("Held native S6 follow-up dialogue. Scenario and signer are disjoint from fit/development."
                if held else "Fit and development signers are disjoint. S12 is present in both roles; "
                "both held scenarios S6 and S10 are excluded.")
    return {"scope": "Native uncut author GSL recordings. Each recording remains on "
                     "its original continuous time axis. Per-turn glosses include only fully contained spans; "
                     "the full official gloss tier is in each dialogue record.",
            "protocol": protocol, "source_page": PAGE, "paper": PAPER, "fps": FPS, "planned_avi_bytes": expected_bytes,
            "samples": turns, "dialogues": dialogues}


def download(dialogue, media=MEDIA):
    name, size, identity = dialogue["name"], dialogue["avi_bytes"], dialogue["avi_id"]
    path = media / f"{name}.avi"
    receipt = media / f"{name}.source.json"
    if receipt.is_file():
        info = json.loads(receipt.read_text(encoding="utf-8"))
        if (info["file_id"] != identity or info["bytes"] != size or path.stat().st_size != size
                or digest(path) != info["sha256"]):
            raise ValueError(f"Cached author video provenance changed: {name}")
        return path, info["sha256"]
    staging = path.with_suffix(".part")
    full = sha256()
    preferred = None
    step = 32 * 1024 ** 2
    with staging.open("wb") as target:
        for start in range(0, size, step):
            end = min(start + step, size) - 1
            sources = sorted((prefix + identity for prefix in DOWNLOADS), key=lambda url: url != preferred)
            for url in sources:
                try:
                    with request(url, f"bytes={start}-{end}") as incoming:
                        if incoming.status != 206 or incoming.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            continue
                        block = incoming.read(end - start + 2)
                except OSError:
                    continue
                if len(block) != end - start + 1:
                    continue
                if start == 0 and (block[:4], block[8:12]) != (b"RIFF", b"AVI "):
                    raise ValueError(f"Public response was not an AVI: {name}")
                target.write(block)
                full.update(block)
                preferred = url
                break
            else:
                raise ValueError(f"No exact author media range {start}-{end}: {name}")
            print(f"{name} {end + 1:,}/{size:,} original bytes", flush=True)
    checksum = full.hexdigest()
    staging.replace(path)
    save(receipt, {"file_id": identity, "bytes": size, "sha256": checksum})
    return path, checksum


def prepare_video(dialogue, media=MEDIA):
    video, checksum = download(dialogue, media)
    frames_dir = media / dialogue["name"] / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    receipt = frames_dir.parent / "frames.json"
    previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else None
    if previous is None:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-xerror", "-y", "-threads", "2", "-i", str(video), "-map", "0:v:0", "-an",
                   "-vf", rf"fps={FPS},scale=w=min(720\,iw):h=-2", "-threads", "2", "-q:v", "3",
                   str(frames_dir / "frame-%06d.jpg")]
        subprocess.run(command, check=True, timeout=800)
    frames = sorted(frames_dir.glob("frame-*.jpg"))
    if not frames or any(p.name != f"frame-{index:06d}.jpg" for index, p in enumerate(frames, 1)):
        raise ValueError(f"Decoded AVI is missing frames: {dialogue['name']}")
    frame_meta = {"fps": FPS, "video_sha256": checksum, "frames": len(frames)}
    if previous is not None and previous != frame_meta:
        raise ValueError("Frame receipt changed")
    if previous is None:
        save(receipt, frame_meta)
    if max(g["end_ms"] for g in dialogue["gloss_intervals"]) > len(frames) * 1000 / FPS:
        raise ValueError(f"Author gloss exceeds full video: {dialogue['name']}")
    result = {"id": dialogue["id"], "fps": FPS, "frame_count": len(frames),
              "frame_paths": [p.relative_to(ROOT).as_posix() for p in frames],
              "source_video_sha256": checksum, "source_video_bytes": video.stat().st_size,
              "source_video": video.relative_to(ROOT).as_posix()}
    print(f"{dialogue['id']} decoded {len(frames)} frames at {FPS} Hz", flush=True)
    return result


def freeze(data, held):
    data.mkdir(parents=True, exist_ok=True)
    plan_path = data / "manifest.planned.json"
    plan = manifest(held)
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Frozen source annotations changed")
    if not plan_path.exists():
        save(plan_path, plan)
    print(json.dumps({"dialogues": [d["id"] for d in plan["dialogues"]],
                      "reference_turns": len(plan["samples"]), "avi_bytes": plan["planned_avi_bytes"]}), flush=True)
    return plan


def prepare(data, held):
    plan = freeze(data, held)
    with ThreadPoolExecutor(max_workers=3) as workers:
        videos = list(workers.map(prepare_video, plan["dialogues"]))
    by_id = {row["id"]: row for row in videos}
    last = {row["dialogue_id"]: row["end_ms"] for row in plan["samples"]}
    if any(last[row["id"]] > row["frame_count"] * 1000 / FPS for row in videos):
        raise ValueError("Author Greek reference extends beyond the decoded video")
    final = {**plan, "complete": True, "dialogues": [
        {**d, **{field: by_id[d["id"]][field] for field in ("source_video", "source_video_sha256", "frame_count")}}
        for d in plan["dialogues"]]}
    source = data / "manifest.json"
    save(source, final)
    save(data / "video-manifest.json", {"scope": final["scope"], "source_manifest_sha256": digest(source),
                                        "samples": videos})


def blocks(data=DATA):
    source = data / "video-manifest.json"
    videos = json.loads(source.read_text(encoding="utf-8"))["samples"]
    chunks = []
    for video in videos:
        paths = video["frame_paths"]
        chunks.extend({"id": f"{video['id']}-block{start // BLOCK:03d}", "fps": FPS,
                       "stream_id": video["id"], "start_frame": start, "frame_paths": paths[start:start + BLOCK]}
                      for start in range(0, len(paths), BLOCK))
    return {"scope": "Original uncut time axis, extracted in fixed blocks unrelated to annotation boundaries. "
                     "The MediaPipe detector restarts at each block.", "video_manifest_sha256": digest(source),
            "samples": chunks}


def merge(data):
    block_manifest = data / "extraction-blocks.json"
    plan = blocks(data)
    if json.loads(block_manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("The original-time block plan changed")
    raw = data / "features-blocks.jsonl"
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["id"]: record for record in records}
    if len(records) != len(by_id) or set(by_id) != {row["id"] for row in plan["samples"]}:
        raise ValueError("Missing or duplicate extraction blocks")
    extractor, = {row["extractor_sha256"] for row in records}
    schema, = {row["feature_schema"] for row in records}
    if schema != "signbridge-landmarks-xy-v1":
        raise ValueError("Unexpected pose schema")
    videos = json.loads((data / "video-manifest.json").read_text(encoding="utf-8"))["samples"]
    output = data / "features.jsonl"
    pending = output.with_suffix(".tmp")
    audit = []
    with pending.open("w", encoding="utf-8") as target:
        for video in videos:
            features, starts, quality, elapsed = [], [], 0., 0.
            for block in plan["samples"]:
                if block["stream_id"] != video["id"]:
                    continue
                row = by_id[block["id"]]
                frames = row["features"]
                if (row["source_key"] != source_key(block) or row["stride"] != 1 or row["fps"] != FPS
                        or block["start_frame"] != len(features) or len(frames) != len(block["frame_paths"])
                        or any(len(frame) != 224 or not all(map(isfinite, frame)) for frame in frames)):
                    raise ValueError(f"Invalid extraction block: {row['id']}")
                starts.append(len(features))
                features.extend(frames)
                quality += row["quality"] * len(frames)
                elapsed += row["elapsed_ms"]
            if len(features) != video["frame_count"]:
                raise ValueError("Feature stream does not cover all sampled frames")
            item = {"id": video["id"], "fps": FPS, "stride": 1, "feature_schema": schema,
                    "extractor_sha256": extractor, "source_key": source_key(video),
                    "source_frames": video["frame_count"], "quality": quality / len(features), "elapsed_ms": elapsed}
            target.write(json.dumps({**item, "features": features}, separators=(",", ":")) + "\n")
            audit.append({**item, "detector_restart_sampled_frames": starts})
    pending.replace(output)
    save(data / "feature-audit.json", {"scope": plan["scope"], "video_manifest_sha256": plan["video_manifest_sha256"],
         "features_sha256": digest(output), "blocks_sha256": digest(raw), "streams": audit})
    print(json.dumps({"verified_streams": len(audit), "fps": FPS, "frames": sum(row["source_frames"] for row in audit)}),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "prepare", "blocks", "merge"), nargs="?", default="prepare")
    parser.add_argument("--held", action="store_true")
    args = parser.parse_args()
    data = HELD if args.held else DATA
    if args.phase == "plan":
        freeze(data, args.held)
    elif args.phase == "prepare":
        prepare(data, args.held)
    elif args.phase == "blocks":
        save(data / "extraction-blocks.json", blocks(data))
    else:
        merge(data)
