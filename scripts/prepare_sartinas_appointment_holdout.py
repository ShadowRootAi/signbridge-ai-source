"""Verify and stage an uncut, external, held-out GSL consultation."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from math import isfinite
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
MAX_MEDIA_BYTES = 500_000_000
FRAME_RATE = 30
BLOCK_FRAMES, STRIDE = 600, 2
PAGE = "https://vagsart.github.io/sl_keyframe_dataset/"
PAPER = "https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1360.pdf"
DOWNLOADS = (
    "https://drive.usercontent.google.com/download?export=download&confirm=t&id=",
    "https://drive.google.com/uc?export=download&confirm=t&id=",
)


@dataclass(frozen=True)
class Recording:
    signer: str
    directory: str
    avi_id: str
    eaf_id: str
    avi_bytes: int
    avi_sha256: str | None
    eaf_sha256: str
    turns: int
    max_media_bytes: int = MAX_MEDIA_BYTES
    preserve_gloss_intervals: bool = False

    @property
    def name(self):
        return f"{self.signer}_s10_t1"

    @property
    def id(self):
        return f"sartinas-{self.signer}-s10-t1"

    @property
    def media(self):
        return ROOT / ".runtime-tmp" / self.directory

    @property
    def data(self):
        return ROOT / "data" / self.directory


RECORDINGS = {
    "n1": Recording("n1", "sartinas-s10-holdout", "1xwTsznpW4WXW7dypYH23eOkalUi2trbD",
                    "1FdZAD2bvSQ_9SA2VMBs9iSt_mlwS1jLf", 290_726_172,
                    "e2ab2602fc050588194c83a43115a9af39c36bb118e8958a3a1ebd3be22201e5",
                    "fabcde4a4acf1716727c43885953cbbfe6195f37843ca6c736a9d6c499e180ac", 40),
    "n6": Recording("n6", "sartinas-s10-n6-holdout", "1T8XUev2ilVZAKgjycCwuVMXvA05yDw1L",
                    "1qDBe0R-zGj_w4H-DYilBsDEhQLD8KzYc", 160_351_566,
                    "af0f27ec20ab7476622458cdec0911e5edafefbca54145cc99de6266683a1d34",
                    "78523181c244d3e947d0590e0bab55c087cdad18c940b382b5f7971d3b644f65", 38),
    "n7": Recording("n7", "sartinas-s10-n7-holdout", "1oupsv_Au5foDASGsh-FrZehXOuSeyJVB",
                    "1T-WrX7KOKE16D7yE_U4MtIKlSTC9RZNW", 174_908_694,
                    "7c6e6b2464dd634963581f11116ca6fa57f3dfd71d52c8b41a576f24c564bb9f",
                    "52a700eac38cf0695b3d14feb263557dfa3c620e46197606180f7ba5e607cda2", 38),
    "n8": Recording("n8", "sartinas-s10-n8-holdout", "1dtbwHpAynYRB0OLPVko5zg2dyYCdmjHF",
                    "1Ek53CqSwFaLO_HOcX7kQsmHz4ajlz8yP", 160_632_224, None,
                    "ef86abd52cefd3f7be3934023447b5058637bd620a07261a13444e82a0709983", 38,
                    max_media_bytes=220_000_000, preserve_gloss_intervals=True),
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def request(url, byte_range=None):
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = byte_range
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def eaf_manifest(recording):
    path = recording.media / f"{recording.name}.eaf"
    if path.exists():
        payload = path.read_bytes()
    else:
        with request("https://drive.google.com/uc?export=download&id=" + recording.eaf_id) as source:
            payload = source.read(524_289)
    if len(payload) > 524_288 or sha256(payload).hexdigest() != recording.eaf_sha256:
        raise ValueError("ELAN bytes differ from the verified holdout annotation")
    doc = ET.fromstring(payload)
    descriptors = doc.findall("./HEADER/MEDIA_DESCRIPTOR")
    if (len(descriptors) != 1 or any(descriptors[0].attrib.get(field, "").replace("\\", "/").rsplit("/", 1)[-1] != f"{recording.name}.avi"
                                     for field in ("RELATIVE_MEDIA_URL", "MEDIA_URL"))):
        raise ValueError("ELAN media descriptor does not match the exact public AVI filename")
    times = {row.attrib["TIME_SLOT_ID"]: int(row.attrib["TIME_VALUE"]) for row in doc.findall("./TIME_ORDER/TIME_SLOT")
             if "TIME_VALUE" in row.attrib}
    [greek] = [row for row in doc.findall("./TIER") if row.attrib["TIER_ID"].casefold() == "protaseis"]
    [gloss] = [row for row in doc.findall("./TIER") if row.attrib["TIER_ID"].casefold() == "glosses"]
    samples = []
    signer = f"sartinas-{recording.signer}"
    for index, row in enumerate(greek.findall(".//ALIGNABLE_ANNOTATION")):
        start, end = [times[row.attrib[f"TIME_SLOT_REF{slot}"]] for slot in (1, 2)]
        text = (row.findtext("ANNOTATION_VALUE") or "").strip()
        if not text or not 0 <= start < end or (samples and start < samples[-1]["end_ms"]):
            raise ValueError("Greek tier contains an empty or overlapping interval")
        samples.append({"id": f"{recording.id}__turn{index+1:03d}", "signer": signer, "split": "test", "text": text,
                        "start_ms": start, "end_ms": end, "elan_annotation_id": row.attrib["ANNOTATION_ID"]})
    if len(samples) != recording.turns or "ραντεβού" not in samples[-1]["text"].casefold():
        raise ValueError(f"Expected all {recording.turns} Greek references, ending with the follow-up appointment")
    dialogue = {"id": recording.id, "scenario": "sartinas-s10-consultation", "signer": signer, "split": "test",
                "turn_ids": [row["id"] for row in samples]}
    if recording.preserve_gloss_intervals:
        from prepare_sartinas_dialogue_fit import annotation_tiers
        _, _, _, intervals = annotation_tiers(payload, recording.name)
        dialogue["gloss_intervals"] = intervals
        for sample in samples:
            sample["dialogue_id"] = recording.id
            sample["glosses"] = [item["label"] for item in intervals
                                 if sample["start_ms"] <= item["start_ms"] < item["end_ms"] <= sample["end_ms"]]
    if not path.exists():
        path.write_bytes(payload)
    return {"scope": "Uncut Sartinas GSL doctor consultation with one signer and a follow-up appointment statement. Strict holdout only.",
            "source_page": PAGE, "paper": PAPER, "video_file_id": recording.avi_id, "elan_file_id": recording.eaf_id,
            "elan_sha256": recording.eaf_sha256, "greek_tier": greek.attrib["TIER_ID"], "gloss_tier": gloss.attrib["TIER_ID"],
            "gloss_annotations": len(gloss.findall(".//ANNOTATION_VALUE")), "samples": samples,
            "dialogues": [dialogue]}


def download_video(recording, path):
    if recording.avi_bytes > recording.max_media_bytes:
        raise ValueError("Source AVI exceeds the authorized research transfer cap")
    receipt = recording.data / "source-video-receipt.json"
    expected_hash = recording.avi_sha256
    if expected_hash is None and receipt.exists():
        stored = json.loads(receipt.read_text(encoding="utf-8"))
        if stored["file_id"] != recording.avi_id or stored["bytes"] != recording.avi_bytes:
            raise ValueError("The public AVI receipt refers to a different source")
        expected_hash = stored["sha256"]
    if path.exists():
        if path.stat().st_size != recording.avi_bytes or digest(path) != expected_hash:
            raise ValueError("Existing AVI bytes differ from the verified public source")
        return expected_hash
    step = 16 * 1024**2
    pending = path.with_suffix(".part")
    previous_url = None
    with pending.open("wb") as target:
        for start in range(0, recording.avi_bytes, step):
            end = min(start + step, recording.avi_bytes) - 1
            expected = f"bytes {start}-{end}/{recording.avi_bytes}"
            urls = [previous_url] if previous_url else [prefix + recording.avi_id for prefix in DOWNLOADS]
            for url in urls:
                with request(url, f"bytes={start}-{end}") as source:
                    if source.status != 206 or source.headers.get("Content-Range") != expected:
                        continue
                    block = source.read(end - start + 2)
                    if len(block) != end - start + 1:
                        raise ValueError("Partial read of a verified AVI byte range")
                    if start == 0 and (block[:4], block[8:12]) != (b"RIFF", b"AVI "):
                        raise ValueError("The Drive response is not an AVI container")
                    target.write(block)
                    previous_url = url
                    break
            else:
                raise ValueError(f"The public Drive server did not return {expected}")
            print(f"Verified {end+1:,}/{recording.avi_bytes:,} AVI bytes", flush=True)
    observed_hash = digest(pending)
    if expected_hash is not None and observed_hash != expected_hash:
        raise ValueError("Downloaded AVI bytes differ from the verified public source")
    if recording.avi_sha256 is None and not receipt.exists():
        save(receipt, {"file_id": recording.avi_id, "bytes": recording.avi_bytes, "sha256": observed_hash,
                       "provenance": "Hash computed from exact public byte ranges on first acquisition"})
    pending.replace(path)
    return observed_hash


def make_frames(recording, video):
    destination = recording.media / "frames"
    destination.mkdir(exist_ok=True)
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
            "-threads", "2", "-i", str(video), "-map", "0:v:0", "-an", "-vf", r"fps=30,scale=w=min(720\,iw):h=-2",
            "-threads", "2", "-q:v", "3", str(destination / "frame-%06d.jpg")]
    subprocess.run(args, check=True, timeout=400)
    frames = sorted(destination.glob("frame-*.jpg"))
    if not frames or any(path.name != f"frame-{i:06d}.jpg" for i, path in enumerate(frames, 1)):
        raise ValueError("Full AVI frame decode did not produce a contiguous time axis")
    return frames


def prepare(recording):
    recording.data.mkdir(parents=True, exist_ok=True)
    recording.media.mkdir(parents=True, exist_ok=True)
    source = eaf_manifest(recording)
    source_path = recording.data / "manifest.json"
    save(source_path, source)
    video = recording.media / f"{recording.name}.avi"
    video_hash = download_video(recording, video)
    frames = make_frames(recording, video)
    turns = source["samples"]
    if turns[-1]["end_ms"] * FRAME_RATE > 1000 * len(frames):
        raise ValueError("A reference interval extends beyond decoded video duration")
    if any(item["end_ms"] * FRAME_RATE > 1000 * len(frames)
           for item in source["dialogues"][0].get("gloss_intervals", ())):
        raise ValueError("A gloss interval extends beyond decoded video duration")
    cuts = [0, *[round((left["end_ms"] + right["start_ms"]) * FRAME_RATE / 2000)
                  for left, right in zip(turns[:-1], turns[1:], strict=True)], len(frames)]
    if any(left >= right for left, right in zip(cuts[:-1], cuts[1:], strict=True)):
        raise ValueError("Reference midpoint cuts are not strictly increasing")
    save(recording.data / "video-manifest.json", {"scope": source["scope"], "source_manifest_sha256": digest(source_path),
         "samples": [{"id": recording.id, "fps": FRAME_RATE, "frame_paths": [p.relative_to(ROOT).as_posix() for p in frames],
                      "frame_count": len(frames), "reference_offsets_30fps": cuts, "source_video_sha256": video_hash,
                      "source_video_bytes": video.stat().st_size, "elan_sha256": source["elan_sha256"]}]})
    print(json.dumps({"frames": len(frames), "duration_s": len(frames) / FRAME_RATE, "reference_turns": len(turns)}), flush=True)


def block_manifest(recording):
    video_path = recording.data / "video-manifest.json"
    [video] = json.loads(video_path.read_text(encoding="utf-8"))["samples"]
    paths = video["frame_paths"]
    chunks = [{"id": f"{recording.id}-block{start // BLOCK_FRAMES:03d}", "fps": FRAME_RATE,
               "frame_paths": paths[start:start + BLOCK_FRAMES]}
              for start in range(0, len(paths), BLOCK_FRAMES)]
    return {"scope": "Fixed blocks for parallel image-mode extraction. Each block starts a new detector. "
                     "No annotations affect block edges. Parity with a single detector has not been established.",
            "video_manifest_sha256": digest(video_path), "samples": chunks}


def source_key(row):
    signature = json.dumps([row["fps"], row["frame_paths"]], ensure_ascii=False, separators=(",", ":"))
    return sha256(signature.encode("utf-8")).hexdigest()


def merge_features(recording):
    plan = block_manifest(recording)
    saved = json.loads((recording.data / "extraction-blocks.json").read_text(encoding="utf-8"))
    if saved != plan:
        raise ValueError("The frame block manifest changed since extraction began")
    [video] = json.loads((recording.data / "video-manifest.json").read_text(encoding="utf-8"))["samples"]
    block_path = recording.data / "features-blocks.jsonl"
    records = [json.loads(line) for line in block_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in records}
    if len(records) != len(by_id) or set(by_id) != {row["id"] for row in plan["samples"]}:
        raise ValueError("Incomplete or duplicate browser feature blocks")
    extractor_hashes = {row["extractor_sha256"] for row in records}
    schemas = {row["feature_schema"] for row in records}
    if len(extractor_hashes) != 1 or schemas != {"signbridge-landmarks-xy-v1"}:
        raise ValueError("Feature blocks use inconsistent detector assets or schemas")
    features, quality, elapsed = [], 0., 0.
    for block in plan["samples"]:
        row = by_id[block["id"]]
        frames = row["features"]
        if (row["source_key"] != source_key(block) or row["fps"] != FRAME_RATE / STRIDE or row["stride"] != STRIDE
                or len(frames) != len(block["frame_paths"][::STRIDE])
                or any(len(frame) != 224 or not all(map(isfinite, frame)) for frame in frames)):
            raise ValueError(f"Invalid browser block provenance or features: {row['id']}")
        features.extend(frames)
        quality += row["quality"] * len(frames)
        elapsed += row["elapsed_ms"]
    if len(features) != len(video["frame_paths"][::STRIDE]):
        raise ValueError("Joined features do not cover the full uninterrupted recording")
    output = recording.data / "features.jsonl"
    pending = output.with_suffix(".tmp")
    pending.write_text(json.dumps({"id": recording.id, "features": features, "fps": FRAME_RATE / STRIDE,
        "feature_schema": next(iter(schemas)), "extractor_sha256": next(iter(extractor_hashes)),
        "source_key": source_key(video), "stride": STRIDE, "source_frames": video["frame_count"],
        "quality": quality / len(features), "elapsed_ms": elapsed}) + "\n", encoding="utf-8")
    pending.replace(output)
    save(recording.data / "feature-audit.json", {"scope": plan["scope"], "video_manifest_sha256": plan["video_manifest_sha256"],
         "blocks_sha256": digest(block_path), "features_sha256": digest(output), "blocks": len(records),
         "fixed_original_frame_block_size": BLOCK_FRAMES,
         "block_start_frames_30fps": list(range(0, video["frame_count"], BLOCK_FRAMES)),
         "frames": len(features), "fps": FRAME_RATE / STRIDE, "quality": quality / len(features)})
    print(f"Verified {len(features)} full-stream frames from {len(records)} blocks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "blocks", "merge"), nargs="?", default="prepare")
    parser.add_argument("--signer", choices=RECORDINGS, default="n1")
    args = parser.parse_args()
    recording = RECORDINGS[args.signer]
    if args.phase == "prepare":
        prepare(recording)
    elif args.phase == "blocks":
        save(recording.data / "extraction-blocks.json", block_manifest(recording))
    else:
        merge_features(recording)


if __name__ == "__main__":
    main()
