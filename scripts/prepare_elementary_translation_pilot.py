"""Build a bounded, signer-disjoint Elementary23 video-to-Greek pilot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5, sha256
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import zipfile
import zlib

import imageio_ffmpeg

from build_elementary_library import VIDEO_DIR, read_original
from dataset_fetch import RemoteZipFile
from prepare_elementary_catalog_plan import ARCHIVE, ROOT, SOURCE_DIR, workbook_rows


OUTPUT = ROOT / "data/elementary-translation-pilot"
GREEK = re.compile(r"[α-ωάέήίόύώϊϋΐΰ]+")
SPLITS = {"A": ("train", 12), "B": ("train", 12), "C": ("train", 12), "E": ("train", 12),
          "F": ("val", 8), "G": ("test", 12)}
RECORD = "https://zenodo.org/records/7847052"
FRAME_RATE = 15


def words(text):
    return GREEK.findall(unicodedata.normalize("NFC", text.lower()))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prior_texts():
    """Keep exact CERTH references out of the second source across all roles."""
    archive = ROOT / "data/original_annotations/supplementary.zip"
    with zipfile.ZipFile(archive) as source:
        name = next(name for name in source.namelist() if name.endswith("merged_continuous.csv"))
        lines = source.read(name).decode("utf-8-sig").splitlines()
    return {" ".join(words(parts[1])) for line in lines if len(parts := line.split("|")) >= 3}


def plan_rows(workbook):
    blocked = prior_texts()
    rows = list(workbook_rows(workbook))
    counts = Counter(identity for _, identity, _, _ in rows)
    usable = []
    for row, identity, signer, text in rows:
        if not identity.startswith("glossa_") or not re.fullmatch(r"[a-z0-9_]+", identity) or signer not in SPLITS or counts[identity] != 1:
            continue
        clean = unicodedata.normalize("NFC", text.strip().lower())
        if not re.fullmatch(r"[α-ωάέήίόύώϊϋΐΰ .!?;·,«»\-]+", clean):
            continue
        tokens = words(clean)
        target = " ".join(tokens)
        if 2 <= len(tokens) <= 5 and target not in blocked:
            usable.append({"id": "elementary-" + identity, "source_path": f"{VIDEO_DIR}/{identity}.flv",
                           "signer": signer, "split": SPLITS[signer][0], "text": target,
                           "original_text": text, "workbook_row": row, "tokens": tokens})
    common = Counter(token for row in usable if row["split"] == "train" for token in set(row["tokens"]))
    used_texts = set()
    selected = []
    seen_train = Counter()
    for signer, (role, limit) in SPLITS.items():
        pool = [row for row in usable if row["signer"] == signer]
        def next_score(row):
            tokens = set(row["tokens"])
            if role == "train":
                # Prefer words that can recur and increase their contextual coverage.
                value = sum(min(common[word], 30) / (1 + seen_train[word]) for word in tokens) / len(tokens)
            else:
                covered = sum(seen_train[word] > 0 for word in tokens) / len(tokens)
                value = covered if chosen_count < limit // 2 else .75 - abs(covered - .60)
            return -value, sha256(row["id"].encode()).hexdigest()
        for chosen_count in range(limit):
            row = min((row for row in pool if row["text"] not in used_texts), key=next_score)
            used_texts.add(row["text"])
            selected.append(row)
            if role == "train":
                seen_train.update(set(row["tokens"]))
    return selected


def materialize(row, archive, info, end, previous, *, output=OUTPUT):
    directory = output / "rgb" / row["id"]
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "source.flv"
    original = video.read_bytes() if video.exists() else b""
    if len(original) != info.file_size or zlib.crc32(original) != info.CRC:
        original, _ = read_original(archive, info, end)
        video.write_bytes(original)
    source_hash = sha256(original).hexdigest()
    crc = f"{info.CRC:08x}"
    if previous and previous["source_sha256"] == source_hash and previous["archive_crc32"] == crc:
        cached = sorted(directory.glob("frame-*.jpg"))
        if ([path.relative_to(ROOT).as_posix() for path in cached] == previous["frame_paths"]
                and [sha256(path.read_bytes()).hexdigest() for path in cached] == previous["frame_sha256"]):
            return previous
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
               "-threads", "1", "-i", str(video), "-map", "0:v:0", "-an",
               "-vf", "fps=15", "-threads", "1", "-q:v", "3", str(directory / "frame-%05d.jpg")]
    subprocess.run(command, check=True, capture_output=True, timeout=45)
    frames = sorted(directory.glob("frame-*.jpg"))
    if not 6 <= len(frames) <= 450:
        raise ValueError(f"{row['id']} has {len(frames)} sampled frames")
    hashes = []
    for path in frames:
        payload = path.read_bytes()
        if payload[:2] != b"\xff\xd8" or payload[-2:] != b"\xff\xd9":
            raise ValueError(f"Unexpected JPEG bytes for {path.name}")
        hashes.append(sha256(payload).hexdigest())
    return {**row, "fps": FRAME_RATE, "frame_paths": [path.relative_to(ROOT).as_posix() for path in frames],
            "frame_sha256": hashes, "source_sha256": source_hash, "archive_crc32": crc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    workbook = SOURCE_DIR / "All_Elementary.xlsx"
    metadata = json.loads((SOURCE_DIR / "elementary23-zenodo-metadata.json").read_text(encoding="utf-8"))
    files = {item["key"]: item for item in metadata["files"]}
    workbook_bytes = workbook.read_bytes()
    if metadata["id"] != 7847052 or metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected official dataset or license")
    if files[workbook.name]["checksum"] != "md5:" + md5(workbook_bytes).hexdigest():
        raise ValueError("Workbook does not match the official checksum")
    samples = plan_rows(workbook)
    plan = {"source": RECORD, "workbook_sha256": sha256(workbook_bytes).hexdigest(), "license": "CC-BY-4.0",
            "sample_fps": FRAME_RATE, "signer_roles": {signer: role for signer, (role, _) in SPLITS.items()},
            "selection": "2–5-word Greek texts. Exact normalized texts and video IDs are unique, and CERTH text is excluded.",
            "label_type": "Original Greek translation only. No gloss or segment timing annotations.",
            "planned_samples": len(samples), "samples": samples}
    frozen = OUTPUT / "manifest.planned.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != plan:
        raise ValueError("Pilot split differs from the frozen manifest")
    save(frozen, plan)
    print(json.dumps(Counter(row["split"] for row in samples)), flush=True)
    if args.plan_only:
        return

    manifest_path = OUTPUT / "manifest.json"
    previous = {}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("workbook_sha256") == plan["workbook_sha256"] and saved.get("sample_fps") == FRAME_RATE:
            previous = {row["id"]: row for row in saved["samples"]}
    spec = files[ARCHIVE]
    remote = {"archiveUrl": spec["links"]["self"], "archiveBytes": spec["size"]}
    with RemoteZipFile(remote["archiveUrl"], remote["archiveBytes"]) as stream, zipfile.ZipFile(stream) as archive:
        index = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
        bounds = {entry.filename: (entry, next_offset) for entry, next_offset in
                  zip(index, [entry.header_offset for entry in index[1:]] + [archive.start_dir], strict=True)}
        initial = stream.bytes_received
    size = initial + sum(bounds[row["source_path"]][1] - bounds[row["source_path"]][0].header_offset for row in samples)
    if size > 85_000_000:
        raise ValueError(f"Planned ranges exceed 85 MB: {size}")
    print(json.dumps({"maximum_payload_bytes": size, "planned": len(samples)}), flush=True)
    done, failed = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {}
        for row in samples:
            cached = previous.get(row["id"])
            if cached and any(cached.get(key) != value for key, value in row.items()):
                cached = None
            pending[pool.submit(materialize, row, remote, *bounds[row["source_path"]], cached)] = row
        for task in as_completed(pending):
            row = pending[task]
            try:
                done[row["id"]] = task.result()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                failed.append({"id": row["id"], "error": str(error)})
                print(json.dumps(failed[-1], ensure_ascii=True), flush=True)
            if len(done) % 10 == 0 and done:
                print(f"Verified {len(done)}/{len(samples)} videos", flush=True)
    ready = [done[row["id"]] for row in samples if row["id"] in done]
    if len({row["source_sha256"] for row in ready}) != len(ready):
        raise ValueError("Same source video payload appears under multiple identities")
    pending_manifest = manifest_path.with_suffix(".tmp")
    save(pending_manifest, {**plan, "complete": not failed, "failed": failed, "samples": ready})
    pending_manifest.replace(manifest_path)
    print(json.dumps({"ready": len(ready), "failures": len(failed)}), flush=True)


if __name__ == "__main__":
    main()

