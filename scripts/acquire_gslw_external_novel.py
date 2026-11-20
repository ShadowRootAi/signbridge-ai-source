"""Stage eight official GSLW phrase videos via bounded split-ZIP range reads."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from pathlib import Path
import re
import struct
import subprocess
import urllib.request
import zlib

import imageio_ffmpeg

from acquire_continuous_scale import near
from acquire_fresh_continuous import DATA, ROOT, digest, read_json, write_manifest
from plan_continuous_composition import normalize_greek


OUTPUT = DATA / "gslw-external-novel"
RECORD = "https://zenodo.org/records/8047231"
CSV_ROWS = (1480, 1486, 1509, 1514, 1607, 1611, 1662, 1680)
PROTECTED = (
    DATA / "continuous-scale/manifest.json",
    DATA / "continuous-scale-novel/manifest.json",
    DATA / "continuous-span-expansion/manifest.planned.json",
    DATA / "sartinas-dialogue-fit-expanded/manifest.json",
)
MAX_MEDIA_BYTES = 92_000_000
FPS = 30
HEADER = struct.Struct("<4s6H3L5H2L")


def central_directory(payload):
    offset = 0
    entries = {}
    while offset < len(payload):
        (signature, _, _, flag, method, _, _, crc, compressed, size, name_size,
         extra, comment, disk, _, _, start) = HEADER.unpack_from(payload, offset)
        if signature != b"PK\x01\x02":
            raise ValueError("Invalid split ZIP directory signature")
        end = offset + HEADER.size + name_size
        name = payload[offset + HEADER.size:end].decode("utf-8" if flag & 2048 else "cp437")
        if name in entries:
            raise ValueError(f"Duplicate ZIP member name: {name}")
        entries[name] = (disk, start, compressed, size, method, crc)
        offset = end + extra + comment
    if offset != len(payload) or len(entries) != 1647:
        raise ValueError("Unexpected split ZIP directory size")
    return entries


def source_files():
    metadata = read_json(OUTPUT / "zenodo-metadata.json")
    if metadata["id"] != 8047231 or metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected GSLW source or license")
    return {row["key"]: row for row in metadata["files"]}


def build_plan():
    files = source_files()
    directory = OUTPUT / "central-directory.bin"
    annotations = OUTPUT / "official.csv"
    entries = central_directory(directory.read_bytes())
    blocked_text, blocked_gloss, vocabulary = set(), set(), set()
    for path in PROTECTED:
        for row in read_json(path)["samples"]:
            if path.parent.name == "sartinas-dialogue-fit-expanded" and row["split"] != "train":
                continue
            glosses = tuple(row.get("glosses", []))
            blocked_gloss.add(glosses)
            blocked_text.add(tuple(normalize_greek(row["text"]).split()))
            if row["split"] == "train":
                vocabulary.update(glosses)
    selected = []
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if line not in CSV_ROWS:
                continue
            source, gold, start, end, signer, flip, text, scenario = row
            glosses = tuple(gold.split())
            greek = tuple(normalize_greek(text).split())
            if (not 2 <= len(glosses) <= 7 or not 2 <= len(greek) <= 9
                    or any(near(greek, prior) for prior in blocked_text)
                    or any(near(glosses, prior) for prior in blocked_gloss)):
                raise ValueError(f"Protected or invalid phrase at official CSV line {line}")
            if float(start) < 0 or float(end) <= float(start) or flip != "False":
                raise ValueError(f"Unexpected timing or orientation at official CSV line {line}")
            disk, offset, compressed, size, method, crc = entries[source]
            part = "gslw.zip" if disk == 21 else f"gslw.z{disk + 1:02d}"
            range_bytes = compressed + 1024
            if (not 0 < compressed <= size < 22_000_000 or method not in (0, 8)
                    or offset + range_bytes > files[part]["size"]):
                raise ValueError(f"Unsafe source range at official CSV line {line}")
            selected.append({
                "id": f"gslw-line-{line}", "split": "test", "signer": f"gslw:{signer}",
                "category": "external_scripted_phrase", "composition": "unseen_template",
                "source_path": source, "source_archive": part, "scenario": scenario, "fps": FPS,
                "text": text, "glosses": list(glosses), "annotation_row": line,
                "out_of_fit_glosses": sorted(set(glosses) - vocabulary),
                "annotation_start_raw": start, "annotation_end_raw": end,
                "annotation_flip": False, "archive_crc32": f"{crc:08x}",
                "offset": offset, "compressed_bytes": compressed, "raw_bytes": size,
                "zip_method": method, "range_bytes": range_bytes,
            })
    counts = Counter(row["signer"] for row in selected)
    expected_signers = {f"gslw:signer{signer}": 2 for signer in (8, 9, 10, 13)}
    if (counts != expected_signers or len({row["archive_crc32"] for row in selected}) != len(CSV_ROWS)
            or len({normalize_greek(row["text"]) for row in selected}) != len(CSV_ROWS)):
        raise ValueError("Expected two unique source videos and phrases per signer")
    requested = sum(row["range_bytes"] for row in selected)
    if requested > MAX_MEDIA_BYTES:
        raise ValueError(f"Planned media transfer {requested} exceeds cap {MAX_MEDIA_BYTES}")
    inputs = (annotations, directory, OUTPUT / "zenodo-metadata.json", *PROTECTED)
    return {
        "schema": 1, "source": RECORD, "license": "CC-BY-4.0", "complete": False,
        "scope": "Eight distinct official whole-phrase videos across four GSLW signer IDs. "
                 "The corpus publishes separate signer codes but provides no identity crosswalk to prior corpora.",
        "selection": "Source annotation, exact and one-token Greek/gloss exclusion against frozen fit/development "
                     "and primary CERTH evaluation inputs. Selection fixed before video inference.",
        "time_basis": "Each official annotation refers to an independently named single-phrase video. "
                      "The raw timing fields are preserved for comparison with verified video duration.",
        "range_bytes": requested, "range_cap_bytes": MAX_MEDIA_BYTES, "planned_samples": len(selected),
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in inputs},
        "samples": selected,
    }


def acquire_member(row, files, budget):
    file = files[row["source_archive"]]
    start, length = row["offset"], row["range_bytes"]
    if length > budget[0]:
        raise ValueError("Remaining media range cap exhausted")
    budget[0] -= length
    end = start + length - 1
    request = urllib.request.Request(file["links"]["self"], headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{file['size']}":
            raise ValueError("Server did not honor the exact planned byte range")
        block = response.read(length + 1)
    if len(block) != length or block[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local range")
    flag, method = struct.unpack_from("<HH", block, 6)
    name_size, extra = struct.unpack_from("<HH", block, 26)
    path = block[30:30 + name_size].decode("utf-8" if flag & 2048 else "cp437")
    if path != row["source_path"] or method != row["zip_method"]:
        raise ValueError("Local ZIP header differs from the planned source")
    offset = 30 + name_size + extra
    compressed = block[offset:offset + row["compressed_bytes"]]
    if len(compressed) != row["compressed_bytes"]:
        raise ValueError("Incomplete planned compressed media member")
    return zlib.decompress(compressed, -15) if method == 8 else compressed


def materialize(row, files, budget):
    target = OUTPUT / "rgb" / row["id"]
    target.mkdir(parents=True, exist_ok=True)
    video = target / ("source" + Path(row["source_path"]).suffix.lower())
    payload = video.read_bytes() if video.exists() else acquire_member(row, files, budget)
    if len(payload) != row["raw_bytes"] or f"{zlib.crc32(payload):08x}" != row["archive_crc32"]:
        raise ValueError(f"CRC verification failed: {row['id']}")
    if not video.exists():
        video.write_bytes(payload)
    reader = imageio_ffmpeg.read_frames(video, input_params=["-threads", "1"])
    try:
        metadata = next(reader)
    finally:
        reader.close()
    duration = metadata["duration"]
    if not 2 <= duration <= 25:
        raise ValueError(f"Unexpected short-phrase media duration {duration}: {row['id']}")
    frame_pattern = re.compile(r"frame-\d{5}\.jpg")
    for path in target.iterdir():
        if path.is_file() and frame_pattern.fullmatch(path.name):
            path.unlink()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
        "-threads", "1", "-i", str(video), "-map", "0:v:0", "-an",
        "-vf", r"fps=30,scale=w=min(720\,iw):h=-2", "-threads", "1", "-q:v", "3", str(target / "frame-%05d.jpg"),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    frames = sorted(target.glob("frame-*.jpg"))
    if (not frames or any(path.name != f"frame-{i:05d}.jpg" for i, path in enumerate(frames, 1))
            or abs(len(frames) / FPS - duration) > .3):
        raise ValueError(f"Decoded frame timeline disagrees with the media duration: {row['id']}")
    return {**row, "source_video_sha256": sha256(payload).hexdigest(), "source_duration_seconds": duration,
            "source_video_fps": metadata["fps"], "frame_count": len(frames),
            "frame_paths": [path.relative_to(ROOT).as_posix() for path in frames],
            "frame_sha256": [digest(path) for path in frames]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    target = OUTPUT / "manifest.planned.json"
    if target.is_file():
        if read_json(target) != plan:
            raise ValueError("The sealed GSLW plan or source evidence changed")
    else:
        write_manifest(target, plan)
    print(f"Sealed {plan['planned_samples']} GSLW phrases, {plan['range_bytes']:,} bytes, SHA256 {digest(target)}", flush=True)
    if not args.acquire:
        return
    budget, files = [MAX_MEDIA_BYTES], source_files()
    completed = []
    for row in plan["samples"]:
        item = materialize(row, files, budget)
        completed.append(item)
        print(f"{len(completed)}/{len(CSV_ROWS)} {row['id']}: {item['source_duration_seconds']} s", flush=True)
    if len({row["source_video_sha256"] for row in completed}) != len(CSV_ROWS):
        raise ValueError("Selected video source bytes were duplicated")
    write_manifest(OUTPUT / "manifest.json", {
        **plan, "complete": True, "samples": completed, "successful_range_bytes_this_run": MAX_MEDIA_BYTES - budget[0],
    })


if __name__ == "__main__":
    main()
