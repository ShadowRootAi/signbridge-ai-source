"""Acquire uncut official GSLW phone recordings with one annotated out-of-scope gloss."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zlib

import imageio_ffmpeg

from acquire_fresh_continuous import ROOT, digest, read_json, write_manifest
from acquire_gslw_external_novel import CSV_ROWS, OUTPUT as SOURCE, RECORD, acquire_member, central_directory, source_files
from plan_continuous_composition import normalize_greek


OUTPUT = ROOT / ".runtime-tmp/recognition-phone-negatives"
STUDIO = ROOT / ".runtime-tmp/recognition-scale-v2/manifest.json"
ROLES = {1474: "train", 1502: "train", 1505: "train", 1744: "validation",
         1731: "test", 1737: "test", 1776: "test"}
EXPECTED_SIGNERS = {"train": {"bill"}, "validation": {"signer11"}, "test": {"signer3", "signer1"}}
TRANSFER_LIMIT = 125_000_000


def make_plan(files):
    csv_path, directory = SOURCE / "official.csv", SOURCE / "central-directory.bin"
    index = central_directory(directory.read_bytes())
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        annotations = list(csv.reader(stream))
    paths = Counter(row[0] for row in annotations if len(row) == 8)
    v2 = read_json(STUDIO)
    blocked = {normalize_greek(word) for word in v2["labels"]}
    blocked.update(normalize_greek(item["original_text"]) for item in
                   (*v2["samples"], *v2.get("previously_used_diagnostic", [])))
    previous_signers = {"signer8", "signer9", "signer10", "signer13"}
    result, used_gloss = [], set()
    for line, role in ROLES.items():
        source, gloss, start, end, signer, flip, text, scenario = annotations[line - 1]
        key = normalize_greek(text)
        start, end = float(start), float(end)
        if (paths[source] != 1 or len(gloss.split()) != 1 or not key or key in blocked
                or gloss in used_gloss or flip != "False" or signer not in EXPECTED_SIGNERS[role]
                or signer in previous_signers or line in CSV_ROWS
                or not (start >= 0 and (end > start or (start == 0 and end == -1)))):
            raise ValueError(f"The frozen sign, signer, timing, or text conflicts at GSLW row {line}")
        disk, offset, compressed, size, method, crc = index[source]
        part = "gslw.zip" if disk == 21 else f"gslw.z{disk + 1:02d}"
        length = compressed + 1024
        if (not 0 < compressed < 22_000_000 or not 0 < size < 22_000_000 or method not in (0, 8)
                or offset + length > files[part]["size"]):
            raise ValueError(f"Unsafe split ZIP range at row {line}")
        result.append({"id": f"gslw-line-{line}", "label": None, "is_known": False, "role": role,
                       "signer": f"gslw:{signer}", "gloss": gloss, "original_text": text, "scenario": scenario,
                       "source_path": source, "source_archive": part, "annotation_row": line,
                       "annotation_start_seconds": start, "annotation_end_seconds": end,
                       "annotation_flip": False, "archive_crc32": f"{crc:08x}", "offset": offset,
                       "compressed_bytes": compressed, "raw_bytes": size, "zip_method": method,
                       "range_bytes": length})
        used_gloss.add(gloss)
        blocked.add(key)
    planned_bytes = sum(item["range_bytes"] for item in result)
    if planned_bytes > TRANSFER_LIMIT or len({item["archive_crc32"] for item in result}) != len(result):
        raise ValueError("Media budget exceeded or source content repeated")
    sources = (csv_path, directory, SOURCE / "zenodo-metadata.json", STUDIO)
    return {"schema_version": 1, "source": RECORD, "license": "CC-BY-4.0", "labels": [],
            "scope": "Smartphone source recordings with one official gloss annotation per unique source file. "
                     "Store the complete RGB video without trimming or horizontal flip. Finite annotation windows "
                     "do not establish the content of frames outside those windows. These files are negatives only.",
            "range_cap_bytes": TRANSFER_LIMIT, "range_bytes": planned_bytes,
            "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
            "samples": result}


def materialize(row, files, media):
    source = media / (row["id"] + ".source" + Path(row["source_path"]).suffix.lower())
    payload = source.read_bytes() if source.exists() else acquire_member(row, files, [row["range_bytes"]])
    if len(payload) != row["raw_bytes"] or f"{zlib.crc32(payload):08x}" != row["archive_crc32"]:
        raise ValueError("Source content fails ZIP directory CRC or size")
    if not source.exists():
        pending = source.with_suffix(".part")
        pending.write_bytes(payload)
        pending.replace(source)
    original_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(source))
    start, end = row["annotation_start_seconds"], row["annotation_end_seconds"]
    if (not .2 <= duration <= 15 or original_frames < 3 or start >= duration or end > duration + .2):
        raise ValueError(f"Source duration or annotation timing invalid: {duration}s with annotation [{start}, {end}]")
    output = media / (row["id"] + ".mp4")
    pending = output.with_suffix(".part.mp4")
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                        "-y", "-threads", "1", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
                        r"scale=w=min(720\,iw):h=-2:flags=lanczos", "-fps_mode", "passthrough", "-enc_time_base", "filter",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", "-crf", "21", "-movflags", "+faststart",
                        str(pending)], check=True, capture_output=True, timeout=60)
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(pending))
        if frames != original_frames or abs(seconds - duration) > .2 or not .2 <= seconds <= 15:
            raise ValueError(f"Transcode changed the full timeline: {frames} frames/{seconds}s vs {original_frames}/{duration}s")
        pending.replace(output)
    except subprocess.CalledProcessError as error:
        raise ValueError(error.stderr.decode(errors="replace").strip()) from error
    finally:
        pending.unlink(missing_ok=True)
    return {**row, "original_sha256": sha256(payload).hexdigest(), "sha256": digest(output),
            "path": output.relative_to(ROOT).as_posix(), "frames": frames, "duration_seconds": seconds,
            "annotation_covers_whole_file": start <= .1 and (end == -1 or end >= duration - .1)}


def run(acquire):
    files = source_files()
    plan = make_plan(files)
    print(json.dumps({"planned": len(plan["samples"]), "range_bytes": plan["range_bytes"], "roles":
                      Counter(row["role"] for row in plan["samples"])}, ensure_ascii=False), flush=True)
    if not acquire:
        return
    media = OUTPUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "manifest.planned.json"
    if frozen.exists() and plan != read_json(frozen):
        raise ValueError("The frozen smartphone cohort changed")
    if not frozen.exists():
        write_manifest(frozen, plan)
    frozen_sha256 = digest(frozen)
    report = OUTPUT / "manifest.json"
    approved = read_json(report) if report.exists() else {}
    if approved and approved.get("plan_sha256") != frozen_sha256:
        raise ValueError("Prior smartphone manifest references a different plan")
    cached = {}
    planned = {row["id"]: row for row in plan["samples"]}
    for sample in approved.get("samples", []):
        row = planned[sample["id"]]
        path = media / f"{sample['id']}.mp4"
        if (all(sample.get(key) == value for key, value in row.items())
                and sample["path"] == path.relative_to(ROOT).as_posix()
                and path.is_file() and digest(path) == sample["sha256"]):
            cached[sample["id"]] = sample
    if approved.get("complete") and len(cached) == len(planned):
        print(json.dumps({"verified": len(cached), "reused": len(cached), "complete": True}), flush=True)
        return

    def stage(row):
        if row["id"] in cached:
            return cached[row["id"]], None
        try:
            return materialize(row, files, media), None
        except (OSError, ValueError, zlib.error, subprocess.SubprocessError) as error:
            return None, {"id": row["id"], "error": str(error)}

    samples, failures, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sample, failure in pool.map(stage, plan["samples"]):
            if failure:
                failures.append(failure)
            elif sample["original_sha256"] in seen:
                failures.append({"id": sample["id"], "error": "Duplicate source bytes"})
            else:
                samples.append(sample)
                seen.add(sample["original_sha256"])
            write_manifest(report, {**plan, "plan_sha256": frozen_sha256,
                "complete": len(samples) == len(plan["samples"]), "samples": samples, "failures": failures})
            print(json.dumps({"verified": len(samples), "failures": failures[-1:] if failure else []}), flush=True)
    if failures:
        raise RuntimeError("Some smartphone recordings failed verification. See manifest.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="Fetch only the frozen source ranges")
    run(parser.parse_args().acquire)
