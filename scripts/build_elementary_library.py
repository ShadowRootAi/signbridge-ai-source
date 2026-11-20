"""Publish complete Elementary videos matched to exact Greek workbook rows."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import zipfile
import zlib

import imageio_ffmpeg

from dataset_fetch import RemoteZipFile
from prepare_elementary_catalog_plan import ROOT, SOURCE_DIR, single_word, word_key, workbook_rows


REPORTS = ROOT / "reports/library-100"
STAGE = ROOT / ".runtime-tmp/elementary-library-100"
LIBRARY = ROOT / "web/public/library"
BASE = ROOT / "web/deployment/expanded-library"
VIDEO_DIR = "Greek Language/Videos"
MODIFICATIONS = (
    "Complete original FLV transcoded to H.264/yuv420p MP4, CRF21, with dimensions "
    "at most 720 pixels wide and audio omitted. Original frame order, speed and orientation preserved."
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_original(source, member, end):
    if not 0 < member.file_size <= 16 * 1024**2 or member.flag_bits & 1:
        raise ValueError(f"Unexpected source member: {member.filename}")
    with RemoteZipFile(source["archiveUrl"], source["archiveBytes"]) as remote:
        remote.seek(member.header_offset)
        block = remote.read(end - member.header_offset)
        transferred = remote.bytes_received

    signature, _, _, method, _, _, _, _, _, name_length, extra_length = struct.unpack_from(
        "<4s5H3I2H", block
    )
    start = 30 + name_length + extra_length
    compressed = block[start:start + member.compress_size]
    if (
        signature != b"PK\x03\x04"
        or method != member.compress_type
        or block[30:30 + name_length] != member.filename.encode("ascii")
        or len(compressed) != member.compress_size
    ):
        raise ValueError(f"Local and central ZIP headers disagree: {member.filename}")

    if method == zipfile.ZIP_STORED:
        data = compressed
    elif method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        data = decoder.decompress(compressed, member.file_size + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError(f"Incomplete or overlong deflate stream: {member.filename}")
    else:
        raise ValueError(f"Unsupported ZIP compression: {method}")
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError(f"ZIP size or CRC mismatch: {member.filename}")
    return data, transferred


def decoded_timestamps(path):
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1",
        "-i", str(path), "-map", "0:v:0", "-an", "-vf", "showinfo", "-f", "null", "-",
    ], check=True, capture_output=True, timeout=60)
    values = [float(value) for value in re.findall(rb"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-+\d.eE]+)", result.stderr)]
    return [value - values[0] for value in values] if values else []


def stage_clip(source, row, member, end, stage=STAGE, *, verify_timestamps=False, preserve_time_base=False):
    source_id = row["source_id"]
    entry_id = f"elementary-{source_id.replace('_', '-')}"
    original = stage / f"{source_id}.flv"
    converted = stage / f"{entry_id}.mp4"
    transferred = 0
    data = original.read_bytes() if original.exists() else b""
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        data, transferred = read_original(source, member, end)
        partial = original.with_suffix(".flv.part")
        partial.write_bytes(data)
        partial.replace(original)

    pending = converted.with_suffix(".part.mp4")
    try:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-y",
                "-threads", "1", "-i", str(original), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", r"scale=w=min(720\,iw):h=-2:flags=lanczos", "-fps_mode", "passthrough",
                *(["-enc_time_base", "filter"] if preserve_time_base else []),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1",
                "-preset", "veryfast", "-crf", "21", "-movflags", "+faststart", str(pending),
            ],
            check=True, capture_output=True, timeout=60,
        )
        original_frames, original_seconds = imageio_ffmpeg.count_frames_and_secs(str(original))
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(pending))
        timing_matches = abs(duration - original_seconds) <= 0.12
        if not timing_matches and verify_timestamps and frames == original_frames:
            original_times, converted_times = decoded_timestamps(original), decoded_timestamps(pending)
            timing_matches = (len(original_times) == len(converted_times) == frames
                              and all(abs(a - b) <= .015 for a, b in zip(original_times, converted_times, strict=True)))
        if frames < 3 or frames != original_frames or duration <= 0 or not timing_matches:
            raise ValueError(
                f"Transcode changed clip timing: {original_frames}/{original_seconds}s to {frames}/{duration}s"
            )
        pending.replace(converted)
    except subprocess.CalledProcessError as error:
        raise ValueError(error.stderr.decode(errors="replace").strip()) from error
    finally:
        pending.unlink(missing_ok=True)

    sha = digest(converted)
    attribution = f"{source['title']} v3 — {'; '.join(source['creators'])}."
    entry = {
        "id": entry_id,
        "phrase": row["phrase"],
        "aliases": [],
        "contentKind": "word",
        "videoUrl": f"/library/{entry_id}.mp4",
        "durationSeconds": duration,
        "sourceUrl": source["recordUrl"],
        "sourceArchiveUrl": source["archiveUrl"],
        "sourceUtterance": member.filename,
        "sha256": sha,
        "attribution": attribution,
        "license": {**source["license"], "name": "Creative Commons Attribution 4.0 International"},
        "modifications": MODIFICATIONS,
        "rights": {
            "redistributionConfirmed": True,
            "basis": "Official open Zenodo record 7847052, CC BY 4.0.",
            "scope": "Dataset copyright license. No endorsement or independent review of participant releases claimed.",
        },
        "verification": {
            "status": "dataset-annotated",
            "basis": "Η λέξη αντιστοιχεί στη μονολεκτική ελληνική μεταγραφή της πηγής για αυτό το πλήρες βίντεο.",
            "newExpertReview": False,
        },
        "contentNote": "Πλήρες βίντεο της πηγής με τη σειρά και την ταχύτητα του πρωτοτύπου.",
    }
    review = {
        "id": entry_id,
        "phrase": row["phrase"],
        "annotation_label": None,
        "source_record_url": source["recordUrl"],
        "source_path": member.filename,
        "workbook_video_id": source_id,
        "workbook_row": row["workbook_row"],
        "workbook_transcription": row["transcription"],
        "signer_id": row["signer"],
        "archive_member_crc32": f"{member.CRC:08x}",
        "original_sha256": hashlib.sha256(data).hexdigest(),
        "original_bytes": len(data),
        "mp4_sha256": sha,
        "mp4_bytes": converted.stat().st_size,
        "frames": frames,
        "duration_seconds": duration,
        "verified_decodable": True,
    }
    return entry, review, transferred


def choose_clip(source, row, members, ends):
    errors = []
    for choice in [row, *row.get("alternate_sources", [])]:
        if choice["transcription"].strip() != row["phrase"]:
            errors.append(f"Workbook row {choice['workbook_row']} is not an exact text match")
            continue
        member_path = f"{VIDEO_DIR}/{choice['source_id']}.flv"
        member = members.get(member_path)
        if member is None:
            errors.append(f"Member missing: {member_path}")
            continue
        try:
            return stage_clip(source, choice, member, ends[member.header_offset])
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            errors.append(f"{member_path}: {error}")
    raise ValueError("; ".join(errors))


def publish(source, results, failed, plan, metadata_path, bytes_received):
    catalog = load_json(BASE / "catalog.json")
    review = load_json(ROOT / "reports/library-expansion/rights-review.json")
    if len(catalog["entries"]) != 18 or len(review["clips"]) != 18:
        raise ValueError("Expected the 18-entry library snapshot")
    if catalog["reviewSha256"] != digest(ROOT / "reports/library-expansion/rights-review.json"):
        raise ValueError("Legacy catalog and review disagree")
    entries = [entry for entry, _, _ in results]
    new_review = [clip for _, clip, _ in results]
    all_entries = [*catalog["entries"], *entries]
    keys = [word_key(term) for entry in all_entries for term in [entry["phrase"], *entry["aliases"]]]
    if len(set(keys)) != len(keys) or len({entry["id"] for entry in all_entries}) != len(all_entries):
        raise ValueError("Duplicate normalized phrases or IDs")
    if len({clip["original_sha256"] for clip in new_review}) != len(new_review):
        raise ValueError("Two Greek transcriptions point to the same original video")
    for entry in catalog["entries"]:
        if digest(LIBRARY / f"{entry['id']}.mp4") != entry["sha256"]:
            raise ValueError(f"Legacy asset mismatch: {entry['id']}")

    review["schema_version"] = 2
    review["elementary_source"] = {
        "title": source["title"],
        "record_url": source["recordUrl"],
        "doi": source["doi"],
        "access_right": "open",
        "creators": source["creators"],
        "license": {"id": "cc-by-4.0", "url": source["license"]["url"]},
        "metadata_sha256": digest(metadata_path),
        "workbook": source["workbook"],
        "workbook_url": source["workbookUrl"],
        "workbook_sha256": source["workbookSha256"],
        "archive": source["archive"],
        "archive_url": source["archiveUrl"],
        "archive_bytes": source["archiveBytes"],
        "archive_publisher_md5": source["archiveMd5"],
        "verification": "Exact workbook row and signer ID, exact FLV member name and ZIP CRC, full-clip decode and transcode. No gloss or new expert review.",
        "modifications": MODIFICATIONS,
    }
    review["elementary_statistics"] = {
        "planned_words": len(plan["entries"]),
        "published_words": len(entries),
        "signers": dict(sorted(Counter(clip["signer_id"] for clip in new_review).items())),
        "range_bytes_this_run": bytes_received,
        "new_mp4_bytes": sum(clip["mp4_bytes"] for clip in new_review),
        "failed_words": failed,
    }
    review["clips"].extend(new_review)
    review_path = REPORTS / "rights-review.json"
    write_json(review_path, review)
    catalog["scope"] = "Exact lookup of full annotated Greek Sign Language clips from two official datasets."
    catalog["reviewSha256"] = digest(review_path)
    catalog["entries"] = all_entries

    for entry in entries:
        name = f"{entry['id']}.mp4"
        pending = LIBRARY / f"{name}.part"
        shutil.copyfile(STAGE / name, pending)
        pending.replace(LIBRARY / name)

    notice = (BASE / "NOTICE.txt").read_text(encoding="utf-8").rstrip()
    lines = [
        notice, "", entries[0]["attribution"], f"Source: {source['recordUrl']}",
        f"DOI: https://doi.org/{source['doi']}", f"License: CC BY 4.0, {source['license']['url']}",
        MODIFICATIONS,
        "Whole video matched to the publisher's Greek transcription. Supplied without warranties.",
    ]
    for entry, clip in zip(entries, new_review, strict=True):
        lines.extend([
            "", f"{entry['videoUrl']}: {entry['phrase']}",
            f"Source archive: {source['archiveUrl']}", f"Source member: {entry['sourceUtterance']}",
            f"Workbook: {source['workbook']}, row {clip['workbook_row']}, signer {clip['signer_id']}",
            f"SHA256: {entry['sha256']}",
        ])
    (LIBRARY / "NOTICE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pending = LIBRARY / "catalog.json.part"
    write_json(pending, catalog)
    destination = LIBRARY / "catalog.json"
    try:
        pending.replace(destination)
    except PermissionError:
        # A local Windows development server can hold the old catalog open for reading.
        destination.write_bytes(pending.read_bytes())
        pending.unlink()
    print(json.dumps(review["elementary_statistics"], ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--minimum", type=int, default=101)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.minimum < 101:
        parser.error("Use 1–8 workers and require at least 101 words")

    plan = load_json(REPORTS / "elementary-plan.json")
    source = plan["source"]
    metadata_path = SOURCE_DIR / "elementary23-zenodo-metadata.json"
    metadata = load_json(metadata_path)
    archive = next(file for file in metadata["files"] if file["key"] == source["archive"])
    workbook = SOURCE_DIR / source["workbook"]
    if (
        metadata["id"] != 7847052
        or metadata["metadata"]["access_right"] != "open"
        or metadata["metadata"]["license"]["id"] != "cc-by-4.0"
        or source["recordUrl"] != "https://zenodo.org/records/7847052"
        or source["creators"] != [creator["name"] for creator in metadata["metadata"]["creators"]]
        or archive["links"]["self"] != source["archiveUrl"]
        or archive["size"] != source["archiveBytes"]
        or archive["checksum"] != f"md5:{source['archiveMd5']}"
        or digest(workbook) != source["workbookSha256"]
    ):
        raise ValueError("The acquisition plan does not match official open Elementary v3 metadata")
    rows = {row: (video_id, signer, transcription) for row, video_id, signer, transcription in workbook_rows(workbook)}
    for entry in plan["entries"]:
        for choice in [entry, *entry.get("alternate_sources", [])]:
            expected = (choice["source_id"], choice["signer"], choice["transcription"])
            if rows[choice["workbook_row"]] != expected or single_word(choice["transcription"]) != entry["phrase"]:
                raise ValueError(f"Plan does not match workbook row {choice['workbook_row']}")

    STAGE.mkdir(parents=True, exist_ok=True)
    with RemoteZipFile(source["archiveUrl"], source["archiveBytes"]) as remote, zipfile.ZipFile(remote) as archive_zip:
        infos = sorted(archive_zip.infolist(), key=lambda member: member.header_offset)
        members = {member.filename: member for member in infos}
        if len(members) != len(infos):
            raise ValueError("Duplicate ZIP member paths")
        offsets = [member.header_offset for member in infos]
        ends = dict(zip(offsets, [*offsets[1:], archive_zip.start_dir], strict=True))
        bytes_received = remote.bytes_received
    print(f"Checked {len(members)} ZIP members; starting {len(plan['entries'])} clips", flush=True)

    completed = {}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(choose_clip, source, row, members, ends): index
            for index, row in enumerate(plan["entries"])
        }
        for done in as_completed(futures):
            index = futures[done]
            try:
                completed[index] = done.result()
                bytes_received += completed[index][2]
            except ValueError as error:
                failed.append({"phrase": plan["entries"][index]["phrase"], "error": str(error)})
                print(f"Failed candidate #{index + 1}: {error}", flush=True)
            count = len(completed) + len(failed)
            if count % 10 == 0:
                print(f"Processed {count}/{len(futures)}, verified {len(completed)}", flush=True)
    if len(completed) < args.minimum:
        raise SystemExit(f"Only {len(completed)} verified words; catalog was not replaced.")
    publish(source, [completed[index] for index in sorted(completed)], failed, plan, metadata_path, bytes_received)


if __name__ == "__main__":
    main()

