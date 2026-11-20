"""Build a bounded local collection of official Elementary23 reference videos."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5, sha256
import json
from pathlib import Path
import re
import struct
import subprocess
import urllib.request
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED
import zlib

from acquire_fresh_continuous import write_manifest
from build_elementary_library import stage_clip
from dataset_fetch import RemoteZipFile
from prepare_elementary_catalog_plan import RECORD_URL, ROOT, SOURCE_DIR, single_word, word_key, workbook_rows


OUTPUT = ROOT / "data/sign-library-expansion"
WORKBOOK = SOURCE_DIR / "All_Elementary.xlsx"
METADATA = SOURCE_DIR / "elementary23-zenodo-metadata.json"
ARCHIVES = {
    "glossa": "Greek Language.zip",
    "mp": "Environmental Studies.zip",
    "math": "Mathematics.zip",
    "anth": "Anthology of Greek Literacy.zip",
    "istoria": "History.zip",
    "thrisk": "Religious Study.zip",
}
PHRASE = re.compile(r"[α-ωάέήίόύώϊϋΐΰ]+(?:[ -]+[α-ωάέήίόύώϊϋΐΰ]+){1,4}")
LOCAL_HEADER = struct.Struct("<4s5H3I2H")
MAX_MEMBER = 16 * 1024**2


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def official():
    source = load(METADATA)
    details = source["metadata"]
    files = {row["key"]: row for row in source["files"]}
    if (source["id"] != 7847052 or details["access_right"] != "open" or details["license"]["id"] != "cc-by-4.0"
            or files[WORKBOOK.name]["checksum"] != "md5:" + md5(WORKBOOK.read_bytes()).hexdigest()):
        raise ValueError("Unexpected official metadata, workbook or license")
    return files, {
        "recordUrl": RECORD_URL, "title": details["title"],
        "creators": [row["name"] for row in details["creators"]],
        "license": {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
    }


def archive_index(spec):
    cached = OUTPUT / "indexes" / (Path(spec["key"]).stem + ".json")
    if cached.is_file():
        payload = load(cached)
        if payload["spec"] != spec:
            raise ValueError(f"Archive metadata changed: {spec['key']}")
        return payload["members"], 0
    with RemoteZipFile(spec["links"]["self"], spec["size"]) as remote, ZipFile(remote) as archive:
        infos = archive.infolist()
        if len(infos) != len({item.filename for item in infos}):
            raise ValueError(f"Duplicate ZIP paths: {spec['key']}")
        members = {}
        for item in infos:
            if Path(item.filename).suffix.lower() != ".flv":
                continue
            identity = Path(item.filename).stem
            if identity in members:
                raise ValueError(f"Duplicate FLV ID: {spec['key']}/{identity}")
            members[identity] = {
                "name": item.filename, "offset": item.header_offset, "compressed": item.compress_size,
                "size": item.file_size, "crc": item.CRC, "method": item.compress_type, "flags": item.flag_bits,
            }
        transferred = remote.bytes_received
    cached.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(cached, {"spec": spec, "members": members})
    return members, transferred


def transcript(text):
    word = single_word(text)
    if word is not None:
        return word, "word"
    phrase = " ".join(text.strip().rstrip("!?;.,·…").lower().split())
    return (phrase, "phrase") if PHRASE.fullmatch(phrase) else (None, None)


def existing_references(official_rows):
    catalog = ROOT / "web/public/library/catalog.json"
    blocked = {word_key(text) for row in load(catalog)["entries"] for text in [row["phrase"], *row.get("aliases", [])]}
    media = [
        *(ROOT / "web/public/library").glob("elementary-*.mp4"),
        *(ROOT / "data/sign-reference-videos/media").glob("elementary-*.mp4"),
        *(ROOT / ".runtime-tmp/elementary-library-100").glob("*.flv"),
        *(ROOT / "data").glob("appointment-*/rgb/elementary-*/source.flv"),
        *(ROOT / "data").glob("elementary-*/rgb/elementary-*/source.flv"),
    ]
    for path in media:
        source_id = (path.parent.name if path.stem == "source" else path.stem)
        source_id = source_id.removeprefix("elementary-").replace("-", "_")
        row = official_rows.get(source_id)
        if row and path.stat().st_size:
            blocked.add(word_key(row["phrase"]))
    return blocked


def inventory(files, workers):
    rows = {}
    for number, identity, signer, text in workbook_rows(WORKBOOK):
        phrase, kind = transcript(text)
        archive = ARCHIVES.get(identity.split("_", 1)[0])
        if phrase and archive:
            if identity in rows:
                raise ValueError(f"Duplicate workbook source ID: {identity}")
            rows[identity] = {"source_id": identity, "signer": signer, "workbook_row": number,
                              "phrase": phrase, "transcription": text, "kind": kind, "archive": archive}
    with ThreadPoolExecutor(max_workers=min(workers, len(ARCHIVES))) as pool:
        indexes = list(pool.map(archive_index, [files[name] for name in ARCHIVES.values()]))
    by_archive = {archive: members for archive, (members, _) in zip(ARCHIVES.values(), indexes, strict=True)}
    blocked = existing_references(rows)
    candidates = defaultdict(list)
    for row in rows.values():
        key = word_key(row["phrase"])
        member = by_archive[row["archive"]].get(row["source_id"])
        if key in blocked or not member:
            continue
        length = member["compressed"] + 1024
        if (member["flags"] & 1 or member["method"] not in (ZIP_DEFLATED, ZIP_STORED)
                or not 0 < member["compressed"] <= member["size"] <= MAX_MEMBER
                or member["offset"] + length > files[row["archive"]]["size"]):
            continue
        candidates[key].append({**row, "key": key, "range_bytes": length, "member": member})
    cheapest = [min(options, key=lambda row: (row["range_bytes"], row["source_id"])) for options in candidates.values()]
    return cheapest, sum(transferred for _, transferred in indexes), len(blocked)


def preflight(candidates, index_bytes, blocked_count):
    by_kind = {}
    for kind in ("word", "phrase"):
        pool = sorted((row for row in candidates if row["kind"] == kind), key=lambda row: row["range_bytes"])
        by_kind[kind] = {
            "unique": len(pool), "cheapest_by_count_bytes": {
                str(n): sum(row["range_bytes"] for row in pool[:n])
                for n in (200, 500, 1000, 1500, 2000) if n <= len(pool)
            },
        }
    return {"existing_reference_keys": blocked_count, "index_transfer_bytes_this_run": index_bytes,
            "by_kind": by_kind, "by_archive": dict(Counter(row["archive"] for row in candidates))}


def choose(candidates, count, kind, reserve, cap, word_target):
    pool = sorted((row for row in candidates if kind == "mixed" or row["kind"] == kind),
                  key=lambda row: (row["range_bytes"], row["source_id"]))
    selected = []
    if kind == "word":
        for archive in list(ARCHIVES.values())[:3]:
            selected.extend([row for row in pool if row["archive"] == archive][:count // 8])
    elif word_target is not None:
        selected.extend([row for row in pool if row["kind"] == "word"][:word_target])
        selected.extend([row for row in pool if row["kind"] == "phrase"][:count - word_target])
        if len(selected) != count:
            raise ValueError("The requested word/phrase mix is unavailable")
    taken = {row["key"] for row in selected}
    selected.extend(row for row in pool if row["key"] not in taken)
    result = selected[:count + reserve]
    if len(result) < count or sum(row["range_bytes"] for row in result) > cap:
        raise ValueError("The requested cohort does not fit the indexed candidate pool or range cap")
    return result


def download(row, spec):
    member = row["member"]
    start, length = member["offset"], row["range_bytes"]
    end = start + length - 1
    request = urllib.request.Request(spec["links"]["self"], headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=22) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{spec['size']}":
            raise ValueError("Server returned a different media range")
        block = response.read(length + 1)
    if len(block) != length:
        raise ValueError("Incomplete range")
    return extract_member(row, block)


def extract_member(row, block):
    member = row["member"]
    signature, _, flags, method, _, _, _, _, _, name_length, extra_length = LOCAL_HEADER.unpack_from(block)
    name = block[LOCAL_HEADER.size:LOCAL_HEADER.size + name_length].decode("utf-8" if flags & 2048 else "cp437")
    compressed_start = LOCAL_HEADER.size + name_length + extra_length
    compressed = block[compressed_start:compressed_start + member["compressed"]]
    if (signature != b"PK\x03\x04" or name != member["name"] or method != member["method"]
            or len(compressed) != member["compressed"]):
        raise ValueError("Local ZIP member does not match the official directory")
    if method == ZIP_DEFLATED:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        payload = decoder.decompress(compressed, member["size"] + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Invalid compressed stream")
    else:
        payload = compressed
    if len(payload) != member["size"] or zlib.crc32(payload) != member["crc"]:
        raise ValueError("Official member length or CRC mismatch")
    return payload


def acquire(row, files, source, media, block=None):
    archive = files[row["archive"]]
    member = row["member"]
    original = media / (row["source_id"] + ".flv")
    payload = original.read_bytes() if original.exists() else b""
    transferred = 0
    if len(payload) != member["size"] or zlib.crc32(payload) != member["crc"]:
        payload = download(row, archive) if block is None else extract_member(row, block)
        temporary = original.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(original)
        transferred = row["range_bytes"] if block is None else 0
    info = ZipInfo(member["name"])
    info.file_size, info.compress_size, info.CRC = member["size"], member["compressed"], member["crc"]
    info.flag_bits, info.compress_type, info.header_offset = member["flags"], member["method"], member["offset"]
    details = {**source, "archiveUrl": archive["links"]["self"], "archiveBytes": archive["size"]}
    entry, review, extra = stage_clip(details, row, info, info.header_offset + row["range_bytes"],
                                      media, verify_timestamps=True, preserve_time_base=True)
    if extra:
        raise ValueError("Transcode unexpectedly requested a second range")
    entry["contentKind"] = row["kind"]
    entry["verification"]["basis"] = "Ακριβής επίσημη ελληνική μεταγραφή για αυτό το πλήρες βίντεο."
    path = media / (entry["id"] + ".mp4")
    return {**review, "kind": row["kind"], "archive": row["archive"], "path": path.relative_to(ROOT).as_posix(),
            "entry": entry, "range_bytes_this_run": transferred}


def run(args):
    files, source = official()
    candidates, index_bytes, blocked_count = inventory(files, args.workers)
    evidence = preflight(candidates, index_bytes, blocked_count)
    print(json.dumps(evidence), flush=True)
    exclusion_plans = [OUTPUT / name / "manifest.planned.json" for name in args.exclude_plan]
    excluded = {row["key"] for path in exclusion_plans for row in load(path)["rows"]}
    candidates = [row for row in candidates if row["key"] not in excluded]
    rows = choose(candidates, args.count, args.kind, args.reserve, args.cap, args.word_target)
    print(json.dumps({"planned": len(rows), "media_range_bytes": sum(row["range_bytes"] for row in rows),
                      "subjects": Counter(row["archive"] for row in rows),
                      "kinds": Counter(row["kind"] for row in rows),
                      "excluded_prior_planned_keys": len(excluded)}), flush=True)
    if args.inventory:
        return
    cohort = OUTPUT / args.name
    media = cohort / "media"
    media.mkdir(parents=True, exist_ok=True)
    plan_path = cohort / "manifest.planned.json"
    plan = {
        "source": source, "workbook_sha256": sha(WORKBOOK), "metadata_sha256": sha(METADATA),
        "requested": args.count, "kind": args.kind, "media_range_cap_bytes": args.cap,
        "media_range_bytes_upper_bound": sum(row["range_bytes"] for row in rows), "rows": rows,
    }
    if exclusion_plans:
        plan["exclusion_plan_sha256"] = {path.relative_to(ROOT).as_posix(): sha(path) for path in exclusion_plans}
    if plan_path.is_file():
        if load(plan_path) != plan:
            raise ValueError("Frozen acquisition plan changed")
    else:
        write_manifest(plan_path, plan)
    output = cohort / "manifest.json"
    prior = load(output) if output.exists() else None
    if prior is not None and prior["plan_sha256"] != sha(plan_path):
        raise ValueError("Existing results came from another plan")
    cached = {row["workbook_video_id"]: row for row in (prior or {}).get("clips", [])}
    clips, failures, seen = [], [], set()

    def task(row):
        previous = cached.get(row["source_id"])
        if previous:
            path = ROOT / previous["path"]
            if path.is_file() and sha(path) == previous["mp4_sha256"]:
                return previous, None
        try:
            return acquire(row, files, source, media), None
        except (OSError, ValueError, zlib.error, subprocess.SubprocessError) as exc:
            return None, {"source_id": row["source_id"], "error": str(exc)}

    def checkpoint():
        write_manifest(output, {
            "source": source, "plan_sha256": sha(plan_path), "requested": args.count,
            "complete": len(clips) == args.count, "clips": clips, "failures": failures,
            "word_count": sum(row["kind"] == "word" for row in clips),
            "phrase_count": sum(row["kind"] == "phrase" for row in clips),
            "selected_range_bytes_upper_bound": plan["media_range_bytes_upper_bound"],
        })

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(rows), args.workers):
            batch = rows[start:start + min(args.workers, args.count - len(clips))]
            for clip, error in pool.map(task, batch):
                if error:
                    failures.append(error)
                elif clip["original_sha256"] in seen:
                    failures.append({"source_id": clip["workbook_video_id"], "error": "Duplicate original content"})
                else:
                    clips.append(clip)
                    seen.add(clip["original_sha256"])
            if args.count <= 240 or start % (args.workers * 3) == 0 or len(clips) == args.count:
                checkpoint()
                print(json.dumps({"processed": len(clips) + len(failures), "verified": len(clips),
                                  "failures": len(failures)}), flush=True)
            if len(clips) == args.count:
                break
    if len(clips) < args.count:
        checkpoint()
        raise ValueError(f"Verified {len(clips)}/{args.count}; partial checkpoint kept")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--kind", choices=("word", "phrase", "mixed"), default="word")
    parser.add_argument("--reserve", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=int, default=100_000_000)
    parser.add_argument("--name", default="words-200")
    parser.add_argument("--exclude-plan", action="append", default=[])
    parser.add_argument("--word-target", type=int)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.count <= 3000 or not 1 <= args.workers <= 16 or not 0 <= args.reserve <= 256
            or not 0 < args.cap <= 2_000_000_000
            or not all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) for name in (args.name, *args.exclude_plan))
            or args.name in args.exclude_plan
            or (args.word_target is not None and (args.kind != "mixed" or not 0 <= args.word_target <= args.count))):
        parser.error("Invalid bounded acquisition arguments")
    run(args)


if __name__ == "__main__":
    main()
