"""Freeze and acquire a small signer-disjoint, exact-text Elementary23 cohort."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import urllib.error
import zlib

from acquire_elementary_coalesced import Pace, download
from acquire_elementary_library_expansion import ARCHIVES, MAX_MEMBER, METADATA, OUTPUT, WORKBOOK, acquire, load, official, sha, transcript
from acquire_fresh_continuous import write_manifest
from prepare_elementary_catalog_plan import ROOT, word_key, workbook_rows


COHORT = ROOT / ".runtime-tmp/recognition-scale-pilot"
VOCABULARY = ("μαθαίνω", "όνομα", "γλώσσα", "θάλασσα", "αυτοκίνητο", "εγώ", "σήμερα", "βουνό")
MEDIA_LIMIT = 40 * 1024**2
WORKERS = 6
TRAIN_SIGNERS = ("A", "B")
VALIDATION_SIGNERS = ("E",)
TEST_SIGNERS = ("C", "G", "F")


def available_rows(files):
    indexes = {}
    for archive in ARCHIVES.values():
        index = load(OUTPUT / "indexes" / (Path(archive).stem + ".json"))
        if index["spec"] != files[archive]:
            raise ValueError(f"Source index does not match the official metadata: {archive}")
        indexes[archive] = index["members"]
    rows = []
    for number, identity, signer, text in workbook_rows(WORKBOOK):
        archive = ARCHIVES.get(identity.split("_", 1)[0])
        label, kind = transcript(text)
        if kind != "word" or not archive or not (member := indexes[archive].get(identity)):
            continue
        length = member["compressed"] + 1024
        if (member["flags"] & 1 or member["method"] not in (0, 8)
                or not 0 < member["compressed"] <= member["size"] <= MAX_MEMBER
                or member["offset"] + length > files[archive]["size"]):
            continue
        rows.append({"source_id": identity, "signer": signer, "workbook_row": number,
                     "phrase": label, "transcription": text, "kind": kind, "archive": archive,
                     "range_bytes": length, "member": member})
    return rows


def previously_verified():
    clips = {}
    for group in ("words-200", "mixed-2512", "coalesced-2331"):
        for row in load(OUTPUT / group / "manifest.json")["clips"]:
            clips[row["workbook_video_id"]] = row
    return clips


def plan_rows(rows, existing):
    by_keyword_signer = defaultdict(list)
    for row in rows:
        by_keyword_signer[word_key(row["phrase"]), row["signer"]].append(row)
    selected, seen_ids, seen_content = [], set(), set()

    def take(options, role, label, count):
        picked = 0
        for row in options:
            member = row["member"]
            signature = (member["size"], member["crc"])
            if row["source_id"] in seen_ids or signature in seen_content:
                continue
            selected.append({**row, "role": role, "label": label})
            seen_ids.add(row["source_id"])
            seen_content.add(signature)
            picked += 1
            if picked == count:
                break
        return picked

    def options(key, signer):
        return sorted(by_keyword_signer[key, signer], key=lambda row: (
            row["source_id"] not in existing, row["range_bytes"], row["source_id"]))

    for label in VOCABULARY:
        key = word_key(label)
        for signer in TRAIN_SIGNERS:
            if not take(options(key, signer), "train", label, 2):
                raise ValueError(f"Missing fit signer {signer} for {label}")
        if not take(options(key, VALIDATION_SIGNERS[0]), "validation", label, 2):
            raise ValueError(f"Missing validation for {label}")
        held_picks = 0
        for signer in TEST_SIGNERS:
            held_picks += take(options(key, signer), "test", label, 2 - held_picks)
            if held_picks == 2:
                break
        if not held_picks:
            raise ValueError(f"Missing held signer for {label}")

    for role, signers in (("validation", VALIDATION_SIGNERS), ("test", TEST_SIGNERS)):
        others = [row for row in rows if row["signer"] in signers and row["source_id"] in existing
                  and word_key(row["phrase"]) not in {word_key(label) for label in VOCABULARY}]
        others.sort(key=lambda row: sha256((role + row["source_id"]).encode()).digest())
        if take(others, role, None, 6) != 6:
            raise ValueError(f"Missing out-of-scope recordings for {role}")
    return selected


def source_metadata(row, clip):
    if (clip["source_path"] != row["member"]["name"] or clip["signer_id"] != row["signer"]
            or clip["workbook_row"] != row["workbook_row"] or clip["workbook_transcription"] != row["transcription"]
            or clip["archive_member_crc32"] != f"{row['member']['crc']:08x}"):
        raise ValueError("Verified media provenance differs from the frozen source row")
    path = (ROOT / clip["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != clip["mp4_sha256"]:
        raise ValueError("Verified MP4 is missing or its hash changed")
    return {"id": row["source_id"], "label": row["label"], "is_known": row["label"] is not None,
            "signer": row["signer"], "role": row["role"], "path": path.relative_to(ROOT).as_posix(),
            "sha256": clip["mp4_sha256"], "original_sha256": clip["original_sha256"],
            "original_text": row["transcription"], "workbook_row": row["workbook_row"],
            "source_archive": row["archive"], "source_member": clip["source_path"],
            "source_crc32": clip["archive_member_crc32"], "duration_seconds": clip["duration_seconds"]}


def run(download_video):
    files, source = official()
    previous = previously_verified()
    rows = plan_rows(available_rows(files), previous)
    uncached = {row["source_id"] for row in rows if row["source_id"] not in previous}
    transfer_limit = sum(row["range_bytes"] for row in rows if row["source_id"] in uncached)
    if transfer_limit > MEDIA_LIMIT:
        raise ValueError(f"Selected media exceeded the {MEDIA_LIMIT}-byte range budget")
    plan = {"source": source, "workbook_sha256": sha(WORKBOOK), "metadata_sha256": sha(METADATA),
            "selection": "Exact complete Greek word annotations in source media, selected before inference."
                         " Train A/B, validate E, test C/G/F. No gloss segmentation or equivalence is asserted.",
            "media_limit": MEDIA_LIMIT, "selected_range_bytes": transfer_limit, "rows": rows}
    print(json.dumps({"planned": len(rows), "cached": len(rows) - len(uncached), "range_bytes": transfer_limit,
                      "by_role": Counter(row["role"] for row in rows)}, ensure_ascii=False), flush=True)
    if not download_video:
        return

    media = COHORT / "media"
    media.mkdir(parents=True, exist_ok=True)
    plan_path = COHORT / "manifest.planned.json"
    output = COHORT / "manifest.json"
    if plan_path.exists() and load(plan_path) != plan:
        raise ValueError("Frozen cohort acquisition plan changed")
    write_manifest(plan_path, plan)
    plan_sha256 = sha(plan_path)
    pace = Pace()

    def retrieve(row):
        try:
            cached = previous.get(row["source_id"])
            if cached is None:
                member = row["member"]
                path = media / f"{row['source_id']}.flv"
                cached_bytes = path.read_bytes() if path.exists() else b""
                block = None
                if len(cached_bytes) != member["size"] or zlib.crc32(cached_bytes) != member["crc"]:
                    ranges = {"start": member["offset"], "end": member["offset"] + row["range_bytes"]}
                    block = download(ranges, files[row["archive"]], pace)
                cached = acquire(row, files, source, media, block=block)
            return source_metadata(row, cached), None
        except (OSError, ValueError, zlib.error, urllib.error.HTTPError, subprocess.SubprocessError) as error:
            return None, {"id": row["source_id"], "error": str(error)}

    def checkpoint(samples, failures):
        write_manifest(output, {"schema_version": 1, "source": source, "plan_sha256": plan_sha256,
                                "scope": plan["selection"], "complete": len(samples) == len(rows),
                                "labels": VOCABULARY, "samples": samples, "failures": failures})

    samples, failures, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, (sample, error) in enumerate(pool.map(retrieve, rows), 1):
            if error:
                failures.append(error)
            elif sample["original_sha256"] in seen:
                failures.append({"id": sample["id"], "error": "Duplicate original source bytes"})
            else:
                samples.append(sample)
                seen.add(sample["original_sha256"])
            if index % WORKERS == 0 or index == len(rows):
                checkpoint(samples, failures)
                print(json.dumps({"processed": index, "verified": len(samples), "failed": len(failures)}), flush=True)
    if failures:
        raise RuntimeError(f"Acquisition incomplete. Frozen plan retained. See {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="Fetch only the missing frozen source ranges")
    run(parser.parse_args().acquire)
