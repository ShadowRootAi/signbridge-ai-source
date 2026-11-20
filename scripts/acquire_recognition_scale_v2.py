"""Acquire a second signer-disjoint Elementary23 cohort with independent negatives."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zlib

from acquire_elementary_coalesced import Pace, download
from acquire_elementary_library_expansion import METADATA, WORKBOOK, acquire, load, official, sha
from acquire_fresh_continuous import write_manifest
from acquire_recognition_scale_pilot import (
    COHORT as PILOT, ROOT, VALIDATION_SIGNERS, VOCABULARY, available_rows,
    previously_verified, source_metadata, word_key,
)


COHORT = ROOT / ".runtime-tmp/recognition-scale-v2"
WORDS = (*tuple(word for word in VOCABULARY if word != "θάλασσα"), "λέξεις", "υλικά", "περιεχόμενα")
TRAIN_SIGNERS = ("A", "B")
TEST_SIGNERS = ("C", "G", "F", "I")
BYTE_LIMIT = 35 * 1024**2
WORKERS = 6


def content(row):
    return row["member"]["size"], row["member"]["crc"]


def inventory(rows, first_plan, first_run, local):
    targets = {word_key(label): label for label in WORDS}
    blocked_keys = {word_key(row["phrase"]) for row in first_plan if row["label"] is None}
    blocked_keys.update(word_key(label) for label in (*VOCABULARY, *WORDS))
    prior = {row["source_id"] for row in first_plan}
    prior_content = {content(row) for row in first_plan}
    groups = defaultdict(list)
    for row in rows:
        key = word_key(row["phrase"])
        if row["source_id"] not in prior and content(row) not in prior_content:
            if key not in targets or targets[key] == row["phrase"]:
                groups[key, row["signer"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["source_id"] not in local, row["range_bytes"], row["source_id"]))

    selected, taken_ids, taken_content = [], set(), set()

    def add(row, role, label):
        if row["source_id"] in taken_ids or content(row) in taken_content:
            return False
        selected.append({**row, "role": role, "label": label})
        taken_ids.add(row["source_id"])
        taken_content.add(content(row))
        return True

    def take(group, role, label, limit, unique_keys=False):
        count = 0
        for row in group:
            text = word_key(row["phrase"])
            if unique_keys and text in blocked_keys:
                continue
            if add(row, role, label):
                if unique_keys:
                    blocked_keys.add(text)
                count += 1
                if count == limit:
                    break
        return count

    for row in first_plan:
        if (row["label"] in WORDS and row["role"] in ("train", "validation")
                and row["source_id"] in first_run):
            if not add(row, row["role"], row["label"]):
                raise ValueError("Unexpected duplicate in the previously verified cohort")

    for key, label in targets.items():
        for signer in (*TRAIN_SIGNERS, *VALIDATION_SIGNERS):
            role = "train" if signer in TRAIN_SIGNERS else "validation"
            take(groups[key, signer], role, label, 2)

    def unknowns(role, signer, count):
        candidates = [row for (key, actor), options in groups.items() if actor == signer and key not in blocked_keys
                      for row in options]
        candidates.sort(key=lambda row: (row["source_id"] not in local,
            sha256(f"signbridge-scale-v2:{role}:{row['source_id']}".encode()).digest()))
        if take(candidates, role, None, count, unique_keys=True) != count:
            raise ValueError(f"Insufficient independent negatives for {role} {signer}")

    for signer in TRAIN_SIGNERS:
        unknowns("train", signer, 7)
    unknowns("validation", VALIDATION_SIGNERS[0], 12)

    held_available = {}
    for key, label in targets.items():
        by_signer = {}
        for signer in TEST_SIGNERS:
            group = groups[key, signer]
            if group:
                by_signer[signer] = len({content(row) for row in group})
                take(group, "test", label, len(group))
        held_available[label] = by_signer

    for signer, count in zip(TEST_SIGNERS, (6, 6, 3, 1), strict=True):
        unknowns("test", signer, count)

    prior_only = [row for row in first_run.values() if row["role"] == "test" and row["label"] in WORDS]
    if len({row["label"] for row in selected if row["role"] == "test" and row["label"]}) < 2:
        raise ValueError("No usable fresh positives for a strict novel test")
    return selected, held_available, prior_only


def pilot_provenance(first_run):
    # Convert earlier normalized rows to the acquisition helper's representation.
    return {item["id"]: {"source_path": item["source_member"], "signer_id": item["signer"],
                         "workbook_row": item["workbook_row"], "workbook_transcription": item["original_text"],
                         "archive_member_crc32": item["source_crc32"], "path": item["path"],
                         "mp4_sha256": item["sha256"], "original_sha256": item["original_sha256"],
                         "duration_seconds": item["duration_seconds"]} for item in first_run.values()}


def run(perform):
    files, source = official()
    pilot_plan_path, pilot_run_path = PILOT / "manifest.planned.json", PILOT / "manifest.json"
    pilot_plan = load(pilot_plan_path)
    pilot = load(pilot_run_path)
    if not pilot["complete"] or pilot["plan_sha256"] != sha(pilot_plan_path):
        raise ValueError("The earlier frozen cohort must be complete")
    first_run = {row["id"]: row for row in pilot["samples"]}
    local = {**previously_verified(), **pilot_provenance(first_run)}
    rows, held_available, diagnostic = inventory(available_rows(files), pilot_plan["rows"], first_run, local)
    transfer = sum(row["range_bytes"] for row in rows if row["source_id"] not in local)
    if transfer > BYTE_LIMIT:
        raise ValueError(f"Cohort exceeds {BYTE_LIMIT}-byte transfer limit")
    scope = ("Ten whole-clip Greek word annotations. New test positives only for words with independent recordings "
             "outside the earlier pilot. Earlier held clips are isolated as retrospective diagnostics. Negative text "
             "groups are disjoint across roles. Train A/B, validation E, test C/F/G/I.")
    plan = {"scope": scope, "source": source, "workbook_sha256": sha(WORKBOOK), "metadata_sha256": sha(METADATA),
            "v1_manifest_sha256": sha(pilot_run_path), "transfer_limit": BYTE_LIMIT, "range_bytes": transfer,
            "fresh_held_by_label_signers": held_available, "rows": rows}
    breakdown = {f"{role}_{'known' if is_known else 'other'}": count for (role, is_known), count in
                 Counter((row["role"], row["label"] is not None) for row in rows).items()}
    print(json.dumps({"planned": len(rows), "range_bytes": transfer, "by_role": breakdown,
                      "fresh_held_by_label_signers": held_available, "prior_diagnostic": len(diagnostic)},
                     ensure_ascii=False), flush=True)
    if not perform:
        return

    media = COHORT / "media"
    media.mkdir(parents=True, exist_ok=True)
    plan_path = COHORT / "manifest.planned.json"
    if plan_path.exists() and load(plan_path) != plan:
        raise ValueError("The frozen V2 plan changed")
    write_manifest(plan_path, plan)
    plan_sha256 = sha(plan_path)
    pace = Pace()

    def retrieve(row):
        try:
            clip = local.get(row["source_id"])
            if clip is None:
                member = row["member"]
                path = media / f"{row['source_id']}.flv"
                payload = path.read_bytes() if path.exists() else b""
                block = None
                if (len(payload), zlib.crc32(payload)) != content(row):
                    start = member["offset"]
                    block = download({"start": start, "end": start + row["range_bytes"]}, files[row["archive"]], pace)
                clip = acquire(row, files, source, media, block=block)
            return source_metadata(row, clip), None
        except (OSError, ValueError, zlib.error, subprocess.SubprocessError) as error:
            return None, {"id": row["source_id"], "error": str(error)}

    def checkpoint(samples, failures):
        write_manifest(COHORT / "manifest.json", {"schema_version": 1, "source": source, "scope": scope,
            "plan_sha256": plan_sha256, "v1_manifest_sha256": plan["v1_manifest_sha256"],
            "complete": len(samples) == len(rows), "labels": WORDS, "samples": samples, "failures": failures,
            "fresh_held_by_label_signers": held_available, "previously_used_diagnostic": diagnostic})

    clips, failures, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, (clip, failure) in enumerate(pool.map(retrieve, rows), 1):
            if failure:
                failures.append(failure)
            elif clip["original_sha256"] in seen:
                failures.append({"id": clip["id"], "error": "Duplicate original bytes"})
            else:
                seen.add(clip["original_sha256"])
                clips.append(clip)
            if index % WORKERS == 0 or index == len(rows):
                checkpoint(clips, failures)
                print(json.dumps({"processed": index, "verified": len(clips), "failed": len(failures)}), flush=True)
    if failures:
        raise RuntimeError(f"Acquisition incomplete. Frozen plan retained in {COHORT}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="Fetch missing ranges for the frozen V2 plan")
    run(parser.parse_args().acquire)
