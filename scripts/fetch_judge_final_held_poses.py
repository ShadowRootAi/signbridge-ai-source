"""Select signer-disjoint CERTH native replay poses for the new recognition head."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re
from zipfile import ZipFile

import numpy as np

from acquire_fresh_continuous import ROOT, digest, read_json, write_manifest
from dataset_fetch import RemoteZipFile
from fetch_openhands_gsl_poses import ARCHIVE_BYTES, METADATA_SHA256, ROOT_CACHE, URL, MemberView, decode_pose


METADATA = ROOT_CACHE / "gsl_metadata.zip"
OUTPUT = ROOT_CACHE / "judge-final-held-poses.npz"
REPORT = ROOT / "reports/judge-final/held-native.json"
PLAN = REPORT.with_name("held-native.planned.json")
POLICY = ROOT / "publication-local-assistant/dist/__signbridge/openhands/policy.json"
TRANSFER_LIMIT = 110_000_000
COUNT_PER_BUCKET = 100
SESSION = re.compile(r"([a-z]+\d+_signer([67])_rep\d+)_(?:sentences|glosses)")


def stable(text):
    return sha256(("signbridge-new-head-held-v1:" + text).encode()).digest()


def excluded_sessions():
    blocked, checked = set(), []
    directories = (ROOT / "data", ROOT / ".runtime-tmp/recognition-rgb-audit", ROOT / ".runtime-tmp/teach-me-rgb-audit")
    for folder in directories:
        for path in sorted(folder.rglob("manifest.json")):
            document = read_json(path)
            used = False
            for key in ("samples", "rows", "clips"):
                for item in document.get(key, []):
                    if not isinstance(item, dict):
                        continue
                    for field in ("id", "source_path", "sourcePath", "source_id", "sourceUtterance"):
                        value = item.get(field)
                        if isinstance(value, str) and (match := SESSION.search(value)):
                            blocked.add(match[1])
                            used = True
            if used:
                checked.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    return blocked, checked


def choose_balanced(groups, count):
    labels = sorted(groups, key=stable)
    ordered = {label: sorted(group, key=lambda item: stable(item["id"])) for label, group in groups.items()}
    selected = []
    for repetition in range(max(map(len, ordered.values()))):
        for label in labels:
            rows = ordered[label]
            if repetition < len(rows):
                selected.append(rows[repetition])
            if len(selected) == count:
                return selected
    return selected


def make_plan():
    if digest(METADATA) != METADATA_SHA256:
        raise ValueError("Native metadata does not match the verified archive")
    active = set(read_json(POLICY)["supportedLabels"])
    if len(active) != 200:
        raise ValueError("Expected the audited active 200-sign roster")
    blocked, exposure = excluded_sessions()
    with ZipFile(METADATA) as archive:
        split_member, = [name for name in archive.namelist() if name.endswith("/train_greek_iso.csv")]
        source = archive.read(split_member).decode("utf-8-sig").splitlines()
    groups = defaultdict(lambda: defaultdict(list))
    for line in source:
        identity, gloss = line.split("|")
        signer = SESSION.match(identity)
        if signer and signer[1] not in blocked:
            actor, known = int(signer[2]), gloss in active
            groups[actor, known][gloss].append({"id": identity, "gloss": gloss, "signer": actor, "is_known": known})
    if set(groups) != {(6, True), (6, False), (7, True), (7, False)}:
        raise ValueError("Missing known or unknown sample groups for actors 6 and 7")
    counts = {key: min(COUNT_PER_BUCKET, sum(map(len, group.values()))) for key, group in groups.items()}
    if sum(counts.values()) < 350:
        raise ValueError(f"Not enough non-overlapping native pose inputs for balanced replay: {counts}")
    selected = [row for actor in (6, 7) for known in (True, False)
                for row in choose_balanced(groups[actor, known], counts[actor, known])]
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Native annotation IDs duplicated")
    return {"schema_version": 1, "source": "https://zenodo.org/records/6674324", "source_url": URL,
            "source_split": "OpenHands published training split", "metadata_sha256": METADATA_SHA256,
            "policy_sha256": digest(POLICY), "transfer_limit": TRANSFER_LIMIT,
            "scope": "Signer 6 and 7 recordings outside the new head's training and calibration. Both signers were "
                     "in upstream OpenHands training. Native archived 75-joint poses, without raw browser footage. "
                     "Local manifest source sessions identified in the exposure audit were excluded. Selected before scoring.",
            "local_exposure_sources": exposure, "excluded_local_rgb_source_sessions": sorted(blocked),
            "samples": selected}


def fetch(plan):
    if OUTPUT.exists():
        raise ValueError(f"Refusing to overwrite {OUTPUT}")
    remote = RemoteZipFile(URL, ARCHIVE_BYTES)
    with ZipFile(remote) as outer:
        nested = MemberView(remote, outer.getinfo("GSL/GSL_pose.zip"))
    results = {}
    with ZipFile(nested) as poses:
        sessions = defaultdict(list)
        for row in plan["samples"]:
            identity = row["id"]
            info = poses.getinfo(f"GSL_pose/{identity}.pkl")
            if not 0 < info.file_size < 4_000_000:
                raise ValueError(f"Invalid pose member size for {identity}")
            sessions[identity.split("/", 1)[0]].append((identity, info))
        members = sorted(poses.infolist(), key=lambda member: member.header_offset)
        ends = {member.filename: following.header_offset for member, following in zip(members, members[1:])}
        ends[members[-1].filename] = poses.start_dir
        blocks = sorted((min(info.header_offset for _, info in group),
                         max(ends[info.filename] for _, info in group), group) for group in sessions.values())
        total = remote.bytes_received + sum(end - start for start, end, _ in blocks)
        if total > TRANSFER_LIMIT:
            raise ValueError(f"Native pose range plan would transfer {total} bytes")
        print(json.dumps({"groups": len(blocks), "planned_transfer_bytes": total}), flush=True)
        for start, end, group in blocks:
            if remote.bytes_received + end - start > TRANSFER_LIMIT:
                raise ValueError("Native pose transfer cap reached")
            nested.seek(start)
            previous = remote.cache.copy()
            block = nested.read(end - start)
            if len(block) != end - start:
                raise ValueError("Incomplete native pose range")
            remote.cache = [(nested.start + start, block), *previous]
            for identity, info in group:
                results[identity] = decode_pose(poses.read(info))
            remote.cache = previous
            print(json.dumps({"verified_poses": len(results), "received_bytes": remote.bytes_received}), flush=True)
    rows = plan["samples"]
    sequences = [results[row["id"]] for row in rows]
    temporary = OUTPUT.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, keypoints=np.concatenate(sequences), split=np.array("judge-final-held"),
                            offsets=np.cumsum([0, *(len(sequence) for sequence in sequences)]),
                            ids=np.array([row["id"] for row in rows]), glosses=np.array([row["gloss"] for row in rows]))
    temporary.replace(OUTPUT)
    write_manifest(REPORT, {**plan, "complete": True, "plan_sha256": digest(PLAN),
        "npz_path": OUTPUT.relative_to(ROOT).as_posix(), "npz_sha256": digest(OUTPUT),
        "requests": remote.requests, "received_bytes": remote.bytes_received})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="Fetch source ranges after the plan is frozen")
    args = parser.parse_args()
    plan = make_plan()
    counts = Counter(f"{row['signer']}_{'known' if row['is_known'] else 'other'}" for row in plan["samples"])
    print(json.dumps({"clips": len(plan["samples"]), "by_role": counts,
                      "distinct_glosses": len({row['gloss'] for row in plan['samples']}),
                      "excluded_sessions": len(plan["excluded_local_rgb_source_sessions"])}), flush=True)
    if args.acquire:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        if PLAN.exists() and read_json(PLAN) != plan:
            raise ValueError("Frozen native pose selection differs")
        if not PLAN.exists():
            write_manifest(PLAN, plan)
        fetch(plan)


if __name__ == "__main__":
    main()
