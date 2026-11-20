"""Plan and acquire signer- and utterance-disjoint supervised CERTH RGB cohorts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
import re
import shutil
import subprocess
import time
from zipfile import ZipFile

from acquire_appointment_daytime import LIBRARY, SYNTHETIC, source_id
from acquire_fresh_continuous import (
    BoundedZip, DATA, DIRECTORY_CACHE, FRAME_NAME, MAX_CLIP_BYTES, METADATA, ROOT, SOURCE, SOURCE_ID,
    SPLIT, SUPPLEMENTARY, TransferBudget, acquire, archive_metadata, digest, read_json, write_manifest,
)
from plan_continuous_composition import normalize_greek, read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "continuous-scale"
TRANCHE = DATA / "continuous-scale-tranche1"
BUDGET = 4_400_000_000
TRAIN, FIRST_TRAIN, DEVELOPMENT_GROUPS, HELD_GROUPS = 520, 160, 12, 14
SIGNERS = {"train": (1, 2, 4, 5), "val": (3,), "test": (6, 7)}
CACHE_BLOCK = re.compile(r"(\d+)-(\d+)-([a-f0-9]{64})\.bin")


def order(value):
    return sha256(f"continuous-scale-v1/{value}".encode()).hexdigest()


def near(a, b):
    if len(a) == len(b):
        return sum(left != right for left, right in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    if len(b) != len(a) + 1:
        return False
    first = next((i for i, (left, right) in enumerate(zip(a, b)) if left != right), len(a))
    return a[first:] == b[first + 1:]


def overlaps(a, b):
    return a[0] == b[0] or near(a[1], b[1]) or near(a[2], b[2])


def prior_sources(canonical, greek, *, excluded=()):
    manifests = [DATA / "manifest.json", *sorted(DATA.glob("*/manifest.json"))]
    manifests = [path for path in manifests if path.parent not in (OUTPUT, TRANCHE, *excluded)]
    used, held = set(), set()
    for manifest in manifests:
        for row in read_json(manifest).get("samples", ()):
            source = source_id(row.get("source_path", ""))
            if source not in canonical:
                continue
            used.add(source)
            if row.get("split") in ("val", "test", "held", "development", "final_test"):
                held.add(source)
    for path, rows, field in ((LIBRARY, "entries", "sourceUtterance"), (SYNTHETIC, "samples", "sourcePath")):
        used.update(source for row in read_json(path)[rows] if (source := source_id(row[field])) in canonical)

    def key(source):
        match = SOURCE_ID.fullmatch(source)
        return (f"{match['scenario']}/{match['sentence']}", canonical[source],
                tuple(normalize_greek(greek.get(source, "")).split()))

    return used, {key(source) for source in used}, {key(source) for source in held}, manifests


def cached_sizes(archive_name, archive, candidates):
    checksum = archive["checksum"].removeprefix("md5:")
    folder = DIRECTORY_CACHE / f"{archive_name}-{checksum}"
    if not folder.is_dir():
        return {}
    remote = BoundedZip(archive, TransferBudget(0))
    for path in sorted(folder.glob("*.bin")):
        match = CACHE_BLOCK.fullmatch(path.name)
        if match is None:
            continue
        offset, length, expected = match.groups()
        data = path.read_bytes()
        if len(data) == int(length) and sha256(data).hexdigest() == expected:
            remote.cache.append((int(offset), data))
    frames = defaultdict(list)
    with ZipFile(remote) as index:
        for member in index.infolist():
            source, _, filename = member.filename.rpartition("/")
            match = FRAME_NAME.fullmatch(filename)
            if source in candidates and match is not None:
                frames[source].append((int(match[1]), member.header_offset,
                                       min(remote.size, member.header_offset + member.compress_size + 1024)))
    result = {}
    for source, members in frames.items():
        numbers = sorted(row[0] for row in members)
        size = max(row[2] for row in members) - min(row[1] for row in members)
        if len(numbers) == candidates[source] and numbers == list(range(numbers[0], numbers[0] + len(numbers))) and size <= MAX_CLIP_BYTES:
            result[source] = size
    return result


def inventory():
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    used, used_groups, held_groups, manifests = prior_sources(canonical, greek)
    metadata = archive_metadata()
    candidates = defaultdict(dict)
    for source, glosses in canonical.items():
        match = SOURCE_ID.fullmatch(source)
        if source in used or source not in timestamps or len(glosses) < 2 or match is None:
            continue
        timing = timestamps[source]
        text = greek.get(source, "")
        normalized = normalize_greek(text)
        words = tuple(normalized.split())
        if not 3 <= len(words) <= 9 or normalized != normalize_greek(timing["translation"]):
            continue
        scene, actor = match["scenario"], int(match["signer"])
        archive = scene + ".zip"
        template = f"{scene}/{match['sentence']}"
        key = template, glosses, words
        role = next(role for role, signers in SIGNERS.items() if actor in signers)
        if any(overlaps(key, old) for old in (held_groups if role == "train" else used_groups)):
            continue
        candidates[archive][source] = (key, {
            "id": source.replace("/", "__"), "source_path": f"{scene}/{source}", "source_archive": archive,
            "source_url": metadata[archive]["links"]["self"], "signer": actor, "repetition": int(match["rep"]),
            "split": role, "text": text, "glosses": list(glosses), "template": template, "fps": 30,
            "take_id": source.split("/", 1)[0].removesuffix("_sentences"),
            "start_ms": timing["start_ms"], "end_ms": timing["end_ms"],
            "annotated_frame_count": timing["implied_inclusive_frame_count"],
            "annotation_evidence": {"file": timing["annotation_file"], "timestamp_file": timing["timestamp_file"],
                                    "line_1based": timing["line_1based"]},
        })
    groups = defaultdict(lambda: defaultdict(list))
    for name, rows in sorted(candidates.items()):
        sizes = cached_sizes(name, metadata[name], {source: row["annotated_frame_count"] for source, (_, row) in rows.items()})
        for source, size in sizes.items():
            key, row = rows[source]
            groups[key][row["signer"]].append({**row, "preflight_rgb_bytes": size})
    return groups, manifests, len(used), len(held_groups)


def choose_held(groups):
    test, development = [], []
    lexical = {token for key, rows in groups.items() if any(actor in rows for actor in SIGNERS["train"]) for token in key[1]}
    for role, selection, target, repetitions in (("test", test, HELD_GROUPS, 2), ("val", development, DEVELOPMENT_GROUPS, 3)):
        scenes = Counter()
        available = {key for key, rows in groups.items() if all(len(rows[actor]) >= repetitions for actor in SIGNERS[role])}
        while len(selection) < target:
            allowed = [key for key in available if not any(overlaps(key, other) for other in (*test, *development))]
            if not allowed:
                raise ValueError(f"Only {len(selection)} disjoint {role} templates available")
            key = min(allowed, key=lambda key: (sum(token not in lexical for token in key[1]),
                                                not 4 <= len(key[2]) <= 8, scenes[key[0].split('/')[0]], order(key)))
            selection.append(key)
            scenes[key[0].split('/')[0]] += 1
            available.remove(key)
    return development, test


def choose_train(groups, held):
    eligible = {key: [row for actor in SIGNERS["train"] for row in rows[actor]] for key, rows in groups.items()
                if not any(overlaps(key, other) for other in held)}
    eligible = {key: rows for key, rows in eligible.items() if rows}
    missing = {token for key in held for token in key[1]}
    actors = Counter()
    ordered = []
    while missing:
        options = [key for key in eligible if key not in ordered and missing.intersection(key[1])]
        if not options:
            break
        key = min(options, key=lambda key: (-len(missing.intersection(key[1])), not 4 <= len(key[2]) <= 8, order(key)))
        ordered.append(key)
        missing.difference_update(key[1])
    ordered.extend(sorted(set(eligible) - set(ordered), key=lambda key: (not 4 <= len(key[2]) <= 8, order(key))))
    result = []
    for desired_length in (True, False):
        for _ in range(20):
            for key in ordered:
                rows = eligible[key]
                if not rows or (4 <= len(key[2]) <= 8) != desired_length:
                    continue
                row = min(rows, key=lambda row: (actors[row["signer"]], order(row["id"])))
                actors[row["signer"]] += 1
                rows.remove(row)
                result.append(row)
                if len(result) == TRAIN:
                    return result
    raise ValueError(f"Only {len(result)} eligible signer-disjoint train recordings")


def plan():
    groups, manifests, observed, prior_held = inventory()
    dev, test = choose_held(groups)
    training = choose_train(groups, [*dev, *test])
    first_glosses = {token for row in training[:FIRST_TRAIN] for token in row["glosses"]}
    all_glosses = {token for row in training for token in row["glosses"]}
    held = []
    for role, keys, repeats in (("val", dev, 3), ("test", test, 2)):
        for key in keys:
            for actor in SIGNERS[role]:
                for row in sorted(groups[key][actor], key=lambda row: order(row["id"]))[:repeats]:
                    missing_first = sorted(set(row["glosses"]) - first_glosses)
                    held.append({**row, "composition": "unseen_template", "category": "unseen_template_unseen_glosses"
                                 if missing_first else "unseen_template_known_glosses", "tranche1_unseen_glosses": missing_first,
                                 "full_train_unseen_glosses": sorted(set(row["glosses"]) - all_glosses)})
    training = [{**row, "composition": "known_template", "category": "train"} for row in training]
    samples = [*held, *training]
    if len({row["id"] for row in samples}) != len(samples) or sum(row["preflight_rgb_bytes"] for row in samples) > BUDGET - 30_000_000:
        raise ValueError("Plan duplicates source IDs or exceeds the capped preflight")
    return {"schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "split_signers": SIGNERS,
            "scope": "Authentic pre-cut CERTH RGB utterances. Signers, template IDs, normalized Greek and ordered glosses "
                     "are split by role. One-edit near reference matches are excluded across roles.",
            "held_policy": "Unused IDs and exact/one-edit Greek or gloss references excluded from prior observations. "
                           "Reserved held roles fixed before any RGB or training. Category is based on first-tranche train glossary.",
            "planned_samples": len(samples), "tranche1_train_count": FIRST_TRAIN,
            "prior_source_count": observed, "prior_held_or_dev_reference_count": prior_held,
            "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                              for path in (SPLIT, SUPPLEMENTARY, METADATA, LIBRARY, SYNTHETIC, *manifests)},
            "samples": samples}


def acquire_stage(output, selection, previous=None, *, max_workers=4):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    saved = read_json(path) if path.exists() else previous
    if saved is not None and saved.get("complete") and saved["planned_samples"] == selection["planned_samples"]:
        return saved
    acquire(selection, output, BUDGET, False, saved, max_workers=max_workers)
    complete = read_json(path)
    if not complete["complete"] or any(row["frame_count"] != row["annotated_frame_count"] for row in complete["samples"]):
        raise ValueError("Committed RGB rows differ from their verified author frame counts")
    return complete


def verify_features(path, samples):
    sources = {row["id"]: row for row in samples}
    seen, extractors = set(), set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identifier = row["id"]
            source = sources[identifier]
            source_key = sha256(json.dumps([source["fps"], source["frame_paths"]],
                                          ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if (identifier in seen or row["source_key"] != source_key or row["fps"] != 15
                    or len(row["features"]) != (source["frame_count"] + 1) // 2
                    or any(len(frame) != 224 or not all(map(isfinite, frame)) for frame in row["features"])):
                raise ValueError(f"Invalid browser features for {identifier}")
            seen.add(identifier)
            extractors.add(row["extractor_sha256"])
    if len(extractors) > 1:
        raise ValueError("Features contain multiple extractor versions")
    return seen


def promote(source, target):
    for attempt in range(4):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.3 * (attempt + 1))


def extract_stage(output, url, previous=None):
    plan = read_json(output / "manifest.planned.json")
    committed = output / "manifest.json"
    partial, final = output / "features.partial.jsonl", output / "features.jsonl"
    fit = output / "features.fit.jsonl"
    snapshot = output / "manifest.extracting.json"
    expected_fit = {row["id"] for row in plan["samples"] if row["split"] != "test"}
    if final.exists():
        existing = read_json(committed)
        if existing["complete"] and len(verify_features(final, existing["samples"])) == len(plan["samples"]):
            return
        if partial.exists():
            raise ValueError(f"Two partial feature files in {output}")
        promote(final, partial)
    if previous is not None and not partial.exists():
        shutil.copyfile(previous / "features.jsonl", partial)
    done, last_progress, failures = set(), time.monotonic(), 0
    while True:
        if committed.exists():
            current = read_json(committed)
            rows = current["samples"]
            if not done:
                done = verify_features(partial, rows)
            if expected_fit <= done and not fit.exists():
                temporary = fit.with_suffix(".tmp")
                with partial.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
                    for line in source:
                        if json.loads(line)["id"] in expected_fit:
                            target.write(line)
                if verify_features(temporary, rows) != expected_fit:
                    raise ValueError("Fitting features incomplete")
                promote(temporary, fit)
                print(f"Fitting features ready: {fit.relative_to(ROOT)} ({len(expected_fit)} clips)", flush=True)
            if current["complete"] and len(done) == len(rows):
                promote(partial, final)
                print(f"All browser features verified: {final.relative_to(ROOT)} ({len(done)} clips)", flush=True)
                return
            fitting = [row for row in rows if row["split"] != "test" or row["id"] in done]
            ready = rows if expected_fit <= done else fitting
            count = len(ready) - len(done)
            if count >= 12 or current["complete"] and count:
                ordered = sorted(ready, key=lambda row: {"val": 0, "train": 1, "test": 2}[row["split"]])
                write_manifest(snapshot, {"samples": ordered})
                result = subprocess.run(["node", "web/scripts/extract-continuous.mjs", "--url", url,
                                         "--manifest", str(snapshot), "--output", str(partial), "--parallel", "4"], cwd=ROOT)
                previous_count = len(done)
                done = verify_features(partial, rows)
                if len(done) > previous_count:
                    last_progress, failures = time.monotonic(), 0
                elif result.returncode:
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError("Browser extraction failed three times without committing a row")
                    time.sleep(4)
                continue
        if time.monotonic() - last_progress > 600:
            raise TimeoutError(f"No new committed data for browser extraction in {output}")
        time.sleep(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "tranche1", "finish", "all", "extract"), default="all", nargs="?")
    parser.add_argument("--url", default="http://127.0.0.1:5208")
    parser.add_argument("--download-workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.phase == "extract":
        extract_stage(TRANCHE, args.url)
        extract_stage(OUTPUT, args.url, previous=TRANCHE)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TRANCHE.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "manifest.planned.json"
    if target.exists():
        selection = read_json(target)
        if any(digest(ROOT / path) != value for path, value in selection["source_hashes"].items()):
            raise ValueError("Prior source evidence changed after freeze")
    else:
        selection = plan()
        write_manifest(target, selection)
    first = [row for row in selection["samples"] if row["split"] != "train"]
    first.extend([row for row in selection["samples"] if row["split"] == "train"][:FIRST_TRAIN])
    tranche = {**selection, "samples": first, "planned_samples": len(first), "full_plan_sha256": digest(target)}
    tranche_path = TRANCHE / "manifest.planned.json"
    if not tranche_path.exists():
        write_manifest(tranche_path, tranche)
    elif read_json(tranche_path) != tranche:
        raise ValueError("The first tranche differs from the frozen full plan")
    for name, rows in (("tranche1", first), ("full", selection["samples"])):
        print(name, dict(Counter(row["split"] for row in rows)), "JPEG bytes", sum(row["preflight_rgb_bytes"] for row in rows), flush=True)
    if args.phase == "plan":
        return
    if args.phase == "finish" and not (TRANCHE / "manifest.json").exists():
        raise ValueError("Complete the first tranche before continuation")
    snapshot = acquire_stage(TRANCHE, tranche, max_workers=args.download_workers)
    if args.phase != "tranche1":
        acquire_stage(OUTPUT, selection, snapshot, max_workers=args.download_workers)


if __name__ == "__main__":
    main()
