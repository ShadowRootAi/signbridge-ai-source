"""Acquire a fixed, signer-disjoint CERTH continuous RGB cohort by ZIP ranges."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
from threading import Lock
import time
import urllib.request
from zipfile import ZipFile

from continuous_dataset import DATA, FRAME_NAME, MAX_CLIP_BYTES, ROOT, verify_rgb, write_manifest
from dataset_fetch import RemoteZipFile
from plan_continuous_composition import normalize_greek, read_annotations


PLAN = ROOT / "reports/continuous-composition-plan.json"
ANNOTATIONS = DATA / "original_annotations"
SPLIT = ANNOTATIONS / "GSL_split.zip"
SUPPLEMENTARY = ANNOTATIONS / "supplementary.zip"
METADATA = ANNOTATIONS / "zenodo_metadata.json"
OUTPUT = DATA / "continuous-fresh"
SOURCE = "https://zenodo.org/records/4756317"
SIGNERS = {"train": [1, 2, 4, 5], "val": [3], "test": [6, 7]}
SOURCE_ID = re.compile(r"(?P<scenario>[a-z]+\d+)_signer(?P<signer>\d+)_rep(?P<rep>\d+)_sentences/(?P<sentence>sentences\d+)")
PREVIOUS = [DATA / name / "manifest.json" for name in ("continuous-v1-expanded", "continuous-v1-holdout", "continuous-composition-pilot")]
COMPACT_HOLDOUTS = {"police3/sentences0006", "kep3/sentences0012", "kep2/sentences0019", "health3/sentences0015"}
DIRECTORY_CACHE = ROOT / ".runtime-tmp/gsl-zip-directory-cache"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def archive_metadata():
    source = read_json(METADATA)
    if (source["id"] != 4756317 or source["metadata"]["access_right"] != "open"
            or source["metadata"]["license"]["id"] != "cc-by-4.0"):
        raise ValueError("Unexpected Zenodo record or rights")
    return {file["key"]: file for file in source["files"]}


def matching_sources(template, official_glosses, official_greek, seen_ids, archives):
    rows = defaultdict(list)
    scenario, sentence = template["template_id"].split("/")
    glosses = tuple(template["glosses"])
    archive = template["source_archive"]
    if archive != scenario + ".zip" or archive not in archives:
        raise ValueError(f"Unexpected archive for {template['template_id']}")
    for source, labels in official_glosses.items():
        if source in seen_ids or labels != glosses or not source.startswith(scenario + "_signer"):
            continue
        match = SOURCE_ID.fullmatch(source)
        text = official_greek.get(source, "")
        if match["sentence"] != sentence or normalize_greek(text) != template["normalized_greek"]:
            continue
        signer = int(match["signer"])
        rows[signer].append({
            "id": source.replace("/", "__"), "source_path": scenario + "/" + source,
            "signer": signer, "repetition": int(match["rep"]), "glosses": list(glosses),
            "text": text, "fps": 30, "template": template["template_id"],
            "source_archive": archive, "source_url": archives[archive]["links"]["self"],
        })
    for signer in rows:
        rows[signer].sort(key=lambda row: (row["repetition"], row["id"]))
    return rows


def select_plan(compact):
    official_glosses, official_greek = read_annotations(SPLIT, SUPPLEMENTARY)
    previous = [row for path in PREVIOUS for row in read_json(path)["samples"]]
    seen_ids = {row["source_path"].split("/", 1)[1] for row in previous}
    seen_ids.update(row["sourceUtterance"] for row in read_json(ROOT / "web/public/library/catalog.json")["entries"])
    old_templates = {row["template"] for row in previous}
    old_holdouts = {tuple(row["glosses"]) for row in previous if row.get("composition") == "unseen_template"}
    roles = read_json(PLAN)["roles"]
    archives = archive_metadata()

    def matched_sources(template):
        return matching_sources(template, official_glosses, official_greek, seen_ids, archives)

    novel = [(split, template, matched_sources(template))
             for split, role in (("val", "development"), ("test", "test")) for template in roles[role]
             if not compact or template["template_id"] in COMPACT_HOLDOUTS]
    held_templates = {template["template_id"] for _, template, _ in novel}
    held_glosses = {tuple(template["glosses"]) for _, template, _ in novel} | old_holdouts
    held_text = {template["normalized_greek"] for _, template, _ in novel}
    open_archives = {template["source_archive"] for _, template, _ in novel}
    if old_templates & held_templates:
        raise ValueError("Novel template already appeared in the cached continuous cohort")

    eligible = {}
    for template in roles["train"]:
        identity = template["template_id"]
        if (identity in old_templates or identity in held_templates or tuple(template["glosses"]) in held_glosses
                or template["normalized_greek"] in held_text):
            continue
        rows = matched_sources(template)
        if all(rows.get(signer) for signers in SIGNERS.values() for signer in signers):
            eligible[identity] = (template, rows)

    # Cover every held gloss in a different scenario before adding diversity.
    missing = {(template["source_archive"], gloss) for _, template, _ in novel for gloss in template["glosses"]}
    chosen = []
    while missing or len(chosen) < (8 if compact else 15):
        covered_glosses = {gloss for template, _ in chosen for gloss in template["glosses"]}

        def priority(identity):
            template, _ = eligible[identity]
            archive, glosses = template["source_archive"], set(template["glosses"])
            supported = sum(held_archive != archive and gloss in glosses for held_archive, gloss in missing)
            return -supported, archive not in open_archives, -len(glosses - covered_glosses), identity

        identity = min(eligible, key=priority)
        if missing and priority(identity)[0] == 0:
            raise ValueError(f"Novel gloss lacks verified cross-scenario training support: {sorted(missing)}")
        template, rows = eligible.pop(identity)
        chosen.append((template, rows))
        archive = template["source_archive"]
        open_archives.add(archive)
        missing.difference_update({(held_archive, gloss) for held_archive, gloss in missing
                                   if held_archive != archive and gloss in template["glosses"]})
    if len(chosen) > 16:
        raise ValueError("Training template support exceeds the bounded clip cohort")

    samples = []
    repetitions = 2 if compact else 3
    for split, template, rows in novel:
        for signer in SIGNERS[split]:
            if len(rows[signer]) < repetitions:
                raise ValueError(f"Fewer than {repetitions} verified holdout repeats for {template['template_id']} signer {signer}")
            samples.extend({**row, "split": split, "composition": "unseen_template"} for row in rows[signer][:repetitions])
    for index, (_, rows) in enumerate(chosen):
        samples.extend({**rows[signer][0], "split": "train", "composition": "known_template"} for signer in SIGNERS["train"])
        if index < (4 if compact else 5):
            for split in ("val", "test"):
                samples.extend({**rows[signer][0], "split": split, "composition": "known_template"} for signer in SIGNERS[split])

    if len({row["id"] for row in samples}) != len(samples):
        raise ValueError("Source clips overlap across roles")
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "split_signers": SIGNERS,
        "scope": "New RGB IDs from CERTH studio recordings. Within-cohort people and novel templates are split. This corpus has previously been used for other experiments.",
        "selection": f"Frozen composition template roles; exact published reference matches; one row per train signer/template; {repetitions} earliest repetitions per novel signer/template before transport preflight; matched known-template evaluation checks.",
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in (PLAN, SPLIT, SUPPLEMENTARY, METADATA, *PREVIOUS)},
        "planned_samples": len(samples), "samples": samples,
    }


class TransferBudget:
    def __init__(self, limit):
        self.limit = limit
        self.reserved = 0
        self.lock = Lock()

    def claim(self, amount):
        with self.lock:
            if self.reserved + amount > self.limit:
                raise ValueError("Cumulative HTTP range budget reached")
            self.reserved += amount


class BoundedZip(RemoteZipFile):
    def __init__(self, metadata, budget):
        super().__init__(metadata["links"]["self"], metadata["size"])
        self.budget = budget

    def read(self, count=-1):
        start = self.position
        end = self.size if count < 0 else min(self.size, start + count)
        length = end - start
        if length <= 0:
            return b""
        if length > MAX_CLIP_BYTES:
            raise ValueError(f"Range of {length} bytes exceeds per-request cap")
        for offset, block in self.cache:
            if offset <= start and offset + len(block) >= end:
                self.position = end
                return block[start - offset:end - offset]

        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"})
        for attempt in range(3):
            self.budget.claim(length)
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{self.size}":
                        raise ValueError("Server did not honor exact ZIP byte range")
                    block = response.read(length)
                    if len(block) != length:
                        raise ValueError("Incomplete ZIP byte range")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        self.requests += 1
        self.bytes_received += length
        self.position = end
        if length > 100_000 or start >= self.size - 100_000:
            self.cache.append((start, block))
        return block


def open_archive(name, samples, metadata, budget):
    remote = BoundedZip(metadata[name], budget)
    checksum = metadata[name]["checksum"].removeprefix("md5:")
    directory = DIRECTORY_CACHE / f"{name}-{checksum}"
    for path in sorted(directory.glob("*.bin")):
        match = re.fullmatch(r"(\d+)-(\d+)-([0-9a-f]{64})\.bin", path.name)
        if not match:
            continue
        start, length = (int(value) for value in match.groups()[:2])
        if not 0 < length <= MAX_CLIP_BYTES or start + length > remote.size or path.stat().st_size != length:
            continue
        data = path.read_bytes()
        if sha256(data).hexdigest() == match[3]:
            remote.cache.append((start, data))
    archive = ZipFile(remote)
    directory.mkdir(parents=True, exist_ok=True)
    for start, block in remote.cache:
        path = directory / f"{start}-{len(block)}-{sha256(block).hexdigest()}.bin"
        if not path.exists():
            path.write_bytes(block)
    wanted = {row["source_path"].split("/", 1)[1]: row["id"] for row in samples}
    by_id = defaultdict(list)
    for member in archive.infolist():
        parent, _, basename = member.filename.rpartition("/")
        if parent in wanted and FRAME_NAME.fullmatch(basename):
            by_id[wanted[parent]].append(member)

    indexes = {}
    for row in samples:
        members = sorted(by_id[row["id"]], key=lambda member: member.filename)
        numbers = [int(FRAME_NAME.fullmatch(member.filename.rsplit("/", 1)[1])[1]) for member in members]
        if not numbers or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"Missing or nonconsecutive RGB frames: {row['id']}")
        start = min(member.header_offset for member in members)
        end = min(remote.size, max(member.header_offset + member.compress_size + 1024 for member in members))
        if end - start > MAX_CLIP_BYTES:
            raise ValueError(f"Clip range exceeds per-request cap: {row['id']}")
        indexes[row["id"]] = (start, end, members)
    print(f"Indexed {name}: {len(samples)} clips, {remote.bytes_received:,} metadata bytes", flush=True)
    return remote, archive, indexes


def acquire(plan, output, limit, allow_compact_trim, previous=None, *, max_workers=4):
    budget = TransferBudget(limit)
    completed, lock = {}, Lock()
    if previous is not None:
        if previous["range_budget_bytes"] != limit or previous["range_bytes_reserved"] > limit:
            raise ValueError("Resume must use the original cumulative transfer budget")
        budget.claim(previous["range_bytes_reserved"])
        wanted = {row["id"]: row for row in plan["samples"]}
        for row in previous["samples"]:
            source = wanted[row["id"]]
            if (any(row[key] != value for key, value in source.items()) or not row["frame_paths"]
                    or len(row["frame_paths"]) != row["frame_count"]
                    or not all((ROOT / path).is_file() for path in row["frame_paths"])):
                raise ValueError(f"Resume source clip changed: {row['id']}")
            completed[row["id"]] = row
    by_archive = defaultdict(list)
    for row in plan["samples"]:
        if row["id"] not in completed:
            by_archive[row["source_archive"]].append(row)
    metadata = archive_metadata()

    def checkpoint():
        rows = [completed[row["id"]] for row in plan["samples"] if row["id"] in completed]
        write_manifest(output / "manifest.json", {
            **plan, "samples": rows, "complete": len(rows) == plan["planned_samples"],
            "range_budget_bytes": limit, "range_bytes_reserved": budget.reserved,
        })

    def acquire_archive(name, indexes):
        remote, archive, by_id = indexes
        with archive:
            directory_cache = list(remote.cache)
            for row in by_archive[name]:
                start, end, members = by_id[row["id"]]
                remote.seek(start)
                block = remote.read(end - start)
                remote.cache = [(start, block), *directory_cache]
                folder = output / "rgb" / row["id"]
                folder.mkdir(parents=True, exist_ok=True)
                clip_hash = sha256()
                paths = []
                for member in members:
                    data = archive.read(member)
                    verify_rgb(data, member)
                    target = folder / member.filename.rsplit("/", 1)[1]
                    # A clip only becomes visible to readers after the entire
                    # frame list is committed in the manifest. Avoid per-frame
                    # rename failures in cloud-synced Windows workspaces.
                    if target.exists():
                        verify_rgb(target.read_bytes(), member)
                    else:
                        for attempt in range(3):
                            try:
                                target.write_bytes(data)
                                break
                            except PermissionError:
                                if attempt == 2:
                                    raise
                                time.sleep(0.25 * (attempt + 1))
                    clip_hash.update(sha256(data).digest())
                    paths.append(target.relative_to(ROOT).as_posix())
                remote.cache = list(directory_cache)
                clip_digest = clip_hash.hexdigest()
                with lock:
                    if any(old["sha256"] == clip_digest for old in completed.values()):
                        raise ValueError(f"Duplicate RGB sequence: {row['id']}")
                    completed[row["id"]] = {**row, "frame_paths": paths, "frame_count": len(paths), "sha256": clip_digest}
                    checkpoint()
                    print(f"{len(completed)}/{plan['planned_samples']} {row['split']} {row['id']} ({len(paths)} frames)", flush=True)

    checkpoint()
    with ThreadPoolExecutor(max_workers=max_workers) as workers:
        pending = {name: workers.submit(open_archive, name, rows, metadata, budget) for name, rows in by_archive.items()}
        indexed = {name: future.result() for name, future in pending.items()}
        sizes = {key: end - start for _, _, indexes in indexed.values() for key, (start, end, _) in indexes.items()}
        forecast = budget.reserved + sum(sizes.values())
        if allow_compact_trim and forecast > limit - 5_000_000:
            trimmed = set()
            for template in reversed(plan.get("budget_optional_train_templates", [])):
                if forecast <= limit - 5_000_000:
                    break
                for row in plan["samples"]:
                    if row["split"] == "train" and row["template"] == template:
                        trimmed.add(row["id"])
                        forecast -= sizes[row["id"]]

            # Keep the first novel take of every template and signer.
            first = {}
            for row in plan["samples"]:
                if row["composition"] == "unseen_template":
                    first.setdefault((row["template"], row["signer"]), row["id"])
            repeated = [row for row in plan["samples"] if row["composition"] == "unseen_template"
                        and first[row["template"], row["signer"]] != row["id"]]
            for row in sorted(repeated, key=lambda row: (-sizes[row["id"]], row["id"])):
                if forecast <= limit - 5_000_000:
                    break
                trimmed.add(row["id"])
                forecast -= sizes[row["id"]]

            minimum = plan.get("minimum_train_signers_per_template", 4)
            training = [row for row in plan["samples"] if row["split"] == "train" and row["id"] not in trimmed]
            per_signer = Counter(row["signer"] for row in training)
            per_template = Counter(row["template"] for row in training)
            while forecast > limit - 5_000_000:
                eligible = [row for row in training if row["id"] not in trimmed and per_template[row["template"]] > minimum
                            and per_signer[row["signer"]] > 1]
                if not eligible:
                    break
                row = min(eligible, key=lambda row: (-per_signer[row["signer"]], -sizes[row["id"]], row["id"]))
                trimmed.add(row["id"])
                forecast -= sizes[row["id"]]
                per_signer[row["signer"]] -= 1
                per_template[row["template"]] -= 1
            plan["samples"] = [row for row in plan["samples"] if row["id"] not in trimmed]
            plan["planned_samples"] = len(plan["samples"])
            plan["transport_trimmed_ids"] = sorted(trimmed)
            for name in by_archive:
                by_archive[name] = [row for row in by_archive[name] if row["id"] not in trimmed]
            write_manifest(output / "manifest.planned.json", plan)
        checkpoint()
        print(f"Preflight range forecast: {forecast:,} of {limit:,} bytes", flush=True)
        if forecast > limit:
            raise ValueError("Preflight forecast exceeds cumulative range budget")
        transfers = [workers.submit(acquire_archive, name, indexes) for name, indexes in indexed.items()]
        try:
            for transfer in transfers:
                transfer.result()
        finally:
            workers.shutdown(wait=True)
            checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-transfer-mb", type=int, default=350)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_transfer_mb <= 350:
        parser.error("Choose a bounded transfer budget of at most 350 MB")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must stay inside the project")
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    if args.resume:
        if args.plan_only:
            parser.error("Cannot combine --resume with --plan-only")
        plan = read_json(output / "manifest.planned.json")
        previous = read_json(output / "manifest.json")
    else:
        if not args.plan_only and (output / "manifest.json").exists():
            parser.error("Acquisition output already exists; use --resume or a separate directory")
        plan = select_plan(args.compact)
        write_manifest(output / "manifest.planned.json", plan)
    counts = Counter((row["split"], row["composition"]) for row in plan["samples"])
    print(f"Planned {plan['planned_samples']} clips: {dict(counts)}", flush=True)
    if not args.plan_only:
        acquire(plan, output, args.max_transfer_mb * 1_000_000, args.compact and not args.resume, previous)


if __name__ == "__main__":
    main()
