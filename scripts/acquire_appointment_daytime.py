"""Acquire verified CERTH appointment clips and a new ordered held recording."""

import argparse
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile

from acquire_appointment_dialogues import LIBRARY, SYNTHETIC
from acquire_fresh_continuous import (
    DATA, METADATA, ROOT, SOURCE, SOURCE_ID, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, read_json, write_manifest,
)
from plan_continuous_composition import normalize_greek, read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "appointment-daytime"
DEVELOPMENT = DATA / "appointment-targeted/manifest.json"
BUDGET = 450_000_000
TRAIN_SIGNERS = (1, 2, 4, 5)
TEST_TAKE = "health5_signer6_rep4"
DEV_TAKE = "health5_signer3_rep4"
RESERVED_TAKE = "health5_signer6_rep5"
TARGETS = {
    ("health5", 1): ("θέλω ραντεβού για υπέρηχο στη κοιλιά", "ΕΓΩ(1) ΘΕΛΩ ΡΑΝΤΕΒΟΥ ΓΙΑ ΥΠΕΡΗΧΟΣ_ΚΟΙΛΙΑ", 14),
    ("health5", 6): ("σας βολεύει τη δευτέρα το πρωί", "ΡΑΝΤΕΒΟΥ ΔΕΥΤΕΡΑ ΠΡΩΙ ΕΣΥ ΑΝΕΤΟ", 12),
    ("health5", 7): ("προτιμώ το μεσημέρι", "ΕΓΩ(1) ΘΕΛΩ ΜΕΣΗΜΕΡΙ", 12),
    ("health1", 8): ("υπάρχει γιατρός μεθαύριο", "ΓΙΑΤΡΟΣ(2) ΜΕΘΑΥΡΙΟ ΕΧΩ", 10),
    ("health1", 6): ("πρώτα σας κλείνω ραντεβού με παθολόγο", "ΠΡΩΤΟΝ ΕΣΥ ΡΑΝΤΕΒΟΥ ΚΡΑΤΗΣΗ ΓΙΑ ΓΙΑΤΡΟΣ(2) ΠΑΘΟΛΟΓΟΣ", 10),
}


def source_id(path):
    for name in (path, path.split("/", 1)[-1]):
        if SOURCE_ID.fullmatch(name):
            return name
    return None


def inputs():
    manifests = [DATA / "manifest.json", *sorted(DATA.glob("*/manifest.json"))]
    manifests = [path for path in manifests if path.parent != OUTPUT]
    seen = set()
    for path in manifests:
        for row in read_json(path).get("samples", ()):
            if item := source_id(row.get("source_path", "")):
                seen.add(item)
    for path, key, field in ((LIBRARY, "entries", "sourceUtterance"), (SYNTHETIC, "samples", "sourcePath")):
        seen.update(item for row in read_json(path)[key] if (item := source_id(row[field])))
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    return manifests, seen, canonical, greek, timestamps


def build_plan():
    manifests, seen, canonical, greek, times = inputs()
    archives = archive_metadata()

    def sample(source, role):
        match = SOURCE_ID.fullmatch(source)
        timing = times[source]
        if normalize_greek(timing["translation"]) != normalize_greek(greek[source]):
            raise ValueError(f"Greek reference differs between official sources: {source}")
        scene = match["scenario"]
        archive = scene + ".zip"
        return {
            "id": source.replace("/", "__"), "source_path": f"{scene}/{source}",
            "source_archive": archive, "source_url": archives[archive]["links"]["self"],
            "template": f"{scene}/{match['sentence']}", "signer": int(match["signer"]),
            "repetition": int(match["rep"]), "split": role, "text": greek[source],
            "glosses": list(canonical[source]), "fps": 30, "start_ms": timing["start_ms"],
            "end_ms": timing["end_ms"], "source_frame_start": timing["frame_start_in_concatenated_sentence_sequence"],
            "source_frame_end": timing["frame_end_in_concatenated_sentence_sequence"],
            "annotated_frame_count": timing["implied_inclusive_frame_count"],
            "annotation_evidence": {"sentence_file": timing["annotation_file"], "time_file": timing["timestamp_file"],
                                    "line_1based": timing["line_1based"], "original_glosses": timing["raw_gloss"]},
        }

    rows = []
    for (scene, index), (text, glosses, expected) in TARGETS.items():
        group = []
        for signer in TRAIN_SIGNERS:
            for repetition in range(1, 6):
                source = f"{scene}_signer{signer}_rep{repetition}_sentences/sentences{index:04d}"
                if (source not in seen and source in times and canonical.get(source) == tuple(glosses.split())
                        and normalize_greek(greek.get(source, "")) == normalize_greek(text)):
                    group.append(sample(source, "train"))
        if len(group) != expected or {item["signer"] for item in group} != set(TRAIN_SIGNERS):
            raise ValueError(f"Fresh official matches changed for {scene}/{index}: {len(group)}")
        rows.extend(group)

    dialogues = []
    for take, role, count in ((DEV_TAKE, "val", 10), (TEST_TAKE, "test", 23)):
        sources = [f"{take}_sentences/sentences{i:04d}" for i in range(count)]
        if role == "test" and (any(source in seen for source in sources) or
                               sum(source.startswith(take + "_sentences/") for source in canonical) != count):
            raise ValueError("Held physical take overlaps prior use or is missing official turns")
        group = [sample(source, role) for source in sources]
        for index, row in enumerate(group):
            row.update(dialogue_id=take, turn_index=index)
        if any(a["end_ms"] > b["start_ms"] or a["source_frame_end"] + 1 != b["source_frame_start"]
               for a, b in zip(group, group[1:])):
            raise ValueError(f"Source rows are not chronological: {take}")
        rows.extend(group)
        dialogues.append({"id": take, "scenario": "health5", "signer": group[0]["signer"], "split": role,
                          "turn_ids": [row["id"] for row in group]})
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0",
        "scope": "Published chronological CERTH utterance cuts. The signer plays both roles. "
                 "The source release omits between-turn frames; the complete cut sequence is a reconstruction.",
        "selection": "Exact Greek and ordered canonical gloss matches for the five target clauses. "
                     "Signer 3 development cuts are reused, all 23 signer 6 repetition 4 test cuts are unused. "
                     "The evaluation script is familiar at the template level.",
        "split_signers": {"train": list(TRAIN_SIGNERS), "val": [3], "test": [6]},
        "reserved_future_take": RESERVED_TAKE,
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (SPLIT, SUPPLEMENTARY, METADATA, LIBRARY, SYNTHETIC, *manifests)},
        "planned_samples": len(rows), "samples": rows, "dialogues": dialogues,
    }


def reuse_development(plan):
    old = {row["id"]: row for row in read_json(DEVELOPMENT)["samples"] if row.get("dialogue_id") == DEV_TAKE}
    rows = []
    for row in plan["samples"]:
        if row["split"] != "val":
            continue
        source = old[row["id"]]
        if source["source_path"] != row["source_path"] or source["glosses"] != row["glosses"]:
            raise ValueError("Reused development identity changed")
        paths = source["frame_paths"]
        observed = sha256(b"".join(bytes.fromhex(digest(ROOT / path)) for path in paths)).hexdigest()
        if len(paths) != row["annotated_frame_count"] or observed != source["sha256"]:
            raise ValueError(f"Reused development bytes changed: {row['id']}")
        rows.append({**row, "frame_paths": paths, "frame_count": len(paths), "sha256": observed})
    return {"range_budget_bytes": BUDGET, "range_bytes_reserved": 0, "samples": rows}


def chronological(manifest):
    by_id = {item["id"]: item for item in manifest["samples"]}
    records = []
    for take in manifest["dialogues"]:
        rows = [by_id[identity] for identity in take["turn_ids"]]
        frames = [path for row in rows for path in row["frame_paths"]]
        records.append({"id": take["id"], "fps": 30, "frame_paths": frames, "frame_count": len(frames)})
    write_manifest(OUTPUT / "chronological-manifest.json", {
        "scope": manifest["scope"], "source_manifest_sha256": digest(OUTPUT / "manifest.json"), "samples": records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    planned = OUTPUT / "manifest.planned.json"
    if planned.exists():
        plan = read_json(planned)
        if any(digest(ROOT / name) != expected for name, expected in plan["source_hashes"].items()):
            raise ValueError("Source evidence changed after the plan was frozen")
    else:
        plan = build_plan()
        write_manifest(planned, plan)
    print(dict(Counter(row["split"] for row in plan["samples"])), flush=True)
    if args.plan_only:
        return
    path = OUTPUT / "manifest.json"
    if path.exists() != args.resume:
        parser.error("Use --resume if and only if acquisition already started")
    previous = read_json(path) if args.resume else reuse_development(plan)
    acquire(plan, OUTPUT, BUDGET, False, previous)
    current = read_json(path)
    if not current["complete"] or any(row["annotated_frame_count"] != row["frame_count"] for row in current["samples"]):
        raise ValueError("RGB frame counts do not match author timestamp annotations")
    chronological(current)


if __name__ == "__main__":
    main()
