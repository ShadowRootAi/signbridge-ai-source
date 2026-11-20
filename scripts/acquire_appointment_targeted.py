"""Acquire independently recorded, officially paired appointment vocabulary and held takes."""

import argparse
from collections import Counter
from zipfile import ZipFile

from acquire_appointment_dialogues import LIBRARY, SYNTHETIC, verified_reuse
from acquire_fresh_continuous import (
    DATA, METADATA, ROOT, SIGNERS, SOURCE, SOURCE_ID, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, read_json, write_manifest,
)
from plan_continuous_composition import read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "appointment-targeted"
INTERNIST = DATA / "appointment-targeted-internist"
BUDGET = 450_000_000
INTERNIST_BUDGET = 150_000_000
REUSED_INTERNIST = "health1_signer7_rep4_sentences/sentences0007"
REUSED_MANIFEST = "data/open-set-v2/manifest.json"
TARGETS = {
    ("health1", 6): ("ΚΡΑΤΗΣΗ", "ΠΑΘΟΛΟΓΟΣ"),
    ("health1", 8): ("ΜΕΘΑΥΡΙΟ",),
    ("health1", 10): ("ΑΙΜΑ", "ΕΞΕΤΑΣΗ"),
    ("health1", 11): ("ΑΙΜΑ", "ΕΞΕΤΑΣΗ"),
    ("health5", 1): ("ΥΠΕΡΗΧΟΣ_ΚΟΙΛΙΑ",),
    ("health5", 6): ("ΔΕΥΤΕΡΑ", "ΠΡΩΙ", "ΑΝΕΤΟ"),
    ("health5", 7): ("ΜΕΣΗΜΕΡΙ",),
    ("health5", 8): ("ΠΡΩΙ", "ΠΙΝΩ", "ΤΡΩΩ"),
}
TAKES = (("val", 3, 4, 10), ("test", 6, 3, 23))


def build_plan(internist=False):
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    archives = archive_metadata()
    prior = sorted({*DATA.glob("continuous-*/manifest.json"), *DATA.glob("appointment-*/manifest.json"),
                    DATA / "manifest.json", DATA / "open-set-v2/manifest.json", DATA / "v3/manifest.json"})
    prior = [path for path in prior if path.parent not in {OUTPUT, INTERNIST}]
    prior = [path for path in prior if not path.parent.name.startswith("appointment-")
             or read_json(path).get("source") == SOURCE]
    if internist:
        prior.append(OUTPUT / "manifest.planned.json")
    used = set()
    for path in prior:
        for row in read_json(path)["samples"]:
            source = row["source_path"]
            used.add(source if SOURCE_ID.fullmatch(source) else source.split("/", 1)[1])
    used.update(row["sourceUtterance"] for row in read_json(LIBRARY)["entries"])
    used.update(row["sourcePath"] for row in read_json(SYNTHETIC)["samples"])

    def sample(scene, signer, repetition, index, split):
        dialogue = f"{scene}_signer{signer}_rep{repetition}"
        name = f"sentences{index:04d}"
        source = f"{dialogue}_sentences/{name}"
        terms = TARGETS.get((scene, index), ())
        reuse = internist and source == REUSED_INTERNIST
        if ((source in used and not reuse) or not greek.get(source) or source not in timestamps
                or source not in canonical or not set(terms) <= set(canonical[source])):
            return None
        archive = scene + ".zip"
        timing = timestamps[source]
        row = {
            "id": source.replace("/", "__"), "source_path": f"{scene}/{source}",
            "source_archive": archive, "source_url": archives[archive]["links"]["self"],
            "template": f"{scene}/{name}", "signer": signer, "repetition": repetition,
            "split": split, "text": greek[source], "glosses": list(canonical[source]), "fps": 30,
            "start_ms": timing["start_ms"], "end_ms": timing["end_ms"],
            "source_frame_start": timing["frame_start_in_concatenated_sentence_sequence"],
            "source_frame_end": timing["frame_end_in_concatenated_sentence_sequence"],
            "annotated_frame_count": timing["implied_inclusive_frame_count"], "target_glosses": list(terms),
        }
        if reuse:
            row["previous_isolated_manifest"] = REUSED_MANIFEST
        return row

    samples = []
    for split in (() if internist else ("train", "val")):
        for scene, index in TARGETS:
            if split == "val" and scene == "health5":
                continue
            for signer in SIGNERS[split]:
                for repetition in range(1, 6):
                    if row := sample(scene, signer, repetition, index, split):
                        samples.append(row)
                        break
                else:
                    raise ValueError(f"No eligible paired source {scene}/{index} for signer {signer}")
    dialogues = []
    selected = (("test", 7, 4, 14),) if internist else TAKES
    scene = "health1" if internist else "health5"
    for split, signer, repetition, length in selected:
        dialogue = f"{scene}_signer{signer}_rep{repetition}"
        rows = [sample(scene, signer, repetition, index, split) for index in range(length)]
        if any(row is None for row in rows):
            raise ValueError(f"Missing or previously used source turn in {dialogue}")
        if any(a["end_ms"] > b["start_ms"] or a["source_frame_end"] + 1 != b["source_frame_start"]
               for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Source turns are not consecutive: {dialogue}")
        for index, row in enumerate(rows):
            staff = index in {0, 2, 4, 6, 8, 9, 11, 13} if internist else index % 2 == 0
            row.update(dialogue_id=dialogue, turn_index=index, role_in_script="staff" if staff else "patient")
        samples.extend(rows)
        dialogues.append({"id": dialogue, "scenario": scene, "signer": signer, "repetition": repetition,
                          "split": split, "turn_ids": [row["id"] for row in rows]})
    if len({row["id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate physical source")
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0",
        "scope": ("Official internist dialogue, same familiar script and a different complete physical take. "
                  "Thirteen source cuts are unused and one «βεβαίως» was earlier evaluated in isolation."
                  if internist else "Official acted GSL utterances and Greek references. Training, development and held recordings and "
                  "signers are disjoint. Both appointment scripts are familiar at the template level. "
                  "One signer enacts both scripted roles in each take."),
        "split_signers": {"train": [], "val": [], "test": [7]} if internist else {"train": SIGNERS["train"], "val": [3], "test": [6]},
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (SPLIT, SUPPLEMENTARY, METADATA, SYNTHETIC, LIBRARY, *prior)},
        "planned_samples": len(samples), "samples": samples, "dialogues": dialogues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--internist", action="store_true", help="Acquire the separate secondary held internist take")
    args = parser.parse_args()
    output = INTERNIST if args.internist else OUTPUT
    budget = INTERNIST_BUDGET if args.internist else BUDGET
    output.mkdir(parents=True, exist_ok=True)
    planned = output / "manifest.planned.json"
    manifest = output / "manifest.json"
    if planned.is_file():
        plan = read_json(planned)
        if any(digest(ROOT / name) != expected for name, expected in plan["source_hashes"].items()):
            raise ValueError("The frozen targeted source evidence changed")
    else:
        plan = build_plan(args.internist)
        write_manifest(planned, plan)
    print(dict(Counter(row["split"] for row in plan["samples"])), flush=True)
    if args.plan_only:
        return
    if manifest.is_file() != args.resume:
        parser.error("Use --resume if and only if acquisition already started")
    previous = read_json(manifest) if args.resume else verified_reuse(plan, budget) if args.internist else None
    acquire(plan, output, budget, False, previous)
    current = read_json(manifest)
    if not current["complete"] or any(row["annotated_frame_count"] != row["frame_count"] for row in current["samples"]):
        raise ValueError("The RGB frames differ from official utterance boundaries")


if __name__ == "__main__":
    main()
