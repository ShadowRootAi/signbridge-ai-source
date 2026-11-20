"""Freeze genuine chronological appointment exchanges and fetch only their RGB ranges."""

import argparse
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile

from acquire_fresh_continuous import (
    DATA, METADATA, ROOT, SIGNERS, SOURCE, SOURCE_ID, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, read_json, write_manifest,
)
from plan_continuous_composition import read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "appointment-dialogues"
UNSEEN = DATA / "appointment-unseen"
SUPPORT = DATA / "appointment-dialogues-support"
SUPPORT_SCOPE = "Official train-only lexical support outside the appointment evaluation scenes."
SYNTHETIC = ROOT / "reports/pretrained-gsl/synthetic-continuous-train-plan.json"
LIBRARY = ROOT / "web/public/library/catalog.json"
GROUPS = (
    ("health1", "train", (1, 2, 4, 5), 6, ("staff", "patient", "staff", "staff", "patient", "staff")),
    ("health4", "train", (1, 2, 4), 19, ("patient", "staff", "patient", "staff", "patient")),
    ("health2", "val", (3,), 18, ("patient", "staff", "patient", "staff")),
    ("health5", "test", (6, 7), 1, ("patient", "staff", "patient", "staff", "patient", "staff", "patient", "staff", "patient")),
)
UNSEEN_GROUPS = (
    ("health5", "train", (1, 2), 1, ("patient", "staff", "patient", "staff", "patient", "staff", "patient", "staff", "patient")),
    ("health4", "train", (4,), 16, ("staff", "patient", "staff", "patient", "staff")),
    ("health3", "val", (3,), 9, ("staff", "patient", "staff", "patient", "staff", "patient", "staff")),
    ("health1", "test", (6, 7), 0, ("staff", "patient", "staff", "patient", "staff", "patient", "staff",
                                 "patient", "staff", "staff", "patient", "staff", "patient", "staff")),
)
UNSEEN_REUSE = {
    "health1_signer6_rep5_sentences/sentences0007": DATA / "v3/manifest.json",
    "health1_signer7_rep1_sentences/sentences0007": DATA / "open-set-v2/manifest.json",
}


def build_plan(unseen=False):
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    archives = archive_metadata()
    prior = sorted(DATA.glob("continuous-*/manifest.json"))
    if unseen:
        prior.extend(sorted(DATA.glob("appointment-dialogues*/manifest.json")))
        prior.extend([DATA / "manifest.json", DATA / "open-set-v2/manifest.json", DATA / "v3/manifest.json"])
    seen = set()
    for path in prior:
        for row in read_json(path)["samples"]:
            source = row["source_path"]
            seen.add(source if SOURCE_ID.fullmatch(source) else source.split("/", 1)[1])
    seen.update(row["sourceUtterance"] for row in read_json(LIBRARY)["entries"])
    seen.update(row["sourcePath"] for row in read_json(SYNTHETIC)["samples"])
    samples, dialogues = [], []
    groups = UNSEEN_GROUPS if unseen else GROUPS
    for scene, split, signers, first, roles in groups:
        for signer in signers:
            if signer not in SIGNERS[split]:
                raise ValueError("Signer split changed")
            for repetition in range(1, 6):
                dialogue = f"{scene}_signer{signer}_rep{repetition}"
                sources = [f"{dialogue}_sentences/sentences{first + n:04d}" for n in range(len(roles))]
                if all((source not in seen or (unseen and source in UNSEEN_REUSE))
                       and source in canonical and source in timestamps and greek.get(source)
                       for source in sources):
                    break
            else:
                raise ValueError(f"No complete unused official take for {scene}, signer {signer}")
            order = [timestamps[source] for source in sources]
            if any(a["end_ms"] > b["start_ms"] for a, b in zip(order, order[1:])):
                raise ValueError(f"Overlapping or out-of-order source time: {dialogue}")
            if any(a["frame_end_in_concatenated_sentence_sequence"] + 1 != b["frame_start_in_concatenated_sentence_sequence"]
                   for a, b in zip(order, order[1:])):
                raise ValueError(f"Nonconsecutive source sentence frames: {dialogue}")
            archive = scene + ".zip"
            turn_ids = []
            for index, (source, role, timing) in enumerate(zip(sources, roles, order, strict=True)):
                sample = source.replace("/", "__")
                turn_ids.append(sample)
                samples.append({
                    "id": sample, "source_path": scene + "/" + source,
                    "source_archive": archive, "source_url": archives[archive]["links"]["self"],
                    "template": f"{scene}/sentences{first + index:04d}", "signer": signer,
                    "repetition": repetition, "split": split, "dialogue_id": dialogue,
                    "turn_index": index, "role_in_script": role, "text": greek[source],
                    "glosses": list(canonical[source]), "fps": 30,
                    "start_ms": timing["start_ms"], "end_ms": timing["end_ms"],
                    "source_frame_start": timing["frame_start_in_concatenated_sentence_sequence"],
                    "source_frame_end": timing["frame_end_in_concatenated_sentence_sequence"],
                    "annotated_frame_count": timing["implied_inclusive_frame_count"],
                })
                if unseen and source in UNSEEN_REUSE:
                    samples[-1]["previous_isolated_manifest"] = UNSEEN_REUSE[source].relative_to(ROOT).as_posix()
            dialogues.append({"id": dialogue, "scenario": scene, "signer": signer, "repetition": repetition,
                              "split": split, "turn_ids": turn_ids})
    if len({row["id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate recording source")
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "split_signers": SIGNERS,
        "scope": "Original chronological pre-cut utterances from acted CERTH scenarios. "
                 "Each take has one source signer. The roles are inferred from the Greek script.",
        "selection": ("Take the first unused eligible rehearsal by each predetermined signer. "
                      "Hold out all of health1 and health3 from new fitting. The health1 script had earlier partial exposure. "
                      "Retain one previously archived isolated answer in each full test take."
                      if unseen else "Take the first unused eligible rehearsal by each predetermined signer. "
                      "Hold out all of health5 and its test signers. References come from the official annotations."),
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (SPLIT, SUPPLEMENTARY, METADATA, SYNTHETIC, LIBRARY, *prior)},
        "planned_samples": len(samples), "samples": samples, "dialogues": dialogues,
    }


def verified_reuse(plan, budget):
    rows = []
    for row in plan["samples"]:
        if "previous_isolated_manifest" not in row:
            continue
        source = row["source_path"].split("/", 1)[1]
        manifest = ROOT / row["previous_isolated_manifest"]
        original, = [item for item in read_json(manifest)["samples"] if item["source_path"] == source]
        frames = original["frame_paths"]
        hashes = [digest(ROOT / frame) for frame in frames]
        if (len(frames) != row["annotated_frame_count"] or hashes != original["frame_sha256"]
                or sha256("".join(hashes).encode("ascii")).hexdigest() != original["sha256"]):
            raise ValueError(f"Archived isolated source bytes differ: {source}")
        binary = sha256(b"".join(bytes.fromhex(value) for value in hashes)).hexdigest()
        rows.append({**row, "frame_paths": frames, "frame_count": len(frames), "sha256": binary})
    return {"range_budget_bytes": budget, "range_bytes_reserved": 0, "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument("--support", action="store_true", help="Acquire the separate frozen train-only lexical support")
    cohort.add_argument("--unseen", action="store_true", help="Acquire the later whole-encounter holdout")
    args = parser.parse_args()
    output = SUPPORT if args.support else UNSEEN if args.unseen else OUTPUT
    planned, manifest = output / "manifest.planned.json", output / "manifest.json"
    budget = 350_000_000 if args.unseen else 450_000_000
    if args.support:
        original = read_json(OUTPUT / "manifest.json")
        plan = read_json(planned)
        if not original["complete"] or any(row["split"] != "train" for row in plan["samples"]):
            raise ValueError("Lexical support needs a completed original cohort and train-only rows")
        if any(digest(ROOT / path) != expected for path, expected in plan["source_hashes"].items()):
            raise ValueError("Lexical support source evidence changed")
        budget -= original["range_bytes_reserved"]
    else:
        plan = build_plan(args.unseen)
    if planned.is_file() and read_json(planned) != plan:
        raise ValueError("The frozen recordings or source evidence changed")
    output.mkdir(parents=True, exist_ok=True)
    if not planned.is_file():
        write_manifest(planned, plan)
    print(dict(Counter(row["split"] for row in plan["samples"])),
          dict(Counter(row["split"] for row in plan.get("dialogues", []))), flush=True)
    if args.plan_only:
        return
    if manifest.is_file() != args.resume:
        parser.error("Use --resume if and only if acquisition already started")
    if args.support:
        plan = {**plan, "scope": SUPPORT_SCOPE}
    previous = read_json(manifest) if args.resume else verified_reuse(plan, budget) if args.unseen else None
    acquire(plan, output, budget, False, previous)
    current = read_json(manifest)
    if not current["complete"] or any(row["annotated_frame_count"] != row["frame_count"] for row in current["samples"]):
        raise ValueError("The RGB ranges do not match the official sentence frame counts")


if __name__ == "__main__":
    main()
