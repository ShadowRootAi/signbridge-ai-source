"""Acquire the reserved signer-6 CERTH health5 take as an ordered held stream."""

import argparse

from acquire_appointment_daytime import LIBRARY, OUTPUT as FIRST_COHORT, RESERVED_TAKE, SYNTHETIC, inputs, source_id
from acquire_fresh_continuous import (
    DATA, METADATA, ROOT, SOURCE, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, read_json, write_manifest,
)
from plan_continuous_composition import normalize_greek


OUTPUT = DATA / "appointment-daytime-rep5"
TAKE = "health5_signer6_rep5"
EXPECTED_TURNS = 23
BUDGET = 150_000_000


def build_plan():
    manifests, seen, glosses, greek, timings = inputs()
    first_path = FIRST_COHORT / "manifest.json"
    first = read_json(first_path)
    if TAKE != RESERVED_TAKE or first["reserved_future_take"] != TAKE or not first["complete"]:
        raise ValueError("Original acquisition does not mark this complete take as reserved")
    seen.update(source_id(row["source_path"]) for row in first["samples"])
    official = sorted(source for source in glosses if source.startswith(TAKE + "_sentences/"))
    expected = [f"{TAKE}_sentences/sentences{number:04d}" for number in range(EXPECTED_TURNS)]
    if official != expected or set(official) & seen:
        raise ValueError("Reserved physical take has missing official cuts or prior exposure")
    archive = "health5.zip"
    url = archive_metadata()[archive]["links"]["self"]
    rows = []
    for number, source in enumerate(official):
        timing = timings[source]
        text = greek[source]
        if not text.strip() or normalize_greek(text) != normalize_greek(timing["translation"]):
            raise ValueError(f"Greek reference mismatch at {source}")
        row = {
            "id": source.replace("/", "__"), "source_path": f"health5/{source}", "source_archive": archive,
            "source_url": url, "template": f"health5/sentences{number:04d}", "signer": 6, "repetition": 5,
            "split": "test", "dialogue_id": TAKE, "turn_index": number, "text": text, "glosses": list(glosses[source]),
            "fps": 30, "start_ms": timing["start_ms"], "end_ms": timing["end_ms"],
            "source_frame_start": timing["frame_start_in_concatenated_sentence_sequence"],
            "source_frame_end": timing["frame_end_in_concatenated_sentence_sequence"],
            "annotated_frame_count": timing["implied_inclusive_frame_count"],
            "annotation_evidence": {"sentence_file": timing["annotation_file"], "time_file": timing["timestamp_file"],
                                    "line_1based": timing["line_1based"], "original_glosses": timing["raw_gloss"]},
        }
        if rows and (rows[-1]["end_ms"] > row["start_ms"] or rows[-1]["source_frame_end"] + 1 != row["source_frame_start"]):
            raise ValueError(f"Nonchronological author cuts at {source}")
        rows.append(row)
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0",
        "scope": "Previously unused physical CERTH health5 repetition 5 by signer 6. The known script uses one signer "
                 "for both roles. Published cuts are kept in author order and omit source frames between cuts.",
        "split_signers": {"train": [], "val": [], "test": [6]},
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (SPLIT, SUPPLEMENTARY, METADATA, LIBRARY, SYNTHETIC, first_path, *manifests)},
        "planned_samples": len(rows), "samples": rows,
        "dialogues": [{"id": TAKE, "scenario": "health5", "signer": 6, "repetition": 5,
                       "split": "test", "turn_ids": [row["id"] for row in rows]}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    planned = OUTPUT / "manifest.planned.json"
    if planned.exists():
        plan = read_json(planned)
        if any(digest(ROOT / path) != expected for path, expected in plan["source_hashes"].items()):
            raise ValueError("Frozen source evidence changed")
    else:
        plan = build_plan()
        write_manifest(planned, plan)
    print(f"Held take {TAKE}: {plan['planned_samples']} official utterances", flush=True)
    if args.plan_only:
        return
    path = OUTPUT / "manifest.json"
    if path.exists() != args.resume:
        parser.error("Use --resume if and only if acquisition already started")
    acquire(plan, OUTPUT, BUDGET, False, read_json(path) if args.resume else None)
    manifest = read_json(path)
    rows = manifest["samples"]
    if not manifest["complete"] or any(row["frame_count"] != row["annotated_frame_count"] for row in rows):
        raise ValueError("RGB frame counts differ from the official annotations")
    frames = [frame for row in rows for frame in row["frame_paths"]]
    write_manifest(OUTPUT / "chronological-manifest.json", {
        "scope": manifest["scope"], "source_manifest_sha256": digest(path),
        "samples": [{"id": TAKE, "fps": 30, "frame_paths": frames, "frame_count": len(frames)}],
    })
    print(f"Verified {len(frames)} ordered JPEGs", flush=True)


if __name__ == "__main__":
    main()
