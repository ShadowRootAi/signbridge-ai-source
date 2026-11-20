"""Freeze a metadata-only, one-text CERTH composition probe; never fetch RGB.

Run from the project root with:
  .venv/Scripts/python.exe -B -X utf8 scripts/prepare_continuous_scale_composition.py

The ten source IDs were selected from annotations before baseline test inference.
This script validates/fixes those IDs, rather than selecting new examples after
observing predictions. An existing frozen plan is verified, never overwritten.
"""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from acquire_continuous_scale import overlaps, prior_sources
from acquire_fresh_continuous import (
    DATA, DIRECTORY_CACHE, METADATA, ROOT, SOURCE, SOURCE_ID, SPLIT,
    SUPPLEMENTARY, archive_metadata, digest, read_json, write_manifest,
)
from acquire_appointment_daytime import LIBRARY, SYNTHETIC, source_id
from plan_continuous_composition import normalize_greek, read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "continuous-scale-composition"
MAIN_PLAN = DATA / "continuous-scale/manifest.planned.json"
TARGET_TEXT = "έχω ταυτότητα όχι φωτοτυπία"
SOURCES = (
    "kep1_signer6_rep1_sentences/sentences0009",
    "kep1_signer6_rep2_sentences/sentences0009",
    "kep1_signer6_rep3_sentences/sentences0010",
    "kep1_signer6_rep4_sentences/sentences0009",
    "kep1_signer6_rep5_sentences/sentences0009",
    "kep1_signer7_rep2_sentences/sentences0009",
    "kep1_signer7_rep3_sentences/sentences0009",
    "kep1_signer7_rep4_sentences/sentences0010",
    "kep1_signer7_rep5_sentences/sentences0010",
    "kep1_signer7_rep6_sentences/sentences0010",
)


def build_plan():
    official, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    used, prior_groups, _, manifests = prior_sources(official, greek)
    primary = read_json(MAIN_PLAN)
    primary_rows = primary["samples"]
    training = [row for row in primary_rows if row["split"] == "train"]
    vocabulary = {gloss for row in training for gloss in row["glosses"]}
    if len(training) != 520 or len(vocabulary) != 101:
        raise ValueError("The frozen primary training cohort no longer has 520 clips / 101 glosses")
    primary_ids = {source_id(row["source_path"]) for row in primary_rows}
    primary_groups = {(row["template"], tuple(row["glosses"]), tuple(normalize_greek(row["text"]).split()))
                      for row in primary_rows}
    blocked = prior_groups | primary_groups
    metadata = archive_metadata()["kep1.zip"]
    rows = []
    for source in SOURCES:
        match = SOURCE_ID.fullmatch(source)
        if match is None or source not in official or source not in timestamps:
            raise ValueError(f"Missing official source or timestamps: {source}")
        if source in used or source in primary_ids:
            raise ValueError(f"The secondary source was already used or reserved: {source}")
        glosses, text, timing = official[source], greek.get(source, ""), timestamps[source]
        template = f"{match['scenario']}/{match['sentence']}"
        key = (template, glosses, tuple(normalize_greek(text).split()))
        if (normalize_greek(text) != normalize_greek(TARGET_TEXT)
                or normalize_greek(text) != normalize_greek(timing["translation"])
                or not set(glosses) <= vocabulary or any(overlaps(key, old) for old in blocked)):
            raise ValueError(f"Reference, glossary or cross-cohort one-edit exclusion failed: {source}")
        frame_count = timing["implied_inclusive_frame_count"]
        if (not isinstance(frame_count, int) or frame_count < 1
                or timing["end_ms"] <= timing["start_ms"]
                or timing["frame_end_in_concatenated_sentence_sequence"]
                - timing["frame_start_in_concatenated_sentence_sequence"] + 1 != frame_count):
            raise ValueError(f"Invalid official temporal/frame annotation: {source}")
        prefix = source.split("/", 1)[0]
        rows.append({
            "id": source.replace("/", "__"), "source_path": "kep1/" + source,
            "source_archive": "kep1.zip", "source_url": metadata["links"]["self"],
            "signer": int(match["signer"]), "repetition": int(match["rep"]), "split": "test",
            "text": text, "glosses": list(glosses), "template": template, "fps": 30,
            "take_id": prefix.removesuffix("_sentences"),
            "start_ms": timing["start_ms"], "end_ms": timing["end_ms"],
            "source_frame_start": timing["frame_start_in_concatenated_sentence_sequence"],
            "source_frame_end": timing["frame_end_in_concatenated_sentence_sequence"],
            "annotated_frame_count": frame_count,
            "annotation_evidence": {"file": timing["annotation_file"], "timestamp_file": timing["timestamp_file"],
                                    "line_1based": timing["line_1based"], "original_glosses": timing["raw_gloss"]},
            "composition": "unseen_template", "category": "unseen_template_known_glosses",
            "tranche1_unseen_glosses": [], "full_train_unseen_glosses": [],
            "prior_observed_other_sources_in_take": sum(item.startswith(prefix + "/") for item in used),
            "primary_plan_other_sources_in_take": sum(item.startswith(prefix + "/") for item in primary_ids),
        })
    first_vocabulary = {gloss for row in training[:160] for gloss in row["glosses"]}
    if (any(not set(row["glosses"]) <= first_vocabulary for row in rows)
            or Counter(row["signer"] for row in rows) != {6: 5, 7: 5}
            or len({row["id"] for row in rows}) != 10
            or len({normalize_greek(row["text"]) for row in rows}) != 1):
        raise ValueError("The fixed ten-take, one-text probe contract changed")
    cache = DIRECTORY_CACHE / f"kep1.zip-{metadata['checksum'].removeprefix('md5:')}"
    blocks = sorted(cache.glob("*.bin"))
    evidence = {
        "status": "cached_blocks_present_but_members_not_verified" if blocks else "no_standard_cached_index_blocks",
        "directory": cache.relative_to(ROOT).as_posix(), "block_count": len(blocks),
        "block_bytes": sum(path.stat().st_size for path in blocks),
        "archive_bytes": metadata["size"], "archive_checksum": metadata["checksum"],
        "network_requests": 0,
        "limitation": "No archive-member listing, JPEG CRC, frame bytes or decoded images were verified by this metadata-only step.",
    }
    paths = (MAIN_PLAN, SPLIT, SUPPLEMENTARY, METADATA, LIBRARY, SYNTHETIC,
             Path(__file__).resolve(), *manifests)
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "complete": False,
        "split_signers": {"train": [], "val": [], "test": [6, 7]},
        "scope": "Secondary one-text composition probe: ten new source utterance clips, five each from corpus signers 6 and 7. "
                 "This is one unique Greek reference, not ten independent sentence templates or a dialogue benchmark. "
                 "Other utterances from the same recording takes can have been observed previously.",
        "selection_context": "These exact source IDs were selected using annotations and the frozen training glossary before "
                             "baseline test inference. This metadata plan was materialized after the first baseline report. "
                             "No probe video, features or predictions have been inspected by this preparation script.",
        "held_policy": "No source ID, template, exact or one-edit Greek/gloss reference overlaps the primary plan or prior "
                       "observed references. All gold glosses occur in both the first160 and full520 continuous training clips. "
                       "The original primary train/development/test splits remain unchanged.",
        "planned_samples": 10, "unique_reference_texts": 1,
        "annotation_template_ids": sorted({row["template"] for row in rows}),
        "expected_annotated_frames": sum(row["annotated_frame_count"] for row in rows),
        "main_plan_sha256": digest(MAIN_PLAN), "continuous_training_gloss_count": len(vocabulary),
        "proposed_future_range_budget_bytes": 180_000_000,
        "rgb_acquisition_started": False, "archive_index_evidence": evidence,
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        "samples": rows,
    }


def main():
    target = OUTPUT / "manifest.planned.json"
    if target.exists():
        plan = read_json(target)
        if any(digest(ROOT / path) != expected for path, expected in plan["source_hashes"].items()):
            raise ValueError("A source bound by the frozen secondary plan changed; no rewrite permitted")
        if tuple(row["source_path"].split("/", 1)[1] for row in plan["samples"]) != SOURCES:
            raise ValueError("The frozen secondary source IDs differ")
    else:
        plan = build_plan()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_manifest(target, plan)
    print(f"Frozen metadata-only plan: {target.relative_to(ROOT)}", flush=True)
    print(f"SHA256 {digest(target)}", flush=True)
    print(f"10 clips / 1 reference / {plan['expected_annotated_frames']} annotated frames; RGB NOT acquired", flush=True)
    print(f"Index status: {plan['archive_index_evidence']['status']}", flush=True)


if __name__ == "__main__":
    main()
