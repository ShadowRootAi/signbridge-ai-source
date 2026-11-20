"""Freeze and acquire a posthoc CERTH short-phrase probe by bounded ZIP ranges."""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from acquire_appointment_daytime import LIBRARY, SYNTHETIC, source_id
from acquire_continuous_scale import OUTPUT as SCALE, TRANCHE, cached_sizes, overlaps, prior_sources
from acquire_fresh_continuous import (
    DATA, METADATA, ROOT, SOURCE, SOURCE_ID, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, read_json, write_manifest,
)
from plan_continuous_composition import normalize_greek, read_annotations
from v3_data_diagnosis import annotation_timestamps


OUTPUT = DATA / "continuous-scale-novel"
ARCHIVE = "kep3.zip"
SIGNERS = (6, 7)
SOURCES = tuple(
    f"kep3_signer{signer}_rep{repetition}_sentences/sentences{sentence:04d}"
    for signer in SIGNERS
    for repetition in range(1, 6)
    for sentence in (2, 4)
)
PROTECTED_ROLES = {"train", "val", "development", "dev", "calibration"}
HELD_ROLES = {"val", "test", "held", "development", "final_test"}
RETRY_ALLOWANCE = 12_000_000


def annotation_key(source, canonical, greek):
    match = SOURCE_ID.fullmatch(source)
    return (
        f"{match['scenario']}/{match['sentence']}", canonical[source],
        tuple(normalize_greek(greek.get(source, "")).split()),
    )


def observed_sources(canonical, greek):
    used, _, previous_held, _ = prior_sources(canonical, greek)
    paths = [DATA / "manifest.json", *sorted(DATA.rglob("manifest*.json"))]
    paths = sorted({path for path in paths if OUTPUT not in path.parents})
    train_dev = set()
    for path in paths:
        for row in read_json(path).get("samples", []):
            raw = next((row[field] for field in (
                "source_path", "sourcePath", "source_id", "source_utterance", "sourceUtterance", "id"
            ) if isinstance(row.get(field), str)), "")
            source = source_id(raw.replace("__sentences", "/sentences"))
            if source not in canonical:
                continue
            used.add(source)
            key = annotation_key(source, canonical, greek)
            if row.get("split") in PROTECTED_ROLES:
                train_dev.add(key)
            if row.get("split") in HELD_ROLES:
                previous_held.add(key)
    for path in DATA.glob("*/rgb/*"):
        if OUTPUT in path.parents:
            continue
        source = source_id(path.name.replace("__sentences", "/sentences"))
        if source in canonical:
            used.add(source)
    return used, train_dev, previous_held, paths


def vocabulary(path):
    return {
        gloss for row in read_json(path)["samples"] if row["split"] == "train"
        for gloss in row["glosses"]
    }


def build_plan():
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    used, train_dev, held, manifests = observed_sources(canonical, greek)
    held_templates, held_sequences, held_texts = map(set, zip(*held, strict=True))
    known_glosses = vocabulary(SCALE / "manifest.json") & vocabulary(TRANCHE / "manifest.json")
    metadata = archive_metadata()[ARCHIVE]
    rows = []
    for source in SOURCES:
        if source in used or source not in timestamps:
            raise ValueError(f"Missing timestamp or previously observed source: {source}")
        match = SOURCE_ID.fullmatch(source)
        key = annotation_key(source, canonical, greek)
        template, glosses, words = key
        if (template in held_templates or glosses in held_sequences or words in held_texts
                or any(overlaps(key, other) for other in train_dev)):
            raise ValueError(f"Reference or template overlaps a previous cohort: {source}")
        timing = timestamps[source]
        text = greek[source]
        if not set(glosses) <= known_glosses or normalize_greek(text) != normalize_greek(timing["translation"]):
            raise ValueError(f"Unexpected glossary or timestamp reference: {source}")
        start = timing["frame_start_in_concatenated_sentence_sequence"]
        end = timing["frame_end_in_concatenated_sentence_sequence"]
        frame_count = timing["implied_inclusive_frame_count"]
        if timing["end_ms"] <= timing["start_ms"] or end - start + 1 != frame_count:
            raise ValueError(f"Invalid official timing: {source}")
        take = source.split("/")[0]
        rows.append({
            "id": source.replace("/", "__"), "source_path": "kep3/" + source,
            "source_archive": ARCHIVE, "source_url": metadata["links"]["self"], "split": "test",
            "signer": int(match["signer"]), "repetition": int(match["rep"]), "template": template,
            "take_id": take.removesuffix("_sentences"), "text": text, "glosses": list(glosses), "fps": 30,
            "start_ms": timing["start_ms"], "end_ms": timing["end_ms"],
            "source_frame_start": start, "source_frame_end": end, "annotated_frame_count": frame_count,
            "annotation_evidence": {"file": timing["annotation_file"], "timestamp_file": timing["timestamp_file"],
                                    "line_1based": timing["line_1based"]},
            "composition": "unseen_template", "category": "unseen_template_known_glosses",
            "prior_observed_other_sources_in_take": sum(item.startswith(take + "/") for item in used),
        })
    sizes = cached_sizes(ARCHIVE, metadata, {
        source: row["annotated_frame_count"] for source, row in zip(SOURCES, rows, strict=True)
    })
    if set(sizes) != set(SOURCES):
        raise ValueError("Cached central directory did not verify all annotated frames")
    for source, row in zip(SOURCES, rows, strict=True):
        row["preflight_rgb_bytes"] = sizes[source]
    repetitions = Counter((row["template"], row["signer"]) for row in rows)
    if sorted(repetitions.values()) != [5, 5, 5, 5]:
        raise ValueError("Expected five repetitions of both phrases from each signer")
    source_paths = sorted({SPLIT, SUPPLEMENTARY, METADATA, LIBRARY, SYNTHETIC, *manifests})
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "complete": False,
        "scope": "Posthoc diagnostic on two new scripted phrases from CERTH signers 6 and 7. "
                 "Both signer identities and other utterances from the recording takes have appeared in earlier evaluation.",
        "selection_context": "Source IDs sealed using only official annotations and cached ZIP directory records "
                             "after prior experiments and before any inference on these clips.",
        "prior_exclusion": {
            "previous_source_id_count": len(used),
            "train_development_annotation_groups": len(train_dev),
            "prior_held_annotation_groups": len(held),
            "rules": ["Exclude all previous source IDs and exact held script, gloss sequence or normalized Greek.",
                      "Exclude matching train/development script and Greek or gloss sequences within one token edit.",
                      "Require all glosses in both frozen CERTH training vocabularies."],
        },
        "split_signers": {"train": [], "val": [], "test": list(SIGNERS)},
        "planned_samples": len(rows), "unique_reference_texts": 2,
        "expected_annotated_frames": sum(row["annotated_frame_count"] for row in rows),
        "archive_checksum": metadata["checksum"], "preflight_rgb_bytes": sum(sizes.values()),
        "range_budget_bytes": sum(sizes.values()) + RETRY_ALLOWANCE,
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="Acquire RGB after sealing the metadata plan")
    args = parser.parse_args()
    target = OUTPUT / "manifest.planned.json"
    if target.exists():
        plan = read_json(target)
        if (tuple(row["source_path"].split("/", 1)[1] for row in plan["samples"]) != SOURCES
                or any(digest(ROOT / path) != expected for path, expected in plan["source_hashes"].items()
                       if ROOT / path != Path(__file__).resolve())):
            raise ValueError("The frozen source IDs or input hashes changed")
    else:
        plan = build_plan()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_manifest(target, plan)
    print(f"Sealed {plan['planned_samples']} IDs, {plan['preflight_rgb_bytes']:,} RGB range bytes, plan SHA256 {digest(target)}", flush=True)
    if args.acquire:
        manifest = OUTPUT / "manifest.json"
        previous = read_json(manifest) if manifest.exists() else None
        acquire(plan, OUTPUT, plan["range_budget_bytes"], False, previous, max_workers=1)
        result = read_json(manifest)
        if not result["complete"] or any(row["frame_count"] != row["annotated_frame_count"] for row in result["samples"]):
            raise ValueError("Acquired cohort failed frame-count verification")


if __name__ == "__main__":
    main()
