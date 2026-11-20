"""Freeze cached RGB and bounded acquisition cohorts before scoring a model."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from prepare_elementary_catalog_plan import word_key


ROOT = Path(__file__).resolve().parents[1]
CERTH_ROLES = {1: "development", 2: "development", 3: "diagnostic", 4: "development",
               5: "development", 6: "calibration", 7: "diagnostic"}
ELEMENTARY_ROLES = {signer: "cross_corpus_calibration" if signer in "ABCD" else "cross_corpus_probe"
                    for signer in "ABCDEFGHI"}
CERTH_MANIFESTS = ("data/manifest.json", "data/open-set-v2/manifest.json", "data/v3/manifest.json")
POLICY_PATH = "reports/pretrained-gsl/stgcn-gate-config.json"
CLASS_PATH = "research/cache/openhands-gsl/stgcn-onnx.json"
LIBRARY_PATH = "web/public/library/catalog.json"
LIBRARY_REVIEW_PATH = "reports/library-100/rights-review.json"
UNUSED_PATH = "data/v3/unused_single_gloss_catalog.json"


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def sha(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def certh_cohorts(roster, public_sources):
    by_source = {}
    for manifest in CERTH_MANIFESTS:
        for row in read_json(manifest)["samples"]:
            previous = by_source.get(row["source_path"])
            if previous and (row["signer_id"], row["raw_label"], row["sha256"]) != (
                previous["signer_id"], previous["raw_label"], previous["sha256"]
            ):
                raise ValueError(f"Conflicting annotation or RGB data: {row['source_path']}")
            by_source[row["source_path"]] = row

    cached = []
    for path, row in sorted(by_source.items()):
        frames = row["frame_paths"]
        if not frames or not all((ROOT / frame).is_file() for frame in frames):
            raise ValueError(f"Missing previously cached RGB data: {path}")
        if len(frames) != row["frame_count"]:
            raise ValueError(f"Incorrect RGB frame count: {path}")
        label = row["raw_label"]
        signer = row["signer_id"]
        cached.append({
            "id": row["id"], "source_path": path, "signer": signer,
            "role": CERTH_ROLES[signer], "gloss": label, "on_roster": label in roster,
            "annotation_raw_glosses": row["annotation_raw_glosses"],
            "source_archive": row["source_archive"], "fps": row["fps"],
            "frame_paths": frames, "source_rgb_sha256": row["sha256"],
        })

    eligible = defaultdict(list)
    for row in read_json(UNUSED_PATH)["samples"]:
        path = row["source_path"]
        if path in by_source or path in public_sources or row["annotation_raw_gloss"] != row["raw_label"]:
            continue
        eligible[row["signer_id"], row["raw_label"]].append(row)

    # One record per signer and gloss. Prefer archives that supply many of the
    # required signer/gloss combinations so the central-directory cost is bounded.
    archive_coverage = Counter(scenario for rows in eligible.values()
                               for scenario in {row["scenario"] for row in rows})
    pending = []
    for (signer, label), rows in sorted(eligible.items()):
        row = min(rows, key=lambda sample: (-archive_coverage[sample["scenario"]], sample["source_path"]))
        pending.append({
            "id": row["source_path"].replace("/", "__"), "source_path": row["source_path"],
            "signer": signer, "role": CERTH_ROLES[signer], "gloss": label,
            "on_roster": label in roster, "source_archive": row["scenario"] + ".zip",
            "annotation_raw_glosses": row["annotation_raw_gloss"], "fps": 30,
        })
    return cached, pending


def elementary_cohort(library, labels, roster):
    # A unique orthographic match is a proposed mapping. Elementary23 supplies
    # whole-video spoken Greek transcriptions rather than sign-level gloss IDs.
    mappings = defaultdict(set)
    for label in labels:
        mappings[word_key(label)].add(label)
    reviewed = {row["id"]: row for row in read_json(LIBRARY_REVIEW_PATH)["clips"] if row["id"].startswith("elementary-")}
    selected = []
    for row in library:
        if row["id"] not in reviewed:
            continue
        labels_for_text = mappings[word_key(row["phrase"])]
        if len(labels_for_text) != 1:
            continue
        label, = labels_for_text
        signer = reviewed[row["id"]]["signer_id"]
        path = "web/public" + row["videoUrl"]
        if sha(path) != row["sha256"]:
            raise ValueError(f"Local library media checksum differs: {path}")
        selected.append({
            "id": row["id"], "source_path": row["sourceUtterance"], "signer": signer,
            "role": ELEMENTARY_ROLES[signer], "translation": row["phrase"],
            "candidate_gloss": label, "gloss_reviewed": False, "on_roster": label in roster,
            "video_path": path, "video_sha256": row["sha256"],
        })
    return selected


def counts(rows, label_key):
    return {
        "clips": len(rows), "distinct_labels": len({row[label_key] for row in rows}),
        "distinct_roster_labels": len({row[label_key] for row in rows if row["on_roster"]}),
        "by_role": dict(sorted(Counter(row["role"] for row in rows).items())),
        "by_signer": dict(sorted(Counter(row["signer"] for row in rows).items())),
        "by_label": dict(sorted(Counter(row[label_key] for row in rows).items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pretrained-gsl/rgb-audit-plan.json")
    args = parser.parse_args()
    policy = read_json(POLICY_PATH)
    labels = read_json(CLASS_PATH)["labels"]
    roster = set(policy["supportedLabels"])
    library = read_json(LIBRARY_PATH)["entries"]
    cached, pending = certh_cohorts(roster, {row["sourceUtterance"] for row in library})
    elementary = elementary_cohort(library, labels, roster)
    report = {
        "schema_version": 1,
        "scope": [
            "CERTH cache includes previously exposed clips. CERTH groups are signer-disjoint inside this plan; the frozen OpenHands weights or checkpoint selection already used all seven source identities.",
            "Pending CERTH samples have unseen clip IDs in the project RGB manifests and public library. Their RGB has not been acquired and upstream model exposure is possible.",
            "Elementary labels are tentative exact text matches. Use for diagnostics only until a GSL reviewer confirms the performed gloss. Cross-corpus personal identity overlap is unknown.",
            "Selections and assignments use source data only and are frozen before any inference.",
        ],
        "source_sha256": {name: sha(name) for name in (*CERTH_MANIFESTS, POLICY_PATH, CLASS_PATH,
                                                         LIBRARY_PATH, LIBRARY_REVIEW_PATH, UNUSED_PATH)},
        "counts": {"certh_cached": counts(cached, "gloss"), "certh_pending": counts(pending, "gloss"),
                   "elementary_text_probe": counts(elementary, "candidate_gloss")},
        "certh_cached": cached,
        "certh_pending": pending,
        "elementary_text_probe": elementary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=True))


if __name__ == "__main__":
    main()
