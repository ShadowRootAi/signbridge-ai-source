"""Freeze and acquire a CERTH cohort with a minimum gloss edit distance."""

import argparse
from collections import Counter, defaultdict

from acquire_fresh_continuous import (
    DATA, METADATA, ROOT, SIGNERS, SOURCE, SOURCE_ID, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, matching_sources, read_json, write_manifest,
)
from plan_continuous_composition import build_templates, normalize_greek, read_annotations, record


OUTPUT = DATA / "continuous-hard-split"
SYNTHETIC = ROOT / "reports/pretrained-gsl/synthetic-continuous-train-plan.json"
LIBRARY = ROOT / "web/public/library/catalog.json"
ELEMENTARY = DATA / "elementary-translation-pilot/manifest.json"
HELD = {
    "kep4/sentences0002": "test",
    "kep3/sentences0006": "test",
    "kep4/sentences0016": "val",
}
SUPPORT = {
    "police4/sentences0001": "ΧΑΝΩ",
    "police4/sentences0004": "ΧΑΝΩ",
    "health3/sentences0013": "ΕΤΟΙΜΟ",
    "health4/sentences0022": "ΕΤΟΙΜΟ",
    "police3/sentences0001": "ΚΑΙΝΟΥΡΙΟ",
    "police4/sentences0008": "ΚΑΙΝΟΥΡΙΟ",
    "kep2/sentences0005": "ΒΕΒΑΙΩΣΗ",
    "kep5/sentences0010": "ΒΕΒΑΙΩΣΗ",
}


def distance(left, right):
    previous = list(range(len(right) + 1))
    for index, token in enumerate(left, 1):
        row = [index]
        for column, other in enumerate(right, 1):
            row.append(min(row[-1] + 1, previous[column] + 1, previous[column - 1] + (token != other)))
        previous = row
    return previous[-1]


def build_plan():
    history_paths = sorted(path for path in DATA.glob("continuous-*/manifest.json") if path.parent != OUTPUT)
    historical = list({row["id"]: row for path in history_paths for row in read_json(path)["samples"]}.values())
    seen_ids = {row["source_path"].split("/", 1)[1] for row in historical}
    seen_ids.update(row["sourceUtterance"] for row in read_json(LIBRARY)["entries"])
    prior_templates = {row["template"] for row in historical}
    prior_rows = [*historical, *read_json(SYNTHETIC)["samples"]]
    prior_glosses = {tuple(row["glosses"]) for row in prior_rows}
    prior_texts = {normalize_greek(row["text"]) for row in [*prior_rows, *read_json(ELEMENTARY)["samples"]]}
    prior_holdouts = {tuple(row["glosses"]) for row in historical if row["split"] == "test"}
    prior_holdout_texts = {normalize_greek(row["text"]) for row in historical if row["split"] == "test"}
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    available, _ = build_templates(canonical, greek, 1, 10)
    held = {item.key: item for item in available if item.key in HELD}
    if set(held) != set(HELD):
        raise ValueError("Missing one of the frozen evaluation templates")
    archives = archive_metadata()
    rows = []
    for key, role in HELD.items():
        item = held[key]
        if (item.key in prior_templates or item.normalized_greek in prior_texts
                or any(distance(item.glosses, past) < 2 for past in prior_glosses)):
            raise ValueError(f"Evaluation template is too close to prior examples: {key}")
        matches = matching_sources(record(item, SIGNERS[role]), canonical, greek, seen_ids, archives)
        for signer in SIGNERS[role]:
            if len(matches[signer]) < 2:
                raise ValueError(f"Need two recordings for {key}, signer {signer}")
            rows.extend({**source, "split": role, "composition": "unseen_template"} for source in matches[signer][:2])

    candidates = defaultdict(list)
    for source, glosses in sorted(canonical.items()):
        if source in seen_ids or not (text := greek.get(source)):
            continue
        match = SOURCE_ID.fullmatch(source)
        scenario, signer = match["scenario"], int(match["signer"])
        key = f"{scenario}/{match['sentence']}"
        if key not in SUPPORT or signer not in SIGNERS["train"] or SUPPORT[key] not in glosses:
            continue
        normalized = normalize_greek(text)
        if (glosses in prior_holdouts or normalized in prior_holdout_texts
                or any(distance(glosses, item.glosses) < 2 or normalized == item.normalized_greek for item in held.values())):
            continue
        archive = scenario + ".zip"
        candidates[key, signer].append({
            "id": source.replace("/", "__"), "source_path": scenario + "/" + source,
            "signer": signer, "repetition": int(match["rep"]), "glosses": list(glosses), "text": text,
            "fps": 30, "template": key, "source_archive": archive, "source_url": archives[archive]["links"]["self"],
            "split": "train", "composition": "known_template",
        })
    for key in SUPPORT:
        for signer in SIGNERS["train"]:
            if not candidates[key, signer]:
                raise ValueError(f"Missing support clip for {key}, signer {signer}")
            rows.append(candidates[key, signer][0])

    training = [row for row in [*historical, *rows] if row["split"] == "train"]
    coverage = {}
    nearest = {}
    for key, item in held.items():
        coverage[key] = {
            gloss: sorted({row["template"] for row in training
                           if gloss in row["glosses"] and row["template"].split("/")[0] != item.scenario})
            for gloss in sorted(set(item.glosses))
        }
        if any(not contexts for contexts in coverage[key].values()):
            raise ValueError(f"Missing training coverage outside held scenario {key}")
        nearest[key] = min(distance(item.glosses, glosses)
                           for glosses in prior_glosses | {tuple(row["glosses"]) for row in training})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "split_signers": SIGNERS,
        "scope": "A CERTH studio diagnostic with held scripts and signer roles.",
        "selection": "Keep at least two token edits from historical and new train gloss sequences. "
                     "Take the first two annotated recordings per held signer and one per train signer.",
        "nearestGlossEditDistance": nearest,
        "heldGlossTrainContexts": coverage,
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (SPLIT, SUPPLEMENTARY, METADATA, SYNTHETIC, LIBRARY, ELEMENTARY, *history_paths)},
        "planned_samples": len(rows), "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    planned, manifest = OUTPUT / "manifest.planned.json", OUTPUT / "manifest.json"
    plan = build_plan()
    if planned.is_file() and read_json(planned) != plan:
        raise ValueError("Planned split or evidence changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not planned.is_file():
        write_manifest(planned, plan)
    print(dict(Counter(row["split"] for row in plan["samples"])), plan["nearestGlossEditDistance"], flush=True)
    if not args.plan_only:
        if manifest.is_file() != args.resume:
            parser.error("Use --resume if and only if acquisition already started")
        acquire(plan, OUTPUT, 300_000_000, False, read_json(manifest) if args.resume else None)


if __name__ == "__main__":
    main()
