"""Acquire authentic GSL contexts with novel, signer-disjoint evaluation phrases."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from acquire_fresh_continuous import (
    DATA, METADATA, PLAN, PREVIOUS, ROOT, SIGNERS, SOURCE, SPLIT, SUPPLEMENTARY,
    acquire, archive_metadata, digest, matching_sources, read_json, write_manifest,
)
from plan_continuous_composition import normalize_greek, read_annotations


OUTPUT = DATA / "continuous-context"
SYNTHETIC = ROOT / "reports/pretrained-gsl/synthetic-continuous-train-plan.json"
LIBRARY = ROOT / "web/public/library/catalog.json"
HISTORY = (*PREVIOUS, DATA / "continuous-fresh-compact/manifest.json")
NOVEL = {
    "val": {"health4/sentences0016"},
    "test": {"kep4/sentences0003", "health1/sentences0009"},
}


def build_plan():
    history = [row for path in HISTORY for row in read_json(path)["samples"]]
    synthetic = read_json(SYNTHETIC)["samples"]
    roles = read_json(PLAN)["roles"]
    prior_templates = {row["template"] for row in history}
    prior_tuples = {tuple(row["glosses"]) for row in [*history, *synthetic]}
    prior_texts = {normalize_greek(row["text"]) for row in [*history, *synthetic]}
    prior_held = [row for row in history if row.get("composition") == "unseen_template"]
    reserved_tuples = {tuple(row["glosses"]) for row in prior_held}
    reserved_texts = {normalize_greek(row["text"]) for row in prior_held}
    seen_ids = {row["source_path"].split("/", 1)[1] for row in history}
    seen_ids.update(row["sourceUtterance"] for row in read_json(LIBRARY)["entries"])
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    archives = archive_metadata()

    def sources(template):
        return matching_sources(template, canonical, greek, seen_ids, archives)

    held = []
    for split, role in (("val", "development"), ("test", "test")):
        for template in roles[role]:
            if template["template_id"] not in NOVEL[split]:
                continue
            if (template["template_id"] in prior_templates or tuple(template["glosses"]) in prior_tuples
                    or template["normalized_greek"] in prior_texts):
                raise ValueError(f"Novel evaluation phrase already exposed: {template['template_id']}")
            held.append((split, template, sources(template)))
    if {template["template_id"] for _, template, _ in held} != NOVEL["val"] | NOVEL["test"]:
        raise ValueError("Frozen contextual holdouts do not match the original annotation plan")

    protected_templates = {template["template_id"] for role in ("development", "test") for template in roles[role]}
    protected_tuples = reserved_tuples | {tuple(template["glosses"]) for role in ("development", "test") for template in roles[role]}
    protected_texts = reserved_texts | {template["normalized_greek"] for role in ("development", "test") for template in roles[role]}
    eligible = {}
    for template in roles["train"]:
        identity = template["template_id"]
        if (identity in prior_templates or identity in protected_templates or tuple(template["glosses"]) in protected_tuples
                or template["normalized_greek"] in protected_texts):
            continue
        recordings = sources(template)
        if all(recordings.get(signer) for signer in SIGNERS["train"]):
            eligible[identity] = (template, recordings)

    training_contexts = defaultdict(set)
    for row in history:
        if row["split"] == "train":
            for gloss in row["glosses"]:
                training_contexts[gloss].add(row["template"])
    held_vocab = {gloss for _, template, _ in held for gloss in template["glosses"]}
    open_archives = {template["source_archive"] for _, template, _ in held}

    def count_external(gloss, scenario):
        return sum(not template.startswith(scenario + "/") for template in training_contexts[gloss])

    def deficits():
        return {(template["template_id"].split("/")[0], gloss) for _, template, _ in held for gloss in template["glosses"]
                if count_external(gloss, template["template_id"].split("/")[0]) < 2}

    chosen = []
    while remaining := deficits():
        def priority(identity):
            template, _ = eligible[identity]
            scenario = identity.split("/")[0]
            coverage = sum(scenario != blocked and gloss in template["glosses"] for blocked, gloss in remaining)
            return -coverage, template["source_archive"] not in open_archives, len(template["glosses"]), identity

        identity = min(eligible, key=priority)
        if priority(identity)[0] == 0:
            raise ValueError(f"No second independent sentence context for {sorted(remaining)}")
        template, recordings = eligible.pop(identity)
        chosen.append((template, recordings))
        open_archives.add(template["source_archive"])
        for gloss in template["glosses"]:
            training_contexts[gloss].add(identity)

    if len(chosen) > 12:
        raise ValueError("Too many essential contexts for the bounded RGB acquisition")
    coverage = {
        template["template_id"]: {
            gloss: sorted(source for source in training_contexts[gloss]
                          if source.split("/")[0] != template["template_id"].split("/")[0])
            for gloss in template["glosses"]
        }
        for _, template, _ in held
    }
    optional = sorted((identity for identity, (template, _) in eligible.items() if template["source_archive"] in open_archives),
                      key=lambda identity: (-len(held_vocab.intersection(eligible[identity][0]["glosses"])),
                                            len(eligible[identity][0]["glosses"]), identity))[:2]
    chosen.extend(eligible[identity] for identity in optional)

    samples = []
    for split, template, recordings in held:
        for signer in SIGNERS[split]:
            if len(recordings[signer]) < 2:
                raise ValueError(f"Missing independent source repetitions: {template['template_id']} signer {signer}")
            samples.extend({**row, "split": split, "composition": "unseen_template"} for row in recordings[signer][:2])
    for template, recordings in chosen:
        samples.extend({**recordings[signer][0], "split": "train", "composition": "known_template"} for signer in SIGNERS["train"])
    if len({row["id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate contextual source recordings")

    return {
        "schema": 1, "source": SOURCE, "license": "CC-BY-4.0", "split_signers": SIGNERS,
        "scope": "New CERTH studio recordings and held utterance gloss sequences. Source persons are separated by role within the experiment.",
        "selection": "Select exact published references. Require two different train template contexts outside each held scenario for every held gloss, counting previous real training and the selected core training rows. Freeze novel roles before acquiring RGB.",
        "heldGlossTrainContexts": coverage,
        "budget_optional_train_templates": optional, "minimum_train_signers_per_template": 3,
        "source_hashes": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (PLAN, SPLIT, SUPPLEMENTARY, METADATA, SYNTHETIC, LIBRARY, *HISTORY)},
        "planned_samples": len(samples), "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-transfer-mb", type=int, default=300)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or not 1 <= args.max_transfer_mb <= 300:
        parser.error("Use project-local output and a bounded transfer budget of at most 300 MB")
    output.mkdir(parents=True, exist_ok=True)
    planned = output / "manifest.planned.json"
    manifest = output / "manifest.json"
    if args.resume:
        if args.plan_only:
            parser.error("Cannot combine --resume with --plan-only")
        plan, previous = read_json(planned), read_json(manifest)
    else:
        if manifest.exists():
            parser.error("Use --resume for an existing acquisition")
        previous = None
        plan = build_plan()
        write_manifest(planned, plan)
    print(f"Planned {plan['planned_samples']} clips: {dict(Counter(row['split'] for row in plan['samples']))}", flush=True)
    if not args.plan_only:
        acquire(plan, output, args.max_transfer_mb * 1_000_000, not args.resume, previous)


if __name__ == "__main__":
    main()
