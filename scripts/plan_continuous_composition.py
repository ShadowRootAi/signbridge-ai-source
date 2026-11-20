"""Plan a signer- and composition-disjoint GSL evaluation from annotations."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "original_annotations"
SD_FILES = tuple(
    f"GSL_split/GSL_continuous/GSL-SD-{role}.txt"
    for role in ("train", "val", "test")
)
SUPPLEMENTARY_FILE = "merged_continuous.csv"
SOURCE_PATH = re.compile(
    r"(?P<scenario>\w+)_signer(?P<signer>\d+)_rep\d+_sentences/(?P<sentence>sentences\d+)"
)
SIGNERS = {"train": (1, 2, 4, 5), "development": (6,), "test": (3, 7)}


@dataclass(frozen=True)
class Template:
    scenario: str
    sentence: str
    glosses: tuple[str, ...]
    greek: str
    normalized_greek: str
    sources: tuple[tuple[int, str], ...]
    annotation_count: int

    @property
    def key(self):
        return f"{self.scenario}/{self.sentence}"


def normalize_greek(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", plain, flags=re.UNICODE))


def read_annotations(split_path, supplementary_path):
    with ZipFile(supplementary_path) as archive:
        lines = archive.read(SUPPLEMENTARY_FILE).decode("utf-8-sig").splitlines()
    greek = {}
    for line in lines:
        parts = line.split("|", 2)
        if len(parts) == 3 and SOURCE_PATH.fullmatch(parts[0]):
            if parts[0] in greek:
                raise ValueError(f"Duplicate Greek annotation: {parts[0]}")
            greek[parts[0]] = parts[1].strip()

    canonical = {}
    with ZipFile(split_path) as archive:
        for name in SD_FILES:
            lines = archive.read(name).decode("utf-8-sig").splitlines()
            for line in lines:
                if not line:
                    continue
                source, glosses = line.split("|", 1)
                if source in canonical:
                    raise ValueError(f"Duplicate canonical annotation: {source}")
                canonical[source] = tuple(unicodedata.normalize("NFC", glosses).split())
    return canonical, greek


def build_templates(canonical, greek, min_repeats, max_glosses):
    scripts = defaultdict(lambda: defaultdict(list))
    counts = Counter(canonical_sources=len(canonical))
    for source, glosses in sorted(canonical.items()):
        reference = greek.get(source, "")
        normalized = normalize_greek(reference)
        if len(glosses) >= 2:
            counts["multigloss_sources"] += 1
            counts["multigloss_with_greek"] += bool(normalized)
        if normalized:
            match = SOURCE_PATH.fullmatch(source)
            if match is None:
                raise ValueError(f"Unexpected source ID: {source}")
            script = match["scenario"], match["sentence"]
            scripts[script][glosses, normalized].append(
                (int(match["signer"]), source, reference)
            )

    required_signers = {signer for signers in SIGNERS.values() for signer in signers}
    templates = []
    for (scenario, sentence), variants in sorted(scripts.items()):
        glosses, normalized = min(variants, key=lambda pair: (-len(variants[pair]), pair))
        examples = sorted(variants[glosses, normalized])
        annotation_count = sum(map(len, variants.values()))
        if not 2 <= len(glosses) <= max_glosses or len(examples) * 5 < annotation_count * 4:
            continue
        signer_counts = Counter(signer for signer, _, _ in examples)
        if any(signer_counts[signer] < min_repeats for signer in required_signers):
            continue
        template = Template(
            scenario, sentence, glosses, examples[0][2], normalized,
            tuple((signer, source) for signer, source, _ in examples),
            annotation_count,
        )
        templates.append(template)
    counts["script_templates_with_greek"] = len(scripts)
    counts["stable_eligible_templates"] = len(templates)
    return templates, counts


def overlaps(first, second):
    return (
        first.key == second.key
        or first.glosses == second.glosses
        or first.normalized_greek == second.normalized_greek
    )


def training_pool(templates, holdouts):
    return [t for t in templates if all(not overlaps(t, h) for h in holdouts)]


def coverage(holdout, train, min_support):
    """Use unique canonical sequences from other scenarios for each gloss."""
    evidence = {}
    for gloss in sorted(set(holdout.glosses)):
        matches = {}
        for template in train:
            if template.scenario != holdout.scenario and gloss in template.glosses:
                matches.setdefault(template.glosses, template.key)
        if len(matches) < min_support:
            return None
        evidence[gloss] = list(matches.values())[:min_support]
    return evidence


def select_holdouts(templates, development_count, test_count, min_support):
    candidates = [
        t for t in templates if coverage(t, training_pool(templates, [t]), min_support)
    ]
    roles = {"development": [], "test": []}
    selected = []
    scenario_counts = Counter()
    lengths = sorted({len(template.glosses) for template in candidates})
    if not lengths:
        roles["train"] = templates
        return roles, 0
    for index in range(max(development_count, test_count)):
        previous_count = len(selected)
        for role, target in (("test", test_count), ("development", development_count)):
            if index >= target:
                continue
            preferred = lengths[(index + (role == "development")) % len(lengths)]
            ranked = sorted(candidates, key=lambda t: (
                scenario_counts[t.scenario], abs(len(t.glosses) - preferred),
                -len(t.sources), t.key,
            ))
            for template in ranked:
                if any(overlaps(template, previous) for previous in selected):
                    continue
                proposed = [*selected, template]
                train = training_pool(templates, proposed)
                if all(coverage(t, train, min_support) for t in proposed):
                    selected.append(template)
                    roles[role].append(template)
                    scenario_counts[template.scenario] += 1
                    break
        if len(selected) == previous_count:
            break
    roles["train"] = training_pool(templates, selected)
    return roles, len(candidates)


def record(template, signers):
    clips = [{"signer": signer, "source_path": source}
             for signer, source in template.sources if signer in signers]
    return {
        "template_id": template.key,
        "source_archive": f"{template.scenario}.zip",
        "greek": template.greek,
        "normalized_greek": template.normalized_greek,
        "glosses": list(template.glosses),
        "modal_annotation_count": len(template.sources),
        "paired_annotation_count": template.annotation_count,
        "verified_signer_repetitions": dict(sorted(Counter(
            signer for signer, _ in template.sources
        ).items())),
        "sources": clips,
    }


def build_plan(split_path, supplementary_path, development_count, test_count,
               min_repeats, min_support, max_glosses):
    canonical, greek = read_annotations(split_path, supplementary_path)
    templates, inventory = build_templates(canonical, greek, min_repeats, max_glosses)
    roles, candidate_count = select_holdouts(templates, development_count, test_count, min_support)
    records = {}
    pilot_holdouts = {}
    pilot_training = set()
    for role in SIGNERS:
        entries = [record(template, SIGNERS[role]) for template in roles[role]]
        if role != "train":
            for template, entry in zip(roles[role], entries, strict=True):
                evidence = coverage(template, roles["train"], min_support)
                if evidence is None:
                    raise ValueError(f"Uncovered holdout: {template.key}")
                entry["cross_scenario_train_support_by_gloss"] = evidence
            pilot_entries = entries[:1 if role == "development" else 2]
            pilot_holdouts[role] = [entry["template_id"] for entry in pilot_entries]
            for entry in pilot_entries:
                for supporters in entry["cross_scenario_train_support_by_gloss"].values():
                    pilot_training.update(supporters)
        records[role] = entries

    for field, key in (
        ("template", lambda t: t.key),
        ("Greek reference", lambda t: t.normalized_greek),
        ("gloss sequence", lambda t: t.glosses),
    ):
        seen = set()
        for group in roles.values():
            keys = {key(t) for t in group}
            if keys & seen:
                raise ValueError(f"Cross-role {field} overlap: {keys & seen}")
            seen.update(keys)

    return {
        "status": "annotation_plan_only",
        "sources": {
            "canonical_glosses": {
                "archive": split_path.name, "members": list(SD_FILES),
                "sha256": sha256(split_path.read_bytes()).hexdigest(),
            },
            "greek_references": {
                "archive": supplementary_path.name, "member": SUPPLEMENTARY_FILE,
                "sha256": sha256(supplementary_path.read_bytes()).hexdigest(),
            },
        },
        "protocol": {
            "requested_holdout_templates": {"development": development_count, "test": test_count},
            "signers_by_role": SIGNERS,
            "required_shared_signer_count": sum(map(len, SIGNERS.values())),
            "minimum_modal_repetitions_per_signer": min_repeats,
            "minimum_modal_fraction_of_paired_annotations": 0.8,
            "maximum_glosses_per_template": max_glosses,
            "minimum_distinct_other_scenario_training_sequences_per_holdout_gloss": min_support,
            "disjoint_across_roles": ["script template", "normalized Greek reference", "gloss sequence"],
            "greek_normalization": "Casefold, remove combining marks, normalize punctuation and spacing.",
            "holdout_selection": "Cycle gloss lengths and prioritize underrepresented scenarios.",
        },
        "inventory": {**inventory, "individually_eligible_holdout_templates": candidate_count},
        "holdout_shortfall": {
            "development": development_count - len(roles["development"]),
            "test": test_count - len(roles["test"]),
        },
        "counts": {
            role: {
                "templates": len(entries),
                "sources": sum(len(entry["sources"]) for entry in entries),
                "scenarios": len({entry["source_archive"] for entry in entries}),
                "templates_by_gloss_count": dict(sorted(Counter(
                    len(entry["glosses"]) for entry in entries
                ).items())),
            }
            for role, entries in records.items()
        },
        "suggested_pilot_templates": {"train": sorted(pilot_training), **pilot_holdouts},
        "roles": records,
        "limitations": [
            "Source IDs and signer repetition counts come from annotations. RGB availability requires separate verification.",
            "The corpus consists of scripted utterances. Spontaneous conversation and model quality require separate video-based evaluation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-zip", type=Path, default=ANNOTATIONS / "GSL_split.zip")
    parser.add_argument("--supplementary-zip", type=Path, default=ANNOTATIONS / "supplementary.zip")
    parser.add_argument("--development", type=int, default=3)
    parser.add_argument("--test", type=int, default=5)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-glosses", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.development, args.test) < 1 or min(args.min_repeats, args.min_support) < 1:
        parser.error("Role sizes, minimum repetitions and minimum support must be positive")
    if args.max_glosses < 2:
        parser.error("The maximum gloss count must be at least two")
    plan = build_plan(args.split_zip, args.supplementary_zip, args.development, args.test,
                      args.min_repeats, args.min_support, args.max_glosses)
    if args.output is None:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "counts": plan["counts"],
                          "suggested_pilot_templates": plan["suggested_pilot_templates"]}, indent=2))


if __name__ == "__main__":
    main()
