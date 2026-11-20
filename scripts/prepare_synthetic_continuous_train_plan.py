"""Index synthetic training sequences from cached official OpenHands isolate poses."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from plan_continuous_composition import ROOT, normalize_greek, read_annotations


SPLIT = "data/original_annotations/GSL_split.zip"
SUPPLEMENTARY = "data/original_annotations/supplementary.zip"
COMPOSITION = "reports/continuous-composition-plan.json"
POSE_FILE = "research/cache/openhands-gsl/official-train-prototypes-poses.npz"
HISTORICAL_HOLDOUTS = ("data/continuous-v1-holdout/manifest.json", "data/continuous-composition-pilot/manifest.json")
TRAIN_SIGNERS = {1, 2, 4}
SESSION = r"(?P<scenario>[a-z]+\d+)_signer(?P<signer>\d+)_rep(?P<repetition>\d+)"
SENTENCE_PATH = re.compile(rf"(?P<session>{SESSION})_sentences/sentences(?P<index>\d+)")
ISOLATE_PATH = re.compile(rf"(?P<session>{SESSION})_glosses/glosses(?P<index>\d+)")


def sha(path):
    with (ROOT / path).open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_labels(archive, names):
    for name in names:
        for line in archive.read(name).decode("utf-8-sig").splitlines():
            yield line.split("|", 1)


def grouped_labels(rows, pattern):
    sessions = defaultdict(dict)
    for path, label in rows:
        match = pattern.fullmatch(path)
        if match is None:
            raise ValueError(f"Unexpected source path: {path}")
        index = int(match["index"])
        session = sessions[match["session"]]
        if index in session:
            raise ValueError(f"Duplicate source path: {path}")
        session[index] = label
    return sessions


def pose_index(official_train):
    with np.load(ROOT / POSE_FILE, allow_pickle=False) as poses:
        ids, glosses, offsets = (poses[name] for name in ("ids", "glosses", "offsets"))
        if (poses["split"].item() != "train-prototypes" or len(ids) != len(glosses)
                or len(set(ids)) != len(ids) or len(offsets) != len(ids) + 1
                or offsets[0] != 0 or offsets[-1] != len(poses["keypoints"]) or np.any(np.diff(offsets) <= 0)):
            raise ValueError("Unexpected cached official TRAIN prototype pose archive")
    index = {}
    for row, (identity, label, frames) in enumerate(zip(ids, glosses, np.diff(offsets))):
        match = ISOLATE_PATH.fullmatch(identity)
        if (match is None or int(match["signer"]) not in TRAIN_SIGNERS
                or int(match["repetition"]) != 1 or official_train.get(identity) != label):
            raise ValueError(f"Pose row is outside the verified TRAIN cohort: {identity}")
        index[str(identity)] = (row, str(label), int(frames))
    return index


def composition_references():
    roles = read_json(COMPOSITION)["roles"]
    approved = {example["source_path"]: (entry["template_id"], tuple(entry["glosses"]), entry["normalized_greek"])
                for entry in roles["train"] for example in entry["sources"]}
    excluded = {(entry["template_id"], tuple(entry["glosses"]), entry["normalized_greek"])
                for role in ("development", "test") for entry in roles[role]}
    historical = defaultdict(Counter)
    for path in HISTORICAL_HOLDOUTS:
        for row in read_json(path)["samples"]:
            if row.get("composition") == "unseen_template":
                historical[row["template"]][tuple(row["glosses"]), normalize_greek(row["text"])] += 1
    for template, variants in historical.items():
        glosses, text = min(variants, key=lambda item: (-variants[item], item))
        excluded.add((template, glosses, text))
    return approved, tuple({row[column] for row in excluded} for column in range(3))


def build_plan():
    sd, greek = read_annotations(ROOT / SPLIT, ROOT / SUPPLEMENTARY)
    with ZipFile(ROOT / SPLIT) as archive:
        si_names = [f"GSL_split/GSL_continuous/gsl_split_SI_{role}.csv" for role in ("train", "dev", "test")]
        si = grouped_labels(((path.split("/", 1)[1], tuple(label.split()))
                             for path, label in read_labels(archive, si_names)), SENTENCE_PATH)
        train_iso = dict(read_labels(archive, ["GSL_split/GSL_isolated/train_greek_iso.csv"]))
    sd_sessions = grouped_labels(sd.items(), SENTENCE_PATH)
    iso = grouped_labels(train_iso.items(), ISOLATE_PATH)
    poses = pose_index(train_iso)
    approved, (excluded_templates, excluded_glosses, excluded_texts) = composition_references()
    samples = []
    for session, sentences in sorted(si.items()):
        first = SENTENCE_PATH.fullmatch(f"{session}_sentences/sentences0000")
        if int(first["signer"]) not in TRAIN_SIGNERS or first["repetition"] != "1":
            continue
        flattened = tuple(gloss for _, labels in sorted(sentences.items()) for gloss in labels)
        isolates = iso.get(session, {})
        if (sentences != sd_sessions.get(session) or sorted(sentences) != list(range(len(sentences)))
                or tuple(isolates.get(i) for i in range(len(isolates))) != flattened):
            continue

        offset = 0
        for number, glosses in sorted(sentences.items()):
            identity = f"{session}_sentences/sentences{number:04d}"
            pose_ids = [f"{session}_glosses/glosses{i:04d}" for i in range(offset, offset + len(glosses))]
            offset += len(glosses)
            text = greek.get(identity, "")
            template = f"{first['scenario']}/sentences{number:04d}"
            normalized = normalize_greek(text)
            if (approved.get(identity) != (template, glosses, normalized) or template in excluded_templates
                    or glosses in excluded_glosses or normalized in excluded_texts
                    or any(pose_id not in poses for pose_id in pose_ids)):
                continue
            indexes, pose_glosses, lengths = zip(*(poses[pose_id] for pose_id in pose_ids))
            if glosses != pose_glosses or not indexes:
                raise ValueError(f"Unexpected pose labels: {identity}")
            samples.append({
                "id": identity.replace("/", "__"), "sourcePath": identity, "signer": int(first["signer"]),
                "scenario": first["scenario"], "template": template, "glosses": list(glosses), "text": text,
                "poseIndices": list(indexes), "poseIds": pose_ids, "frameCount": sum(lengths),
            })

    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Empty or duplicate synthetic training manifest")
    return {
        "schemaVersion": 1,
        "scope": "Synthetic TRAIN-only ordered concatenation of isolated pose segments. Index consistency does not prove source frame correspondence, and original inter-sign transitions are not reconstructed.",
        "poseFile": POSE_FILE,
        "poseFileSha256": sha(POSE_FILE),
        "sourceSha256": {path: sha(path) for path in (SPLIT, SUPPLEMENTARY, COMPOSITION, *HISTORICAL_HOLDOUTS)},
        "counts": {
            "samples": len(samples), "templates": len({sample["template"] for sample in samples}),
            "glosses": len({gloss for sample in samples for gloss in sample["glosses"]}),
            "bySigner": dict(sorted(Counter(sample["signer"] for sample in samples).items())),
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pretrained-gsl/synthetic-continuous-train-plan.json")
    args = parser.parse_args()
    plan = build_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan["counts"]))


if __name__ == "__main__":
    main()
