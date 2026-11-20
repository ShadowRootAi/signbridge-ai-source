"""Join fixed CERTH and Sartinas clip manifests for a paired runtime comparison."""

import argparse
from hashlib import file_digest
import json
from os.path import relpath
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/openhands-engine/replay-manifest.json"
COHORTS = (
    ("certh", ".runtime-tmp/recognition-rgb-audit/manifest.json", "clips", 96),
    ("certh-expansion", "data/isolated-expansion-rgb/manifest.json", "samples", 12),
    ("sartinas-context", "data/isolated-expansion-sartinas/manifest.json", "samples", 12),
)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Include the 96 prior CERTH development and diagnostic clips")
    args = parser.parse_args()
    cohorts = COHORTS if args.full else COHORTS[1:]
    policy = ROOT / "reports/pretrained-gsl/stgcn-gate-roster-200.json"
    supported = set(json.loads(policy.read_bytes())["supportedLabels"])
    dataset = {"scope": "Paired replay of previously examined genuine RGB clips with one fixed 200-label policy. "
                        "CERTH clips share corpus with model training; Sartinas excerpts are author-timed in context. "
                        "Model outputs were not used to choose this combined comparison.",
               "policySha256": digest(policy), "sourceManifests": {}, "clips": []}
    seen_ids, seen_media = set(), set()
    for group, relative, key, count in cohorts:
        manifest = ROOT / relative
        rows = json.loads(manifest.read_bytes())[key]
        if len(rows) != count:
            raise ValueError(f"Unexpected source count for {group}")
        dataset["sourceManifests"][relative] = digest(manifest)
        for row in rows:
            path = (manifest.parent / row["path"]).resolve()
            fingerprint = row["sha256"]
            if (not path.is_relative_to(ROOT) or digest(path) != fingerprint
                    or row["id"] in seen_ids or fingerprint in seen_media):
                raise ValueError(f"Invalid, duplicate or changed clip: {row['id']}")
            seen_ids.add(row["id"])
            seen_media.add(fingerprint)
            dataset["clips"].append({"id": row["id"], "gloss": row["gloss"], "group": group,
                                     "path": Path(relpath(path, OUTPUT.parent)).as_posix(),
                                     "sha256": fingerprint, "signer": row["signer"],
                                     "on_roster": row["gloss"] in supported})
    output = OUTPUT.with_name("replay-full-manifest.json") if args.full else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(dataset['clips'])} verified clips for paired browser replay")


if __name__ == "__main__":
    main()
