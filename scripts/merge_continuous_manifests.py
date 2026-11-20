"""Combine the preregistered cohorts without changing roles or references."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    inputs, samples = [], []
    ids, sources, signer_roles = set(), set(), {}
    for path in args.inputs:
        payload = path.read_bytes()
        manifest = json.loads(payload)
        if not manifest["complete"] or len(manifest["samples"]) != manifest["planned_samples"]:
            raise ValueError(f"Incomplete acquisition: {path}")
        inputs.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        for row in manifest["samples"]:
            signer = row["signer"]
            if row["id"] in ids or row["source_path"] in sources or signer_roles.setdefault(signer, row["split"]) != row["split"]:
                raise ValueError(f'Duplicate source or signer overlap: {row["id"]}')
            ids.add(row["id"])
            sources.add(row["source_path"])
            samples.append(row)
    result = {"source_manifests": inputs, "planned_samples": len(samples), "complete": True, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Merged {len(samples)} verified utterance references into {args.output}")


if __name__ == "__main__":
    main()
