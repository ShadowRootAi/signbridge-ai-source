"""Audit browser JSONL against manifest, verify Python parity, then pack NPZ."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import FeatureDataset, dataset_summary, validate_dataset
from preprocess import FEATURE_COUNT, FEATURE_SCHEMA, TIMESTEPS, preprocess


def pack(manifest_path: Path, jsonl_paths: list[Path], output: Path, allow_partial: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    samples = manifest["samples"]
    identities = [str(sample["id"]) for sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("Manifest has duplicate sample IDs")
    by_id = dict(zip(identities, samples))
    records = {}
    for path in jsonl_paths:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                identity = str(record["id"])
                if identity not in by_id:
                    raise ValueError(f"{path}:{line_number}: ID absent from manifest: {identity}")
                if identity in records:
                    previous = records[identity]
                    if previous["x"] != record["x"] or previous["quality"] != record["quality"]:
                        raise ValueError(f"Conflicting extraction records for {identity}")
                    if "raw" not in previous and "raw" in record:
                        records[identity] = record
                else:
                    records[identity] = record
    missing = [identity for identity in identities if identity not in records]
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} manifest samples; first IDs: {missing[:5]}")
    if not records:
        raise ValueError("No extracted records")
    selected = [sample for sample in samples if str(sample["id"]) in records]
    # Detect exact duplicate sampled frame sequences across different splits.
    hash_splits = {}
    for sample in selected:
        hashes = sample.get("frame_sha256")
        content_hash = sample.get("sha256")
        if hashes:
            content_hash = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
        if content_hash:
            previous = hash_splits.setdefault(content_hash, sample["split"])
            if previous != sample["split"]:
                raise ValueError(f"Identical frame content crosses splits: {sample['id']}")
    parity_examples = []
    xs, qualities = [], []
    for sample in selected:
        record = records[str(sample["id"])]
        x = np.asarray(record["x"], dtype=np.float32)
        if x.shape != (TIMESTEPS, FEATURE_COUNT):
            raise ValueError(f"{sample['id']}: expected x shape {(TIMESTEPS, FEATURE_COUNT)}, got {x.shape}")
        quality = float(record["quality"])
        if "raw" in record and record["raw"] is not None:
            raw = record["raw"]
            frames = raw["frames"] if isinstance(raw, dict) else raw
            python_x, python_quality = preprocess(frames)
            np.testing.assert_allclose(python_x, x, rtol=1e-6, atol=1e-6,
                                       err_msg=f"Python/browser preprocessing mismatch: {sample['id']}")
            if abs(python_quality - quality) > 1e-10:
                raise ValueError(f"Python/browser quality mismatch: {sample['id']}")
            parity_examples.append({"id": sample["id"], "split": sample["split"],
                                    "frame_count": len(frames), "max_abs_error": float(np.max(np.abs(python_x - x))),
                                    "quality": quality, "quality_abs_error": abs(python_quality - quality)})
        xs.append(x)
        qualities.append(quality)
    metadata = {
        "featureSchema": FEATURE_SCHEMA, "T": TIMESTEPS, "F": FEATURE_COUNT,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset": manifest.get("dataset"), "dataset_version": manifest.get("dataset_version"),
        "subset_type": manifest.get("subset_type"),
        "not_official_isolated_benchmark": manifest.get("not_official_isolated_benchmark", True),
        "license": manifest.get("license"), "license_url": manifest.get("license_url"),
        "source": manifest.get("source"),
        "lexical_validation": "original dataset annotation; no new expert validation",
        "selected_samples": len(selected), "missing_manifest_samples": missing,
        "preprocessing_parity_examples": len(parity_examples),
        "preprocessing_parity": {"fixture_count": len(parity_examples),
                                 "status": "passed" if parity_examples else "not_tested_no_raw_fixtures",
                                 "max_abs_error": max((item["max_abs_error"] for item in parity_examples), default=None),
                                 "atol": 1e-6, "rtol": 1e-6},
        "extraction_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in jsonl_paths},
    }
    data = FeatureDataset(
        x=np.stack(xs), y=np.asarray([sample["class_id"] for sample in selected], dtype=np.int64),
        signer=np.asarray([str(sample["signer_id"]) for sample in selected]),
        source_id=np.asarray([sample.get("source_path", sample["id"]) for sample in selected]),
        split=np.asarray([sample["split"] for sample in selected]),
        classes=manifest["classes"], raw_label=np.asarray([sample["raw_label"] for sample in selected]),
        quality=np.asarray(qualities, dtype=np.float32), metadata=metadata,
    )
    validate_dataset(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=data.x, y=data.y, signer=data.signer, source_id=data.source_id,
                        clip_id=np.asarray([sample["id"] for sample in selected]), split=data.split,
                        classes=np.asarray(data.classes), raw_label=data.raw_label, quality=data.quality,
                        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    report = {"status": "passed" if parity_examples else "not_tested_no_raw_fixtures",
              "examples": parity_examples, "atol": 1e-6, "rtol": 1e-6,
              "max_abs_error": max((item["max_abs_error"] for item in parity_examples), default=None),
              "dataset": dataset_summary(data),
              "quality_by_split": {split: {"min": float(data.quality[data.mask(split)].min()),
                                            "mean": float(data.quality[data.mask(split)].mean()),
                                            "zero_quality_n": int((data.quality[data.mask(split)] == 0).sum())}
                                   for split in ("train", "val", "test")}}
    report_path = output.with_name(output.stem + "-preprocessing-parity.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"features": str(output.resolve()), "parity_report": str(report_path.resolve()),
            "samples": len(selected), "shape": list(data.x.shape), "parity_status": report["status"],
            "parity_examples": len(parity_examples), "max_abs_error": report["max_abs_error"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(pack(args.manifest, args.jsonl, args.output, args.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
