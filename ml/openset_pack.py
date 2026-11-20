"""Audit the locked open-set manifest and write separately sealed split NPZ files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import FeatureDataset, validate_dataset
from preprocess import FEATURE_COUNT, FEATURE_SCHEMA, TIMESTEPS, preprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(manifest_path, plan_path, jsonl_paths, frozen_path, feature_code, output):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    samples = manifest["samples"]
    planned = {sample["id"]: sample for sample in plan["samples"]}
    by_id = {sample["id"]: sample for sample in samples}
    if len(by_id) != len(samples) or set(by_id) != set(planned):
        raise ValueError("Completed manifest sample identities differ from the locked plan")
    for identity, sample in by_id.items():
        for field in ("source_path", "class_id", "signer_id", "split", "raw_label"):
            if sample[field] != planned[identity][field]:
                raise ValueError(f"Locked plan field changed: {identity} {field}")
    with np.load(frozen_path, allow_pickle=False) as old:
        old_sources = set(old["source_id"].astype(str))
        old_test_sources = set(old["source_id"][old["split"].astype(str) == "test"].astype(str))
    for sample in samples:
        if sample["source_path"] in old_test_sources:
            raise ValueError(f"An old diagnostic test clip entered the new experiment: {sample['id']}")
        if sample["split"] == "final_test" and sample["source_path"] in old_sources:
            raise ValueError(f"Fresh final contains an old clip: {sample['id']}")
    expected_feature_hash = sha(feature_code)
    records, ignored, pipeline_hashes = {}, 0, set()
    for path in jsonl_paths:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["id"] not in by_id:
                    ignored += 1
                    continue
                if record.get("feature_sha256") != expected_feature_hash:
                    raise ValueError(f"Feature-code hash mismatch: {record['id']}")
                pipeline_hashes.add(record.get("pipeline_sha256", "unrecorded"))
                if record["id"] in records:
                    previous = records[record["id"]]
                    if previous["x"] != record["x"] or previous["quality"] != record["quality"]:
                        raise ValueError(f"Conflicting duplicate extraction: {record['id']}")
                    if not previous.get("raw") and record.get("raw"):
                        records[record["id"]] = record
                else:
                    records[record["id"]] = record
    missing = set(by_id) - set(records)
    if missing:
        raise ValueError(f"Missing {len(missing)} locked extraction records; first: {sorted(missing)[:5]}")
    seen_frame_hashes = {}
    xs, quality, frame_counts, spans, parity = [], [], [], [], []
    for sample in samples:
        identity = sample["id"]; record = records[identity]
        hashes = sample.get("frame_sha256")
        if not isinstance(hashes, list) or not hashes or not all(isinstance(h, str) for h in hashes):
            raise ValueError(f"Missing acquired frame hashes: {identity}")
        for digest in hashes:
            previous_split = seen_frame_hashes.setdefault(digest, sample["split"])
            if previous_split != sample["split"]:
                raise ValueError(f"Identical frame crosses splits: {identity}")
        x = np.asarray(record["x"], dtype=np.float32)
        if x.shape != (TIMESTEPS, FEATURE_COUNT):
            raise ValueError(f"Feature shape mismatch: {identity}")
        count = int(record["frames"])
        if count != len(sample["frame_paths"]):
            raise ValueError(f"Extracted frame count differs from source frames: {identity}")
        span = (count - 1) * 1000 / float(sample["fps"])
        if record.get("raw"):
            frames = record["raw"]
            px, pq = preprocess(frames)
            np.testing.assert_allclose(px, x, atol=1e-6, rtol=1e-6, err_msg=identity)
            if abs(pq - float(record["quality"])) > 1e-10:
                raise ValueError(f"Quality parity mismatch: {identity}")
            actual_span = float(frames[-1]["t"] - frames[0]["t"])
            if abs(actual_span - span) > 1e-6:
                raise ValueError(f"Observed timestamp span contradicts JPEG sampling contract: {identity}")
            parity.append({"id": identity, "split": sample["split"], "max_abs_error": float(np.abs(px - x).max())})
        xs.append(x); quality.append(float(record["quality"])); frame_counts.append(count); spans.append(span)
    internal_splits = np.asarray(["test" if sample["split"] == "final_test" else sample["split"] for sample in samples])
    metadata = {"featureSchema": FEATURE_SCHEMA, "T": 48, "F": 224,
                "dataset_plan_sha256": sha(plan_path), "completed_manifest_sha256": sha(manifest_path),
                "feature_code_sha256": expected_feature_hash, "pipeline_hashes": sorted(pipeline_hashes),
                "pipeline_hash_note": "Different worker hashes may reflect inference-only changes; unchanged feature code and raw Python parity are required. See extraction reports.",
                "source": manifest.get("source", plan["source"]), "classes": plan["classes"],
                "exact_official_aliases": plan.get("exact_official_aliases", {}),
                "final_identity_caveat": plan["final_identity_caveat"],
                "preprocessing_parity_fixture_count": len(parity),
                "observed_span_ms_basis": "actual last-first extraction timestamps; JPEG contract (frame_count-1)*1000/fps, checked against available raw fixtures"}
    data = FeatureDataset(np.stack(xs), np.asarray([s["class_id"] for s in samples], dtype=np.int64),
                          np.asarray([str(s["signer_id"]) for s in samples]), np.asarray([s["source_path"] for s in samples]),
                          internal_splits, plan["classes"], np.asarray([s["raw_label"] for s in samples]),
                          np.asarray(quality, dtype=np.float32), metadata)
    validate_dataset(data)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, internal in (("train", "train"), ("val", "val"), ("final_test", "test")):
        mask = data.split == internal
        local_metadata = {**metadata, "original_split_role": name, "internal_split_role": internal,
                          "fresh_examples": name == "final_test", "sealed_until_validation_selection": name == "final_test"}
        path = output / f"{name}.npz"
        np.savez_compressed(path, x=data.x[mask], y=data.y[mask], quality=data.quality[mask],
                            signer=data.signer[mask], source_id=data.source_id[mask], split=data.split[mask],
                            raw_label=data.raw_label[mask], classes=np.asarray(data.classes),
                            clip_id=np.asarray([s["id"] for s in samples])[mask],
                            scenario=np.asarray([s["scenario"] for s in samples])[mask],
                            session=np.asarray([s["session"] for s in samples])[mask],
                            annotation_raw=np.asarray([s.get("annotation_raw_glosses", s["raw_label"]) for s in samples])[mask],
                            frame_count=np.asarray(frame_counts, dtype=np.int64)[mask],
                            observed_span_ms=np.asarray(spans, dtype=np.float64)[mask],
                            metadata_json=np.asarray(json.dumps(local_metadata, ensure_ascii=False)))
        outputs[name] = {"path": str(path.resolve()), "sha256": sha(path), "n": int(mask.sum()),
                         "known_n": int((mask & (data.y >= 0)).sum()), "unknown_n": int((mask & (data.y == -1)).sum())}
    audit = {"dataset_plan_sha256": sha(plan_path), "outputs": outputs, "ignored_unselected_records": ignored,
             "old_69_test_excluded_from_every_split": True, "fresh_final_excludes_all_old_243": True,
             "feature_code_sha256": expected_feature_hash, "pipeline_hashes": sorted(pipeline_hashes),
             "raw_preprocessing_parity": parity, "fixture_count": len(parity),
             "max_abs_error": max((p["max_abs_error"] for p in parity), default=None),
             "final_test_status": "packed and hashed only; no model inference or tuning performed"}
    (output / "packing-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan-lock", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--feature-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = pack(args.manifest, args.plan_lock, args.jsonl, args.frozen_features, args.feature_code, args.output)
    print(json.dumps({"outputs": report["outputs"], "fixtures": report["fixture_count"], "max_abs_error": report["max_abs_error"]}, ensure_ascii=False, indent=2))
