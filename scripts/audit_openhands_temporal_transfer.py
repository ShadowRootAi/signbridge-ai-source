"""Check upstream isolated overlap and inspect native OpenHands temporal features."""

from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

from plan_continuous_composition import ROOT, read_annotations
from prepare_synthetic_continuous_train_plan import (
    ISOLATE_PATH, SENTENCE_PATH, grouped_labels, read_labels,
)

sys.path.insert(0, str(ROOT / "ml"))
from openhands_gsl_stgcn import load_openhands_gsl_stgcn, prepare_stgcn_pose


CACHE = ROOT / "research/cache/openhands-gsl"
HARD_PLAN = ROOT / "data/continuous-hard-split/manifest.planned.json"
SPLIT = ROOT / "data/original_annotations/GSL_split.zip"
SUPPLEMENTARY = ROOT / "data/original_annotations/supplementary.zip"
OUTPUT = ROOT / "reports/pretrained-gsl/temporal-transfer-audit.json"


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def split_overlap(metadata, plan):
    isolates, role_of, signer_roles = {}, {}, {}
    with ZipFile(metadata) as native, ZipFile(SPLIT) as official:
        for role, file_prefix in (("train", "train"), ("val", "val"), ("test", "test")):
            member = f"GSL_split/GSL_isolated/{file_prefix}_greek_iso.csv"
            if native.read(member) != official.read(member):
                raise ValueError(f"The isolated {role} annotation differs across releases")
            rows = dict(read_labels(native, [member]))
            if isolates.keys() & rows.keys():
                raise ValueError("Duplicate isolated IDs across roles")
            isolates.update(rows)
            role_of.update(dict.fromkeys(rows, role))
            signer_roles[role] = dict(Counter(int(ISOLATE_PATH.fullmatch(item)["signer"]) for item in rows))

    canonical, _ = read_annotations(SPLIT, SUPPLEMENTARY)
    sentences = grouped_labels(canonical.items(), SENTENCE_PATH)
    segments = grouped_labels(isolates.items(), ISOLATE_PATH)
    items = []
    for row in plan["samples"]:
        if row["split"] == "train":
            continue
        identity = row["source_path"].split("/", 1)[1]
        match = SENTENCE_PATH.fullmatch(identity)
        session, index = match["session"], int(match["index"])
        ordered_sentences = sorted(sentences[session].items())
        ordered_segments = sorted(segments[session].items())
        if ([i for i, _ in ordered_sentences] != list(range(len(ordered_sentences)))
                or [i for i, _ in ordered_segments] != list(range(len(ordered_segments)))
                or tuple(gloss for _, text in ordered_sentences for gloss in text) != tuple(gloss for _, gloss in ordered_segments)
                or list(sentences[session][index]) != row["glosses"]):
            raise ValueError(f"No exact annotation sequence alignment for {identity}")
        offset = sum(len(glosses) for i, glosses in ordered_sentences if i < index)
        aligned = [f"{session}_glosses/glosses{i:04d}" for i in range(offset, offset + len(row["glosses"]))]
        items.append({"id": identity, "role": row["split"], "template": row["template"], "signer": row["signer"],
                      "segments": [{"id": item, "role": role_of[item], "gloss": isolates[item]} for item in aligned]})
    return signer_roles, items


@torch.inference_mode()
def probe(poses):
    model, labels = load_openhands_gsl_stgcn(CACHE)
    blocks = model.encoder.st_gcn_networks
    observed, results = {}, []
    handles = [blocks[index].register_forward_hook(lambda _, inputs, output, i=index: observed.__setitem__(i, output))
               for index in (3, 6, 9)]
    try:
        with np.load(poses, allow_pickle=False) as cache:
            if cache["split"].item() != "train-prototypes":
                raise ValueError("The feature probe expects the official training prototype cache")
            offsets = cache["offsets"]
            order = np.argsort(np.diff(offsets), kind="stable")
            for row in order[[0, len(order) // 2, -1]]:
                raw = cache["keypoints"][offsets[row]:offsets[row + 1]]
                values = prepare_stgcn_pose(raw)
                observed.clear()
                logits = model(values)
                pooled = model.decoder(observed[9].mean(dim=(2, 3)))
                torch.testing.assert_close(pooled, logits, rtol=0, atol=0)
                if not all(torch.isfinite(tensor).all() for tensor in observed.values()):
                    raise ValueError("Nonfinite temporal activations")
                results.append({"id": str(cache["ids"][row]), "input_shape": list(values.shape),
                                "features_shape_batch_time_channels": {
                                    str(i): list(tensor.mean(dim=3).transpose(1, 2).shape) for i, tensor in observed.items()},
                                "isolated_logits_shape": list(logits.shape), "global_pool_exact": True,
                                "predicted_isolated_label": labels[int(logits[0].argmax())]})
    finally:
        for handle in handles:
            handle.remove()

    timing, stride, receptive_field = {}, 1, 1
    for i, block in enumerate(blocks):
        convolution = block.tcn[2]
        receptive_field += (convolution.kernel_size[0] - 1) * convolution.dilation[0] * stride
        stride *= convolution.stride[0]
        if i in (3, 6, 9):
            timing[str(i)] = {"frame_stride": stride, "receptive_field_input_frames": receptive_field}
    return timing, results


def main():
    torch.set_num_threads(2)
    metadata = CACHE / "gsl_metadata.zip"
    poses = CACHE / "official-train-prototypes-poses.npz"
    plan = json.loads(HARD_PLAN.read_text(encoding="utf-8"))
    signer_roles, overlap = split_overlap(metadata, plan)
    timing, activations = probe(poses)
    report = {
        "scope": "Exact released annotation alignment. Associated isolated and continuous video bytes were not compared. "
                 "Temporal features were probed on native training poses without fitting or continuous predictions.",
        "hashes": {path.relative_to(ROOT).as_posix(): sha(path) for path in
                   (metadata, SPLIT, SUPPLEMENTARY, HARD_PLAN, poses, CACHE / "gsl_stgcn.zip")},
        "native_isolated_signer_counts_by_role": signer_roles,
        "continuous_holdout_segment_alignment": overlap,
        "zero_indexed_graph_block_timing": timing,
        "native_temporal_feature_probes": activations,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
