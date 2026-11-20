"""Export the official Greek OpenHands ST-GCN and verify native pose parity."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch

from openhands_gsl_stgcn import STGCN_ARCHIVE_SHA256, load_openhands_gsl_stgcn, prepare_stgcn_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/cache/openhands-gsl"))
    parser.add_argument("--poses", type=Path)
    args = parser.parse_args()
    poses_path = args.poses or args.cache / "official-val-poses.npz"
    torch.set_num_threads(1)
    model, labels = load_openhands_gsl_stgcn(args.cache)
    with np.load(poses_path, allow_pickle=False) as dataset:
        keypoints, offsets, ids = (dataset[name] for name in ("keypoints", "offsets", "ids"))
        split = str(dataset["split"].item())
    if (keypoints.ndim != 3 or keypoints.shape[1:] != (75, 3) or len(offsets) != len(ids) + 1
            or offsets[0] != 0 or offsets[-1] != len(keypoints) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Invalid native pose archive index")
    lengths, first = np.unique(np.diff(offsets), return_index=True)
    if len(lengths) < 6:
        raise ValueError("Need at least six distinct native clip lengths to verify the dynamic axis")
    selected = first[np.linspace(0, len(first) - 1, 6, dtype=int)]
    clips = [(str(ids[i]), prepare_stgcn_pose(keypoints[offsets[i]:offsets[i+1]])) for i in selected]

    with TemporaryDirectory(prefix="stgcn-export-", dir=args.cache) as temp:
        model_path = Path(temp) / "stgcn.onnx"
        config_path = Path(temp) / "stgcn-onnx.json"
        with torch.no_grad():
            torch.onnx.export(
                model, (clips[len(clips) // 2][1],), model_path,
                input_names=["landmarks"], output_names=["logits"],
                dynamic_axes={"landmarks": {2: "time"}},
                opset_version=17, dynamo=False,
            )
        onnx.checker.check_model(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        errors = []
        evidence = []
        for identity, clip in clips:
            with torch.inference_mode():
                expected = model(clip).numpy()
            actual, = session.run(["logits"], {"landmarks": clip.numpy()})
            if expected.shape != (1, len(labels)) or not np.isfinite(expected).all():
                raise ValueError(f"Invalid PyTorch logits for {identity}")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
            expected_index, actual_index = int(expected.argmax()), int(actual.argmax())
            if expected_index != actual_index:
                raise ValueError(f"Top-1 parity failed for {identity}")
            error = float(np.max(np.abs(expected - actual)))
            errors.append(error)
            evidence.append({"id": identity, "frames": int(clip.shape[2]), "top1": labels[actual_index], "max_abs_logit_error": error})
        del session
        with model_path.open("rb") as handle:
            model_hash = file_digest(handle, "sha256").hexdigest()
        config = {
            "architecture": "OpenHands Greek ST-GCN",
            "labels": labels,
            "input": "landmarks",
            "input_shape": [1, 2, "time", 27],
            "output": "logits",
            "opset": 17,
            "source_checkpoint_sha256": STGCN_ARCHIVE_SHA256,
            "model_sha256": model_hash,
            "native_pose_split": split,
            "native_pose_parity": evidence,
            "max_abs_logit_error": max(errors),
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model_path.replace(args.cache / model_path.name)
        config_path.replace(args.cache / config_path.name)
    print(json.dumps({"clips": len(clips), "frames": [entry["frames"] for entry in evidence], "max_abs_logit_error": max(errors), "model_sha256": model_hash}))


if __name__ == "__main__":
    main()
