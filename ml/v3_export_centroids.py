"""Export frozen research centroids without training, recalibration or deployment.

Run after artifacts/v3-centroid/fit.locked.json exists. Every parity input is
from the validation split, never from the exposed regression split.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from openset_core import motion_metrics, score_gate
from openset_train import load_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/v3-centroid"
OUTPUT = ARTIFACTS / "browser"
ATOL, RTOL = 1e-6, 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def numpy_logits(x, source):
    # Independent transcription of the frozen formula, not a new fit.
    z = np.clip((x - source["mean"][None, None, :]) / source["std"][None, None, :], -8, 8)
    z = (z * source["keep"][None, None, :]).reshape(len(x), -1)
    distances = np.stack([np.mean((z - center) ** 2, axis=1) for center in source["centroids"]], axis=1)
    return np.concatenate((-distances[:, :3], -np.min(distances[:, 3:], axis=1, keepdims=True)), axis=1)


class FrozenCentroids(nn.Module):
    def __init__(self, source):
        super().__init__()
        for name in ("mean", "std", "keep", "centroids"):
            self.register_buffer(name, torch.from_numpy(source[name].copy()))

    def forward(self, landmarks):
        z = (torch.clamp((landmarks - self.mean) / self.std, -8, 8) * self.keep).flatten(1)
        distances = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).mean(dim=2)
        return torch.cat((-distances[:, :3], -distances[:, 3:].amin(dim=1, keepdim=True)), dim=1)


def decisions_equal(expected, actual, description):
    for key in ("accepted", "predicted"):
        if not np.array_equal(expected[key], actual[key]):
            indices = np.flatnonzero(expected[key] != actual[key]).tolist()
            raise AssertionError(f"{description}: exact {key} mismatch at validation rows {indices}")
    if set(expected["reasons"]) != set(actual["reasons"]):
        raise AssertionError(f"{description}: reason schema differs")
    for key in expected["reasons"]:
        if not np.array_equal(expected["reasons"][key], actual["reasons"][key]):
            raise AssertionError(f"{description}: exact rejection reason {key} differs")


def main():
    if (OUTPUT / "export.locked.json").exists():
        raise ValueError("Browser export already locked; do not silently overwrite research evidence.")
    fitted = read(ARTIFACTS / "fit.locked.json")
    protocol = read(ARTIFACTS / "protocol.locked.json")
    protocol_hashes = {name.replace("\\", "/"): digest for name, digest in protocol["file_hashes"].items()}
    if fitted["selection_uses_old_test"] is not False:
        raise ValueError("Expected validation-only parameter fitting, with adaptive-design caveat.")
    if fitted["protocol_sha256"] != sha(ARTIFACTS / "protocol.locked.json"):
        raise ValueError("Protocol changed after fit lock.")
    for filename, digest in protocol["file_hashes"].items():
        if sha(ROOT / filename) != digest:
            raise ValueError(f"Locked source changed: {filename}")
    data = load_split(ROOT / "data/v3/val.npz", "val")
    packing = read(ROOT / "data/v3/packing-audit.json")
    if sha(ROOT / "data/v3/val.npz") != packing["outputs"]["val"]["sha256"]:
        raise ValueError("Validation data changed after packing audit.")
    methods = [(name, ARTIFACTS / name / "centroids.npz", ARTIFACTS / name / "config.json", hashes)
               for name, hashes in fitted["candidates"].items()]
    old = ROOT / "artifacts/open-set-v2"
    methods.append(("frozen_v2_centroid", old / "reference-centroid.npz", old / "reference-centroid-config.json", {
        "model_sha256": protocol_hashes["artifacts/open-set-v2/reference-centroid.npz"],
        "config_sha256": protocol_hashes["artifacts/open-set-v2/reference-centroid-config.json"]}))
    torch.set_num_threads(2)
    diagnostics = motion_metrics(data.x)
    summary = {"role": "validation_export_parity_only", "developmentOnly": True,
               "deploymentChanged": False, "humanWebcamTested": False,
               "fitLockSha256": sha(ARTIFACTS / "fit.locked.json"),
               "exporterSha256": sha(Path(__file__)), "validationSha256": sha(ROOT / "data/v3/val.npz"),
               "numericAtol": ATOL, "numericRtol": RTOL, "methods": {}}
    for name, npz, config_path, hashes in methods:
        if sha(npz) != hashes["model_sha256"] or sha(config_path) != hashes["config_sha256"]:
            raise ValueError(f"Frozen {name} artifact identity changed.")
        with np.load(npz, allow_pickle=False) as values:
            source = {key: values[key].copy() for key in ("mean", "std", "keep", "centroids")}
        for key, values in source.items():
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"{name}/{key}: expected finite frozen Float32 values.")
        if any(source[key].shape != (224,) for key in ("mean", "std", "keep")):
            raise ValueError("Frozen normalization must have224 channels.")
        if source["centroids"].ndim != 2 or source["centroids"].shape[1] != 48 * 224 or len(source["centroids"]) < 4:
            raise ValueError("Expected3 known prototypes and one or more negative prototypes.")
        if np.any(source["std"] <= 0) or not np.isin(source["keep"], [0, 1]).all():
            raise ValueError("Invalid frozen normalization.")
        config = read(config_path)
        browser_config = deepcopy(config)
        browser_config["output"].pop("embeddingName", None)
        if browser_config.get("prototypeGate") is not None:
            raise ValueError("Raw centroid logits must not use the TCN embedding prototype gate.")
        if len(config["labels"]) != 3 or config["backgroundIndex"] != 3:
            raise ValueError("Unexpected known/background output order.")
        directory = OUTPUT / name; directory.mkdir(parents=True, exist_ok=True)
        model = FrozenCentroids(source).eval()
        np_logits = numpy_logits(data.x, source)
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(data.x)).numpy()
        np.testing.assert_allclose(torch_logits, np_logits, atol=ATOL, rtol=RTOL)
        model_path = directory / "model.onnx"
        torch.onnx.export(model, (torch.from_numpy(data.x[:1]),), str(model_path),
                          input_names=["landmarks"], output_names=["logits"],
                          dynamo=True, opset_version=18, external_data=False)
        onnx.checker.check_model(str(model_path))
        session_options = ort.SessionOptions(); session_options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 48, 224] or [v.name for v in session.get_outputs()] != ["logits"]:
            raise ValueError("Incorrect exported I/O contract.")
        ort_logits = np.concatenate([session.run(["logits"], {"landmarks": x[None]})[0] for x in data.x])
        np.testing.assert_allclose(ort_logits, np_logits, atol=ATOL, rtol=RTOL)
        scored = score_gate(np_logits, data.quality, diagnostics["motionBodyRms"], config)
        decisions_equal(scored, score_gate(torch_logits, data.quality, diagnostics["motionBodyRms"], config), f"{name}: PyTorch")
        decisions_equal(scored, score_gate(ort_logits, data.quality, diagnostics["motionBodyRms"], config), f"{name}: ONNX CPU")
        cases = []
        for i in range(len(data.x)):
            cases.append({"id": str(data.source_id[i]), "rawLabel": str(data.raw_label[i]), "signer": str(data.signer[i]),
                          "split": "val", "x": data.x[i].tolist(), "quality": float(data.quality[i]),
                          "numpy_logits": np_logits[i].tolist(), "pytorch_logits": torch_logits[i].tolist(), "onnx_logits": ort_logits[i].tolist(),
                          "diagnostics": {key: value[i].tolist() for key, value in diagnostics.items()},
                          "accepted": bool(scored["accepted"][i]), "predicted": int(scored["predicted"][i]),
                          "confidence": float(scored["confidence"][i]),
                          "reasons": [key for key, value in scored["reasons"].items() if value[i]]})
        write(directory / "browser-config.json", browser_config)
        write(directory / "validation-parity.json", {"role": "validation_only", "cases": cases, "numericAtol": ATOL, "numericRtol": RTOL, "exactDecisionsRequired": True})
        entry = {"sourceCentroidsSha256": sha(npz), "sourceConfigSha256": sha(config_path),
                 "modelSha256": sha(model_path), "browserConfigSha256": sha(directory / "browser-config.json"),
                 "fixtureSha256": sha(directory / "validation-parity.json"), "modelBytes": model_path.stat().st_size,
                 "prototypeCount": len(source["centroids"]), "validationCases": len(cases),
                 "maxNumpyPytorchLogitError": float(np.abs(np_logits - torch_logits).max()),
                 "maxNumpyOnnxLogitError": float(np.abs(np_logits - ort_logits).max()),
                 "exactDecisionAgreement": True, "browserConfigOnlyChange": "omit output.embeddingName"}
        summary["methods"][name] = entry
        print(json.dumps({"method": name, **entry}), flush=True)
    write(OUTPUT / "export.locked.json", summary)
    print(json.dumps({"status": "passed", "methods": len(summary["methods"]), "validationCasesPerMethod": len(data.x)}))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
