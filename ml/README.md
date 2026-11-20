# Landmark model proof of concept

## Current open-set v2 experiment

The current research comparison lives in `artifacts/open-set-v2`; the original
corrected POC is preserved in `artifacts/poc-v2` and `reports/poc-v2`. The
algorithm grid and dataset plan were frozen before fitting. Separate files hold
training, validation and final examples. The final clips are new, but test
people 3 and 7 were seen in earlier diagnostics; they are not new identities.

Candidate B (TCN retrained with diverse OTHER examples) was selected on
validation before final evaluation. On the final set it accepts 59/79 known
examples, all correctly, but also falsely accepts 24/50 unsupported examples.
The full-feature centroid reference performs better on this final set; it was
not a selection candidate and must not silently replace B after test inspection.
All methods remain experimental. Human webcam performance is untested.

The following is the recorded reproduction sequence, run from `signbridge-poc`
with its configured virtual environment. Existing locked selections and final
evaluations deliberately refuse accidental overwrites/repeated evaluation.
Do not delete those locks to seek a better test result. A genuinely new research
run requires separate artifacts and an appropriate untouched evaluation set.

```powershell
$extractionFiles = @(
  'data/features/extracted-isolated-0.jsonl',
  'data/features/extracted-isolated-1.jsonl',
  'data/features/extracted-isolated-2.jsonl',
  'data/open-set-v2/features/development-0.jsonl',
  'data/open-set-v2/features/development-1.jsonl',
  'data/open-set-v2/features/development-2.jsonl',
  'data/open-set-v2/features/final-0.jsonl',
  'data/open-set-v2/features/final-1.jsonl',
  'data/open-set-v2/features/final-2.jsonl'
)
& '.venv/Scripts/python.exe' -X utf8 ml/openset_protocol.py --output artifacts/open-set-v2/algorithm-protocol.json --plan-lock data/open-set-v2/plan.locked.json
& '.venv/Scripts/python.exe' -X utf8 ml/openset_pack.py --manifest data/open-set-v2/manifest.json --plan-lock data/open-set-v2/plan.locked.json --jsonl $extractionFiles --frozen-features artifacts/poc-v2/features.npz --feature-code web/src/features.ts --output data/open-set-v2
& '.venv/Scripts/python.exe' -X utf8 ml/openset_train.py fit --train data/open-set-v2/train.npz --validation data/open-set-v2/val.npz --packing-audit data/open-set-v2/packing-audit.json --frozen artifacts/poc-v2 --output artifacts/open-set-v2
```

Only after `selection.locked.json` exists does the separate final command read
the sealed final tensors. It checks final data, model, config, algorithm-protocol
and gate-code hashes, evaluates the three frozen candidates and references, and
retains the validation-selected winner regardless of final rankings:

```powershell
& '.venv/Scripts/python.exe' -X utf8 ml/openset_train.py evaluate --test data/open-set-v2/final_test.npz --frozen artifacts/poc-v2 --output artifacts/open-set-v2
& '.venv/Scripts/python.exe' -X utf8 ml/openset_final_diagnose.py --data data/open-set-v2 --artifacts artifacts/open-set-v2 --output reports/open-set-v2/final-failure-analysis.json
& '.venv/Scripts/python.exe' -X utf8 ml/openset_check.py --output reports/open-set-v2/ml-test-summary.json
```

`openset_diagnose.py` operates only on already-seen frozen POC data. It includes
temporal reversal/shuffling, static means, feature masking and train-only
feature baselines. Such altered inputs are distribution-shift diagnostics, not
causal proof. The post-final diagnostic uses saved final predictions for its
error summary and restricts additional temporal ablations to development data.

Gate mathematics are implemented in `openset_core.py`. The C prototype option
uses a frozen 48-dimensional embedding, training-known scaling/prototypes and
training leave-one-out distance quantiles; validation chooses the gate. Quality
and motion gates include a disabled-motion option because legitimate signs can
be static. Each exported candidate includes validation-only JSON fixtures for
independent JavaScript numerical and exact-decision parity checks.

## Archived three-sign POC workflow

This folder trains a small Temporal CNN on real extracted landmark sequences. It does not download videos, invent labels, or generate synthetic training observations. The tests use synthetic arrays only to check leakage detection and metric accounting.

The NPZ contract is `x: float32[N,T,F]`, `y: int[N]` (known classes `0..K-1`, unknown `-1`), `signer: str[N]`, `source_id: str[N]`, `split: str[N]` (`train`, `val`, `test`), and `classes: str[K]`. Unknown examples require `raw_label: str[N]` containing the actual semantic unknown category. Optional `quality: float[N]` must be in `[0,1]`; it defaults to one. Optional scalar `metadata_json` contains the feature schema and extraction provenance. Arrays must load with `allow_pickle=False`.

Signer identities, source-video identities, and unknown semantic categories must be disjoint between splits. Every known class must occur in every split. The script fails on violations rather than reshuffling examples. Unknown training examples train an explicit background output; otherwise the output has only known classes. Training-only normalization is embedded inside the exported graph: continuous standard deviations have a fixed `0.01` floor, training-invariant continuous dimensions (`std < 1e-6`) are ignored, and the four schema-v1 presence flags retain their native `[0,1]` scale. Standardized values are clipped to `[-8,8]`. These rules are fixed before examining any test results and also apply to the centroid baseline. Model selection, temperature scaling, and rejection thresholds use validation data only. Threshold search maximizes correct-known acceptance subject to declared validation targets; these targets are not claims about final test performance. Config marks that fitting was performed (`calibrated: true`) and records the validation split and sample counts; this does not assert universal probability calibration or unknown-sign detection.

Run from the `signbridge-poc` project root with the configured Python environment.
The following commands reproduce the corrected clip-isolated pipeline; they
must not be interpreted as a fresh untouched test after earlier inspection:

```powershell
& '.venv/Scripts/python.exe' -X utf8 ml/pack_features.py --manifest data/manifest.json --jsonl data/features/extracted-isolated-0.jsonl data/features/extracted-isolated-1.jsonl data/features/extracted-isolated-2.jsonl --output data/features.npz
& '.venv/Scripts/python.exe' -X utf8 ml/train.py --features data/features.npz --output artifacts --model-version signbridge-poc-tcn-v2-isolated --evaluation-context corrected_pipeline_repeated_holdout
& '.venv/Scripts/python.exe' -X utf8 ml/analyze_failure.py --features data/features.npz --artifacts artifacts --output reports/failure_analysis.json
& '.venv/Scripts/python.exe' -X utf8 -m unittest discover -s ml -p "test_*.py" -v
```

The three JSONL files each contain 81 records. Their extraction provenance is
recorded in `reports/extraction-0.json`, `extraction-1.json`, and
`extraction-2.json`. The original v1 model, metrics and features are archived in
`artifacts/v1`, `reports/v1`, and `data/features-v1.npz`. Keep those originals;
do not rerun training merely to seek a better test result.

The packer accepts JSONL records `{id, x, quality, raw?}`, matches identities and targets to the manifest, and compares any raw frames with the independent Python preprocessing implementation. `raw` is either an array of landmark frames or `{frames: [...]}`. It rejects incomplete extraction by default and never silently excludes low-quality clips. The separate preprocessing report records exactly how many real fixtures were checked. No raw fixtures means `not_tested_no_raw_fixtures`, never a claimed pass.

Dependencies: NumPy, PyTorch, ONNX, ONNX Script, ONNX Runtime. The baseline is a dependency-free nearest-centroid model using the same normalized, time-ordered features. The classifier uses Conv1d, ReLU, averaging, and a linear output, and exports a fixed `[1,T,F]` input.

Artifacts include ONNX and PyTorch weights, browser config, labels, all-example predictions, training history and metrics, and numeric ONNX/PyTorch parity fixtures in NPZ and JSON. The test report separates forced known-class prediction from selective decisions. Always show accepted accuracy alongside known coverage, known rejection, and unknown false acceptance. A reject-all result has zero coverage and undefined accepted accuracy, not perfect performance.

The report's Python CPU classifier timing excludes tracking, video processing, and browser overhead. Browser parity and genuinely new-user webcam evaluation remain separate checks and must not be represented as completed by this script.

An engineering validity gate requires quality of at least `0.25`: a quarter of the original sampled frames must contain valid shoulders and at least one detected hand. Validation can raise this threshold, never lower it. Zero-hand and empty-tracking sequences therefore cannot be accepted even if the classifier produces a high score. This floor is a declared input-validity policy, not an empirically established guarantee of correct recognition; partially tracked sequences above the floor can still be wrong.
