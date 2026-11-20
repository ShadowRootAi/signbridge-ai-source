# SignBridge V3 — reproducible diagnosis and bounded centroid experiment

The Greek result is in [the technical report](reports/v3/TECHNICAL_REPORT.md).
No new large CNN training, production-model replacement, human webcam trial or
public deployment was performed. The existing B model remains a frozen research
reference. The newly fitted augmented pooled centroid is not a validated product
replacement: it rejects all11 validation ΓΕΙΑ clips.

## Evidence and limitations

- Ten ΚΑΤΑΛΑΒΑΙΝΩ failures retain consistent source labels and frame counts.
  Contact sheets, raw landmark overlays, trajectories, shape and embedding plots
  are in `reports/v3`.
- Quality1 can hide active-hand loss because the resting hand satisfies the
  existing pose-plus-either-hand gate. Seven of ten failures nevertheless retain
  complete right-hand presence, so dropout is not the complete explanation.
- Frozen temporal ablations cover all129 exposed V2 final examples, plus the
  focal19 subset. These are sensitivity tests, not fresh benchmark results.
- Added75 real development clips,1364 original RGB frames:20train+5validation
  per ΚΑΤΑΛΑΒΑΙΝΩ, ΝΑΙ, ΗΜΕΡΑ_ΤΡΕΙΣ. They were selected after observing V2
  category failures. Train338 (people1/2/4/5); validation86 (person6).
- All98 eligible whole-single-gloss ΓΕΙΑ clips were previously exposed.
  A new complete independent three-class test is unavailable in this pool.
- Normalization and three known prototypes remain unchanged. Pooled versus
  category-specific negative prototypes are fitted from train only. Separate
  negatives collapse to one nearest-negative OTHER logit for a fair four-way
  softmax comparison. Validation temperatures/gates are frozen before regression.
- Regression uses the same129 V2 examples. Augmented pool:71/79 known accepted
  correctly and0/50 unknown falsely accepted; old baseline77/79 and12/50.
  This adaptive gain does not establish new-user or new-category generalization.

## Reproduction

Run from the project root with the existing Python venv and pinned web packages.
Scripts preserve original artifacts and refuse conflicting locks or repeated
fitting/evaluation writes. Do not remove locks to tune against exposed outcomes.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\v3_data_diagnosis.py
.\.venv\Scripts\python.exe -X utf8 scripts\v3_data_diagnosis.py --tracking-overlay
.\.venv\Scripts\python.exe -X utf8 ml\v3_diagnose.py
.\.venv\Scripts\python.exe -X utf8 scripts\v3_data_diagnosis.py --acquire-targeted-development
```

Data selection is locked in `data/v3/targeted-development.plan.locked.json`.
Acquisition downloads only selected RGB ranges, validates archive CRC, image
decode, official frame intervals, hashes and disjointness from old547 clips.
No depth archive is used.

Start the scientific extraction server from `web`:

```powershell
node node_modules\vite\bin\vite.js --host 127.0.0.1 --port 5173 --strictPort
```

From the project root, run the extraction command for shards0,1,2, changing the
output/report/shard suffix together. Each shard contains25 genuine clips.

```powershell
node web/scripts/extract.mjs --url http://127.0.0.1:5173 --manifest data/v3/manifest.json --output data/v3/features/targeted-0.jsonl --report reports/v3/extraction-targeted-0.json --shard 0 --shards 3 --quiet
```

The following phases were executed once in this order:

```powershell
.\.venv\Scripts\python.exe -X utf8 ml\v3_centroid_experiment.py lock
.\.venv\Scripts\python.exe -X utf8 ml\v3_centroid_experiment.py pack --manifest data/v3/manifest.json --plan data/v3/targeted-development.plan.locked.json --jsonl data/v3/features/targeted-0.jsonl data/v3/features/targeted-1.jsonl data/v3/features/targeted-2.jsonl
.\.venv\Scripts\python.exe -X utf8 ml\v3_centroid_experiment.py fit
.\.venv\Scripts\python.exe -X utf8 ml\v3_centroid_experiment.py regression
.\.venv\Scripts\python.exe -X utf8 ml\v3_export_centroids.py
.\.venv\Scripts\python.exe -X utf8 ml\v3_audit.py
node web/scripts/qa-v3-centroids.mjs --url http://127.0.0.1:5173
.\.venv\Scripts\python.exe -X utf8 scripts\build_v3_report.py
```

The fit command reads only train/validation. Regression cannot run until the fit
is locked. Prototype and gate source files, reused train/validation NPZs and the
new plan/manifest are bound into the algorithm lock. Original V2 artifacts stay
intact. New browser models live under `artifacts/v3-centroid/browser`; none are
copied to the production `web/public/models` directory.

The18 new Python/TypeScript raw preprocessing fixtures match exactly. All344
validation model/gate parity cases and48 actual MP4 runs passed. Scientific QA
records repeated old examples explicitly; it never presents them as webcam or
independent final evaluation. Actual human webcam data remain unavailable.

Four mathematical tests cover manual distances, equivalent single-background
behavior, negative-order invariance and duplicate-negative softmax mass:

```powershell
Push-Location ml
..\.venv\Scripts\python.exe -X utf8 -m unittest test_v3_centroid_experiment -v
Pop-Location
```

## Publication preparation

See [deployment instructions](web/deployment/README.md) and
`signbridge-publication-review.zip`. This is the unchanged frozen B research
app, visibly reporting24/50 unknown errors; no public URL exists. Upload and
webcam share preprocessing; camera has countdown/autostop and no automatic
speech. A real person using a webcam has not completed a trial.

The package excludes raw dataset media, capture helpers and local paths, has
exact dependency license texts and proposed hosting headers, and passed real
MP4 inference under its CSP. `Start-SignBridge.cmd` / `.ps1` now verify exact
HTML/config/model identity before reusing a server. Incompatible servers are
reported without stopping them. Review servers were stopped after QA.

Do not promote augmented_pool solely because of0/50 exposed regression errors.
Validate active-hand tracking, per-class coverage and independent new captures
before choosing a replacement or expanding the supported vocabulary.
