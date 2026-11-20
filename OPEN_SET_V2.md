# SignBridge Open-Set Recognition V2

This experiment keeps the three supported glosses: ΓΕΙΑ, ΕΥΧΑΡΙΣΤΩ, ΕΝΤΑΞΕΙ.
It does not expand the vocabulary or claim general Greek Sign Language translation.

## Frozen data and evaluation

`data/open-set-v2/plan.locked.json` is the exact reviewed annotation-only plan.
Its byte hash and locking time are in `reports/open-set-v2/protocol-lock.json`.
Its original embedded `proposed_not_locked` field was retained to preserve those
reviewed bytes; the external lock is authoritative. No outcome-based sample
selection was performed.

| Role | Signers | Known | Unsupported |
|---|---|---:|---:|
| Training | 1, 2, 4, 5 | 178 | 100 |
| Validation | 6 | 41 | 30 |
| Final evaluation | 3, 7 | 79 | 50 |

All 129 final clips are unused examples, and their session identifiers differ
from the old test sessions. Their two signer identities were viewed in earlier
experiments. They are not new identities, and recording-day independence is not
established. The final known classes are imbalanced: 9 ΓΕΙΑ, 30 ΕΥΧΑΡΙΣΤΩ,
40 ΕΝΤΑΞΕΙ; report macro F1 and per-class counts with accuracy.

Unsupported categories are disjoint across roles:

- Training: ΦΥΣΙΚΑ, ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ, ΟΚ, ΤΕΛΕΙΑ.
- Validation: ΟΧΙ, ΚΑΛΟ.
- Final: ΝΑΙ, ΚΑΤΑΛΑΒΑΙΝΩ, ΗΜΕΡΑ_ΤΡΕΙΣ.

The raw annotation ΜΕΡΕΣ_ΤΡΕΙΣ is mapped only through its verified official
canonical alias ΗΜΕΡΑ_ΤΡΕΙΣ. ΟΚ and ΕΝΤΑΞΕΙ remain separate official labels;
this is not a new linguistic judgment that their meanings or realizations are
necessarily distinguishable. Project-specific expert review is still absent.

Original POC weights, reports and features are preserved under `artifacts/poc-v2`
and `reports/poc-v2`. Old test clips are diagnostic only and are excluded from
every new split. Existing training/validation material is explicitly marked as
reused; 304 of the 478 clips are new.

## Three candidates and controls

The frozen algorithm specification is `artifacts/open-set-v2/algorithm-protocol.json`.
Candidate A recalibrates the old TCN. B trains the same architecture with the
larger OTHER pool. C adds train-fitted prototype distances to the frozen TCN's
48-dimensional embeddings. The old unchanged configuration and a nearest-centroid
classifier are reference controls, not additional contenders for selection.

Thresholds use validation only. Eligibility requires at least 60% known coverage
and 90% accepted-known accuracy. Eligible settings minimize unknown false
acceptance first, then maximize correct-known acceptance. Rejecting everything
cannot pass. An ineligible candidate's diagnostic fallback is not relabeled as
successful selection. Quality and tracked-pair motion statistics are shared by
Python and TypeScript; static valid signs are not automatically invalidated.

Calibration fits confidence; it does not establish universal unknown detection.
See [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html) for temperature
scaling and [Bendale and Boult, 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)
for the open-set problem. This implementation uses a simple prototype gate;
it is not an implementation of OpenMax or a claim of its theoretical guarantees.

## Reproduction from this project folder

The pinned Python environment and npm dependencies are already installed here.
Start the Vite development server on loopback port 5174. Dataset acquisition is
resumable: `.venv/Scripts/python.exe -X utf8 scripts/openset_dataset.py --acquire`.

Run extraction from `web`, separately for shard 0, 1 and 2:

```powershell
node scripts/extract.mjs --url http://127.0.0.1:5174 --manifest data/open-set-v2/manifest.new.development.json --output data/open-set-v2/features/development-0.jsonl --report reports/open-set-v2/extraction-development-0.json --shards 3 --shard 0 --quiet
```

Use the corresponding shard number in all three arguments. Repeat with
`manifest.new.final.json`, `features/final-0.jsonl`, and `extraction-final-0.json`.
Each clip starts a fresh MediaPipe worker; feature math and detector settings
were unchanged from the corrected POC. Do not edit the source during extraction.

`ml/openset_pack.py` combines the old three isolated feature files with the six
new JSONLs, audits real Python/TypeScript parity, and creates separate train,
validation and final NPZ files. `--help` lists explicit paths. Fitting then reads
only the development NPZ files:

```powershell
.venv/Scripts/python.exe -X utf8 ml/openset_train.py fit --train data/open-set-v2/train.npz --validation data/open-set-v2/val.npz --packing-audit data/open-set-v2/packing-audit.json --frozen artifacts/poc-v2 --output artifacts/open-set-v2
.venv/Scripts/python.exe -X utf8 ml/openset_train.py evaluate --test data/open-set-v2/final_test.npz --frozen artifacts/poc-v2 --output artifacts/open-set-v2
```

These commands refuse to overwrite existing locked selection/evaluation files.
Keep the recorded experiment intact. A new scientific experiment needs a new
output directory and an explicit protocol, and these final outcomes are now
known once evaluated.

## Webcam and browser evidence

`Start-Webcam-Check.cmd` opens a separate local recording helper on port 5175.
The user must grant camera access, choose a trial, review it and explicitly save.
It records no audio and displays no model prediction during collection.
Confirmed clips remain in `data/open-set-v2/webcam`, with provenance and hashes;
they are a separate observational holdout, never threshold-fitting data.

The user could not complete this session's recordings. Real human webcam
performance must therefore remain untested. Brightness, translation and mirror
stress checks derived from GSL images are transformed-data diagnostics, not
physical lighting/camera tests. In particular, mirror transforms are not
assumed to preserve linguistic meaning.

Twelve MP4 fixtures were selected by label, signer and source path before any
model outcomes. Their actual encoding and frame-count audit is in
`data/open-set-v2/clips/mapping.json`. These small integration checks do not
replace the 129-example final evaluation.
