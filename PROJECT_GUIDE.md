# SignBridge AI — limited research web application

## Local Ollama assistant

On Windows, start Ollama with `qwen3:4b` installed and open
`Start-Local-Assistant.cmd`. **Τοπικός βοηθός** opens the assistant inside the
existing result card. The local edition sends a typed message and, when
available, the latest recognition result to Ollama over loopback.

**Αποθήκευση διόρθωσης** saves an explicit original and replacement. When a
later message contains that phrase, the assistant retrieves the correction
from `%LOCALAPPDATA%/SignBridge/assistant/memory.sqlite3`. Use **Εκκαθάριση
μνήμης** in the dialog to remove saved corrections. These corrections are text
memory. Training the visual recognizer is a separate workflow below.

```powershell
ollama pull qwen3:4b
.\.venv\Scripts\python.exe -X utf8 scripts/package_video_library.py words-200 mixed-2512 coalesced-2331 --minimum 3000 --assistant --frame-offset
node web/scripts/qa-local-assistant.mjs
```

### Teach Me from a video correction

In the local assistant edition, analyze a short clip on `#/try` and select
**Διόρθωσε στο Teach Me** under the result. Enter the correct Greek meaning,
add two more training clips of that sign, two fresh clips to check recognition,
and two clips of different signs to check rejection. Use separate recordings.
**Έλεγχος και αποθήκευση** activates the sign after all four checks pass.

The browser compares new motion to the saved examples by aligning landmark
sequences across time. A unique accepted personal match appears in the regular
recognition result. Teach Me keeps the derived motion templates in the same
browser profile and local origin. The original videos are processed locally.
The profile stores only the training templates, label and acceptance threshold.
Delete an entry under **Τα νοήματά σου** to remove its influence. The method is
for single signs and primarily for the person who supplied the examples. The
Ollama assistant uses the resulting text. Teach Me fits its own motion templates.

Replay the restart test with `node web/scripts/qa-teach-restart.mjs` after
building the local assistant edition.

## Local WAICY judge demo

On Windows, run `Start-Judge-Demo.cmd`. Select **Εντάξει**, **Παγωμένο καρέ**,
**Μαύρη εικόνα** or **Εκτός λεξιλογίου**, then **Ανάλυση βίντεο**. The decision
trace shows the actual tracking coverage, classifier execution and acceptance
rule. On a recovered result, expand **Έλεγχος δεύτερου μοντέλου** to see the
runtime scores. The local 30 fps boundary correction can allow **Εντάξει**
to pass directly. The Library tab opens the local 3,000-video catalog.

The frozen and black clips are controlled variants of the same CERTH clip.
The linked report documents what happened on these examples and how to rebuild
the separate local edition.

[Engine changes and browser comparisons](reports/openhands-engine/README.md) ·
[Fixture sources](reports/judge-robustness/README.md) ·
[Final judge walkthrough and training comparison](reports/judge-final/README.md) ·
[AI development experiments and release checks](reports/judge-ai-development/README.md).

## Local video library and isolated recognition experiment

On Windows, open `Start-Video-Library.cmd` to search the local collection of 3,000
attributed Greek Sign Language videos. It includes 1,566 single-word references
and 1,434 short phrases or educational excerpts. The package audit records the
video count and source totals. The bundled recognition preview supports 200
isolated signs.

[Library audit](publication-video-library/library-audit.json) ·
[Chrome library QA](reports/video-library/browser-qa.json) ·
[Recognition preview smoke test](reports/video-library/recognition-smoke.json).

The separate 309-class research model learned a similarity metric across signers.
On the previously examined signer-3 isolated poses it improved first-choice
accuracy from 79.69% to 81.26%. Correct accepted results rose from 1,194 to 1,392
and incorrect accepted results rose from 12 to 23. Webcam and unknown-sign
accuracy remain unmeasured for this model. The Chrome WASM check covers the
exported metric branch with prepared input tensors and reference fusion scores.

[Model benchmark](reports/isolated-cross-signer-metric/benchmark.json) ·
[Chrome ONNX parity](reports/isolated-cross-signer-metric/browser-onnx-parity.json).

### Library recordings for video-to-text research

The [coverage audit](reports/video-library/recognition-coverage.json) checks
the catalog against the official Elementary23 workbook and archive indexes.
One catalog clip per text is enough for lookup. Of 1,566 word entries, 134
have indexed clips with the same Greek transcription from at least three
documented signer IDs. A shared transcription is weak sign-level supervision.

The [scale pilot benchmark](reports/recognition-scale-pilot/benchmark.json)
uses 67 actual Elementary23 studio MP4s, with signer IDs held apart for fit,
threshold calibration and test. The validation and test sets use different
out-of-vocabulary words. The experiment uses Chrome's landmark extractor
and compares a temporal CNN with nearest class centroids. Seven of the eight
planned words had eligible training videos from both fit signers. On the
held recordings of those classes, the CNN ranked 9 of 12 known words first,
accepted 4 correctly, and falsely accepted 1 of 6 out-of-vocabulary clips.
The weights and thresholds are research artifacts outside the app release.

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_video_recognition_coverage.py
.\.venv\Scripts\python.exe -B scripts/acquire_recognition_scale_pilot.py --acquire
$pilot = '.runtime-tmp/recognition-scale-pilot'
node web/scripts/extract-recognition-scale-pilot.mjs "$pilot/manifest.json" "$pilot/features.jsonl"
.\.venv\Scripts\python.exe -B ml/evaluate_recognition_scale_pilot.py fit "$pilot/manifest.json" "$pilot/features.jsonl"
.\.venv\Scripts\python.exe -B ml/evaluate_recognition_scale_pilot.py evaluate "$pilot/manifest.json" "$pilot/features.jsonl"
```

The [V2 benchmark](reports/recognition-scale-v2/benchmark.json) uses 118
hash-verified Elementary23 MP4s for ten candidates. Fit includes 14 real OTHER
clips and uses different unknown words for calibration and final scoring.
The 31 final MP4s are outside the V1 cohort, with signer IDs held out from
fitting and calibration. Six of the ten targets have fresh positive test
examples. Validation selected the temporal hand prototype branch. It accepted
four correct, one incorrect known, and three unknown clips in the final test.
On the matched six known and sixteen unknown fresh clips, V1 and V2 each
accepted one correct result. V1 made one false acceptance and V2 made four.
These results keep V2 outside the application release.

[Chrome ONNX parity](reports/recognition-scale-v2/browser-onnx-parity.json)
compares all 31 decisions with PyTorch using the extracted landmark tensors.
Both runtimes made the same decisions. This measures model execution, separate
from webcam or video decoder generalization.

```powershell
.\.venv\Scripts\python.exe -B scripts/acquire_recognition_scale_v2.py --acquire
$scale = '.runtime-tmp/recognition-scale-v2'
node web/scripts/extract-recognition-scale-pilot.mjs "$scale/manifest.json" "$scale/features.jsonl" --reuse "$pilot/manifest.json" "$pilot/features.jsonl"
.\.venv\Scripts\python.exe -B ml/evaluate_recognition_scale_v2.py fit "$scale/manifest.json" "$scale/features.jsonl"
.\.venv\Scripts\python.exe -B ml/evaluate_recognition_scale_v2.py evaluate "$scale/manifest.json" "$scale/features.jsonl"
node web/scripts/qa-recognition-scale-v2-onnx.mjs artifacts/recognition-scale-v2/model.onnx "$scale/browser-fixture.json" reports/recognition-scale-v2/browser-onnx-parity.json
```

### Phone recordings for V2

The [GSLW audit](reports/recognition-phone-adaptation/source-audit.json)
checks the complete official annotations against the ten V2 words. It found
no whole-file positive. Occurrences inside longer utterances require reviewed
time boundaries before they can be used for isolated-sign training. The
project's [phone recording guide](docs/PHONE_RECORDINGS.md) lists the ten
prompts and a signer-disjoint collection plan.

The [phone adaptation benchmark](reports/recognition-phone-adaptation/benchmark.json)
compares frozen V2 and a fit using real phone OTHER examples. Its phone test
contains three out-of-vocabulary clips from two held GSLW signer IDs. All
three full clips failed the tracking guard. Two windows from those same
sources pass the guard, and both models reject both windows. This replay
uses the numeric annotation bounds as seconds for a tracking diagnostic.
The separate external diagnostic uses
three author-timed Sartinas `ΕΓΩ` segments recorded with a Ximea camera.
The previously scored Elementary23 test is a retrospective regression check.

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_recognition_phone_positives.py
.\.venv\Scripts\python.exe -B scripts/acquire_recognition_phone_negatives.py --acquire
.\.venv\Scripts\python.exe -B scripts/prepare_recognition_phone_studio_diagnostic.py
.\.venv\Scripts\python.exe -B scripts/prepare_recognition_phone_windows.py
$phone = '.runtime-tmp/recognition-phone-negatives'
$diagnostic = '.runtime-tmp/recognition-phone-studio-diagnostic'
$windows = '.runtime-tmp/recognition-phone-windows'
node web/scripts/extract-recognition-scale-pilot.mjs "$phone/manifest.json" "$phone/features.jsonl"
node web/scripts/extract-recognition-scale-pilot.mjs "$diagnostic/manifest.json" "$diagnostic/features.jsonl"
node web/scripts/extract-recognition-scale-pilot.mjs "$windows/manifest.json" "$windows/features.jsonl"
.\.venv\Scripts\python.exe -B ml/evaluate_recognition_phone_adaptation.py fit "$phone/manifest.json" "$phone/features.jsonl"
.\.venv\Scripts\python.exe -B ml/evaluate_recognition_phone_adaptation.py evaluate "$phone/manifest.json" "$phone/features.jsonl" --external-manifest "$diagnostic/manifest.json" --external-features "$diagnostic/features.jsonl" --window-manifest "$windows/manifest.json" --window-features "$windows/features.jsonl"
```

## V4 product and raw tracking guard

Latest work: **V4 product and raw tracking guard**.
[Greek delivery report](reports/v4/TECHNICAL_REPORT.md) ·
[Static publication instructions](web/deployment/V4_DEPLOYMENT.md).

The React/TypeScript application has five Greek screens: Home, Sign to Text,
Text to Sign, Try the AI, and About the AI. Camera and uploaded MP4 inference
run locally with MediaPipe and the frozen ONNX B model. Speech requires a
separate explicit confirmation and an available Greek voice. The reverse
direction is exact lookup of three licensed, originally annotated GSL videos,
not sentence translation or word concatenation.

V4 checks original landmarks **before interpolation**: at least four frames,
with pose and both hands together in at least 90%. Missing active-hand data can
no longer be hidden behind the old pose-plus-either-hand score. This is an
engineering completeness rule, not an accuracy guarantee; it can also reject
a valid one-handed sign whose resting hand is out of view. The trained model,
feature representation and learned thresholds have not changed.

The baseline's rejection of all 11 validation ΓΕΙΑ examples predates the V3
negative augmentation. ΓΕΙΑ is their nearest known centroid, but its scores
remain below acceptance thresholds; augmentation lowers them further. Raw
frames show substantial detector-right dropout. This is a diagnosis, not proof
of a single causal explanation. All seven available signers have been exposed;
no new independent human comparison or real human webcam test is claimed.

Double-click `Start-SignBridge.cmd` for the built app. The exact static release
is prepared separately in `publication-v4`; it needs no Python server. No public
HTTPS deployment has been made. Historical V2 numbers below precede the V4 guard
and must not be described as current independent V4 performance.

Verification completed: TypeScript/production build, 8 Python tracking tests,
17 exact Python/JavaScript policy fixtures, 6 real MP4 product runs and 16 bounded
keyboard/accessibility checks. The camera-flow checks use an explicitly generated
blank stream; speech-flow checks emulate the API. Neither is a human webcam or
audible speech test. A real local Greek voice was detected. An unsupported
ΚΑΤΑΛΑΒΑΙΝΩ still falsely produces ΓΕΙΑ and one library ΕΝΤΑΞΕΙ is rejected.
These failures remain visible; no independent improvement claim is made.

[Product QA](reports/v4/product-qa/report.json) ·
[Keyboard QA](reports/v4/product-qa/accessibility.json) ·
[Static ZIP](signbridge-v4-static-preview.zip) ·
[Verified archive hashes](reports/v4/release-package.json).
The ZIP includes 32 audited static files plus deployment configuration, inventory
and instructions. The two QA reports bind to that exact publication inventory.

## Previous V3 experiment

Latest work: **V3 diagnosis and adaptive centroid experiment**.
[Greek technical report](reports/v3/TECHNICAL_REPORT.md) ·
[Reproduction and limitations](V3_DIAGNOSIS.md).

Added 75 genuine development clips and compared pooled versus separate negative
prototypes. Augmented pooled OTHER accepts 71/79 known signs correctly and falsely
accepts 0/50 unknowns on the **already exposed** V2 regression set. It also rejects
all 11 validation ΓΕΙΑ clips; it is not an independently validated replacement.
Active-hand dropout can be hidden by the existing quality score. All 344 new
browser parity checks and 48 real MP4 runs passed. No new human webcam recordings
or public deployment exist. The [publication review bundle](publication-review/README.md)
preserves the frozen B model below; the supported vocabulary is unchanged.

This repository is a real three-sign experiment. It is not a full Greek Sign
Language translator and is not yet validated on a new person using a webcam.

Current model: **`signbridge-open-set-v2-B_tcn_diverse_other`**, selected on
validation before its then-new 129-example final evaluation (now exposed). It classifies 78/79 known
examples correctly, accepts 59/79 known examples (all 59 correct), and falsely
accepts 24/50 unsupported examples. The unchanged old model falsely accepts
42/50 on this same new set; the centroid reference does better at 12/50. The
new mechanism improves rejection but remains unreliable. Do not expand the
vocabulary or claim trustworthy open-set communication from these results.

Previous experiment, split provenance and reproduction: [OPEN_SET_V2.md](OPEN_SET_V2.md).
Full Greek report: [Open-set technical report](reports/open-set-v2/TECHNICAL_REPORT.md).
All three candidates and frozen selection: `artifacts/open-set-v2`.
Double-click `Start-SignBridge.cmd` for the built app. Twelve current test clips
are in `data/open-set-v2/clips`, with original-frame/encoding provenance.

`Start-Webcam-Check.cmd` opens a separate optional local capture helper. It
requires explicit camera consent and a manual save. No human recordings were
completed in this session; genuine webcam recognition remains untested.

## Archived initial PoC

The remaining sections describe the initial experiment and its reproduction.
Use `OPEN_SET_V2.md` for the current open-set experiment. The original weights,
reports and features were snapshotted in `artifacts/poc-v2` and `reports/poc-v2`.

## Data and consent boundary

The input is the official CERTH GSL Zenodo v2 release, CC BY 4.0. We use a custom
subset of complete utterances annotated with exactly one gloss, not the official
isolated benchmark. See `data/ATTRIBUTION.md`, `data/manifest.json` and
`reports/dataset_verification.json` for the source, all eleven authors, exact
utterance text, signer identities, hashes and exclusions. No original source
video is uploaded when using the local web application.

| Partition | Signers | Known examples | Unknown examples |
|---|---|---:|---:|
| Training | 1, 2, 4, 5 | 119 | 20 |
| Validation | 6 | 30 | 5 |
| Final test | 3, 7 | 60 | 9 |

Known glosses: ΓΕΙΑ, ΕΥΧΑΡΙΣΤΩ, ΕΝΤΑΞΕΙ. The official utterance translations are
«γεια σας», «ευχαριστώ», «εντάξει». The selected ΓΕΙΑ occurs at dialogue closure.
These mappings were checked against annotations, not newly certified by a
linguist. Unknown classes differ across partitions. No everyday-motion or live
webcam negative collection is claimed.

## Run locally (PowerShell)

Run from this directory. Python 3.12 and Node 24 were used. The existing `.venv`
and `web/node_modules` are already installed in this working copy.

Double-click `Start-SignBridge.cmd` to start the already-built app and open it in
your browser. This launcher starts a hidden local preview server; it does not
train, download data, upload video or turn on the camera. For development:

```powershell
Set-Location web
npm run dev -- --port 5174 --strictPort
```

Open http://127.0.0.1:5174 in Chrome or Edge. Select a licensed local MP4 file,
or enable the camera. The application fetches local model/runtime assets; frame
analysis and classifier inference run in a Web Worker on the device. The camera
is unmirrored and the microphone is never requested. The selected camera
interval must contain one sign. Long idle padding is outside the training
distribution. The default interval is one second after a visual countdown.

Eight real transcoded test clips are available in `data/clips/`, with provenance
in `mapping.json`. They are test examples, not training examples, and their file
names/expected labels are never inputs to the classifier.

## Reproduce

1. Create a virtual environment and install `requirements.lock.txt` with the
   PyTorch CPU wheel index available. Install web packages with `npm ci` in `web`.
2. Run `scripts/dataset_audit.py`, `dataset_prepare.py`, `dataset_fetch.py`, and
   `dataset_verify.py` using the options documented by each script's `--help`.
   Acquisition uses HTTP byte ranges for selected RGB JPEGs; it does not fetch
   the 134.8 GB collection or depth data.
3. Run `.venv/Scripts/python.exe scripts/prepare_assets.py` to acquire the official
   holistic model and copy runtime assets from pinned npm dependencies. Exact
   runtime hashes are stored in `artifacts/runtime_assets.json`.
4. Keep Vite running, then extract real landmarks in headless Chrome:

```powershell
Set-Location web
node scripts/extract.mjs --url http://127.0.0.1:5174
Set-Location ..
.venv/Scripts/python.exe -X utf8 ml/pack_features.py --manifest data/manifest.json --jsonl data/features/extracted-isolated.jsonl --output data/features.npz
.venv/Scripts/python.exe -X utf8 ml/train.py --features data/features.npz --output artifacts --model-version signbridge-poc-tcn-v2-isolated
```

5. Copy `artifacts/model.onnx` and `artifacts/config.json` to
   `web/public/models/model.onnx` and `web/public/models/model_config.json`.
6. Run the browser parity/video checks and inspect the reports before making
   any performance claim. Build the app with `npm run build` inside `web`.

The recorded corrected run used three independent extraction processes for
speed: add `--shards 3 --shard 0` (then 1 and 2 in separate processes) to the
extraction command. Pack the three `extracted-isolated-0.jsonl`,
`extracted-isolated-1.jsonl`, and `extracted-isolated-2.jsonl` paths together after
`--jsonl`. Each contains 81 clips. Every clip starts a fresh MediaPipe worker;
records retain the extractor source hashes. Do not change extractor sources
mid-run or mix caches from different pipelines.

The initial v1 artifacts and audit are preserved under `artifacts/v1` and
`reports/v1`. The current v2 corrects detector state isolation and initial video
frame decoding. It repeats the same holdout after those pipeline fixes; it is
not a new untouched test. Future model or threshold experiments require new
test examples.

The trained ONNX includes training-only normalization. It expects raw
preprocessed float32 `[1,48,224]` inputs, not already standardized tensors.
Preprocessing is separately implemented in Python and TypeScript and compared
on actual landmark fixtures. No filename, label or signer ID is a model input.

## Evaluation rules

- Train/validation/test people are disjoint. No augmentation or cropping crosses
  partitions. Original frames were checked for duplicates across partitions.
- Checkpoints, temperature and rejection thresholds are selected using training
  and validation only. The final test is not a hyperparameter search set.
- Closed-set metrics exclude unknown classes and are separate from accepted
  accuracy, known coverage and unknown false acceptance. A rejection is not
  silently counted as a correct recognition.
- A small validation/test negative set cannot establish universal open-set
  reliability. Webcam operation and webcam recognition accuracy are different
  claims; a fluent new signer is still needed for the latter.
- Landmarks are detected using MediaPipe Holistic CPU IMAGE mode in a newly
  created worker for every clip. IMAGE mode alone showed history-dependent
  differences and was not sufficient isolation. The Temporal CNN learns
  from the ordered sequence of those landmarks.

Detailed results and limitations are written to `reports/TECHNICAL_REPORT.md` and
the machine-readable reports after the experiment completes.

## Local continual learning

`scripts/Install-SignBridgeLearning.ps1 -RunNow` installs an hourly Windows task for
the signed-in user and requests its first check. The worker keeps its registry,
approved landmark batches and candidate models in `%LOCALAPPDATA%/SignBridge/continual`.
It trains once for each distinct set of approved inputs and records a check when
the inputs are unchanged. A local candidate must improve recognition on difficult
development phrases, preserve familiar phrase performance and pass the text checks.
The development cohorts are reused for monitoring. Release evaluation requires
independent recordings.

```powershell
.venv/Scripts/python.exe -X utf8 ml/continual_learning.py status
.venv/Scripts/python.exe -X utf8 ml/continual_learning.py approve --manifest <reviewed-manifest.json> --features <landmarks.jsonl> --reviewer <reviewer-id>
```

Prepare the manifest and 15 Hz MediaPipe feature file locally. The manifest uses a
`samples` array. Each row contains `id`, `source_sha256`, `start_ms`, `end_ms`,
`signer`, `session`, `consent_local_ml`, `glosses` and `text`. Use a consistent signer
alias, explicit consent for local training, and the exact ordered glosses verified
by a reviewer. Each JSONL feature record uses the matching `id`, `feature_schema`,
`fps`, `quality` and `features`. Approval seals a local copy for future checks.
The command accepts 0.4–12 second clips and glosses from the frozen vocabulary.

Use Task Scheduler to inspect or disable `SignBridge-ContinualLearning`. For an
interactive run, `ml/continual_learning.py watch` checks hourly while it runs.

## Local GSL library and recognition

The library keeps official glosses, Greek translations, playable references and
motion recordings as separate evidence. Rebuild the local SQLite index after
changing the verified data. Search is accent-insensitive, and results retain the
source spelling. The isolated recognizer returns the nearest verified training
example with its prediction.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/build_gsl_provenance_index.py
.venv/Scripts/python.exe -X utf8 ml/sign_library.py stats
.venv/Scripts/python.exe -X utf8 ml/sign_library.py search "γιατρός"
.venv/Scripts/python.exe -X utf8 ml/sign_library.py recognize --source-id health1_signer3_rep3_glosses/glosses0000
```

`recognize --pose` also accepts a native NumPy array with shape `[frames,75,3]`.
This experiment measures isolated signs from one held CERTH signer. See
`reports/isolated-exemplar-retrieval-expanded/fusion-expanded-benchmark.json`
for the recognition results and `data/sign-reference-videos/manifest.json` for
the new verified clips and their attribution.
