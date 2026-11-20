"""Replay frozen direct Greek translation checkpoints with altered motion."""

from dataclasses import replace
import json
from pathlib import Path

import torch

from train_continuous import altered_frames, ctc_decode, edit_counts
from train_greek_translation_pilot import (
    Translator, certh_sets, contextual_sets, elementary_sets, encoded_input, evaluate, metrics, sha,
)


FAMILIES = ("greek-translation-pilot-context", "greek-translation-certh-context")
MODES = ("original", "reversed", "static_midpoint", "static_mean")


def replay_rows(samples, mode):
    if mode == "static_mean":
        return [replace(row, frames=row.frames.mean(0).expand_as(row.frames)) for row in samples]
    return [replace(row, frames=altered_frames(row, mode)) for row in samples]


@torch.inference_mode()
def gloss_metrics(model, rows, labels):
    frames, lengths = encoded_input(rows)
    memory, _, _, _ = model.encode(frames, lengths)
    scores = model.gloss(memory).log_softmax(-1)
    edits, exact, references = 0, 0, 0
    for row, values, length in zip(rows, scores, lengths, strict=True):
        prediction, _ = ctc_decode(values[:length], labels)
        edits += sum(edit_counts(row.glosses, prediction))
        exact += prediction == row.glosses
        references += len(row.glosses)
    return {"gloss_edits": edits, "reference_glosses": references, "exact_gloss_sequences": exact}


def main():
    torch.set_num_threads(2)
    elementary = elementary_sets()
    cohorts = {
        "certh_development": certh_sets()["val"],
        "certh_context_diagnostic": contextual_sets()["test"],
        "elementary_development": elementary["val"],
        "elementary_diagnostic": elementary["test"],
    }
    report = {
        "scope": "Retrospective audit of previously evaluated clips and fixed checkpoints. No parameter or threshold fitting.",
        "controls": {
            "original": "Original time order",
            "reversed": "Original frames in reversed time order",
            "static_midpoint": "Repeat the original midpoint frame for the original clip duration",
            "static_mean": "Repeat the mean feature vector for the original clip duration. Masks are averaged too.",
        },
        "families": {},
    }
    for family in FAMILIES:
        frozen = json.loads((Path("reports") / family / "freeze.json").read_text(encoding="utf-8"))
        for path, checksum in frozen["sources"].items():
            if sha(Path(path.replace("\\", "/"))) != checksum:
                raise ValueError(f"Frozen source changed: {path}")
        results = report["families"][family] = {}
        for arm, record in frozen["models"].items():
            checkpoint = Path("artifacts") / family / f"{arm}.pt"
            if sha(checkpoint) != record["artifact_sha256"]:
                raise ValueError(f"Frozen checkpoint changed: {checkpoint}")
            artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
            labels, state = artifact["glosses"], artifact["state_dict"]
            model = Translator(state["mean"], state["scale"], len(labels))
            model.load_state_dict(state)
            model.eval()
            scores = {}
            for cohort, rows in cohorts.items():
                has_glosses = rows[0].corpus == "certh"
                if not has_glosses and family == FAMILIES[1]:
                    continue
                scores[cohort] = {}
                for mode in MODES:
                    replayed = replay_rows(rows, mode)
                    scores[cohort][mode] = metrics(evaluate(model, replayed))
                    if has_glosses:
                        scores[cohort][mode].update(gloss_metrics(model, replayed, labels))
            results[arm] = {"checkpoint_sha256": record["artifact_sha256"], "cohorts": scores}
    output = Path("reports") / FAMILIES[0] / "visual-ablation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.as_posix())


if __name__ == "__main__":
    main()
