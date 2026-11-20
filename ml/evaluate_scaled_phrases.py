"""Independent exact/edit scoring for frozen GSL phrase predictions.

This module performs no inference, fitting, threshold selection, or corpus loading.
Text scores measure transcription agreement, not semantic translation accuracy.
Counterfactual video predictions must be scored separately with evaluate_paired_rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata


SCHEMA = "signbridge-scaled-phrases-scoring-v1"
NORMALIZATION = (
    "Greek/text: Unicode NFC, lowercase, Unicode letter/digit tokens; punctuation, "
    "underscores and whitespace ignored; accents and Latin names retained. "
    "Glosses: individual NFC tokens, exact case and label spelling retained."
)


def normalize_text(text: str) -> str:
    """Normalize surface form without translating or removing Greek diacritics."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).lower()))


def edit_counts(reference: list[str], prediction: list[str]) -> dict[str, int]:
    """Levenshtein counts; ties prefer diagonal, deletion, then insertion.

    Counts describe one deterministic minimum-edit alignment. In ambiguous
    alignments the individual S/D/I counts can differ across implementations.
    """
    n, m = len(reference), len(prediction)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    result = {"correct": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            result["correct" if reference[i - 1] == prediction[j - 1] else "substitutions"] += 1
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            result["deletions"] += 1
            i -= 1
        else:
            result["insertions"] += 1
            j -= 1
    result["edits"] = result["substitutions"] + result["deletions"] + result["insertions"]
    return result


def _glosses(value, name: str, identity: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{identity}: {name} must be a {'nonempty ' if nonempty else ''}token list")
    if any(not isinstance(token, str) or not token or any(c.isspace() for c in token) for token in value):
        raise ValueError(f"{identity}: {name} contains an empty, non-string or whitespace-bearing token")
    return [unicodedata.normalize("NFC", token) for token in value]


def _tag(row: dict, key: str) -> str:
    value = row.get(key)
    if key == "signer" and value is None:
        value = row.get("signer_id")
    if value is None:
        return "unspecified"
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"{row.get('id')}: invalid {key} metadata")
    return str(value)


def _prepare(rows) -> list[dict]:
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Scoring requires a nonempty list of prediction rows")
    records, ids, conditions = [], set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each prediction row must be a dictionary")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Prediction IDs must be nonempty and unique within a condition")
        ids.add(identity)
        reference = row.get("reference")
        if not isinstance(reference, str) or not normalize_text(reference):
            raise ValueError(f"{identity}: a nonempty reference is required; background belongs in a separate detection score")
        if "text" not in row or (row["text"] is not None and not isinstance(row["text"], str)):
            raise ValueError(f"{identity}: text must be a string or explicit null for abstention")
        text = normalize_text(row["text"] or "")
        accepted = bool(text)
        if "accepted" in row and (not isinstance(row["accepted"], bool) or row["accepted"] != accepted):
            raise ValueError(f"{identity}: accepted must agree with emitted nonempty text")
        record = {
            "id": identity,
            "gold": _glosses(row.get("gold_glosses"), "gold_glosses", identity, nonempty=True),
            "prediction": _glosses(row.get("predicted_glosses"), "predicted_glosses", identity, nonempty=False),
            "reference": normalize_text(reference), "text": text,
            "raw_reference": reference, "raw_text": row["text"], "accepted": accepted,
            **{key: _tag(row, key) for key in ("signer", "category", "composition", "split", "condition")},
        }
        conditions.add(record["condition"])
        records.append(record)
    if len(conditions) != 1:
        raise ValueError("Do not pool correct-video, mismatched-video or other conditions; score them separately")
    return records


def _aggregate(records: list[dict]) -> dict:
    counts = {kind: {key: 0 for key in ("correct", "substitutions", "deletions", "insertions", "edits")}
              for kind in ("gloss", "greek")}
    glossary = words = exact_gloss = exact_text = verbatim = accepted = correct_accepted = 0
    for row in records:
        glossary += len(row["gold"])
        words += len(row["reference"].split())
        for kind, ref, pred in (("gloss", row["gold"], row["prediction"]),
                                ("greek", row["reference"].split(), row["text"].split())):
            for key, value in edit_counts(ref, pred).items():
                counts[kind][key] += value
        exact_gloss += row["gold"] == row["prediction"]
        exact_text += row["reference"] == row["text"]
        verbatim += row["raw_reference"] == row["raw_text"]
        accepted += row["accepted"]
        correct_accepted += row["accepted"] and row["reference"] == row["text"]
    total = len(records)
    return {
        "utterances": total,
        "reference_gloss_count": glossary,
        "gloss_edit_counts": counts["gloss"],
        "gloss_edits": counts["gloss"]["edits"],
        "gloss_wer": counts["gloss"]["edits"] / glossary,
        "gloss_sequence_exact": exact_gloss,
        "gloss_sequence_exact_rate": exact_gloss / total,
        "reference_word_count": words,
        "greek_edit_counts": counts["greek"],
        "greek_word_edits": counts["greek"]["edits"],
        "greek_wer": counts["greek"]["edits"] / words,
        "greek_reference_exact": exact_text,
        "greek_reference_exact_rate": exact_text / total,
        "greek_verbatim_exact": verbatim,
        "accepted_texts": accepted,
        "abstained_texts": total - accepted,
        "text_coverage": accepted / total,
        "correct_accepted_texts": correct_accepted,
        "incorrect_accepted_texts": accepted - correct_accepted,
        "accepted_text_exact_rate": correct_accepted / accepted if accepted else None,
    }


def evaluate_rows(rows: list[dict]) -> dict:
    """Score one prediction condition without mutating the supplied rows.

    Required: id, gold_glosses, predicted_glosses, reference, text (str or None).
    Recommended: signer, category, composition, split. Missing grouping metadata
    appears explicitly as 'unspecified'. Gloss WER is micro-averaged token edit
    distance. Greek WER includes abstentions as deletions; accepted-only exactness
    is reported alongside coverage. An incorrect exact match is not necessarily
    a semantic mistranslation, and a match alone does not prove visual grounding.
    """
    prepared = _prepare(rows)
    result = {"schema": SCHEMA, "normalization": NORMALIZATION,
              "interpretation": "Exact surface/reference agreement; not semantic accuracy or temporal localization.",
              "condition": prepared[0]["condition"], **_aggregate(prepared)}
    for key in ("signer", "category", "composition", "split"):
        groups = defaultdict(list)
        for row in prepared:
            groups[row[key]].append(row)
        result[f"by_{key}"] = {name: _aggregate(group) for name, group in sorted(groups.items())}
    return result


def evaluate_paired_rows(correct_rows: list[dict], mismatched_rows: list[dict]) -> dict:
    """Compare matched targets under correct versus different input videos.

    Both lists use the same target IDs/references and must supply input_video_id.
    Input video identity must change for every pair. If provided, input_reference
    and input_signer are validated; no such declaration is invented when absent.
    This diagnostic is not an independent test or a semantic-grounding proof.
    """
    correct, mismatched = _prepare(correct_rows), _prepare(mismatched_rows)
    a, b = {r["id"]: r for r in correct}, {r["id"]: r for r in mismatched}
    raw_a, raw_b = {r["id"]: r for r in correct_rows}, {r["id"]: r for r in mismatched_rows}
    if set(a) != set(b):
        raise ValueError("Paired conditions must contain exactly the same target IDs")
    details, length_checked, source_reference_checked, same_signer_checked = [], 0, 0, 0
    for identity in sorted(a):
        left, right = a[identity], b[identity]
        if any(left[key] != right[key] for key in ("gold", "raw_reference", "signer", "category", "composition", "split")):
            raise ValueError(f"{identity}: the gold target and grouping metadata must stay identical across conditions")
        left_raw, right_raw = raw_a[identity], raw_b[identity]
        source_ids = [r.get("input_video_id") for r in (left_raw, right_raw)]
        if any(not isinstance(value, str) or not value.strip() for value in source_ids) or source_ids[0] == source_ids[1]:
            raise ValueError(f"{identity}: paired scoring requires two different explicit input_video_id values")
        if "input_reference" in right_raw:
            if not isinstance(right_raw["input_reference"], str) or not normalize_text(right_raw["input_reference"]):
                raise ValueError(f"{identity}: invalid mismatched input_reference")
            if normalize_text(right_raw["input_reference"]) == left["reference"]:
                raise ValueError(f"{identity}: mismatched video has the same normalized reference")
            source_reference_checked += 1
        if "input_signer" in right_raw:
            if str(right_raw["input_signer"]) != left["signer"]:
                raise ValueError(f"{identity}: declared mismatched input_signer differs from target signer")
            same_signer_checked += 1
        if "input_frames" in left_raw or "input_frames" in right_raw:
            frame_counts = [r.get("input_frames") for r in (left_raw, right_raw)]
            if any(type(value) is not int or value < 1 for value in frame_counts) or frame_counts[0] != frame_counts[1]:
                raise ValueError(f"{identity}: declared paired input frame counts must be positive and equal")
            length_checked += 1
        true_gloss = edit_counts(left["gold"], left["prediction"])["edits"]
        wrong_gloss = edit_counts(right["gold"], right["prediction"])["edits"]
        true_text = edit_counts(left["reference"].split(), left["text"].split())["edits"]
        wrong_text = edit_counts(right["reference"].split(), right["text"].split())["edits"]
        details.append({"id": identity, "signer": left["signer"], "category": left["category"],
                        "composition": left["composition"], "correct_input_video_id": source_ids[0],
                        "mismatched_input_video_id": source_ids[1],
                        "gloss_edits_correct": true_gloss, "gloss_edits_mismatched": wrong_gloss,
                        "greek_edits_correct": true_text, "greek_edits_mismatched": wrong_text,
                        "same_gloss_prediction": left["prediction"] == right["prediction"],
                        "same_text_prediction": left["text"] == right["text"],
                        "same_nonempty_text_prediction": bool(left["text"]) and left["text"] == right["text"]})
    return {
        "schema": SCHEMA + "-paired", "pairs": len(details),
        "interpretation": "Controlled paired diagnostic; no semantic accuracy or causal grounding claim.",
        "verified_equal_input_frames": length_checked,
        "verified_distinct_input_references": source_reference_checked,
        "verified_same_input_signer": same_signer_checked,
        "correct_video": evaluate_rows(correct_rows), "mismatched_video": evaluate_rows(mismatched_rows),
        "gloss_correct_video_better": sum(r["gloss_edits_correct"] < r["gloss_edits_mismatched"] for r in details),
        "gloss_correct_video_worse": sum(r["gloss_edits_correct"] > r["gloss_edits_mismatched"] for r in details),
        "gloss_ties": sum(r["gloss_edits_correct"] == r["gloss_edits_mismatched"] for r in details),
        "greek_correct_video_better": sum(r["greek_edits_correct"] < r["greek_edits_mismatched"] for r in details),
        "greek_correct_video_worse": sum(r["greek_edits_correct"] > r["greek_edits_mismatched"] for r in details),
        "greek_ties": sum(r["greek_edits_correct"] == r["greek_edits_mismatched"] for r in details),
        "same_gloss_predictions": sum(r["same_gloss_prediction"] for r in details),
        "same_text_predictions": sum(r["same_text_prediction"] for r in details),
        "same_nonempty_text_predictions": sum(r["same_nonempty_text_prediction"] for r in details),
        "pairs_detail": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list, or {rows: [...]} from one frozen condition")
    parser.add_argument("--mismatched-predictions", type=Path, help="Same target IDs with different explicit input_video_id")
    parser.add_argument("--freeze", type=Path, required=True, help="Existing fit/protocol lock; scorer never fits models")
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_hash = sha256(args.freeze.read_bytes()).hexdigest()
    if freeze_hash != args.expected_freeze_sha256:
        parser.error("The supplied freeze does not match the required SHA256")
    paths = [args.predictions] + ([args.mismatched_predictions] if args.mismatched_predictions else [])
    if args.output.resolve() in {path.resolve() for path in [*paths, args.freeze]}:
        parser.error("Output must not overwrite predictions or the freeze")
    inputs, hashes = [], {}
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        inputs.append(data["rows"] if isinstance(data, dict) else data)
        hashes[str(path)] = sha256(raw).hexdigest()
    report = evaluate_paired_rows(*inputs) if len(inputs) == 2 else evaluate_rows(inputs[0])
    report["provenance"] = {"freeze_sha256": freeze_hash, "prediction_files_sha256": hashes,
                            "scorer_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
