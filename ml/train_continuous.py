"""Train a gloss-sequence recognizer on complete GSL utterances."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from preprocess import FEATURE_COUNT, FEATURE_SCHEMA


ROLES = ("train", "val", "test")


@dataclass
class Utterance:
    id: str
    source: str
    signer: str
    split: str
    glosses: tuple[str, ...]
    text: str
    frames: torch.Tensor
    fps: float
    quality: float
    extraction_ms: float | None


def read_utterances(manifest_path, feature_paths, allow_missing, required_roles=ROLES):
    samples = json.loads(manifest_path.read_text(encoding="utf-8-sig"))["samples"]
    identifiers, sources, signers = set(), set(), defaultdict(set)
    for sample in samples:
        identity, source, signer, split = (sample[key] for key in ("id", "source_path", "signer", "split"))
        if identity in identifiers or source in sources:
            raise ValueError(f"Duplicate utterance id or source: {identity}")
        if split not in required_roles:
            raise ValueError(f"Unrecognized split for {identity}: {split}")
        if not isinstance(sample["glosses"], list) or not all(
                isinstance(gloss, str) and gloss.strip() for gloss in sample["glosses"]):
            raise ValueError(f"Invalid glosses: {identity}")
        if not sample["glosses"] or not isinstance(sample["text"], str):
            raise ValueError(f"Missing glosses or invalid Greek reference: {identity}")
        identifiers.add(identity)
        sources.add(source)
        signers[str(signer)].add(split)
    overlaps = [signer for signer, splits in signers.items() if len(splits) != 1]
    if overlaps:
        raise ValueError(f"Signer leakage between roles: {overlaps}")

    features = {}
    for path in feature_paths:
        with path.open(encoding="utf-8-sig") as records:
            for line in records:
                if not line.strip():
                    continue
                record = json.loads(line)
                identity = record["id"]
                if identity not in identifiers or identity in features:
                    raise ValueError(f"Unexpected or duplicate feature record: {identity}")
                features[identity] = record

    missing = sorted(identifiers - features.keys())
    if missing and not allow_missing:
        raise ValueError(f"Missing features for {len(missing)} of {len(samples)} clips (first: {missing[0]})")
    roles = {role: [] for role in required_roles}
    for sample in samples:
        identity = sample["id"]
        if identity not in features:
            continue
        record = features[identity]
        if record.get("feature_schema", FEATURE_SCHEMA) != FEATURE_SCHEMA:
            raise ValueError(f"Unexpected feature schema: {identity}")
        frames = torch.as_tensor(record["features"], dtype=torch.float32)
        fps, quality = float(record["fps"]), float(record["quality"])
        if frames.ndim != 2 or len(frames) < 2 or frames.shape[1] != FEATURE_COUNT or not frames.isfinite().all():
            raise ValueError(f"Invalid frame features for {identity}: {list(frames.shape)}")
        if not math.isfinite(fps) or fps <= 0 or not 0 <= quality <= 1:
            raise ValueError(f"Invalid tracking metadata: {identity}")
        elapsed_ms = record.get("elapsed_ms")
        if elapsed_ms is not None and (not math.isfinite(elapsed_ms) or elapsed_ms < 0):
            raise ValueError(f"Invalid extraction time: {identity}")
        roles[sample["split"]].append(Utterance(
            identity, sample["source_path"], str(sample["signer"]), sample["split"],
            tuple(gloss.strip() for gloss in sample["glosses"]), sample["text"],
            frames, fps, quality, elapsed_ms,
        ))
    if any(not utterances for utterances in roles.values()):
        raise ValueError("Each signer-disjoint role requires extracted utterances")
    return roles, missing


class ContinuousCTC(nn.Module):
    def __init__(self, output_size, mean, scale, hidden=64):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.projection = nn.Sequential(nn.Linear(FEATURE_COUNT, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden * 2, output_size))

    def forward(self, frames, lengths):
        frame_features = self.projection((frames - self.mean) / self.scale)
        packed = pack_padded_sequence(frame_features, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.temporal(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)
        return self.classifier(encoded).log_softmax(-1)


def batch_loss(model, examples, vocabulary, loss_fn):
    sequences = [item.frames for item in examples]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    targets = [torch.tensor([vocabulary[gloss] for gloss in item.glosses]) for item in examples]
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    scores = model(pad_sequence(sequences, batch_first=True), lengths)
    return loss_fn(scores.transpose(0, 1), torch.cat(targets), lengths, target_lengths)


def ctc_decode(scores, labels):
    # Merge consecutive nonblank runs; blank-separated repeats remain distinct.
    values, indexes = scores.max(dim=-1)
    glosses, confidences = [], []
    start = 0
    for end in range(1, len(indexes) + 1):
        if end < len(indexes) and indexes[end] == indexes[start]:
            continue
        label = int(indexes[start])
        if label:
            glosses.append(labels[label])
            confidences.append(float(values[start:end].mean().exp()))
        start = end
    return tuple(glosses), min(confidences, default=0.0)


def edit_counts(reference, predicted):
    """Return Levenshtein substitutions, deletions and insertions."""
    previous = [(0, 0, index) for index in range(len(predicted) + 1)]
    for row, expected in enumerate(reference, 1):
        current = [(0, row, 0)]
        for column, actual in enumerate(predicted, 1):
            sub, delete, insert = previous[column - 1]
            replaced = (sub + (expected != actual), delete, insert)
            sub, delete, insert = previous[column]
            removed = (sub, delete + 1, insert)
            sub, delete, insert = current[-1]
            added = (sub, delete, insert + 1)
            current.append(min((replaced, removed, added), key=sum))
        previous = current
    return previous[-1]


def normalize_text(text):
    return " ".join(text.casefold().split())


def reference_lookup(train):
    candidates = defaultdict(dict)
    for item in train:
        normalized = normalize_text(item.text)
        if normalized:
            candidates[item.glosses].setdefault(normalized, item.text.strip())
    return {glosses: next(iter(texts.values())) for glosses, texts in candidates.items() if len(texts) == 1}


def altered_frames(item, mode):
    if mode == "reversed":
        return item.frames.flip(0)
    if mode == "shuffled":
        seed = int.from_bytes(hashlib.sha256(item.id.encode()).digest()[:4], "little")
        order = torch.randperm(len(item.frames), generator=torch.Generator().manual_seed(seed))
        return item.frames[order]
    if mode == "static_midpoint":
        return item.frames[len(item.frames) // 2].expand_as(item.frames)
    return item.frames


@torch.inference_mode()
def replay(model, examples, labels, lookup, threshold=0.0, mode="original"):
    model.eval()
    rows = []
    for item in examples:
        frames = altered_frames(item, mode)
        started = time.perf_counter()
        scores = model(frames.unsqueeze(0), [len(frames)])[0]
        glosses, confidence = ctc_decode(scores, labels)
        inference_ms = (time.perf_counter() - started) * 1_000
        candidate = lookup.get(glosses)
        text = candidate if confidence >= threshold else None
        reason = None if text else ("low_confidence" if candidate else "no_unambiguous_training_reference")
        substitutions, deletions, insertions = edit_counts(item.glosses, glosses)
        rows.append({
            "id": item.id, "source_path": item.source, "signer": item.signer, "split": item.split,
            "reference_glosses": item.glosses, "predicted_glosses": glosses,
            "reference_text": item.text, "predicted_text": text, "abstention": reason,
            "confidence": confidence, "tracking_quality": item.quality, "frames": len(frames), "fps": item.fps,
            "inference_ms": inference_ms, "extraction_ms": item.extraction_ms,
            "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
            "gloss_sequence_exact": item.glosses == glosses,
            "greek_reference_exact": bool(text and normalize_text(text) == normalize_text(item.text)),
        })
    return rows


def summarize(rows):
    accepted = sum(row["predicted_text"] is not None for row in rows)
    correct = sum(row["greek_reference_exact"] for row in rows)
    counts = {key: sum(row[key] for row in rows) for key in ("substitutions", "deletions", "insertions")}
    target_count = sum(len(row["reference_glosses"]) for row in rows)
    extraction = [row["extraction_ms"] for row in rows if row["extraction_ms"] is not None]
    return {
        "utterances": len(rows), "reference_gloss_count": target_count, **counts,
        "gloss_wer": sum(counts.values()) / target_count,
        "gloss_sequence_exact": sum(row["gloss_sequence_exact"] for row in rows),
        "gloss_sequence_exact_rate": sum(row["gloss_sequence_exact"] for row in rows) / len(rows),
        "accepted": accepted, "abstained": len(rows) - accepted, "coverage": accepted / len(rows),
        "greek_reference_exact": correct, "accepted_wrong_reference": accepted - correct,
        "greek_reference_exact_among_accepted": correct / accepted if accepted else None,
        "median_cpu_model_ms": statistics.median(row["inference_ms"] for row in rows),
        "median_browser_offline_extraction_ms": statistics.median(extraction) if extraction else None,
    }


def choose_threshold(rows):
    # Only development rows select this threshold. Penalize false texts before
    # maximizing correct text outputs and keep all actual gloss predictions.
    candidates = sorted({0.0, *(row["confidence"] for row in rows), 1.000001})

    def score(threshold):
        retained = [row for row in rows if row["predicted_text"] and row["confidence"] >= threshold]
        correct = sum(row["greek_reference_exact"] for row in retained)
        return len(retained) - correct, -correct, threshold

    return min(candidates, key=score)


def split_summary(examples, train_sequences, vocabulary, lookup):
    absent = sorted({gloss for item in examples for gloss in item.glosses} - vocabulary.keys())
    return {
        "utterances": len(examples), "signers": sorted({item.signer for item in examples}),
        "gloss_sequence_templates": len({item.glosses for item in examples}),
        "new_sequence_utterances": sum(item.glosses not in train_sequences for item in examples),
        "new_sequence_in_vocabulary_utterances": sum(
            item.glosses not in train_sequences and set(item.glosses) <= vocabulary.keys() for item in examples
        ),
        "out_of_training_vocabulary_glosses": absent,
        "reference_lookup_eligible": sum(item.glosses in lookup for item in examples),
        "min_frames": min(len(item.frames) for item in examples),
        "max_frames": max(len(item.frames) for item in examples),
        "min_quality": min(item.quality for item in examples),
        "median_quality": statistics.median(item.quality for item in examples),
    }


def train_model(roles, labels, vocabulary, lookup, epochs, batch_size, seed):
    train = roles["train"]
    all_frames = torch.cat([item.frames for item in train])
    mean, scale = all_frames.mean(0), all_frames.std(0, unbiased=False).clamp_min(0.08)
    model = ContinuousCTC(len(labels), mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    loss_fn = nn.CTCLoss(blank=0)
    history, best_key, best_state = [], None, None
    generator = random.Random(seed)

    for item in train:
        minimum = len(item.glosses) + sum(left == right for left, right in zip(item.glosses, item.glosses[1:]))
        if len(item.frames) < minimum:
            raise ValueError(f"CTC needs at least {minimum} frames for {item.id}")

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = generator.sample(train, len(train))
        losses = []
        for start in range(0, len(shuffled), batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, shuffled[start:start + batch_size], vocabulary, loss_fn)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = summarize(replay(model, roles["val"], labels, lookup))
        mean_loss = statistics.mean(losses)
        history.append({"epoch": epoch, "train_ctc_loss": mean_loss, "validation_gloss_wer": validation["gloss_wer"],
                        "validation_gloss_sequence_exact": validation["gloss_sequence_exact"]})
        key = (validation["gloss_wer"], -validation["gloss_sequence_exact"], mean_loss)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch in (1, epochs) or epoch % 10 == 0:
            print(json.dumps(history[-1]), flush=True)

    model.load_state_dict(best_state)
    return model, history, best_epoch


def evaluate_holdout(args):
    roles, missing = read_utterances(args.manifest, args.features, args.allow_missing, ("test",))
    checkpoint = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if checkpoint["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("Checkpoint uses a different feature schema")
    examples = roles["test"]
    ids, sources, signers = (set(checkpoint[key]) for key in ("prior_ids", "prior_sources", "development_signers"))
    for item in examples:
        if item.id in ids or item.source in sources or item.signer in signers:
            raise ValueError(f"Holdout overlaps earlier utterances or development signers: {item.id}")
    labels = checkpoint["labels"]
    lookup = checkpoint["training_greek_references"]
    train_sequences = set(checkpoint["training_gloss_sequences"])
    state = checkpoint["state_dict"]
    model = ContinuousCTC(len(labels), state["mean"], state["scale"])
    model.load_state_dict(state)
    threshold = checkpoint["confidence_threshold"]
    vocabulary = {label: index for index, label in enumerate(labels) if index}
    rows = replay(model, examples, labels, lookup, threshold)
    novel = [row for row in rows if row["reference_glosses"] not in train_sequences]
    report = {
        "setting": "frozen checkpoint; additional held-out utterances; no threshold changes",
        "checkpoint_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "features_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.features},
        "confidence_threshold": threshold, "missing_feature_ids": missing,
        "split_summary": split_summary(examples, train_sequences, vocabulary, lookup),
        "test": summarize(rows), "unseen_sequence_test": summarize(novel) if novel else None,
        "predictions": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": report["test"], "unseen_sequence_test": report["unseen_sequence_test"],
                      "report": str(args.report)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/continuous-v1/ctc.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/continuous-v1/ctc.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Score new test clips with the frozen --artifact")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size and threads must be positive")
    if args.report.exists() or (args.artifact.exists() and not args.evaluate_only):
        parser.error("Refusing to overwrite an existing experiment. Specify new artifact and report paths.")

    torch.set_num_threads(args.threads)
    if args.evaluate_only:
        evaluate_holdout(args)
        return
    torch.manual_seed(args.seed)
    roles, missing = read_utterances(args.manifest, args.features, args.allow_missing)
    train = roles["train"]
    vocabulary = {gloss: index for index, gloss in enumerate(sorted({token for item in train for token in item.glosses}), 1)}
    labels = [None, *vocabulary]
    lookup = reference_lookup(train)
    train_sequences = {item.glosses for item in train}
    splits = {role: split_summary(examples, train_sequences, vocabulary, lookup) for role, examples in roles.items()}
    print(json.dumps({"splits": splits, "missing_extraction_count": len(missing)}, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    model, history, best_epoch = train_model(roles, labels, vocabulary, lookup, args.epochs, args.batch_size, args.seed)
    duration = time.perf_counter() - started
    validation_raw = replay(model, roles["val"], labels, lookup)
    threshold = choose_threshold(validation_raw)
    validation = replay(model, roles["val"], labels, lookup, threshold)
    test = replay(model, roles["test"], labels, lookup, threshold)
    perturbations = {mode: replay(model, roles["test"], labels, lookup, threshold, mode)
                     for mode in ("reversed", "shuffled", "static_midpoint")}
    report = {
        "model": "bidirectional GRU with CTC, greedy gloss decoding", "feature_schema": FEATURE_SCHEMA,
        "setting": "offline utterance clips; supplied boundaries; signer-disjoint roles within this experiment",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "features_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.features},
        "seed": args.seed, "training_seconds": duration, "checkpoint_epoch": best_epoch,
        "selection": "minimum validation gloss WER, maximum validation exact gloss sequences, minimum train CTC loss",
        "split_summary": splits, "missing_feature_ids": missing, "training_vocabulary": list(vocabulary),
        "translation": "exact decoded gloss tuple matched to an unambiguous Greek reference seen in training",
        "abstention": {"confidence_threshold": threshold, "confidence_definition": "minimum geometric mean of frame posterior across each emitted nonblank run", "threshold_selection": "minimize wrong accepted texts, then maximize correct accepted texts, on validation only", "unknown_rejection_validated": False},
        "validation": summarize(validation), "test": summarize(test),
        "test_frame_perturbations": {mode: summarize(rows) for mode, rows in perturbations.items()},
        "validation_predictions": validation, "test_predictions": test,
        "test_frame_perturbation_predictions": perturbations, "history": history,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    examples = [item for utterances in roles.values() for item in utterances]
    torch.save({"state_dict": model.state_dict(), "labels": labels, "feature_schema": FEATURE_SCHEMA,
                "confidence_threshold": threshold, "training_greek_references": lookup,
                "training_gloss_sequences": sorted(train_sequences),
                "prior_ids": [item.id for item in examples], "prior_sources": [item.source for item in examples],
                "development_signers": sorted({item.signer for role in ("train", "val") for item in roles[role]})},
               args.artifact)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_epoch": best_epoch, "threshold": threshold, "test": report["test"],
                      "report": str(args.report)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
