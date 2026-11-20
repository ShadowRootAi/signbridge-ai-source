"""Train a compact Greek decoder with paired video, gloss supervision, and visual negatives."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
import random
import re
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from evaluate_continuous_transfer import load_inputs
from preprocess import FEATURE_COUNT, FEATURE_SCHEMA
from train_continuous import edit_counts, read_utterances


PAD, BOS, EOS = 0, 1, 2
ALPHABET = " αβγδεζηθικλμνξοπρστυφχψωάέήίόύώϊϋΐΰς0123456789"
TOKENS = ("<pad>", "<bos>", "<eos>", *ALPHABET)
CHARACTER_IDS = {char: index for index, char in enumerate(TOKENS)}
ROLES = ("train", "val", "test")
SEED = 73
WIDTH = 64
ELEMENTARY = Path("data/elementary-translation-pilot")
CONTEXT = Path("data/continuous-context")


@dataclass
class Sample:
    id: str
    corpus: str
    signer: str
    text: str
    glosses: tuple[str, ...]
    frames: torch.Tensor


def greek(text):
    return " ".join(re.findall(r"[α-ωάέήίόύώϊϋΐΰ0-9]+", unicodedata.normalize("NFC", text.lower())))


def sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def certh_sets():
    pools = [load_inputs(name) for name in ("expanded", "composition")]
    path = Path("data/continuous-fresh-compact")
    compact, _ = read_utterances(path / "manifest.json", [path / "features.jsonl"], False)
    out = {}
    for role in ROLES:
        rows = compact[role] if role == "test" else [row for pool in [*pools, compact] for row in pool[role]]
        out[role] = [Sample(row.id, "certh", row.signer, greek(row.text), tuple(row.glosses), row.frames)
                     for row in rows]
    return out


def contextual_sets():
    rows, _ = read_utterances(CONTEXT / "manifest.json", [CONTEXT / "features.jsonl"], False)
    return {role: [Sample(row.id, "certh", row.signer, greek(row.text), tuple(row.glosses), row.frames)
                   for row in rows[role]] for role in ("train", "test")}


def elementary_sets():
    path = ELEMENTARY / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest["complete"] or len(manifest["samples"]) != manifest["planned_samples"]:
        raise ValueError("The Elementary cohort must be complete")
    feature_path = ELEMENTARY / "features.jsonl"
    records = [json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != {row["id"] for row in manifest["samples"]}:
        raise ValueError("Elementary features and manifest differ")
    if len({row["extractor_sha256"] for row in records}) != 1:
        raise ValueError("Multiple Elementary feature extractor versions")
    out = {role: [] for role in ROLES}
    for row in manifest["samples"]:
        record = by_id[row["id"]]
        frames = torch.tensor(record["features"], dtype=torch.float32)
        text = greek(row["text"])
        source = json.dumps([row["fps"], row["frame_paths"]], ensure_ascii=False, separators=(",", ":"))
        if (record["feature_schema"] != FEATURE_SCHEMA or float(record["fps"]) != 15
                or record["source_key"] != sha256(source.encode()).hexdigest()
                or frames.shape != (len(row["frame_paths"]), FEATURE_COUNT) or not frames.isfinite().all()
                or row["split"] != manifest["signer_roles"][row["signer"]] or text != row["text"]):
            raise ValueError(f"Invalid Elementary evidence: {row['id']}")
        out[row["split"]].append(Sample(row["id"], "elementary23", row["signer"], text, (), frames))
    return out


def check_splits(roles):
    all_samples = [row for role in ROLES for row in roles[role]]
    if len({row.id for row in all_samples}) != len(all_samples):
        raise ValueError("Duplicate recording IDs")
    seen_signers = {}
    for role in ROLES:
        for row in roles[role]:
            if seen_signers.setdefault((row.corpus, row.signer), role) != role:
                raise ValueError("Signer leakage")
            if not row.text or any(char not in CHARACTER_IDS for char in row.text):
                raise ValueError(f"Unsupported target text: {row.id}")
    source_texts = defaultdict(set)
    for role in ROLES:
        for row in roles[role]:
            if row.corpus == "elementary23":
                source_texts[role].add(row.text)
    if sum(map(len, source_texts.values())) != len(set().union(*source_texts.values())):
        raise ValueError("Elementary text crosses roles")


class Translator(nn.Module):
    def __init__(self, mean, scale, glosses):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.project = nn.Sequential(nn.Linear(FEATURE_COUNT, WIDTH), nn.LayerNorm(WIDTH), nn.GELU())
        self.encode_video = nn.GRU(WIDTH, WIDTH, batch_first=True, bidirectional=True)
        self.gloss = nn.Linear(2 * WIDTH, glosses)
        self.embed = nn.Embedding(len(TOKENS), WIDTH, padding_idx=PAD)
        self.decode_text = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.initial = nn.Linear(2 * WIDTH, WIDTH)
        self.keys = nn.Linear(2 * WIDTH, WIDTH)
        self.characters = nn.Linear(3 * WIDTH, len(TOKENS))

    def encode(self, frames, lengths):
        inputs = self.project((frames - self.mean) / self.scale)
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        sequence, _ = self.encode_video(packed)
        memory, _ = pad_packed_sequence(sequence, batch_first=True)
        mask = torch.arange(memory.shape[1])[None] < lengths[:, None]
        seed = self.initial(memory.sum(dim=1) / lengths[:, None]).tanh().unsqueeze(0)
        return memory, self.keys(memory), mask, seed

    def decode(self, tokens, memory, keys, mask, hidden):
        query, hidden = self.decode_text(self.embed(tokens), hidden)
        attention = (query @ keys.transpose(1, 2)) / WIDTH ** .5
        attended = attention.masked_fill(~mask[:, None], -torch.inf).softmax(-1) @ memory
        return self.characters(torch.cat((query, attended), dim=-1)), hidden

    def decode_batch(self, memory, keys, mask, hidden):
        tokens = torch.full((len(memory), 1), BOS, dtype=torch.long)
        ended = torch.zeros(len(memory), dtype=torch.bool)
        texts = [""] * len(memory)
        for _ in range(90):
            logits, hidden = self.decode(tokens, memory, keys, mask, hidden)
            values = logits[:, 0]
            values[:, :EOS] = -torch.inf
            chosen = values.argmax(dim=1)
            for index, char in enumerate(chosen.tolist()):
                if not ended[index] and char > EOS:
                    texts[index] += TOKENS[char]
            ended |= chosen == EOS
            if bool(ended.all()):
                break
            tokens = torch.where(ended, EOS, chosen)[:, None]
        return texts, ended.tolist()


def encoded_input(rows):
    lengths = torch.tensor([len(row.frames) for row in rows])
    return pad_sequence([row.frames for row in rows], batch_first=True), lengths


def teacher_targets(rows):
    tokens = [torch.tensor([BOS, *(CHARACTER_IDS[char] for char in row.text), EOS]) for row in rows]
    targets = pad_sequence(tokens, batch_first=True, padding_value=PAD)
    return targets[:, :-1], targets[:, 1:]


def nll(logits, targets):
    char_loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD, reduction="none")
    return char_loss.sum(1) / (targets != PAD).sum(1)


def hard_rivals(rows):
    word_sets = [set(row.text.split()) for row in rows]
    rivals = []
    for i, row in enumerate(rows):
        options = [j for j, other in enumerate(rows) if row.text != other.text and row.corpus == other.corpus
                   and row.signer == other.signer]
        if not options:
            rivals.append(i)
            continue
        rivals.append(max(options, key=lambda j: (len(word_sets[i] & word_sets[j]),
                                                   -abs(len(row.text) - len(rows[j].text)), -j)))
    return torch.tensor(rivals)


def batches(rows, generator, size):
    by_signer = defaultdict(list)
    for row in rows:
        by_signer[(row.corpus, row.signer)].append(row)
    groups = []
    for items in by_signer.values():
        items = generator.sample(items, len(items))
        groups.extend(items[start:start + size] for start in range(0, len(items), size))
    return generator.sample(groups, len(groups))


def train_epoch(model, optimizer, rows, vocabulary, generator, visual_loss):
    model.train()
    losses = []
    for group in batches(rows, generator, 12):
        frames, lengths = encoded_input(group)
        memory, keys, mask, seed = model.encode(frames, lengths)
        incoming, target = teacher_targets(group)
        logits, _ = model.decode(incoming, memory, keys, mask, seed)
        right = nll(logits, target)
        loss = right.mean()
        indices = [i for i, row in enumerate(group) if row.glosses]
        if indices:
            labels = [torch.tensor([vocabulary[gloss] for gloss in group[i].glosses]) for i in indices]
            predictions = model.gloss(memory[indices]).log_softmax(-1).transpose(0, 1)
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            gloss_loss = F.ctc_loss(predictions, torch.cat(labels), lengths[indices], label_lengths)
            loss += .30 * gloss_loss
        if visual_loss:
            rivals = hard_rivals(group)
            valid = rivals != torch.arange(len(group))
            if valid.any():
                wrong, _ = model.decode(incoming, memory[rivals], keys[rivals], mask[rivals], seed[:, rivals])
                loss += .40 * F.relu(.12 + right[valid] - nll(wrong, target)[valid]).mean()
        if not loss.isfinite():
            raise ValueError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 4)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


@torch.inference_mode()
def evaluate(model, samples):
    model.eval()
    outcomes = []
    for group in batches(samples, random.Random(SEED), 48):
        frames, lengths = encoded_input(group)
        memory, keys, mask, seed = model.encode(frames, lengths)
        decoded, ended = model.decode_batch(memory, keys, mask, seed)
        incoming, target = teacher_targets(group)
        right, _ = model.decode(incoming, memory, keys, mask, seed)
        rivals = hard_rivals(group)
        wrong, _ = model.decode(incoming, memory[rivals], keys[rivals], mask[rivals], seed[:, rivals])
        gaps = (nll(wrong, target) - nll(right, target)).tolist()
        for index, (row, text) in enumerate(zip(group, decoded, strict=True)):
            prediction = greek(text)
            errors = sum(edit_counts(row.text.split(), prediction.split()))
            outcomes.append({"id": row.id, "corpus": row.corpus, "signer": row.signer, "reference": row.text,
                             "prediction": prediction, "terminated": ended[index], "edits": errors,
                             "true_video_nll_advantage": gaps[index] if rivals[index] != index else None})
    return sorted(outcomes, key=lambda row: row["id"])


def metrics(rows):
    possible = sum(len(row["reference"].split()) for row in rows)
    pairs = [row["true_video_nll_advantage"] for row in rows if row["true_video_nll_advantage"] is not None]
    return {"clips": len(rows), "word_errors": sum(row["edits"] for row in rows), "reference_words": possible,
            "word_error_rate": sum(row["edits"] for row in rows) / possible,
            "exact_sentences": sum(row["reference"] == row["prediction"] for row in rows),
            "natural_endings": sum(row["terminated"] for row in rows),
            "true_video_preferred": sum(gap > 0 for gap in pairs), "same_signer_video_pairs": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--certh-only", action="store_true")
    parser.add_argument("--context", action="store_true", help="Include the frozen CERTH contextual train and test splits")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(2)
    certh = certh_sets()
    roles = {role: list(certh[role]) for role in ROLES}
    cohorts = {"certh": certh["test"]}
    if args.context:
        context = contextual_sets()
        if {row.text for row in context["test"]} & {row.text for row in certh["train"] + context["train"]}:
            raise ValueError("Contextual Greek reference crosses roles")
        for role in ("train", "test"):
            roles[role].extend(context[role])
        cohorts["certh_context"] = context["test"]
    if not args.certh_only:
        elementary = elementary_sets()
        for role in ROLES:
            roles[role].extend(elementary[role])
        cohorts["elementary23"] = elementary["test"]
    check_splits(roles)
    folder = "greek-translation-certh" if args.certh_only else "greek-translation-pilot"
    if args.context:
        folder += "-context"
    report_dir, artifacts = Path("reports") / folder, Path("artifacts") / folder
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    glosses = sorted({gloss for row in roles["train"] for gloss in row.glosses})
    vocabulary = {gloss: index for index, gloss in enumerate(glosses, 1)}
    train = torch.cat([row.frames for row in roles["train"]])
    train_words = {word for row in roles["train"] for word in row.text.split()}
    result = {
        "scope": "Offline direct Greek decoder pilot. CERTH test was previously observed and is diagnostic. "
                 "CERTH context test uses new reference texts. Elementary23 text and source signer are excluded across roles. "
                 "These are supplied clip boundaries, not dialogue.",
        "label_sources": {"certh": "Greek text and gloss order", "elementary23": "Greek text only"},
        "epochs": args.epochs, "seed": SEED,
        "counts": {role: {corpus: sum(row.corpus == corpus for row in roles[role])
                          for corpus in sorted({row.corpus for row in roles[role]})} for role in ROLES},
        "signers": {role: {corpus: sorted({row.signer for row in roles[role] if row.corpus == corpus})
                           for corpus in sorted({row.corpus for row in roles[role]})} for role in ROLES},
        "source_files": {},
        "models": {},
    }
    if not args.certh_only:
        for path in (ELEMENTARY / "manifest.json", ELEMENTARY / "features.jsonl"):
            result["source_files"][str(path)] = sha(path)
    if args.context:
        for path in (CONTEXT / "manifest.json", CONTEXT / "features.jsonl"):
            result["source_files"][str(path)] = sha(path)

    # Both arms are trained and frozen on development alone before opening test results.
    frozen = {}
    for name, visual_loss in (("dual_supervision", False), ("visual_contrast", True)):
        torch.manual_seed(SEED)
        model = Translator(train.mean(0), train.std(0, unbiased=False).clamp_min(.08), len(vocabulary) + 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        generator = random.Random(SEED)
        best = None
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, optimizer, roles["train"], vocabulary, generator, visual_loss)
            if epoch % 5 and epoch != args.epochs:
                continue
            val = metrics(evaluate(model, roles["val"]))
            score = val["word_error_rate"], -val["exact_sentences"], -val["true_video_preferred"]
            if best is None or score < best[0]:
                best = score, epoch, val, {key: value.detach().clone() for key, value in model.state_dict().items()}
            if epoch == 5 or epoch % 15 == 0:
                print(json.dumps({"model": name, "epoch": epoch, "loss": round(loss, 4), "val": val}), flush=True)
        _, epoch, val, weights = best
        model.load_state_dict(weights)
        artifact = artifacts / f"{name}.pt"
        torch.save({"state_dict": weights, "characters": TOKENS, "glosses": [None, *glosses],
                    "feature_schema": FEATURE_SCHEMA, "width": WIDTH, "selected_epoch": epoch}, artifact)
        frozen[name] = model
        result["models"][name] = {"epoch": epoch, "validation": val, "artifact_sha256": sha(artifact)}
    freeze = {"criterion": ["minimum validation WER", "maximum validation exact match", "maximum matched-video preference"],
              "models": result["models"], "seed": SEED, "sources": result["source_files"]}
    (report_dir / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    for name, model in frozen.items():
        reports = {}
        for corpus, rows in cohorts.items():
            predictions = evaluate(model, rows)
            words = [word for row in rows for word in row.text.split()]
            summary = {**metrics(predictions), "reference_words_seen_in_training": sum(word in train_words for word in words)}
            reports[corpus] = {"summary": summary, "predictions": predictions}
        result["models"][name]["diagnostic"] = reports
        print(json.dumps({"model": name, "diagnostic": {key: value["summary"] for key, value in reports.items()}}), flush=True)
    (report_dir / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

