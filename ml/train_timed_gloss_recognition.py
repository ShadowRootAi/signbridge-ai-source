"""Compare whole-stream gloss recognition with and without CERTH visual transfer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

from train_continuous import edit_counts
from train_native_greek import (
    FPS, align_takes, check_fit_split, ctc_loss, embed_stream, frame_spans, load_streams,
    resample, save_checkpoint, visual_checkpoint, warp_timeline,
)
from train_grounded_greek import load_visual, write
from train_greek_translation_pilot import sha


DATA = Path('data/sartinas-dialogue-fit-expanded')
REPORT = Path('reports/timed-gloss')
ARTIFACTS = Path('artifacts/timed-gloss')
ARMS = ('landmarks', 'certh_transfer')
SEED = 746
WIDTH = 80
FLOORS = (0., .15, .30, .45, .60)


def normalized(label):
    return unicodedata.normalize('NFC', label.strip().upper())


class GlossRecognizer(nn.Module):
    def __init__(self, inputs, classes, *, factorized=False):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, WIDTH), nn.GELU())
        self.layers = nn.ModuleList([
            nn.Conv1d(WIDTH, WIDTH, 3, dilation=dilation, padding=dilation)
            for dilation in (1, 2, 4, 8)
        ])
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in self.layers)
        self.dropout = nn.Dropout(.12)
        self.output = nn.Linear(WIDTH, classes)
        self.timing = nn.Linear(WIDTH, 2) if factorized else None

    def forward(self, features):
        memory = self.input(features)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            context = layer(memory.T[None])[0].T
            memory = norm(memory + self.dropout(F.gelu(context)))
        return memory, self.output(memory)


def labeled_spans(stream, count, classes):
    return [(start, end, classes.get(normalized(label), -1))
            for start, end, label in frame_spans(stream.glosses, count)]


def frame_targets(count, spans):
    labels = torch.zeros(count, dtype=torch.long)
    for start, end, target in spans:
        overlap = labels[start:end] != 0
        labels[start:end] = target
        labels[start:end].masked_fill_(overlap, -100)
    return labels


def train_loss(model, features, streams, classes, weights):
    frame_losses, sequence_losses, span_scores, span_targets, pooled, owners = [], [], [], [], [], []
    for owner, (values, stream) in enumerate(zip(features, streams, strict=True)):
        memory, scores = model(values)
        spans = labeled_spans(stream, len(values), classes)
        labels = frame_targets(len(values), spans)
        if model.timing is None:
            frame_losses.append(F.cross_entropy(scores, labels, weight=weights, ignore_index=-100))
            sequence_losses.append(ctc_loss(scores[None], [tuple(label for _, _, label in spans)], [len(values)]))
        else:
            visible = labels > 0
            frame_losses.append(F.cross_entropy(scores[visible, 1:], labels[visible] - 1, weight=weights[1:]))
            time_labels = torch.zeros((len(scores), 2))
            for start, end, _ in spans:
                time_labels[start:end, 0] = 1.
                time_labels[max(0, start - 1):start + 2, 1].clamp_(min=.35)
                time_labels[start, 1] = 1.
            sequence_losses.append(F.binary_cross_entropy_with_logits(model.timing(memory), time_labels,
                                                                       pos_weight=torch.tensor([1.5, 4.])))
        for start, end, label in spans:
            span_scores.append(scores[start:end].mean(0))
            span_targets.append(label)
            pooled.append(memory[start:end].mean(0))
            owners.append((owner, 0, 1))
    target = torch.tensor(span_targets)
    logits = torch.stack(span_scores)
    sign_loss = (F.cross_entropy(logits, target, weight=weights) if model.timing is None
                 else F.cross_entropy(logits[:, 1:], target - 1, weight=weights[1:]))
    alignment = align_takes(torch.stack(pooled)[:, None], torch.ones((len(pooled), 1), dtype=torch.bool),
                            owners, [(label,) for label in span_targets])
    return .5 * torch.stack(frame_losses).mean() + torch.stack(sequence_losses).mean() + .7 * sign_loss + .12 * alignment


def greedy_runs(scores, floor):
    probabilities, ids = scores.softmax(-1).max(-1)
    spans, previous = [], 0
    for index, (label, probability) in enumerate(zip(ids.tolist(), probabilities.tolist(), strict=True)):
        if probability < floor:
            label = 0
        if label == 0:
            previous = 0
            continue
        if label == previous:
            start, _, key, peak = spans[-1]
            spans[-1] = (start, index + 1, key, max(peak, probability))
        else:
            spans.append((index, index + 1, label, probability))
        previous = label
    return spans


def recognize_segments(scores, timing, policy):
    active, onset = timing.sigmoid().unbind(1)
    above = active >= policy['activity']
    if len(above) > 2:
        above[1:-1] |= above[:-2] & above[2:]
    boundaries = F.pad(above.to(torch.int8), (1, 1)).diff()
    starts = torch.where(boundaries == 1)[0].tolist()
    ends = torch.where(boundaries == -1)[0].tolist()
    spans = []
    for start, end in zip(starts, ends, strict=True):
        candidates = [index for index in range(start + 2, end - 1)
                      if onset[index] >= policy['onset'] and onset[index - 1] < onset[index] >= onset[index + 1]]
        chosen = [start, end]
        for candidate in sorted(candidates, key=lambda index: (-float(onset[index]), index)):
            if all(abs(candidate - other) >= 2 for other in chosen):
                chosen.append(candidate)
        chosen.sort()
        for left, right in zip(chosen, chosen[1:]):
            if right - left < 2:
                continue
            probability, index = scores[left:right, 1:].mean(0).softmax(-1).max(0)
            if probability >= policy['confidence']:
                spans.append((left, right, int(index) + 1, float(probability)))
    return spans


def interval_recognition(stream, scores, vocabulary, classes):
    rows = []
    for (start, end, label), (left, right, target) in zip(
            stream.glosses, labeled_spans(stream, len(scores), classes), strict=True):
        logits = scores[left:right, 1:].mean(0)
        ranked = [vocabulary[index + 1] for index in logits.topk(min(5, len(logits))).indices.tolist()]
        rows.append({'start_s': start, 'end_s': end, 'reference': normalized(label),
                     'seen': target > 0, 'ranked': ranked})
    return rows


def evaluate_scores(stream, scores, vocabulary, classes, policy, *, timing=None, details=False):
    runs = (greedy_runs(scores, policy['confidence']) if timing is None
            else recognize_segments(scores, timing, policy))
    expected = [normalized(label) for _, _, label in stream.glosses]
    predicted = [vocabulary[index] for _, _, index, _ in runs]
    edits = edit_counts(expected, predicted)
    intervals = interval_recognition(stream, scores, vocabulary, classes)
    known = [row for row in intervals if row['seen']]
    matched = sum(row['ranked'][0] == row['reference'] for row in known)
    result = {'reference_glosses': len(expected), 'predicted_glosses': len(predicted), 'edits': sum(edits),
              'substitutions': edits[0], 'deletions': edits[1], 'insertions': edits[2],
              'gloss_error_rate': sum(edits) / len(expected), 'known_glosses': len(known),
              'correct_at_given_interval': matched,
              'top5_at_given_interval': sum(row['reference'] in row['ranked'] for row in known)}
    if details:
        result['detected_glosses'] = [
            {'start_s': start / FPS, 'end_s': end / FPS, 'gloss': vocabulary[index], 'score': peak}
            for start, end, index, peak in runs
        ]
        result['given_interval_diagnostic'] = intervals
    return result


def summary(results):
    totals = {key: sum(row[key] for row in results) for key in (
        'reference_glosses', 'predicted_glosses', 'edits', 'substitutions', 'deletions', 'insertions',
        'known_glosses', 'correct_at_given_interval', 'top5_at_given_interval')}
    return {**totals, 'gloss_error_rate': totals['edits'] / totals['reference_glosses']}


def values_for(arm, streams, visual):
    if arm == 'landmarks':
        return [resample(row.frames, row.fps, FPS) for row in streams]
    return [embed_stream(visual, row).clone() for row in streams]


def dev_policies(factorized):
    if not factorized:
        return [{'confidence': floor} for floor in FLOORS]
    return [{'activity': activity, 'onset': onset, 'confidence': confidence}
            for activity in (.35, .50, .65) for onset in (.35, .55, .70) for confidence in (.0, .08, .15, .30)]


def sources(streams):
    return [{'id': row.id, 'signer': row.signer, 'scenario': row.scenario,
             'glosses': len(row.glosses), 'source_key': row.source_key} for row in streams]


def fit(args):
    if args.freeze.exists():
        raise ValueError(f'Existing frozen experiment: {args.freeze}')
    streams, provenance = load_streams(args.data, ('train', 'val'))
    train, dev = check_fit_split(streams, include_n2=True)
    if any(not row.glosses for row in streams):
        raise ValueError('Official timed gloss labels are required for every fitting stream')
    counts = Counter(normalized(label) for row in train for _, _, label in row.glosses)
    vocabulary = ['', *sorted(counts)]
    classes = {label: index for index, label in enumerate(vocabulary) if index}
    weights = torch.tensor([.20, *[min(2., (4. / counts[label]) ** .5) for label in vocabulary[1:]]])
    groups = [[index for index, row in enumerate(train) if row.scenario == scenario]
              for scenario in sorted({row.scenario for row in train})]
    visual_path, visual_hash = visual_checkpoint()
    visual, _ = load_visual(visual_path, visual_hash)
    result = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        rng = random.Random(SEED)
        train_features, dev_features = (values_for(arm, group, visual) for group in (train, dev))
        model = GlossRecognizer(train_features[0].shape[1], len(vocabulary), factorized=args.factorized)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0017, weight_decay=.015)
        best, history = None, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for group in rng.sample(groups, len(groups)):
                timed = [warp_timeline(train[i], train_features[i], rng) if epoch % 3
                         else (train[i], train_features[i]) for i in group]
                rows, values = map(list, zip(*timed, strict=True))
                loss = train_loss(model, values, rows, classes, weights)
                if not loss.isfinite():
                    raise ValueError(f'Nonfinite timed gloss loss: {arm}, epoch {epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch % 5 and epoch not in (1, args.epochs):
                continue
            model.eval()
            with torch.inference_mode():
                output_rows = [model(value) for value in dev_features]
                timing_rows = [model.timing(memory) if model.timing is not None else None for memory, _ in output_rows]
                metrics = [(policy, summary([
                    evaluate_scores(row, scores, vocabulary, classes, policy, timing=timing)
                    for row, (_, scores), timing in zip(dev, output_rows, timing_rows, strict=True)
                ])) for policy in dev_policies(args.factorized)]
            policy, selected = min(metrics, key=lambda item: (item[1]['edits'], -item[1]['correct_at_given_interval'],
                                                               item[1]['insertions']))
            key = selected['edits'], -selected['correct_at_given_interval'], epoch
            record = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'policy': policy, 'dev': selected}
            history.append(record)
            print(json.dumps({'arm': arm, **record}, ensure_ascii=False), flush=True)
            if best is None or key < best[0]:
                best = key, record, {name: value.detach().clone() for name, value in model.state_dict().items()}
        _, selected, state = best
        checkpoint = args.artifacts / f'{arm}.pt'
        save_checkpoint(checkpoint, {'state': state, 'inputs': train_features[0].shape[1], 'labels': vocabulary,
                                     'factorized': args.factorized})
        result[arm] = {'checkpoint': checkpoint.as_posix(), 'checkpoint_sha256': sha(checkpoint),
                       'selected': selected, 'history': history}
    write(args.freeze, {'scope': 'Standalone recognition trained on full timed native streams; no Greek text targets.',
                        'objective': 'timing_and_gloss' if args.factorized else 'ctc_and_gloss',
                        'data': provenance, 'train': sources(train), 'development': sources(dev), 'seed': SEED,
                        'vocabulary': vocabulary[1:], 'visual_path': visual_path.as_posix(), 'visual_sha256': visual_hash,
                        'arms': result})


def evaluate(args):
    frozen = json.loads(args.freeze.read_bytes())
    streams, provenance = load_streams(args.data, ('test',))
    if any(not row.glosses for row in streams):
        raise ValueError('Evaluation requires source-verified gloss intervals')
    fitted = frozen['train'] + frozen['development']
    for key in ('id', 'signer', 'scenario', 'source_key'):
        if {row[key] for row in fitted} & {getattr(row, key) for row in streams}:
            raise ValueError(f'Evaluation overlaps fitting data by {key}')
    visual, _ = load_visual(Path(frozen['visual_path']), frozen['visual_sha256'])
    result = {}
    for arm, config in frozen['arms'].items():
        checkpoint = Path(config['checkpoint'])
        if sha(checkpoint) != config['checkpoint_sha256']:
            raise ValueError(f'Frozen {arm} checkpoint changed')
        saved = torch.load(checkpoint, weights_only=True)
        vocabulary = saved['labels']
        classes = {label: index for index, label in enumerate(vocabulary) if index}
        model = GlossRecognizer(saved['inputs'], len(vocabulary), factorized=saved.get('factorized', False))
        model.load_state_dict(saved['state'])
        model.eval()
        items = []
        with torch.inference_mode():
            for row, features in zip(streams, values_for(arm, streams, visual), strict=True):
                memory, scores = model(features)
                timing = model.timing(memory) if model.timing is not None else None
                selected = config['selected']
                policy = selected.get('policy') or {'confidence': selected['confidence_floor']}
                metric = evaluate_scores(row, scores, vocabulary, classes, policy, timing=timing, details=True)
                items.append({'id': row.id, 'signer': row.signer, 'scenario': row.scenario, **metric})
        result[arm] = {'summary': summary(items), 'recordings': items}
        print(json.dumps({'arm': arm, **result[arm]['summary']}, ensure_ascii=False), flush=True)
    write(args.output, {'scope': 'Uncut inference. Official gloss intervals enter only the post-inference scoring diagnostic.',
                        'cohort': args.cohort, 'freeze_sha256': sha(args.freeze), 'data': provenance, 'arms': result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--factorized', action='store_true', help='Supervise sign identity, activity and onsets separately')
    parser.add_argument('--freeze', type=Path, default=REPORT / 'freeze.json')
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--output', type=Path, default=REPORT / 'evaluation.json')
    parser.add_argument('--cohort', choices=('retrospective', 'new-recording'), default='retrospective')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Epoch count must be positive')
    torch.set_num_threads(3)
    (fit if args.phase == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
