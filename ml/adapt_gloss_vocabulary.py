"""Adapt a verified CERTH isolated-sign recognizer to continuous Sartinas recordings."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from train_continuous import edit_counts
from train_continuous_library import isolated_rows
from train_greek_translation_pilot import sha
from train_grounded_greek import load_visual, write
from train_isolated_large_vocabulary import load_checkpoint, resample_features
from train_native_greek import FPS, check_fit_split, frame_spans, load_streams, resample, save_checkpoint
from train_timed_gloss_recognition import (
    DATA, REPORT, GlossRecognizer, normalized, recognize_segments, sources, values_for,
)


FOCUS = ('ΡΑΝΤΕΒΟΥ', 'ΜΗΝΑΣ', 'ΩΡΑ', 'ΔΕΥΤΕΡΑ', 'ΜΕΘΑΥΡΙΟ')
SEED = 1153


def source_model():
    model, labels, freeze = load_checkpoint('expanded')
    vocabulary = [normalized(label) for label in labels]
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError('Canonical CERTH glosses are ambiguous')
    return model, vocabulary, freeze


def load_timing(path):
    freeze = json.loads(path.read_bytes())
    arm = freeze['arms']['certh_transfer']
    checkpoint = Path(arm['checkpoint'])
    if sha(checkpoint) != arm['checkpoint_sha256']:
        raise ValueError('Frozen timing checkpoint changed')
    data = torch.load(checkpoint, weights_only=True)
    timing = GlossRecognizer(data['inputs'], len(data['labels']), factorized=True)
    timing.load_state_dict(data['state'])
    timing.eval()
    visual, _ = load_visual(Path(freeze['visual_path']), freeze['visual_sha256'])
    return timing, visual, {**arm['selected']['policy'], 'confidence': 0.}


def clip_embeddings(model, clips):
    if not clips:
        return torch.empty((0, model.head.in_features))
    hidden = []
    hook = model.embedding.register_forward_hook(lambda _module, _inputs, output: hidden.append(output.detach()))
    try:
        with torch.no_grad():
            for start in range(0, len(clips), 96):
                batch = torch.stack([resample_features(clip) for clip in clips[start:start + 96]])
                model(batch)
    finally:
        hook.remove()
    return torch.cat(hidden)


def raw_intervals(row, frames):
    return frame_spans(row.glosses, len(frames))


@torch.no_grad()
def evaluation_features(row, model, timing, visual, policy):
    raw = resample(row.frames, row.fps, FPS)
    features = values_for('certh_transfer', [row], visual)[0]
    memory, scores = timing(features)
    spans = recognize_segments(scores, timing.timing(memory), policy)
    reference = raw_intervals(row, raw)
    given = clip_embeddings(model, [raw[start:end] for start, end, _ in reference])
    uncut = clip_embeddings(model, [raw[start:end] for start, end, _, _ in spans])
    return given, uncut, spans


def measure(row, encoded, head, vocabulary, floor, *, details=False):
    given, uncut, spans = encoded
    with torch.no_grad():
        given_ids = head(given).argmax(-1).tolist()
        probability, ids = head(uncut).softmax(-1).max(-1)
    expected = [normalized(label) for _, _, label in row.glosses]
    oracle = [vocabulary[index] for index in given_ids]
    output = [{'start_s': start / FPS, 'end_s': end / FPS, 'gloss': vocabulary[index], 'score': score}
              for (start, end, _, _), index, score in zip(spans, ids.tolist(), probability.tolist(), strict=True)
              if score >= floor]
    predicted = [item['gloss'] for item in output]
    substitutions, deletions, insertions = edit_counts(expected, predicted)
    metrics = {'reference': len(expected), 'predicted': len(predicted), 'edits': substitutions + deletions + insertions,
               'correct_at_given_interval': sum(a == b for a, b in zip(expected, oracle, strict=True)),
               'label_coverage': sum(label in vocabulary for label in expected)}
    if details:
        metrics.update(prediction=output, focus=[
            {'reference': expected[i], 'given_interval_guess': oracle[i],
             'automatic_predictions': [item for item in output if item['start_s'] < end and item['end_s'] > start]}
            for i, (start, end, _) in enumerate(row.glosses) if expected[i] in FOCUS])
    return metrics


def aggregate(rows):
    return {key: sum(row[key] for row in rows)
            for key in ('reference', 'predicted', 'edits', 'correct_at_given_interval', 'label_coverage')}


def best_floor(dev, features, head, vocabulary):
    records = [(floor, aggregate([measure(row, feature, head, vocabulary, floor)
                                  for row, feature in zip(dev, features, strict=True)]))
               for floor in (0., .08, .15, .25, .40, .55)]
    return min(records, key=lambda item: (item[1]['edits'], -item[1]['correct_at_given_interval'], item[1]['predicted']))


def fit(args):
    if args.freeze.exists():
        raise ValueError(f'Already frozen: {args.freeze}')
    base, labels, base_record = source_model()
    timing, visual, policy = load_timing(args.timing)
    streams, evidence = load_streams(args.data, ('train', 'val'))
    train, dev = check_fit_split(streams, include_n2=True)
    dev_features = [evaluation_features(row, base, timing, visual, policy) for row in dev]
    native_clips, native_labels = [], []
    for row in train:
        raw = resample(row.frames, row.fps, FPS)
        for start, end, label in raw_intervals(row, raw):
            native_clips.append(raw[start:end])
            native_labels.append(normalized(label))
    vocabulary = labels + sorted(set(native_labels) - set(labels))
    lookup = {label: index for index, label in enumerate(vocabulary)}
    native = clip_embeddings(base, native_clips)
    y_native = torch.tensor([lookup[label] for label in native_labels])
    certh, _, source_evidence = isolated_rows()
    grouped = defaultdict(list)
    for row in certh:
        grouped[normalized(row.glosses[0])].append(row)
    rng = random.Random(SEED)
    selected = [row for label, rows in sorted(grouped.items())
                for row in rng.sample(rows, min(len(rows), 24 if label in FOCUS else 5))]
    source = clip_embeddings(base, [row.frames for row in selected])
    y_source = torch.tensor([lookup[normalized(row.glosses[0])] for row in selected])
    torch.manual_seed(SEED)
    head = nn.Linear(base.head.in_features, len(vocabulary))
    with torch.no_grad():
        head.weight[:len(labels)].copy_(base.head.weight)
        head.bias[:len(labels)].copy_(base.head.bias)
    counts = Counter(native_labels)
    weights = torch.tensor([min(2., (4. / max(1, counts[label])) ** .5) for label in vocabulary])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.015)
    best = None
    for epoch in range(1, 51):
        order = rng.sample(range(len(native)), len(native))
        for start in range(0, len(order), 48):
            group = order[start:start + 48]
            rehearsal = rng.sample(range(len(source)), 64)
            loss = F.cross_entropy(head(native[group]), y_native[group], weight=weights)
            loss += .55 * F.cross_entropy(head(source[rehearsal]), y_source[rehearsal])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 5 and epoch != 1:
            continue
        floor, score = best_floor(dev, dev_features, head, vocabulary)
        key = score['edits'], -score['correct_at_given_interval'], epoch
        if best is None or key < best[0]:
            best = key, {'epoch': epoch, 'confidence_floor': floor, 'dev': score}, {
                name: value.detach().clone() for name, value in head.state_dict().items()
            }
    _, selected_epoch, state = best
    baseline_floor, baseline = best_floor(dev, dev_features, base.head, labels)
    save_checkpoint(args.checkpoint, {'state': state, 'labels': vocabulary})
    write(args.freeze, {'data': evidence, 'train': sources(train), 'dev': sources(dev), 'timing_sha256': sha(args.timing),
                        'base_sha256': base_record['checkpoint_sha256'], 'checkpoint_sha256': sha(args.checkpoint),
                        'adapted': selected_epoch, 'baseline': {'confidence_floor': baseline_floor, 'dev': baseline},
                        'native_clips': len(native), 'source_clips': len(source), 'certh_sources': source_evidence,
                        'focus_source_counts': {label: sum(normalized(row.glosses[0]) == label for row in selected)
                                                for label in FOCUS}})
    print(json.dumps({'baseline': baseline, 'adapted': selected_epoch}, ensure_ascii=False), flush=True)


def evaluate(args):
    freeze = json.loads(args.freeze.read_bytes())
    if sha(args.checkpoint) != freeze['checkpoint_sha256'] or sha(args.timing) != freeze['timing_sha256']:
        raise ValueError('Adapted model or timing changed')
    base, labels, base_record = source_model()
    if base_record['checkpoint_sha256'] != freeze['base_sha256']:
        raise ValueError('CERTH recognizer changed')
    streams, proof = load_streams(args.data, ('test',))
    fit_rows = freeze['train'] + freeze['dev']
    for key in ('id', 'signer', 'scenario', 'source_key'):
        if {row[key] for row in fit_rows} & {getattr(row, key) for row in streams}:
            raise ValueError(f'Holdout overlaps training by {key}')
    timing, visual, policy = load_timing(args.timing)
    adapted = torch.load(args.checkpoint, weights_only=True)
    head = nn.Linear(base.head.in_features, len(adapted['labels']))
    head.load_state_dict(adapted['state'])
    features = [evaluation_features(row, base, timing, visual, policy) for row in streams]
    results = {}
    for name, arm, words in [('baseline', base.head, labels), ('adapted', head, adapted['labels'])]:
        rows = [{'id': row.id, **measure(row, encoded, arm, words, freeze[name]['confidence_floor'], details=True)}
                for row, encoded in zip(streams, features, strict=True)]
        results[name] = {'summary': aggregate(rows), 'recordings': rows}
        print(json.dumps({'arm': name, **results[name]['summary']}, ensure_ascii=False), flush=True)
    write(args.output, {'scope': 'Shared automatically proposed spans and a separate given-interval scoring diagnostic.',
                        'data': proof, 'freeze_sha256': sha(args.freeze), 'arms': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--timing', type=Path, default=REPORT / 'factorized-freeze.json')
    parser.add_argument('--freeze', type=Path, default=REPORT / 'vocabulary-freeze.json')
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/timed-gloss/adapted-vocabulary.pt'))
    parser.add_argument('--output', type=Path, default=REPORT / 'n8-vocabulary.json')
    args = parser.parse_args()
    torch.set_num_threads(3)
    (fit if args.phase == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
