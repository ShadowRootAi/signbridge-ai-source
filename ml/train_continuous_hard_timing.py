"""Fine-tune a clean CTC checkpoint using visually estimated same-source windows."""

import argparse
import json
from pathlib import Path
import random
import sys
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from evaluate_synthetic_ctc_transfer import POSE_CACHE, frame_feature_array, verify_feature_math
from train_continuous import reference_lookup, replay, summarize
from train_continuous_hard import BATCH_SIZE, SEED, OrderAwareCTC, datasets, digest, read_selected, write
from train_continuous_hard_isolated import CHECKPOINT as INITIAL, FREEZE as INITIAL_FREEZE, METADATA
from train_continuous_multitask import ORDER_WEIGHT, PRESENCE_WEIGHT, loss_components, warp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from prepare_synthetic_continuous_train_plan import (
    ISOLATE_PATH, ROOT, SENTENCE_PATH, SPLIT, SUPPLEMENTARY, grouped_labels, read_annotations, read_labels,
)


REPORT = Path('reports/continuous-hard')
ARTIFACTS = Path('artifacts/continuous-hard')
FREEZE = REPORT / 'timing-freeze.json'
FEATURE_COLUMNS = [*range(174, 182), 0, 1, 84, 85]
MAX_DISTANCE, MIN_REVERSE_GAIN, ANCHOR_WEIGHT = .15, .04, .30
EPOCHS, LR = 12, .0008


def align_windows(sequence, segments):
    """Find the lowest-cost ordered nonoverlapping windows and allow unlabelled gaps."""
    landmarks = sequence[:, FEATURE_COLUMNS]
    states = {0: (0., ())}
    for segment in segments:
        prefix, best = [], None
        for frame in range(len(sequence)):
            state = states.get(frame)
            if state is not None and (best is None or state[0] < best[0]):
                best = state
            prefix.append(best)
        updated = {}
        for phase in (0, 1):
            clip = segment[phase::2, FEATURE_COLUMNS]
            if not 0 < len(clip) <= len(landmarks):
                continue
            windows = np.lib.stride_tricks.sliding_window_view(landmarks, len(clip), axis=0).transpose(0, 2, 1)
            distances = np.abs(windows - clip).mean(axis=(1, 2))
            for start, cost in enumerate(distances):
                previous = prefix[start]
                if previous is None:
                    continue
                end = start + len(clip)
                score = previous[0] + float(cost)
                if end not in updated or score < updated[end][0]:
                    updated[end] = score, (*previous[1], (start, end))
        states = updated
    if not states:
        return float('inf'), ()
    total, windows = min(states.values(), key=lambda result: result[0])
    return total / len(segments), windows


def training_anchors(train, excluded, source_freeze):
    if digest(POSE_CACHE) != source_freeze['isolated']['pose_cache_sha256'] or digest(METADATA) != source_freeze['isolated']['isolated_metadata_sha256']:
        raise ValueError('The original isolated sources changed')
    canonical, _ = read_annotations(ROOT / SPLIT, ROOT / SUPPLEMENTARY)
    sentences = grouped_labels(canonical.items(), SENTENCE_PATH)
    with ZipFile(METADATA) as archive:
        original = dict(read_labels(archive, ['GSL_split/GSL_isolated/train_greek_iso.csv']))
    isolates = grouped_labels(original.items(), ISOLATE_PATH)
    with np.load(POSE_CACHE, allow_pickle=False) as cache:
        if cache['split'].item() != 'train-prototypes':
            raise ValueError('The pose cache must contain only the official train prototype cohort')
        raw = cache['keypoints']
        offsets = cache['offsets']
        ids = {identity: slice(start, end) for identity, start, end in zip(cache['ids'], offsets[:-1], offsets[1:], strict=True)}
    pose_features = frame_feature_array(raw)
    verify_feature_math(raw, pose_features)
    aligned, diagnostics = {}, []
    for row in train:
        source = SENTENCE_PATH.fullmatch(row.source.split('/', 1)[1])
        session, index = source['session'], int(source['index'])
        if session not in isolates or source['scenario'] in excluded:
            continue
        ordered_sentences = sorted(sentences[session].items())
        ordered_isolates = sorted(isolates[session].items())
        if ([i for i, _ in ordered_sentences] != list(range(len(ordered_sentences)))
                or [i for i, _ in ordered_isolates] != list(range(len(ordered_isolates)))
                or tuple(gloss for _, phrase in ordered_sentences for gloss in phrase) != tuple(gloss for _, gloss in ordered_isolates)
                or sentences[session][index] != row.glosses):
            continue
        offset = sum(len(phrase) for number, phrase in ordered_sentences if number < index)
        segment_ids = [f'{session}_glosses/glosses{i:04d}' for i in range(offset, offset + len(row.glosses))]
        if not all(identity in ids for identity in segment_ids):
            continue
        segments = [pose_features[ids[identity]] for identity in segment_ids]
        forward, windows = align_windows(row.frames.numpy(), segments)
        reverse, _ = align_windows(row.frames.numpy()[::-1], segments)
        if not np.isfinite(forward) or not np.isfinite(reverse):
            continue
        keep = forward <= MAX_DISTANCE and reverse - forward >= MIN_REVERSE_GAIN
        diagnostics.append({'id': row.id, 'signer': row.signer, 'isolated_sources': segment_ids, 'glosses': row.glosses,
                            'forward_distance': forward, 'reversed_distance': reverse, 'frames': len(row.frames),
                            'windows_half_open_at_15hz': windows, 'kept': keep})
        if keep:
            aligned[row.id] = windows
    if not aligned:
        raise ValueError('No training clips have sufficiently strong temporal alignment')
    return aligned, diagnostics


def temporal_loss(model, rows, anchors, indexes):
    lengths = torch.tensor([len(row.frames) for row in rows])
    with torch.random.fork_rng(devices=[]):
        scores = model(pad_sequence([row.frames for row in rows], batch_first=True), lengths)
    losses = []
    for logits, row in zip(scores, rows, strict=True):
        for (start, end), gloss in zip(anchors[row.id], row.glosses, strict=True):
            positions = torch.arange(start, end, dtype=logits.dtype)
            width = max(1., (end - start) / 4.)
            log_weights = torch.log_softmax(-.5 * ((positions - (start + end - 1) / 2.) / width).square(), dim=0)
            losses.append(-torch.logsumexp(logits[start:end, indexes[gloss]] + log_weights, dim=0))
    return torch.stack(losses).mean()


def load_model(checkpoint):
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    weights, labels = state['state_dict'], state['labels']
    model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    model.load_state_dict(weights)
    return model, labels


def fine_tune(train, dev, anchors):
    torch.manual_seed(SEED)
    model, labels = load_model(INITIAL)
    indexes = {gloss: index for index, gloss in enumerate(labels) if index}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    generator = random.Random(SEED)
    lookup = reference_lookup(train)
    history, best = [], None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        shuffled = generator.sample(train, len(train))
        for start in range(0, len(shuffled), BATCH_SIZE):
            batch = shuffled[start:start + BATCH_SIZE]
            ctc, presence, order = loss_components(model, [warp(row, generator) for row in batch], indexes)
            loss = ctc + PRESENCE_WEIGHT * presence + ORDER_WEIGHT * order
            anchored = [row for row in batch if row.id in anchors]
            if anchored:
                loss = loss + ANCHOR_WEIGHT * temporal_loss(model, anchored, anchors, indexes)
            if not loss.isfinite():
                raise ValueError('Non-finite timing objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        score = summarize(replay(model, dev, labels, lookup))
        key = (score['gloss_wer'], -score['gloss_sequence_exact'], score['deletions'])
        history.append({'epoch': epoch, 'gloss_wer': key[0], 'gloss_sequence_exact': -key[1], 'deletions': key[2]})
        if best is None or key < best[0]:
            best = key, epoch, {name: value.detach().clone() for name, value in model.state_dict().items()}
    return labels, history, best


def fit(manifest, features):
    if FREEZE.exists():
        raise ValueError(f'Existing timing freeze: {FREEZE}')
    _, train, dev, provenance = datasets(manifest, features)
    initial = json.loads(INITIAL_FREEZE.read_text(encoding='utf-8'))
    if initial['provenance'] != provenance or digest(INITIAL) != initial['checkpoint_sha256']:
        raise ValueError('Initial training split or checkpoint changed')
    reserved = [row for row in json.loads(manifest.read_text(encoding='utf-8'))['samples'] if row['split'] != 'train']
    excluded = {row['template'].split('/')[0] for row in reserved} | {row.id.split('_')[0] for row in dev}
    anchors, diagnostic = training_anchors(train, excluded, initial)
    alignment = REPORT / 'timing-alignments.json'
    write(alignment, {'scope': 'Training-only estimated windows, without original per-gloss frame timestamps.',
                      'excluded_scenarios': sorted(excluded), 'max_mean_distance': MAX_DISTANCE,
                      'minimum_forward_gain_over_reversed': MIN_REVERSE_GAIN, 'compared_clips': len(diagnostic),
                      'kept_clips': len(anchors), 'kept_segments': sum(map(len, anchors.values())), 'rows': diagnostic})
    print(f'{len(anchors)} train clips with {sum(map(len, anchors.values()))} timing anchors', flush=True)
    variants = {}
    for name, selected_anchors in (('continuation_control', {}), ('timing_anchors', anchors)):
        labels, history, (key, epoch, weights) = fine_tune(train, dev, selected_anchors)
        path = ARTIFACTS / f'{name}.pt'
        torch.save({'state_dict': weights, 'labels': labels}, path)
        variants[name] = {'checkpoint_sha256': digest(path), 'selected_epoch': epoch, 'development_key': key,
                          'history': history}
        print(f'{name}: epoch {epoch}, dev key {key}', flush=True)
    write(FREEZE, {'provenance': provenance, 'initial_checkpoint_sha256': initial['checkpoint_sha256'],
                   'alignment_sha256': digest(alignment), 'continuation_epochs': EPOCHS, 'learning_rate': LR,
                   'anchor_weight': ANCHOR_WEIGHT, 'arms': variants})


def score(manifest, features):
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    _, _, _, provenance = datasets(manifest, features)
    if provenance != freeze['provenance']:
        raise ValueError('Frozen development provenance changed')
    held, feature_hash = read_selected(manifest, features, ('test',))
    output = {'scope': 'Retrospective hard CERTH cohort with supplied clip boundaries.',
              'freeze_sha256': digest(FREEZE), 'test_features_sha256': feature_hash, 'arms': {}}
    for name, record in freeze['arms'].items():
        path = ARTIFACTS / f'{name}.pt'
        if digest(path) != record['checkpoint_sha256']:
            raise ValueError(f'Frozen weights changed for {name}')
        model, labels = load_model(path)
        rows = replay(model, held['test'], labels, {})
        output['arms'][name] = {'summary': summarize(rows), 'predictions': rows}
        print(json.dumps({'arm': name, **output['arms'][name]['summary']}, ensure_ascii=False), flush=True)
    write(REPORT / 'timing-benchmark.json', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'score'))
    parser.add_argument('--manifest', type=Path, default=Path('data/continuous-hard-split/manifest.json'))
    parser.add_argument('--features', type=Path, default=Path('data/continuous-hard-split/features.jsonl'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    (fit if args.phase == 'train' else score)(args.manifest, args.features)


if __name__ == '__main__':
    main()
