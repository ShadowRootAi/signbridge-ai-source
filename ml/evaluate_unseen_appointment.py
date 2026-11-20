"""Fit turn detection on training takes and decode whole held appointment streams."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import torch
from torch import nn
from torch.nn import functional as F

from evaluate_appointment_stream import FPS, automatic_boundaries, boundary_counts, report_dialogue, stream_rows, write
from train_continuous import edit_counts
from train_greek_translation_pilot import greek


DATA = Path('data/appointment-unseen')
VIDEO = Path('.runtime-tmp/appointment-unseen-video')
REPORT = Path('reports/appointment-unseen')
ARTIFACT = Path('artifacts/appointment-unseen/turn-boundaries.pt')
SEED = 593
MOTION_BLOCKS = ((0, 42), (42, 84), (84, 126), (126, 168), (168, 186), (186, 220))
THRESHOLDS = (.25, .35, .45, .55, .65, .75, .85)
MIN_SPACING = (round(.7 * FPS), FPS, round(1.3 * FPS))


def data_rows(manifest, video, features, roles):
    source = manifest.read_bytes()
    digest = sha256(source).hexdigest()
    plan = json.loads(source)
    dialogue_ids = {row['id'] for row in plan['dialogues'] if row['split'] in roles}
    lines = {}
    with features.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if item['id'] in dialogue_ids:
                if item['id'] in lines:
                    raise ValueError(f'Duplicate continuous landmark stream: {item["id"]}')
                lines[item['id']] = line
    ordered = [lines[key] for key in sorted(lines)]
    rows = stream_rows(plan, json.loads(video.read_bytes()), list(map(json.loads, ordered)), digest, roles=roles)
    return plan, rows, digest, sha256(''.join(ordered).encode()).hexdigest()


def signals(frames):
    change = torch.diff(frames, dim=0, prepend=frames[:1])
    moved = torch.stack([change[:, start:stop].square().mean(1).sqrt() for start, stop in MOTION_BLOCKS], dim=1)
    smoothed = F.avg_pool1d(moved.T[None], 5, stride=1, padding=2)[0].T
    flags = frames[:, 220:224]
    height = frames[:, [1, 85]] * flags[:, :2]
    return torch.cat((moved, smoothed, flags, change[:, 220:224].abs(), height), dim=1)


class BoundaryDetector(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)
        self.net = nn.Sequential(nn.Conv1d(len(center), 32, 7, padding=3), nn.GELU(), nn.Dropout(.12),
                                 nn.Conv1d(32, 24, 7, padding=6, dilation=2), nn.GELU(), nn.Conv1d(24, 1, 1))

    def forward(self, sequence):
        value = ((sequence - self.center) / self.scale).T[None]
        return self.net(value)[0, 0]


def nms_boundaries(probabilities, threshold, spacing):
    count = len(probabilities)
    peaks = [i for i in range(spacing, count - spacing)
             if probabilities[i] >= threshold and probabilities[i] >= probabilities[i - 1]
             and probabilities[i] > probabilities[i + 1]]
    chosen = [0, count]
    for index in sorted(peaks, key=lambda i: (-probabilities[i], i)):
        if all(abs(index - previous) >= spacing for previous in chosen):
            chosen.append(index)
    return sorted(chosen)


def aggregate(measures):
    reference = sum(row['reference'] for row in measures)
    predicted = sum(row['predicted'] for row in measures)
    matched = sum(row['within_half_second'] for row in measures)
    return {'reference': reference, 'predicted': predicted, 'within_half_second': matched,
            'f1': 2 * matched / (reference + predicted) if reference or predicted else 1.}


def checkpoint_score(result):
    return (result['f1'], result['within_half_second'], -result['predicted'])


def fit_boundary(args):
    plan, data, manifest_hash, stream_hash = data_rows(args.manifest, args.video_manifest, args.features, ('train', 'val'))
    training = [data[row['id']] for row in plan['dialogues'] if row['split'] == 'train']
    development = [data[row['id']] for row in plan['dialogues'] if row['split'] == 'val']
    if len(training) < 3 or not development:
        raise ValueError('Need at least three chronological training takes and one development take')
    signal_rows = [signals(frames) for frames, _, _ in training]
    joined = torch.cat(signal_rows)
    torch.manual_seed(SEED)
    model = BoundaryDetector(joined.mean(0), joined.std(0, unbiased=False).clamp_min(.02))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.015)
    targets = []
    for frames, boundaries, _ in training:
        target = torch.zeros(len(frames))
        for index in boundaries[1:-1]:
            target[max(0, index - 2):index + 3] = 1.
        targets.append(target)
    positive = sum(float(target.sum()) for target in targets)
    pos_weight = torch.tensor(min(12., (sum(map(len, targets)) - positive) / max(1., positive)))
    dev_inputs = [signals(frames) for frames, _, _ in development]
    best = None
    for epoch in range(1, 81):
        model.train()
        for features, target in zip(signal_rows, targets, strict=True):
            loss = F.binary_cross_entropy_with_logits(model(features), target, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite turn detector loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 10:
            continue
        model.eval()
        with torch.inference_mode():
            probabilities = [model(features).sigmoid().tolist() for features in dev_inputs]
        for threshold in THRESHOLDS:
            for spacing in MIN_SPACING:
                result = aggregate([boundary_counts(boundaries, nms_boundaries(probability, threshold, spacing))
                                    for (_, boundaries, _), probability in zip(development, probabilities, strict=True)])
                key = (*checkpoint_score(result), -epoch, threshold, spacing)
                if best is None or key > best[0]:
                    best = (key, epoch, threshold, spacing, result,
                            {name: value.detach().clone() for name, value in model.state_dict().items()})
    _, epoch, threshold, spacing, learned_score, state = best
    typical = median(row['frame_count'] / 2 for row in plan['samples'] if row['split'] == 'train')
    duration_score = aggregate([boundary_counts(boundaries, automatic_boundaries(frames, typical, 0.))
                                for frames, boundaries, _ in development])
    selection = 'learned' if checkpoint_score(learned_score) >= checkpoint_score(duration_score) else 'duration'
    args.boundary_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pending = args.boundary_checkpoint.with_suffix('.pending.pt')
    torch.save(state, pending)
    pending.replace(args.boundary_checkpoint)
    freeze = {'manifest_sha256': manifest_hash, 'train_dev_stream_sha256': stream_hash,
              'checkpoint_sha256': sha256(args.boundary_checkpoint.read_bytes()).hexdigest(), 'train_takes': len(training),
              'development_takes': len(development), 'epochs_selected': epoch, 'threshold': threshold,
              'spacing_samples': spacing, 'train_typical_turn_samples': typical, 'selected_method': selection,
              'development_learned': learned_score, 'development_duration': duration_score, 'seed': SEED}
    write(args.report_dir / 'turn-freeze.json', freeze)
    print(json.dumps(freeze, ensure_ascii=False), flush=True)


def with_raw_result(decoder, frames, predicted, reference, expected):
    result = report_dialogue(decoder, frames, predicted, reference, expected)
    target = greek(' '.join(expected)).split()
    raw = ' '.join(text for turn in result['turns'] if (text := turn.get('raw_text', turn['text']).strip()))
    raw_tokens = greek(raw).split()
    result.update(joined_text=' '.join(turn['text'].strip() for turn in result['turns'] if turn['text'].strip()),
                  raw_joined_text=raw, raw_word_edits=sum(edit_counts(target, raw_tokens)), raw_exact_full_text=target == raw_tokens,
                  emitted_segments=sum(bool(turn['text'].strip()) for turn in result['turns']))
    return result


def score(args):
    from train_grounded_greek import load_grounded_decoder

    frozen = json.loads((args.report_dir / 'turn-freeze.json').read_bytes())
    _, _, manifest_hash, stream_hash = data_rows(args.manifest, args.video_manifest, args.features, ('train', 'val'))
    if (manifest_hash != frozen['manifest_sha256'] or stream_hash != frozen['train_dev_stream_sha256']
            or sha256(args.boundary_checkpoint.read_bytes()).hexdigest() != frozen['checkpoint_sha256']):
        raise ValueError('Turn detector, annotations or development inputs changed')
    decoder_freeze = json.loads(args.decoder_freeze.read_bytes())
    if decoder_freeze['manifest_sha256'] != manifest_hash:
        raise ValueError('The translator was not frozen for this unseen dialogue split')
    state = torch.load(args.boundary_checkpoint, weights_only=True, map_location='cpu')
    detector = BoundaryDetector(state['center'], state['scale'])
    detector.load_state_dict(state)
    detector.eval()
    plan, held, _, held_hash = data_rows(args.manifest, args.video_manifest, args.features, ('test',))
    by_id = {row['id']: row for row in plan['samples']}
    models = {arm: load_grounded_decoder(args.decoder_freeze, arm=arm)
              for arm in ('continuous_only', 'isolated_transfer')}
    rows = []
    with torch.inference_mode():
        for dialogue in plan['dialogues']:
            if dialogue['split'] != 'test':
                continue
            frames, reference, quality = held[dialogue['id']]
            expected = [by_id[key]['text'] for key in dialogue['turn_ids']]
            methods = {
                'learned': nms_boundaries(detector(signals(frames)).sigmoid().tolist(), frozen['threshold'], frozen['spacing_samples']),
                'duration': automatic_boundaries(frames, frozen['train_typical_turn_samples'], 0.),
            }
            edges = methods[frozen['selected_method']]
            results = {arm: {'automatic': with_raw_result(decoder, frames, edges, reference, expected),
                             'published_cuts': with_raw_result(decoder, frames, reference, reference, expected)}
                       for arm, decoder in models.items()}
            comparisons = {method: boundary_counts(reference, values) for method, values in methods.items()}
            rows.append({'id': dialogue['id'], 'scenario': dialogue['scenario'], 'signer': dialogue['signer'],
                         'duration_seconds': len(frames) / FPS, 'tracking_quality': quality, 'reference': expected,
                         'boundary_diagnostics': comparisons, 'models': results})
            print(json.dumps({'id': dialogue['id'], 'chosen_boundaries': comparisons[frozen['selected_method']],
                              'guarded_text': results['isolated_transfer']['automatic']['joined_text'],
                              'raw_text': results['isolated_transfer']['automatic']['raw_joined_text']}, ensure_ascii=False), flush=True)
    write(args.report_dir / 'whole-stream-benchmark.json', {
        'scope': 'Consecutive official source cuts presented as one stream with no supplied turn count or boundaries. '
                 'The original pauses between published cuts are absent. Each studio signer performs both roles.',
        'turn_freeze_sha256': sha256((args.report_dir / 'turn-freeze.json').read_bytes()).hexdigest(),
        'decoder_freeze_sha256': sha256(args.decoder_freeze.read_bytes()).hexdigest(), 'held_stream_sha256': held_hash,
        'selected_boundary_method': frozen['selected_method'],
        'summary': {arm: {
            'takes': len(rows),
            'raw_exact_dialogues': sum(row['models'][arm]['automatic']['raw_exact_full_text'] for row in rows),
            'guarded_exact_dialogues': sum(row['models'][arm]['automatic']['exact_full_text'] for row in rows),
            'raw_word_edits': sum(row['models'][arm]['automatic']['raw_word_edits'] for row in rows),
            'reference_words': sum(row['models'][arm]['automatic']['reference_words'] for row in rows),
            'emitted_segments': sum(row['models'][arm]['automatic']['emitted_segments'] for row in rows),
        } for arm in models}, 'dialogues': rows,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('--manifest', type=Path, default=DATA / 'manifest.json')
    parser.add_argument('--video-manifest', type=Path, default=VIDEO / 'manifest.json')
    parser.add_argument('--features', type=Path, default=VIDEO / 'features.jsonl')
    parser.add_argument('--report-dir', type=Path, default=REPORT)
    parser.add_argument('--boundary-checkpoint', type=Path, default=ARTIFACT)
    parser.add_argument('--decoder-freeze', type=Path, default=Path('reports/grounded-greek/appointment-unseen-freeze.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    (fit_boundary if args.phase == 'fit' else score)(args)


if __name__ == '__main__':
    main()
