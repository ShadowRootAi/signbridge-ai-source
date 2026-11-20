"""Evaluate the frozen appointment translator on full reconstructed dialogue streams."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import torch
from torch.nn import functional as F

from preprocess import FEATURE_COUNT, FEATURE_SCHEMA
from train_appointment_dialogues import FREEZE, load_frozen_dialogue_decoder
from train_continuous import edit_counts
from train_greek_translation_pilot import greek


DATA = Path('data/appointment-dialogues')
VIDEO = Path('.runtime-tmp/appointment-video')
REPORTS = Path('reports/appointment-dialogues')
FPS = 15
WEIGHTS = (0., .75, 1.5, 3.)
TOLERANCE = .5 * FPS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix('.pending.json')
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staging.replace(path)


def motion(frames):
    delta = (frames[1:, :168] - frames[:-1, :168]).reshape(-1, 2, 84)
    visible = (frames[1:, 220:222] == 1) & (frames[:-1, 220:222] == 1)
    magnitude = ((delta.square().mean(-1) * visible).sum(-1) / visible.sum(-1).clamp_min(1)).sqrt()
    smooth = F.avg_pool1d(magnitude[None, None], kernel_size=5, stride=1, padding=2).flatten()
    return (smooth / smooth.quantile(.75).clamp_min(1e-4)).clamp(max=4).tolist()


def automatic_boundaries(frames, typical, weight):
    count = len(frames)
    if count < 2:
        raise ValueError('A dialogue stream needs at least two samples')
    minimum = max(FPS, round(.4 * typical))
    maximum = min(12 * FPS, max(round(2.3 * typical), minimum * 2))
    choices = [*range(0, count, 3), count]
    movement = motion(frames)
    costs, previous = [0.] + [float('inf')] * (len(choices) - 1), [-1] * len(choices)
    for j, end in enumerate(choices[1:], 1):
        for i in range(j - 1, -1, -1):
            length = end - choices[i]
            if length > maximum:
                break
            if length < minimum and (i or j != len(choices) - 1):
                continue
            cut = 0. if end == count else .15 + weight * movement[end - 1]
            cost = costs[i] + ((length - typical) / typical) ** 2 + cut
            if cost < costs[j]:
                costs[j], previous[j] = cost, i
    if previous[-1] < 0:
        raise ValueError('No feasible dialogue segmentation at the required sample rate')
    order, index = [count], len(choices) - 1
    while index:
        index = previous[index]
        order.append(choices[index])
    return list(reversed(order))


def boundary_counts(reference, predicted):
    reference, predicted = reference[1:-1], predicted[1:-1]
    a = b = matched = 0
    while a < len(reference) and b < len(predicted):
        delta = predicted[b] - reference[a]
        if abs(delta) <= TOLERANCE:
            matched += 1
            a += 1
            b += 1
        elif delta < 0:
            b += 1
        else:
            a += 1
    return {'reference': len(reference), 'predicted': len(predicted), 'within_half_second': matched,
            'f1': 2 * matched / (len(reference) + len(predicted)) if reference or predicted else 1.}


def stream_rows(manifest, video_manifest, features, source_hash, roles=('val', 'test')):
    if video_manifest['source_manifest_sha256'] != source_hash:
        raise ValueError('Whole-video reconstruction refers to a different appointment split')
    expected = {row['id'] for row in manifest['dialogues'] if row['split'] in roles}
    videos = {row['id']: row for row in video_manifest['samples'] if row['id'] in expected}
    by_id = {row['id']: row for row in features}
    if set(by_id) != expected or set(videos) != expected or len(features) != len(expected):
        raise ValueError('Missing or duplicate whole-video landmark rows')
    rows = {}
    for identity, row in by_id.items():
        source = videos[identity]
        key = json.dumps([source['fps'], source['frame_paths']], ensure_ascii=False, separators=(',', ':'))
        frames = torch.tensor(row['features'], dtype=torch.float32)
        offsets = [(value + 1) // 2 for value in source['reference_offsets_30fps']]
        if (row['fps'] != FPS or row['stride'] != 2 or row['feature_schema'] != FEATURE_SCHEMA
                or row['source_frames'] != source['frame_count'] or row['source_key'] != sha256(key.encode()).hexdigest()
                or frames.shape != (offsets[-1], FEATURE_COUNT) or not torch.isfinite(frames).all()
                or offsets != sorted(set(offsets)) or offsets[0] != 0 or not 0 <= row['quality'] <= 1):
            raise ValueError(f'Invalid continuous frame provenance for {identity}')
        rows[identity] = (frames, offsets, row['quality'])
    return rows


def report_dialogue(decoder, frames, edges, reference, expected):
    segments = [frames[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]
    outcomes = decoder.predict_dialogue(segments)
    texts = [item['text'] for item in outcomes]
    tokens = greek(' '.join(texts)).split()
    target = greek(' '.join(expected)).split()
    edits = sum(edit_counts(target, tokens))
    return {'turns': [{**item, 'start_s': round(edges[i] / FPS, 3), 'end_s': round(edges[i + 1] / FPS, 3)}
                      for i, item in enumerate(outcomes)], 'joined_text': ' '.join(texts),
            'boundaries': boundary_counts(reference, edges), 'word_edits': edits, 'reference_words': len(target),
            'word_error_rate': edits / len(target), 'exact_full_text': tokens == target,
            'exact_turns_if_same_count': (sum(greek(a) == greek(b) for a, b in zip(expected, texts, strict=True))
                                          if len(expected) == len(texts) else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DATA / 'manifest.json')
    parser.add_argument('--video-manifest', type=Path, default=VIDEO / 'manifest.json')
    parser.add_argument('--video-features', type=Path, default=VIDEO / 'features.jsonl')
    parser.add_argument('--decoder-freeze', type=Path, default=FREEZE)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = args.manifest.read_bytes()
    manifest = json.loads(source.decode('utf-8-sig'))
    source_hash = sha256(source).hexdigest()
    model_manifest = json.loads(args.decoder_freeze.read_text(encoding='utf-8'))
    if model_manifest['manifest_sha256'] != source_hash:
        raise ValueError('Decoder was not frozen for this dialogue split')
    videos = json.loads(args.video_manifest.read_text(encoding='utf-8'))
    features = stream_rows(manifest, videos, read_jsonl(args.video_features), source_hash)
    by_id = {row['id']: row for row in manifest['samples']}
    typical = median(row['frame_count'] / 2 for row in by_id.values() if row['split'] == 'train')
    dev = [features[row['id']] for row in manifest['dialogues'] if row['split'] == 'val']
    scores = {weight: [boundary_counts(ref, automatic_boundaries(frames, typical, weight))
                       for frames, ref, _ in dev] for weight in WEIGHTS}
    weight = max(WEIGHTS, key=lambda value: (sum(row['f1'] for row in scores[value]),
                                             -sum(abs(row['reference'] - row['predicted']) for row in scores[value]), -value))
    freeze = {'train_typical_turn_samples': typical, 'motion_weight': weight, 'development_results': scores,
              'tolerance_s': TOLERANCE / FPS, 'decoder_freeze_sha256': sha256(args.decoder_freeze.read_bytes()).hexdigest(),
              'source_manifest_sha256': source_hash,
              'video_feature_sha256': sha256(args.video_features.read_bytes()).hexdigest()}
    write(args.report_dir / 'stream-freeze.json', freeze)
    decoder = load_frozen_dialogue_decoder(args.decoder_freeze, arm='contextual')
    outcomes = []
    for row in manifest['dialogues']:
        if row['split'] != 'test':
            continue
        frames, reference, quality = features[row['id']]
        expected = [by_id[key]['text'] for key in row['turn_ids']]
        with torch.inference_mode():
            automatic = report_dialogue(decoder, frames, automatic_boundaries(frames, typical, weight), reference, expected)
            given = report_dialogue(decoder, frames, reference, reference, expected)
        outcomes.append({'id': row['id'], 'scenario': row['scenario'], 'signer': row['signer'], 'tracking_quality': quality,
                         'reference': expected, 'automatic': automatic, 'published_cuts': given})
        print(json.dumps({'id': row['id'], 'auto_segments': len(automatic['turns']), 'reference_turns': len(expected),
                          'matched_boundaries': automatic['boundaries']['within_half_second'], 'auto_text': automatic['joined_text'],
                          'word_edits': automatic['word_edits'], 'reference_words': automatic['reference_words']}, ensure_ascii=False), flush=True)
    write(args.report_dir / 'whole-stream-benchmark.json', {'scope': videos['scope'], 'freeze': freeze, 'dialogues': outcomes})


if __name__ == '__main__':
    main()
