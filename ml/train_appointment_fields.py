"""Detect supported appointment fields in a chronological stream of verified GSL footage."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from continuous_selective import tracked_hand_motion
from preprocess import FEATURE_COUNT, FEATURE_SCHEMA
from train_continuous_library import datasets
from train_greek_translation_pilot import greek, sha
from train_grounded_greek import encode, load_visual, write
from train_native_greek import save_checkpoint, visual_checkpoint
from train_targeted_greek import compatible_training, split_rows, verify_manifest


REPORT = Path('reports/appointment-fields')
ARTIFACTS = Path('artifacts/appointment-fields')
OLD = Path('data/appointment-targeted')
BASE = Path('data/appointment-unseen')
DEVELOPMENT = Path('.runtime-tmp/appointment-targeted-video')
BACKGROUND_DEVELOPMENT = Path('.runtime-tmp/appointment-unseen-video')
NEW = Path('data/appointment-daytime')
FPS = 15
SEED = 534
FIELDS = {
    'intent': ('background', 'book_request', 'book_confirmation'),
    'day': ('background', 'monday', 'day_after_tomorrow'),
    'time_period': ('background', 'morning', 'noon'),
}
LABELS = {
    'θέλω ραντεβού για υπέρηχο στη κοιλιά': {'intent': 'book_request'},
    'θέλω ραντεβού για υπέρηχο στην κοιλιά': {'intent': 'book_request'},
    'πρώτα σας κλείνω ραντεβού με παθολόγο': {'intent': 'book_confirmation'},
    'υπάρχει γιατρός μεθαύριο': {'day': 'day_after_tomorrow'},
    'σας βολεύει τη δευτέρα το πρωί': {'day': 'monday', 'time_period': 'morning'},
    'προτιμώ το μεσημέρι': {'time_period': 'noon'},
}
WINDOW_SECONDS = (1.2, 1.7, 2.3, 3.1, 4.1, 5.2)
FLOORS = (.45, .55, .65, .75, .82, .88, .92, .95, .97, .985, .995, 1.01)


class FieldHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 96), nn.GELU(), nn.Dropout(.20))
        self.fields = nn.ModuleDict({name: nn.Linear(96, len(values)) for name, values in FIELDS.items()})

    def forward(self, memory):
        pooled = self.shared(memory)
        return {name: layer(pooled) for name, layer in self.fields.items()}


def pool_memories(encoder, clips):
    windows = encode(encoder, clips)
    pooled = []
    for row in windows:
        quarters = F.adaptive_avg_pool1d(row.T[None], 4)[0].T.flatten()
        pooled.append(torch.cat((quarters, row.amax(0))))
    return torch.stack(pooled)


def targets(rows):
    entries = [LABELS.get(greek(row.text), {}) for row in rows]
    return {field: torch.tensor([values.index(row.get(field, 'background')) for row in entries])
            for field, values in FIELDS.items()}


def fit_head(encoder, rows, updates):
    expected = targets(rows)
    for name, values in FIELDS.items():
        present = set(expected[name].tolist())
        if present != set(range(len(values))):
            raise ValueError(f'Missing authentic examples of supported {name} values')
    original = [row.frames for row in rows]
    cropped = [clip[round(.08 * len(clip)):max(round(.08 * len(clip)) + 2, round(.92 * len(clip)))]
               for clip in original]
    features = pool_memories(encoder, [*original, *cropped]).clone()
    actual = {name: torch.cat((value, value)) for name, value in expected.items()}
    positive = torch.stack([value > 0 for value in actual.values()]).any(0)
    sampling = torch.where(positive, 3., 1.)
    weights = {
        name: (4 / torch.bincount(value, minlength=len(FIELDS[name])).float().clamp_min(1)).sqrt().clamp(.32, 1.8)
        for name, value in actual.items()
    }
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = FieldHead(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    for _ in range(updates):
        ids = torch.multinomial(sampling, 32, replacement=True, generator=generator)
        outputs = model(features[ids])
        loss = sum(F.cross_entropy(output, actual[name][ids], weight=weights[name])
                   for name, output in outputs.items()) / len(outputs)
        if not loss.isfinite():
            raise ValueError('Non-finite field loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), features.shape[1], {name: dict(zip(FIELDS[name], torch.bincount(value).tolist(), strict=True))
                                             for name, value in expected.items()}


def reference_stream(data_manifest, stream_manifest, features, role):
    data = json.loads(data_manifest.read_bytes())
    source = json.loads(stream_manifest.read_bytes())
    if source.get('source_manifest_sha256', sha(data_manifest)) != sha(data_manifest):
        raise ValueError('Chronological frames refer to a different annotation manifest')
    by_id = {row['id']: row for row in data['samples']}
    feature_rows = [json.loads(line) for line in features.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({row['id'] for row in feature_rows}) != len(feature_rows):
        raise ValueError('Duplicate landmark streams')
    landmark_rows = {row['id']: row for row in feature_rows}
    streams = []
    for scene in data['dialogues']:
        if scene['split'] != role:
            continue
        video = next(row for row in source['samples'] if row['id'] == scene['id'])
        observed = landmark_rows[video['id']]
        source_key = sha256(json.dumps([video['fps'], video['frame_paths']], ensure_ascii=False,
                                       separators=(',', ':')).encode()).hexdigest()
        turns = [by_id[identity] for identity in scene['turn_ids']]
        original_paths = [path for turn in turns for path in turn['frame_paths']]
        values = torch.tensor(observed['features'], dtype=torch.float32)
        if (source_key != observed['source_key'] or observed['feature_schema'] != FEATURE_SCHEMA
                or observed['fps'] != FPS or video['fps'] != 2 * FPS or observed['stride'] != 2
                or video['frame_paths'] != original_paths or values.shape != (len(original_paths[::2]), FEATURE_COUNT)
                or not values.isfinite().all()
                or any(str(row['signer']) != str(scene['signer']) or row['split'] != role for row in turns)):
            raise ValueError(f'Invalid chronological feature provenance: {scene["id"]}')
        offset, gold = 0, []
        for turn in turns:
            next_offset = offset + len(turn['frame_paths'])
            for field, value in LABELS.get(greek(turn['text']), {}).items():
                gold.append({'field': field, 'value': value, 'start_s': offset / video['fps'],
                             'end_s': next_offset / video['fps'], 'official_text': turn['text']})
            offset = next_offset
        streams.append((scene, values, gold))
    if not streams:
        raise ValueError(f'No verified {role} stream')
    return streams


def window_inputs(encoder, frames):
    clips, spans = [], []
    for seconds in WINDOW_SECONDS:
        size = min(len(frames), round(seconds * FPS))
        if size < 2:
            continue
        starts = sorted({*range(0, len(frames) - size + 1, 5), len(frames) - size})
        for start in starts:
            clip = frames[start:start + size]
            flags = clip[:, 220:223] >= .75
            tracked = (flags[:, 2] & flags[:, :2].any(1)).float().mean().item()
            if tracked >= .45 and tracked_hand_motion(clip) >= .003:
                clips.append(clip)
                spans.append((start / FPS, (start + size) / FPS))
    return (pool_memories(encoder, clips) if clips else None), spans


def overlap(left, right):
    span = min(left['end_s'], right['end_s']) - max(left['start_s'], right['start_s'])
    if span <= 0:
        return 0.
    return span / (max(left['end_s'], right['end_s']) - min(left['start_s'], right['start_s']))


@torch.inference_mode()
def detect(model, prepared, floors):
    memory, spans = prepared
    if memory is None:
        return []
    outputs = model(memory)
    detected = []
    for field, logits in outputs.items():
        score, ids = logits.softmax(-1).max(-1)
        choices = sorted([
            {'field': field, 'value': FIELDS[field][index], 'score': confidence, 'start_s': start, 'end_s': end}
            for (start, end), index, confidence in zip(spans, ids.tolist(), score.tolist(), strict=True)
            if index and confidence >= floors[field]
        ], key=lambda row: (-row['score'], row['start_s']))
        retained = []
        for event in choices:
            if all(overlap(event, previous) < .15 for previous in retained):
                retained.append(event)
        detected.extend(retained)
    return sorted(detected, key=lambda row: (row['start_s'], row['field']))


def measure(gold, output):
    found = set()
    unmatched = []
    for guess in sorted(output, key=lambda row: -row['score']):
        candidates = [(overlap(guess, event), index) for index, event in enumerate(gold)
                      if index not in found and guess['field'] == event['field'] and guess['value'] == event['value']]
        score, index = max(candidates, default=(0., -1))
        if score >= .3:
            found.add(index)
        else:
            unmatched.append(guess)
    return {'correct': len(found), 'expected': len(gold), 'false': len(unmatched),
            'all_fields_without_extras': bool(gold) and len(found) == len(gold) and not unmatched,
            'fields': {name: {'correct': sum(gold[index]['field'] == name for index in found),
                              'expected': sum(row['field'] == name for row in gold),
                              'false': sum(row['field'] == name for row in unmatched)} for name in FIELDS},
            'reference': [{**row, 'matched': index in found} for index, row in enumerate(gold)],
            'predictions': output, 'unmatched_predictions': unmatched}


def choose_floors(model, prepared):
    floors = {}
    for name in FIELDS:
        choices = []
        for floor in FLOORS:
            policy = dict.fromkeys(FIELDS, 1.01)
            policy[name] = floor
            metrics = [measure(gold, detect(model, windows, policy))['fields'][name] for windows, gold in prepared]
            correct = sum(row['correct'] for row in metrics)
            denominator = sum(row['expected'] + row['correct'] + row['false'] for row in metrics)
            false = sum(row['false'] for row in metrics)
            f1 = 2 * correct / denominator if denominator else 1.
            choices.append((f1, -false, floor))
        floors[name] = max(choices)[2]
    return floors


def train(args):
    if args.freeze.exists():
        raise ValueError(f'The comparison is already frozen: {args.freeze}')
    visual_path, visual_hash = visual_checkpoint()
    encoder, _ = load_visual(visual_path, visual_hash)
    historical, _, prior = datasets(BASE / 'manifest.json', BASE / 'features.jsonl')
    old_manifest, old, _, old_hash = split_rows(OLD / 'manifest.json', OLD / 'features.jsonl')
    extra_manifest, added, _, added_hash = split_rows(args.data / 'manifest.json', args.data / 'features.jsonl')
    held = [row for row in old_manifest['samples'] + extra_manifest['samples'] if row['split'] != 'train']
    baseline = compatible_training(historical, old, held)
    expanded = compatible_training(baseline, added, held)
    dev = reference_stream(OLD / 'manifest.json', DEVELOPMENT / 'manifest.json', DEVELOPMENT / 'features.jsonl', 'val')
    dev += reference_stream(BASE / 'manifest.json', BACKGROUND_DEVELOPMENT / 'manifest.json',
                            BACKGROUND_DEVELOPMENT / 'features.jsonl', 'val')
    prepared = [(window_inputs(encoder, frames), gold) for _, frames, gold in dev]
    arms = {}
    for name, rows in (('existing_data', baseline), ('more_appointment_data', expanded)):
        model, width, counts = fit_head(encoder, rows, args.updates)
        floors = choose_floors(model, prepared)
        metrics = [{'id': scene['id'], **measure(gold, detect(model, windows, floors))}
                   for (scene, _, _), (windows, gold) in zip(dev, prepared, strict=True)]
        checkpoint = ARTIFACTS / f'{name}.pt'
        save_checkpoint(checkpoint, {'width': width, 'weights': model.state_dict()})
        arms[name] = {'checkpoint': checkpoint.as_posix(), 'checkpoint_sha256': sha(checkpoint),
                      'thresholds': floors, 'training_clips': len(rows), 'labels': counts, 'development': metrics}
        print(json.dumps({'arm': name, 'train': len(rows), 'thresholds': floors,
                          'dev': [{key: row[key] for key in ('correct', 'expected', 'false')} for row in metrics]},
                         ensure_ascii=False), flush=True)
    write(args.freeze, {'scope': 'Official CERTH appointment fields. No cancellation, confirmed appointment change or clock time labels.',
                        'annotation_sha256': sha(args.data / 'manifest.json'), 'visual_path': visual_path.as_posix(),
                        'visual_sha256': visual_hash, 'verified_training': {'historical': prior, 'old': old_hash, 'new': added_hash},
                        'training_signers': sorted({row.signer for row in expanded}),
                        'development_signers': sorted({str(scene['signer']) for scene, _, _ in dev}),
                        'development_streams': {str(path): sha(path) for path in (
                            DEVELOPMENT / 'features.jsonl', BACKGROUND_DEVELOPMENT / 'features.jsonl')},
                        'updates_per_arm': args.updates, 'arms': arms})


def evaluation_manifest(fitted_manifest, reserved):
    if reserved is None:
        return fitted_manifest
    original = json.loads(fitted_manifest.read_bytes())
    new_manifest = reserved / 'manifest.json'
    candidate = verify_manifest(new_manifest)
    old_takes = [row for row in original['dialogues'] if row['split'] == 'test']
    if len(old_takes) != 1 or len(candidate['dialogues']) != 1:
        raise ValueError('Expected one original test take and its reserved repeat')
    [old_take], [new_take] = old_takes, candidate['dialogues']
    if (new_take['id'] != original['reserved_future_take'] or new_take['split'] != 'test'
            or new_take['scenario'] != old_take['scenario'] or str(new_take['signer']) != str(old_take['signer'])):
        raise ValueError('The test must use the physical recording reserved in the frozen manifest')
    old_by_id = {row['id']: row for row in original['samples']}
    new_by_id = {row['id']: row for row in candidate['samples']}
    old_sources = {row['source_path'] for row in original['samples']}
    old_digests = {row['sha256'] for row in original['samples']}
    expected_sources = [old_by_id[identity]['source_path'].replace(old_take['id'], new_take['id'], 1)
                        for identity in old_take['turn_ids']]
    turns = [new_by_id[identity] for identity in new_take['turn_ids']]
    if (len(turns) != len(candidate['samples']) or len(turns) != len(new_by_id)
            or [row['source_path'] for row in turns] != expected_sources
            or any(row['source_path'] in old_sources or row['sha256'] in old_digests for row in turns)):
        raise ValueError('The reserved recording is incomplete, out of order, or repeats previously used data')
    return new_manifest


def evaluate(args):
    frozen = json.loads(args.freeze.read_bytes())
    fitted_manifest = args.data / 'manifest.json'
    if sha(fitted_manifest) != frozen['annotation_sha256']:
        raise ValueError('The fitting manifest differs from the frozen protocol')
    manifest = evaluation_manifest(fitted_manifest, args.reserved_evaluation)
    stream_manifest = args.stream_manifest or manifest.parent / 'chronological-manifest.json'
    stream_features = args.stream_features or manifest.parent / 'chronological-features.jsonl'
    held = reference_stream(manifest, stream_manifest, stream_features, 'test')
    if {str(scene['signer']) for scene, _, _ in held} & set(frozen['training_signers'] + frozen['development_signers']):
        raise ValueError('Held signers overlap fitting or calibration')
    encoder, _ = load_visual(Path(frozen['visual_path']), frozen['visual_sha256'])
    prepared = [window_inputs(encoder, frames) for _, frames, _ in held]
    arms = {}
    for name, arm in frozen['arms'].items():
        checkpoint = Path(arm['checkpoint'])
        if sha(checkpoint) != arm['checkpoint_sha256']:
            raise ValueError('Frozen appointment field checkpoint changed')
        weights = torch.load(checkpoint, weights_only=True)
        model = FieldHead(weights['width'])
        model.load_state_dict(weights['weights'])
        model.eval()
        records = []
        for (scene, frames, gold), windows in zip(held, prepared, strict=True):
            predicted = detect(model, windows, arm['thresholds'])
            result = measure(gold, predicted)
            correct, expected, false = (result[key] for key in ('correct', 'expected', 'false'))
            records.append({'id': scene['id'], 'signer': scene['signer'], 'duration_s': len(frames) / FPS,
                            'precision': correct / (correct + false) if correct + false else 0.,
                            'recall': correct / expected if expected else None,
                            'f1': 2 * correct / (expected + correct + false) if expected + correct + false else None,
                            **result})
        arms[name] = records
        print(json.dumps({'arm': name, 'results': [{key: row[key] for key in ('correct', 'expected', 'false', 'fields')}
                                                    for row in records]}, ensure_ascii=False), flush=True)
    output = args.output or REPORT / ('benchmark-reserved.json' if args.reserved_evaluation else 'benchmark.json')
    write(output, {'scope': 'All published consecutive cuts from a new take of the same script, joined without '
                                'reconstructing original gaps. Inference receives one full stream and no reference boundaries.',
                        'freeze_sha256': sha(args.freeze), 'stream_sha256': sha(stream_features),
                        'evaluation_manifest_sha256': sha(manifest),
                        'unsupported': ['reschedule', 'cancel', 'exact_clock_time'], 'arms': arms})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate'))
    parser.add_argument('--data', type=Path, default=NEW)
    parser.add_argument('--reserved-evaluation', type=Path, help='Verified folder for the repeat reserved at fitting time')
    parser.add_argument('--stream-manifest', type=Path)
    parser.add_argument('--stream-features', type=Path)
    parser.add_argument('--freeze', type=Path, default=REPORT / 'freeze.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--updates', type=int, default=350)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('Training updates must be positive')
    if args.phase != 'evaluate' and args.reserved_evaluation:
        parser.error('A reserved recording can only be used for evaluation')
    torch.set_num_threads(3)
    (train if args.phase == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
