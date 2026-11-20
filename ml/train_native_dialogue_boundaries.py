"""Learn causal signing activity and interval boundaries on native timed dialogues."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from preprocess import FEATURE_COUNT, FEATURE_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('data/sartinas-dialogue-fit')
REPORT = Path('reports/native-dialogue-boundaries')
ARTIFACT = Path('artifacts/native-dialogue-boundaries/causal.pt')
FREEZE = REPORT / 'freeze.json'
RATE = 10
SEED, WIDTH = 715, 48
DILATIONS = (1, 2, 4, 8)
KERNEL = 5
TRAILING = (3, 9)
HISTORY = (KERNEL - 1) * sum(DILATIONS) + max(TRAILING)
MOTION_FLOOR, STILL_TIMEOUT = .002, 15
RESERVED = re.compile(r'(?:^|[^a-z0-9])(?:s6|s10|n1|n2|n6|n7)(?:[^a-z0-9]|$)', re.IGNORECASE)
JSON_ID = re.compile(r'^\s*\{"id":\s*("(?:[^"\\]|\\.)*")\s*,')
MOTION_BLOCKS = ((0, 42), (42, 84), (84, 126), (126, 168), (168, 186), (186, 220))
MODELS = ('native_causal', 'previous_detector')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix('.pending.json')
    stage.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    stage.replace(path)


@dataclass
class NativeStream:
    id: str
    scene: str
    signer: str
    split: str
    frames: torch.Tensor
    fps: int
    intervals_ms: list[tuple[int, int]]


def read_streams(manifest, features, roles, *, fitting=False):
    metadata = json.loads(manifest.read_text(encoding='utf-8-sig'))
    manifest_hash = digest(manifest)
    video_path = manifest.with_name('video-manifest.json')
    videos = {}
    if video_path.exists():
        video_manifest = json.loads(video_path.read_text(encoding='utf-8-sig'))
        if video_manifest['source_manifest_sha256'] != manifest_hash:
            raise ValueError('Native video manifest refers to a different timing annotation')
        videos = {row['id']: row for row in video_manifest['samples']}
        if len(videos) != len(video_manifest['samples']):
            raise ValueError('Duplicate native video identities')
    dialogues = [row for row in metadata['dialogues'] if row['split'] in roles]
    by_id = {row['id']: row for row in dialogues}
    samples = {row['id']: row for row in metadata['samples']}
    if len(by_id) != len(dialogues) or len(samples) != len(metadata['samples']) or not dialogues:
        raise ValueError('Duplicate or absent native timed dialogue IDs')
    selected = {}
    with features.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            prefix = JSON_ID.match(line)
            if not prefix:
                raise ValueError('Whole-stream feature rows must put the ID first')
            identity = json.loads(prefix[1])
            if identity in by_id:
                if identity in selected:
                    raise ValueError('Duplicate whole-stream feature row')
                selected[identity] = line
    if set(by_id) != set(selected):
        raise ValueError('Missing whole native dialogue landmarks')
    streams = []
    signers, used_turns, used_videos = {}, set(), set()
    for row in dialogues:
        identity, scene, signer, split = (str(row[key]) for key in ('id', 'scenario', 'signer', 'split'))
        if fitting and RESERVED.search(' '.join((identity, scene, signer))):
            raise ValueError('Reserved S6/S10 or a held signer entered native boundary fitting')
        if fitting and signers.setdefault(signer, split) != split:
            raise ValueError('A native signer crosses the train/development split')
        record = json.loads(selected[identity])
        fps = record['fps']
        if videos:
            video = videos[identity]
            stride = record['stride']
            key = json.dumps([video['fps'], video['frame_paths']], ensure_ascii=False, separators=(',', ':'))
            if (type(stride) is not int or stride < 1 or video['fps'] != fps * stride
                    or len(video['frame_paths']) != video['frame_count'] or record['source_frames'] != video['frame_count']
                    or record['source_key'] != sha256(key.encode()).hexdigest()
                    or len(record['features']) != (video['frame_count'] + stride - 1) // stride):
                raise ValueError(f'Native landmark provenance does not match the complete video: {identity}')
            video_hash = video.get('source_video_sha256')
            if video_hash is not None:
                if not re.fullmatch(r'[a-f0-9]{64}', video_hash) or video_hash in used_videos:
                    raise ValueError('Invalid or duplicate source recording in native boundary data')
                used_videos.add(video_hash)
        frames = torch.tensor(record['features'], dtype=torch.float32)
        if (fps not in (10, 15) or record['feature_schema'] != FEATURE_SCHEMA or frames.ndim != 2
                or frames.shape[1] != FEATURE_COUNT or len(frames) < fps or not torch.isfinite(frames).all()):
            raise ValueError(f'Invalid full native landmark sequence: {identity}')
        duration_ms = 1000 * len(frames) / fps
        intervals = []
        for key in row['turn_ids']:
            sample = samples[key]
            start, end = sample['start_ms'], sample['end_ms']
            if (key in used_turns or str(sample['signer']) != signer or sample['split'] != split
                    or type(start) is not int or type(end) is not int or not 0 <= start < end <= duration_ms + 1e-6
                    or (intervals and start < intervals[-1][1])):
                raise ValueError('Invalid or nonchronological native timing annotations')
            used_turns.add(key)
            intervals.append((start, end))
        if len(intervals) < 2:
            raise ValueError('A native boundary training or evaluation sequence needs multiple timed turns')
        streams.append(NativeStream(identity, scene, signer, split, frames, int(fps), intervals))
    if used_turns != {row['id'] for row in metadata['samples'] if row['split'] in roles}:
        raise ValueError('A native timing annotation was omitted from its complete recording')
    proof = {'manifest_sha256': manifest_hash, 'video_manifest_sha256': digest(video_path) if videos else None,
             'selected_feature_rows_sha256': sha256(''.join(selected[key] for key in sorted(selected)).encode()).hexdigest(),
             'streams': [{'id': row.id, 'scenario': row.scene, 'signer': row.signer, 'split': row.split,
                          'frames': len(row.frames), 'fps': row.fps, 'turns': len(row.intervals_ms)} for row in streams]}
    return streams, proof


def to_rate(frames, source_fps, target_fps):
    count = (len(frames) * target_fps + source_fps - 1) // source_fps
    return frames[torch.arange(count) * source_fps // target_fps]


def causal_signals(frames):
    flags = (frames[:, 220:224] >= .75).float()
    left, right, pose, face = flags.T
    visibility = torch.stack((left * pose, left, right * pose, right, pose, face), dim=1)
    prior_visibility = torch.cat((visibility[:1], visibility[:-1]))
    delta = torch.diff(frames, dim=0, prepend=frames[:1])
    movement = torch.stack([delta[:, start:end].square().mean(1).sqrt() for start, end in MOTION_BLOCKS], dim=1)
    movement = movement * visibility * prior_visibility
    smoothed = [F.avg_pool1d(F.pad(movement.T[None], (count - 1, 0)), count, stride=1)[0].T for count in TRAILING]
    wrists = torch.cat((frames[:, :2] * (left * pose)[:, None], frames[:, 84:86] * (right * pose)[:, None]), dim=1)
    torso = frames[:, 168:186] * pose[:, None]
    flag_changes = torch.diff(flags, dim=0, prepend=flags[:1]).abs()
    return torch.cat((movement, *smoothed, flags, flag_changes, wrists, torso), dim=1)


class CausalBoundaryHead(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)
        self.project = nn.Linear(len(center), WIDTH)
        self.convolutions = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, KERNEL, dilation=d) for d in DILATIONS)
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in DILATIONS)
        self.dropout = nn.Dropout(.12)
        self.output = nn.Linear(WIDTH, 3)

    def forward(self, features):
        signal = F.gelu(self.project(((features - self.center) / self.scale).clamp(-8, 8)))
        for dilation, convolution, norm in zip(DILATIONS, self.convolutions, self.norms, strict=True):
            past = F.pad(signal.T[None], ((KERNEL - 1) * dilation, 0))
            signal = norm(signal + self.dropout(F.gelu(convolution(past)[0].T)))
        return self.output(signal)


def boundary_probabilities(model, frames):
    signals = causal_signals(frames)
    probabilities = model(signals).sigmoid()
    visible = (frames[:, 222] >= .75) & (frames[:, 220:222] >= .75).any(1)
    moving = signals[:, len(MOTION_BLOCKS):len(MOTION_BLOCKS) + 4].amax(1) >= MOTION_FLOOR
    # Recent tracked hand motion must support a new interval.
    return torch.cat((probabilities[:, :2] * visible[:, None], probabilities[:, 2:], (moving & visible)[:, None]), dim=1)


def targets(count, intervals_ms):
    result = torch.zeros((count, 3))
    for start_ms, end_ms in intervals_ms:
        start = (start_ms * RATE + 999) // 1000
        end = (end_ms * RATE + 999) // 1000
        result[start:end, 0] = 1.
        result[start:start + 3, 1] = 1.
        result[end:end + 3, 2] = 1.
    return result


@dataclass(frozen=True)
class Policy:
    onset: float
    offset: float
    event: float | None
    off_patience: int
    minimum: int


class IntervalState:
    def __init__(self, policy):
        self.policy = policy
        self.index = 0
        self.start = None
        self.high = self.low = self.still = 0

    def consume(self, probabilities):
        spans = []
        settings = self.policy
        for activity, onset, offset, moving in probabilities.tolist():
            index = self.index
            self.index += 1
            starts = settings.event is not None and onset >= settings.event
            ends = settings.event is not None and offset >= settings.event
            if self.start is None:
                self.high = self.high + 1 if moving and activity >= settings.onset else 0
                if self.high >= 2 or (moving and starts and activity >= settings.offset):
                    self.start = index - max(1, self.high) + 1
                    self.high = 0
                continue
            self.low = self.low + 1 if activity < settings.offset else 0
            self.still = 0 if moving else self.still + 1
            if self.low >= settings.off_patience or self.still >= STILL_TIMEOUT:
                end = index - max(self.low, self.still if self.still >= STILL_TIMEOUT else 0) + 1
                if end - self.start >= settings.minimum:
                    spans.append((self.start, end))
                self.start, self.low, self.still = None, 0, 0
            elif ends and index - self.start >= settings.minimum:
                if starts and moving and activity >= settings.onset:
                    spans.append((self.start, index))
                    self.start, self.low, self.still = index, 0, 0
                elif activity < settings.onset:
                    spans.append((self.start, index))
                    self.start, self.low, self.still = None, 0, 0
        return spans

    def finish(self):
        if self.start is None:
            return []
        start, self.start = self.start, None
        end = self.index - self.low
        return [(start, end)] if end - start >= self.policy.minimum else []


def source_intervals(spans, count, fps):
    def convert(index):
        return min(count, (index * fps + RATE // 2) // RATE)
    return [(convert(start), convert(end)) for start, end in spans]


class NativeBoundarySession:
    """Feed contiguous source chunks. Returns completed source-frame intervals."""

    def __init__(self, model, policy, fps):
        if fps not in (10, 15):
            raise ValueError('Native boundary input must be 10 or 15 Hz')
        self.model, self.fps = model, int(fps)
        self.decoder = IntervalState(policy)
        self.source_count = self.grid_count = 0
        self.history = torch.empty((0, FEATURE_COUNT))
        self.finished = False

    @torch.inference_mode()
    def push(self, frames):
        if self.finished:
            raise ValueError('Cannot append landmarks after finalizing a boundary session')
        values = torch.as_tensor(frames, dtype=torch.float32, device='cpu')
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT or not torch.isfinite(values).all():
            raise ValueError('Native boundary chunks require finite [frames,224] landmarks')
        start = self.source_count
        self.source_count += len(values)
        count = (self.source_count * RATE + self.fps - 1) // self.fps
        if count == self.grid_count:
            return []
        indexes = torch.arange(self.grid_count, count) * self.fps // RATE - start
        chunk = values[indexes]
        self.grid_count = count
        combined = torch.cat((self.history, chunk))
        probabilities = boundary_probabilities(self.model, combined)[-len(chunk):]
        self.history = combined[-HISTORY:].clone()
        return source_intervals(self.decoder.consume(probabilities), self.source_count, self.fps)

    def finish(self):
        self.finished = True
        return source_intervals(self.decoder.finish(), self.source_count, self.fps)


@dataclass
class NativeBoundaryDetector:
    model: CausalBoundaryHead
    policy: Policy

    def start_stream(self, fps=15):
        return NativeBoundarySession(self.model, self.policy, fps)

    def predict(self, frames, fps=15):
        session = self.start_stream(fps)
        return session.push(frames) + session.finish()


def matching(reference, predicted, quality):
    """Maximize ordered one-to-one matches, then their overlap."""
    row = [(0, 0.)] * (len(predicted) + 1)
    for gold in reference:
        current = [(0, 0.)]
        for index, guess in enumerate(predicted, 1):
            value = quality(gold, guess)
            options = [row[index], current[-1]]
            if value is not None:
                count, score = row[index - 1]
                options.append((count + 1, score + value))
            current.append(max(options))
        row = current
    return row[-1]


def overlap(left, right):
    return max(0., min(left[1], right[1]) - max(left[0], right[0]))


def detection_metrics(reference, predicted, duration):
    def iou(left, right):
        intersection = overlap(left, right)
        value = intersection / (max(left[1], right[1]) - min(left[0], right[0]))
        return value if value >= .5 else None

    matches, iou_total = matching(reference, predicted, iou)
    onset, _ = matching([row[0] for row in reference], [row[0] for row in predicted],
                        lambda a, b: .5 - abs(a - b) if abs(a - b) <= .5 else None)
    offset, _ = matching([row[1] for row in reference], [row[1] for row in predicted],
                         lambda a, b: .5 - abs(a - b) if abs(a - b) <= .5 else None)
    total = len(reference) + len(predicted)
    wanted = sum(end - start for start, end in reference)
    found = sum(end - start for start, end in predicted)
    covered = sum(overlap(left, right) for left in reference for right in predicted)
    outside = max(0., found - covered)
    return {'reference_intervals': len(reference), 'predicted_intervals': len(predicted), 'matches_iou_at_least_half': matches,
            'interval_f1': 2 * matches / total, 'mean_matched_iou': iou_total / matches if matches else None,
            'onsets_within_half_second': onset, 'offsets_within_half_second': offset,
            'reference_active_seconds': wanted, 'predicted_active_seconds': found, 'covered_reference_seconds': covered,
            'missed_reference_seconds': max(0., wanted - covered), 'activity_outside_reference_seconds': outside,
            'outside_reference_fraction': outside / max(1e-8, duration - wanted), 'activity_f1': 2 * covered / (wanted + found)}


def reference_seconds(stream):
    return [(start / 1000, end / 1000) for start, end in stream.intervals_ms]


def policy_grid():
    for onset, offset, event, off_patience, minimum in product((.45, .55, .65, .75), (.25, .35, .45),
                                                                (None, .65, .8), (2, 4), (3, 5)):
        if offset < onset:
            yield Policy(onset, offset, event, off_patience, minimum)


def choose_policy(model, development):
    with torch.inference_mode():
        examples = [(boundary_probabilities(model, to_rate(row.frames, row.fps, RATE)), reference_seconds(row),
                     len(row.frames), row.fps) for row in development]
    best = None
    for policy in policy_grid():
        results = []
        for probabilities, reference, count, fps in examples:
            decoder = IntervalState(policy)
            spans = decoder.consume(probabilities) + decoder.finish()
            predicted = [(start / fps, end / fps) for start, end in source_intervals(spans, count, fps)]
            results.append(detection_metrics(reference, predicted, count / fps))
        key = (sum(row['interval_f1'] for row in results), sum(row['activity_f1'] for row in results),
               -sum(row['activity_outside_reference_seconds'] for row in results))
        if best is None or key > best[0]:
            best = key, policy, results
    return best[1:]


def train(manifest, features, epochs):
    if FREEZE.exists():
        raise ValueError('The native boundary experiment is already frozen')
    streams, proof = read_streams(manifest, features, ('train', 'val'), fitting=True)
    training = [row for row in streams if row.split == 'train']
    development = [row for row in streams if row.split == 'val']
    if len(training) < 2 or not development:
        raise ValueError('Need at least two complete training recordings and one distinct development recording')
    def examples(rows):
        clips = [causal_signals(to_rate(row.frames, row.fps, RATE)) for row in rows]
        return [(clip, targets(len(clip), row.intervals_ms)) for clip, row in zip(clips, rows, strict=True)]

    fit_rows, dev_rows = examples(training), examples(development)
    joined = torch.cat([clip for clip, _ in fit_rows])
    labels = torch.cat([target for _, target in fit_rows])
    counts = labels.sum(0).clamp_min(1)
    weights = ((len(labels) - counts) / counts).clamp(1., 18.)
    torch.manual_seed(SEED)
    model = CausalBoundaryHead(joined.mean(0), joined.std(0, unbiased=False).clamp_min(.025))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        for index in torch.randperm(len(fit_rows)).tolist():
            clip, target = fit_rows[index]
            loss = F.binary_cross_entropy_with_logits(model(clip), target, pos_weight=weights)
            if not bool(loss.isfinite()):
                raise ValueError('Native boundary training produced a nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
        if epoch % 5 and epoch not in (1, epochs):
            continue
        model.eval()
        with torch.no_grad():
            criterion = sum(float(F.binary_cross_entropy_with_logits(model(clip), target, pos_weight=weights))
                            for clip, target in dev_rows) / len(dev_rows)
        history.append({'epoch': epoch, 'development_weighted_bce': criterion})
        if best is None or criterion < best[0]:
            best = criterion, epoch, {key: value.detach().clone() for key, value in model.state_dict().items()}
            print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best[2])
    model.eval()
    policy, dev_score = choose_policy(model, development)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    stage = ARTIFACT.with_suffix('.pending.pt')
    torch.save({'state_dict': model.state_dict(), 'policy': vars(policy)}, stage)
    stage.replace(ARTIFACT)
    write(FREEZE, {'scope': 'Whole native TRAIN and development streams. Causal temporal encoder and event policy. '
                            'Train and development signers are distinct. Scripts S6/S10 and held signer IDs are excluded.',
                   'model_path': ARTIFACT.as_posix(), 'model_sha256': digest(ARTIFACT), 'data': proof,
                   'model_hz': RATE, 'seed': SEED, 'selected_epoch': best[1], 'history': history,
                   'tracking_policy': {'motion_floor': MOTION_FLOOR, 'maximum_still_samples': STILL_TIMEOUT},
                   'selection': 'Minimum development weighted BCE, then dev interval F1 and activity F1 for the fixed policy grid.',
                   'shared_train_development_scenes': sorted({row.scene for row in training} & {row.scene for row in development}),
                   'development_policy_fit': dev_score, 'policy': vars(policy)})
    print('Frozen native boundary model', digest(FREEZE), flush=True)


def load_native_boundary_detector(freeze_path=FREEZE):
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    path = ROOT / freeze['model_path']
    if (digest(path) != freeze['model_sha256'] or freeze['model_hz'] != RATE
            or freeze['tracking_policy'] != {'motion_floor': MOTION_FLOOR, 'maximum_still_samples': STILL_TIMEOUT}):
        raise ValueError('Frozen native boundary checkpoint or sampling rate changed')
    artifact = torch.load(path, weights_only=True, map_location='cpu')
    if artifact['policy'] != freeze['policy']:
        raise ValueError('Frozen boundary policy differs from the checkpoint')
    weights = artifact['state_dict']
    model = CausalBoundaryHead(weights['center'], weights['scale'])
    model.load_state_dict(weights)
    return NativeBoundaryDetector(model.eval(), Policy(**artifact['policy']))


def score(manifest, features, role, output, old_freeze, old_checkpoint):
    from evaluate_targeted_appointment import load_boundaries

    streams, evidence = read_streams(manifest, features, (role,))
    native = load_native_boundary_detector()
    if role == 'val':
        freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
        if digest(manifest) != freeze['data']['manifest_sha256']:
            raise ValueError('Development manifest differs from the one used to select the policy')
        _, fit_evidence = read_streams(manifest, features, ('train', 'val'), fitting=True)
        if fit_evidence != freeze['data']:
            raise ValueError('Native fitting features changed')
    _, previous = load_boundaries(old_freeze, old_checkpoint)
    results = []
    with torch.inference_mode():
        for stream in streams:
            reference = reference_seconds(stream)
            duration = len(stream.frames) / stream.fps
            actual = [(start / stream.fps, end / stream.fps) for start, end in native.predict(stream.frames, stream.fps)]
            old_edges = previous(to_rate(stream.frames, stream.fps, 15))
            baseline = [(start / 15, min(duration, end / 15)) for start, end in zip(old_edges[:-1], old_edges[1:], strict=True)]
            models = {name: {'metrics': detection_metrics(reference, intervals, duration), 'intervals_seconds': intervals}
                      for name, intervals in zip(MODELS, (actual, baseline), strict=True)}
            tracked = torch.where((stream.frames[:, 222] >= .75) & (stream.frames[:, 220:222] >= .75).any(1))[0]
            static_seconds = 40
            control = None
            if len(tracked):
                static = stream.frames[tracked[0]].expand(static_seconds * stream.fps, -1)
                control = {'seconds': static_seconds, 'intervals': native.predict(static, stream.fps),
                           'scope': 'One tracked source frame repeated as a synthetic motion control.'}
            results.append({'id': stream.id, 'signer': stream.signer, 'scenario': stream.scene, 'duration_seconds': duration,
                            'reference_intervals_seconds': reference, 'static_motion_control': control, 'models': models})
            print(stream.id, {name: item['metrics'] for name, item in models.items()}, flush=True)
    write(output, {'scope': 'Native timed full-stream evaluation. Development is reused for model and policy selection. '
                            'Any earlier observed test recordings are retrospective. Outside-reference time can include unlabeled movements.',
                   'native_freeze_sha256': digest(FREEZE), 'previous_freeze_sha256': digest(old_freeze),
                   'role': role, 'data': evidence, 'streams': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'score'))
    parser.add_argument('--manifest', type=Path, default=DATA / 'manifest.json')
    parser.add_argument('--features', type=Path, default=DATA / 'features.jsonl')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--role', choices=('val', 'test'), default='val')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--old-freeze', type=Path, default=Path('reports/appointment-unseen/turn-freeze.json'))
    parser.add_argument('--old-checkpoint', type=Path, default=Path('artifacts/appointment-unseen/turn-boundaries.pt'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Training epochs must be positive')
    torch.set_num_threads(2)
    if args.phase == 'train':
        train(args.manifest, args.features, args.epochs)
    else:
        score(args.manifest, args.features, args.role, args.output or REPORT / f'{args.role}-benchmark.json',
              args.old_freeze, args.old_checkpoint)


if __name__ == '__main__':
    main()
