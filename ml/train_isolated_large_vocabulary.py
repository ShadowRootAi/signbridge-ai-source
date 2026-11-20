"""Train isolated GSL gloss recognition on actors 1/2, develop on 4, and hold actor 3."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_synthetic_ctc_transfer import POSE_CACHE, file_hash, frame_feature_array, verify_feature_math
from openhands_gsl_stgcn import METADATA_ARCHIVE_SHA256


CACHE = POSE_CACHE.parent
METADATA = CACHE / 'gsl_metadata.zip'
HELD_POSES = CACHE / 'official-test-poses.npz'
EXPANDED_POSES = CACHE / 'official-train-expanded-poses.npz'
ARTIFACTS = Path('artifacts/isolated-large-vocabulary')
REPORT = Path('reports/isolated-large-vocabulary')
POSE_ID = re.compile(r'[a-z]+\d+_signer(?P<signer>\d+)_rep\d+_glosses/glosses\d+')
SEED, FRAMES, WIDTH, BATCH, EPOCHS = 111, 32, 96, 48, 42


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def resample_features(frames):
    sampled = F.interpolate(frames.T[None], size=FRAMES, mode='linear', align_corners=True)[0].T
    indices = torch.linspace(0, len(frames) - 1, FRAMES).round().long()
    sampled[:, 220:] = frames[indices, 220:]
    return sampled


def prepare(raw, offsets):
    converted = frame_feature_array(raw)
    verify_feature_math(raw, converted)
    return torch.stack([resample_features(torch.from_numpy(converted[start:end]))
                        for start, end in zip(offsets[:-1], offsets[1:], strict=True)])


def load_poses(path, split):
    if file_hash(METADATA) != METADATA_ARCHIVE_SHA256:
        raise ValueError('Official GSL split metadata changed')
    with ZipFile(METADATA) as official:
        folder = 'GSL_split/GSL_isolated/'
        vocabulary = official.read(folder + 'iso_classes.csv').decode('utf-8-sig').splitlines()
        member = folder + ('test' if split == 'test' else 'train') + '_greek_iso.csv'
        references = dict(line.split('|', 1) for line in official.read(member).decode('utf-8-sig').splitlines())
    with np.load(path, allow_pickle=False) as cache:
        raw, offsets, ids, glosses = (cache[k] for k in ('keypoints', 'offsets', 'ids', 'glosses'))
        if cache['split'].item() != split:
            raise ValueError(f'Incorrect cache split: {path}')
    if (len(set(ids)) != len(ids) or len(ids) != len(glosses) or len(offsets) != len(ids) + 1 or offsets[0] != 0
            or offsets[-1] != len(raw) or np.any(np.diff(offsets) <= 0) or raw.shape[1:] != (75, 3)
            or not np.isfinite(raw).all()):
        raise ValueError(f'Malformed poses: {path}')
    signers = []
    for identity, gloss in zip(ids.tolist(), glosses.tolist(), strict=True):
        match = POSE_ID.fullmatch(identity)
        if match is None or references.get(identity) != gloss:
            raise ValueError(f'Unverified annotation: {identity}')
        signers.append(int(match['signer']))
    return prepare(raw, offsets), ids, glosses, np.array(signers), vocabulary


class IsolatedRecognizer(nn.Module):
    def __init__(self, labels, center, scale):
        super().__init__()
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)
        self.project = nn.Linear(224, WIDTH)
        self.motion = nn.Linear(168, WIDTH, bias=False)
        self.normalize = nn.LayerNorm(WIDTH)
        self.temporal = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in self.temporal)
        self.recurrent = nn.GRU(WIDTH, WIDTH // 2, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(WIDTH, 1)
        self.embedding = nn.Sequential(nn.Linear(WIDTH * 3, WIDTH), nn.LayerNorm(WIDTH), nn.GELU())
        self.dropout = nn.Dropout(.17)
        self.head = nn.Linear(WIDTH, len(labels))

    def forward(self, frames):
        values = (frames - self.center) / self.scale
        hands = values[:, :, :168]
        visible = frames[:, :, 220:222] > .5
        stable = (visible[:, 1:] & visible[:, :-1]).repeat_interleave(84, dim=-1)
        delta = F.pad(torch.diff(hands, dim=1).clamp(-4, 4) * stable, (0, 0, 1, 0))
        encoded = F.gelu(self.normalize(self.project(values) + .35 * self.motion(delta)))
        for convolution, norm in zip(self.temporal, self.norms, strict=True):
            update = convolution(encoded.transpose(1, 2)).transpose(1, 2)
            encoded = norm(encoded + self.dropout(F.gelu(update)))
        recurrent, _ = self.recurrent(encoded)
        attention = self.attention(recurrent).softmax(dim=1)
        pooled = torch.cat(((attention * recurrent).sum(1), recurrent.amax(1), encoded.mean(1)), dim=-1)
        return self.head(self.dropout(self.embedding(pooled)))


def augment(frames):
    axis = torch.linspace(0., 1., FRAMES)[None]
    random = torch.rand(len(frames), 4)
    start, end = .1 * random[:, :1], 1. - .1 * random[:, 1:2]
    positions = (start + (end - start) * axis.pow(.75 + random[:, 2:3] * .55)) * (FRAMES - 1)
    left, right = positions.floor().long(), positions.ceil().long()
    index = torch.arange(len(frames))[:, None]
    sample = torch.lerp(frames[index, left], frames[index, right], (positions - left)[..., None])
    sample[..., 220:] = frames[index, positions.round().long(), 220:]
    coordinates = sample[..., :220].reshape(len(frames), FRAMES, 110, 2)
    angle = .22 * (random[:, 3:4] - .5)
    x, y = coordinates.unbind(-1)
    sine, cosine = angle.sin()[:, :, None], angle.cos()[:, :, None]
    scale = .88 + .24 * torch.rand(len(frames), 1, 1, 2)
    coordinates[:] = torch.stack((cosine * x - sine * y, sine * x + cosine * y), -1) * scale
    return sample


@torch.inference_mode()
def infer(model, data):
    model.eval()
    return torch.cat([model(chunk) for chunk in data.split(128)])


def class_metrics(logits, glosses, labels):
    index = {label: i for i, label in enumerate(labels)}
    target = torch.tensor([index.get(label, -1) for label in glosses])
    known = target >= 0
    rank = logits.topk(5).indices
    hit = rank[:, 0] == target
    total = torch.bincount(target[known], minlength=len(labels))
    correct = torch.bincount(target[hit], minlength=len(labels))
    present = total > 0
    return {'clips': int(known.sum()), 'top1_correct': int(hit.sum()), 'top1': float(hit[known].float().mean()),
            'top5': float((rank[known] == target[known, None]).any(1).float().mean()),
            'macro_top1': float((correct[present] / total[present]).mean()),
            'classes_present': int(present.sum()), 'classes_with_top1_hit': int((correct > 0).sum()),
            'classes_at_least_five_and_80_percent_top1': int(((total >= 5) & (correct >= .8 * total)).sum()),
            'per_gloss': {label: {'clips': int(total[i]), 'correct': int(correct[i])}
                          for i, label in enumerate(labels) if total[i]}}, target


def control_frames(data):
    random = torch.Generator().manual_seed(SEED + 5)
    order = torch.rand((len(data), FRAMES), generator=random).argsort(dim=1)
    return {'static_midpoint': data[:, FRAMES // 2:FRAMES // 2 + 1].expand_as(data),
            'shuffled': data[torch.arange(len(data))[:, None], order]}


def motion(data):
    tracked = data[:, :, 220:223] > .5
    hands = data[:, :, [0, 1, 84, 85]]
    visible = (tracked[:, 1:, :2] & tracked[:, :-1, :2] & tracked[:, 1:, 2:]).repeat_interleave(2, dim=-1)
    change = torch.diff(hands, dim=1).square() * visible
    return (change.sum(dim=(1, 2)) / visible.sum(dim=(1, 2)).clamp_min(1)).sqrt()


def confidence(logits):
    return logits.softmax(-1).max(-1).values


def accept(logits, data, policy):
    return (confidence(logits) >= policy['confidence_floor']) & (motion(data) >= policy['motion_floor'])


def choose_policy(model, data, logits, target, lexical_unknown, *, strict=False):
    known = target >= 0
    correct = (logits.argmax(1) == target) & known
    scores, displacement = confidence(logits), motion(data)
    negatives = [(confidence(infer(model, clips)), motion(clips)) for clips in control_frames(data[known]).values()]
    if lexical_unknown.any():
        negatives.append((scores[lexical_unknown], displacement[lexical_unknown]))
    negative_scores, negative_motion = map(torch.cat, zip(*negatives, strict=True))
    thresholds = sorted({float(item) for item in torch.linspace(.025, .975, 39)} | {0., 1.})
    trials = []
    for floor in (.0025, .005, .01, .02, .035, .05):
        for value in thresholds:
            selected = (scores >= value) & (displacement >= floor)
            false_accept = float(((negative_scores >= value) & (negative_motion >= floor)).float().mean())
            risk = max(float(((group_score >= value) & (group_motion >= floor)).float().mean())
                       for group_score, group_motion in negatives) if strict else false_accept
            false_known = int((selected & known & ~correct).sum())
            trials.append(((max(0., risk - .1), -int((selected & correct).sum()), false_known, risk),
                           {'confidence_floor': value, 'motion_floor': floor, 'dev_negative_acceptance': false_accept,
                            'dev_risk': risk, 'criterion': 'worst negative group' if strict else 'pooled negatives'}))
    return min(trials, key=lambda trial: trial[0])[1]


def fit(arm):
    freeze = REPORT / f'{arm}-freeze.json'
    if freeze.exists():
        raise ValueError(f'Already frozen: {freeze}')
    data, ids, glosses, signers, vocabulary = load_poses(POSE_CACHE, 'train-prototypes')
    if set(signers) != {1, 2, 4}:
        raise ValueError('Expected actors 1, 2, and 4 in the verified pose subset')
    development = signers == 4
    if arm == 'expanded':
        source = EXPANDED_POSES
        train_data, train_ids, train_glosses, train_signers, expanded_vocabulary = load_poses(source, 'train-expanded')
        if (set(train_signers) != {1, 2, 5} or vocabulary != expanded_vocabulary
                or not set(ids[signers != 4]) <= set(train_ids) or set(ids[development]) & set(train_ids)):
            raise ValueError('Expanded training identities differ from the planned signer split')
    else:
        source = POSE_CACHE
        training = signers != 4
        train_data, train_ids, train_glosses, train_signers = data[training], ids[training], glosses[training], signers[training]
    known = Counter(train_glosses)
    candidates = [label for label in set(glosses[development]) if known[label] >= 3]
    reserved = sorted(candidates, key=lambda label: sha256(f'isolated-open-set:{label}'.encode()).digest())[:24]
    dev_unknown, test_unknown = (reserved[:12], reserved[12:]) if arm == 'open' else ([], [])
    labels = [label for label in vocabulary if label in known and label not in dev_unknown + test_unknown]
    training = np.isin(train_glosses, labels)
    development &= ~np.isin(glosses, test_unknown)
    train, dev = train_data[training], data[development]
    train_gloss, dev_gloss = train_glosses[training], glosses[development]
    index = {label: i for i, label in enumerate(labels)}
    targets = torch.tensor([index[label] for label in train_gloss])
    weights = torch.bincount(targets).float().clamp_min(1).rsqrt()
    torch.manual_seed(SEED)
    model = IsolatedRecognizer(labels, train.mean((0, 1)), train.std((0, 1), unbiased=False).clamp_min(.08))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.006)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.00025)
    interval = ceil(int((signers != 4).sum()) / BATCH) if arm == 'expanded' else ceil(len(train) / BATCH)
    budget = EPOCHS * interval
    history, losses, best, updates = [], [], None, 0
    print(f'{arm}: {len(train)} train, {len(dev)} dev, {len(labels)} labels', flush=True)
    while updates < budget:
        model.train()
        for batch in torch.randperm(len(train)).split(BATCH):
            logits = model(augment(train[batch]))
            loss = F.cross_entropy(logits, targets[batch], weight=weights, label_smoothing=.035)
            if not loss.isfinite():
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
            if updates % interval == 0:
                schedule.step()
                result, _ = class_metrics(infer(model, dev), dev_gloss, labels)
                del result['per_gloss']
                key = result['macro_top1'], result['top1'], result['top5']
                epoch = updates // interval
                if best is None or key > best[0]:
                    best = key, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}
                history.append({'epoch': epoch, 'updates': updates, 'loss': sum(losses) / len(losses), **result})
                losses.clear()
                if epoch == 1 or epoch % 7 == 0:
                    print(json.dumps(history[-1]), flush=True)
                model.train()
            if updates == budget:
                break
    model.load_state_dict(best[2])
    dev_logits = infer(model, dev)
    _, dev_targets = class_metrics(dev_logits, dev_gloss, labels)
    policy = choose_policy(model, dev, dev_logits, dev_targets, torch.from_numpy(np.isin(dev_gloss, dev_unknown)))
    checkpoint = ARTIFACTS / f'{arm}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'labels': labels, 'state_dict': model.state_dict(), 'policy': policy, 'frames': FRAMES}, checkpoint)
    write(freeze, {'checkpoint_sha256': file_hash(checkpoint), 'pose_cache_sha256': file_hash(source),
                   'development_pose_cache_sha256': file_hash(POSE_CACHE), 'updates': updates,
                   'train_ids_sha256': sha256('\n'.join(train_ids[training]).encode()).hexdigest(),
                   'development_ids_sha256': sha256('\n'.join(ids[development]).encode()).hexdigest(),
                   'train_clips': len(train), 'development_clips': len(dev), 'labels': labels, 'seed': SEED,
                   'train_signers': sorted(set(train_signers[training].tolist())), 'development_signers': [4], 'test_signers': [3],
                   'dev_unknown': dev_unknown, 'test_unknown': test_unknown, 'selected_epoch': best[1],
                   'policy': policy, 'development_history': history})


def load_checkpoint(arm):
    freeze = json.loads((REPORT / f'{arm}-freeze.json').read_text(encoding='utf-8'))
    path = ARTIFACTS / f'{arm}.pt'
    source = EXPANDED_POSES if arm == 'expanded' else POSE_CACHE
    if (file_hash(path) != freeze['checkpoint_sha256'] or file_hash(source) != freeze['pose_cache_sha256']
            or file_hash(POSE_CACHE) != freeze.get('development_pose_cache_sha256', freeze['pose_cache_sha256'])):
        raise ValueError('Frozen checkpoint or source poses changed')
    state = torch.load(path, weights_only=True, map_location='cpu')
    weights, labels = state['state_dict'], state['labels']
    model = IsolatedRecognizer(labels, weights['center'], weights['scale'])
    model.load_state_dict(weights)
    return model, labels, freeze


def score(arm):
    model, labels, freeze = load_checkpoint(arm)
    development, dev_ids, dev_glosses, dev_signers, _ = load_poses(POSE_CACHE, 'train-prototypes')
    dev_mask = (dev_signers == 4) & ~np.isin(dev_glosses, freeze['test_unknown'])
    if sha256('\n'.join(dev_ids[dev_mask]).encode()).hexdigest() != freeze['development_ids_sha256']:
        raise ValueError('Frozen development IDs changed')
    dev_clips, dev_references = development[dev_mask], dev_glosses[dev_mask]
    dev_logits = infer(model, dev_clips)
    np.savez_compressed(REPORT / f'{arm}-dev-scores.npz', ids=dev_ids[dev_mask], glosses=dev_references,
                        labels=np.array(labels), logits=dev_logits.numpy())
    policies = {'pooled': freeze['policy']}
    if arm == 'open':
        _, targets = class_metrics(dev_logits, dev_references, labels)
        dev_unknown = torch.from_numpy(np.isin(dev_references, freeze['dev_unknown']))
        policies['worst_group'] = choose_policy(model, dev_clips, dev_logits, targets, dev_unknown, strict=True)
        write(REPORT / 'open-strict-policy.json', {
            'source_freeze_sha256': file_hash(REPORT / 'open-freeze.json'), 'policy': policies['worst_group'],
        })
    data, ids, glosses, signers, _ = load_poses(HELD_POSES, 'test')
    if set(signers) != {3}:
        raise ValueError('Expected only held signer 3')
    keep = ~np.isin(glosses, freeze['dev_unknown'])
    data, ids, glosses = data[keep], ids[keep], glosses[keep]
    logits = infer(model, data)
    result, target = class_metrics(logits, glosses, labels)
    known, unknown = target >= 0, target < 0
    correct = logits.argmax(1) == target
    controls = {name: (clips, infer(model, clips)) for name, clips in control_frames(data[known]).items()}
    outcomes = {}
    for name, policy in policies.items():
        accepted = accept(logits, data, policy)
        selective = {'known_accepted': int((accepted & known).sum()), 'correct_known_accepted': int((accepted & correct).sum()),
                     'unknown_clips': int(unknown.sum()), 'unknown_accepted': int((accepted & unknown).sum())}
        negatives = {key: {'clips': len(clips), 'accepted': int(accept(values, clips, policy).sum())}
                     for key, (clips, values) in controls.items()}
        outcomes[name] = {'policy': policy, 'selective': selective, 'controls': negatives}
    result.update({key: outcomes['pooled'][key] for key in ('selective', 'controls')})
    np.savez_compressed(REPORT / f'{arm}-scores.npz', ids=ids, glosses=glosses, labels=np.array(labels), logits=logits.numpy())
    output = {'scope': 'Native CERTH isolated pose replay. Held actor 3. No webcam or continuous recognition.',
              'freeze_sha256': file_hash(REPORT / f'{arm}-freeze.json'), 'test_pose_sha256': file_hash(HELD_POSES),
              'policy': freeze['policy'], 'reserved_test_glosses': freeze['test_unknown'], 'summary': result,
              'policy_outcomes': outcomes}
    write(REPORT / f'{arm}-benchmark.json', output)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_gloss'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'score'))
    parser.add_argument('--arm', choices=('full', 'open', 'expanded'), default='full')
    args = parser.parse_args()
    torch.set_num_threads(2)
    (fit if args.phase == 'train' else score)(args.arm)


if __name__ == '__main__':
    main()
