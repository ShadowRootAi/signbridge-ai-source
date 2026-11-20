"""Train a signer-disjoint motion classifier and evaluate it alongside OpenHands."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import decisions, fit_temperature, softmax, tune_rejection
from evaluate_openhands_scope_veto import session
from evaluate_recognition_scale_pilot import write_json
from openhands_gsl_stgcn import METADATA_ARCHIVE_SHA256, prepare_stgcn_pose


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'research/cache/openhands-gsl'
MODEL_DIR = ROOT / 'artifacts/judge-final-motion'
REPORT = ROOT / 'reports/judge-final'
POLICY = ROOT / 'publication-judge-demo/dist/__signbridge/openhands/policy.json'
PRIMARY = CACHE / 'stgcn.onnx'
SECONDARY = CACHE / 'lstm.onnx'
CONFIG = CACHE / 'stgcn-onnx.json'
TRAIN = CACHE / 'official-train-expanded-poses.npz'
CALIBRATION = CACHE / 'official-train-prototypes-poses.npz'
HELD = CACHE / 'judge-final-held-poses.npz'
STEPS = 32
SEED = 81


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def authority():
    metadata = CACHE / 'gsl_metadata.zip'
    if digest(metadata) != METADATA_ARCHIVE_SHA256:
        raise ValueError('Original class metadata changed')
    with ZipFile(metadata) as archive:
        return dict(line.split('|', 1) for line in archive.read('GSL_split/GSL_isolated/train_greek_iso.csv')
                    .decode('utf-8-sig').splitlines())


def load_native(path, signers, official):
    with np.load(path, allow_pickle=False) as archive:
        ids, words, points, offsets = (archive[key] for key in ('ids', 'glosses', 'keypoints', 'offsets'))
    if (len(set(ids)) != len(ids) or offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(points)
            or np.any(np.diff(offsets) <= 0) or not np.isfinite(points).all()):
        raise ValueError('Unexpected pose archive layout')
    speakers = np.array([int(re.search(r'_signer(\d+)_', name)[1]) for name in ids])
    keep = np.flatnonzero(np.isin(speakers, signers))
    if set(speakers[keep]) != set(signers) or any(official.get(ids[i]) != words[i] for i in keep):
        raise ValueError('Missing expected signers or invalid source annotation')
    poses = [prepare_stgcn_pose(points[offsets[i]:offsets[i + 1]]).numpy()[0] for i in keep]
    return ids[keep], words[keep], speakers[keep], poses


def resize(pose, size):
    axis = np.linspace(0, pose.shape[1] - 1, size)
    prior = axis.astype(np.int64)
    following = np.minimum(prior + 1, pose.shape[1] - 1)
    fraction = (axis - prior).astype(np.float32)[None, :, None]
    return pose[:, prior, :] + fraction * (pose[:, following, :] - pose[:, prior, :])


class MotionNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(148, 80, 5, padding=2), nn.BatchNorm1d(80), nn.GELU(),
            nn.Conv1d(80, 96, 3, padding=2, dilation=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 96, 3, padding=4, dilation=4), nn.BatchNorm1d(96), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 5, 128), nn.GELU(), nn.Dropout(.18), nn.Linear(128, classes + 1))

    def forward(self, poses):
        location = poses.transpose(2, 3).reshape(-1, 54, STEPS)
        speed = F.pad(location[:, :, 1:] - location[:, :, :-1], (1, 0))
        relative = torch.cat((poses[:, :, :, 7:17] - poses[:, :, :, 7:8],
                              poses[:, :, :, 17:27] - poses[:, :, :, 17:18]), dim=3)
        relative = relative.transpose(2, 3).reshape(-1, 40, STEPS)
        x = torch.cat((location.clamp(-12, 12), (speed * 4).clamp(-8, 8), relative.clamp(-6, 6)), dim=1)
        features = self.features(x)
        pooled = torch.cat((features.reshape(-1, 96, 4, 8).mean(dim=3), features.amax(dim=2, keepdim=True)), dim=2)
        return self.head(pooled)


def infer(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(chunk) for chunk in torch.from_numpy(x).split(128)]).numpy()


def train_model(train_poses, target, calibration_poses, calibration_targets, count):
    x = torch.from_numpy(np.stack([resize(pose, STEPS) for pose in train_poses]))
    y = torch.from_numpy(target)
    valid = np.stack([resize(pose, STEPS) for pose in calibration_poses])
    values = torch.from_numpy(calibration_targets)
    torch.manual_seed(SEED)
    model = MotionNet(count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.004)
    generator = torch.Generator().manual_seed(SEED)
    frequency = y.bincount(minlength=count + 1).float().clamp_min(1)
    weight = (frequency.median() / frequency).sqrt().clamp(.4, 3.5)
    # Keep varied OTHER recordings in the optimization despite pooling many
    # separate lexical signs into this one output.
    weight[-1] = 1.
    best, best_epoch, checkpoint, history = float('inf'), 0, None, []
    for epoch in range(1, 19):
        model.train()
        for batch in torch.randperm(len(x), generator=generator).split(160):
            sample = x[batch]
            variation = 1 + (torch.rand((len(batch), 1, 1, 1), generator=generator) - .5) * .16
            jitter = torch.randn(sample.shape, generator=generator) * .008
            loss = F.cross_entropy(model(sample * variation + jitter), y[batch], weight=weight, label_smoothing=.025)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
        prediction = torch.from_numpy(infer(model, valid))
        loss = float(F.cross_entropy(prediction, values, weight=weight))
        accuracy = float((prediction.argmax(dim=1) == values).float().mean())
        history.append({'epoch': epoch, 'validation_cross_entropy': loss, 'validation_accuracy_with_other': accuracy})
        if loss < best:
            best, best_epoch = loss, epoch
            checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
        print(f'epoch {epoch}: validation loss {loss:.4f}, accuracy {accuracy:.3f}', flush=True)
        if epoch - best_epoch >= 4:
            break
    model.load_state_dict(checkpoint)
    return model, valid, best_epoch, history


def openhands_logits(model, poses, lstm=False):
    # The released ONNX graphs support variable time but one clip per call.
    output = []
    for pose in poses:
        x = pose[None]
        if lstm:
            x = x.transpose(0, 2, 1, 3).reshape(1, -1, 54).copy()
        output.append(model.run(['logits'], {'landmarks': x})[0][0])
    return np.stack(output)


def native_openhands(poses, upstream, policy):
    primary = session(PRIMARY, policy['runtimeModelSha256'])
    agreement = policy['agreementGate']
    secondary = session(SECONDARY, agreement['modelSha256'])
    logits = openhands_logits(primary, poses)
    winner = logits.argmax(axis=1)
    wins = (np.arange(len(poses)), winner)
    p = softmax(logits)[wins]
    selected = np.asarray(upstream)[winner]
    added = np.isin(selected, policy['expansionGate']['labels'])
    evidence = logits[wins] >= policy['decisionGate']['minimumEvidence']
    eligible = np.isin(selected, policy['supportedLabels']) & evidence
    thresholds = np.where(added, policy['expansionGate']['minimumConfidence'], policy['decisionGate']['minimumConfidence'])
    accepted = eligible & (p >= thresholds)
    candidates = np.flatnonzero(eligible & ~accepted & ~added & (p >= agreement['minimumConfidence']))
    if len(candidates):
        second = openhands_logits(secondary, [poses[i] for i in candidates], lstm=True)
        reference = winner[candidates]
        score = second[np.arange(len(candidates)), reference]
        ranked = (second > score[:, None]).sum(axis=1) < agreement['maximumSecondaryRank']
        support = np.where(ranked, softmax(second)[np.arange(len(second)), reference], 0.)
        accepted[candidates[support >= agreement['minimumSecondaryProbability']]] = True
        temporal = agreement['temporal']
        tests, origins = [], []
        for i, s in zip(candidates, support, strict=True):
            sizes = [max(4, int(poses[i].shape[1] * factor + .5)) for factor in (.65, .8)]
            if (not accepted[i] and s >= temporal['minimumSecondaryProbability']
                    and p[i] >= temporal['minimumPrimaryConfidence'] and sizes[0] < sizes[1] < poses[i].shape[1]):
                tests.extend(resize(poses[i], length) for length in sizes)
                origins.append(i)
        if tests:
            check = openhands_logits(primary, tests)
            indices = np.repeat(winner[origins], 2)
            matches = check.argmax(axis=1) == indices
            scores = softmax(check)[np.arange(len(tests)), indices].reshape(-1, 2)
            passes = (matches.reshape(-1, 2).all(axis=1)
                      & (check[np.arange(len(tests)), indices].reshape(-1, 2).min(axis=1) >= policy['decisionGate']['minimumEvidence'])
                      & (scores.min(axis=1) >= temporal['minimumConsistencyConfidence'])
                      & (scores.max(axis=1) >= temporal['minimumPeakConfidence']))
            accepted[np.asarray(origins)[passes]] = True
    return selected, p, accepted, eligible


def summarize(reference, predicted, accepted, classes):
    known = np.isin(reference, classes)
    correct = reference == predicted
    return {'known': int(known.sum()), 'outside_roster': int((~known).sum()), 'known_top1_correct': int((known & correct).sum()),
            'correct_accepted': int((known & correct & accepted).sum()),
            'wrong_known_accepted': int((known & ~correct & accepted).sum()),
            'outside_accepted': int((~known & accepted).sum()), 'abstained': int((~accepted).sum())}


def hybrid_accept(base, probability, classes, gate):
    words, confidence, accepted, eligible = base
    best = probability.argmax(axis=1)
    head_word = np.asarray([*classes, ''])[best]
    other = probability[:, -1]
    veto = accepted & (best == len(classes)) & (other >= gate['other_veto'])
    rescue = (~accepted & eligible & (head_word == words) & (confidence >= gate['primary'])
              & (probability.max(axis=1) >= gate['head']))
    return (accepted & ~veto) | rescue


def choose_hybrid(base, probabilities, reference, classes):
    current = summarize(reference, base[0], base[2], classes)
    options = []
    for veto in (.8, .9, .97, 1.01):
        for primary in (.97, .99, .995, 1.01):
            for head in (.55, .75, .9, 1.01):
                gate = {'other_veto': veto, 'primary': primary, 'head': head}
                summary = summarize(reference, base[0], hybrid_accept(base, probabilities, classes, gate), classes)
                if (summary['wrong_known_accepted'] > current['wrong_known_accepted']
                        or summary['outside_accepted'] > current['outside_accepted']):
                    continue
                ranking = (summary['correct_accepted'], -summary['outside_accepted'], -summary['wrong_known_accepted'], veto, primary, head)
                options.append((ranking, gate, summary))
    _, gate, summary = max(options, key=lambda value: value[0])
    return gate, current, summary


def target_ids(reference, classes):
    lookup = {label: i for i, label in enumerate(classes)}
    return np.array([lookup.get(word, len(classes)) for word in reference], dtype=np.int64)


def fit():
    policy = json.loads(POLICY.read_bytes())
    upstream = json.loads(CONFIG.read_bytes())['labels']
    classes = sorted(policy['supportedLabels'])
    official = authority()
    train_ids, train_labels, train_people, train_poses = load_native(TRAIN, [1, 2, 5], official)
    dev_ids, dev_labels, dev_people, dev_poses = load_native(CALIBRATION, [4], official)
    if set(train_ids) & set(dev_ids) or (set(classes) - set(train_labels)):
        raise ValueError('Train and validation overlap or classes are missing')
    train_y, dev_y = target_ids(train_labels, classes), target_ids(dev_labels, classes)
    count = len(classes)
    network, dev_x, epoch, history = train_model(train_poses, train_y, dev_poses, dev_y, count)
    logits = infer(network, dev_x)
    temperature = fit_temperature(logits, dev_y)
    probability = softmax(logits, temperature)
    quality = np.ones(len(dev_y))
    standalone_policy = tune_rejection(probability, np.where(dev_y == count, -1, dev_y), quality, count,
                                       target_far=.012, target_accepted_accuracy=.97)
    base = native_openhands(dev_poses, upstream, policy)
    gate, baseline, hybrid = choose_hybrid(base, probability, dev_labels, classes)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), MODEL_DIR / 'motion.pt')
    torch.onnx.export(network.eval(), (torch.from_numpy(dev_x[:1]),), str(MODEL_DIR / 'motion.onnx'),
                      input_names=['landmarks'], output_names=['logits'],
                      dynamic_axes={'landmarks': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=17, dynamo=False, external_data=False)
    metadata = {'classes': classes, 'temperature': temperature, 'standalone': standalone_policy, 'hybrid': gate,
                'baseline_policy_sha256': digest(POLICY), 'source_sha256': digest(Path(__file__)),
                'files': {name: digest(MODEL_DIR / name) for name in ('motion.pt', 'motion.onnx')},
                'train_sha256': digest(TRAIN), 'calibration_sha256': digest(CALIBRATION),
                'train': {'signers': sorted(set(train_people.tolist())), 'clips': len(train_ids), 'other': int((train_y == count).sum())},
                'validation': {'signers': sorted(set(dev_people.tolist())), 'clips': len(dev_ids), 'baseline': baseline, 'hybrid': hybrid},
                'best_epoch': epoch, 'history': history, 'parameters': sum(parameter.numel() for parameter in network.parameters())}
    write_json(MODEL_DIR / 'freeze.json', metadata)
    print(json.dumps({'train': metadata['train'], 'validation': metadata['validation'], 'hybrid': gate}), flush=True)


def restore():
    state = json.loads((MODEL_DIR / 'freeze.json').read_bytes())
    if (state['baseline_policy_sha256'] != digest(POLICY) or state['source_sha256'] != digest(Path(__file__))
            or any(digest(MODEL_DIR / name) != value for name, value in state['files'].items())):
        raise ValueError('Evaluation inputs differ from the frozen model')
    model = MotionNet(len(state['classes']))
    model.load_state_dict(torch.load(MODEL_DIR / 'motion.pt', map_location='cpu', weights_only=True))
    return state, model


def compare(state, model, ids, reference, poses, base, usable):
    classes = state['classes']
    p = np.zeros((len(poses), len(classes) + 1))
    x = np.stack([resize(pose, STEPS) for pose, good in zip(poses, usable, strict=True) if good])
    logits = infer(model, x)
    p[usable] = softmax(logits, state['temperature'])
    p[~usable, -1] = 1
    standalone = state['standalone']
    predicted, accepted, _ = decisions(p, usable.astype(float), len(classes), standalone['confidence_threshold'], standalone['min_quality'])
    compared = hybrid_accept(base, p, classes, state['hybrid']) & usable
    candidate = summarize(reference, base[0], compared, classes)
    current = summarize(reference, base[0], base[2], classes)
    onnx = session(MODEL_DIR / 'motion.onnx', state['files']['motion.onnx'])
    parity = float(np.abs(onnx.run(['logits'], {'landmarks': x})[0] - logits).max())
    if parity > .0002:
        raise ValueError(f'Model export parity failed: {parity}')
    return {'baseline': current, 'learned_alone': summarize(reference, np.asarray(classes)[predicted], accepted, classes),
            'hybrid': candidate, 'usable': int(usable.sum()), 'onnx_max_abs_logit_error': parity,
            'changed': [{'id': str(ids[i]), 'reference': str(reference[i]), 'candidate': str(base[0][i]),
                         'baseline_accepts': bool(base[2][i]), 'hybrid_accepts': bool(compared[i])}
                        for i in np.flatnonzero(base[2] != compared)]}


def native(held):
    state, model = restore()
    ids, words, signers, poses = load_native(held, [6, 7], authority())
    policy = json.loads(POLICY.read_bytes())
    base = native_openhands(poses, json.loads(CONFIG.read_bytes())['labels'], policy)
    result = compare(state, model, ids, words, poses, base, np.ones(len(ids), dtype=bool))
    report = {'scope': 'Native-pose data from signers 6 and 7 held out of the new head. The upstream OpenHands '
                       'training split includes these actors. Raw browser tracking/motion gates are unobserved here.',
              'held_sha256': digest(held), 'freeze_sha256': digest(MODEL_DIR / 'freeze.json'),
              'held_signers': sorted(set(signers.tolist())), **result}
    write_json(REPORT / 'native-comparison.json', report)
    print(json.dumps({key: value for key, value in result.items() if key != 'changed'}))


def rgb(source):
    state, model = restore()
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    inputs = [ROOT / 'reports/openhands-engine/replay-manifest.json', ROOT / 'data/isolated-sartinas-outside-roster/manifest.json']
    clip_sources = {}
    for path in inputs:
        for item in json.loads(path.read_bytes())['clips']:
            if item['id'] in clip_sources:
                raise ValueError('The RGB source manifests overlap')
            clip_sources[item['id']] = (digest(path), item['sha256'])
    if (len(rows) != len(clip_sources) or set(clip_sources) != {row['id'] for row in rows}
            or any((row['manifestSha256'], row['sha256']) != clip_sources[row['id']]
                   or row['seekOffsetMs'] != 2 or row['policySha256'] != state['baseline_policy_sha256'] for row in rows)):
        raise ValueError('RGB records do not match the fixed manifest, runtime policy and sampling offset')
    words = np.asarray([row['gloss'] for row in rows])
    poses = [np.asarray(row['pose'], dtype=np.float32).reshape(2, row['frames'], 27) if row['pose'] is not None else None for row in rows]
    outcomes = [row['base'] for row in rows]
    usable = np.asarray([prediction['classifierExecuted'] and pose is not None for pose, prediction in zip(poses, outcomes, strict=True)])
    winning = np.asarray([row['topCandidates'][0]['gloss'] if row['topCandidates'] else '' for row in outcomes])
    probability = np.asarray([row['confidence'] or 0 for row in outcomes])
    accepted = np.asarray([row['accepted'] for row in outcomes])
    eligible = np.isin(winning, state['classes']) & usable & np.asarray([row['rejectionCode'] != 'low_evidence' for row in outcomes])
    result = compare(state, model, [row['id'] for row in rows], words, poses, (winning, probability, accepted, eligible), usable)
    report = {'scope': 'Previously examined RGB videos through the exact judge-build browser landmark extractor and '
                       'frozen browser decision. The new hybrid decision is replayed on those same pose tensors.',
              'browser_export_sha256': digest(source), 'freeze_sha256': digest(MODEL_DIR / 'freeze.json'), **result}
    write_json(REPORT / 'rgb-comparison.json', report)
    cases = np.flatnonzero(usable)
    selected = cases[np.linspace(0, len(cases) - 1, num=min(6, len(cases)), dtype=int)]
    test_x = np.stack([resize(poses[i], STEPS) for i in selected])
    write_json(source.with_name('motion-onnx-fixture.json'), {'model_sha256': state['files']['motion.onnx'],
        'rows': [{'id': rows[i]['id'], 'features': x.ravel().tolist(), 'logits': row_logits.tolist()}
                 for i, x, row_logits in zip(selected, test_x, infer(model, test_x), strict=True)]})
    print(json.dumps({key: value for key, value in result.items() if key != 'changed'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'native', 'rgb'))
    parser.add_argument('--held', type=Path, default=HELD)
    parser.add_argument('--rgb', type=Path, default=ROOT / '.runtime-tmp/judge-final-rgb/poses.jsonl')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == 'fit':
        fit()
    elif args.phase == 'native':
        native(args.held)
    else:
        rgb(args.rgb)
