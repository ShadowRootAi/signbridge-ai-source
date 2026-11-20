"""Transfer a CTC encoder from synthetic official pose sequences to real GSL RGB."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import random
import re
from itertools import accumulate
from statistics import mean
import time

import numpy as np
import torch
from torch import nn

from evaluate_continuous_transfer import CHECKPOINT, SOURCES, load_inputs
from preprocess import POSE_INDICES, frame_features
from train_continuous import ContinuousCTC, Utterance, batch_loss, choose_threshold, reference_lookup, replay, summarize


PLAN = Path('reports/pretrained-gsl/synthetic-continuous-train-plan.json')
REPORT = Path('reports/continuous-v1-expanded/synthetic-pretraining-transfer.json')
POSE_CACHE = Path('research/cache/openhands-gsl/official-train-prototypes-poses.npz')
ASPECT = 848 / 480
SOURCE_SIGNERS = {1, 2, 4}
MAX_SYNTHETIC_SAMPLES = 720
SEED = 41


def file_hash(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def frame_feature_array(keypoints):
    points = keypoints[..., :2].astype(np.float64)
    pose, left, right = points[:, :33], points[:, 33:54], points[:, 54:75]
    pose_ok, left_ok, right_ok = (np.any(part != 0, axis=(1, 2)) for part in (pose, left, right))
    points[..., 0] *= ASPECT
    shoulders = pose[:, [11, 12]]
    origin = shoulders.mean(1)[:, None, :]
    scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1).clip(min=.05)[:, None, None]
    blocks = []
    for hand, visible in ((left, left_ok), (right, right_ok)):
        shape_scale = np.linalg.norm(hand[:, 0] - hand[:, 9], axis=1).clip(min=.01)[:, None, None]
        blocks.extend((np.where((visible & pose_ok)[:, None, None], (hand - origin) / scale, 0.),
                       np.where(visible[:, None, None], (hand - hand[:, :1]) / shape_scale, 0.)))
    body = np.where(pose_ok[:, None, None], (pose[:, POSE_INDICES] - origin) / scale, 0.)
    return np.column_stack((*(block.reshape(len(points), -1) for block in [*blocks, body]),
                            np.zeros((len(points), 34)), left_ok, right_ok, pose_ok, np.zeros(len(points)))).clip(-8., 8.).astype(np.float32)


def verify_feature_math(keypoints, features):
    for i in np.linspace(0, len(keypoints) - 1, 16, dtype=int):
        frame = {'width': 848, 'height': 480, 'face': []}
        for part, start, end in (('pose', 0, 33), ('left', 33, 54), ('right', 54, 75)):
            coordinates = keypoints[i, start:end, :2]
            frame[part] = [{'x': float(x), 'y': float(y)} for x, y in coordinates] if coordinates.any() else []
        expected, _ = frame_features(frame)
        np.testing.assert_allclose(features[i], expected, atol=2e-7, rtol=2e-7)


def read_synthetic(plan_path, *, excluded_templates=(), excluded_sequences=(), alignments=None):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    pose_hash = file_hash(POSE_CACHE)
    if Path(plan['poseFile']).resolve() != POSE_CACHE.resolve() or plan['poseFileSha256'].lower() != pose_hash:
        raise ValueError('The ordered synthetic plan does not bind to the exact official training pose cache')
    with np.load(POSE_CACHE, allow_pickle=False) as cache:
        poses, ids, offsets, refs = (cache[key] for key in ('keypoints', 'ids', 'offsets', 'glosses'))
    if (len(set(ids)) != len(ids) or not np.isfinite(poses).all() or len(offsets) != len(ids) + 1
            or offsets[0] != 0 or offsets[-1] != len(poses) or np.any(np.diff(offsets) < 1)):
        raise ValueError('Malformed pose cache')
    features = frame_feature_array(poses)
    verify_feature_math(poses, features)
    rows = [row for row in plan['samples'] if row['template'] not in excluded_templates
            and tuple(row['glosses']) not in excluded_sequences]
    pools = defaultdict(list)
    for index, (identity, gloss) in enumerate(zip(ids, refs, strict=True)):
        match = re.search(r'_signer(\d+)_rep1_glosses/', identity)
        if not match or int(match[1]) not in SOURCE_SIGNERS:
            raise ValueError(f'Unexpected signer or repetition in official TRAIN cache: {identity}')
        pools[int(match[1]), gloss].append(index)
    aligned = []
    combinations = set()
    for row in rows:
        signer = int(row['signer'])
        indices = row['poseIndices']
        if (signer not in SOURCE_SIGNERS or any(type(i) is not int or not 0 <= i < len(ids) for i in indices)
                or [str(ids[i]) for i in indices] != row['poseIds'] or [str(refs[i]) for i in indices] != row['glosses']
                or any(f'_signer{signer}_rep1_glosses/' not in str(ids[i]) for i in indices)):
            raise ValueError(f'Synthetic label, provenance or signer mismatch: {row["id"]}')
        choice = tuple(indices)
        if choice in combinations:
            raise ValueError(f'Duplicate ordered pose list: {row["id"]}')
        combinations.add(choice)
        aligned.append((row, row['id'], choice))

    # Source-aligned examples anchor the ordering. Additional examples retain
    # the label sequence but choose alternate isolated tokens from the same signer.
    variants = []
    generator = random.Random(SEED)
    for attempt in range(1, 9):
        for row, _, source_indices in generator.sample(aligned, len(aligned)):
            if len(aligned) + len(variants) >= MAX_SYNTHETIC_SAMPLES:
                break
            signer = int(row['signer'])
            selected = tuple(generator.choice(pools[signer, label]) for label in row['glosses'])
            if selected not in combinations and selected != source_indices:
                combinations.add(selected)
                variants.append((row, f'{row["id"]}-resampled-{attempt}', selected))
    recipe = [*aligned, *variants]
    result = []
    for row, identity, indices in recipe:
        signer = int(row['signer'])
        # Published pose cache approximates 30 FPS. Keep each independently
        # clipped token at 15 FPS and retain its observed frames and order.
        tokens = [features[offsets[i]:offsets[i + 1]:2] for i in indices]
        sequence = np.concatenate(tokens, axis=0)
        minimum = len(indices) + sum(a == b for a, b in zip(row['glosses'], row['glosses'][1:]))
        if len(sequence) < minimum:
            raise ValueError(f'CTC cannot align the synthetic sample: {row["id"]}')
        if alignments is not None:
            endpoints = [0, *accumulate(map(len, tokens))]
            alignments[identity] = tuple(zip(endpoints[:-1], endpoints[1:], indices, strict=True))
        result.append(Utterance(identity, row['sourcePath'], str(signer), 'train', tuple(row['glosses']),
                                row['text'], torch.from_numpy(sequence), 15., 1., None))
    if len(result) <= 600 or len({item.id for item in result}) != len(result):
        raise ValueError('Expected more than 600 uniquely identified synthetic sequences')
    recipe_json = json.dumps([(identity, indices) for _, identity, indices in recipe], separators=(',', ':'))
    return result, {'planSha256': file_hash(plan_path), 'poseFileSha256': pose_hash,
                    'orderedRecipeSha256': sha256(recipe_json.encode()).hexdigest(),
                    'usedPoseIds': len({index for combination in combinations for index in combination}),
                    'excludedSourceAlignedRows': len(plan['samples']) - len(rows),
                    'sourceAlignedSynthetic': len(aligned), 'sameSignerResampledSynthetic': len(variants),
                    'sourceTemplates': len({row['template'] for row in rows}),
                    'uniqueOrderedGlossTuples': len({item.glosses for item in result}), 'clips': len(result),
                    'signers': dict(Counter(item.signer for item in result)), 'featureParityFrames': 16,
                    'frameLengths': [min(len(item.frames) for item in result), max(len(item.frames) for item in result)]}


def vocabulary(items):
    labels = [None, *sorted({gloss for item in items for gloss in item.glosses})]
    return labels, {label: index for index, label in enumerate(labels) if index}


def optimize(model, training, labels, indices, epochs, size, validation=(), lookup=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    objective = nn.CTCLoss(blank=0)
    generator = random.Random(SEED)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = generator.sample(training, len(training))
        losses = []
        for start in range(0, len(shuffled), size):
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, shuffled[start:start + size], indices, objective)
            if not math.isfinite(float(loss.detach())):
                raise ValueError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {'epoch': epoch, 'train_ctc_loss': mean(losses)}
        if validation:
            metrics = summarize(replay(model, validation, labels, lookup))
            row.update(validation_gloss_wer=metrics['gloss_wer'], validation_gloss_sequence_exact=metrics['gloss_sequence_exact'])
            key = (metrics['gloss_wer'], -metrics['gloss_sequence_exact'], row['train_ctc_loss'])
            if best is None or key < best[0]:
                best = key, epoch, {name: value.detach().clone() for name, value in model.state_dict().items()}
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % 4 == 0:
            print(json.dumps(row), flush=True)
    if best is not None:
        model.load_state_dict(best[2])
    return history, best[1] if best else epochs


def transfer_weights(model, prior, labels, previous_labels):
    current = model.state_dict()
    previous = prior.state_dict()
    head = ('classifier.1.weight', 'classifier.1.bias')
    for key in current.keys() - set(head):
        current[key].copy_(previous[key])
    for index, label in enumerate(labels):
        if label in previous_labels:
            source = previous_labels.index(label)
            for key in head:
                current[key][index].copy_(previous[key][source])
    model.load_state_dict(current)


def evaluation(model, cohorts, labels, lookup, threshold):
    result = {}
    for name, items in cohorts.items():
        rows = replay(model, items, labels, lookup, threshold)
        result[name] = {**summarize(rows), 'predictions': rows}
        if name == 'genuine_out_of_scope':
            result[name]['false_accepts'] = result[name]['accepted']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--pretrain-epochs', type=int, default=12)
    parser.add_argument('--fine-tune-epochs', type=int, default=20)
    args = parser.parse_args()
    if not 10 <= args.pretrain_epochs <= 20 or not 10 <= args.fine_tune_epochs <= 20:
        parser.error('Use 10–20 epochs per stage for this bounded experiment')
    torch.set_num_threads(2)
    synthetic, provenance = read_synthetic(args.plan)
    roles = load_inputs('expanded')
    training, validation = roles['train'], roles['val']
    if {item.signer for item in synthetic} & {item.signer for role in ('val', 'test') for item in roles[role]}:
        raise ValueError('Synthetic signers overlap real validation or held-out signers')
    labels, index = vocabulary(training)
    previous_labels, previous_index = vocabulary(synthetic)
    lookup = reference_lookup(training)
    all_frames = torch.cat([item.frames for item in training])
    center, scale = all_frames.mean(0), all_frames.std(0, unbiased=False).clamp_min(.08)
    print(json.dumps({'synthetic': provenance, 'synthetic_vocabulary': len(previous_index),
                      'matched_fine_tune_labels': sum(label in previous_index for label in index)}), flush=True)
    started = time.perf_counter()
    torch.manual_seed(SEED)
    prior = ContinuousCTC(len(previous_labels), center, scale)
    print('Pretraining on synthetic clips', flush=True)
    pretrain_history, _ = optimize(prior, synthetic, previous_labels, previous_index, args.pretrain_epochs, 32)

    models = {}
    for name in ('equal_budget_real_only', 'synthetic_transfer'):
        torch.manual_seed(SEED)
        model = ContinuousCTC(len(labels), center, scale)
        if name == 'synthetic_transfer':
            transfer_weights(model, prior, labels, previous_labels)
        print(name, flush=True)
        history, epoch = optimize(model, training, labels, index, args.fine_tune_epochs, 12, validation, lookup)
        threshold = choose_threshold(replay(model, validation, labels, lookup))
        models[name] = {'model': model, 'threshold': threshold, 'checkpoint_epoch': epoch, 'history': history}

    old = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = old['state_dict']
    frozen = ContinuousCTC(len(old['labels']), state['mean'], state['scale'])
    frozen.load_state_dict(state)
    models['archived_full_budget'] = {'model': frozen, 'threshold': old['confidence_threshold']}
    if labels != old['labels'] or lookup != old['training_greek_references']:
        raise ValueError('Archived model and new training use different output classes or Greek lookup')
    cohorts = {'development': validation, 'known_held_signers': roles['test'],
               'genuine_out_of_scope': load_inputs('old_unknowns', ('test',))['test']}
    output = {name: {**settings, 'cohorts': evaluation(settings['model'], cohorts, labels, lookup, settings['threshold'])}
              for name, settings in models.items()}
    for settings in output.values():
        settings.pop('model')
    report = {
        'scope': 'Synthetic pretraining joins separately clipped tokens. Evaluation uses authentic utterance MP4 landmarks, '
                 'presegmented. Test recordings were already evaluated in previous experiments and are diagnostic only.',
        'source': provenance, 'synthetic_pretraining_vocabulary': previous_labels[1:], 'real_output_vocabulary': labels[1:],
        'pretrain_history': pretrain_history, 'settings': {'seed': SEED, 'pretrain_epochs': args.pretrain_epochs,
            'fine_tune_epochs': args.fine_tune_epochs, 'pretrain_batch_size': 32, 'fine_tune_batch_size': 12,
            'architecture': 'Linear projection 224→64 with LayerNorm and GELU, bidirectional GRU 64, .1 dropout '
                            'then linear classifier and log-softmax. CTC blank 0, AdamW .002, gradient clip 5.',
            'real_normalization': 'Mean and std from 48 real training clips only; std floor .08.',
            'transfer': 'Copy projection and GRU parameters and matching output rows including CTC blank. Train all weights.',
            'source_features': 'Canonical aspect 848/480, absent face and absent hands encoded as zero. Select every other frame per isolated segment.',
            'selection': 'Minimum development WER, maximum exact sequences, minimum train CTC loss. Text threshold uses known development only.'},
        'real_signer_splits': {role: sorted({item.signer for item in rows}) for role, rows in roles.items()},
        'real_input_sha256': {path.as_posix(): file_hash(path) for key in ('expanded', 'old_unknowns') for path in SOURCES[key]},
        'archived_checkpoint_sha256': file_hash(CHECKPOINT), 'elapsed_seconds': time.perf_counter() - started, 'models': output,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, settings in output.items():
        print(name, {key: {metric: val[metric] for metric in ('gloss_wer', 'greek_reference_exact', 'coverage', 'accepted')}
                     for key, val in settings['cohorts'].items()}, flush=True)


if __name__ == '__main__':
    main()
