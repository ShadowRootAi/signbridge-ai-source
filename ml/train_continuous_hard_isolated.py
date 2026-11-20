"""Pretrain on signer- and scenario-disjoint isolated poses before authentic CTC training."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil, lcm
from pathlib import Path
import random
import re
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from evaluate_synthetic_ctc_transfer import PLAN, POSE_CACHE, SOURCE_SIGNERS, frame_feature_array, verify_feature_math
from train_continuous import reference_lookup, replay, summarize
from train_continuous_hard import BATCH_SIZE, SEED, OrderAwareCTC, datasets, digest, read_selected, train_candidate, write


REPORT = Path('reports/continuous-hard')
CHECKPOINT = Path('artifacts/continuous-hard/isolated_transfer.pt')
FREEZE = REPORT / 'isolated-transfer-freeze.json'
METADATA = Path('research/cache/openhands-gsl/gsl_metadata.zip')
POSE_ID = re.compile(r'(?P<scenario>[a-z]+\d+)_signer(?P<signer>\d+)_rep1_glosses/glosses\d+')
PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, PRESENCE_WEIGHT = 8, 32, .35


def isolated_inputs(manifest, development, labels):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    pose_hash = digest(POSE_CACHE)
    if Path(plan['poseFile']).resolve() != POSE_CACHE.resolve() or plan['poseFileSha256'].lower() != pose_hash:
        raise ValueError('Unexpected isolated pose cache')
    with ZipFile(METADATA) as archive:
        member = 'GSL_split/GSL_isolated/train_greek_iso.csv'
        official = dict(line.split('|', 1) for line in archive.read(member).decode('utf-8-sig').splitlines())
    reserved = [row for row in json.loads(manifest.read_text(encoding='utf-8'))['samples'] if row['split'] != 'train']
    excluded = {row['template'].split('/')[0] for row in reserved}
    excluded.update(row.id.split('_')[0] for row in development)
    with np.load(POSE_CACHE, allow_pickle=False) as archive:
        poses, ids, offsets, refs = (archive[key] for key in ('keypoints', 'ids', 'offsets', 'glosses'))
        if archive['split'].item() != 'train-prototypes':
            raise ValueError('Unexpected isolated split')
    if (len(set(ids)) != len(ids) or len(ids) != len(refs) or len(offsets) != len(ids) + 1 or offsets[0] != 0
            or offsets[-1] != len(poses) or np.any(np.diff(offsets) <= 0) or not np.isfinite(poses).all()):
        raise ValueError('Malformed isolated pose cache')
    features = frame_feature_array(poses)
    verify_feature_math(poses, features)
    items, counts, signer_counts = [], Counter(), Counter()
    selected = []
    for row, (identity, gloss) in enumerate(zip(ids.tolist(), refs.tolist(), strict=True)):
        source = POSE_ID.fullmatch(identity)
        if source is None or int(source['signer']) not in SOURCE_SIGNERS or official.get(identity) != gloss:
            raise ValueError(f'Isolated source outside verified train: {identity}')
        if source['scenario'] not in excluded and gloss in labels:
            frames = torch.from_numpy(features[offsets[row]:offsets[row + 1]:2])
            items.append((frames, labels[gloss]))
            counts[gloss] += 1
            signer_counts[source['signer']] += 1
            selected.append(identity)
    if not {gloss for row in reserved for gloss in row['glosses']} <= counts.keys():
        raise ValueError('An evaluation gloss is absent from isolated pretraining')
    return items, {
        'pose_cache_sha256': pose_hash, 'isolated_metadata_sha256': digest(METADATA), 'pose_plan_sha256': digest(PLAN),
        'selected_ids_sha256': sha256('\n'.join(selected).encode()).hexdigest(), 'source_clips': len(ids),
        'clips': len(items), 'glosses': len(counts), 'signer_counts': signer_counts, 'gloss_counts': counts,
        'excluded_scenarios': sorted(excluded), 'source_frame_stride': 2,
    }


def pretrain(model, items, indexes, counts):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    rng = random.Random(SEED + 1)
    weight = torch.zeros(len(indexes) + 1)
    for gloss, count in counts.items():
        weight[indexes[gloss]] = count ** -.5
    history = []
    model.train()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        shuffled = rng.sample(items, len(items))
        losses = []
        for start in range(0, len(items), PRETRAIN_BATCH_SIZE):
            batch = shuffled[start:start + PRETRAIN_BATCH_SIZE]
            clips, classes = zip(*batch, strict=True)
            lengths = torch.tensor([len(clip) for clip in clips])
            target = torch.tensor(classes)
            scores, presence = model.outputs(pad_sequence(clips, batch_first=True), lengths)
            ctc = F.ctc_loss(scores.transpose(0, 1), target, lengths, torch.ones_like(target), reduction='none')
            loss = ((ctc + PRESENCE_WEIGHT * F.cross_entropy(presence, target - 1, reduction='none')) * weight[target]).sum() / weight[target].sum()
            if not loss.isfinite():
                raise ValueError('Non-finite isolated training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses) / len(losses))
        print(f'isolated epoch {epoch}, loss {history[-1]:.3f}', flush=True)
    return history


def fit(manifest, features):
    if FREEZE.exists():
        raise ValueError(f'Existing frozen experiment: {FREEZE}')
    base, train, dev, provenance = datasets(manifest, features)
    labels = [None, *sorted({gloss for row in train for gloss in row.glosses})]
    if (len(base), len(train), len(dev), len(labels)) != (144, 176, 6, 86):
        raise ValueError('The matched hard comparison changed')
    indexes = {gloss: index for index, gloss in enumerate(labels) if index}
    items, isolated = isolated_inputs(manifest, dev, indexes)
    print(f"isolated {isolated['clips']} clips / {isolated['glosses']} glosses", flush=True)
    torch.manual_seed(SEED)
    frames = torch.cat([row.frames for row in train])
    model = OrderAwareCTC(len(labels), frames.mean(0), frames.std(0, unbiased=False).clamp_min(.08))
    fine_tuning_rng = torch.random.get_rng_state()
    history = pretrain(model, items, indexes, isolated['gloss_counts'])
    torch.random.set_rng_state(fine_tuning_rng)
    steps = ceil(len(train) / BATCH_SIZE)
    quantum = lcm(ceil(len(base) / BATCH_SIZE), steps)
    updates = quantum * ceil(500 / quantum)
    if updates != 540:
        raise ValueError('The matched optimizer budget changed')
    model, development, epoch = train_candidate({'train': train, 'val': dev}, labels, indexes, reference_lookup(train),
                                                 initial_model=model, epochs=updates // steps)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'labels': labels}, CHECKPOINT)
    write(FREEZE, {'checkpoint_sha256': digest(CHECKPOINT), 'provenance': provenance, 'isolated': isolated,
                   'seed': SEED, 'authentic_training_clips': len(train), 'authentic_updates': updates,
                   'pretraining_epochs': PRETRAIN_EPOCHS, 'pretraining_batch_size': PRETRAIN_BATCH_SIZE,
                   'pretraining_updates': ceil(len(items) / PRETRAIN_BATCH_SIZE) * PRETRAIN_EPOCHS,
                   'pretraining_loss_history': history, 'development_ids': [row.id for row in dev],
                   'development_history': development, 'selected_epoch': epoch})


def score(manifest, features):
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    _, _, dev, provenance = datasets(manifest, features)
    if (provenance != freeze['provenance'] or freeze['development_ids'] != [row.id for row in dev]
            or digest(CHECKPOINT) != freeze['checkpoint_sha256']):
        raise ValueError('The frozen experiment changed')
    held, feature_hash = read_selected(manifest, features, ('test',))
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    template = {row['id']: row['template'] for row in raw['samples'] if row['split'] == 'test'}
    state = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
    weights, labels = state['state_dict'], state['labels']
    model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    model.load_state_dict(weights)
    rows = replay(model, held['test'], labels, {})
    result = summarize(rows)
    result['per_template'] = {key: summarize([row for row in rows if template[row['id']] == key])
                              for key in sorted(set(template.values()))}
    write(REPORT / 'isolated-transfer-benchmark.json', {
        'scope': 'Held CERTH clips with supplied boundaries. Isolated pretraining excludes held signers and scenarios.',
        'freeze_sha256': digest(FREEZE), 'test_features_sha256': feature_hash, 'summary': result, 'predictions': rows,
    })
    print(json.dumps(result, ensure_ascii=False), flush=True)


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
