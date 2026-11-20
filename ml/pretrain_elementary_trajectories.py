"""Pretrain shared Greek translation layers on verified Elementary23-SLT trajectories."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from train_greek_translation_pilot import greek, sha
from train_native_greek import (
    ARTIFACTS, FPS, MODEL_VERSION, SEED, Stream, SubtitleModel, check_fit_split, load_streams,
    loss_for, preparation, prepare_native_model, save_checkpoint, step,
)


def load_pairs(folder, existing_ids):
    manifest_path = folder / 'manifest.json'
    poses_path = folder / 'poses.npz'
    manifest = json.loads(manifest_path.read_bytes())
    pose_hash = sha(poses_path)
    if manifest['npz_sha256'] != pose_hash:
        raise ValueError('Trajectory archive differs from its verified manifest')
    with np.load(poses_path, allow_pickle=False) as archive:
        features, offsets, ids, texts, signers = (archive[key] for key in ('features', 'offsets', 'ids', 'texts', 'signers'))
    if (features.ndim != 2 or features.shape[1] != 250 or features.dtype != np.float32
            or not np.isfinite(features).all() or len(offsets) != len(ids) + 1
            or not offsets.dtype.kind == 'i' or offsets[0] != 0 or offsets[-1] != len(features)
            or np.any(np.diff(offsets) < 2) or len(texts) != len(ids) or len(signers) != len(ids)
            or any(items.dtype.kind != 'U' or items.ndim != 1 for items in (ids, texts, signers))
            or len(set(ids)) != len(ids) or len(manifest['samples']) != len(ids)):
        raise ValueError('Malformed verified Elementary trajectories')
    existing = {identity.removeprefix('elementary-').casefold() for identity in existing_ids}
    pairs, used = [], []
    for index, (identity, text, signer) in enumerate(zip(ids.tolist(), texts.tolist(), signers.tolist(), strict=True)):
        if signer in {'F', 'G'} or identity.removeprefix('elementary-').casefold() in existing:
            continue
        normalized = greek(text)
        if not normalized:
            raise ValueError(f'Empty verified Greek reference for {identity}')
        clip = features[offsets[index]:offsets[index + 1]]
        # The release does not establish a compatible frame clock. Keep source
        # order, bounding the input length for this complete-clip task.
        positions = np.linspace(0, len(clip) - 1, min(128, len(clip))).round().astype(np.int64)
        sampled = torch.from_numpy(clip[positions])
        pairs.append(Stream(identity, f'elementary23-{signer}', 'elementary23-slt', 'train',
                            sampled, FPS, ((0., len(sampled) / FPS, normalized),)))
        used.append(identity)
    if not pairs:
        raise ValueError('No independent paired trajectory rows remain')
    return pairs, {'manifest_sha256': sha(manifest_path), 'poses_sha256': pose_hash,
                   'source_pairs': len(ids), 'used_pairs': len(pairs), 'signers': sorted({row.signer for row in pairs}),
                   'excluded_existing_ids_or_previous_held_signers': len(ids) - len(pairs), 'training_ids': used}


def train(args):
    cache = torch.load(args.pairs, weights_only=True)
    existing_ids = [row['id'] for row in cache['rows']]
    pilot = json.loads(Path('data/elementary-translation-pilot/manifest.json').read_bytes())
    existing_ids.extend(row['id'] for row in pilot['samples'])
    source, evidence = load_pairs(args.source, existing_ids)
    streams, _ = load_streams(args.data, ('train', 'val'))
    fit, _ = check_fit_split(streams, include_n2=True)
    codec, gloss_ids = preparation([text for row in fit for _, _, text in row.turns],
                                   [label for row in fit for _, _, label in row.glosses], cache['rows'])
    torch.manual_seed(SEED)
    model = SubtitleModel(250, len(codec.symbols), max(1, len(gloss_ids)))
    rng = random.Random(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    print(f'Verified {len(source)} independent trajectory pairs for the source projection', flush=True)
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        order = rng.sample(source, len(source))
        losses = []
        for start in range(0, len(order), 16):
            group = order[start:start + 16]
            loss = loss_for(model, codec, group, [row.frames for row in group], gloss_ids, timed=False)
            step(optimizer, model, loss)
            losses.append(float(loss.detach()))
        print(f'Trajectory pretraining {epoch}: loss {sum(losses) / len(losses):.3f}', flush=True)
    torch.manual_seed(SEED)
    native = prepare_native_model(args, codec, gloss_ids, cache['rows'], initial_state=model.state_dict())
    save_checkpoint(args.output, {'model_version': MODEL_VERSION, 'state': native.state_dict(), 'merges': codec.merges,
                                  'gloss_ids': gloss_ids, 'pairs_sha256': sha(args.pairs), 'warmup_epochs': args.warmup_epochs,
                                  'additional_pretraining': {**evidence, 'source_epochs': args.source_epochs,
                                      'training_text': sorted({row.turns[0][2] for row in source})}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/elementary-slt-prefix'))
    parser.add_argument('--data', type=Path, default=Path('data/sartinas-dialogue-fit-expanded'))
    parser.add_argument('--pairs', type=Path, default=ARTIFACTS / 'warmup-expanded-features.pt')
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'warmup-trajectories.pt')
    parser.add_argument('--source-epochs', type=int, default=12)
    parser.add_argument('--warmup-epochs', type=int, default=30)
    args = parser.parse_args()
    if min(args.source_epochs, args.warmup_epochs) < 1:
        parser.error('Epoch counts must be positive')
    torch.set_num_threads(3)
    train(args)


if __name__ == '__main__':
    main()
