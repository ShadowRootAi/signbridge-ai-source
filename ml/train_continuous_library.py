"""Train matched continuous GSL recognizers with and without clean isolated library transfer."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
import torch

from evaluate_synthetic_ctc_transfer import frame_feature_array, verify_feature_math
from openhands_gsl_stgcn import METADATA_ARCHIVE_SHA256
from train_continuous import Utterance, reference_lookup, replay, summarize
from train_continuous_hard import BATCH_SIZE, SEED, OrderAwareCTC, datasets as hard_datasets, digest, read_selected, train_candidate, write
from train_continuous_hard_isolated import METADATA, pretrain as train_isolated
from train_isolated_large_vocabulary import EXPANDED_POSES, REPORT as ISOLATED_REPORT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plan_continuous_composition import normalize_greek, read_annotations


DATA = Path('data')
FRESH = DATA / 'appointment-unseen'
REPORT = Path('reports/continuous-library-ctc')
ARTIFACTS = Path('artifacts/continuous-library-ctc')
FREEZE = REPORT / 'freeze.json'
PRETRAIN_FREEZE = REPORT / 'pretraining-freeze.json'
PRETRAIN_ARTIFACT = ARTIFACTS / 'isolated_pretraining.pt'
RESERVED_SCENES = {'health1', 'health3'}
RESERVED_SIGNERS = {'3', '6', '7'}
ARM_NAMES = ('continuous_only', 'isolated_transfer')
SOURCE_ISOLATE = re.compile(r'(?P<scene>[a-z]+\d+)_signer(?P<signer>[125])_rep[123]_glosses/glosses\d+')


def annotations():
    folder = DATA / 'original_annotations'
    return read_annotations(folder / 'GSL_split.zip', folder / 'supplementary.zip')


def eligibility():
    canonical, references = annotations()
    held_sources = {source for source in canonical if source.split('_', 1)[0] in RESERVED_SCENES}
    return ({canonical[source] for source in held_sources},
            {normalize_greek(references[source]) for source in held_sources if source in references})


def filter_training(rows):
    held_glosses, held_texts = eligibility()
    kept, excluded = {}, Counter()
    for row in rows:
        if row.signer in RESERVED_SIGNERS or row.source.split('/', 1)[0] in RESERVED_SCENES:
            excluded['scene_or_signer'] += 1
        elif row.glosses in held_glosses or normalize_greek(row.text) in held_texts:
            excluded['exact_reference'] += 1
        elif row.source in kept:
            previous = kept[row.source]
            if (row.id, row.glosses, row.text) != (previous.id, previous.glosses, previous.text):
                raise ValueError(f'Conflicting repeated source: {row.source}')
            excluded['duplicate_source'] += 1
        else:
            kept[row.source] = row
    return list(kept.values()), dict(excluded)


def historical_training():
    hard = DATA / 'continuous-hard-split'
    _, rows, _, previous = hard_datasets(hard / 'manifest.json', hard / 'features.jsonl')
    sources = {'historical': previous}
    for folder in ('appointment-dialogues', 'appointment-dialogues-support'):
        manifest, features = DATA / folder / 'manifest.json', DATA / folder / 'features.jsonl'
        selected, selected_sha = read_selected(manifest, features, ('train',))
        rows.extend(selected['train'])
        sources[folder] = {'manifest_sha256': digest(manifest), 'train_features_sha256': selected_sha}
    train, excluded = filter_training(rows)
    return train, {**sources, 'eligible_historical_ids': sorted(row.id for row in train),
                   'excluded_historical': excluded}


def read_fresh_manifest(path):
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if not manifest.get('complete'):
        raise ValueError('Fresh dialogue acquisition is incomplete')
    canonical, references = annotations()
    held, development = [], []
    for row in manifest['samples']:
        scene, identity = row['source_path'].split('/', 1)
        if (not identity.startswith(f"{scene}_signer{row['signer']}_") or canonical.get(identity) != tuple(row['glosses'])
                or normalize_greek(references.get(identity, '')) != normalize_greek(row['text'])):
            raise ValueError(f"Invalid official pairing: {row['id']}")
        if row['split'] == 'test':
            if scene != 'health1' or str(row['signer']) not in {'6', '7'}:
                raise ValueError('Unexpected fresh test scene or signer')
            held.append(row)
        elif row['split'] == 'val':
            if scene != 'health3' or str(row['signer']) != '3':
                raise ValueError('Unexpected fresh development scene or signer')
            development.append(row)
        elif row['split'] != 'train' or scene in RESERVED_SCENES or str(row['signer']) in RESERVED_SIGNERS:
            raise ValueError('An invalid source crosses into the fresh training cohort')
    if len(held) != 28 or len(development) != 7 or Counter(str(row['signer']) for row in held) != {'6': 14, '7': 14}:
        raise ValueError('Unexpected fresh dialogue count')
    return manifest, held


def datasets(manifest, features):
    metadata, held = read_fresh_manifest(manifest)
    historical, previous = historical_training()
    roles = ('train', 'val') if any(row['split'] == 'train' for row in metadata['samples']) else ('val',)
    fresh, fresh_feature_hash = read_selected(manifest, features, roles)
    fresh_train = fresh.get('train', [])
    train, excluded = filter_training(historical + fresh_train)
    development = fresh['val']
    ids = [row.id for row in train + development] + [row['id'] for row in held]
    if len(set(ids)) != len(ids) or any(row.fps != 15 for row in train + development):
        raise ValueError('Continuous feature cadence or split identities changed')
    accepted = {row.id for row in train}
    provenance = {**previous, 'fresh_manifest_sha256': digest(manifest), 'fresh_train_dev_features_sha256': fresh_feature_hash,
                  'train_ids': sorted(row.id for row in train), 'development_ids': sorted(row.id for row in development),
                  'held_ids': sorted(row['id'] for row in held), 'fresh_train_included': sum(row.id in accepted for row in fresh_train),
                  'fresh_train_candidates': len(fresh_train), 'fresh_train_excluded': excluded,
                  'reserved_scenes': sorted(RESERVED_SCENES), 'reserved_signers': sorted(RESERVED_SIGNERS)}
    return train, development, provenance


def isolated_rows():
    freeze = json.loads((ISOLATED_REPORT / 'expanded-freeze.json').read_text(encoding='utf-8'))
    pose_hash = digest(EXPANDED_POSES)
    if pose_hash != freeze['pose_cache_sha256'] or digest(METADATA) != METADATA_ARCHIVE_SHA256:
        raise ValueError('Verified isolated library assets changed')
    with ZipFile(METADATA) as archive:
        official = dict(line.split('|', 1) for line in archive.read('GSL_split/GSL_isolated/train_greek_iso.csv').decode('utf-8-sig').splitlines())
    with np.load(EXPANDED_POSES, allow_pickle=False) as archive:
        raw, offsets, ids, glosses = (archive[name] for name in ('keypoints', 'offsets', 'ids', 'glosses'))
        if archive['split'].item() != 'train-expanded':
            raise ValueError('Unexpected isolated library role')
    if (len(ids) != 10415 or len(set(ids)) != len(ids) or len(glosses) != len(ids) or len(offsets) != len(ids) + 1
            or offsets[0] != 0 or offsets[-1] != len(raw) or np.any(np.diff(offsets) < 1) or not np.isfinite(raw).all()):
        raise ValueError('Invalid expanded isolated features')
    converted = frame_feature_array(raw)
    verify_feature_math(raw, converted)
    rows = []
    for identity, gloss, start, end in zip(ids.tolist(), glosses.tolist(), offsets[:-1], offsets[1:], strict=True):
        source = SOURCE_ISOLATE.fullmatch(identity)
        if source is None or official.get(identity) != gloss:
            raise ValueError(f'Unexpected isolated library member: {identity}')
        if source['scene'] in RESERVED_SCENES:
            continue
        frames = torch.from_numpy(converted[start:end:2])
        rows.append(Utterance(identity, identity, source['signer'], 'train', (gloss,), '', frames, 15., 1., None))
    counts = Counter(row.glosses[0] for row in rows)
    record = {'source_cache_sha256': pose_hash, 'metadata_sha256': METADATA_ARCHIVE_SHA256, 'source_clips': len(ids),
              'clips': len(rows), 'glosses': len(counts), 'signer_counts': dict(Counter(row.signer for row in rows)),
              'selected_ids_sha256': sha256('\n'.join(row.id for row in rows).encode()).hexdigest(),
              'excluded_scenes': sorted(RESERVED_SCENES), 'counts': dict(counts)}
    return rows, counts, record


def pretrain():
    if PRETRAIN_FREEZE.exists():
        raise ValueError(f'Existing frozen library initialization: {PRETRAIN_FREEZE}')
    historical, previous = historical_training()
    isolated, counts, source = isolated_rows()
    labels = [None, *sorted(counts)]
    indexes = {label: index for index, label in enumerate(labels) if index}
    frames = torch.cat([row.frames for row in historical])
    torch.manual_seed(SEED)
    model = OrderAwareCTC(len(labels), frames.mean(0), frames.std(0, unbiased=False).clamp_min(.08))
    items = [(row.frames, indexes[row.glosses[0]]) for row in isolated]
    print(f'Library pretrain: {len(items)} isolated clips / {len(counts)} labels', flush=True)
    history = train_isolated(model, items, indexes, counts)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'labels': labels}, PRETRAIN_ARTIFACT)
    write(PRETRAIN_FREEZE, {'checkpoint_sha256': digest(PRETRAIN_ARTIFACT), 'source': source,
                            'normalization': previous, 'history': history})


def transfer_state(model, labels, pretrained):
    state, old = model.state_dict(), pretrained['state_dict']
    old_index = {label: index for index, label in enumerate(pretrained['labels'])}
    for name in state.keys() - {'ctc.weight', 'ctc.bias', 'gloss_presence.weight', 'gloss_presence.bias'}:
        state[name] = old[name]
    for index, label in enumerate(labels):
        if label not in old_index:
            continue
        previous = old_index[label]
        for kind in ('weight', 'bias'):
            state[f'ctc.{kind}'][index].copy_(old[f'ctc.{kind}'][previous])
            if index:
                state[f'gloss_presence.{kind}'][index - 1].copy_(old[f'gloss_presence.{kind}'][previous - 1])
    model.load_state_dict(state)


def fit(manifest, features):
    if FREEZE.exists():
        raise ValueError(f'Existing frozen comparison: {FREEZE}')
    train, dev, provenance = datasets(manifest, features)
    initialization = json.loads(PRETRAIN_FREEZE.read_text(encoding='utf-8'))
    if digest(PRETRAIN_ARTIFACT) != initialization['checkpoint_sha256']:
        raise ValueError('The frozen library initialization changed')
    historical, previous = historical_training()
    isolated, counts, isolated_record = isolated_rows()
    if previous != initialization['normalization'] or isolated_record != initialization['source']:
        raise ValueError('The library pretraining inputs changed before fine-tuning')
    pretrained = torch.load(PRETRAIN_ARTIFACT, weights_only=True, map_location='cpu')
    authentic = {gloss for row in train for gloss in row.glosses}
    labels = [None, *sorted(authentic | counts.keys())]
    indexes = {label: index for index, label in enumerate(labels) if index}
    steps = ceil(len(train) / BATCH_SIZE)
    epochs = ceil(540 / steps)
    frames = torch.cat([row.frames for row in historical])
    center, scale = frames.mean(0), frames.std(0, unbiased=False).clamp_min(.08)
    if not torch.equal(center, pretrained['state_dict']['mean']) or not torch.equal(scale, pretrained['state_dict']['scale']):
        raise ValueError('Isolated initialization and authentic training normalization differ')
    arms = {}
    for arm in ARM_NAMES:
        torch.manual_seed(SEED)
        model = OrderAwareCTC(len(labels), center, scale)
        rehearsal = ()
        if arm == 'isolated_transfer':
            transfer_state(model, labels, pretrained)
            rehearsal = isolated
        print(f'{arm}: {len(train)} train, {len(dev)} dev, {len(labels)-1} labels, {steps*epochs} authentic updates', flush=True)
        model, history, epoch = train_candidate({'train': train, 'val': dev}, labels, indexes, reference_lookup(train),
                                                 initial_model=model, epochs=epochs, rehearsal=rehearsal)
        path = ARTIFACTS / f'{arm}.pt'
        torch.save({'state_dict': model.state_dict(), 'labels': labels}, path)
        arms[arm] = {'checkpoint_sha256': digest(path), 'selected_epoch': epoch, 'development_history': history}
    write(FREEZE, {'provenance': provenance, 'arms': arms, 'authentic_updates': steps * epochs, 'epochs': epochs,
                   'authentic_glosses': sorted(authentic), 'library_only_glosses': sorted(counts.keys() - authentic),
                   'pretraining_freeze_sha256': digest(PRETRAIN_FREEZE), 'rehearsal': 'one isolated CTC batch per fourth authentic update'})


def load_checkpoint(arm):
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    path = ARTIFACTS / f'{arm}.pt'
    if digest(path) != freeze['arms'][arm]['checkpoint_sha256']:
        raise ValueError(f'The frozen weights changed for {arm}')
    state = torch.load(path, weights_only=True, map_location='cpu')
    weights, labels = state['state_dict'], state['labels']
    model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    model.load_state_dict(weights)
    return model, labels, freeze


def exact_aligned_glosses(reference, prediction):
    previous = [(column, ()) for column in range(len(prediction) + 1)]
    for row, expected in enumerate(reference, 1):
        current = [(row, ())]
        for column, actual in enumerate(prediction, 1):
            diagonal, deletion, insertion = previous[column - 1], previous[column], current[column - 1]
            substitution = (diagonal[0] + 1, diagonal[1]) if expected != actual else (diagonal[0], (*diagonal[1], expected))
            candidates = (substitution, (deletion[0] + 1, deletion[1]), (insertion[0] + 1, insertion[1]))
            current.append(min(candidates, key=lambda item: (item[0], -len(item[1]))))
        previous = current
    return previous[-1][1]


def score(manifest, features):
    _, _, provenance = datasets(manifest, features)
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if provenance != freeze['provenance']:
        raise ValueError('Fresh split provenance changed')
    held, feature_hash = read_selected(manifest, features, ('test',))
    authentic, library = set(freeze['authentic_glosses']), set(freeze['library_only_glosses'])

    def support(gloss):
        return 'authentic' if gloss in authentic else 'library_only' if gloss in library else 'unseen'

    results = {}
    for arm in ARM_NAMES:
        model, labels, _ = load_checkpoint(arm)
        predictions = replay(model, held['test'], labels, {})
        result = summarize(predictions)
        result['reference_gloss_support'] = dict(Counter(support(gloss) for row in held['test'] for gloss in row.glosses))
        result['exact_aligned_matches_by_support'] = dict(Counter(
            support(gloss) for row in predictions
            for gloss in exact_aligned_glosses(row['reference_glosses'], row['predicted_glosses'])
        ))
        results[arm] = {'summary': result, 'predictions': predictions}
        print(json.dumps({'arm': arm, **result}, ensure_ascii=False), flush=True)
    write(REPORT / 'benchmark.json', {'scope': 'Fresh actors and held scenes, using supplied utterance boundaries.',
                                      'freeze_sha256': digest(FREEZE), 'test_features_sha256': feature_hash, 'arms': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pretrain', 'train', 'score'))
    parser.add_argument('--manifest', type=Path, default=FRESH / 'manifest.json')
    parser.add_argument('--features', type=Path, default=FRESH / 'features.jsonl')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'pretrain':
        pretrain()
    else:
        (fit if args.phase == 'train' else score)(args.manifest, args.features)


if __name__ == '__main__':
    main()
