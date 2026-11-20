"""Measure the effect of more authentic contexts on a separately reserved CERTH test."""

import argparse
from collections import Counter
import json
from math import ceil, lcm
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from train_continuous import edit_counts, read_utterances, reference_lookup, replay, summarize
from train_continuous_context_transfer import BASE_FREEZE, FEATURES, ID_PREFIX, MANIFEST, read_expansion
from train_continuous_contrastive import BATCH_SIZE, SEED, OrderAwareCTC, digest, real_data, train_candidate, write


REPORT = Path('reports/continuous-hard')
ARTIFACTS = Path('artifacts/continuous-hard')
FREEZE = REPORT / 'matched-freeze.json'


def read_selected(manifest, features, roles):
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if not metadata.get('complete'):
        raise ValueError(f'Incomplete acquisition: {manifest}')
    samples = [row for row in metadata['samples'] if row['split'] in roles]
    ids = {row['id'] for row in samples}
    workspace = Path('.runtime-tmp')
    workspace.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='continuous-hard-', dir=workspace) as directory:
        selected = Path(directory) / 'manifest.json'
        features_path = Path(directory) / 'features.jsonl'
        write(selected, {'samples': samples})
        with features.open(encoding='utf-8') as source, features_path.open('w', encoding='utf-8') as target:
            for line in source:
                prefix = ID_PREFIX.match(line)
                if not prefix:
                    raise ValueError('Unexpected extraction record ID prefix')
                if json.loads(prefix[1]) in ids:
                    target.write(line)
        rows, _ = read_utterances(selected, [features_path], False, roles)
        return rows, digest(features_path)


def distance(reference, other):
    return sum(edit_counts(reference, other))


def datasets(manifest, features):
    baseline, archived, sources = real_data()
    base_freeze = json.loads(BASE_FREEZE.read_text(encoding='utf-8'))
    context, context_hash = read_expansion(base_freeze)
    baseline['train'].extend(context)
    old_hard, _ = read_selected(MANIFEST, FEATURES, ('val',))
    hard, fresh_hash = read_selected(manifest, features, ('train', 'val'))
    development = old_hard['val'] + hard['val'] + [row for row in baseline['val']
                                                 if row.id.startswith('police3_') and row.id.endswith('__sentences0006')]
    training = baseline['train'] + hard['train']
    rows = [row for group in (training, baseline['val'], old_hard['val'], hard['val'], *archived.values()) for row in group]
    reserved = [row for row in json.loads(manifest.read_text(encoding='utf-8'))['samples'] if row['split'] == 'test']
    held_texts = {row['text'].casefold().strip() for row in reserved}
    train_orders = {row.glosses for row in training}
    train_labels = {label for order in train_orders for label in order}
    if (len({row.id for row in rows}) != len(rows) or set(row.id for row in rows) & {row['id'] for row in reserved}
            or held_texts & {row.text.casefold().strip() for row in training + development}
            or {row.signer for row in training} != {'1', '2', '4', '5'} or {row.signer for row in development} != {'3'}
            or {str(row['signer']) for row in reserved} != {'6', '7'}
            or any(min(distance(row['glosses'], order) for order in train_orders) < 2
                   or not set(row['glosses']) <= train_labels for row in reserved)):
        raise ValueError('Data differ from the independent hard-template protocol')
    provenance = {'existing_manifests_and_features': {path.as_posix(): digest(path) for path in sources},
                  'context_train_features': context_hash, 'fresh_train_val_features': fresh_hash,
                  'fresh_manifest': digest(manifest), 'held_ids': sorted(row['id'] for row in reserved)}
    return baseline['train'], training, development, provenance


def fit(manifest, features):
    if FREEZE.exists():
        raise ValueError(f'Existing frozen comparison: {FREEZE}')
    base, extended, dev, provenance = datasets(manifest, features)
    labels = [None, *sorted({gloss for row in extended for gloss in row.glosses})]
    indexes = {label: index for index, label in enumerate(labels) if index}
    train_sets = {'previous_contexts': base, 'extended_contexts': extended}
    steps = {name: ceil(len(rows) / BATCH_SIZE) for name, rows in train_sets.items()}
    quantum = lcm(*steps.values())
    updates = quantum * ceil(500 / quantum)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    arms = {}
    for name, rows in train_sets.items():
        epochs = updates // steps[name]
        print(f'{name} {len(rows)} train / {len(dev)} hard dev, {epochs} epochs', flush=True)
        torch.manual_seed(SEED)
        model, history, epoch = train_candidate({'train': rows, 'val': dev}, labels, indexes, reference_lookup(rows), epochs=epochs)
        checkpoint = ARTIFACTS / f'{name}.pt'
        torch.save({'state_dict': model.state_dict(), 'labels': labels}, checkpoint)
        arms[name] = {'train_clips': len(rows), 'epochs': epochs, 'epoch': epoch, 'sha256': digest(checkpoint),
                      'development_history': history}
    write(FREEZE, {'budget_updates': updates, 'provenance': provenance, 'development_ids': [row.id for row in dev], 'arms': arms})


def score(manifest, features):
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    base, extended, _, provenance = datasets(manifest, features)
    if provenance != freeze['provenance']:
        raise ValueError('The frozen training or held split changed')
    held, feature_hash = read_selected(manifest, features, ('test',))
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    template = {row['id']: row['template'] for row in raw['samples'] if row['split'] == 'test'}
    results = {}
    for name, train in (('previous_contexts', base), ('extended_contexts', extended)):
        path = ARTIFACTS / f'{name}.pt'
        if digest(path) != freeze['arms'][name]['sha256']:
            raise ValueError(f'The frozen weights changed for {name}')
        state = torch.load(path, weights_only=True, map_location='cpu')
        weights, labels = state['state_dict'], state['labels']
        model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
        model.load_state_dict(weights)
        rows = replay(model, held['test'], labels, {})
        support = {gloss for row in train for gloss in row.glosses}
        result = summarize(rows)
        result['template_counts'] = Counter(template[row.id] for row in held['test'])
        result['per_template'] = {key: summarize([row for row in rows if template[row['id']] == key])
                                  for key in result['template_counts']}
        result['reference_tokens_with_authentic_support'] = sum(gloss in support for row in held['test'] for gloss in row.glosses)
        results[name] = {'summary': result, 'predictions': rows}
        print(json.dumps({'arm': name, **result}, ensure_ascii=False), flush=True)
    write(REPORT / 'matched-benchmark.json', {'scope': 'New held CERTH recordings. Raw sign recognition on supplied clip boundaries.',
                                             'freeze_sha256': digest(FREEZE), 'test_features_sha256': feature_hash,
                                             'results': results})


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
