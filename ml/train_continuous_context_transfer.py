"""Add audited authentic contexts at the same real optimizer-update budget."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import torch

from evaluate_synthetic_ctc_transfer import PLAN, read_synthetic
from preprocess import FEATURE_SCHEMA
from train_continuous import read_utterances, reference_lookup
from train_continuous_contrastive import (
    ARTIFACTS, BATCH_SIZE, CONTEXT, CONTEXT_HOLDOUT, REPORTS, SEED, OrderAwareCTC, context_rows,
    digest, fit_text_policies, pretrain, real_data, score_arms, train_candidate, write,
)


BASE_FREEZE = REPORTS / 'freeze.json'
EXPANDED_FREEZE = REPORTS / 'context-expanded-freeze.json'
COMBINED_REPORT = REPORTS / 'context-matched-comparison.json'
MANIFEST = CONTEXT / 'manifest.json'
FEATURES = CONTEXT / 'features.jsonl'
ID_PREFIX = re.compile(r'^\s*\{"id":("(?:[^"\\]|\\.)*")\s*,')


def read_expansion(base):
    raw = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if not raw.get('complete') or sorted(row['id'] for row in context_rows(raw)) != base['reserved_context_ids']:
        raise ValueError('The context manifest is incomplete or its held IDs differ from the original freeze')
    samples = [item for item in raw['samples'] if item['split'] == 'train']
    expected = {row['id'] for row in samples}
    if len(expected) != 40 or {str(row['signer']) for row in samples} != {'1', '2', '4', '5'}:
        raise ValueError('Expected 40 exact new training utterances from original training signers')
    # Identify rows by the JSONL ID prefix alone, and deserialize only training features.
    workspace = Path('.runtime-tmp').resolve()
    if workspace.parent != Path.cwd().resolve():
        raise ValueError('Temporary evaluation path must stay inside the project')
    workspace.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='continuous-context-train-', dir=workspace) as directory:
        manifest, features = (Path(directory) / name for name in ('manifest.json', 'features.jsonl'))
        write(manifest, {'samples': samples})
        with FEATURES.open(encoding='utf-8') as source, features.open('w', encoding='utf-8', newline='\n') as target:
            for line in source:
                prefix = ID_PREFIX.match(line)
                if not prefix:
                    raise ValueError('Unexpected extraction record ID prefix')
                if json.loads(prefix[1]) in expected:
                    target.write(line.rstrip('\n') + '\n')
        selected, _ = read_utterances(manifest, [features], False, ('train',))
        train_features_hash = digest(features)
    return selected['train'], train_features_hash


def train():
    if EXPANDED_FREEZE.exists():
        raise ValueError('The expanded comparison is frozen already')
    base = json.loads(BASE_FREEZE.read_text(encoding='utf-8'))
    roles, archived, _ = real_data()
    new, features_hash = read_expansion(base)
    roles['train'].extend(new)
    reserved = {labels for _, labels in CONTEXT_HOLDOUT.values()}
    if len({item.id for rows in roles.values() for item in rows}) != 168:
        raise ValueError('Expanded real IDs are not disjoint')
    if any(item.glosses in reserved for rows in roles.values() for item in rows):
        raise ValueError('An expanded training order overlaps the context holdout')
    forbidden = reserved | {item.glosses for name, rows in archived.items() if 'known' not in name for item in rows}
    alignment = {}
    synthetic, provenance = read_synthetic(PLAN, excluded_templates=CONTEXT_HOLDOUT,
                                           excluded_sequences=forbidden, alignments=alignment)
    if provenance['orderedRecipeSha256'] != base['synthetic']['orderedRecipeSha256']:
        raise ValueError('Expanded arms require identical synthetic recipes')
    real_glosses = {token for item in roles['train'] for token in item.glosses}
    synthetic_glosses = {token for item in synthetic for token in item.glosses}
    labels = [None, *sorted(real_glosses | synthetic_glosses)]
    if set(labels[1:]) != set(base['real_glosses']) | set(base['synthetic_only_glosses']):
        raise ValueError('Expanded arms must have exactly the base output vocabulary')
    indices = {token: i for i, token in enumerate(labels) if i}
    lookup = reference_lookup(roles['train'])
    duration = defaultdict(int)
    for row in roles['train']:
        duration[row.glosses] = max(duration[row.glosses], len(row.frames))
    frames = torch.cat([item.frames for item in roles['train']])
    center, scale = frames.mean(0), frames.std(0, unbiased=False).clamp_min(.08)
    target_steps = base['settings']['finetuning_epochs'] * ((104 + BATCH_SIZE - 1) // BATCH_SIZE)
    steps_per_epoch = len(roles['train']) // BATCH_SIZE
    epochs, remainder = divmod(target_steps, steps_per_epoch)
    if remainder or len(roles['train']) % BATCH_SIZE:
        raise ValueError('The expanded authentic arm cannot match the exact update budget')
    arms = {}
    for name, contrast in (('expanded_authentic_only', False), ('expanded_authentic_contrastive', True)):
        print(name, epochs, 'epochs', target_steps, 'real updates', flush=True)
        torch.manual_seed(SEED)
        model = OrderAwareCTC(len(labels), center, scale)
        warmup = []
        if contrast:
            unused = [indices[label] for label in sorted(real_glosses - synthetic_glosses)]
            initial_weight = model.ctc.weight[unused].detach().clone()
            initial_bias = model.ctc.bias[unused].detach().clone()
            warmup = pretrain(model, synthetic, alignment, indices, base['settings']['pretraining_epochs'], contrast=True)
            with torch.no_grad():
                model.ctc.weight[unused] = initial_weight
                model.ctc.bias[unused] = initial_bias
        torch.manual_seed(SEED + 2)
        model, history, epoch = train_candidate(roles, labels, indices, lookup, initial_model=model, epochs=epochs,
                                                rehearsal=synthetic if contrast else ())
        confidence, guard, development = fit_text_policies(model, roles['val'], labels, lookup, duration)
        target = ARTIFACTS / f'{name}.pt'
        torch.save({'state_dict': model.state_dict(), 'labels': labels, 'feature_schema': FEATURE_SCHEMA,
                    'training_greek_references': lookup, 'train_max_frames': dict(duration),
                    'confidence': confidence, 'joint_guard': guard}, target)
        arms[name] = {'artifact': target.as_posix(), 'sha256': digest(target), 'selected_finetune_epoch': epoch,
                      'pretraining_history': warmup, 'finetuning_history': history, 'development': development}
    all_arms = {**base['arms'], **arms}

    def choose(name):
        development = all_arms[name]['development']
        return (development['validation_raw']['gloss_wer'], -development['validation_raw']['gloss_sequence_exact'],
                -development['validation_guard']['correct_text'], name)

    ranked = sorted(all_arms, key=choose)
    freeze = {'scope': 'All five candidates and dev ordering fixed before reading context held features.',
              'base_freeze_sha256': digest(BASE_FREEZE), 'implementation_sha256': digest(Path(__file__)),
              'context_manifest_sha256': digest(MANIFEST), 'context_train_features_sha256': features_hash,
              'context_train_ids': [item.id for item in new], 'real_counts': {'train': 144, 'val': 24},
              'nominal_real_minibatch_update_budget': target_steps,
              'expanded_finetune_epochs': epochs, 'pretraining_recipe_sha256': provenance['orderedRecipeSha256'],
              'real_glosses': sorted(real_glosses), 'synthetic_only_glosses': sorted(synthetic_glosses - real_glosses),
              'ranked_on_existing_dev_only': ranked, 'arms': arms}
    write(EXPANDED_FREEZE, freeze)
    print('FROZEN all', ranked, digest(EXPANDED_FREEZE), flush=True)


def evaluate():
    base, extra = (json.loads(path.read_text(encoding='utf-8')) for path in (BASE_FREEZE, EXPANDED_FREEZE))
    if digest(BASE_FREEZE) != extra['base_freeze_sha256'] or digest(MANIFEST) != extra['context_manifest_sha256']:
        raise ValueError('Frozen base/held data metadata changed')
    _, selected_train_hash = read_expansion(base)
    if selected_train_hash != extra['context_train_features_sha256']:
        raise ValueError('Frozen authentic context training features changed')
    for path, expected in base['real_input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Original experiment input changed: {path}')
    held, _ = read_utterances(MANIFEST, [FEATURES], False)
    if sorted(item.id for role in ('val', 'test') for item in held[role]) != base['reserved_context_ids']:
        raise ValueError('The held input IDs changed')
    _, archived, _ = real_data()
    cohorts = {'context_development_diagnostic': held['val'], 'context_novel': held['test'], **archived}
    combined = {'arms': {**base['arms'], **extra['arms']}}
    result = score_arms(combined, cohorts)
    lexical = {}
    for name in combined['arms']:
        source = extra if name in extra['arms'] else base
        real, synthetic = (set(source[key]) for key in ('real_glosses', 'synthetic_only_glosses'))
        lexical[name] = {group: Counter('authentic_train' if token in real else 'synthetic_only' if token in synthetic else 'unseen'
                                        for item in rows for token in item.glosses)
                         for group, rows in cohorts.items() if group.startswith('context')}
    report = {'scope': 'All arms locked and ranked on existing development before new held feature inference. '
                       'The context test contains new orders from known corpus signers and pre-cut utterances. '
                       'Archived results are diagnostics. No webcam, spontaneous conversation, or runtime claim.',
              'base_freeze_sha256': digest(BASE_FREEZE), 'expanded_freeze_sha256': digest(EXPANDED_FREEZE),
              'context_features_sha256': digest(FEATURES), 'ranked_on_existing_dev_only': extra['ranked_on_existing_dev_only'],
              'lexical_support': lexical, 'arms': result}
    write(COMBINED_REPORT, report)
    for name, groups in result.items():
        test = groups['context_novel']
        print(name, 'context WER', test['raw']['gloss_wer'], 'exact', test['raw']['gloss_sequence_exact'],
              'out-of-scope text', test['joint_guard']['wrong_or_out_of_scope_text'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'train':
        train()
    else:
        evaluate()


if __name__ == '__main__':
    main()
