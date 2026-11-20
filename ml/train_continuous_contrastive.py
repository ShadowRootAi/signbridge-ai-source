"""Freeze matched-budget Greek continuous-recognition transfer experiments."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import random
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from continuous_selective import PERTURBATIONS, guard_rows, summarize_guard
from evaluate_continuous_joint_guard import as_base_guard, fit, tag_candidates
from evaluate_continuous_transfer import SOURCES, load_inputs
from evaluate_synthetic_ctc_transfer import PLAN, read_synthetic
from preprocess import FEATURE_SCHEMA
from train_continuous import choose_threshold, read_utterances, reference_lookup, replay, summarize
from train_continuous_multitask import BATCH_SIZE, SEED, WIDTH, OrderAwareCTC, train_candidate


COMPACT = Path('data/continuous-fresh-compact')
CONTEXT = Path('data/continuous-context')
ARTIFACTS = Path('artifacts/continuous-contrastive')
REPORTS = Path('reports/continuous-contrastive')
PRETRAIN_BATCH_SIZE = 32
TOKEN_WEIGHT = .35
CONTRAST_WEIGHT = .15
TEMPERATURE = .15
ARM_NAMES = ('real_only', 'synthetic_ctc', 'synthetic_boundary_contrastive')
CONTEXT_HOLDOUT = {
    'health4/sentences0016': ('val', ('ΣΥΝ', 'ΤΑΥΤΟΤΗΤΑ', 'ΦΩΤΟΤΥΠΙΑ', 'ΕΣΥ_ΔΙΝΩ_ΕΜΕΝΑ')),
    'kep4/sentences0003': ('test', ('ΕΣΥ', 'ΕΥΘΥΝΗ', 'ΥΠΟΓΡΑΦΗ', 'ΣΥΜΠΛΗΡΩΝΩ')),
    'health1/sentences0009': ('test', ('ΠΡΕΠΕΙ', 'ΕΣΥ', 'ΒΙΒΛΙΟ', 'ΥΓΕΙΑ', 'ΕΧΩ')),
}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def context_rows(metadata):
    rows = [item for item in metadata['samples'] if item['split'] in ('val', 'test')]
    if ({item['template'] for item in rows} != set(CONTEXT_HOLDOUT)
            or any(CONTEXT_HOLDOUT[item['template']] != (item['split'], tuple(item['glosses'])) for item in rows)):
        raise ValueError('The context utterances differ from the reserved templates')
    if {str(row['signer']) for row in rows if row['split'] == 'test'} != {'6', '7'}:
        raise ValueError('Fresh context test must contain both reserved test signers')
    if {str(row['signer']) for row in rows if row['split'] == 'val'} != {'3'}:
        raise ValueError('Fresh context development recordings must only contain signer 3')
    if Counter(row['split'] for row in rows) != {'val': 2, 'test': 8}:
        raise ValueError('Expected 2 predeclared context development and 8 test recordings')
    return rows


def real_data():
    sources = [p for name in ('expanded', 'composition', 'old_unknowns') for p in SOURCES[name]]
    sources.extend([COMPACT / 'manifest.json', COMPACT / 'features.jsonl'])
    expanded, composition = (load_inputs(name) for name in ('expanded', 'composition'))
    compact, _ = read_utterances(sources[-2], [sources[-1]], False)
    roles = {role: expanded[role] + composition[role] + compact[role] for role in ('train', 'val')}
    if len(roles['train']) != 104 or len(roles['val']) != 24:
        raise ValueError('Expected the frozen 104 real train / 24 real development cohort')
    signer_sets = {role: {item.signer for item in rows} for role, rows in roles.items()}
    if signer_sets != {'train': {'1', '2', '4', '5'}, 'val': {'3'}}:
        raise ValueError('Unexpected real signer roles')
    known = {item.glosses for item in roles['train']}
    unknown = load_inputs('old_unknowns', ('test',))['test']
    held = {
        'archived_known': [item for item in expanded['test'] + composition['test'] if item.glosses in known],
        'archived_novel': [item for item in composition['test'] if item.glosses not in known],
        'compact_known': [item for item in compact['test'] if item.glosses in known],
        'compact_novel': [item for item in compact['test'] if item.glosses not in known],
        'archived_outside_scope': unknown,
    }
    ids = [item.id for rows in (*roles.values(), *held.values()) for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('A real recording crosses role boundaries')
    return roles, held, sources


def contrastive_loss(embedding, labels, signers, source_segments):
    identity = source_segments[:, None] == source_segments[None, :]
    same_label = labels[:, None] == labels[None, :]
    cross_signer = signers[:, None] != signers[None, :]
    positives = same_label & cross_signer & ~identity
    comparable = (~same_label | cross_signer) & ~identity
    valid = positives.any(dim=1)
    if not valid.any():
        return embedding.sum() * 0., 0
    z = F.normalize(embedding, dim=-1)
    logits = (z[valid] @ z.T) / TEMPERATURE
    denom = torch.logsumexp(logits.masked_fill(~comparable[valid], -torch.inf), dim=1)
    positive_logits = logits.masked_fill(~positives[valid], 0.).sum(dim=1) / positives[valid].sum(dim=1)
    return (denom - positive_logits).mean(), int(valid.sum())


def pretrain(model, synthetic, alignments, indices, epochs, *, contrast):
    token_head = nn.Linear(WIDTH, len(indices))
    contrast_head = nn.Linear(WIDTH, 48)
    parameters = list(model.parameters()) + list(token_head.parameters()) + list(contrast_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.002)
    generator = random.Random(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = defaultdict(list)
        shuffled = generator.sample(synthetic, len(synthetic))
        for start in range(0, len(shuffled), PRETRAIN_BATCH_SIZE):
            batch = shuffled[start:start + PRETRAIN_BATCH_SIZE]
            lengths = torch.tensor([len(item.frames) for item in batch])
            targets = [torch.tensor([indices[label] for label in item.glosses], dtype=torch.long) for item in batch]
            target_lengths = torch.tensor([len(target) for target in targets])
            optimizer.zero_grad(set_to_none=True)
            local, temporal, _ = model.encode(pad_sequence([item.frames for item in batch], batch_first=True), lengths)
            scores = model.ctc(model.dropout(temporal)).log_softmax(dim=-1).transpose(0, 1)
            ctc = F.ctc_loss(scores, torch.cat(targets), lengths, target_lengths)
            loss = ctc
            if contrast:
                segments, source_ids, signers = [], [], []
                for row, item in enumerate(batch):
                    for left, right, source_id in alignments[item.id]:
                        # Only interior frames, excluding synthetic cut points where possible.
                        inset = min(2, (right - left - 1) // 3)
                        segments.append(local[row, left + inset:right - inset].mean(dim=0))
                        source_ids.append(source_id)
                        signers.append(int(item.signer))
                token_vectors = torch.stack(segments)
                token_ids = torch.cat(targets) - 1
                classifier = F.cross_entropy(token_head(token_vectors), token_ids)
                visual, anchor_count = contrastive_loss(contrast_head(token_vectors), token_ids,
                                                       torch.tensor(signers), torch.tensor(source_ids))
                loss = loss + TOKEN_WEIGHT * classifier + CONTRAST_WEIGHT * visual
                totals['segment_ce'].append(float(classifier.detach()))
                totals['contrast'].append(float(visual.detach()))
                totals['cross_signer_anchors'].append(anchor_count)
            if not bool(loss.isfinite()):
                raise ValueError('Non-finite synthetic training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            totals['ctc'].append(float(ctc.detach()))
        row = {'epoch': epoch, **{name: mean(values) for name, values in totals.items()}}
        history.append(row)
        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    return history


def fit_text_policies(model, validation, labels, lookup, durations):
    guesses = replay(model, validation, labels, lookup)
    for row in guesses:
        if row['reference_glosses'] not in lookup:
            row['greek_reference_exact'] = False
    confidence = choose_threshold(guesses)
    original = guard_rows(model, None, validation, labels, lookup, 0.)
    transformations = [row for mode in PERTURBATIONS
                       for row in guard_rows(model, None, validation, labels, lookup, 0., mode, False)]
    tag_candidates([*original, *transformations], durations)
    _, guard = fit([*original, *transformations], agreement=False, duration=True)
    return confidence, guard, {
        'validation_raw': summarize(guesses),
        'validation_guard': summarize_guard(as_base_guard(original, guard), guard, True),
        'validation_transformed_guard': summarize_guard(as_base_guard(transformations, guard), guard, False),
    }


def train(pretrain_epochs, finetune_epochs):
    freeze_path = REPORTS / 'freeze.json'
    if freeze_path.exists():
        raise ValueError('A frozen experiment exists. Do not overwrite it after any held data is observed.')
    roles, archived, sources = real_data()
    context_plan = CONTEXT / 'manifest.planned.json'
    planned_context = context_rows(json.loads(context_plan.read_text(encoding='utf-8')))
    if {row['id'] for row in planned_context} & {item.id for items in (*roles.values(), *archived.values()) for item in items}:
        raise ValueError('Fresh held recording ID overlaps a previous real cohort')
    reserved = {labels for _, labels in CONTEXT_HOLDOUT.values()}
    forbidden = reserved | {item.glosses for name, items in archived.items() if 'known' not in name for item in items}
    if any(item.glosses in reserved for items in roles.values() for item in items):
        raise ValueError('Fresh reserved sequences already occur in real development')
    alignment = {}
    synthetic, provenance = read_synthetic(PLAN, excluded_templates=CONTEXT_HOLDOUT,
                                           excluded_sequences=forbidden, alignments=alignment)
    if {item.glosses for item in synthetic} & reserved:
        raise ValueError('Fresh reserved order found in synthetic training')
    train_glosses = {token for item in roles['train'] for token in item.glosses}
    synthetic_glosses = {token for item in synthetic for token in item.glosses}
    labels = [None, *sorted(train_glosses | synthetic_glosses)]
    indices = {token: i for i, token in enumerate(labels) if i}
    lookup = reference_lookup(roles['train'])
    durations = defaultdict(int)
    for row in roles['train']:
        durations[row.glosses] = max(durations[row.glosses], len(row.frames))
    frames = torch.cat([item.frames for item in roles['train']])
    center, scale = frames.mean(0), frames.std(0, unbiased=False).clamp_min(.08)
    outcomes = {}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print('Frozen plan uses', len(synthetic), 'synthetic clips and', len(indices), 'union labels', flush=True)
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = OrderAwareCTC(len(labels), center, scale)
        warmup = []
        print(name, flush=True)
        if name != 'real_only':
            unused = [indices[label] for label in sorted(train_glosses - synthetic_glosses)]
            initial_weight = model.ctc.weight[unused].detach().clone()
            initial_bias = model.ctc.bias[unused].detach().clone()
            warmup = pretrain(model, synthetic, alignment, indices, pretrain_epochs,
                              contrast=name == 'synthetic_boundary_contrastive')
            with torch.no_grad():
                model.ctc.weight[unused] = initial_weight
                model.ctc.bias[unused] = initial_bias
        torch.manual_seed(SEED + 2)
        model, history, epoch = train_candidate(roles, labels, indices, lookup, initial_model=model,
                                                epochs=finetune_epochs, rehearsal=synthetic if warmup else ())
        confidence, guard, development = fit_text_policies(model, roles['val'], labels, lookup, durations)
        path = ARTIFACTS / f'{name}.pt'
        torch.save({'state_dict': model.state_dict(), 'labels': labels, 'feature_schema': FEATURE_SCHEMA,
                    'training_greek_references': lookup, 'train_max_frames': dict(durations),
                    'confidence': confidence, 'joint_guard': guard}, path)
        outcomes[name] = {'artifact': path.as_posix(), 'sha256': digest(path), 'selected_finetune_epoch': epoch,
                          'pretraining_history': warmup, 'finetuning_history': history, 'development': development}
    freeze = {
        'scope': 'Training and development only. All arm weights and text policies are fixed before reading context test features.',
        'real_counts': {role: len(items) for role, items in roles.items()},
        'real_input_sha256': {path.as_posix(): digest(path) for path in sources},
        'implementation_sha256': {path.as_posix(): digest(path) for path in map(Path, (
            'ml/train_continuous_contrastive.py', 'ml/train_continuous_multitask.py', 'ml/evaluate_synthetic_ctc_transfer.py',
            'ml/continuous_selective.py', 'ml/evaluate_continuous_joint_guard.py', 'ml/train_continuous.py'))},
        'synthetic': provenance, 'reserved_context_templates': CONTEXT_HOLDOUT,
        'context_plan_sha256': digest(context_plan), 'reserved_context_ids': sorted(row['id'] for row in planned_context),
        'real_glosses': sorted(train_glosses), 'synthetic_only_glosses': sorted(synthetic_glosses - train_glosses),
        'settings': {'pretraining_epochs': pretrain_epochs, 'finetuning_epochs': finetune_epochs, 'seed': SEED,
            'pretraining_batch_size': PRETRAIN_BATCH_SIZE, 'finetuning_batch_size': BATCH_SIZE,
            'segment_ce_weight': TOKEN_WEIGHT, 'cross_signer_contrast_weight': CONTRAST_WEIGHT,
            'temperature': TEMPERATURE, 'rehearsal': '.20 synthetic CTC loss every fourth real minibatch for both transfer arms',
            'normalization': '104 authentic train clips only, all arms',
            'finetune_checkpoint_choice': 'Min dev WER, max dev exact sequences, min training CTC loss.'},
        'arms': outcomes,
    }
    write(freeze_path, freeze)
    print(f'FROZEN {freeze_path}: {digest(freeze_path)}', flush=True)


def score_arms(freeze, cohorts):
    result = {}
    for name, item in freeze['arms'].items():
        path = Path(item['artifact'])
        if digest(path) != item['sha256']:
            raise ValueError(f'Frozen model changed: {name}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        labels, lookup, state = (checkpoint[k] for k in ('labels', 'training_greek_references', 'state_dict'))
        model = OrderAwareCTC(len(labels), state['mean'], state['scale'])
        model.load_state_dict(state)
        details = {}
        for group, examples in cohorts.items():
            known = group.endswith('_known')
            rows = guard_rows(model, None, examples, labels, lookup, 0., genuine=known)
            tag_candidates(rows, checkpoint['train_max_frames'])
            gate = checkpoint['joint_guard']
            conf = replay(model, examples, labels, lookup, checkpoint['confidence'])
            details[group] = {
                'raw': {**summarize(rows), 'lookup_emissions_without_threshold': sum(row['eligible_text'] for row in rows)},
                'confidence_only': {**summarize(conf),
                    'false_out_of_scope_emissions': None if known else sum(row['predicted_text'] is not None for row in conf)},
                'joint_guard': summarize_guard(as_base_guard(rows, gate), gate, True), 'rows': rows,
            }
        result[name] = details
    return result


def evaluate(mode, manifest, features):
    freeze_path = REPORTS / 'freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    for path, value in freeze['real_input_sha256'].items():
        if digest(Path(path)) != value:
            raise ValueError(f'Frozen development data changed: {path}')
    sources = {}
    if mode == 'archived':
        _, cohorts, _ = real_data()
        scope = 'Previously observed diagnostic recordings only.'
    else:
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if not metadata.get('complete'):
            raise ValueError('Await the final verified context manifest')
        rows = context_rows(metadata)
        if sorted(row['id'] for row in rows) != freeze['reserved_context_ids']:
            raise ValueError('The actual context recordings differ from the frozen IDs')
        # Extraction emits all 50 feature records. Only held rows are passed to scoring.
        fresh, _ = read_utterances(manifest, [features], False)
        cohorts = {'new_context_development_diagnostic': fresh['val'], 'new_context_novel': fresh['test']}
        scope = 'New real utterances from fixed signers and novel ordered label tuples. No fresh result tuned weights or policy.'
        sources = {path.as_posix(): digest(path) for path in (manifest, features)}
    outcomes = score_arms(freeze, cohorts)
    actual = set(freeze['real_glosses'])
    synthetic_only = set(freeze['synthetic_only_glosses'])
    lexical = {group: Counter('authentic_train' if token in actual else 'synthetic_only' if token in synthetic_only else 'unseen'
                              for row in rows for token in row.glosses) for group, rows in cohorts.items()}
    path = REPORTS / f'{mode}.json'
    write(path, {'scope': scope, 'freeze_sha256': digest(freeze_path), 'inputs': sources,
                 'lexical_support_by_reference_token': lexical, 'arms': outcomes})
    for name, groups in outcomes.items():
        print(name, {group: {'wer': result['raw']['gloss_wer'], 'exact': result['raw']['gloss_sequence_exact'],
                            'correct_text': result['joint_guard']['correct_text'],
                            'false_text': result['joint_guard']['wrong_or_out_of_scope_text']}
                     for group, result in groups.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'archived', 'context'))
    parser.add_argument('--pretrain-epochs', type=int, default=12)
    parser.add_argument('--finetune-epochs', type=int, default=60)
    parser.add_argument('--context-manifest', type=Path, default=CONTEXT / 'manifest.json')
    parser.add_argument('--context-features', type=Path, default=CONTEXT / 'features.jsonl')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'train':
        if not 10 <= args.pretrain_epochs <= 20 or not 40 <= args.finetune_epochs <= 100:
            parser.error('Use 10–20 pretraining epochs and 40–100 fine-tuning epochs')
        train(args.pretrain_epochs, args.finetune_epochs)
    else:
        evaluate(args.phase, args.context_manifest, args.context_features)


if __name__ == '__main__':
    main()
