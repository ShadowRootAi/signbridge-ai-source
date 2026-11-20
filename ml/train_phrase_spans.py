"""Compare matched CTC continuation recipes for recognizing signs inside phrases."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from evaluate_scaled_phrases import evaluate_rows
from train_continuous import ctc_decode
from train_continuous_multitask import OrderAwareCTC, digest
from train_grounded_greek import write
from train_scaled_phrases import ARTIFACTS, DATA, REPORT, cohort_digest, greek, input_batch, read_selected, static_videos


MANIFEST = DATA / 'manifest.json'
FIT_FEATURES = DATA / 'features.fit.jsonl'
ANCHORS = DATA / 'estimated-gloss-spans.json'
SOURCE_FREEZE = REPORT / 'scaled-freeze.json'
SOURCE_CHECKPOINT = ARTIFACTS / 'scaled.pt'
FREEZE = REPORT / 'span-freeze.json'
SEED = 18723
ARMS = ('continued_ctc', 'contrastive_ctc')


def starting_point():
    original = json.loads(SOURCE_FREEZE.read_bytes())
    if (digest(SOURCE_CHECKPOINT) != original['checkpoint_sha256']
            or digest(MANIFEST) != original['manifest_sha256']):
        raise ValueError('Original training inputs or checkpoint changed')
    checkpoint = torch.load(SOURCE_CHECKPOINT, weights_only=True, map_location='cpu')
    weights = {key.removeprefix('visual.'): value for key, value in checkpoint['state_dict'].items()
               if key.startswith('visual.')}
    return original, checkpoint['labels'], weights


def make_model(labels, weights):
    model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    model.load_state_dict(weights)
    return model


def read_anchors(train):
    raw = json.loads(ANCHORS.read_bytes())
    if raw['manifest_sha256'] != digest(MANIFEST):
        raise ValueError('Span mapping is for a different RGB manifest')
    rows = {row.id: row for row in train}
    anchors = {}
    for identity, spans in raw['spans'].items():
        if identity not in rows or rows[identity].fps != 15:
            raise ValueError(f'Non-training or wrong-cadence span: {identity}')
        row = rows[identity]
        if tuple(label for _, _, label in spans) != row.glosses:
            raise ValueError(f'Span labels differ from source order: {identity}')
        windows, end_previous = [], 0
        for start, end, _ in spans:
            if not 0 <= end_previous <= start < end <= 2 * len(row.frames):
                raise ValueError(f'Invalid or overlapping source span: {identity}')
            end_previous = end
            left, right = (start + 1) // 2, (end + 1) // 2
            if right <= left:
                raise ValueError(f'No extracted frame falls inside source span: {identity}')
            windows.append((left, right))
        anchors[identity] = windows
    if not anchors:
        raise ValueError('No verified training spans')
    return anchors


def contrastive_loss(scores, lengths, targets, gold, rng):
    alternatives = []
    for clip, (target, length) in enumerate(zip(targets, lengths.tolist(), strict=True)):
        position = rng.randrange(len(target))
        visual = scores[:length, clip, 1:].detach().amax(0)
        visual[target[position] - 1] = -torch.inf
        rival = (visual.argmax() + 1).reshape(1)
        alternatives.append((
            torch.cat((target[:position], target[position + 1:])),
            torch.cat((target[:position], rival, target[position + 1:])),
            torch.cat((target[:position], rival, target[position:])),
        ))
    target_lengths = torch.tensor([len(target) for target in targets])
    losses = []
    for candidate in zip(*alternatives, strict=True):
        wrong = F.ctc_loss(scores, torch.cat(candidate), lengths,
                           torch.tensor([len(target) for target in candidate]), reduction='none')
        losses.append(F.relu(.40 + (gold - wrong) / target_lengths).mean())
    return torch.stack(losses).mean()


def train_step(model, rows, indexes, anchors, rival_rng=None):
    frames, lengths = input_batch(rows)
    _, memory, _ = model.encode(frames, lengths)
    logits = model.ctc(model.dropout(memory))
    targets = [torch.tensor([indexes[gloss] for gloss in row.glosses]) for row in rows]
    count = torch.tensor([len(target) for target in targets])
    scores = logits.log_softmax(-1).transpose(0, 1)
    gold = F.ctc_loss(scores, torch.cat(targets), lengths, count, reduction='none')
    loss = (gold / count).mean()
    if rival_rng is not None:
        loss = loss + .45 * contrastive_loss(scores, lengths, targets, gold, rival_rng)
    pooled, reference = [], []
    for i, (row, target) in enumerate(zip(rows, targets, strict=True)):
        for (left, right), label in zip(anchors.get(row.id, ()), target):
            # Use the source window to discriminate visually similar nonblank labels.
            pooled.append(logits[i, left:right, 1:].mean(0))
            reference.append(int(label) - 1)
    if pooled:
        loss = loss + .40 * F.cross_entropy(torch.stack(pooled), torch.tensor(reference))
    return loss


@torch.inference_mode()
def predictions(model, rows, labels, inputs=None):
    model.eval()
    inputs = rows if inputs is None else inputs
    results = []
    for start in range(0, len(rows), 24):
        group, videos = rows[start:start + 24], inputs[start:start + 24]
        frames, lengths = input_batch(videos)
        logits = model(frames, lengths)
        for row, scores, length in zip(group, logits, lengths.tolist(), strict=True):
            glosses, confidence = ctc_decode(scores[:length], labels)
            results.append({'id': row.id, 'signer': row.signer, 'split': row.split,
                            'gold_glosses': list(row.glosses), 'predicted_glosses': list(glosses),
                            'reference': row.text, 'text': None, 'gloss_confidence': confidence})
    return results


def quality(results):
    summary = evaluate_rows(results)
    counts = summary['gloss_edit_counts']
    return {'clips': summary['utterances'], 'reference_glosses': summary['reference_gloss_count'],
            'exact': summary['gloss_sequence_exact'], **counts}


def fit(updates, with_spans):
    if FREEZE.exists():
        raise ValueError(f'Previous fit already frozen: {FREEZE}')
    original, labels, weights = starting_point()
    roles, features_hash = read_selected(MANIFEST, FIT_FEATURES, ('train', 'val'))
    train, dev = roles['train'], roles['val']
    if (features_hash != original['train_dev_features_sha256'] or cohort_digest(dev) != original['dev_cohort_sha256']
            or any(sorted(row.id for row in roles[role]) != original['splits'][role] for role in roles)):
        raise ValueError('The original 520/36 split has changed')
    anchors = read_anchors(train) if with_spans else {}
    print(f'{len(anchors)} training clips with estimated windows; {sum(map(len, anchors.values()))} signs', flush=True)
    indexes = {label: i for i, label in enumerate(labels) if i}
    groups = defaultdict(list)
    for row in train:
        groups[row.signer].append(row)
    groups = list(groups.values())
    base = make_model(labels, weights)
    baseline = quality(predictions(base, dev, labels))
    static = static_videos(dev)
    outcomes = {}
    for arm in (*ARMS, 'estimated_spans') if with_spans else ARMS:
        model = make_model(labels, weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00045)
        rng = random.Random(SEED)
        rival_rng = random.Random(SEED + 1) if arm == 'contrastive_ctc' else None
        torch.manual_seed(SEED)
        best, history = None, []
        for step in range(1, updates + 1):
            model.train()
            group = rng.choice(groups)
            batch = rng.sample(group, min(16, len(group)))
            loss = train_step(model, batch, indexes, anchors if arm == 'estimated_spans' else {}, rival_rng)
            if not loss.isfinite():
                raise ValueError(f'Non-finite loss in {arm} at update {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
            if step % 45 and step != updates:
                continue
            metrics = quality(predictions(model, dev, labels))
            history.append({'step': step, 'loss': float(loss.detach()), **metrics})
            key = (metrics['edits'], metrics['substitutions'], -metrics['exact'])
            if best is None or key < best[0]:
                best = key, step, {key: value.detach().clone() for key, value in model.state_dict().items()}
            print(arm, history[-1], flush=True)
        model.load_state_dict(best[2])
        target = ARTIFACTS / f'{arm}.pt'
        pending = target.with_suffix('.tmp')
        torch.save({'state_dict': model.state_dict(), 'labels': labels}, pending)
        pending.replace(target)
        outcomes[arm] = {'checkpoint_sha256': digest(target), 'selected_update': best[1], 'history': history,
                         'development': quality(predictions(model, dev, labels)),
                         'static_nonempty': sum(bool(row['predicted_glosses']) for row in predictions(model, dev, labels, static))}
    write(FREEZE, {'scope': 'Matched recognition-only continuation and single-edit contrastive CTC. Optional estimated spans are training-only. Selection on signer-3 development.',
                   'implementation_sha256': digest(Path(__file__)), 'source_freeze_sha256': digest(SOURCE_FREEZE),
                   'source_manifest_sha256': digest(MANIFEST), 'source_checkpoint_sha256': digest(SOURCE_CHECKPOINT),
                   'fit_features_sha256': features_hash, 'anchors_sha256': digest(ANCHORS) if with_spans else None,
                   'anchors': len(anchors), 'sign_windows': sum(map(len, anchors.values())), 'updates_per_arm': updates,
                   'baseline_development': baseline, 'arms': outcomes})


def score(manifest, features, *, retrospective):
    _, labels, weights = starting_point()
    freeze = json.loads(FREEZE.read_bytes())
    if (freeze['implementation_sha256'] != digest(Path(__file__))
            or freeze['source_freeze_sha256'] != digest(SOURCE_FREEZE)
            or freeze['anchors_sha256'] is not None and freeze['anchors_sha256'] != digest(ANCHORS)):
        raise ValueError('The frozen experiment was modified')
    verified = json.loads(manifest.read_bytes())
    if not verified.get('complete'):
        raise ValueError('The RGB test cohort is incomplete')
    raw = [row for row in verified['samples'] if row['split'] == 'test']
    original = json.loads(MANIFEST.read_bytes())['samples']
    if not retrospective:
        source_ids = {row['source_path'] for row in original}
        training_text = {greek(row['text']) for row in original if row['split'] != 'test'}
        training_glosses = {tuple(row['glosses']) for row in original if row['split'] != 'test'}
        if any(row['source_path'] in source_ids or greek(row['text']) in training_text
               or tuple(row['glosses']) in training_glosses for row in raw):
            raise ValueError('A fresh source or its training reference overlaps the original cohort')
    elif digest(manifest) != freeze['source_manifest_sha256']:
        raise ValueError('Retrospective scoring requires the original observed cohort')
    rows, feature_hash = read_selected(manifest, features, ('test',))
    test = rows['test']
    if {row.signer for row in test} != {'6', '7'} or any(row.fps != 15 for row in test):
        raise ValueError('Unexpected signer or cadence in the evaluation cohort')
    models = {'scaled': make_model(labels, weights)}
    for arm in freeze['arms']:
        checkpoint = ARTIFACTS / f'{arm}.pt'
        if digest(checkpoint) != freeze['arms'][arm]['checkpoint_sha256']:
            raise ValueError(f'Frozen weights differ for {arm}')
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        models[arm] = make_model(state['labels'], state['state_dict'])
    outcomes = {}
    for name, model in models.items():
        results = predictions(model, test, labels)
        outcomes[name] = {'metrics': quality(results), 'predictions': results}
        print(name, outcomes[name]['metrics'], flush=True)
    destination = 'span-observed-benchmark.json' if retrospective else 'span-fresh-benchmark.json'
    write(REPORT / destination, {'scope': 'Observed original clips' if retrospective else 'Previously unused clips',
                                 'freeze_sha256': digest(FREEZE), 'manifest_sha256': digest(manifest),
                                 'test_features_sha256': feature_hash, 'arms': outcomes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('--updates', type=int, default=360)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--features', type=Path, default=DATA / 'features.jsonl')
    parser.add_argument('--retrospective', action='store_true')
    parser.add_argument('--with-estimated-spans', action='store_true')
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('Expected a positive number of updates')
    torch.set_num_threads(3)
    if args.phase == 'fit':
        fit(args.updates, args.with_estimated_spans)
    else:
        score(args.manifest, args.features, retrospective=args.retrospective)


if __name__ == '__main__':
    main()
