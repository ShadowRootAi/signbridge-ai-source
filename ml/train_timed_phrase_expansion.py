"""Compare matched phrase training with and without added ELAN sign timing."""

import argparse
from collections import Counter, defaultdict
import json
from math import ceil
from pathlib import Path
import random
import re
import unicodedata

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from evaluate_scaled_phrases import edit_counts, evaluate_rows, normalize_text
from train_continuous import Utterance, read_utterances
from train_grounded_greek import write
from train_native_greek import Pieces, load_streams, resample
from train_phrase_spans import FIT_FEATURES, MANIFEST, SOURCE_CHECKPOINT, starting_point
from train_scaled_phrases import (
    ARTIFACTS, REPORT, PhraseModel, apply_policy, calibrate, candidates, cohort_digest, digest,
    greek, input_batch, matched_videos, predict, read_selected, static_videos,
)


NATIVE = Path('data/sartinas-dialogue-fit-expanded')
PRIOR_SPAN_FREEZE = REPORT / 'span-freeze.json'
PRIOR_VISUAL = ARTIFACTS / 'estimated_spans.pt'
FREEZE = REPORT / 'timed-expansion-freeze.json'
ARMS = ('expanded_untimed', 'expanded_timed')
SEED = 6509
FPS = 15


def gloss_form(label):
    # Preserve the author's lexical annotation and explicit variants.
    return re.sub(r'\s+', '_', unicodedata.normalize('NFC', label).strip().upper())


def load_native(folder, role):
    streams, evidence = load_streams(folder, (role,))
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    index = {(row['dialogue_id'], row['start_ms'], row['end_ms']): row for row in manifest['samples'] if row['split'] == role}
    rows, anchors, info = [], {}, {}
    for stream in streams:
        frames = resample(stream.frames, stream.fps, FPS)
        for start, end, _ in stream.turns:
            raw = index[(stream.id, round(start * 1000), round(end * 1000))]
            spans = [(a, b, label) for a, b, label in stream.glosses if start <= a < b <= end]
            if [label for _, _, label in spans] != raw['glosses']:
                raise ValueError(f'EAF gloss labels differ from the utterance annotation: {raw["id"]}')
            if not 2 <= len(spans) <= 14:
                continue
            first, last = ceil(start * FPS - 1e-7), min(len(frames), ceil(end * FPS - 1e-7))
            windows = [(max(0, ceil(a * FPS - 1e-7) - first), min(last - first, ceil(b * FPS - 1e-7) - first))
                       for a, b, _ in spans]
            if any(a >= b for a, b in windows):
                continue
            identity = raw['id']
            sequence = tuple(gloss_form(label) for _, _, label in spans)
            rows.append(Utterance(identity, stream.id + '/' + identity, stream.signer, role, sequence,
                                  raw['text'], frames[first:last], FPS, 1., None))
            anchors[identity] = windows
            info[identity] = {'category': 'sartinas', 'composition': stream.scenario}
    return rows, anchors, info, evidence


def initialize():
    original, labels, _ = starting_point()
    previous = json.loads(PRIOR_SPAN_FREEZE.read_bytes())
    if digest(PRIOR_VISUAL) != previous['arms']['estimated_spans']['checkpoint_sha256']:
        raise ValueError('The prior timing checkpoint changed')
    visual = torch.load(PRIOR_VISUAL, weights_only=True, map_location='cpu')
    model = torch.load(SOURCE_CHECKPOINT, weights_only=True, map_location='cpu')
    if labels != visual['labels'] or labels != model['labels']:
        raise ValueError('The starting visual models disagree on vocabulary')
    model['state_dict'].update({f'visual.{key}': value for key, value in visual['state_dict'].items()})
    return original, model


def build_model(state, labels):
    old = state['state_dict']
    codec = Pieces(state['merges'])
    model = PhraseModel(labels, codec, {name: old[f'visual.{name}'] for name in ('mean', 'scale')})
    weights = model.state_dict()
    expanded = {'visual.ctc.weight', 'visual.ctc.bias', 'visual.gloss_presence.weight', 'visual.gloss_presence.bias'}
    for name, value in old.items():
        if name not in expanded:
            weights[name].copy_(value)
        else:
            weights[name][:len(value)].copy_(value)
    model.load_state_dict(weights)
    return model, codec


def joint_loss(model, rows, vocabulary, codec, windows):
    frames, lengths = input_batch(rows)
    _, memory, active = model.visual.encode(frames, lengths)
    mask = active.squeeze(-1)
    logits = model.visual.ctc(model.visual.dropout(memory))
    target = [torch.tensor([vocabulary[label] for label in row.glosses]) for row in rows]
    counts = torch.tensor([len(item) for item in target])
    recognition = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.cat(target), lengths, counts)
    pieces = pad_sequence([torch.tensor([codec.bos, *codec.encode(row.text), codec.eos]) for row in rows], batch_first=True)
    incoming, outgoing = pieces[:, :-1], pieces[:, 1:]
    right = model.text_loss(memory, mask, incoming, outgoing)
    loss = recognition + .65 * right.mean()
    wrong_index = torch.tensor(candidates(rows))
    valid = wrong_index != torch.arange(len(rows))
    if valid.any():
        wrong = model.text_loss(memory[wrong_index], mask[wrong_index], incoming, outgoing)
        loss = loss + .25 * F.relu(.25 + right[valid] - wrong[valid]).mean()
    segment_logits, segment_labels = [], []
    for i, row in enumerate(rows):
        for (start, end), label in zip(windows.get(row.id, ()), target[i]):
            segment_logits.append(logits[i, start:end, 1:].mean(0))
            segment_labels.append(int(label) - 1)
    if segment_logits:
        loss = loss + .40 * F.cross_entropy(torch.stack(segment_logits), torch.tensor(segment_labels))
    return loss


def fit(updates, extra):
    if FREEZE.exists():
        raise ValueError('The comparison is already frozen')
    old, state = initialize()
    roles, features_hash = read_selected(MANIFEST, FIT_FEATURES, ('train', 'val'))
    if features_hash != old['train_dev_features_sha256'] or cohort_digest(roles['val']) != old['dev_cohort_sha256']:
        raise ValueError('The original fit and development features changed')
    training, native_windows, info, evidence = load_native(NATIVE, 'train')
    validation, _, native_info, val_evidence = load_native(NATIVE, 'val')
    known_train = {row.signer for row in training}
    known_val = {row.signer for row in validation}
    if known_train & known_val or len(known_train) < 3 or len(training) < 80:
        raise ValueError('Unexpected Sartinas signer split')
    protected = json.loads(MANIFEST.read_bytes())['samples']
    protected_text = {normalize_text(row['text']) for row in protected if row['split'] != 'train'}
    protected_gloss = {tuple(row['glosses']) for row in protected if row['split'] != 'train'}
    training = [row for row in training if normalize_text(row.text) not in protected_text and row.glosses not in protected_gloss]
    extra_rows, extra_windows, extra_evidence = [], {}, None
    if extra:
        extra_manifest, extra_features = extra / 'manifest.json', extra / 'features.jsonl'
        data, extra_features_hash = read_selected(extra_manifest, extra_features, ('train',))
        extra_rows = data['train']
        extra_anchors = extra / 'estimated-gloss-spans.json'
        anchor_data = json.loads(extra_anchors.read_bytes())
        if anchor_data['manifest_sha256'] != digest(extra_manifest):
            raise ValueError('Additional spans do not match their verified RGB sources')
        extra_by_id = {row.id: row for row in extra_rows}
        for identity, spans in anchor_data['spans'].items():
            row = extra_by_id[identity]
            if tuple(label for _, _, label in spans) != row.glosses or row.fps != FPS:
                raise ValueError(f'Unexpected extra alignment: {identity}')
            windows = [((a + 1) // 2, (b + 1) // 2) for a, b, _ in spans]
            if any(not 0 <= a < b <= len(row.frames) for a, b in windows):
                raise ValueError(f'Out-of-range extra alignment: {identity}')
            extra_windows[identity] = windows
        if not extra_windows:
            raise ValueError('Additional cohort lacks eligible timing')
        extra_evidence = {'manifest': digest(extra_manifest), 'features': extra_features_hash, 'anchors': digest(extra_anchors)}
    train = roles['train'] + training + extra_rows
    dev = roles['val'] + validation
    if len({row.id for row in train + dev}) != len(train) + len(dev):
        raise ValueError('Duplicate short-phrase examples')
    info.update(native_info)
    info.update({row.id: {'category': 'certh'} for row in roles['val']})
    used = set(state['labels'][1:])
    labels = [*state['labels'], *sorted({token for row in train for token in row.glosses} - used)]
    vocabulary = {label: i for i, label in enumerate(labels) if i}
    windows = {row.id: native_windows[row.id] for row in training} | extra_windows
    corpus_groups = []
    for source in (roles['train'] + extra_rows, training):
        grouped = defaultdict(list)
        for row in source:
            grouped[row.signer].append(row)
        corpus_groups.append(list(grouped.values()))
    outcomes = {}
    print(f'{len(train)} paired clips, {len(training)} timed Sartinas, {len(extra_rows)} additional CERTH, {len(labels)-1} gloss labels', flush=True)
    for arm in ARMS:
        rng = random.Random(SEED)
        torch.manual_seed(SEED)
        model, codec = build_model(state, labels)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00055)
        history, best = [], None
        for step in range(1, updates + 1):
            model.train()
            group = rng.choice(corpus_groups[step % 2])
            batch = rng.sample(group, min(12, len(group)))
            loss = joint_loss(model, batch, vocabulary, codec, windows if arm == 'expanded_timed' else {})
            if not loss.isfinite():
                raise ValueError(f'Non-finite loss in {arm}, step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
            if step % 60 and step != updates:
                continue
            results = predict(model, dev, info, labels, codec)
            metrics = evaluate_rows(results)
            entry = {'step': step, 'loss': float(loss.detach()), 'gloss_edits': metrics['gloss_edits'],
                     'word_edits': metrics['greek_word_edits'], 'greek_exact': metrics['greek_reference_exact'],
                     'gloss_exact': metrics['gloss_sequence_exact']}
            history.append(entry)
            rank = (entry['gloss_edits'] + entry['word_edits'], -entry['greek_exact'], -entry['gloss_exact'])
            if best is None or rank < best[0]:
                best = (rank, step, {name: value.detach().clone() for name, value in model.state_dict().items()})
            print(arm, entry, flush=True)
        model.load_state_dict(best[2])
        results = predict(model, dev, info, labels, codec)
        originals, alternatives = matched_videos(dev)
        negatives = predict(model, originals, info, labels, codec, alternatives)
        static = predict(model, dev, info, labels, codec, static_videos(dev))
        policy, calibration = calibrate(results, negatives, static)
        path = ARTIFACTS / f'{arm}.pt'
        pending = path.with_suffix('.tmp')
        torch.save({'state_dict': model.state_dict(), 'labels': labels, 'merges': codec.merges}, pending)
        pending.replace(path)
        outcomes[arm] = {'checkpoint_sha256': digest(path), 'selected_update': best[1], 'history': history,
                         'development': evaluate_rows(results), 'policy': policy, 'calibration': calibration}
    write(FREEZE, {'scope': 'Matched paired utterance training with and without new timing supervision.',
                   'implementation_sha256': digest(Path(__file__)), 'source_visual_sha256': digest(PRIOR_VISUAL),
                   'source_translator_sha256': digest(SOURCE_CHECKPOINT), 'certh_features_sha256': features_hash,
                   'native_training': evidence, 'native_development': val_evidence, 'extra_training': extra_evidence,
                   'train_signers': sorted({row.signer for row in train}),
                   'dev_signers': sorted({row.signer for row in dev}), 'train_ids': sorted(row.id for row in train),
                   'dev_ids': sorted(row.id for row in dev), 'training_texts': sorted({normalize_text(row.text) for row in train}),
                   'timed_clips': len(windows), 'timed_signs': sum(map(len, windows.values())),
                   'native_fit_clips': len(training), 'native_unique_texts': len({greek(row.text) for row in training}),
                   'updates': updates, 'arms': outcomes})


def text_metrics(rows):
    counts = Counter()
    for row in rows:
        reference = normalize_text(row['reference']).split()
        prediction = normalize_text(row['text'] or '').split()
        counts.update(edit_counts(reference, prediction))
        counts['reference_words'] += len(reference)
        counts['exact'] += reference == prediction
        counts['accepted'] += bool(prediction)
    return {'clips': len(rows), **counts}


def score(folder, name, native):
    freeze = json.loads(FREEZE.read_bytes())
    if (freeze['implementation_sha256'] != digest(Path(__file__))
            or freeze['source_visual_sha256'] != digest(PRIOR_VISUAL)
            or freeze['source_translator_sha256'] != digest(SOURCE_CHECKPOINT)):
        raise ValueError('Frozen source changed')
    if native:
        held, _, info, evidence = load_native(folder, 'test')
    else:
        manifest, features = folder / 'manifest.json', folder / 'features.jsonl'
        verified = json.loads(manifest.read_bytes())
        raw = verified['samples']
        if not verified.get('complete') or not raw or any(row['split'] != 'test' for row in raw):
            raise ValueError('External test requires a dedicated held manifest')
        rows, _ = read_utterances(manifest, [features], False, ('test',))
        held = rows['test']
        info = {row['id']: {'category': name, 'composition': row.get('composition', name)} for row in raw}
        evidence = {'manifest_sha256': digest(manifest), 'features_sha256': digest(features)}
    excluded = set(freeze['train_ids']) | set(freeze['dev_ids'])
    if (not held or any(row.id in excluded for row in held)
            or {row.signer for row in held} & (set(freeze['train_signers']) | set(freeze['dev_signers']))):
        raise ValueError('The external split overlaps fitting or selection')
    used_text = set(freeze['training_texts'])
    unseen = [row for row in held if normalize_text(row.text) not in used_text]
    if not unseen:
        raise ValueError('No held text is distinct from training')
    original, state = initialize()
    outcomes = {}
    for arm in ('prior_timed', *ARMS):
        if arm == 'prior_timed':
            checkpoint = state
            policy = original['policy']
        else:
            path = ARTIFACTS / f'{arm}.pt'
            if digest(path) != freeze['arms'][arm]['checkpoint_sha256']:
                raise ValueError(f'Frozen weights changed: {arm}')
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            policy = freeze['arms'][arm]['policy']
        model, codec = build_model(checkpoint, checkpoint['labels'])
        predictions = predict(model, unseen, info, checkpoint['labels'], codec)
        guarded = apply_policy(predictions, policy)
        metrics = evaluate_rows(predictions)
        vocabulary = set(checkpoint['labels'][1:])
        outcomes[arm] = {'gloss_and_text': metrics, 'text': text_metrics(predictions),
                         'guarded_text': text_metrics(guarded), 'predictions': predictions,
                         'test_signers': sorted({row.signer for row in unseen}),
                         'glosses_in_vocabulary': sum(token in vocabulary for row in unseen for token in row.glosses),
                         'clips_fully_in_vocabulary': sum(set(row.glosses) <= vocabulary for row in unseen)}
        print(arm, outcomes[arm]['text'],
              metrics['gloss_edit_counts'], 'gloss_exact', metrics['gloss_sequence_exact'], flush=True)
    write(REPORT / f'timed-expansion-{name}.json', {'scope': 'Whole utterance clips, no sign boundaries at inference.',
                                                   'freeze_sha256': digest(FREEZE), 'source': evidence,
                                                   'train_text_overlap_excluded': len(held) - len(unseen),
                                                   'arms': outcomes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('--updates', type=int, default=540)
    parser.add_argument('--extra', type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--name', default='external')
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    if args.updates < 1 or args.phase == 'score' and not args.test:
        parser.error('Fit needs positive updates, score needs a test folder')
    torch.set_num_threads(3)
    if args.phase == 'fit':
        fit(args.updates, args.extra)
    else:
        score(args.test, args.name, args.native)


if __name__ == '__main__':
    main()
