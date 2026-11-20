"""Fit and compare short-phrase GSL models on separately reserved CERTH recordings."""

import argparse
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from train_continuous import ctc_decode, edit_counts
from train_continuous_hard import read_selected
from train_continuous_library import PRETRAIN_ARTIFACT, PRETRAIN_FREEZE, transfer_state
from train_continuous_multitask import OrderAwareCTC, WIDTH, digest
from train_grounded_greek import write
from train_greek_translation_pilot import greek
from train_native_greek import Pieces


DATA = Path('data/continuous-scale')
REPORT = Path('reports/continuous-scale')
ARTIFACTS = Path('artifacts/continuous-scale')
SEED = 1129
BATCH_SIZE = 16
DECODER_WIDTH = 96
MAX_PIECES = 40
SIGNERS = {'train': {'1', '2', '4', '5'}, 'val': {'3'}, 'test': {'6', '7'}}
ARMS = ('baseline', 'scaled')


class PhraseModel(nn.Module):
    def __init__(self, labels, codec, weights):
        super().__init__()
        self.visual = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
        self.embedding = nn.Embedding(codec.eos + 1, DECODER_WIDTH, padding_idx=0)
        self.initial = nn.Linear(WIDTH * 2, DECODER_WIDTH)
        self.key = nn.Linear(WIDTH * 2, DECODER_WIDTH)
        self.decoder = nn.GRU(DECODER_WIDTH, DECODER_WIDTH, batch_first=True)
        self.text = nn.Linear(DECODER_WIDTH + WIDTH * 2, codec.eos + 1)

    def context(self, memory, mask):
        mean = (memory * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.initial(mean).tanh().unsqueeze(0)

    def decode(self, memory, mask, keys, tokens, state):
        query, state = self.decoder(self.embedding(tokens), state)
        similarity = (query @ keys.transpose(1, 2)) / sqrt(DECODER_WIDTH)
        attention = similarity.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        return self.text(torch.cat((query, attention @ memory), dim=-1)), state

    def text_loss(self, memory, mask, source, target):
        logits, _ = self.decode(memory, mask, self.key(memory), source, self.context(memory, mask))
        losses = F.cross_entropy(logits.transpose(1, 2), target, reduction='none', ignore_index=0)
        return losses.sum(1) / (target != 0).sum(1)

    def generate(self, memory, mask, codec):
        keys, state = self.key(memory), self.context(memory, mask)
        tokens = torch.full((len(memory), 1), codec.bos, dtype=torch.long)
        ended = torch.zeros(len(memory), dtype=torch.bool)
        outputs = [[] for _ in memory]
        scores = torch.zeros(len(memory))
        steps = torch.zeros(len(memory))
        for _ in range(MAX_PIECES):
            logits, state = self.decode(memory, mask, keys, tokens, state)
            scores_for_token = logits[:, 0].log_softmax(-1)
            scores_for_token[:, [0, codec.bos]] = -torch.inf
            values, ids = scores_for_token.max(-1)
            scores += torch.where(~ended, values, 0.)
            steps += ~ended
            for i, token in enumerate(ids.tolist()):
                if not ended[i] and token != codec.eos:
                    outputs[i].append(token)
            ended |= ids == codec.eos
            if bool(ended.all()):
                break
            tokens = ids[:, None]
        return [(codec.decode(row), float(confidence), bool(finished))
                for row, confidence, finished in zip(outputs, (scores / steps).exp(), ended, strict=True)]


def metadata(path):
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    if not raw.get('complete'):
        raise ValueError('The source cohort must have completed RGB verification')
    rows = raw['samples']
    if any(row['split'] not in SIGNERS for row in rows):
        raise ValueError('Unexpected split')
    for role, signers in SIGNERS.items():
        if {str(row['signer']) for row in rows if row['split'] == role} != signers:
            raise ValueError(f'Unexpected signer assignment for {role}')
    for identity in ('id', 'source_path'):
        if len({row[identity] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate {identity}')
    payloads = {}
    references = {}
    for row in rows:
        for key in (('Greek', greek(row['text'])), ('glosses', tuple(row['glosses']))):
            if not key[1] or references.setdefault(key, row['split']) != row['split']:
                raise ValueError(f'Empty or cross-role reference: {row["id"]}')
        signature = row.get('sha256')
        if not signature or payloads.setdefault(signature, row['split']) != row['split']:
            raise ValueError(f'Cross-role RGB digest: {row["id"]}')
    return {row['id']: row for row in rows}


def candidates(rows):
    text = [greek(row.text) for row in rows]
    words = [set(reference.split()) for reference in text]
    glosses = [set(row.glosses) for row in rows]
    return [max(
        (j for j, other in enumerate(rows) if other.signer == row.signer
         and text[j] != text[i] and other.glosses != row.glosses),
        key=lambda j: (len(glosses[i] & glosses[j]) / len(glosses[i] | glosses[j]),
                       len(words[i] & words[j]), -abs(len(row.frames) - len(rows[j].frames)), -j),
        default=i) for i, row in enumerate(rows)]


def input_batch(rows):
    lengths = torch.tensor([len(row.frames) for row in rows], dtype=torch.long)
    return pad_sequence([row.frames for row in rows], batch_first=True), lengths


def training_loss(model, batch, gloss_ids, codec, *, joint):
    frames, lengths = input_batch(batch)
    _, memory, active = model.visual.encode(frames, lengths)
    mask = active.squeeze(-1)
    labels = [torch.tensor([gloss_ids[gloss] for gloss in row.glosses]) for row in batch]
    needed = [len(label) + int((label[1:] == label[:-1]).sum()) for label in labels]
    if any(length < size for length, size in zip(lengths.tolist(), needed, strict=True)):
        raise ValueError('Insufficient frames for CTC alignment')
    probabilities = model.visual.ctc(model.visual.dropout(memory)).log_softmax(-1).transpose(0, 1)
    loss = F.ctc_loss(probabilities, torch.cat(labels), lengths, torch.tensor([len(label) for label in labels]))
    pooled = torch.cat((memory.sum(1) / lengths[:, None],
                        memory.masked_fill(~active, -torch.inf).amax(1)), dim=-1)
    presence = model.visual.gloss_presence(model.visual.dropout(pooled))
    target_presence = torch.zeros_like(presence)
    for i, label in enumerate(labels):
        target_presence[i, label - 1] = 1
    errors = F.binary_cross_entropy_with_logits(presence, target_presence, reduction='none')
    positive = (errors * target_presence).sum(1) / target_presence.sum(1)
    negative = (errors * (1 - target_presence)).sum(1) / (1 - target_presence).sum(1).clamp_min(1)
    loss = loss + .10 * (positive + negative).mean()
    if joint:
        targets = pad_sequence([torch.tensor([codec.bos, *codec.encode(row.text), codec.eos])
                                for row in batch], batch_first=True)
        source, target = targets[:, :-1], targets[:, 1:]
        correct = model.text_loss(memory, mask, source, target)
        loss = loss + correct.mean()
        rivals = torch.tensor(candidates(batch))
        valid = rivals != torch.arange(len(batch))
        if valid.any():
            wrong = model.text_loss(memory[rivals], mask[rivals], source, target)
            loss = loss + .35 * F.relu(.25 + correct[valid] - wrong[valid]).mean()
    return loss


@torch.inference_mode()
def predict(model, rows, info, labels, codec, inputs=None):
    model.eval()
    inputs = rows if inputs is None else inputs
    output = []
    for start in range(0, len(rows), 32):
        group, videos = rows[start:start + 32], inputs[start:start + 32]
        frames, lengths = input_batch(videos)
        _, memory, active = model.visual.encode(frames, lengths)
        scores = model.visual.ctc(memory).log_softmax(-1)
        generated = model.generate(memory, active.squeeze(-1), codec)
        for row, video, (text, confidence, finished), logits in zip(group, videos, generated, scores, strict=True):
            glosses, gloss_confidence = ctc_decode(logits[:len(video.frames)], labels)
            sample = info[row.id]
            output.append({'id': row.id, 'input_video_id': video.id, 'signer': row.signer,
                           'input_frames': len(video.frames), 'input_reference': video.text, 'input_signer': video.signer,
                           'category': sample.get('category') or sample.get('composition', 'unspecified'),
                           'composition': sample.get('composition', 'unspecified'),
                           'gold_glosses': list(row.glosses), 'predicted_glosses': list(glosses),
                           'reference': row.text, 'text': (text or None) if finished else None,
                           'text_confidence': confidence, 'gloss_confidence': gloss_confidence,
                           'tracking': video.quality})
    return output


def development_key(rows):
    words = sum(sum(edit_counts(greek(row['reference']).split(), (row['text'] or '').split())) for row in rows)
    glosses = sum(sum(edit_counts(row['gold_glosses'], row['predicted_glosses'])) for row in rows)
    return (words + glosses, words, glosses)


def calibrate(rows, rivals, static):
    confidence = (.0, .45, .60, .70, .80, .90, .95, 1.01)
    best = None
    for floor in confidence:
        for visual_floor in confidence:
            policy = {'text_confidence': floor, 'gloss_confidence': visual_floor}
            accepted = apply_policy(rows, policy)
            wrong_videos = apply_policy(rivals, policy)
            exact = sum(row['text'] == greek(row['reference']) for row in accepted if row['text'])
            emitted = sum(bool(row['text']) for row in accepted)
            false = sum(row['text'] == greek(row['reference']) for row in wrong_videos if row['text'])
            static_emitted = sum(bool(row['text']) for row in apply_policy(static, policy))
            if (emitted and exact / emitted < .80) or false > len(rivals) * .10 or static_emitted > len(static) * .10:
                continue
            key = (exact, -(emitted - exact), -false, -static_emitted, floor + visual_floor)
            if best is None or key > best[0]:
                best = (key, policy, {'exact': exact, 'emitted': emitted, 'mismatched_anchor_exact': false,
                                      'static_emitted': static_emitted})
    return best[1], best[2]


def apply_policy(rows, policy):
    return [dict(row, text=row['text'] if row['predicted_glosses']
                 and row['text_confidence'] >= policy['text_confidence']
                 and row['gloss_confidence'] >= policy['gloss_confidence'] else None) for row in rows]


def matched_videos(rows):
    anchors, videos = [], []
    for i, j in enumerate(candidates(rows)):
        if i == j:
            continue
        source, donor = rows[i], rows[j]
        indices = torch.linspace(0, len(donor.frames) - 1, len(source.frames)).round().long()
        anchors.append(source)
        videos.append(replace(donor, frames=donor.frames[indices]))
    return anchors, videos


def static_videos(rows):
    return [replace(row, id=f'{row.id}:static', frames=row.frames[len(row.frames) // 2].expand_as(row.frames))
            for row in rows]


def cohort_digest(rows):
    digestor = sha256()
    for row in sorted(rows, key=lambda row: row.id):
        attributes = (row.id, row.source, row.signer, row.glosses, row.text, row.fps, row.quality)
        digestor.update(json.dumps(attributes, ensure_ascii=False).encode())
        digestor.update(row.frames.numpy().tobytes())
    return digestor.hexdigest()


def fit(args):
    freeze = REPORT / f'{args.arm}-freeze.json'
    if freeze.exists():
        raise ValueError(f'The arm is already frozen: {freeze}')
    info = metadata(args.manifest)
    extracted, feature_hash = read_selected(args.manifest, args.features, ('train', 'val'))
    train, development = extracted['train'], extracted['val']
    if len(train) < (140 if args.arm == 'baseline' else 500) or len(development) < 20:
        raise ValueError('The arm has too few verified clips')
    if any(row.fps != 15 or not greek(row.text) for row in train + development):
        raise ValueError('Expected valid 15 Hz paired features')
    initialization = json.loads(PRETRAIN_FREEZE.read_bytes())
    if (digest(PRETRAIN_ARTIFACT) != initialization['checkpoint_sha256']
            or set(initialization['source']['signer_counts']) & (SIGNERS['val'] | SIGNERS['test'])):
        raise ValueError('The isolated initialization no longer matches its signer-clean freeze')
    pretrained = torch.load(PRETRAIN_ARTIFACT, weights_only=True, map_location='cpu')
    labels = [None, *sorted(set(pretrained['labels'][1:]) | {label for row in train for label in row.glosses})]
    gloss_ids = {label: i for i, label in enumerate(labels) if i}
    codec = Pieces.fit([row.text for row in train])
    if any(len(codec.encode(row.text)) >= MAX_PIECES for row in train):
        raise ValueError('A training text exceeds the fixed decoder limit')
    torch.manual_seed(SEED)
    model = PhraseModel(labels, codec, pretrained['state_dict'])
    transfer_state(model.visual, labels, pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.01)
    by_signer = defaultdict(list)
    for row in train:
        by_signer[row.signer].append(row)
    groups = list(by_signer.values())
    rng = random.Random(SEED)
    best, history = None, []
    print(f'{args.arm}: {len(train)} train / {len(development)} dev / {len(labels) - 1} glosses', flush=True)
    for step in range(1, args.updates + 1):
        model.train()
        group = rng.choices(groups, weights=list(map(len, groups)))[0]
        batch = rng.sample(group, min(BATCH_SIZE, len(group)))
        loss = training_loss(model, batch, gloss_ids, codec, joint=step > args.warmup)
        if not loss.isfinite():
            raise ValueError(f'Non-finite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 4.)
        optimizer.step()
        if step <= args.warmup or (step % 60 and step != args.updates):
            continue
        rows = predict(model, development, info, labels, codec)
        key = development_key(rows)
        entry = {'step': step, 'loss': float(loss.detach()), 'dev_word_edits': key[1], 'dev_gloss_edits': key[2],
                 'dev_exact_text': sum(row['text'] == greek(row['reference']) for row in rows)}
        history.append(entry)
        if best is None or key < best[0]:
            best = (key, step, {name: value.detach().clone() for name, value in model.state_dict().items()})
        print(json.dumps(entry), flush=True)
    model.load_state_dict(best[2])
    dev = predict(model, development, info, labels, codec)
    originals, alternatives = matched_videos(development)
    negatives = predict(model, originals, info, labels, codec, alternatives)
    static = predict(model, development, info, labels, codec, static_videos(development))
    policy, calibration = calibrate(dev, negatives, static)
    path = ARTIFACTS / f'{args.arm}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp')
    torch.save({'state_dict': model.state_dict(), 'labels': labels, 'merges': codec.merges}, pending)
    pending.replace(path)
    write(freeze, {'scope': 'Fitted on train, selected and calibrated on val before any test prediction.',
                   'manifest': args.manifest.as_posix(), 'manifest_sha256': digest(args.manifest),
                   'train_dev_features_sha256': feature_hash, 'dev_cohort_sha256': cohort_digest(development),
                   'checkpoint_sha256': digest(path),
                   'initialization_sha256': initialization['checkpoint_sha256'],
                   'implementation_sha256': digest(Path(__file__)),
                   'splits': {role: sorted(row['id'] for row in info.values() if row['split'] == role)
                              for role in SIGNERS}, 'updates': args.updates, 'warmup': args.warmup,
                   'selected_update': best[1], 'development': history, 'policy': policy,
                   'calibration': calibration, 'train_glosses': sorted({label for row in train for label in row.glosses}),
                   'train_templates': len({info[row.id]['template'] for row in train})})


def score(args):
    from evaluate_scaled_phrases import evaluate_paired_rows, evaluate_rows

    info = metadata(args.manifest)
    arms = (args.arm,) if args.arm else ARMS
    freezes = {name: json.loads((REPORT / f'{name}-freeze.json').read_bytes()) for name in arms}
    ids = {role: sorted(row['id'] for row in info.values() if row['split'] == role) for role in SIGNERS}
    fitted = freezes[arms[-1]]
    if (fitted['splits'] != ids or fitted['manifest_sha256'] != digest(args.manifest)
            or any(arm['implementation_sha256'] != digest(Path(__file__)) for arm in freezes.values())):
        raise ValueError('The frozen experiment differs from the scoring inputs')
    if not args.arm:
        base, scaled = (freezes[name] for name in arms)
        if (any(base['splits'][role] != ids[role] for role in ('val', 'test'))
                or not set(base['splits']['train']) < set(ids['train'])
                or any(base[key] != scaled[key] for key in ('updates', 'warmup', 'initialization_sha256', 'dev_cohort_sha256'))):
            raise ValueError('The frozen arms are not a matched comparison')
    manifest, features = args.manifest, args.features
    scope = 'Whole CERTH utterance clips with supplied clip boundaries, unseen signers and phrase references.'
    suffix = f'-{args.arm}' if args.arm else ''
    destination = REPORT / f'benchmark{suffix}.json'
    if args.probe_manifest:
        manifest, features = args.probe_manifest, args.probe_features
        existing = list(info.values())
        probe = json.loads(manifest.read_bytes())
        if not probe.get('complete'):
            raise ValueError('The secondary probe must have completed RGB verification')
        members = probe['samples']
        reserved = {key: {row[key] for row in existing} for key in ('id', 'source_path', 'sha256')}
        reserved_texts = {greek(row['text']) for row in existing}
        reserved_glosses = {tuple(row['glosses']) for row in existing}
        covered = set.intersection(*(set(arm['train_glosses']) for arm in freezes.values()))
        if ({str(row['signer']) for row in members} != SIGNERS['test']
                or len({greek(row['text']) for row in members}) != 1
                or any(row['split'] != 'test' or not set(row['glosses']) <= covered
                       or any(not row.get(key) or row[key] in values for key, values in reserved.items())
                       or greek(row['text']) in reserved_texts or tuple(row['glosses']) in reserved_glosses
                       for row in members)):
            raise ValueError('The secondary probe has overlapping sources or unsupported glosses')
        info = {row['id']: row for row in members}
        scope = 'Secondary probe of a single new Greek reference and new gloss orders, with every gloss observed in continuous training.'
        destination = REPORT / f'composition-probe{suffix}.json'
    rows, feature_hash = read_selected(manifest, features, ('test',))
    held = rows['test']
    if any(row.fps != 15 for row in held):
        raise ValueError('The test cadence differs from training')
    originals, alternatives = matched_videos(held)
    result = {}
    for name in arms:
        arm = freezes[name]
        path = ARTIFACTS / f'{name}.pt'
        if digest(path) != arm['checkpoint_sha256']:
            raise ValueError(f'Frozen checkpoint changed: {name}')
        state = torch.load(path, weights_only=True, map_location='cpu')
        labels, codec = state['labels'], Pieces(state['merges'])
        norm = {key: state['state_dict'][f'visual.{key}'] for key in ('mean', 'scale')}
        model = PhraseModel(labels, codec, norm)
        model.load_state_dict(state['state_dict'])
        predictions = predict(model, held, info, labels, codec)
        indexed = {row['id']: row for row in predictions}
        negatives = predict(model, originals, info, labels, codec, alternatives)
        static = predict(model, held, info, labels, codec, static_videos(held))
        guarded = apply_policy(predictions, arm['policy'])
        paired = evaluate_paired_rows([indexed[row.id] for row in originals], negatives) if originals else None
        continuous_labels, model_labels = set(arm['train_glosses']), set(labels[1:])
        result[name] = {'raw': evaluate_rows(predictions), 'guarded': evaluate_rows(guarded),
                        'paired_input_diagnostic': paired, 'rows': predictions,
                        'guarded_rows': guarded, 'mismatched_rows': negatives,
                        'lexical_support': {
                            'all_glosses_seen_in_continuous_training': sum(set(row.glosses) <= continuous_labels for row in held),
                            'all_glosses_in_model_vocabulary': sum(set(row.glosses) <= model_labels for row in held),
                            'test_reference_texts': len({greek(row.text) for row in held}),
                        },
                        'static_text_emissions': {
                            'clips': len(static), 'raw': sum(bool(row['text']) for row in static),
                            'guarded': sum(bool(row['text']) for row in apply_policy(static, arm['policy'])),
                        }}
        print(name, development_key(predictions), flush=True)
    write(destination, {'scope': scope, 'manifest_sha256': digest(manifest), 'test_features_sha256': feature_hash,
                                     'freezes': {name: digest(REPORT / f'{name}-freeze.json') for name in arms},
                                     'arms': result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--manifest', type=Path, default=DATA / 'manifest.json')
    parser.add_argument('--features', type=Path, default=DATA / 'features.jsonl')
    parser.add_argument('--probe-manifest', type=Path)
    parser.add_argument('--probe-features', type=Path)
    parser.add_argument('--updates', type=int, default=780)
    parser.add_argument('--warmup', type=int, default=120)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.phase == 'fit' and (not args.arm or not 0 <= args.warmup < args.updates):
        parser.error('Fit requires an arm and a valid training schedule')
    if bool(args.probe_manifest) != bool(args.probe_features) or args.phase != 'score' and args.probe_manifest:
        parser.error('The secondary scoring probe requires both manifest and feature paths')
    torch.set_num_threads(args.threads)
    (fit if args.phase == 'fit' else score)(args)


if __name__ == '__main__':
    main()
