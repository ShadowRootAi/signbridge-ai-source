"""Train and compare an order-aware multi-task continuous GSL model offline."""

import argparse
from collections import defaultdict
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
import random
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from continuous_selective import evaluate_selective
from evaluate_continuous_transfer import SOURCES, load_inputs
from preprocess import FEATURE_COUNT, FEATURE_SCHEMA
from train_continuous import batch_loss, choose_threshold, read_utterances, reference_lookup, replay, summarize, train_model


SEED, EPOCHS, BATCH_SIZE, WIDTH = 41, 100, 12, 64
DILATIONS = (1, 2, 4)
PRESENCE_WEIGHT, ORDER_WEIGHT, ORDER_MARGIN = .35, .15, .35
REPORT = Path('reports/continuous-multitask/benchmark.json')
ARTIFACT = Path('artifacts/continuous-multitask/ctc.pt')
FRESH_MANIFEST = Path('data/continuous-fresh/manifest.json')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


class OrderAwareCTC(nn.Module):
    def __init__(self, count, center, scale):
        super().__init__()
        self.register_buffer('mean', center)
        self.register_buffer('scale', scale)
        self.projection = nn.Linear(FEATURE_COUNT, WIDTH)
        self.hand_motion = nn.Linear(168, WIDTH, bias=False)
        self.project_norm = nn.LayerNorm(WIDTH)
        self.convolutions = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATIONS)
        self.block_norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in DILATIONS)
        self.temporal = nn.GRU(WIDTH, WIDTH, batch_first=True, bidirectional=True)
        self.ctc = nn.Linear(2 * WIDTH, count)
        self.gloss_presence = nn.Linear(4 * WIDTH, count - 1)
        self.dropout = nn.Dropout(.1)

    def encode(self, frames, lengths):
        active = torch.arange(frames.shape[1])[None, :, None] < torch.as_tensor(lengths)[:, None, None]
        values = (frames - self.mean) / self.scale
        hand = values[:, :, :168]
        visible = frames[:, :, 220:222] >= .75
        present_in_both = visible[:, 1:] & visible[:, :-1]
        valid = present_in_both.repeat_interleave(84, dim=-1)
        motion = F.pad((hand[:, 1:] - hand[:, :-1]).clamp(-3, 3) * valid, (0, 0, 1, 0))
        features = F.gelu(self.project_norm(self.projection(values) + .3 * self.hand_motion(motion))) * active
        for convolution, norm in zip(self.convolutions, self.block_norms, strict=True):
            context = convolution(features.transpose(1, 2)).transpose(1, 2)
            features = norm(features + self.dropout(F.gelu(context))) * active
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        temporal, _ = self.temporal(packed)
        encoded, _ = pad_packed_sequence(temporal, batch_first=True, total_length=frames.shape[1])
        return features, encoded, active

    def outputs(self, frames, lengths):
        _, encoded, active = self.encode(frames, lengths)
        pooled = torch.cat((encoded.sum(dim=1) / active.sum(dim=1), encoded.masked_fill(~active, -torch.inf).amax(dim=1)), dim=-1)
        return self.ctc(self.dropout(encoded)).log_softmax(dim=-1), self.gloss_presence(self.dropout(pooled))

    def forward(self, frames, lengths):
        scores, _ = self.outputs(frames, lengths)
        return scores


def warp(item, generator):
    if generator.random() < .35:
        return item
    count = max(2 * len(item.glosses), round(len(item.frames) * generator.uniform(.85, 1.18)))
    elapsed = torch.linspace(0, 1, count).pow(generator.uniform(.85, 1.18))
    indexes = (elapsed * (len(item.frames) - 1)).round().long()
    return replace(item, frames=item.frames[indexes])


def loss_components(model, examples, vocabulary):
    targets = [torch.tensor([vocabulary[token] for token in item.glosses], dtype=torch.long) for item in examples]
    reversed_targets = [target.flip(0) for target in targets]
    lengths = torch.tensor([len(item.frames) for item in examples])
    target_lengths = torch.tensor([len(target) for target in targets])
    scores, present = model.outputs(pad_sequence([item.frames for item in examples], batch_first=True), lengths)
    ctc_scores = scores.transpose(0, 1)
    forward = F.ctc_loss(ctc_scores, torch.cat(targets), lengths, target_lengths, reduction='none') / target_lengths
    reverse = F.ctc_loss(ctc_scores, torch.cat(reversed_targets), lengths, target_lengths, reduction='none') / target_lengths
    orderable = torch.tensor([not torch.equal(a, b) for a, b in zip(targets, reversed_targets, strict=True)])
    order = (F.relu(ORDER_MARGIN + forward - reverse) * orderable).sum() / orderable.sum().clamp_min(1)
    reference_presence = torch.zeros_like(present)
    for i, target in enumerate(targets):
        reference_presence[i, target - 1] = 1
    binary = F.binary_cross_entropy_with_logits(present, reference_presence, reduction='none')
    positive = (binary * reference_presence).sum(1) / reference_presence.sum(1)
    negative = (binary * (1 - reference_presence)).sum(1) / (1 - reference_presence).sum(1).clamp_min(1)
    presence = ((positive + negative) / 2).mean()
    return forward.mean(), presence, order


def train_candidate(roles, labels, vocabulary, lookup, *, initial_model=None, epochs=EPOCHS, rehearsal=(), learning_rate=.002):
    training = roles['train']
    frames = torch.cat([item.frames for item in training])
    model = initial_model if initial_model is not None else OrderAwareCTC(
        len(labels), frames.mean(0), frames.std(0, unbiased=False).clamp_min(.08))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = random.Random(SEED)
    rehearsal_generator = random.Random(SEED + 1)
    rehearsal_loss = nn.CTCLoss(blank=0)
    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = defaultdict(list)
        shuffled = generator.sample(training, len(training))
        for start in range(0, len(shuffled), BATCH_SIZE):
            examples = [warp(item, generator) for item in shuffled[start:start + BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            ctc, presence, order = loss_components(model, examples, vocabulary)
            loss = ctc + PRESENCE_WEIGHT * presence + ORDER_WEIGHT * order
            if rehearsal and (start // BATCH_SIZE + 1) % 4 == 0:
                synthetic_ctc = batch_loss(model, rehearsal_generator.sample(rehearsal, BATCH_SIZE), vocabulary, rehearsal_loss)
                loss = loss + .20 * synthetic_ctc
                losses['synthetic_rehearsal_ctc'].append(float(synthetic_ctc.detach()))
            if not bool(loss.isfinite()):
                raise ValueError(f'Non-finite candidate loss in epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            for name, value in [('total', loss), ('ctc', ctc), ('presence', presence), ('order', order)]:
                losses[name].append(float(value.detach()))
        dev = summarize(replay(model, roles['val'], labels, lookup))
        row = {'epoch': epoch, 'train_losses': {name: mean(values) for name, values in losses.items()},
               'validation_gloss_wer': dev['gloss_wer'], 'validation_exact_sequences': dev['gloss_sequence_exact']}
        history.append(row)
        key = (dev['gloss_wer'], -dev['gloss_sequence_exact'], row['train_losses']['ctc'])
        if best is None or key < best[0]:
            best = key, epoch, {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
    model.load_state_dict(best[2])
    return model, history, best[1]


def score(model, labels, lookup, threshold, cohorts):
    result = {}
    for name, examples in cohorts.items():
        mode = name if name in ('reversed', 'shuffled', 'static_midpoint') else 'original'
        predictions = replay(model, examples, labels, lookup, threshold, mode)
        result[name] = {**summarize(predictions), 'predictions': predictions}
        if name != 'development' and not name.startswith('known_'):
            result[name]['out_of_scope_emissions'] = result[name]['accepted']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-manifest', type=Path, default=FRESH_MANIFEST,
                        help='Frozen complete manifest for the additional fresh cohort')
    parser.add_argument('--fresh-features', type=Path, nargs='+', help='Also train on the fresh train cohort and evaluate its untouched test')
    args = parser.parse_args()
    torch.set_num_threads(2)
    expanded, composition = (load_inputs(name) for name in ('expanded', 'composition'))
    unknown = load_inputs('old_unknowns', ('test',))['test']
    roles = {role: expanded[role] + composition[role] for role in ('train', 'val', 'test')}
    sources = [path for paths in SOURCES.values() for path in paths]
    report_path, artifact_path = REPORT, ARTIFACT
    fresh_references = None
    composition_manifest = json.loads(SOURCES['composition'][0].read_text(encoding='utf-8'))
    expected_novel = {item['id'] for item in composition_manifest['samples'] if item.get('composition') == 'unseen_template'}
    if args.fresh_features:
        manifest_path = args.fresh_manifest
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not manifest.get('complete') or len(manifest['samples']) != manifest['planned_samples']:
            raise ValueError('The frozen fresh cohort must be complete before evaluating it')
        fresh_references = {item['id']: item['composition'] for item in manifest['samples']}
        fresh, _ = read_utterances(manifest_path, args.fresh_features, False)
        roles = {'train': roles['train'] + fresh['train'], 'val': roles['val'] + fresh['val'], 'test': fresh['test']}
        expected_novel = {item.id for item in roles['test'] if fresh_references[item.id] == 'unseen_template'}
        sources.extend([manifest_path, *args.fresh_features])
        tag = manifest_path.resolve().parent.name
        report_path = Path('reports') / tag / 'multitask-benchmark.json'
        artifact_path = Path('artifacts') / tag / 'multitask.pt'
    signer_sets = {role: {item.signer for item in items} for role, items in roles.items()}
    if sum(map(len, signer_sets.values())) != len(set().union(*signer_sets.values())):
        raise ValueError('A signer crosses role boundaries')
    all_items = [item for items in roles.values() for item in items] + unknown
    if len({item.id for item in all_items}) != len(all_items) or len({item.source for item in all_items}) != len(all_items):
        raise ValueError('A real recording appears more than once')
    vocab = {gloss: i for i, gloss in enumerate(sorted({token for item in roles['train'] for token in item.glosses}), 1)}
    labels, lookup = [None, *vocab], reference_lookup(roles['train'])
    train_templates = {item.glosses for item in roles['train']}
    val_templates = {item.glosses for item in roles['val']}
    novel = [item for item in roles['test'] if item.glosses not in train_templates]
    if (not expected_novel or len(expected_novel) == len(roles['test'])
            or any(item.glosses in val_templates or set(item.glosses) - vocab.keys() for item in novel)):
        raise ValueError('Expected only predesignated novel test utterances with known individual glosses')
    if {item.id for item in novel} != expected_novel:
        raise ValueError('The fresh held-out templates no longer match the frozen roles')

    models = {}
    torch.manual_seed(SEED)
    print('Combined real-only GRU baseline', flush=True)
    models['plain_ctc'] = train_model(roles, labels, vocab, lookup, EPOCHS, BATCH_SIZE, SEED)
    torch.manual_seed(SEED)
    print('Order-aware temporal-convolution/BiGRU multi-task', flush=True)
    models['order_aware_multitask'] = train_candidate(roles, labels, vocab, lookup)
    calibrated = {}
    for name, (model, _, _) in models.items():
        development = replay(model, roles['val'], labels, lookup)
        for row in development:
            if row['reference_glosses'] not in train_templates:
                row['greek_reference_exact'] = False
        calibrated[name] = choose_threshold(development)

    known = ({'known_fresh': [item for item in roles['test'] if item.glosses in train_templates]}
             if fresh_references is not None else {'known_police1': expanded['test'],
                 'known_police2': [item for item in composition['test'] if item.glosses in train_templates]})
    cohorts = {'development': roles['val'], **known, 'unseen_composition': novel, 'genuine_out_of_scope': unknown,
               **{mode: roles['test'] for mode in ('reversed', 'shuffled', 'static_midpoint')}}
    selective = evaluate_selective(models['order_aware_multitask'][0], models['plain_ctc'][0], roles['val'],
                                   {key: items for key, items in cohorts.items() if key != 'development'},
                                   labels, lookup, calibrated['order_aware_multitask'])
    outcomes = {name: {'selected_epoch': epoch, 'threshold': calibrated[name], 'history': history,
                       'cohorts': score(model, labels, lookup, calibrated[name], cohorts)}
                for name, (model, history, epoch) in models.items()}
    config = {'width': WIDTH, 'dilations': DILATIONS, 'presence_weight': PRESENCE_WEIGHT,
              'order_weight': ORDER_WEIGHT, 'order_margin': ORDER_MARGIN, 'seed': SEED, 'max_epochs': EPOCHS,
              'batch_size': BATCH_SIZE, 'augmentation': 'Probability .65; nearest-frame time scale and '
                                                  'nonlinear warp sampled from .85–1.18, preserving both endpoints.'}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    gate = selective['variants'][selective['selected_on_development']]['policy']
    weights = {'state_dict': models['order_aware_multitask'][0].state_dict(), 'feature_schema': FEATURE_SCHEMA,
               'labels': labels, 'config': config, 'threshold': calibrated['order_aware_multitask'],
               'training_greek_references': lookup, 'experimental_selective_policy': gate}
    if gate['require_plain_ctc_agreement']:
        weights['agreement_state_dict'] = models['plain_ctc'][0].state_dict()
    torch.save(weights, artifact_path)
    report = {
        'scope': ('Real MediaPipe features from presegmented GSL video. Frozen fresh test with no reused earlier RGB '
                  'test clips; earlier unknown clips remain diagnostic. No fresh webcam or free conversation test.'
                  if fresh_references is not None else 'Real MediaPipe features from presegmented GSL video. The held-out '
                  'recordings were evaluated in previous experiments and are reused diagnostics. No fresh webcam or free conversation test.'),
        'sources': {path.as_posix(): digest(path) for path in sources},
        'real_clips': len(all_items), 'roles': {role: {'clips': len(items), 'signers': sorted(signer_sets[role])}
                                              for role, items in roles.items()},
        'labels': labels[1:], 'novel_templates': sorted({item.glosses for item in novel}), 'model_config': config,
        'selection': 'Minimum development gloss WER, maximum development exact sequences, minimum train CTC loss. '
                     'Threshold then selected from development predictions, with any emission on a novel development '
                     'sequence counted as wrong even if Greek strings match.',
        'weak_supervision': 'No official local token boundaries. The auxiliary heads use real utterance gloss '
                            'presence and contrast original versus reversed label order for the same real frames.',
        'candidate_artifact': {'file': artifact_path.as_posix(), 'sha256': digest(artifact_path)},
        'outcomes': outcomes, 'selective_guard': selective,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, item in outcomes.items():
        print(name, {group: {key: stats[key] for key in ('gloss_wer', 'gloss_sequence_exact', 'greek_reference_exact', 'accepted')}
                     for group, stats in item['cohorts'].items()}, flush=True)
    name = selective['selected_on_development']
    guard = selective['variants'][name]
    print('selective', name, guard['policy'], {key: (metrics['correct_text'], metrics['wrong_or_out_of_scope_text'])
                                            for key, metrics in guard['test'].items()}, flush=True)


if __name__ == '__main__':
    main()
