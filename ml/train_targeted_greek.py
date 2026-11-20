"""Train a bounded Greek appointment decoder backed by verified paired examples."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from continuous_selective import tracked_hand_motion
from train_continuous import edit_counts
from train_continuous_hard import read_selected
from train_continuous_library import annotations, datasets, normalize_greek
from train_grounded_greek import CONTROLS, MIN_MOTION, MIN_TRACKING, alter, encode, load_visual, word_error_summary, write
from train_greek_translation_pilot import greek, sha


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path('data/appointment-targeted')
SECONDARY = Path('data/appointment-targeted-internist')
BASE = Path('data/appointment-unseen')
VISUAL = Path('reports/grounded-greek/appointment-unseen-freeze.json')
REPORT = Path('reports/appointment-targeted-greek')
ARTIFACT = Path('artifacts/appointment-targeted-greek')
FREEZE = REPORT / 'freeze.json'
ARMS = ('baseline', 'targeted')
SEED, BATCH, WIDTH = 923, 16, 96
FUNCTION_WORDS = frozenset('και που του την των τον από για στο στη στα τις μας σας μου της σας αυτό αυτή είναι'.split())


def content_words(text):
    return {word for word in greek(text).split() if (len(word) >= 3 or word.isdigit()) and word not in FUNCTION_WORDS}


class PhraseHead(nn.Module):
    def __init__(self, input_width, phrases, words):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, WIDTH), nn.GELU())
        self.weight = nn.Linear(WIDTH, 1)
        self.embedding = nn.Sequential(nn.Linear(3 * WIDTH, WIDTH), nn.LayerNorm(WIDTH), nn.GELU())
        self.dropout = nn.Dropout(.2)
        self.phrase = nn.Linear(WIDTH, phrases + 1)
        self.words = nn.Linear(WIDTH, words)

    def forward(self, memory, lengths):
        features = self.project(memory)
        valid = torch.arange(memory.shape[1])[None, :] < lengths[:, None]
        weights = self.weight(features).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        pooled = torch.cat(((weights[..., None] * features).sum(1),
                            features.masked_fill(~valid[..., None], -torch.inf).amax(1),
                            (features * valid[..., None]).sum(1) / lengths[:, None]), dim=-1)
        embedding = self.embedding(pooled)
        hidden = self.dropout(embedding)
        return self.phrase(hidden), self.words(hidden), F.normalize(embedding, dim=-1)


def head_outputs(head, memories):
    return head(pad_sequence(memories, batch_first=True), torch.tensor([len(item) for item in memories]))


def verify_manifest(manifest):
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    if not raw.get('complete'):
        raise ValueError('Targeted paired recordings must be fully verified before use')
    canonical, reference = annotations()
    expected = {'train': {'1', '2', '4', '5'}, 'val': {'3'}, 'test': {'6', '7'}}
    for row in raw['samples']:
        source = row['source_path'].split('/', 1)[-1]
        if (str(row['signer']) not in expected[row['split']] or canonical.get(source) != tuple(row['glosses'])
                or normalize_greek(reference.get(source, '')) != normalize_greek(row['text'])):
            raise ValueError('An appointment turn differs from its official paired reference or signer split')
    return raw


def split_rows(manifest, features):
    raw = verify_manifest(manifest)
    rows, digest = read_selected(manifest, features, ('train', 'val'))
    if any(row.fps != 15 for row in rows['train'] + rows['val']):
        raise ValueError('Targeted appointment landmarks do not use the same cadence')
    return raw, rows['train'], rows['val'], digest


def compatible_training(base, extra, held):
    held_ids = {row['id'] for row in held}
    kept = {}
    for row in [*base, *extra]:
        if row.id in held_ids or row.signer not in {'1', '2', '4', '5'}:
            raise ValueError('A development or held signer or recording entered targeted fitting')
        if row.source in kept:
            old = kept[row.source]
            if (old.id, old.text, old.glosses) != (row.id, row.text, row.glosses):
                raise ValueError('The same recorded source has conflicting paired evidence')
        else:
            kept[row.source] = row
    return list(kept.values())


@torch.no_grad()
def infer_head(head, memories):
    groups = [head_outputs(head, memories[index:index + BATCH]) for index in range(0, len(memories), BATCH)]
    return tuple(torch.cat(parts) for parts in zip(*groups, strict=True))


def fit_head(encoder, train, dev, budget, interval):
    torch.manual_seed(SEED)
    generator = random.Random(SEED)
    phrases = [None, *sorted({greek(row.text) for row in train})]
    words = sorted(set().union(*(content_words(row.text) for row in train)))
    phrase_ids = {text: i for i, text in enumerate(phrases)}
    word_ids = {word: i for i, word in enumerate(words)}
    targets = torch.tensor([phrase_ids[greek(row.text)] for row in train])
    lexical = torch.zeros((len(train), len(words)))
    for index, row in enumerate(train):
        lexical[index, [word_ids[word] for word in content_words(row.text)]] = 1.
    phrase_weights = torch.bincount(targets, minlength=len(phrases)).float().clamp_min(1).rsqrt()
    word_weights = ((len(train) - lexical.sum(0)) / lexical.sum(0)).sqrt().clamp(1., 12.)
    clips = [row.frames for row in train]
    memory = encode(encoder, clips)
    static = encode(encoder, [alter(clip, 'static') for clip in clips])
    development = encode(encoder, [row.frames for row in dev])
    dev_target = [greek(row.text) for row in dev]
    input_width = memory[0].shape[1]
    head = PhraseHead(input_width, len(phrases) - 1, len(words))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    best, history, updates = None, [], 0
    while updates < budget:
        order = generator.sample(range(len(train)), len(train))
        for start in range(0, len(train), BATCH):
            head.train()
            batch = order[start:start + BATCH]
            whole, part, _ = head_outputs(head, [memory[index] for index in batch])
            negative, blank, _ = head_outputs(head, [static[index] for index in batch])
            loss = F.cross_entropy(whole, targets[batch], weight=phrase_weights)
            loss += .5 * F.binary_cross_entropy_with_logits(part, lexical[batch], pos_weight=word_weights)
            loss += .25 * (-negative.log_softmax(-1)[:, 0].mean() + F.softplus(blank).mean())
            if not bool(loss.isfinite()):
                raise ValueError('Targeted appointment loss became nonfinite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 4.)
            optimizer.step()
            updates += 1
            if updates % interval == 0:
                head.eval()
                logits, lexical_scores, _ = infer_head(head, development)
                best_phrase = [phrases[index] or '' for index in logits.argmax(1).tolist()]
                exact = sum(actual == expected for actual, expected in zip(best_phrase, dev_target, strict=True))
                word_hits = sum(words[index] in content_words(expected) for index, expected in
                                zip(lexical_scores.argmax(1).tolist(), dev_target, strict=True))
                edits = sum(sum(edit_counts(expected.split(), actual.split())) for actual, expected in zip(best_phrase, dev_target, strict=True))
                key = exact, word_hits, -edits
                history.append({'updates': updates, 'dev_exact': exact, 'dev_first_keyword_correct': word_hits, 'dev_edits': edits})
                if best is None or key > best[0]:
                    best = key, updates, {name: value.detach().clone() for name, value in head.state_dict().items()}
                    print(json.dumps(history[-1]), flush=True)
            if updates >= budget:
                break
    head.load_state_dict(best[2])
    head.eval()
    bank = infer_head(head, memory)[2]
    return head, input_width, phrases, words, bank, targets, best[1], history


@dataclass
class TargetedDecoder:
    visual: nn.Module
    head: PhraseHead
    phrases: list[str | None]
    words: list[str]
    exemplar_vectors: torch.Tensor
    exemplar_targets: torch.Tensor
    exemplar_ids: list[str]
    policy: dict

    @torch.inference_mode()
    def inspect(self, clips):
        if not clips:
            return []
        originals = infer_head(self.head, encode(self.visual, clips))
        controls = [infer_head(self.head, encode(self.visual, [alter(clip, name) for clip in clips])) for name in CONTROLS]
        whole, lexical, vectors = originals
        scores = whole.softmax(1)
        partial_scores = lexical.sigmoid()
        worst = torch.stack([item[0].softmax(1) for item in controls]).amax(0)
        worst_partial = torch.stack([item[1].sigmoid() for item in controls]).amax(0)
        best = scores.topk(2, dim=1)
        example_texts = [content_words(self.phrases[index]) for index in self.exemplar_targets.tolist()]
        output = []
        for i, clip in enumerate(clips):
            choice = int(best.indices[i, 0])
            quality = (clip[:, 222] >= .75) & (clip[:, 220:222] >= .75).any(1)
            sample_id = None
            if choice:
                candidates = self.exemplar_targets == choice
                nearest = (self.exemplar_vectors[candidates] @ vectors[i]).argmax()
                sample_id = self.exemplar_ids[int(torch.where(candidates)[0][nearest])]
            raw_parts = []
            for index in partial_scores[i].argsort(descending=True)[:4].tolist():
                word = self.words[index]
                matches = torch.tensor([j for j, sample in enumerate(example_texts) if word in sample])
                nearest = matches[(self.exemplar_vectors[matches] @ vectors[i]).argmax()]
                raw_parts.append({'word': word, 'score': float(partial_scores[i, index]),
                                  'gain': float(partial_scores[i, index] - worst_partial[i, index]),
                                  'example_id': self.exemplar_ids[int(nearest)]})
            output.append({'raw_text': self.phrases[choice] or '', 'confidence': float(scores[i, choice]),
                           'gain': float(scores[i, choice] - worst[i, choice]),
                           'margin': float(best.values[i, 0] - best.values[i, 1]),
                           'quality': float(quality.float().mean()), 'motion': tracked_hand_motion(clip),
                           'example_id': sample_id, 'raw_partials': raw_parts})
        return output

    def predict_dialogue(self, clips):
        output = []
        for row in self.inspect(clips):
            full = full_accept(row, self.policy['full'])
            partials = [] if full else partial_accept(row, self.policy['partial'])
            output.append({**row, 'text': row['raw_text'] if full else '', 'accepted': full, 'partials': partials,
                           'terminated': True, 'glosses': ()})
        return output


def adequate(row):
    return row['quality'] >= MIN_TRACKING and row['motion'] >= MIN_MOTION


def full_accept(row, policy):
    return bool(row['raw_text'] and adequate(row) and row['confidence'] >= policy['score'] and row['gain'] >= policy['gain'])


def partial_accept(row, policy):
    if not adequate(row):
        return []
    return [part for part in row['raw_partials'] if part['score'] >= policy['score'] and part['gain'] >= policy['gain']]


def choose_policy(actual, references, controls):
    all_rows = [*actual, *(row for group in controls.values() for row in group)]
    text = [*references, *[''] * (len(all_rows) - len(actual))]
    lexical = [content_words(reference) for reference in text]
    best_full, best_partial = None, None
    for score in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.01):
        for gain in (.005, .025, .05, .10, .20, .35, .5, .75, 1.01):
            policy = {'score': score, 'gain': gain}
            accepted = [index for index, row in enumerate(all_rows) if full_accept(row, policy)]
            if all(all_rows[index]['raw_text'] == text[index] for index in accepted):
                key = len(accepted), score, gain
                if best_full is None or key > best_full[0]:
                    best_full = key, policy
            partial = [partial_accept(row, policy) for row in all_rows]
            if all(part['word'] in expected for candidates, expected in zip(partial, lexical, strict=True) for part in candidates):
                key = sum(map(len, partial)), score, gain
                if best_partial is None or key > best_partial[0]:
                    best_partial = key, policy
    return {'full': best_full[1], 'partial': best_partial[1]}, {'correct_full': best_full[0][0],
                                                                  'correct_partial': best_partial[0][0],
                                                                  'controls': {key: len(rows) for key, rows in controls.items()}}


def train(manifest, features, epochs):
    if FREEZE.exists():
        raise ValueError('The targeted Greek trial has already been frozen')
    raw, extra, dev, feature_hash = split_rows(manifest, features)
    base, _, provenance = datasets(BASE / 'manifest.json', BASE / 'features.jsonl')
    held = [row for row in raw['samples'] if row['split'] != 'train']
    baseline = compatible_training(base, [], held)
    targeted = compatible_training(base, extra, held)
    frozen_visual = json.loads(VISUAL.read_text(encoding='utf-8'))['arms']['isolated_transfer']
    visual_path, visual_sha = Path(frozen_visual['vision_path']), frozen_visual['vision_sha256']
    visual, _ = load_visual(visual_path, visual_sha)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    interval = ceil(len(baseline) / BATCH)
    budget = epochs * interval
    arms = {}
    for arm, rows in zip(ARMS, (baseline, targeted), strict=True):
        print(arm, len(rows), 'paired turns', budget, 'updates', flush=True)
        head, width, phrases, words, bank, targets, update, history = fit_head(visual, rows, dev, budget, interval)
        ids = [row.id for row in rows]
        model = TargetedDecoder(visual, head, phrases, words, bank, targets, ids, {})
        clips = [row.frames for row in dev]
        raw_dev = model.inspect(clips)
        controls = {name: model.inspect([alter(clip, name) for clip in clips]) for name in CONTROLS}
        policy, dev_results = choose_policy(raw_dev, [greek(row.text) for row in dev], controls)
        path = ARTIFACT / f'{arm}.pt'
        torch.save({'state_dict': head.state_dict(), 'width': width, 'phrases': phrases, 'words': words,
                    'vectors': bank, 'targets': targets, 'ids': ids}, path)
        arms[arm] = {'path': path.as_posix(), 'sha256': sha(path), 'selected_update': update,
                     'development_history': history, 'development_gate': dev_results, 'policy': policy,
                     'training_turns': len(rows), 'reference_phrases': len(phrases) - 1, 'keyword_classes': len(words)}
    write(FREEZE, {'scope': 'Signer- and recording-disjoint appointment replay, allowing familiar exact phrase references. '
                            'Word fragments never become a composed sentence. Policies fitted only on development.',
                   'manifest_sha256': sha(manifest), 'train_dev_features_sha256': feature_hash, 'prior_provenance': provenance,
                   'visual_path': visual_path.as_posix(), 'visual_sha256': visual_sha, 'dev_turns': len(dev),
                   'target_train_turns': len(extra), 'budget_updates': budget, 'seed': SEED,
                   'fitted_role_counts': {'new_train': len(extra), 'development': len(dev)},
                   'manifest_role_counts': {role: sum(row['split'] == role for row in raw['samples'])
                                            for role in ('train', 'val', 'test')},
                   'signers_by_role': {role: sorted({str(row['signer']) for row in raw['samples'] if row['split'] == role})
                                       for role in ('train', 'val', 'test')}, 'arms': arms})
    print('Frozen targeted Greek', sha(FREEZE), flush=True)


def load_targeted_decoder(freeze_path=FREEZE, arm='targeted'):
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    record = freeze['arms'][arm]
    path = ROOT / record['path']
    if sha(path) != record['sha256']:
        raise ValueError('The frozen targeted Greek head changed')
    visual, _ = load_visual(ROOT / freeze['visual_path'], freeze['visual_sha256'])
    artifact = torch.load(path, weights_only=True, map_location='cpu')
    head = PhraseHead(artifact['width'], len(artifact['phrases']) - 1, len(artifact['words']))
    head.load_state_dict(artifact['state_dict'])
    return TargetedDecoder(visual, head.eval(), artifact['phrases'], artifact['words'], artifact['vectors'],
                           artifact['targets'], artifact['ids'], record['policy'])


def evaluate(manifest, features, secondary=False):
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if sha(manifest) != freeze['manifest_sha256']:
        raise ValueError('The frozen appointment manifest changed')
    raw, _, _, digest = split_rows(manifest, features)
    if digest != freeze['train_dev_features_sha256']:
        raise ValueError('The frozen appointment training or development landmarks changed')
    if secondary:
        manifest, features = SECONDARY / 'manifest.json', SECONDARY / 'features.jsonl'
        raw = verify_manifest(manifest)
        if any(row['split'] != 'test' for row in raw['samples']):
            raise ValueError('The secondary appointment can only provide held clips')
    held, held_digest = read_selected(manifest, features, ('test',))
    by_id = {row.id: row for row in held['test']}
    scenes = [row for row in raw['dialogues'] if row['split'] == 'test']
    if set(by_id) != {identity for scene in scenes for identity in scene['turn_ids']}:
        raise ValueError('Held appointment landmarks and chronological takes differ')
    results = {}
    for arm in ARMS:
        model = load_targeted_decoder(arm=arm)
        if set(by_id) & set(model.exemplar_ids):
            raise ValueError('A held appointment clip entered the verified phrase bank')
        report, references, proposals, final = [], [], [], []
        controls = {name: {'full': 0, 'partial_words': 0} for name in CONTROLS}
        for scene in scenes:
            items = [by_id[identity] for identity in scene['turn_ids']]
            clips = [row.frames for row in items]
            expected = [greek(row.text) for row in items]
            predictions = model.predict_dialogue(clips)
            actual = [row['text'] for row in predictions]
            for name in CONTROLS:
                guesses = model.predict_dialogue([alter(clip, name) for clip in clips])
                controls[name]['full'] += sum(row['accepted'] for row in guesses)
                controls[name]['partial_words'] += sum(len(row['partials']) for row in guesses)
            references.extend(expected)
            proposals.extend(row['raw_text'] for row in predictions)
            final.extend(actual)
            report.append({'id': scene['id'], 'turns': [{**guess, 'id': item.id, 'reference': reference}
                                                      for item, reference, guess in zip(items, expected, predictions, strict=True)],
                           'exact_full_dialogue': actual == expected,
                           'joined_text_word_edits': sum(edit_counts(' '.join(expected).split(), ' '.join(actual).split()))})
        correct = wrong = 0
        for scene in report:
            for turn in scene['turns']:
                supported = content_words(turn['reference'])
                for part in turn['partials']:
                    correct += int(part['word'] in supported)
                    wrong += int(part['word'] not in supported)
        result = {'raw': word_error_summary([tuple(text.split()) for text in references], [tuple(text.split()) for text in proposals]),
                  'guarded': word_error_summary([tuple(text.split()) for text in references], [tuple(text.split()) for text in final]),
                  'known_exact_references': sum(text in model.phrases for text in references),
                  'wrong_full': sum(bool(actual) and actual != expected for actual, expected in zip(final, references, strict=True)),
                  'correct_partial_words': correct, 'wrong_partial_words': wrong,
                  'exact_dialogues': sum(scene['exact_full_dialogue'] for scene in report),
                  'temporal_controls': controls, 'dialogues': report}
        if secondary:
            result['critical_word_diagnostics'] = {}
            for word in ('ραντεβού', 'παθολόγο', 'μεθαύριο', 'αίματος'):
                support = [turn for scene in report for turn in scene['turns'] if word in turn['reference'].split()]
                result['critical_word_diagnostics'][word] = {
                    'reference_turns': len(support),
                    'raw_exact_phrase': sum(turn['reference'] == turn['raw_text'] for turn in support),
                    'guarded_exact_phrase': sum(turn['reference'] == turn['text'] for turn in support),
                    'raw_word': sum(word in turn['raw_text'].split() for turn in support),
                    'guarded_word': sum(word in turn['text'].split() for turn in support),
                    'partial_word': sum(any(part['word'] == word for part in turn['partials']) for turn in support),
                    'rows': [{'id': turn['id'], 'reference': turn['reference'], 'raw': turn['raw_text'], 'guarded': turn['text'],
                              'partials': [part['word'] for part in turn['partials']]} for turn in support],
                }
        results[arm] = result
        print(json.dumps({'arm': arm, **{key: value for key, value in result.items() if key != 'dialogues'}}, ensure_ascii=False), flush=True)
    name = 'secondary-published-cuts.json' if secondary else 'published-cuts.json'
    scope = ('Secondary held signer-7 internist take. The «βεβαίως» clip was already used as a diagnostic in a prior project; '
             'it was outside both current fits. Thirteen other cuts are newly acquired. Official cuts supplied.' if secondary
             else 'New actual take from a held signer, with familiar reference phrases allowed. Official cuts supplied.')
    write(REPORT / name, {'freeze_sha256': sha(FREEZE), 'test_manifest_sha256': sha(manifest),
                           'test_landmarks_sha256': held_digest, 'scope': scope, 'results': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate'))
    parser.add_argument('--manifest', type=Path, default=TARGET / 'manifest.json')
    parser.add_argument('--features', type=Path, default=TARGET / 'features.jsonl')
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--secondary', action='store_true', help='Replay the separately verified held internist take')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Training epochs must be positive')
    if args.secondary and args.phase != 'evaluate':
        parser.error('--secondary is available only for held evaluation')
    torch.set_num_threads(2)
    if args.phase == 'train':
        train(args.manifest, args.features, args.epochs)
    else:
        evaluate(args.manifest, args.features, args.secondary)


if __name__ == '__main__':
    main()
