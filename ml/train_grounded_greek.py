"""Train matched Greek word CTC heads on frozen continuous GSL visual encoders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from continuous_selective import sequence_probability, tracked_hand_motion
from preprocess import FEATURE_COUNT
from train_continuous import ctc_decode, edit_counts
from train_continuous_hard import read_selected
from train_continuous_multitask import OrderAwareCTC
from train_greek_translation_pilot import greek, sha


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('data/appointment-unseen')
VISUAL_REPORT = Path('reports/continuous-library-ctc/freeze.json')
VISUAL_ARTIFACTS = Path('artifacts/continuous-library-ctc')
REPORT = Path('reports/grounded-greek')
ARTIFACTS = Path('artifacts/grounded-greek')
FREEZE = REPORT / 'appointment-unseen-freeze.json'
BENCHMARK = REPORT / 'appointment-unseen-translation.json'
ARMS = ('continuous_only', 'isolated_transfer')
CONTROLS = ('static', 'reversed', 'shuffled')
BATCH = 12
SEED = 1904
STATIC_LOSS_WEIGHT = .25
MIN_TRACKING = .60
MIN_MOTION = .004


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


class WordHead(nn.Module):
    def __init__(self, width, vocabulary):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 96), nn.GELU(), nn.Dropout(.15), nn.Linear(96, vocabulary))

    def forward(self, memory):
        return self.layers(memory).log_softmax(-1)


def load_visual(path, expected):
    if sha(path) != expected:
        raise ValueError('The frozen continuous visual checkpoint changed')
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    labels, weights = checkpoint['labels'], checkpoint['state_dict']
    model = OrderAwareCTC(len(labels), weights['mean'], weights['scale'])
    model.load_state_dict(weights)
    return model.eval(), labels


@torch.no_grad()
def encode(encoder, clips):
    memories = []
    for start in range(0, len(clips), BATCH):
        group = clips[start:start + BATCH]
        if any(clip.ndim != 2 or clip.shape[1] != FEATURE_COUNT or len(clip) < 2 or not clip.isfinite().all() for clip in group):
            raise ValueError('Grounded Greek expects nonempty 15 Hz, 224-value landmark clips')
        lengths = torch.tensor([len(clip) for clip in group])
        embedded = encoder.encode(pad_sequence(group, batch_first=True), lengths)[1]
        memories.extend(memory[:length].clone() for memory, length in zip(embedded, lengths.tolist(), strict=True))
    return memories


def alter(frames, mode):
    if mode == 'static':
        return frames[len(frames) // 2].expand_as(frames)
    if mode == 'reversed':
        return frames.flip(0)
    generator = torch.Generator().manual_seed(SEED + len(frames))
    return frames[torch.randperm(len(frames), generator=generator)]


def decode_word_sequences(head, memories, words):
    decoded = []
    for start in range(0, len(memories), BATCH):
        group = memories[start:start + BATCH]
        for memory, scores in zip(group, head(pad_sequence(group, batch_first=True)), strict=True):
            decoded.append(ctc_decode(scores[:len(memory)], words)[0])
    return decoded


def word_error_summary(references, guesses):
    metrics = [edit_counts(reference, guess) for reference, guess in zip(references, guesses, strict=True)]
    substitutions, deletions, insertions = map(sum, zip(*metrics, strict=True))
    words = sum(map(len, references))
    return {'turns': len(references), 'reference_words': words, 'word_edits': substitutions + deletions + insertions,
            'substitutions': substitutions, 'deletions': deletions, 'insertions': insertions,
            'word_error_rate': (substitutions + deletions + insertions) / words,
            'exact_turns': sum(reference == guess for reference, guess in zip(references, guesses, strict=True)),
            'nonempty_turns': sum(bool(guess) for guess in guesses)}


def fit_head(visual, train, dev, words, epochs):
    torch.manual_seed(SEED)
    generator = random.Random(SEED)
    clips = [row.frames for row in train]
    train_memory = encode(visual, clips)
    static_memory = encode(visual, [alter(clip, 'static') for clip in clips])
    dev_memory = encode(visual, [row.frames for row in dev])
    vocabulary = {word: index for index, word in enumerate(words) if index}
    references = [tuple(greek(row.text).split()) for row in dev]
    targets = [torch.tensor([vocabulary[word] for word in greek(row.text).split()]) for row in train]
    for clip, target in zip(clips, targets, strict=True):
        needed = len(target) + int((target[1:] == target[:-1]).sum())
        if not len(target) or len(clip) < needed:
            raise ValueError('A Greek training turn cannot be aligned at the visual frame rate')
    width = train_memory[0].shape[1]
    head = WordHead(width, len(words))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004)
    best, history = None, []
    for epoch in range(1, epochs + 1):
        head.train()
        order = generator.sample(range(len(train)), len(train))
        losses = []
        for start in range(0, len(order), BATCH):
            indexes = order[start:start + BATCH]
            actual = [train_memory[index] for index in indexes]
            negatives = [static_memory[index] for index in indexes if len(targets[index]) >= 2]
            labels = [targets[index] for index in indexes]
            ctc = F.ctc_loss(head(pad_sequence(actual, batch_first=True)).transpose(0, 1), torch.cat(labels),
                             [len(item) for item in actual], [len(target) for target in labels])
            if negatives:
                blank = -head(torch.cat(negatives))[:, 0].mean()
                loss = ctc + STATIC_LOSS_WEIGHT * blank
            else:
                loss = ctc
            if not bool(loss.isfinite()):
                raise ValueError('Nonfinite Greek word CTC loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 4.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 and epoch not in (1, epochs):
            continue
        head.eval()
        with torch.no_grad():
            dev_metrics = word_error_summary(references, decode_word_sequences(head, dev_memory, words))
        key = (dev_metrics['word_edits'], -dev_metrics['exact_turns'], dev_metrics['insertions'])
        history.append({'epoch': epoch, 'loss': sum(losses) / len(losses), 'development_raw': dev_metrics})
        if best is None or key < best[0]:
            best = key, epoch, {name: value.detach().clone() for name, value in head.state_dict().items()}
            print(json.dumps({'epoch': epoch, 'development_raw': dev_metrics}), flush=True)
    head.load_state_dict(best[2])
    return head.eval(), width, best[1], history


def accepts(row, policy):
    return bool(row['raw_text'] and row['glosses'] and row['tracking_quality'] >= MIN_TRACKING
                and row['hand_motion'] >= MIN_MOTION and row['confidence'] >= policy['minimum_confidence']
                and row['visual_gain'] >= policy['minimum_visual_gain'])


@dataclass
class GroundedDecoder:
    encoder: OrderAwareCTC
    labels: list[str | None]
    word_head: WordHead
    words: list[str | None]
    policy: dict[str, float]

    @torch.inference_mode()
    def inspect(self, clips):
        base = encode(self.encoder, clips)
        altered = {name: encode(self.encoder, [alter(clip, name) for clip in clips]) for name in CONTROLS}
        word_ids = {word: index for index, word in enumerate(self.words) if index}
        predictions = []
        for index, (clip, memory) in enumerate(zip(clips, base, strict=True)):
            logits = self.word_head(memory)
            words, confidence = ctc_decode(logits, self.words)
            glosses, gloss_confidence = ctc_decode(self.encoder.ctc(memory).log_softmax(-1), self.labels)
            tracking = (clip[:, 222] >= .75) & (clip[:, 220:222] >= .75).any(1)
            gain = None
            if words:
                target = torch.tensor([word_ids[word] for word in words])
                log_probability = sequence_probability(logits, target)
                rival = max(sequence_probability(self.word_head(altered[name][index]), target) for name in CONTROLS)
                gain = (log_probability - rival) / len(clip)
            predictions.append({'raw_text': ' '.join(words), 'glosses': glosses, 'confidence': confidence,
                                'gloss_confidence': gloss_confidence, 'tracking_quality': float(tracking.float().mean()),
                                'hand_motion': tracked_hand_motion(clip), 'visual_gain': gain})
        return predictions

    def predict_dialogue(self, clips):
        """Decode ordered landmark segments and expose guarded and raw text separately."""
        outcomes = []
        for row in self.inspect(clips):
            accepted = accepts(row, self.policy)
            outcomes.append({**row, 'text': row['raw_text'] if accepted else '', 'terminated': True, 'accepted': accepted})
        return outcomes


def choose_policy(positive, references, controls):
    rows = [*positive, *(item for group in controls.values() for item in group)]
    correct = [row['raw_text'] == text for row, text in zip(positive, references, strict=True)] + [False] * (len(rows) - len(positive))
    minimum_confidence = sorted({0., .4, .55, .7, .85, .95, 1.01, *(row['confidence'] for row in rows)})
    minimum_gain = sorted({.001, .01, .025, .05, .1, .2, .4, 1e6, *(row['visual_gain'] for row in rows
                                                                  if row['visual_gain'] is not None and row['visual_gain'] > 0)})
    best = None
    for confidence in minimum_confidence:
        for gain in minimum_gain:
            policy = {'minimum_confidence': confidence, 'minimum_visual_gain': gain}
            accepted = [index for index, row in enumerate(rows) if accepts(row, policy)]
            if all(correct[index] for index in accepted):
                score = len(accepted), confidence, gain
                if best is None or score > best[0]:
                    best = score, policy
    return best[1], {'accepted_correct': best[0][0], 'actual_turns': len(positive),
                     'raw_correct': sum(correct), 'control_turns': {name: len(group) for name, group in controls.items()},
                     'raw_control_nonempty': {name: sum(bool(row['raw_text']) for row in group) for name, group in controls.items()}}


def assert_roles(train, dev, manifest):
    source = json.loads(manifest.read_text(encoding='utf-8'))
    protected = [row for row in source['samples'] if row['split'] != 'train']
    reserved_scenes = {row['source_path'].split('/', 1)[0] for row in protected}
    reserved_signers = {str(row['signer']) for row in protected}
    reserved_text = {greek(row['text']) for row in protected}
    reserved_gloss = {tuple(row['glosses']) for row in protected}
    if any(row.source.split('/', 1)[0] in reserved_scenes or row.signer in reserved_signers
           or greek(row.text) in reserved_text or row.glosses in reserved_gloss for row in train):
        raise ValueError('A protected appointment scene, signer, exact Greek text or gloss order entered the text fit')
    if {row.id for row in dev} != {row['id'] for row in protected if row['split'] == 'val'}:
        raise ValueError('Greek model development differs from the frozen appointment set')
    if len({row.id for row in train}) != len(train) or not all(greek(row.text) for row in train + dev):
        raise ValueError('Duplicate or incomplete approved Greek translation rows')


def train(manifest, features, epochs):
    if FREEZE.exists():
        raise ValueError('Grounded Greek weights already frozen')
    from train_continuous_library import datasets
    visual = json.loads(VISUAL_REPORT.read_text(encoding='utf-8'))
    train_rows, dev, provenance = datasets(manifest, features)
    if json.loads(json.dumps(provenance)) != visual['provenance']:
        raise ValueError('Visual and Greek text stage training inputs differ')
    assert_roles(train_rows, dev, manifest)
    words = [None, *sorted({word for row in train_rows for word in greek(row.text).split()})]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ARMS:
        print(arm, len(train_rows), 'verified Greek training turns', flush=True)
        visual_path = VISUAL_ARTIFACTS / f'{arm}.pt'
        visual_sha = visual['arms'][arm]['checkpoint_sha256']
        encoder, labels = load_visual(visual_path, visual_sha)
        head, width, epoch, history = fit_head(encoder, train_rows, dev, words, epochs)
        decoder = GroundedDecoder(encoder, labels, head, words, {})
        clips = [row.frames for row in dev]
        positive = decoder.inspect(clips)
        controls = {name: decoder.inspect([alter(clip, name) for clip in clips]) for name in CONTROLS}
        policy, dev_result = choose_policy(positive, [greek(row.text) for row in dev], controls)
        path = ARTIFACTS / f'appointment-unseen-{arm}.pt'
        torch.save({'weights': head.state_dict(), 'width': width, 'words': words, 'policy': policy}, path)
        arms[arm] = {'path': path.as_posix(), 'sha256': sha(path), 'vision_path': visual_path.as_posix(),
                     'vision_sha256': visual_sha, 'selected_epoch': epoch, 'policy': policy, 'development_gate': dev_result,
                     'development_history': history}
    write(FREEZE, {'scope': 'Greek word CTC on vetted paired visual turns. Static frames are synthetic negative controls. '
                            'Select word weights and guarded emission floors using development only.',
                   'manifest_sha256': sha(manifest), 'visual_freeze_sha256': sha(VISUAL_REPORT), 'provenance': provenance,
                   'train_ids_sha256': sha256('\n'.join(row.id for row in train_rows).encode()).hexdigest(),
                   'train_turns': len(train_rows), 'development_turns': len(dev), 'greek_word_classes': len(words) - 1,
                   'seed': SEED, 'arms': arms})
    print('Frozen grounded Greek checkpoint', sha(FREEZE), flush=True)


def load_grounded_decoder(freeze_path=FREEZE, arm='isolated_transfer'):
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    record = freeze['arms'][arm]
    path = ROOT / record['path']
    if sha(path) != record['sha256']:
        raise ValueError('Frozen Greek word checkpoint changed')
    encoder, labels = load_visual(ROOT / record['vision_path'], record['vision_sha256'])
    word = torch.load(path, weights_only=True, map_location='cpu')
    if word['policy'] != record['policy']:
        raise ValueError('Frozen Greek evidence gate differs from checkpoint')
    head = WordHead(word['width'], len(word['words']))
    head.load_state_dict(word['weights'])
    return GroundedDecoder(encoder, labels, head.eval(), word['words'], word['policy'])


def evaluate(manifest, features):
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if sha(manifest) != freeze['manifest_sha256'] or sha(VISUAL_REPORT) != freeze['visual_freeze_sha256']:
        raise ValueError('Frozen appointment or visual source changed')
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    held, held_hash = read_selected(manifest, features, ('test',))
    by_id = {row.id: row for row in held['test']}
    scripts = [scene for scene in raw['dialogues'] if scene['split'] == 'test']
    if set(by_id) != {identifier for scene in scripts for identifier in scene['turn_ids']}:
        raise ValueError('Held cut order and features differ')
    results = {}
    for arm in ARMS:
        decoder = load_grounded_decoder(arm=arm)
        grouped = []
        raw_references, raw_outputs, emitted = [], [], []
        control_emissions = {name: {'accepted': 0, 'raw_nonempty': 0} for name in CONTROLS}
        vocabulary = set(decoder.words[1:])
        for scene in scripts:
            rows = [by_id[identifier] for identifier in scene['turn_ids']]
            clips = [row.frames for row in rows]
            guesses = decoder.predict_dialogue(clips)
            references = [tuple(greek(row.text).split()) for row in rows]
            raw_texts = [tuple(guess['raw_text'].split()) for guess in guesses]
            texts = [tuple(guess['text'].split()) for guess in guesses]
            raw_references.extend(references)
            raw_outputs.extend(raw_texts)
            emitted.extend(texts)
            for name in CONTROLS:
                controlled = decoder.predict_dialogue([alter(clip, name) for clip in clips])
                control_emissions[name]['accepted'] += sum(row['accepted'] for row in controlled)
                control_emissions[name]['raw_nonempty'] += sum(bool(row['raw_text']) for row in controlled)
            grouped.append({'id': scene['id'], 'scenario': scene['scenario'], 'signer': scene['signer'],
                            'raw_full_text_exact': references == raw_texts, 'guarded_full_text_exact': references == texts,
                            'raw_full_text_word_edits': sum(edit_counts(sum(references, ()), sum(raw_texts, ()))),
                            'guarded_full_text_word_edits': sum(edit_counts(sum(references, ()), sum(texts, ()))),
                            'turns': [{**guess, 'id': row.id, 'reference': greek(row.text), 'reference_glosses': row.glosses,
                                       'unknown_reference_words': sorted(set(greek(row.text).split()) - vocabulary)}
                                      for row, guess in zip(rows, guesses, strict=True)]})
        turn_predictions = [row for scene in grouped for row in scene['turns']]
        signs = word_error_summary([tuple(row['reference_glosses']) for row in turn_predictions],
                                   [tuple(row['glosses']) for row in turn_predictions])
        results[arm] = {'raw': word_error_summary(raw_references, raw_outputs),
                        'guarded': word_error_summary(raw_references, emitted), 'dialogues': grouped,
                        'recognition': {'reference_glosses': signs['reference_words'], 'gloss_edits': signs['word_edits'],
                                        'gloss_error_rate': signs['word_error_rate'], 'exact_turns': signs['exact_turns']},
                        'word_vocabulary': {'reference_words': sum(len(words) for words in raw_references),
                                            'known_reference_words': sum(word in vocabulary for words in raw_references for word in words),
                                            'entirely_known_turns': sum(set(words) <= vocabulary for words in raw_references)},
                        'accepted_wrong': sum(bool(actual) and reference != actual for reference, actual in zip(raw_references, emitted, strict=True)),
                        'raw_exact_dialogues': sum(row['raw_full_text_exact'] for row in grouped),
                        'guarded_exact_dialogues': sum(row['guarded_full_text_exact'] for row in grouped),
                        'raw_full_text_word_edits': sum(row['raw_full_text_word_edits'] for row in grouped),
                        'guarded_full_text_word_edits': sum(row['guarded_full_text_word_edits'] for row in grouped),
                        'temporal_controls': control_emissions}
        print(json.dumps({'arm': arm, 'raw': results[arm]['raw'], 'guarded': results[arm]['guarded']}, ensure_ascii=False), flush=True)
    write(BENCHMARK, {'scope': 'Two held official appointment takes, with annotation-provided cuts. '
                              'Guarded outputs and all raw proposals are recorded separately. No semantic review.',
                      'freeze_sha256': sha(FREEZE), 'held_landmarks_sha256': held_hash, 'results': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate'))
    parser.add_argument('--manifest', type=Path, default=DATA / 'manifest.json')
    parser.add_argument('--features', type=Path, default=DATA / 'features.jsonl')
    parser.add_argument('--epochs', type=int, default=45)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Training epochs must be positive')
    torch.set_num_threads(2)
    if args.phase == 'train':
        train(args.manifest, args.features, args.epochs)
    else:
        evaluate(args.manifest, args.features)


if __name__ == '__main__':
    main()
