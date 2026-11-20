"""Train a text-only decoder on the official CERTH gloss and Greek annotation pairs."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from train_continuous import edit_counts
from train_greek_translation_pilot import greek, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from plan_continuous_composition import build_templates, normalize_greek, read_annotations


DATA = ROOT / 'data'
REPORT = ROOT / 'reports/greek-translation-pilot-context/official-gloss-to-greek.json'
FREEZE = ROOT / 'reports/greek-translation-pilot-context/official-gloss-split.json'
CHECKPOINT = ROOT / 'artifacts/greek-translation-pilot-context/official-gloss-to-greek.pt'
SEED = 431
PAD, START, END, UNKNOWN = 0, 1, 2, 3
HIDDEN = 64


def nearby(left, right):
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    if len(right) != len(left) + 1:
        return False
    offset = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), len(left))
    return left[offset:] == right[offset + 1:]


def signature(glosses, text):
    return tuple(glosses), tuple(normalize_greek(text).split())


def near_reference(key, other):
    return nearby(key[0], other[0]) or nearby(key[1], other[1])


def reference_files():
    exposed, reserved = set(), set()
    sources = sorted(DATA.glob('continuous-*/manifest.json'))
    for path in sources:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for row in manifest['samples']:
            key = signature(row['glosses'], row['text'])
            if not key[0] or not key[1]:
                continue
            exposed.add(key)
            if row['split'] in ('test', 'val'):
                reserved.add(key)
    return exposed, reserved, sources


def split_templates():
    files = [DATA / 'original_annotations' / name for name in ('GSL_split.zip', 'supplementary.zip')]
    templates, inventory = build_templates(*read_annotations(*files), min_repeats=1, max_glosses=24)
    unique = {}
    for item in templates:
        key = signature(item.glosses, item.greek)
        if key[0] and key[1] and greek(item.greek):
            unique.setdefault(key, item)
    exposed, reserved, prior = reference_files()
    pool = {key: item for key, item in unique.items() if not any(near_reference(key, known) for known in reserved)}
    exposed_gloss = {key[0] for key in exposed}
    exposed_greek = {key[1] for key in exposed}
    candidate = {key: item for key, item in pool.items()
                 if key[0] not in exposed_gloss and key[1] not in exposed_greek}

    # The fixed hash orders scenes without inspecting model outputs or choosing easy texts.
    scenarios = sorted({item.scenario for item in candidate.values()},
                       key=lambda name: sha256(f'{SEED}/{name}'.encode()).hexdigest())
    development, test = {}, {}
    for name in scenarios:
        group = {key: item for key, item in candidate.items() if item.scenario == name
                 and not any(near_reference(key, other) for other in development)}
        if len(group) >= 8:
            if not development:
                development = group
            else:
                test = group
                break
    if not development or not test:
        raise ValueError('No two scenario holdouts satisfy the minimum template count')
    held_scenes = {item.scenario for item in (*development.values(), *test.values())}
    held = set(development) | set(test)
    train = {key: item for key, item in pool.items() if item.scenario not in held_scenes
             and not any(near_reference(key, other) for other in held)}
    if not train:
        raise ValueError('All official annotation pairs were excluded from training')
    if any(near_reference(key, other) for key in train for other in reserved | held):
        raise ValueError('An exact or near-reference overlap crossed into training')

    roles = {'train': list(train.values()), 'development': list(development.values()), 'test': list(test.values())}
    inventory.update({'deduped_pairs': len(unique), 'eligible_after_prior_holdouts': len(pool),
                      'prior_holdout_pairs': len(reserved), 'excluded_near_prior_holdouts': len(unique) - len(pool)})
    proof = {'seed': SEED, 'inventory': inventory, 'holdout_scenarios': sorted(held_scenes),
             'sources': {path.relative_to(ROOT).as_posix(): sha(path) for path in files + prior},
             'exclusion': 'No training pair is within one token edit in gloss order or stress-insensitive Greek text '
                          'of any previous evaluation pair or either new held scenario. New evaluation labels were '
                          'previously unseen as exact sequences or normalized references in archived clip manifests.',
             'roles': {role: [{'template': item.key, 'glosses': item.glosses, 'text': item.greek,
                               'agreeing_records': len(item.sources), 'official_records': item.annotation_count}
                              for item in rows] for role, rows in roles.items()}}
    return roles, proof


class GlossTranslator(nn.Module):
    def __init__(self, n_gloss, n_character):
        super().__init__()
        self.gloss_embedding = nn.Embedding(n_gloss, HIDDEN, padding_idx=PAD)
        self.encoder = nn.GRU(HIDDEN, HIDDEN, bidirectional=True, batch_first=True)
        self.decoder_embedding = nn.Embedding(n_character, HIDDEN, padding_idx=PAD)
        self.decoder = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        self.init = nn.Linear(2 * HIDDEN, HIDDEN)
        self.keys = nn.Linear(2 * HIDDEN, HIDDEN)
        self.classify = nn.Linear(3 * HIDDEN, n_character)

    def encode(self, glosses):
        lengths = glosses.ne(PAD).sum(1)
        packed = pack_padded_sequence(self.gloss_embedding(glosses), lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, state = self.encoder(packed)
        memory, _ = pad_packed_sequence(encoded, batch_first=True)
        first = self.init(torch.cat(tuple(state), dim=1)).tanh().unsqueeze(0)
        return memory, self.keys(memory), glosses.ne(PAD), first

    def decode(self, chars, state, memory, keys, valid):
        query, state = self.decoder(self.decoder_embedding(chars), state)
        attention = (query @ keys.transpose(1, 2) / HIDDEN ** 0.5).masked_fill(~valid[:, None], -torch.inf)
        context = attention.softmax(-1) @ memory
        return self.classify(torch.cat([query, context], dim=-1)), state


def tensor_rows(rows, vocab, field):
    ids = []
    for row in rows:
        if field == 'gloss':
            tokens = [vocab.get(token, UNKNOWN) for token in row.glosses]
        else:
            tokens = [START] + [vocab.get(token, UNKNOWN) for token in greek(row.greek)] + [END]
        ids.append(torch.tensor(tokens))
    return pad_sequence(ids, batch_first=True, padding_value=PAD)


@torch.inference_mode()
def generate(model, rows, gloss_vocab, characters):
    model.eval()
    memory, keys, valid, state = model.encode(tensor_rows(rows, gloss_vocab, 'gloss'))
    cursor = torch.full((len(rows), 1), START)
    ended = torch.zeros(len(rows), dtype=torch.bool)
    texts = [[] for _ in rows]
    for _ in range(120):
        logits, state = model.decode(cursor, state, memory, keys, valid)
        logits[:, :, [PAD, START, UNKNOWN]] = -torch.inf
        next_character = logits[:, -1].argmax(-1)
        for i, token in enumerate(next_character.tolist()):
            if not ended[i] and token != END:
                texts[i].append(characters[token])
        ended |= next_character.eq(END)
        if ended.all():
            break
        cursor = next_character.masked_fill(ended, END)[:, None]
    return [''.join(chars) for chars in texts]


def nearest(row, train):
    scores = [(sum(edit_counts(row.glosses, item.glosses)), greek(item.greek)) for item in train]
    best = min(distance for distance, _ in scores)
    candidates = {text for distance, text in scores if distance == best}
    return next(iter(candidates)) if len(candidates) == 1 else ''


def char_overlap(rows):
    scores = []
    for size in range(1, 7):
        matches = predicted = total = 0
        for row in rows:
            counts = []
            for key in ('reference', 'prediction'):
                text = row[key].replace(' ', '')
                counts.append(Counter(text[i:i + size] for i in range(len(text) - size + 1)))
            gold, guess = counts
            total += sum(gold.values())
            predicted += sum(guess.values())
            matches += sum((gold & guess).values())
        if total or predicted:
            scores.append(5 * matches / (4 * total + predicted))
    return sum(scores) / len(scores) if scores else None


def subset_metrics(rows):
    edits = [edit_counts(row['reference'].split(), row['prediction'].split()) for row in rows]
    substitutions, deletions, insertions = (sum(counts) for counts in zip(*edits)) if rows else (0, 0, 0)
    words = sum(len(row['reference'].split()) for row in rows)
    errors = substitutions + deletions + insertions
    return {'templates': len(rows), 'reference_words': words, 'word_edits': errors,
            'substitutions': substitutions, 'deletions': deletions, 'insertions': insertions,
            'wer': errors / words if words else None,
            'exact': sum(row['reference'] == row['prediction'] for row in rows),
            'corpus_character_f2_orders_1_to_6': char_overlap(rows)}


def summarize(rows):
    known = [row for row in rows if not row['unseen_glosses']]
    unknown = [row for row in rows if row['unseen_glosses']]
    return {**subset_metrics(rows), 'all_glosses_seen_templates': len(known),
            'by_gloss_vocabulary': {'all_glosses_seen': subset_metrics(known),
                                    'one_or_more_unseen_glosses': subset_metrics(unknown)}, 'rows': rows}


def metrics(rows, predictions, vocab):
    details = []
    for row, prediction in zip(rows, predictions, strict=True):
        reference, prediction = greek(row.greek), greek(prediction)
        details.append({'template': row.key, 'reference': reference, 'prediction': prediction,
                        'glosses': row.glosses, 'unseen_glosses': sorted(set(row.glosses) - vocab.keys()),
                        'word_edits': sum(edit_counts(reference.split(), prediction.split()))})
    return summarize(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=90)
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error('--epochs must be at least 5')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    roles, proof = split_templates()
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    split_json = json.dumps(proof, indent=2, ensure_ascii=False) + '\n'
    if FREEZE.exists() and FREEZE.read_text(encoding='utf-8') != split_json:
        raise ValueError('The source manifests changed after the official annotation split was frozen')
    FREEZE.write_text(split_json, encoding='utf-8')
    print('Frozen role counts', {role: len(rows) for role, rows in roles.items()}, flush=True)
    train = roles['train']
    glossary = sorted({token for row in train for token in row.glosses})
    alphabet = ['<pad>', '<start>', '<end>', '<unknown>'] + sorted({token for row in train for token in greek(row.greek)})
    vocab = {token: i for i, token in enumerate(glossary, 4)}
    char_vocab = {token: i for i, token in enumerate(alphabet)}
    model = GlossTranslator(len(vocab) + 4, len(alphabet))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = rng.sample(train, len(train))
        for i in range(0, len(batches), 24):
            batch = batches[i:i + 24]
            memory, keys, valid, state = model.encode(tensor_rows(batch, vocab, 'gloss'))
            target = tensor_rows(batch, char_vocab, 'text')
            logits, _ = model.decode(target[:, :-1], state, memory, keys, valid)
            loss = F.cross_entropy(logits.flatten(0, 1), target[:, 1:].reshape(-1), ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if epoch % 5 and epoch != args.epochs:
            continue
        valid_rows = roles['development']
        validation = metrics(valid_rows, generate(model, valid_rows, vocab, alphabet), vocab)
        rank = (validation['wer'], -validation['corpus_character_f2_orders_1_to_6'])
        if best is None or rank < best[0]:
            best = rank, epoch, {key: value.detach().clone() for key, value in model.state_dict().items()}
            print('Development checkpoint', epoch, rank, flush=True)
    _, epoch, state = best
    model.load_state_dict(state)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state, 'gloss_vocab': vocab, 'alphabet': alphabet, 'epoch': epoch,
                'frozen_split_sha256': sha(FREEZE)}, CHECKPOINT)

    reports = {}
    for role, rows in roles.items():
        reports[role] = {'learned_character_decoder': metrics(rows, generate(model, rows, vocab, alphabet), vocab)}
        if role != 'train':
            reports[role]['nearest_training_text'] = metrics(rows, [nearest(row, train) for row in rows], vocab)
            reports[role]['literal_gloss_text'] = metrics(rows, [' '.join(row.glosses) for row in rows], vocab)
    report = {'scope': 'Gold annotated gloss input only. Evaluation is per unique official script pair with scene-disjoint '
                       'new holdouts. The experiment did not perform video recognition or synthesize any labels.',
              'split_sha256': sha(FREEZE), 'checkpoint_sha256': sha(CHECKPOINT), 'selected_epoch': epoch,
              'criterion': ['minimum development word error rate', 'maximum development character F2 on ties'],
              'reports': reports}
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for role, methods in reports.items():
        print(role, {method: {key: value for key, value in score.items() if key != 'rows'}
                     for method, score in methods.items()})


if __name__ == '__main__':
    main()
