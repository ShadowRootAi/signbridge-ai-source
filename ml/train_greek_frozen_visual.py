"""Compare word decoders on fixed joint and dedicated continuous-recognition encoders."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from train_continuous import ctc_decode, edit_counts
from train_continuous_hard import ARTIFACTS as HARD_ARTIFACTS, FREEZE as HARD_FREEZE, datasets, read_selected
from train_continuous_multitask import OrderAwareCTC
from train_greek_translation_pilot import (
    BOS, EOS, PAD, SEED, WIDTH, Translator, batches, certh_sets, check_splits,
    Sample, contextual_sets, elementary_sets, encoded_input, greek, hard_rivals, metrics, nll, sha,
)


UNK = 3
REPORTS = Path('reports/greek-frozen-visual')
ARTIFACTS = Path('artifacts/greek-frozen-visual')


class WordDecoder(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embed = nn.Embedding(size, WIDTH, padding_idx=PAD)
        self.initial = nn.Linear(2 * WIDTH, WIDTH)
        self.keys = nn.Linear(2 * WIDTH, WIDTH)
        self.gru = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.output = nn.Linear(3 * WIDTH, size)
        self.dropout = nn.Dropout(.15)

    def prepare(self, memory, lengths):
        mask = torch.arange(memory.shape[1])[None] < lengths[:, None]
        hidden = self.initial(memory.sum(1) / lengths[:, None]).tanh().unsqueeze(0)
        return memory, self.keys(memory), mask, hidden

    def step(self, ids, memory, keys, mask, hidden):
        states, hidden = self.gru(self.dropout(self.embed(ids)), hidden)
        scores = (states @ keys.transpose(1, 2)) / WIDTH ** .5
        attention = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        return self.output(self.dropout(torch.cat((states, attention @ memory), dim=-1))), hidden

    def generate(self, prepared, limit):
        memory, keys, mask, hidden = prepared
        ids = torch.full((len(memory), 1), BOS)
        finished = torch.zeros(len(memory), dtype=torch.bool)
        sequences = [[] for _ in range(len(memory))]
        for _ in range(limit + 1):
            logits, hidden = self.step(ids, memory, keys, mask, hidden)
            logits[:, 0, :EOS] = -torch.inf
            next_ids = logits[:, 0].argmax(-1)
            for sequence, token, done in zip(sequences, next_ids.tolist(), finished.tolist(), strict=True):
                if not done and token != EOS and len(sequence) < limit:
                    sequence.append(token)
            finished |= next_ids == EOS
            if finished.all():
                break
            ids = torch.where(finished, EOS, next_ids)[:, None]
        return sequences, finished.tolist()


def load_frozen(name):
    if name == 'joint_encoder':
        folder = Path('greek-translation-pilot-context')
        path = Path('artifacts') / folder / 'dual_supervision.pt'
        freeze = json.loads((Path('reports') / folder / 'freeze.json').read_text(encoding='utf-8'))
        expected = freeze['models']['dual_supervision']['artifact_sha256']
    elif name == 'ctc_encoder':
        freeze = json.loads(Path('reports/continuous-contrastive/context-expanded-freeze.json').read_text(encoding='utf-8'))
        entry = freeze['arms']['expanded_authentic_only']
        path, expected = Path(entry['artifact']), entry['sha256']
    else:
        path = HARD_ARTIFACTS / f'{name}.pt'
        freeze = json.loads(HARD_FREEZE.read_text(encoding='utf-8'))
        expected = freeze['arms'][name]['sha256']
    if sha(path) != expected:
        raise ValueError(f'Frozen encoder checkpoint changed: {path}')
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    state = checkpoint['state_dict']
    if name == 'joint_encoder':
        labels = checkpoint['glosses']
        model = Translator(state['mean'], state['scale'], len(labels))
    else:
        labels = checkpoint['labels']
        model = OrderAwareCTC(len(labels), state['mean'], state['scale'])
    model.load_state_dict(state)
    return model.eval(), labels, expected


@torch.no_grad()
def cache_video(model, labels, rows):
    embeddings, signs = [], []
    for offset in range(0, len(rows), 12):
        group = rows[offset:offset + 12]
        frames, lengths = encoded_input(group)
        if isinstance(model, Translator):
            memory = model.encode(frames, lengths)[0]
            scores = model.gloss(memory).log_softmax(-1)
        else:
            memory = model.encode(frames, lengths)[1]
            scores = model.ctc(memory).log_softmax(-1)
        for row, hidden, logits, length in zip(group, memory, scores, lengths.tolist(), strict=True):
            embeddings.append(replace(row, frames=hidden[:length].clone()))
            if row.glosses:
                prediction, _ = ctc_decode(logits[:length], labels)
                signs.append({'id': row.id, 'reference': row.glosses, 'prediction': prediction,
                              'edits': sum(edit_counts(row.glosses, prediction))})
    return embeddings, signs


def targets(group, ids):
    vectors = [torch.tensor([BOS, *(ids.get(word, UNK) for word in row.text.split()), EOS]) for row in group]
    tokens = nn.utils.rnn.pad_sequence(vectors, batch_first=True, padding_value=PAD)
    return tokens[:, :-1], tokens[:, 1:]


@torch.no_grad()
def evaluate(model, rows, words, ids, limit):
    model.eval()
    outcomes = []
    for group in batches(rows, random.Random(SEED), 48):
        prepared = model.prepare(*encoded_input(group))
        decoded, ended = model.generate(prepared, limit)
        incoming, target = targets(group, ids)
        correct = nll(model.step(incoming, *prepared)[0], target)
        rivals = hard_rivals(group)
        memory, keys, mask, hidden = prepared
        wrong = nll(model.step(incoming, memory[rivals], keys[rivals], mask[rivals], hidden[:, rivals])[0], target)
        for i, row in enumerate(group):
            output = [words[index] for index in decoded[i]]
            outcomes.append({'id': row.id, 'reference': row.text, 'prediction': ' '.join(output),
                             'edits': sum(edit_counts(row.text.split(), output)), 'terminated': ended[i],
                             'true_video_nll_advantage': float(wrong[i] - correct[i]) if rivals[i] != i else None})
    return sorted(outcomes, key=lambda row: row['id'])


def train(train_rows, validation_rows, words, ids, limit, epochs):
    torch.manual_seed(SEED)
    generator = random.Random(SEED)
    decoder = WordDecoder(len(words))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=.003)
    best = None
    for epoch in range(1, epochs + 1):
        decoder.train()
        for group in batches(train_rows, generator, 12):
            incoming, target = targets(group, ids)
            scores, _ = decoder.step(incoming, *decoder.prepare(*encoded_input(group)))
            loss = nll(scores, target).mean()
            if not loss.isfinite():
                raise ValueError('Non-finite word-decoder loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 4.)
            optimizer.step()
        if epoch % 5 and epoch != epochs:
            continue
        dev = metrics(evaluate(decoder, validation_rows, words, ids, limit))
        score = (dev['word_error_rate'], -dev['exact_sentences'], -dev['true_video_preferred'])
        if best is None or score < best[0]:
            best = score, epoch, dev, {key: value.detach().clone() for key, value in decoder.state_dict().items()}
        if epoch == 5 or epoch % 15 == 0:
            print(json.dumps({'epoch': epoch, 'validation': dev}), flush=True)
    decoder.load_state_dict(best[3])
    return decoder, best[1], best[2]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def certh_samples(rows):
    return [Sample(row.id, 'certh', row.signer, greek(row.text), row.glosses, row.frames) for row in rows]


def evaluation_inputs(hard):
    elementary = elementary_sets()
    if hard:
        source = Path('data/continuous-hard-split')
        manifest, features = source / 'manifest.json', source / 'features.jsonl'
        base, extended, dev, provenance = datasets(manifest, features)
        freeze = json.loads(HARD_FREEZE.read_text(encoding='utf-8'))
        if provenance != freeze['provenance']:
            raise ValueError('The hard visual training data changed')
        training = {name: certh_samples(rows) + elementary['train']
                    for name, rows in (('previous_contexts', base), ('extended_contexts', extended))}

        def test_cohorts():
            held, _ = read_selected(manifest, features, ('test',))
            return {'certh_hard': certh_samples(held['test']), 'elementary23': elementary['test']}

        return training, certh_samples(dev) + elementary['val'], test_cohorts
    certh, context = certh_sets(), contextual_sets()
    training = certh['train'] + context['train'] + elementary['train']

    def test_cohorts():
        return {'certh_previous': certh['test'], 'certh_context': context['test'], 'elementary23': elementary['test']}

    return dict.fromkeys(('joint_encoder', 'ctc_encoder'), training), certh['val'] + elementary['val'], test_cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--hard', action='store_true', help='Use the separately frozen hard CERTH comparison')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    train_sets, dev, test_cohorts = evaluation_inputs(args.hard)
    reports = REPORTS.with_name(REPORTS.name + '-hard') if args.hard else REPORTS
    artifacts = ARTIFACTS.with_name(ARTIFACTS.name + '-hard') if args.hard else ARTIFACTS
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    frozen, decoders = {}, {}
    for name, rows in train_sets.items():
        print(f'Training {name}', flush=True)
        check_splits({'train': rows, 'val': dev, 'test': []})
        train_text = [row.text.split() for row in rows]
        words = ('<pad>', '<bos>', '<eos>', '<unk>', *sorted({word for text in train_text for word in text}))
        ids = {word: index for index, word in enumerate(words)}
        limit = max(map(len, train_text))
        encoder, labels, digest = load_frozen(name)
        training, _ = cache_video(encoder, labels, rows)
        validation, _ = cache_video(encoder, labels, dev)
        decoder, epoch, validation_summary = train(training, validation, words, ids, limit, args.epochs)
        path = artifacts / f'{name}.pt'
        torch.save({'state_dict': decoder.state_dict(), 'words': words, 'max_words': limit,
                    'encoder_sha256': digest, 'epoch': epoch}, path)
        frozen[name] = {'epoch': epoch, 'validation': validation_summary, 'encoder_sha256': digest,
                        'decoder_sha256': sha(path)}
        decoders[name] = encoder, labels, decoder, words, ids, limit
    write(reports / 'freeze.json', {'selection': 'Minimum development word WER, then exact and same-signer pair counts.',
                                    'seed': SEED, 'epochs': args.epochs, 'arms': frozen})

    results = {}
    cohorts = test_cohorts()
    for name, (encoder, labels, decoder, words, ids, limit) in decoders.items():
        check_splits({'train': train_sets[name], 'val': dev, 'test': [row for rows in cohorts.values() for row in rows]})
        results[name] = {}
        for cohort, rows in cohorts.items():
            inputs, glosses = cache_video(encoder, labels, rows)
            predictions = evaluate(decoder, inputs, words, ids, limit)
            summary = {**metrics(predictions),
                       'references_with_all_words_seen': sum(set(row.text.split()) <= ids.keys() for row in rows)}
            gloss_count = sum(len(row['reference']) for row in glosses)
            sign_summary = {'glosses': gloss_count, 'edits': sum(row['edits'] for row in glosses),
                            'exact_sequences': sum(row['reference'] == row['prediction'] for row in glosses)}
            results[name][cohort] = {'words': summary, 'signs': sign_summary,
                                     'predictions': predictions, 'gloss_predictions': glosses}
            print(json.dumps({'arm': name, 'cohort': cohort, 'text': summary, 'signs': sign_summary}), flush=True)
    write(reports / 'benchmark.json', {
        'scope': ('New CERTH scripts held from all fit. Previously scored Elementary23 is diagnostic.' if args.hard
                  else 'Previously scored recordings, diagnostic only. Word vocabulary from train.'),
        'counts': {'train_by_arm': {name: len(rows) for name, rows in train_sets.items()}, 'val': len(dev),
                   'test_by_cohort': {name: len(rows) for name, rows in cohorts.items()}},
        'frozen': frozen, 'results': results,
    })


if __name__ == '__main__':
    main()
