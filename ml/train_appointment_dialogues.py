"""Fit and replay matched appointment decoders on ordered, verified CERTH takes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from preprocess import FEATURE_COUNT
from train_continuous import Utterance, ctc_decode, edit_counts
from train_continuous_hard import datasets, read_selected
from train_continuous_multitask import OrderAwareCTC, train_candidate
from train_greek_frozen_visual import UNK, WordDecoder, targets, write
from train_greek_translation_pilot import BOS, PAD, SEED, greek, nll, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from plan_continuous_composition import normalize_greek, read_annotations

DATA = Path('data/appointment-dialogues')
SUPPORT = Path('data/appointment-dialogues-support')
REPORTS = Path('reports/appointment-dialogues')
ARTIFACTS = Path('artifacts/appointment-dialogues')
FREEZE = REPORTS / 'translation-freeze.json'
BENCHMARK = REPORTS / 'translation-benchmark.json'
ANNOTATIONS = Path('data/original_annotations')
HARD = Path('data/continuous-hard-split')
ROLES = ('train', 'val', 'test')
ARMS = ('individual', 'contextual')
SOURCE = re.compile(r'(?P<scene>[a-z]+\d+)/(?P=scene)_signer(?P<signer>\d+)_rep(?P<take>\d+)_sentences/sentences(?P<turn>\d+)')
BATCH_SIZE, HISTORY_LIMIT, APPOINTMENT_COPIES = 12, 4, 3


@dataclass
class Dialogue:
    id: str
    scenario: str
    signer: str
    turns: list[Utterance]


def plan(path):
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    samples = {row['id']: row for row in manifest['samples']}
    if len(samples) != len(manifest['samples']):
        raise ValueError('Duplicate appointment sample IDs')
    archives = [ANNOTATIONS / name for name in ('GSL_split.zip', 'supplementary.zip')]
    canonical, references = read_annotations(*archives)
    seen_ids, seen_sources, seen_dialogues, scene_roles, signer_roles = set(), set(), set(), {}, {}
    dialogues = {role: [] for role in ROLES}
    for dialogue in manifest['dialogues']:
        identity, scene, role = (dialogue[key] for key in ('id', 'scenario', 'split'))
        signer, repetition = str(dialogue['signer']), str(dialogue['repetition'])
        turn_ids = dialogue['turn_ids']
        if identity in seen_dialogues or role not in ROLES or len(turn_ids) < 3:
            raise ValueError('Unexpected appointment dialogue ID, role, or incomplete ordered turn list')
        if scene_roles.setdefault(scene, role) != role or signer_roles.setdefault(signer, role) != role:
            raise ValueError('An appointment scene or signer crosses roles')
        seen_dialogues.add(identity)
        official_numbers, indexes = [], []
        for identifier in turn_ids:
            row = samples[identifier]
            source = row['source_path']
            source_match = SOURCE.fullmatch(source)
            annotation = source.split('/', 1)[-1]
            if (not source_match or (scene, signer, repetition) != tuple(source_match[key] for key in ('scene', 'signer', 'take'))
                    or row['dialogue_id'] != identity or row['split'] != role or str(row['signer']) != signer
                    or identifier in seen_ids or source in seen_sources or canonical.get(annotation) != tuple(row['glosses'])
                    or normalize_greek(references.get(annotation, '')) != normalize_greek(row['text']) or not greek(row['text'])):
                raise ValueError(f'Appointment turn does not match the official same-take annotations: {identifier}')
            seen_ids.add(identifier)
            seen_sources.add(source)
            official_numbers.append(int(source_match['turn']))
            indexes.append(row['turn_index'])
        for order in (official_numbers, indexes):
            if any(type(number) is not int for number in order) or order != list(range(order[0], order[0] + len(order))):
                raise ValueError('Appointment turns must be consecutive in official source order')
        dialogues[role].append(dialogue)
    if seen_ids != samples.keys() or any(not group for group in dialogues.values()):
        raise ValueError('All appointment rows must belong to a complete train, development, or test dialogue')
    held_scenes = {scene for scene, role in scene_roles.items() if role != 'train'}
    held_annotations = [source for source in canonical if source.split('_', 1)[0] in held_scenes]
    held_glosses = {canonical[source] for source in held_annotations}
    held_texts = {normalize_greek(references[source]) for source in held_annotations if source in references}
    held_signers = {signer for signer, role in signer_roles.items() if role != 'train'}
    return dialogues, held_scenes, held_signers, held_glosses, held_texts, archives


def with_features(dialogues, rows):
    by_id = {row.id: row for row in rows}
    if any(row.fps != 15 or row.frames.shape[1] != FEATURE_COUNT for row in rows):
        raise ValueError('Appointment video must use the 15 Hz, 224-value landmark schema')
    return [Dialogue(scene['id'], scene['scenario'], str(scene['signer']),
                     [by_id[identifier] for identifier in scene['turn_ids']]) for scene in dialogues]


def normalized(row):
    return Utterance(row.id, row.source, row.signer, row.split, row.glosses, greek(row.text), row.frames,
                     row.fps, row.quality, row.extraction_ms)


def training_pool(train_dialogues, held_scenes, held_signers, held_glosses, held_texts, *, legacy, support=()):
    prior, provenance = [], {}
    if legacy:
        _, prior, _, provenance = datasets(HARD / 'manifest.json', HARD / 'features.jsonl')
    accepted, counts = {}, Counter()
    used_appointment = set()
    pairs = [(row, 'legacy') for row in prior]
    pairs.extend((row, 'appointment') for dialogue in train_dialogues for row in dialogue.turns)
    pairs.extend((row, 'support') for row in support)
    for row, origin in pairs:
        scene = row.source.split('/', 1)[0]
        if row.signer in held_signers or scene in held_scenes:
            counts[f'{origin}_held_scene_or_signer'] += 1
        elif row.glosses in held_glosses or normalize_greek(row.text) in held_texts:
            counts[f'{origin}_held_reference_overlap'] += 1
        elif row.glosses and greek(row.text):
            if origin == 'support' and row.id in accepted:
                raise ValueError('The added vocabulary support duplicates a baseline training clip')
            accepted[row.id] = normalized(row)
            if origin == 'appointment':
                used_appointment.add(row.id)
            if origin == 'support':
                counts['eligible_support'] += 1
    counts['eligible_appointment'] = len(used_appointment)
    counts['eligible_legacy'] = len(accepted) - len(used_appointment) - counts['eligible_support']
    if counts['eligible_support'] != len(support):
        raise ValueError('A planned added vocabulary support fails held-data exclusion')
    if len(used_appointment) < 6:
        raise ValueError('Too few appointment turns survived held-reference exclusions')
    return accepted, used_appointment, dict(counts), provenance


def read_support(annotations):
    manifest, features = SUPPORT / 'manifest.json', SUPPORT / 'features.jsonl'
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    if (not raw.get('complete') or len(raw['samples']) != raw['planned_samples']
            or any(row['split'] != 'train' for row in raw['samples'])):
        raise ValueError('Added vocabulary support must be the complete predeclared train-only cohort')
    canonical, references = read_annotations(*annotations)
    for row in raw['samples']:
        source = row['source_path']
        match = SOURCE.fullmatch(source)
        annotation = source.split('/', 1)[-1]
        if (not match or str(row['signer']) != match['signer'] or canonical.get(annotation) != tuple(row['glosses'])
                or normalize_greek(references.get(annotation, '')) != normalize_greek(row['text'])):
            raise ValueError('An added support clip does not match its official paired annotation')
    rows, digest = read_selected(manifest, features, ('train',))
    if any(row.fps != 15 for row in rows['train']):
        raise ValueError('Added support features must use the same 15 Hz cadence')
    return rows['train'], {'manifest_sha256': sha(manifest), 'features_sha256': digest,
                           'ids': sorted(row.id for row in rows['train'])}


def variant_paths(support):
    if not support:
        return ARTIFACTS, FREEZE, BENCHMARK
    reports = REPORTS.with_name(SUPPORT.name)
    return ARTIFACTS.with_name(SUPPORT.name), reports / FREEZE.name, reports / BENCHMARK.name


def encode_clips(encoder, labels, clips):
    memories, predictions = [], []
    for start in range(0, len(clips), BATCH_SIZE):
        batch = clips[start:start + BATCH_SIZE]
        if any(clip.ndim != 2 or clip.shape[1] != FEATURE_COUNT or len(clip) < 2 or not clip.isfinite().all()
               for clip in batch):
            raise ValueError('Invalid continuous landmark clip')
        lengths = torch.tensor([len(clip) for clip in batch])
        hidden = encoder.encode(pad_sequence(batch, batch_first=True), lengths)[1]
        logits = encoder.ctc(hidden).log_softmax(-1)
        for length, memory, scores in zip(lengths.tolist(), hidden, logits, strict=True):
            memories.append(memory[:length].clone())
            glosses, confidence = ctc_decode(scores[:length], labels)
            predictions.append({'glosses': glosses, 'gloss_confidence': confidence})
    return memories, predictions


def prepare(decoder, memories, histories, vocabulary):
    lengths = torch.tensor([len(memory) for memory in memories])
    prepared = decoder.prepare(pad_sequence(memories, batch_first=True), lengths)
    if not any(histories):
        return prepared
    indices, contexts = [], []
    for index, previous in enumerate(histories):
        if previous:
            indices.append(index)
            contexts.append(torch.tensor([token for phrase in previous[-HISTORY_LIMIT:]
                                          for token in (BOS, *(vocabulary.get(word, UNK) for word in phrase.split()))]))
    inputs = pad_sequence(contexts, batch_first=True, padding_value=PAD)
    packed = pack_padded_sequence(decoder.dropout(decoder.embed(inputs)), [len(context) for context in contexts],
                                  batch_first=True, enforce_sorted=False)
    hidden = prepared[-1].clone()
    _, conditioned = decoder.gru(packed, hidden[:, indices].contiguous())
    hidden[:, indices] = conditioned
    return (*prepared[:-1], hidden)


@dataclass
class FrozenDialogueDecoder:
    encoder: OrderAwareCTC
    labels: list[str | None]
    decoder: WordDecoder
    words: tuple[str, ...]
    vocabulary: dict[str, int]
    limit: int
    contextual: bool

    def decode_memories(self, memories, *, mode='autoregressive', history=None):
        previous = []
        outcomes = []
        for index, memory in enumerate(memories):
            context = previous if self.contextual and mode != 'no_history' else []
            inputs = prepare(self.decoder, [memory], [context], self.vocabulary)
            tokens, terminated = self.decoder.generate(inputs, self.limit)
            text = ' '.join(self.words[token] for token in tokens[0])
            outcomes.append({'text': text, 'terminated': terminated[0]})
            previous.append(history[index] if history is not None else text)
        return outcomes

    @torch.inference_mode()
    def predict_dialogue(self, clips):
        """Decode chronological landmark segments using past generated text only."""
        memories, signs = encode_clips(self.encoder, self.labels, clips)
        return [{**words, **gloss} for words, gloss in zip(self.decode_memories(memories), signs, strict=True)]


def load_frozen_dialogue_decoder(freeze_path=FREEZE, arm='contextual'):
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if arm not in ARMS:
        raise ValueError(f'Unknown dialogue decoder arm: {arm}')
    encoder_record, arm_record = freeze['encoder'], freeze['arms'][arm]
    paths = [ROOT / row['path'] for row in (encoder_record, arm_record)]
    if any(sha(path) != row['sha256'] for path, row in zip(paths, (encoder_record, arm_record), strict=True)):
        raise ValueError('Frozen dialogue model bytes changed')
    enc, dec = (torch.load(path, weights_only=True, map_location='cpu') for path in paths)
    labels, state = enc['labels'], enc['state_dict']
    model = OrderAwareCTC(len(labels), state['mean'], state['scale'])
    model.load_state_dict(state)
    decoder = WordDecoder(len(dec['words']))
    decoder.load_state_dict(dec['state_dict'])
    words = tuple(dec['words'])
    return FrozenDialogueDecoder(model.eval(), labels, decoder.eval(), words,
                                 {word: index for index, word in enumerate(words)}, dec['limit'], arm == 'contextual')


def training_turns(accepted, appointment_ids, dialogues):
    contexts = {}
    for scene in dialogues:
        previous = []
        for row in scene.turns:
            if row.id in appointment_ids:
                contexts[row.id] = tuple(previous[-HISTORY_LIMIT:])
                previous.append(accepted[row.id].text)
            else:
                previous.append('')  # Preserve position without using excluded labels.
    batch = []
    for identity, row in accepted.items():
        example = row, contexts.get(identity, ())
        batch.extend([example] * (APPOINTMENT_COPIES if identity in appointment_ids else 1))
    return batch


def score_dialogues(dialogues, decoded, signs=None):
    all_rows = []
    for scene, predictions in zip(dialogues, decoded, strict=True):
        turns = []
        for row, predicted in zip(scene.turns, predictions, strict=True):
            reference = greek(row.text)
            outcome = {'id': row.id, 'reference': reference, 'prediction': predicted['text'],
                       'reference_glosses': row.glosses, 'terminated': predicted['terminated'],
                       'edits': dict(zip(('substitutions', 'deletions', 'insertions'),
                                         edit_counts(reference.split(), predicted['text'].split()), strict=True))}
            if signs is not None:
                outcome.update(signs[row.id])
            turns.append(outcome)
        exact = all(turn['reference'] == turn['prediction'] for turn in turns)
        raw_prediction = '\n'.join(turn['prediction'] for turn in turns)
        raw_reference = '\n'.join(turn['reference'] for turn in turns)
        all_rows.append({'id': scene.id, 'scenario': scene.scenario, 'signer': scene.signer, 'complete_exact': exact,
                         'raw_prediction': raw_prediction, 'raw_reference': raw_reference,
                         'full_text_word_edits': sum(edit_counts(raw_reference.split(), raw_prediction.split())),
                         'turns': turns})
    turns = [turn for row in all_rows for turn in row['turns']]
    word_count = sum(len(turn['reference'].split()) for turn in turns)
    word_edits = sum(sum(turn['edits'].values()) for turn in turns)
    full_text_edits = sum(row['full_text_word_edits'] for row in all_rows)
    summary = {'takes': len(all_rows), 'scenarios': len({row['scenario'] for row in all_rows}), 'turns': len(turns),
               'reference_words': word_count, 'word_edits': word_edits, 'word_error_rate': word_edits / word_count,
               'full_text_word_edits': full_text_edits, 'full_text_word_error_rate': full_text_edits / word_count,
               'exact_turns': sum(row['reference'] == row['prediction'] for row in turns),
               'exact_full_dialogues': sum(row['complete_exact'] for row in all_rows),
               'nonempty_turns': sum(bool(row['prediction']) for row in turns),
               'terminated_turns': sum(row['terminated'] for row in turns)}
    if signs is not None:
        summary['reference_glosses'] = sum(len(row['reference_glosses']) for row in turns)
        summary['gloss_edits'] = sum(sum(edit_counts(row['reference_glosses'], row['glosses'])) for row in turns)
    return {'summary': summary, 'dialogues': all_rows}


def lexical_coverage(dialogues, labels, words):
    result = {}
    turns = [row for scene in dialogues for row in scene.turns]
    for name, vocabulary, sequences in (
        ('glosses', set(labels[1:]), [row.glosses for row in turns]),
        ('greek_words', set(words[4:]), [greek(row.text).split() for row in turns]),
    ):
        reference = [token for sequence in sequences for token in sequence]
        result[name] = {'known_occurrences': sum(token in vocabulary for token in reference), 'occurrences': len(reference),
                        'known_types': len(set(reference) & vocabulary), 'types': len(set(reference)),
                        'turns_entirely_in_vocabulary': sum(set(sequence) <= vocabulary for sequence in sequences),
                        'out_of_vocabulary': sorted(set(reference) - vocabulary)}
    return result


def fit_decoder(arm, encoder, labels, memories, examples, development, dev_memories, words, limit, epochs):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    decoder = WordDecoder(len(words))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=.003)
    vocabulary = {word: index for index, word in enumerate(words)}
    run = FrozenDialogueDecoder(encoder, labels, decoder, words, vocabulary, limit, arm == 'contextual')
    best = None
    for epoch in range(1, epochs + 1):
        decoder.train()
        shuffled = rng.sample(examples, len(examples))
        for offset in range(0, len(shuffled), BATCH_SIZE):
            rows, history = zip(*shuffled[offset:offset + BATCH_SIZE], strict=True)
            inputs = prepare(decoder, [memories[row.id] for row in rows], history if run.contextual else [()] * len(rows), vocabulary)
            incoming, target = targets(rows, vocabulary)
            logits, _ = decoder.step(incoming, *inputs)
            loss = nll(logits, target).mean()
            if not bool(loss.isfinite()):
                raise ValueError('Non-finite appointment decoder loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 4.)
            optimizer.step()
        if epoch % 5 and epoch != epochs:
            continue
        decoder.eval()
        with torch.no_grad():
            result = score_dialogues(development, [run.decode_memories(group) for group in dev_memories])['summary']
        rank = result['word_edits'], -result['exact_turns'], -result['terminated_turns']
        if best is None or rank < best[0]:
            best = rank, epoch, result, {key: value.detach().clone() for key, value in decoder.state_dict().items()}
            print(json.dumps({'arm': arm, 'epoch': epoch, 'development': result}), flush=True)
    return best[1:]


def paths_for_model(path):
    return {'path': path.resolve().relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def train(manifest, features, vision_epochs, text_epochs, legacy, support=False):
    artifacts, freeze_path, _ = variant_paths(support)
    if freeze_path.exists():
        raise ValueError(f'The appointment comparison is already frozen at {freeze_path}')
    groups, scenes, signers, excluded_glosses, excluded_texts, annotations = plan(manifest)
    rows, feature_digest = read_selected(manifest, features, ('train', 'val'))
    train_dialogues = with_features(groups['train'], rows['train'])
    development = with_features(groups['val'], rows['val'])
    support_rows, support_provenance = read_support(annotations) if support else ([], None)
    accepted, appointment_ids, audit, provenance = training_pool(train_dialogues, scenes, signers, excluded_glosses,
                                                                excluded_texts, legacy=legacy, support=support_rows)
    examples = training_turns(accepted, appointment_ids, train_dialogues)
    labels = [None, *sorted({gloss for row in accepted.values() for gloss in row.glosses})]
    vocabulary = {label: index for index, label in enumerate(labels) if index}
    torch.manual_seed(SEED)
    encoder, vision_history, vision_epoch = train_candidate({'train': list(accepted.values()), 'val': rows['val']},
                                                            labels, vocabulary, {}, epochs=vision_epochs)
    encoder.eval()
    artifacts.mkdir(parents=True, exist_ok=True)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path = artifacts / 'encoder.pt'
    torch.save({'state_dict': encoder.state_dict(), 'labels': labels}, encoder_path)
    with torch.no_grad():
        encoded, _ = encode_clips(encoder, labels, [row.frames for row in accepted.values()])
        train_memory = dict(zip(accepted, encoded, strict=True))
        dev_memories = [encode_clips(encoder, labels, [row.frames for row in scene.turns])[0] for scene in development]
    words = ('<pad>', '<bos>', '<eos>', '<unk>', *sorted({word for row in accepted.values() for word in row.text.split()}))
    limit = max(len(row.text.split()) for row in accepted.values())
    arms = {}
    for arm in ARMS:
        epoch, validation, state = fit_decoder(arm, encoder, labels, train_memory, examples, development, dev_memories,
                                               words, limit, text_epochs)
        path = artifacts / f'{arm}.pt'
        torch.save({'state_dict': state, 'words': words, 'limit': limit}, path)
        arms[arm] = {**paths_for_model(path), 'epoch': epoch, 'development': validation}
    freeze = {'scope': 'Model selection by development only. Test features are read only by evaluation.',
              'manifest_sha256': sha(manifest), 'training_development_features_sha256': feature_digest,
              'annotation_sha256': {str(path): sha(path) for path in annotations}, 'legacy_sources': provenance,
              'excluded_scenes': sorted(scenes), 'excluded_signers': sorted(signers), 'training_audit': audit,
              'training_ids': sorted(accepted), 'appointment_training_ids': sorted(appointment_ids),
              'support': support_provenance, 'vision_epochs': vision_epochs, 'text_epochs': text_epochs,
              'selection': {'vision': 'Development gloss WER, then exact and CTC training loss.',
                            'decoder': 'Autoregressive development word edits, then exact and termination.'},
              'seed': SEED, 'appointment_copies_per_decoder_epoch': APPOINTMENT_COPIES,
              'encoder': {**paths_for_model(encoder_path), 'epoch': vision_epoch, 'history': vision_history}, 'arms': arms}
    write(freeze_path, freeze)
    print('Frozen appointment weights', sha(freeze_path), flush=True)


@torch.inference_mode()
def evaluate(manifest, features, support=False):
    _, freeze_path, benchmark = variant_paths(support)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    if sha(manifest) != freeze['manifest_sha256']:
        raise ValueError('The frozen appointment manifest changed')
    groups, _, _, _, _, annotations = plan(manifest)
    _, training_digest = read_selected(manifest, features, ('train', 'val'))
    if training_digest != freeze['training_development_features_sha256']:
        raise ValueError('Training or development landmarks changed after freezing')
    if support and read_support(annotations)[1] != freeze.get('support'):
        raise ValueError('Added vocabulary support changed after freezing')
    test, held_digest = read_selected(manifest, features, ('test',))
    dialogues = with_features(groups['test'], test['test'])
    outcomes, baseline_signs, coverage = {}, {}, {}
    for arm in ARMS:
        model = load_frozen_dialogue_decoder(freeze_path, arm)
        if arm == 'individual':
            coverage = lexical_coverage(dialogues, model.labels, model.words)
        actual, no_history, reference_history, blank_current = [], [], [], []
        for scene in dialogues:
            clips = [row.frames for row in scene.turns]
            memories, glosses = encode_clips(model.encoder, model.labels, clips)
            decoded = model.decode_memories(memories)
            actual.append(decoded)
            if arm == 'individual':
                baseline_signs.update(zip((row.id for row in scene.turns), glosses, strict=True))
            else:
                no_history.append(model.decode_memories(memories, mode='no_history'))
                reference_history.append(model.decode_memories(memories, history=[greek(row.text) for row in scene.turns]))
                blanks, _ = encode_clips(model.encoder, model.labels, [torch.zeros_like(clip) for clip in clips])
                blank_current.append(model.decode_memories(blanks, history=[item['text'] for item in decoded]))
        result = {'autoregressive': score_dialogues(dialogues, actual, baseline_signs)}
        if arm == 'contextual':
            result['ablations'] = {'no_history': score_dialogues(dialogues, no_history),
                                   'oracle_previous_text': score_dialogues(dialogues, reference_history),
                                   'blank_current_with_previous_model_text': score_dialogues(dialogues, blank_current)}
        outcomes[arm] = result
        print(json.dumps({'arm': arm, 'held': result['autoregressive']['summary']}, ensure_ascii=False), flush=True)
    write(benchmark, {'scope': 'Held appointment takes in official order, with supplied annotation turn boundaries. '
                              'Each take is one CERTH studio signer performing both roles. No human semantic review.',
                      'history': 'Autoregressive evaluation sees only the current video and the model’s own earlier outputs. '
                                 'Only the explicitly labeled oracle ablation uses earlier reference texts.',
                      'freeze_sha256': sha(freeze_path), 'held_features_sha256': held_digest,
                      'training_vocabulary_coverage': coverage, 'results': outcomes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'train', 'evaluate'))
    parser.add_argument('--manifest', type=Path, default=DATA / 'manifest.json')
    parser.add_argument('--features', type=Path, default=DATA / 'features.jsonl')
    parser.add_argument('--vision-epochs', type=int, default=45)
    parser.add_argument('--text-epochs', type=int, default=60)
    parser.add_argument('--appointment-only', action='store_true')
    parser.add_argument('--with-support', action='store_true', help='Add the separately frozen vocabulary training clips')
    args = parser.parse_args()
    if args.vision_epochs < 1 or args.text_epochs < 1:
        parser.error('Training epoch counts must be positive')
    torch.set_num_threads(2)
    if args.phase == 'plan':
        groups, scenes, signers, glosses, texts, _ = plan(args.manifest)
        print(json.dumps({'takes': {role: len(rows) for role, rows in groups.items()}, 'excluded_scenes': sorted(scenes),
                          'excluded_signers': sorted(signers), 'excluded_gloss_sequences': len(glosses),
                          'excluded_text_references': len(texts)}, ensure_ascii=False))
    elif args.phase == 'train':
        train(args.manifest, args.features, args.vision_epochs, args.text_epochs, not args.appointment_only, args.with_support)
    else:
        evaluate(args.manifest, args.features, args.with_support)


if __name__ == '__main__':
    main()
