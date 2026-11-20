"""Frozen development-only decoder grounding: true versus length-matched wrong video.

Run from the project root with Python -B. This is an oracle-boundary diagnostic,
not a new independent benchmark. No weights, vocabulary, gates or splits are fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from train_native_greek import (FPS, MODEL_VERSION, Pieces, SubtitleModel, describe,
                                embed_stream, frame_spans, generate, load_streams,
                                memory_batch)
from train_continuous import edit_counts
from train_grounded_greek import load_visual
from train_greek_translation_pilot import sha

ROOT = Path(__file__).resolve().parents[1]


def same_length(memory, count):
    """Endpoint-preserving nearest selection; never average latent states."""
    if count < 1 or len(memory) < 1:
        raise ValueError('Empty decoder interval')
    indices = torch.linspace(0, len(memory) - 1, count).round().long()
    return memory[indices]


def match_donors(rows):
    matched = []
    for index, row in enumerate(rows):
        candidates = [i for i, other in enumerate(rows)
                      if i != index and other['signer'] == row['signer']
                      and other['reference'] != row['reference']]
        if not candidates:
            raise ValueError('No different-text same-signer donor')
        donor = min(candidates, key=lambda i: (
            abs(rows[i]['frame_count'] - row['frame_count']),
            abs(rows[i]['duration_s'] - row['duration_s']), rows[i]['id']))
        matched.append(donor)
    return matched


@torch.inference_mode()
def teacher_forced(model, codec, memories, references):
    lengths = torch.tensor([len(item) for item in memories])
    memory = pad_sequence(memories, batch_first=True)
    mask = torch.arange(memory.shape[1])[None] < lengths[:, None]
    tokens = pad_sequence([torch.tensor((codec.bos, *codec.encode(text), codec.eos))
                           for text in references], batch_first=True)
    target = tokens[:, 1:]
    logits = model.translate(memory, mask, tokens[:, :-1])
    # Match the frozen greedy decoder's legal next-token vocabulary.
    logits[:, :, [0, codec.bos]] = -torch.inf
    losses = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=0, reduction='none')
    predicted = logits.argmax(-1)
    result = []
    for i in range(len(references)):
        valid = target[i] != 0
        count = int(valid.sum())
        result.append({'target_tokens_including_eos': count,
                       'nll_sum': float(losses[i, valid].sum()),
                       'nll_per_token': float(losses[i, valid].mean()),
                       'correct_tokens': int((predicted[i, valid] == target[i, valid]).sum()),
                       'eos_correct': bool(predicted[i, count - 1] == codec.eos)})
    return result


@torch.inference_mode()
def free_generation(model, codec, memories):
    result = []
    for memory in memories:
        text, score, ended = generate(model, codec, memory[None], [(0, len(memory))])[0]
        words = text.split()
        bigrams = Counter(zip(words, words[1:]))
        result.append({'raw_text': text, 'eos_completed': ended, 'mean_selected_token_score': score,
                       'completion_only_text': text if ended else '',
                       'repeated_bigram_occurrences': sum(value - 1 for value in bigrams.values())})
    return result


def word_metrics(references, predictions):
    target = [text.split() for text in references]
    guessed = [text.split() for text in predictions]
    per_turn = [edit_counts(left, right) for left, right in zip(target, guessed, strict=True)]
    aggregate = edit_counts([word for row in target for word in row], [word for row in guessed for word in row])
    words = sum(map(len, target))
    return {'turns': len(target), 'reference_words': words,
            'exact_turns': sum(a == b for a, b in zip(target, guessed, strict=True)),
            'nonempty_turns': sum(bool(row) for row in guessed),
            'per_turn_word_edits': sum(sum(row) for row in per_turn),
            'per_turn_wer': sum(sum(row) for row in per_turn) / words,
            'concatenated_word_edits': sum(aggregate), 'concatenated_wer': sum(aggregate) / words,
            'concatenated_substitutions': aggregate[0], 'concatenated_deletions': aggregate[1],
            'concatenated_insertions': aggregate[2]}


def summarize(references, teacher, generated):
    tokens = sum(row['target_tokens_including_eos'] for row in teacher)
    return {'teacher_forced': {
                'target_tokens_including_eos': tokens,
                'token_nll': sum(row['nll_sum'] for row in teacher) / tokens,
                'mean_turn_nll': sum(row['nll_per_token'] for row in teacher) / len(teacher),
                'correct_tokens': sum(row['correct_tokens'] for row in teacher),
                'token_accuracy': sum(row['correct_tokens'] for row in teacher) / tokens,
                'eos_correct': sum(row['eos_correct'] for row in teacher)},
            'free_generation_raw': word_metrics(references, [row['raw_text'] for row in generated]),
            'free_generation_completion_only': word_metrics(references, [row['completion_only_text'] for row in generated]),
            'eos_completed': sum(row['eos_completed'] for row in generated),
            'repeated_bigram_occurrences': sum(row['repeated_bigram_occurrences'] for row in generated),
            'turns_with_repeated_bigrams': sum(row['repeated_bigram_occurrences'] > 0 for row in generated)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/timed-gloss/visual-grounding.json'))
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run this script from the project root')
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(491)
    torch.use_deterministic_algorithms(True)
    folder = Path('data/sartinas-dialogue-fit-expanded')
    freeze_path = Path('reports/native-greek/expanded-freeze.json')
    checkpoint = Path('artifacts/native-greek/subtitle-expanded.pt')
    frozen = json.loads(freeze_path.read_bytes())
    if sha(checkpoint) != frozen['checkpoint_sha256']:
        raise ValueError('Expanded checkpoint no longer matches its freeze')
    streams, provenance = load_streams(folder, ('val',))
    if provenance != frozen['data']:
        raise ValueError('Expanded development source changed after freezing')
    if [describe(row) for row in streams] != frozen['native_dev']:
        raise ValueError('Only the exact frozen native development recording is allowed')
    artifact = torch.load(checkpoint, weights_only=True)
    if artifact['model_version'] != MODEL_VERSION:
        raise ValueError('Incompatible frozen native decoder')
    codec = Pieces(artifact['merges'])
    model = SubtitleModel(artifact['inputs'], len(codec.symbols), artifact['gloss_classes'])
    model.load_state_dict(artifact['state'])
    model.eval()
    visual_path = Path(frozen['visual_path'])
    visual, _ = load_visual(visual_path, frozen['visual_sha256'])
    visual.eval()
    protected = [freeze_path, checkpoint, visual_path, folder / 'manifest.json', folder / 'features.jsonl',
                 folder / 'video-manifest.json', Path(__file__).relative_to(ROOT),
                 Path('ml/train_native_greek.py'), Path('ml/train_grounded_greek.py'),
                 Path('ml/train_greek_translation_pilot.py'), Path('ml/train_continuous.py')]
    before = {path.as_posix(): sha(path) for path in protected}
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    sample_ids = {row['id']: row['turn_ids'] for row in manifest['dialogues'] if row['split'] == 'val'}
    rows, memories = [], []
    for stream in streams:
        features = embed_stream(visual, stream)
        whole = memory_batch(model, [features])[0][0]
        for index, (start, end, reference) in enumerate(frame_spans(stream.turns, len(whole))):
            seconds = stream.turns[index]
            rows.append({'id': sample_ids[stream.id][index], 'stream_id': stream.id, 'signer': stream.signer,
                         'reference': reference, 'start_s': seconds[0], 'end_s': seconds[1],
                         'duration_s': seconds[1] - seconds[0], 'frame_count': end - start,
                         'reference_seen_in_training': reference in artifact['observed_training_text'],
                         'gold_tokens_including_eos': len(codec.encode(reference)) + 1})
            memories.append(whole[start:end].clone())
    if len(rows) != 33 or {row['signer'] for row in rows} != {'sartinas-n4'}:
        raise ValueError('Expected the frozen 33-turn single-signer development diagnostic')
    donors = match_donors(rows)
    if donors != match_donors(rows):
        raise AssertionError('Matching must be deterministic')
    wrong = [same_length(memories[donor], len(memories[i])) for i, donor in enumerate(donors)]
    assert all(len(a) == len(b) for a, b in zip(memories, wrong, strict=True))
    assert all(torch.equal(same_length(memory, len(memory)), memory) for memory in memories)
    references = [row['reference'] for row in rows]
    good_tf, wrong_tf = [teacher_forced(model, codec, value, references) for value in (memories, wrong)]
    good_free, wrong_free = [free_generation(model, codec, value) for value in (memories, wrong)]
    differences = [b['nll_per_token'] - a['nll_per_token'] for a, b in zip(good_tf, wrong_tf, strict=True)]
    tolerance = 1e-7
    for i, row in enumerate(rows):
        row.update(donor_id=rows[donors[i]]['id'], donor_reference=rows[donors[i]]['reference'],
                   donor_original_frames=rows[donors[i]]['frame_count'],
                   donor_original_duration_s=rows[donors[i]]['duration_s'],
                   wrong_minus_true_nll=differences[i],
                   true={'teacher_forced': good_tf[i], 'free_generation': good_free[i]},
                   wrong={'teacher_forced': wrong_tf[i], 'free_generation': wrong_free[i]})
    summaries = {'true_video': summarize(references, good_tf, good_free),
                 'wrong_video_same_length': summarize(references, wrong_tf, wrong_free)}
    subgroup = {}
    for seen in (True, False):
        indices = [i for i, row in enumerate(rows) if row['reference_seen_in_training'] == seen]
        subgroup['seen_training_text' if seen else 'novel_training_text'] = {
            'turns': len(indices),
            'true_video': summarize([references[i] for i in indices], [good_tf[i] for i in indices], [good_free[i] for i in indices]),
            'wrong_video': summarize([references[i] for i in indices], [wrong_tf[i] for i in indices], [wrong_free[i] for i in indices]),
            'true_preferred': sum(differences[i] > tolerance for i in indices)}
    after = {path.as_posix(): sha(path) for path in protected}
    if before != after:
        raise ValueError('Protected source or frozen evidence changed during diagnostic')
    report = {
        'schema': 'signbridge-native-visual-grounding-v1',
        'scope': 'Retrospective frozen expanded-model diagnostic on development only; not independent test evidence.',
        'protocol': {'fit_performed': False, 'test_recordings_loaded': False,
                     'gold_intervals_supplied': True, 'frames_per_second': FPS,
                     'representation': 'Frozen full-stream visual + native encoder memory, sliced at official development intervals.',
                     'donor_rule': 'Same signer, different normalized reference; minimize absolute frame-count difference, then absolute real-duration difference, then lexicographic donor ID. Donors may be reused.',
                     'exact_length_control': 'Resample donor latent states with round(linspace(0, donor_length-1, target_length)); no averaging or model refit.',
                     'teacher_forcing': 'Correct Greek prefix supplied only for next-token scoring; legal vocabulary excludes PAD/BOS; EOS counted. This is not autonomous translation accuracy.',
                     'free_generation': 'Unmodified frozen greedy decoder, maximum38 tokens; report raw text and completion-only text separately. No confidence, tracking or motion gate applied in decoder isolation.',
                     'selection': 'No threshold/model selection; all33 frozen dev turns retained.'},
        'source_sha256': before, 'protected_files_unchanged': before == after, 'data': provenance,
        'runtime': {'torch': torch.__version__, 'device': 'cpu', 'threads': 2, 'seconds': perf_counter() - started},
        'matching': {'turns': len(rows), 'unique_signers': 1, 'unique_donors': len(set(donors)),
                     'exact_original_frame_matches': sum(rows[i]['frame_count'] == rows[j]['frame_count'] for i, j in enumerate(donors)),
                     'median_absolute_frame_difference': median(abs(rows[i]['frame_count'] - rows[j]['frame_count']) for i, j in enumerate(donors)),
                     'maximum_absolute_frame_difference': max(abs(rows[i]['frame_count'] - rows[j]['frame_count']) for i, j in enumerate(donors)),
                     'all_decoder_lengths_exactly_matched': True},
        'summary': summaries,
        'paired': {'true_preferred': sum(value > tolerance for value in differences),
                   'wrong_preferred': sum(value < -tolerance for value in differences),
                   'ties': sum(abs(value) <= tolerance for value in differences),
                   'mean_wrong_minus_true_nll': sum(differences) / len(differences),
                   'median_wrong_minus_true_nll': median(differences), 'tie_tolerance': tolerance,
                   'identical_free_text': sum(a['raw_text'] == b['raw_text'] for a, b in zip(good_free, wrong_free, strict=True))},
        'by_reference_familiarity': subgroup,
        'limitations': ['One already-used development signer and one shared training scenario; turns and reused donors are dependent.',
                       'Latent-memory swapping isolates decoder grounding, not the causal accuracy of raw tracking or the whole vision encoder.',
                       'Length matching changes donor temporal sampling; frozen full-stream memories retain neighbouring context.',
                       'Different official reference text is not a guarantee of completely different signed meaning.',
                       'Teacher-forced gains can rely on correct language prefixes and do not establish usable free translation.',
                       'No new participant, webcam test, model improvement, deployment or untouched benchmark is claimed.'],
        'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'matching': report['matching'], 'summary': summaries,
                      'paired': report['paired'], 'elapsed_seconds': report['runtime']['seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
