"""Compare frozen Greek translators on a held chronological appointment stream."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from evaluate_appointment_stream import FPS, automatic_boundaries, boundary_counts, write
from evaluate_unseen_appointment import BoundaryDetector, data_rows, nms_boundaries, signals, with_raw_result
from train_continuous_library import exact_aligned_glosses
from train_greek_translation_pilot import greek
from train_grounded_greek import CONTROLS, alter, load_grounded_decoder


REPORT = Path('reports/appointment-targeted')
PREVIOUS = Path('reports/appointment-unseen/turn-freeze.json')
PREVIOUS_BOUNDARY = Path('artifacts/appointment-unseen/turn-boundaries.pt')
PREVIOUS_DECODER = Path('reports/grounded-greek/appointment-unseen-freeze.json')
FOCUS_WORDS = ('ραντεβού', 'κράτηση', 'παθολόγο', 'παθολόγος', 'μεθαύριο', 'αίμα', 'αίματος', 'υπέρηχο', 'κλείσω', 'μήνα')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_boundaries(freeze_path, checkpoint_path):
    frozen = json.loads(freeze_path.read_bytes())
    if digest(checkpoint_path) != frozen['checkpoint_sha256']:
        raise ValueError('Frozen boundary checkpoint changed')
    state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    detector = BoundaryDetector(state['center'], state['scale'])
    detector.load_state_dict(state)
    detector.eval()

    def predict(frames):
        if frozen['selected_method'] == 'duration':
            return automatic_boundaries(frames, frozen['train_typical_turn_samples'], 0.)
        probabilities = detector(signals(frames)).sigmoid().tolist()
        return nms_boundaries(probabilities, frozen['threshold'], frozen['spacing_samples'])

    return frozen, predict


def aligned_turn(start, end, intervals):
    best_index, best_overlap = None, 0
    for index, (left, right) in enumerate(intervals):
        overlap = max(0, min(right, end) - max(left, start))
        if overlap * 2 < max(end - start, right - left):
            continue
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
        elif overlap == best_overlap:
            best_index = None
    return best_index


@torch.inference_mode()
def calibrate_stream_policy(args, propose_boundaries, decoder):
    from train_targeted_greek import choose_policy

    plan, development, manifest_hash, feature_hash = data_rows(
        args.manifest, args.video_manifest, args.features, ('val',))
    if not development or manifest_hash != digest(args.fit_manifest):
        raise ValueError('Use the targeted fitting manifest with its chronological development video')
    by_id = {row['id']: row for row in plan['samples']}
    clips, references = [], []
    for scene in plan['dialogues']:
        if scene['split'] != 'val':
            continue
        frames, reference, _ = development[scene['id']]
        edges = propose_boundaries(frames)
        intervals = list(zip(reference[:-1], reference[1:], strict=True))
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            index = aligned_turn(start, end, intervals)
            references.append(greek(by_id[scene['turn_ids'][index]]['text']) if index is not None else '')
            clips.append(frames[start:end])
    actual = decoder.inspect(clips)
    controls = {name: decoder.inspect([alter(clip, name) for clip in clips]) for name in CONTROLS}
    policy, development_result = choose_policy(actual, references, controls)
    result = {'scope': 'Emission thresholds calibrated only on the signer-3 development stream. '
                       'Automatically proposed intervals were matched to references only to score calibration.',
              'fit_manifest_sha256': manifest_hash, 'development_stream_sha256': feature_hash,
              'targeted_freeze_sha256': digest(args.targeted_freeze),
              'boundary_freeze_sha256': digest(args.boundary_freeze), 'policy': policy,
              'development': development_result, 'development_segments': len(clips),
              'development_aligned_segments': sum(bool(text) for text in references)}
    write(args.calibrate_stream, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def score_translator(decoder, frames, edges, reference, expected, intervals=None):
    start = perf_counter()
    result = with_raw_result(decoder, frames, edges, reference, expected)
    result['decoder_ms'] = round((perf_counter() - start) * 1000, 1)
    target = greek(' '.join(expected)).split()
    output = greek(result['joined_text']).split()
    raw = greek(result['raw_joined_text']).split()
    wanted, emitted = Counter(target), Counter(output)
    correct = Counter(exact_aligned_glosses(target, output))
    raw_correct = Counter(exact_aligned_glosses(target, raw))
    result['focus_words'] = {word: {'reference': wanted[word], 'emitted': emitted[word],
                                    'aligned_correct': correct[word], 'raw_aligned_correct': raw_correct[word]}
                             for word in FOCUS_WORDS if wanted[word] or emitted[word]}
    if hasattr(decoder, 'phrases'):
        phrase_bank = set(decoder.phrases)
        result['reference_turns_in_phrase_bank'] = sum(greek(text) in phrase_bank for text in expected)

    # Match one exact predicted phrase to one temporal reference. An accidental
    # repeat in a different part of the dialogue does not count as a correct turn.
    if intervals is None:
        intervals = list(zip(reference[:-1], reference[1:], strict=True))
    matched = set()
    correct_partials = total_partials = 0
    for start, end, guess in zip(edges[:-1], edges[1:], result['turns'], strict=True):
        parts = [greek(part['word']) for part in guess.get('partials', ())]
        total_partials += len(parts)
        index = aligned_turn(start, end, intervals)
        reference_text = greek(expected[index]) if index is not None else ''
        supported = [part for part in parts if part in reference_text.split()]
        guess['temporally_supported_partials'] = supported
        correct_partials += len(supported)
        if reference_text and greek(guess['text']) == reference_text:
            matched.add(index)
    result['exact_reference_turns'] = len(matched)
    result['accepted_wrong_segments'] = result['emitted_segments'] - len(matched)
    result['correct_partial_words'] = correct_partials
    result['partial_words'] = total_partials
    return result


def summarize(rows, model, mode):
    records = [row['models'][model][mode] for row in rows]
    sums = {key: sum(row[key] for row in records) for key in (
        'word_edits', 'raw_word_edits', 'reference_words', 'exact_reference_turns',
        'emitted_segments', 'accepted_wrong_segments', 'correct_partial_words', 'partial_words')}
    return {**sums, 'takes': len(records), 'reference_turns': sum(len(row['reference']) for row in rows),
            'exact_dialogues': sum(row['exact_full_text'] for row in records),
            'raw_exact_dialogues': sum(row['raw_exact_full_text'] for row in records),
            'word_error_rate': sums['word_edits'] / sums['reference_words']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/appointment-targeted/manifest.json'))
    parser.add_argument('--fit-manifest', type=Path, default=Path('data/appointment-targeted/manifest.json'))
    parser.add_argument('--video-manifest', type=Path, default=Path('.runtime-tmp/appointment-targeted-video/manifest.json'))
    parser.add_argument('--features', type=Path, default=Path('.runtime-tmp/appointment-targeted-video/features.jsonl'))
    parser.add_argument('--targeted-freeze', type=Path, required=True)
    parser.add_argument('--boundary-freeze', type=Path, default=PREVIOUS)
    parser.add_argument('--boundary-checkpoint', type=Path, default=PREVIOUS_BOUNDARY)
    parser.add_argument('--native-timing', action='store_true',
                        help='Score a native video using its exact annotated intervals after inference')
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--calibrate-stream', type=Path, help='Fit emission floors on development, then exit')
    calibration.add_argument('--stream-policy', type=Path, help='Also score a frozen whole-stream emission policy')
    parser.add_argument('--output', type=Path, default=REPORT / 'whole-stream-benchmark.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    from train_targeted_greek import load_targeted_decoder

    fit_source = args.fit_manifest.read_bytes()
    decoder_freeze = json.loads(args.targeted_freeze.read_bytes())
    if sha256(fit_source).hexdigest() != decoder_freeze['manifest_sha256']:
        raise ValueError('Targeted Greek checkpoint was frozen for different data')
    _, propose_boundaries = load_boundaries(args.boundary_freeze, args.boundary_checkpoint)
    targeted = load_targeted_decoder(args.targeted_freeze)
    if args.calibrate_stream:
        calibrate_stream_policy(args, propose_boundaries, targeted)
        return
    models = {'previous': load_grounded_decoder(PREVIOUS_DECODER),
              'matched_without_new_pairs': load_targeted_decoder(args.targeted_freeze, arm='baseline'),
              'targeted': targeted}
    if args.stream_policy:
        calibrated = json.loads(args.stream_policy.read_bytes())
        if (calibrated['targeted_freeze_sha256'] != digest(args.targeted_freeze)
                or calibrated['boundary_freeze_sha256'] != digest(args.boundary_freeze)
                or calibrated['fit_manifest_sha256'] != decoder_freeze['manifest_sha256']):
            raise ValueError('The stream policy was calibrated with different data or frozen models')
        models['targeted_stream_calibrated'] = replace(targeted, policy=calibrated['policy'])
    plan, held, manifest_hash, held_hash = data_rows(args.manifest, args.video_manifest, args.features, ('test',))
    fit_rows = [row for row in json.loads(fit_source)['samples'] if row['split'] in ('train', 'val')]
    held_rows = [row for row in plan['samples'] if row['split'] == 'test']
    if not held or {str(row['signer']) for row in fit_rows} & {str(row['signer']) for row in held_rows}:
        raise ValueError('The whole-dialogue evaluation needs an independent held signer')
    by_id = {row['id']: row for row in plan['samples']}
    rows = []
    modes = ('automatic',) if args.native_timing else ('automatic', 'published_cuts_diagnostic')
    with torch.inference_mode():
        for scene in plan['dialogues']:
            if scene['split'] != 'test':
                continue
            frames, reference, quality = held[scene['id']]
            edges = propose_boundaries(frames)
            annotated = [by_id[key] for key in scene['turn_ids']]
            expected = [turn['text'] for turn in annotated]
            intervals = None
            if args.native_timing:
                intervals = [(turn['start_ms'] * FPS / 1000, turn['end_ms'] * FPS / 1000) for turn in annotated]
                if any(not 0 <= left < right <= len(frames) for left, right in intervals):
                    raise ValueError('Native annotation intervals fall outside the video')
            comparisons = {
                model: {'automatic': score_translator(decoder, frames, edges, reference, expected, intervals)}
                for model, decoder in models.items()
            }
            if not args.native_timing:
                for model, decoder in models.items():
                    comparisons[model]['published_cuts_diagnostic'] = score_translator(
                        decoder, frames, reference, reference, expected)
            row = {'id': scene['id'], 'scenario': scene['scenario'], 'signer': scene['signer'],
                   'duration_seconds': len(frames) / FPS, 'tracking_quality': quality, 'reference': expected,
                   'boundary_diagnostics': boundary_counts(reference, edges), 'models': comparisons}
            rows.append(row)
            print(json.dumps({'id': row['id'], 'boundaries': row['boundary_diagnostics'],
                              'previous': comparisons['previous']['automatic']['joined_text'],
                              'targeted': comparisons['targeted']['automatic']['joined_text']},
                             ensure_ascii=False), flush=True)
    write(args.output, {
        'scope': ('One uninterrupted video in native mode. Reference midpoints between timed utterances are '
                  'used for boundary diagnostics only. ' if args.native_timing else
                  'Original CERTH cuts joined in sequence, with the original between-cut frames unavailable. '
                  'Published cuts are diagnostic only. ')
                 + 'The automatic pass receives one landmark stream, no reference text, turn count or boundaries.',
        'data_scope': plan['scope'],
        'manifest_sha256': manifest_hash, 'held_stream_sha256': held_hash,
        'boundary_freeze_sha256': digest(args.boundary_freeze), 'targeted_freeze_sha256': digest(args.targeted_freeze),
        'stream_policy_sha256': digest(args.stream_policy) if args.stream_policy else None,
        'previous_freeze_sha256': digest(PREVIOUS_DECODER),
        'summary': {model: {mode: summarize(rows, model, mode) for mode in modes} for model in models},
        'dialogues': rows,
    })


if __name__ == '__main__':
    main()
