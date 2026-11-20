"""Compare source-supported CTC decoding and multi-scenario training offline."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from replay_continuous_beam import BLANK_OFFSETS, collect_logits, replay as beam_replay
from train_continuous import (
    ContinuousCTC, altered_frames, choose_threshold, edit_counts, normalize_text, read_utterances,
    reference_lookup, replay, split_summary, summarize, train_model,
)


DATA = Path('data')
CHECKPOINT = Path('artifacts/continuous-v1-expanded/ctc.pt')
REPORT = Path('reports/continuous-v1-expanded/transfer-diagnostic.json')
SOURCES = {
    'expanded': [DATA / 'continuous-v1-expanded/manifest.json', DATA / 'continuous-v1/features.jsonl',
                 DATA / 'continuous-v1-extra/features.jsonl'],
    'composition': [DATA / 'continuous-composition-pilot/manifest.json', DATA / 'continuous-composition-pilot/features.jsonl'],
    'old_unknowns': [DATA / 'continuous-v1-holdout/manifest.json', DATA / 'continuous-v1-holdout/features.jsonl'],
}
SEED, EPOCHS, BATCH_SIZE = 41, 100, 12


def exact_template_scores(scores, target, target_lengths):
    # Sum the probability of every valid CTC path for each training sequence.
    return -F.ctc_loss(scores[:, None, :].expand(-1, len(target_lengths), -1), target,
                       [len(scores)] * len(target_lengths), target_lengths, reduction='none')


@torch.inference_mode()
def score_templates(model, items, labels, lookup, transform='original'):
    vocabulary = {label: index for index, label in enumerate(labels) if index}
    templates = tuple(lookup)
    encoded = [[vocabulary[label] for label in phrase] for phrase in templates]
    targets = torch.tensor([label for phrase in encoded for label in phrase])
    lengths = [len(phrase) for phrase in encoded]
    rows = []
    for item in items:
        frames = altered_frames(item, transform)
        n = len(frames)
        original = model(frames[None, :, :], [n])[0]
        scores = exact_template_scores(original, targets, lengths)
        best, indexes = scores.topk(2)
        index = int(indexes[0])
        candidates = (frames.flip(0), frames[n // 2].expand_as(frames))
        controls = [exact_template_scores(model(control[None, :, :], [n])[0], targets, lengths)[index]
                    for control in candidates]
        winner = templates[index]
        rows.append({
            'id': item.id, 'signer': item.signer, 'reference_glosses': item.glosses, 'candidate_glosses': winner,
            'reference_text': item.text, 'candidate_text': lookup[winner],
            'candidate_exact': winner == item.glosses and normalize_text(lookup[winner]) == normalize_text(item.text),
            'log_probability_per_frame': float(best[0] / n),
            'template_margin_per_frame': float((best[0] - best[1]) / n),
            'order_gain_per_frame': float((best[0] - torch.stack(controls).max()) / n),
            'minimum_token_peak': float(original[:, encoded[index]].amax(dim=0).min().exp()),
        })
    return rows


FLOOR_KEYS = ('log_probability_per_frame', 'template_margin_per_frame', 'minimum_token_peak')


def accepts(row, thresholds):
    return row['order_gain_per_frame'] > 0 and all(row[key] >= thresholds[key] for key in FLOOR_KEYS)


def choose_template_thresholds(known, controls):
    # Exact CTC score, separation from the next training template and evidence
    # for each token. Candidate thresholds come only from the positive dev clips.
    levels = [sorted({row[key] for row in known}) + [1.000001] for key in FLOOR_KEYS]
    best = None
    for values in product(*levels):
        floors = dict(zip(FLOOR_KEYS, values, strict=True))
        if any(accepts(row, floors) for row in controls):
            continue
        eligible = [row for row in known if accepts(row, floors)]
        errors = sum(not row['candidate_exact'] for row in eligible)
        key = (errors, -len(eligible), *(-value for value in values))
        if best is None or key < best[0]:
            best = key, floors
    return best[1]


def template_summary(rows, floors, has_reference):
    accepted = [row for row in rows if accepts(row, floors)]
    correct = sum(row['candidate_exact'] for row in accepted) if has_reference else 0
    edits = [0, 0, 0]
    target_count = sum(len(row['reference_glosses']) for row in rows)
    for row in rows:
        predicted = row['candidate_glosses'] if accepts(row, floors) else ()
        edits = [old + count for old, count in zip(edits, edit_counts(row['reference_glosses'], predicted), strict=True)]
    return {'utterances': len(rows), 'accepted': len(accepted), 'coverage': len(accepted) / len(rows),
            'correct_texts': correct, 'wrong_or_out_of_scope_texts': len(accepted) - correct,
            'gloss_wer_with_abstentions_as_deletions': sum(edits) / target_count if has_reference else None,
            'rows': [{**row, 'accepted': accepts(row, floors)} for row in rows]}


def load_inputs(name, roles=('train', 'val', 'test')):
    manifest, *features = SOURCES[name]
    data, _ = read_utterances(manifest, features, False, roles)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    expanded, composition = (load_inputs(name) for name in ('expanded', 'composition'))
    unknown = load_inputs('old_unknowns', ('test',))['test']
    old = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    labels, lookup, state = (old[key] for key in ('labels', 'training_greek_references', 'state_dict'))
    if {item.id for cohort in expanded.values() for item in cohort} != set(old['prior_ids']):
        raise ValueError('Frozen checkpoint does not belong to the expanded cohort')
    frozen = ContinuousCTC(len(labels), state['mean'], state['scale']).eval()
    frozen.load_state_dict(state)

    # Neither original validation nor distractor validation contains test signers.
    original_val = score_templates(frozen, expanded['val'], labels, lookup)
    dev = {'outside_scenario': score_templates(frozen, composition['val'], labels, lookup)}
    for mode in ('static_midpoint', 'reversed', 'shuffled'):
        dev[mode] = score_templates(frozen, expanded['val'], labels, lookup, mode)
    floors = choose_template_thresholds(original_val, [row for batch in dev.values() for row in batch])

    joined = {role: expanded[role] + composition[role] for role in ('train', 'val', 'test')}
    signer_roles = {role: {item.signer for item in items} for role, items in joined.items()}
    if sum(map(len, signer_roles.values())) != len(set().union(*signer_roles.values())):
        raise ValueError('A signer appears in more than one joint role')
    dev_ids = [item.id for role in ('train', 'val') for item in joined[role]]
    if len(dev_ids) != len(set(dev_ids)) or set(dev_ids) & {item.id for item in joined['test'] + unknown}:
        raise ValueError('A recording appears in more than one development or test cohort')
    vocabulary = {label: i for i, label in enumerate(sorted({token for row in joined['train'] for token in row.glosses}), 1)}
    joint_labels, joint_lookup = [None, *vocabulary], reference_lookup(joined['train'])
    train_templates = {item.glosses for item in joined['train']}
    novel = [item for item in composition['test'] if item.glosses not in train_templates]
    if not novel or any(set(item.glosses) - vocabulary.keys() for item in novel) or any(
        item.glosses in {other.glosses for other in joined['val']} for item in novel
    ):
        raise ValueError('The composition test must contain genuinely new sequences of known glosses')
    joint, history, epoch = train_model(joined, joint_labels, vocabulary, joint_lookup, EPOCHS, BATCH_SIZE, SEED)
    greedy_threshold = choose_threshold(replay(joint, joined['val'], joint_labels, joint_lookup))
    dev_logits = collect_logits(joint, joined['val'])
    dev_beams = {offset: beam_replay(joined['val'], dev_logits, joint_labels, joint_lookup, offset)
                 for offset in BLANK_OFFSETS}
    selected_offset = min(BLANK_OFFSETS, key=lambda offset: (dev_beams[offset]['gloss_wer'],
                                                           -dev_beams[offset]['exact_sequences'], abs(offset)))

    # Freeze all model and threshold choices before opening prediction outcomes on these cohorts.
    grammar = {'validation': template_summary(original_val, floors, True), 'thresholds': floors,
               'validation_controls': {key: template_summary(rows, floors, False) for key, rows in dev.items()}}
    for name, items, mode, known in (
        ('known_same_scenario', expanded['test'], 'original', True),
        ('genuine_same_scenario_unknown', unknown, 'original', False),
        ('genuine_other_scenario', composition['test'], 'original', False),
        *((mode, expanded['test'], mode, False) for mode in ('static_midpoint', 'reversed', 'shuffled')),
    ):
        grammar[name] = template_summary(score_templates(frozen, items, labels, lookup, mode), floors, known)
    novel_ids = {item.id for item in novel}
    cohorts = {'known_police1': expanded['test'],
               'known_police2': [item for item in composition['test'] if item.id not in novel_ids],
               'new_in_vocabulary_sequences': novel, 'outside_training_scope': unknown}
    transfer = {}
    for name, items in cohorts.items():
        scores = collect_logits(joint, items)
        rows = replay(joint, items, joint_labels, joint_lookup, greedy_threshold)
        beam = beam_replay(items, scores, joint_labels, joint_lookup, selected_offset)
        transfer[name] = {'greedy': {**summarize(rows), 'predictions': rows},
                          'beam_raw_reference_lookup': {**beam, 'coverage': beam['accepted_texts'] / beam['utterances']}}
        if name in ('new_in_vocabulary_sequences', 'outside_training_scope'):
            transfer[name]['out_of_scope_text_emissions'] = {
                'greedy': transfer[name]['greedy']['accepted'], 'beam_raw_reference_lookup': beam['accepted_texts'],
            }
    report = {
        'scope': 'Local experiments on complete, already segmented GSL utterances. All test clips were used in earlier '
                 'experiments. No independent test, webcam, unknown speech-like movement, or spontaneous conversation claim.',
        'sources': {path.as_posix(): sha256(path.read_bytes()).hexdigest() for paths in SOURCES.values() for path in paths},
        'frozen_checkpoint_sha256': sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'grammar_decoding': grammar,
        'grammar_selection': 'Positive signer-3 validation only for thresholds; signer-3 alternative scenario and '
                             'synthetic temporal validation controls constrain false accepts. Positive order gain required.',
        'combined_training': {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'selected_epoch': epoch,
                              'architecture': 'ContinuousCTC: Linear 224→64, LayerNorm+GELU, bidirectional GRU 64, '
                                              'dropout .1 then log-softmax. AdamW lr .002, CTC blank 0.',
                              'splits': {role: split_summary(items, train_templates, vocabulary, joint_lookup)
                                         for role, items in joined.items()},
                              'labels': joint_labels, 'greedy_threshold': greedy_threshold, 'beam_offset': selected_offset,
                              'beam_note': 'Beam raw lookup has no calibrated rejection. Every lookup emission on a '
                                           'new or outside-scope sequence counts as an out-of-scope output, even if '
                                           'its Greek text happens to match a reference string.',
                              'history': history, 'outcomes': transfer},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, results in grammar.items():
        if 'accepted' in results:
            print('grammar', name, results['accepted'], results['correct_texts'], results['utterances'],
                  results['gloss_wer_with_abstentions_as_deletions'])
    for name, results in transfer.items():
        g, b = results['greedy'], results['beam_raw_reference_lookup']
        print('combined', name, 'greedy', g['gloss_wer'], g['greek_reference_exact'], g['accepted'],
              'beam', b['gloss_wer'], b['correct_texts'], b['accepted_texts'], flush=True)


if __name__ == '__main__':
    main()
