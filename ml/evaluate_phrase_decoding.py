"""Select a CTC decoding rule on signer-3 development and compare frozen rules."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from evaluate_scaled_phrases import evaluate_rows
from replay_continuous_beam import prefix_beam
from train_continuous import ctc_decode
from train_grounded_greek import write
from train_phrase_spans import FIT_FEATURES, MANIFEST, SOURCE_CHECKPOINT, make_model, starting_point
from train_scaled_phrases import REPORT, digest, input_batch, read_selected, static_videos


FREEZE = REPORT / 'phrase-decoding-freeze.json'
BEAM_WIDTH = 8
POLICIES = [('greedy', 0., 0.), *[
    ('beam', blank, presence) for blank in (0., -.5, -1., -1.5) for presence in (0., .25)
]]


@torch.inference_mode()
def score_features(model, rows):
    result = []
    for start in range(0, len(rows), 20):
        clips = rows[start:start + 20]
        frames, lengths = input_batch(clips)
        logits, presence = model.outputs(frames, lengths)
        result.extend((scores[:length], prior)
                      for scores, prior, length in zip(logits, presence, lengths.tolist(), strict=True))
    return result


def recognize(features, labels, policy):
    scores, presence = features
    mode, blank_offset, weight = policy
    if mode == 'greedy':
        return list(ctc_decode(scores, labels)[0])
    # Phrase-level evidence can break local ties. Retain enough classes to preserve
    # weak candidates and never restrict decoding to a memorized phrase.
    evidence = scores[:, 1:] + weight * torch.nn.functional.logsigmoid(presence.clamp(-8, 8))
    candidate_ids = evidence.amax(dim=0).topk(min(32, evidence.shape[1])).indices.sort().values
    clipped = torch.cat((scores[:, :1], evidence[:, candidate_ids]), dim=-1)
    sequence = prefix_beam(clipped.tolist(), BEAM_WIDTH, blank_offset)
    return [labels[int(candidate_ids[index - 1]) + 1] for index in sequence]


def assess(rows, features, static, labels, policy):
    predictions = [{'id': row.id, 'signer': row.signer, 'split': row.split, 'reference': row.text,
                    'gold_glosses': list(row.glosses), 'predicted_glosses': recognize(item, labels, policy), 'text': None}
                   for row, item in zip(rows, features, strict=True)]
    metrics = evaluate_rows(predictions)
    return {'policy': list(policy), 'metrics': metrics, 'predictions': predictions,
            'static_glosses': sum(len(recognize(item, labels, policy)) for item in static)}


def fit():
    original, labels, weights = starting_point()
    model = make_model(labels, weights).eval()
    roles, feature_hash = read_selected(MANIFEST, FIT_FEATURES, ('val',))
    rows = roles['val']
    if sorted(row.id for row in rows) != original['splits']['val']:
        raise ValueError('Unexpected development cohort')
    real = score_features(model, rows)
    static = score_features(model, static_videos(rows))
    outcomes = []
    for policy in POLICIES:
        result = assess(rows, real, static, labels, policy)
        outcomes.append(result)
        print(policy, result['metrics']['gloss_edit_counts'], 'static', result['static_glosses'], flush=True)
    baseline = outcomes[0]
    eligible = [outcome for outcome in outcomes if outcome['static_glosses'] <= baseline['static_glosses']]
    def rank(outcome):
        metrics = outcome['metrics']
        return metrics['gloss_edits'], metrics['gloss_edit_counts']['substitutions'], -metrics['gloss_sequence_exact']
    selected = min(eligible, key=rank)
    write(FREEZE, {'scope': 'Only signer-3 development, selected on total edits with a static-control emission constraint.',
                   'implementation_sha256': digest(Path(__file__)),
                   'model_sha256': digest(SOURCE_CHECKPOINT), 'val_features_sha256': feature_hash,
                   'selected': selected['policy'], 'outcomes': outcomes})
    print('selected', selected['policy'], flush=True)


def score(manifest, features):
    _, labels, weights = starting_point()
    freeze = json.loads(FREEZE.read_bytes())
    if freeze['implementation_sha256'] != digest(Path(__file__)) or freeze['model_sha256'] != digest(SOURCE_CHECKPOINT):
        raise ValueError('The decoding freeze was modified')
    model = make_model(labels, weights).eval()
    roles, feature_hash = read_selected(manifest, features, ('test',))
    rows = roles['test']
    if {row.signer for row in rows} != {'6', '7'}:
        raise ValueError('Unexpected evaluation signers')
    real = score_features(model, rows)
    static = score_features(model, static_videos(rows))
    predictions = {'greedy': assess(rows, real, static, labels, POLICIES[0]),
                   'development_selected': assess(rows, real, static, labels, freeze['selected'])}
    reversed_features = score_features(model, [replace(row, frames=row.frames.flip(0)) for row in rows])
    reversed_metrics = assess(rows, reversed_features, [], labels, freeze['selected'])['metrics']
    write(REPORT / 'phrase-decoding-novel.json', {'scope': 'Previously unused CERTH clips of two phrase references.',
                                                  'freeze_sha256': digest(FREEZE), 'manifest_sha256': digest(manifest),
                                                  'test_features_sha256': feature_hash, 'arms': predictions,
                                                  'reversed_selected': reversed_metrics})
    for name, outcome in predictions.items():
        print(name, outcome['metrics']['gloss_edit_counts'], 'static', outcome['static_glosses'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--features', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'fit':
        fit()
    elif not args.manifest or not args.features:
        parser.error('Score requires a verified manifest and features')
    else:
        score(args.manifest, args.features)


if __name__ == '__main__':
    main()
