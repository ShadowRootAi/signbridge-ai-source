"""Recognize isolated CERTH signs by matching independently indexed training examples."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256 as digest_bytes
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import torch

from evaluate_openhands_verifier import read_native_poses, sha256, write_json
from evaluate_synthetic_ctc_transfer import frame_feature_array, verify_feature_math
from openhands_gsl_stgcn import METADATA_ARCHIVE_SHA256


CACHE = Path('research/cache/openhands-gsl')
REPORT = Path('reports/isolated-exemplar-retrieval')
ARTIFACTS = Path('artifacts/isolated-exemplar-retrieval')
CLASSIFIER = Path('reports/isolated-large-vocabulary')
TRAIN = 'official-train-prototypes-poses.npz'
EXPANDED = 'official-train-expanded-poses.npz'
TEST = 'official-test-poses.npz'
SIGNER = re.compile(r'_signer([1-7])_rep(\d+)_glosses/')
STEPS = 16
METHODS = ('nearest_exemplar', 'class_centroid', 'exemplar_and_centroid')
ERROR_LIMIT = .01
HOLDOUT_LABELS = 30


def annotations(path):
    if sha256(path) != METADATA_ARCHIVE_SHA256:
        raise ValueError('Unexpected official CERTH isolated annotations')
    prefix = 'GSL_split/GSL_isolated/'
    with ZipFile(path) as archive:
        values = {role: dict(line.split('|', 1) for line in archive.read(prefix + f'{role}_greek_iso.csv').decode('utf-8-sig').splitlines())
                  for role in ('train', 'test')}
    return values


def open_source(path, official, *, role):
    poses, offsets, ids, glosses = read_native_poses(path)
    identities = [SIGNER.search(str(value)) for value in ids]
    if any(match is None for match in identities):
        raise ValueError('Isolated source identity has no verified signer')
    signers = np.array([int(match[1]) for match in identities])
    if (poses.ndim != 3 or poses.shape[1:] != (75, 3) or not np.isfinite(poses).all()
            or any(official.get(str(identity)) != gloss for identity, gloss in zip(ids, glosses, strict=True))):
        raise ValueError('Isolated pose rows differ from official labels or pose shape')
    if role == 'test':
        if set(signers) != {3} or dict(zip(ids, glosses, strict=True)) != official:
            raise ValueError('Expected the complete isolated evaluation from signer 3')
    else:
        actors, takes = ('125', '123') if role == 'expanded' else ('124', '1')
        if set(signers) != set(map(int, actors)) or any(match[2] not in takes for match in identities):
            raise ValueError('Expected source training poses with the declared actors and repetitions')
        selected = {identity for identity in official if (match := SIGNER.search(identity)) and match[1] in actors and match[2] in takes}
        if set(ids) != selected:
            raise ValueError('Official isolated source-training coverage changed')
    values = frame_feature_array(poses)
    verify_feature_math(poses, values)
    sequences = [sample_sequence(values, start, end) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    return np.stack(sequences), ids, glosses, signers


def sample_sequence(values, start, end):
    columns = np.r_[0:186, 220:223]  # No face landmarks are available in native isolates.
    positions = np.linspace(start, end - 1, STEPS)
    left = positions.astype(np.int64)
    right = np.minimum(left + 1, end - 1)
    fraction = (positions - left).astype(np.float32)[:, None]
    first = values[np.ix_(left, columns)]
    return first + fraction * (values[np.ix_(right, columns)] - first)


def motion(sequences):
    return np.diff(sequences, axis=1, prepend=sequences[:, :1])


def unit(values):
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)


def descriptors(sequences, location, scale, motion_scale):
    pose = ((sequences - location) / scale).clip(-8, 8).reshape(len(sequences), -1)
    velocity = (motion(sequences) / motion_scale).clip(-8, 8).reshape(len(sequences), -1)
    return unit(np.concatenate((pose, .6 * velocity), axis=1))


def create_bank(sequences, ids, glosses, signers, actors):
    if set(signers) != set(actors) or len(ids) != len(set(ids)):
        raise ValueError('An exemplar bank must use only distinct clips from its training actors')
    order = np.argsort(glosses, kind='stable')
    sequences, ids, glosses, signers = (values[order] for values in (sequences, ids, glosses, signers))
    labels, starts, counts = np.unique(glosses, return_index=True, return_counts=True)
    location = sequences.mean(axis=(0, 1))
    scale = sequences.std(axis=(0, 1)).clip(min=.15)
    motion_scale = motion(sequences).std(axis=(0, 1)).clip(min=.12)
    vectors = descriptors(sequences, location, scale, motion_scale)
    centers = unit(np.add.reduceat(vectors, starts, axis=0) / counts[:, None]).astype(np.float32)
    return {'ids': ids, 'glosses': glosses, 'signers': signers, 'labels': labels, 'starts': starts, 'counts': counts,
            'location': location, 'scale': scale, 'motion_scale': motion_scale, 'vectors': vectors, 'centers': centers}


def scores(bank, sequences):
    vectors = descriptors(sequences, bank['location'], bank['scale'], bank['motion_scale'])
    examples = (torch.from_numpy(vectors) @ torch.from_numpy(bank['vectors']).T).numpy()
    nearest = np.maximum.reduceat(examples, bank['starts'], axis=1)
    centroid = (torch.from_numpy(vectors) @ torch.from_numpy(bank['centers']).T).numpy()
    return {'nearest_exemplar': nearest, 'class_centroid': centroid, 'exemplar_and_centroid': (nearest + centroid) * .5}, examples


def ranks(matrix):
    order = np.argsort(-matrix, axis=1, kind='stable')[:, :5]
    rows = np.arange(len(order))
    confidence = matrix[rows, order[:, 0]]
    return order, confidence, confidence - matrix[rows, order[:, 1]]


def select_gate(matrix, labels, references):
    order, similarities, margins = ranks(matrix)
    winners = labels[order[:, 0]]
    known = np.isin(references, labels)
    correct = known & (winners == references)
    wrong = known & ~correct
    unknown = ~known
    grid = np.linspace(0, 1, 41)
    def thresholds(values):
        return np.unique(np.r_[np.quantile(values, grid), float(values.max()) + .001])
    candidates = thresholds(margins)
    margin_matches = margins >= candidates[:, None]
    best = None
    for similarity in thresholds(similarities):
        eligible = (similarities >= similarity) & margin_matches
        right_counts = (eligible & correct).sum(1)
        wrong_counts = (eligible & wrong).sum(1)
        unknown_counts = (eligible & unknown).sum(1)
        good = np.flatnonzero((wrong_counts <= ERROR_LIMIT * known.sum()) & (unknown_counts <= ERROR_LIMIT * unknown.sum()))
        for index in good:
            key = (int(right_counts[index]), -int(unknown_counts[index]), -int(wrong_counts[index]),
                   float(similarity), float(candidates[index]))
            if best is None or key > best:
                best = key
    return {'minimum_similarity': best[3], 'minimum_margin': best[4], 'dev_correct': best[0],
            'dev_outside_false_accepts': -best[1], 'dev_wrong_accepts': -best[2]}


def summarize(matrix, labels, references, policy):
    order, similarity, margin = ranks(matrix)
    candidates = labels[order]
    known = np.isin(references, labels)
    right = candidates[:, 0] == references
    accepted = (similarity >= policy['minimum_similarity']) & (margin >= policy['minimum_margin'])
    counts = Counter(references[known].tolist())
    raw_class_correct = Counter(references[right].tolist())
    accepted_class_correct = Counter(references[right & accepted].tolist())
    result = {'clips': len(references), 'known_clips': int(known.sum()), 'outside_clips': int((~known).sum()),
              'top1_correct': int(right.sum()), 'top3_correct': int((candidates[:, :3] == references[:, None]).any(1).sum()),
              'top5_correct': int((candidates == references[:, None]).any(1).sum()),
              'observed_known_labels': len(counts), 'labels_with_raw_correct': len(raw_class_correct),
              'macro_top1': float(np.mean([raw_class_correct[label] / count for label, count in counts.items()])),
              'accepted_correct': int((accepted & right).sum()), 'accepted_wrong_known': int((accepted & known & ~right).sum()),
              'accepted_outside': int((accepted & ~known).sum()), 'known_abstained': int((known & ~accepted).sum()),
              'labels_with_accepted_correct': len(accepted_class_correct)}
    return result, order, accepted


def choose(matrix, labels, references):
    gate = {method: select_gate(value, labels, references) for method, value in matrix.items()}
    summary = {method: summarize(value, labels, references, gate[method])[0] for method, value in matrix.items()}
    best = max(METHODS, key=lambda method: (gate[method]['dev_correct'], summary[method]['top1_correct']))
    return {'selected': best, 'gate': gate, 'development': summary}


def reference_examples(bank, examples, class_order):
    ids = []
    for row, label in enumerate(class_order):
        start, count = bank['starts'][label], bank['counts'][label]
        nearest = start + examples[row, start:start + count].argmax()
        ids.append('certh:isolated:' + str(bank['ids'][nearest]))
    return ids


def fit(cache, report, artifacts, expanded=False):
    target = report / 'freeze.json'
    if target.exists():
        raise ValueError('The exemplar retrieval experiment is already frozen')
    official_path, native_path = cache / 'gsl_metadata.zip', cache / TRAIN
    original = annotations(official_path)
    sequences, ids, glosses, signers = open_source(native_path, original['train'], role='train')
    original_train, dev = np.isin(signers, [1, 2]), signers == 4
    dev_sequences, dev_ids, dev_glosses = sequences[dev], ids[dev], glosses[dev]
    labels = sorted(set(glosses[original_train]), key=lambda label: digest_bytes(('exemplar-held-labels/' + label).encode()).hexdigest())
    bank_path, actors = (cache / EXPANDED, [1, 2, 5]) if expanded else (native_path, [1, 2])
    if expanded:
        sequences, ids, glosses, signers = open_source(bank_path, original['train'], role='expanded')
    else:
        sequences, ids, glosses, signers = (values[original_train] for values in (sequences, ids, glosses, signers))
    if set(ids) & set(dev_ids):
        raise ValueError('Source sample overlap across bank and development')
    dev_unknown, test_unknown = labels[:HOLDOUT_LABELS], labels[HOLDOUT_LABELS:2 * HOLDOUT_LABELS]
    artifacts.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    variants = {}
    for name, exclude in (('all_verified_labels', []), ('held_label_challenge', dev_unknown + test_unknown)):
        exemplar = ~np.isin(glosses, exclude)
        calibration = ~np.isin(dev_glosses, test_unknown if exclude else [])
        bank = create_bank(sequences[exemplar], ids[exemplar], glosses[exemplar], signers[exemplar], actors)
        matrix, _ = scores(bank, dev_sequences[calibration])
        path = artifacts / f'{name}.npz'
        np.savez_compressed(path, **bank)
        variants[name] = {'bank_path': path.as_posix(), 'bank_sha256': sha256(path), 'bank_clips': int(exemplar.sum()),
                          'bank_labels': len(bank['labels']), 'bank_signers': actors,
                          **choose(matrix, bank['labels'], dev_glosses[calibration])}
        print(name, variants[name]['selected'], variants[name]['development'][variants[name]['selected']], flush=True)
    write_json(target, {'scope': 'Fit deterministic pose descriptor normalization and original exemplars using only the declared training signers. '
                                'Select methods and acceptance floors on signer 4 only. No upstream pretrained model is used.',
                        'metadata_sha256': sha256(official_path), 'source_train_pose_sha256': sha256(native_path),
                        'source_bank_pose_file': bank_path.name, 'source_bank_pose_sha256': sha256(bank_path),
                        'development_unknown_labels': dev_unknown, 'test_unknown_labels': test_unknown,
                        'wrong_acceptance_limit_on_development': ERROR_LIMIT, 'variants': variants})


def read_bank(record):
    path = Path(record['bank_path'])
    if sha256(path) != record['bank_sha256']:
        raise ValueError('Frozen reference exemplars changed')
    with np.load(path, allow_pickle=False) as archive:
        bank = {key: archive[key] for key in archive.files}
    if set(bank['signers']) != set(record.get('bank_signers', [1, 2])) or set(bank['signers']) & {3, 4}:
        raise ValueError('Frozen exemplars include a development or held signer')
    return bank


def evaluate(cache, report):
    freeze_path, source = report / 'freeze.json', cache / TEST
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    official_path = cache / 'gsl_metadata.zip'
    if sha256(official_path) != freeze['metadata_sha256'] or sha256(cache / TRAIN) != freeze['source_train_pose_sha256']:
        raise ValueError('Source annotation or training pose identity changed after freezing')
    bank_source = cache / freeze.get('source_bank_pose_file', TRAIN)
    if sha256(bank_source) != freeze.get('source_bank_pose_sha256', freeze['source_train_pose_sha256']):
        raise ValueError('Expanded source pose cache changed after freezing')
    original_policy = (json.loads((REPORT / 'freeze.json').read_text(encoding='utf-8'))
                       if bank_source.name == EXPANDED else None)
    sequences, ids, glosses, _ = open_source(source, annotations(official_path)['test'], role='test')
    variants = {}
    for name, record in freeze['variants'].items():
        bank = read_bank(record)
        if set(ids) & set(bank['ids']):
            raise ValueError('Held pose clip entered the exemplar bank')
        held = ~np.isin(glosses, freeze['development_unknown_labels'] if name == 'held_label_challenge' else [])
        matrix, examples = scores(bank, sequences[held])
        summary = {method: summarize(value, bank['labels'], glosses[held], record['gate'][method])[0] for method, value in matrix.items()}
        winner = record['selected']
        _, order, accepted = summarize(matrix[winner], bank['labels'], glosses[held], record['gate'][winner])
        units = reference_examples(bank, examples, order[:, 0])
        rows = [{'id': str(clip), 'reference_gloss': str(reference), 'predicted_gloss': str(bank['labels'][index]),
                 'supporting_lexical_unit': unit_id, 'accepted': bool(decision)}
                for clip, reference, index, unit_id, decision in zip(ids[held], glosses[held], order[:, 0], units, accepted, strict=True)]
        variants[name] = {'held': summary, 'selected': winner, 'predictions': rows}
        if original_policy:
            prior = original_policy['variants'][name]
            mode = prior['selected']
            variants[name]['original_frozen_gate_on_expanded_bank'] = summarize(matrix[mode], bank['labels'], glosses[held],
                                                                                prior['gate'][mode])[0]
        print(name, winner, summary[winner], flush=True)
    write_json(report / 'benchmark.json', {'scope': 'Held signer 3 native pre-cut isolated pose clips. The held-label challenge '
                                                     'reserves different real glosses outside the bank for development and test. '
                                                     'No independent webcam, nonsign, or continuous translation measurement.',
                                           'freeze_sha256': sha256(freeze_path), 'held_pose_sha256': sha256(source), 'variants': variants})


def align_scores(path, ids, glosses, labels):
    with np.load(path, allow_pickle=False) as dataset:
        score_ids, references, source_labels, logits = (dataset[name] for name in ('ids', 'glosses', 'labels', 'logits'))
    by_clip = {identity: index for index, identity in enumerate(score_ids)}
    by_gloss = {label: index for index, label in enumerate(source_labels)}
    if (len(by_clip) != len(score_ids) or set(by_clip) != set(ids) or len(by_gloss) != len(source_labels)
            or set(by_gloss) != set(labels) or logits.shape != (len(score_ids), len(source_labels))
            or not np.isfinite(logits).all()):
        raise ValueError('Classifier scores do not match the independently verified exemplar cohort')
    rows = [by_clip[identity] for identity in ids]
    if not np.array_equal(references[rows], glosses):
        raise ValueError('Classifier and exemplar labels disagree for at least one clip')
    columns = [by_gloss[label] for label in labels]
    return logits[np.ix_(rows, columns)].astype(np.float64)


def probabilities(logits):
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def mix(logits, retrieval, weight):
    centered = retrieval - retrieval.mean(axis=1, keepdims=True)
    return logits + weight * centered / retrieval.std(axis=1, keepdims=True).clip(min=.02)


@dataclass
class FrozenExemplarFusion:
    bank: dict[str, np.ndarray]
    classifier_labels: tuple[str, ...]
    method: str
    weight: float
    policy: dict[str, float]

    def predict(self, raw_pose, classifier_logits, classifier_labels=None):
        pose = np.asarray(raw_pose, dtype=np.float32)
        logits = np.asarray(classifier_logits, dtype=np.float64).reshape(-1)
        labels = self.classifier_labels if classifier_labels is None else tuple(classifier_labels)
        if pose.ndim != 3 or pose.shape[1:] != (75, 3) or not len(pose) or not np.isfinite(pose).all():
            raise ValueError('Expected a nonempty native pose clip with shape [frames,75,3]')
        if len(labels) != len(self.classifier_labels) or set(labels) != set(self.classifier_labels) or logits.shape != (len(labels),) or not np.isfinite(logits).all():
            raise ValueError('Single-clip classifier logits do not match the frozen fusion vocabulary')
        by_label = {label: index for index, label in enumerate(labels)}
        logits = logits[[by_label[label] for label in self.bank['labels']]][None, :]
        features = sample_sequence(frame_feature_array(pose), 0, len(pose))
        evidence, examples = scores(self.bank, features[None])
        weighted = probabilities(mix(logits, evidence[self.method], self.weight))
        rank, similarity, margin = ranks(weighted)
        index = rank[0, 0]
        return {'gloss': str(self.bank['labels'][index]),
                'accepted': bool(similarity[0] >= self.policy['minimum_similarity'] and margin[0] >= self.policy['minimum_margin']),
                'score': float(similarity[0]), 'margin': float(margin[0]),
                'supporting_lexical_unit': reference_examples(self.bank, examples, [index])[0],
                'top5': [{'gloss': str(self.bank['labels'][candidate]), 'score': float(weighted[0, candidate])}
                         for candidate in rank[0]]}


def load_frozen_exemplar_fusion(arm='expanded', report=None):
    if arm not in ('full', 'expanded'):
        raise ValueError('Choose a supported isolated classifier arm')
    report = Path(report) if report is not None else (REPORT.with_name(REPORT.name + '-expanded') if arm == 'expanded' else REPORT)
    retrieval_path = report / 'freeze.json'
    classifier_path = CLASSIFIER / f'{arm}-freeze.json'
    fusion = json.loads((report / f'fusion-{arm}-freeze.json').read_text(encoding='utf-8'))
    if sha256(retrieval_path) != fusion['exemplar_freeze_sha256'] or sha256(classifier_path) != fusion['classifier_freeze_sha256']:
        raise ValueError('Frozen exemplar or classifier identity changed')
    retrieval = json.loads(retrieval_path.read_text(encoding='utf-8'))['variants']['all_verified_labels']
    classifier = json.loads(classifier_path.read_text(encoding='utf-8'))
    bank = read_bank(retrieval)
    if set(bank['labels']) != set(classifier['labels']):
        raise ValueError('Frozen exemplar and classifier labels differ')
    return FrozenExemplarFusion(bank, tuple(classifier['labels']), retrieval['selected'], fusion['weight'], fusion['weighted_policy'])


def decision_rows(labels, references, winners, accepted):
    correct = labels[winners] == references
    return {'correct': int((accepted & correct).sum()), 'wrong': int((accepted & ~correct).sum()),
            'abstained': int((~accepted).sum()), 'correct_glosses': len(set(references[accepted & correct]))}


def fit_rescue(direct, retrieval, labels, refs, baseline_policy):
    baseline_summary, direct_rank, base = summarize(direct, labels, refs, baseline_policy)
    bank_rank, similarity, margin = ranks(retrieval)
    winner = bank_rank[:, 0]
    correct = labels[winner] == refs
    remaining_errors = max(0, int(ERROR_LIMIT * len(refs)) - baseline_summary['accepted_wrong_known'])
    sims = np.unique(np.r_[np.quantile(similarity, np.linspace(0, 1, 41)), similarity.max() + .001])
    margins = np.unique(np.r_[np.quantile(margin, np.linspace(0, 1, 41)), margin.max() + .001])
    best = None
    for maximum_rank in (1, 2, 3):
        eligible = ~base & (direct_rank[:, :maximum_rank] == winner[:, None]).any(axis=1)
        for minimum_similarity in sims:
            variants = eligible & (similarity >= minimum_similarity) & (margin >= margins[:, None])
            wrong = (variants & ~correct).sum(1)
            right = (variants & correct).sum(1)
            for i in np.flatnonzero(wrong <= remaining_errors):
                key = (int(right[i]), -int(wrong[i]), -maximum_rank, float(minimum_similarity), float(margins[i]))
                if best is None or key > best:
                    best = key
    return {'max_classifier_rank': -best[2], 'minimum_similarity': best[3], 'minimum_margin': best[4],
            'development_extra_correct': best[0], 'development_extra_wrong': -best[1]}


def rescue_decisions(direct, retrieval, labels, references, baseline_policy, rescue_policy):
    _, ranking, baseline = summarize(direct, labels, references, baseline_policy)
    bank_rank, similarity, margin = ranks(retrieval)
    winner = bank_rank[:, 0]
    rescued = (~baseline & (similarity >= rescue_policy['minimum_similarity']) & (margin >= rescue_policy['minimum_margin'])
               & (ranking[:, :rescue_policy['max_classifier_rank']] == winner[:, None]).any(axis=1))
    guesses = np.where(rescued, winner, ranking[:, 0])
    result = {**decision_rows(labels, references, guesses, baseline | rescued),
              'extra_correct': int((rescued & (labels[guesses] == references)).sum()),
              'extra_wrong': int((rescued & (labels[guesses] != references)).sum())}
    return result, guesses, baseline | rescued, rescued


def fuse(cache, report, arm):
    source = report / 'freeze.json'
    freeze = json.loads(source.read_text(encoding='utf-8'))
    output = report / f'fusion-{arm}-freeze.json'
    if output.exists():
        raise ValueError('The exemplar fusion experiment is already frozen')
    model_freeze = CLASSIFIER / f'{arm}-freeze.json'
    model = json.loads(model_freeze.read_text(encoding='utf-8'))
    record = freeze['variants']['all_verified_labels']
    if model['train_signers'] != record.get('bank_signers', [1, 2]) or model['development_signers'] != [4] or model['test_signers'] != [3]:
        raise ValueError('Classifier and exemplar signer roles differ')
    bank = read_bank(record)
    official = annotations(cache / 'gsl_metadata.zip')
    sequences, ids, glosses, signers = open_source(cache / TRAIN, official['train'], role='train')
    mask = signers == 4
    dev_path = CLASSIFIER / f'{arm}-dev-scores.npz'
    logits = align_scores(dev_path, ids[mask], glosses[mask], bank['labels'])
    selected = record['selected']
    retrieval = scores(bank, sequences[mask])[0][selected]
    weights = (0., .125, .25, .5, 1., 2., 4.)
    weight = max(weights, key=lambda value: (int((bank['labels'][mix(logits, retrieval, value).argmax(1)] == glosses[mask]).sum()), -value))
    base = probabilities(logits)
    fused = probabilities(mix(logits, retrieval, weight))
    baseline_policy = select_gate(base, bank['labels'], glosses[mask])
    weighted_policy = select_gate(fused, bank['labels'], glosses[mask])
    rescue_policy = fit_rescue(base, retrieval, bank['labels'], glosses[mask], baseline_policy)
    fit = {'selection': 'Maximize development top-1 on the fixed mixture grid. Acceptance policies use a 1% development '
                        'wrong-known limit, including any additional accepted fallback predictions.',
           'exemplar_freeze_sha256': sha256(source), 'classifier_freeze_sha256': sha256(model_freeze),
           'classifier_dev_scores_sha256': sha256(dev_path), 'weight_grid': weights, 'weight': weight,
           'baseline_policy': baseline_policy, 'weighted_policy': weighted_policy, 'rescue_policy': rescue_policy,
           'development': {'classifier': summarize(base, bank['labels'], glosses[mask], baseline_policy)[0],
                           'weighted': summarize(fused, bank['labels'], glosses[mask], weighted_policy)[0],
                           'rescue': rescue_decisions(base, retrieval, bank['labels'], glosses[mask], baseline_policy, rescue_policy)[0]}}
    write_json(output, fit)
    evaluate_fusion(cache, report, arm)


def evaluate_fusion(cache, report, arm):
    source = report / 'freeze.json'
    output = report / f'fusion-{arm}-freeze.json'
    fit = json.loads(output.read_text(encoding='utf-8'))
    freeze = json.loads(source.read_text(encoding='utf-8'))
    model_freeze = CLASSIFIER / f'{arm}-freeze.json'
    dev_path = CLASSIFIER / f'{arm}-dev-scores.npz'
    if (sha256(source) != fit['exemplar_freeze_sha256'] or sha256(model_freeze) != fit['classifier_freeze_sha256']
            or sha256(dev_path) != fit['classifier_dev_scores_sha256']):
        raise ValueError('Frozen fusion model, retrieval bank, or development scores changed')
    record = freeze['variants']['all_verified_labels']
    bank = read_bank(record)
    official = annotations(cache / 'gsl_metadata.zip')
    held_path = cache / TEST
    held, ids, glosses, _ = open_source(held_path, official['test'], role='test')
    score_file = CLASSIFIER / f'{arm}-scores.npz'
    logits = align_scores(score_file, ids, glosses, bank['labels'])
    matrix, examples = scores(bank, held)
    retrieval = matrix[record['selected']]
    base = probabilities(logits)
    fused = probabilities(mix(logits, retrieval, fit['weight']))
    rescue, guesses, accepted, rescued = rescue_decisions(base, retrieval, bank['labels'], glosses, fit['baseline_policy'], fit['rescue_policy'])
    reference_ids = reference_examples(bank, examples, guesses)
    weighted, rank, weighted_accepts = summarize(fused, bank['labels'], glosses, fit['weighted_policy'])
    weighted_references = reference_examples(bank, examples, rank[:, 0])
    result = {'classifier': summarize(base, bank['labels'], glosses, fit['baseline_policy'])[0],
              'weighted': weighted, 'rescue': rescue}
    rows = []
    for i, identity in enumerate(ids):
        rows.append({'id': str(identity), 'reference_gloss': str(glosses[i]), 'predicted_gloss': str(bank['labels'][guesses[i]]),
                     'supporting_lexical_unit': reference_ids[i], 'accepted': bool(accepted[i]), 'rescued': bool(rescued[i]),
                     'weighted_gloss': str(bank['labels'][rank[i, 0]]), 'weighted_accepted': bool(weighted_accepts[i]),
                     'weighted_supporting_lexical_unit': weighted_references[i]})
    write_json(report / f'fusion-{arm}-benchmark.json', {
        'scope': 'Native isolated held signer. Library exemplar support is source-verified, and threshold and mixture selection '
                 'use signer 4. Evaluation only covers known CERTH classes, with no out-of-distribution proof.',
        'freeze_sha256': sha256(output), 'held_pose_sha256': sha256(held_path),
        'held_classifier_scores_sha256': sha256(score_file), 'held': result, 'rows': rows,
    })
    print(json.dumps({'weight': fit['weight'], 'held': result}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate', 'fuse', 'evaluate-fusion'))
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--expanded', action='store_true', help='Use the separately verified larger pose bank')
    parser.add_argument('--classifier-arm', choices=('full', 'expanded'))
    args = parser.parse_args()
    if args.expanded:
        if args.report == REPORT:
            args.report = REPORT.with_name(REPORT.name + '-expanded')
        if args.artifacts == ARTIFACTS:
            args.artifacts = ARTIFACTS.with_name(ARTIFACTS.name + '-expanded')
    torch.set_num_threads(2)
    if args.phase == 'fit':
        fit(args.cache, args.report, args.artifacts, args.expanded)
    elif args.phase == 'evaluate':
        evaluate(args.cache, args.report)
    elif args.phase == 'fuse':
        fuse(args.cache, args.report, args.classifier_arm or ('expanded' if args.expanded else 'full'))
    else:
        evaluate_fusion(args.cache, args.report, args.classifier_arm or ('expanded' if args.expanded else 'full'))


if __name__ == '__main__':
    main()
