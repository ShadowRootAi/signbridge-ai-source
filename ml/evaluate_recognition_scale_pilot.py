"""Fit a landmark TCN and centroid baseline and replay a signer-disjoint MP4 pilot."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import numpy as np
import torch

from common import (FeatureDataset, classification_metrics, decisions, fit_temperature, softmax,
                    tune_rejection)
from preprocess import FEATURE_SCHEMA
from train import TemporalClassifier, normalize_numpy, predict, seed_everything, train_model


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def reference(row):
    return row['label'] if row['is_known'] else row['original_text']


def text_key(value):
    return ''.join(c for c in unicodedata.normalize('NFD', value.casefold().strip(' !?;.,·…'))
                   if unicodedata.category(c) != 'Mn')


def load_inputs(manifest):
    info = json.loads(manifest.read_bytes())
    if not info['complete']:
        raise ValueError('The frozen acquisition cohort is incomplete')
    samples = info['samples']
    if len({row['id'] for row in samples}) != len(samples) or len({row['sha256'] for row in samples}) != len(samples):
        raise ValueError('Duplicate clip ID or video checksum')
    roles = ('train', 'validation', 'test')
    if set(row['role'] for row in samples) != set(roles):
        raise ValueError('Expected three independent roles')
    signers = {role: {row['signer'] for row in samples if row['role'] == role} for role in roles}
    if sum(map(len, signers.values())) != len(set().union(*signers.values())):
        raise ValueError('A signer occurs in multiple roles')
    labels = {role: {row['label'] for row in samples if row['role'] == role and row['is_known']} for role in roles}
    unknown = {role: {text_key(reference(row)) for row in samples if row['role'] == role and not row['is_known']} for role in roles}
    if (len(labels['train']) < 2 or any(label_set != labels['train'] for label_set in labels.values())
            or unknown['train'] or not unknown['validation'] or not unknown['test']
            or unknown['validation'] & unknown['test']
            or set(map(text_key, labels['train'])) & (unknown['validation'] | unknown['test'])):
        raise ValueError('Evaluation requires all known words and disjoint unknown words')
    return samples, sorted(labels['train'])


def arrays(samples, labels, features, manifest, roles):
    samples = [row for row in samples if row['role'] in roles]
    ids = {row['id'] for row in samples}
    records = {}
    with features.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['id'] in ids:
                if record['id'] in records:
                    raise ValueError('Duplicate feature row')
                records[record['id']] = record
    if records.keys() != ids:
        raise ValueError(f'Missing extracted clips: {sorted(ids - records.keys())}')
    aligned = [records[row['id']] for row in samples]
    if any(record['fingerprint'] != row['sha256'] for record, row in zip(aligned, samples, strict=True)):
        raise ValueError('The extracted video checksums disagree with the manifest')
    expected_manifest = digest(manifest)
    if ({record['manifestSha256'] for record in aligned} != {expected_manifest}
            or {record['featureSchema'] for record in aligned} != {FEATURE_SCHEMA}
            or len({record['extractorSha256'] for record in aligned}) != 1):
        raise ValueError('Incompatible or mixed feature provenance')
    x = np.asarray([record['x'] for record in aligned], dtype=np.float32)
    quality = np.asarray([record['tracking']['eitherHandDetection'] for record in aligned], dtype=np.float32)
    motion = np.asarray([record['motion']['motionBodyRms'] for record in aligned])
    if x.shape != (len(samples), 48, 224) or not np.isfinite(x).all() or not np.isfinite(quality).all() or not np.isfinite(motion).all():
        raise ValueError('Invalid landmarks, tracking or motion')
    # The zero quality of a static or short clip blocks acceptance, but the clip is still counted.
    quality[(motion < .004) | (np.asarray([record['frames'] for record in aligned]) < 6)] = 0
    labels_index = {label: index for index, label in enumerate(labels)}
    return FeatureDataset(x, np.array([labels_index[row['label']] if row['is_known'] else -1 for row in samples]),
                          np.array([row['signer'] for row in samples]), np.array([row['id'] for row in samples]),
                          np.array(['val' if row['role'] == 'validation' else row['role'] for row in samples], dtype='U8'), labels,
                          np.array([reference(row) for row in samples]), quality,
                          {'featureSchema': FEATURE_SCHEMA, 'extractor_sha256': aligned[0]['extractorSha256']})


def centroid_scores(x, mean, std, keep, centers):
    normalized = normalize_numpy(x, mean, std, keep)
    return -np.square(normalized[:, None] - centers[None, :]).mean(axis=(2, 3)) * 12


def scores_and_policy(logits, y, quality, count):
    known = y >= 0
    temperature = fit_temperature(logits[known], y[known])
    probability = softmax(logits, temperature)
    policy = tune_rejection(probability, y, np.where(quality >= .6, quality, 0), count,
                            target_far=0, target_accepted_accuracy=1)
    return {**policy, 'temperature': temperature}


def active_classes(data, labels):
    kept = (data.y < 0) | np.isin(data.raw_label, labels)
    indices = {label: index for index, label in enumerate(labels)}
    remapped = np.array([indices[label] if target >= 0 else -1
                         for label, target in zip(data.raw_label[kept], data.y[kept], strict=True)])
    return replace(data, classes=labels, x=data.x[kept], y=remapped, raw_label=data.raw_label[kept],
                   split=data.split[kept], source_id=data.source_id[kept], signer=data.signer[kept], quality=data.quality[kept])


def fit(data, output, manifest):
    planned = data.classes
    usable = data.mask('train') & (data.quality >= .6)
    eligibility = {label: sorted(set(data.signer[usable & (data.raw_label == label)])) for label in planned}
    active = [label for label in planned if len(eligibility[label]) >= 2]
    if len(active) < 2:
        raise ValueError('Fewer than two labels have usable fit clips from two people')
    # A label with insufficient fit coverage is excluded before calibration and held scoring.
    data = active_classes(data, active)
    fit_rows = data.mask('train')
    val_rows = data.mask('val')
    # Keep the measured tracking on validation so poor inputs count in outcomes.
    good_train = fit_rows & (data.quality >= .6)
    data.split[fit_rows & ~good_train] = 'unusable'
    args = SimpleNamespace(seed=42, channels=48, learning_rate=.001, epochs=70, patience=12, batch_size=8)
    seed_everything(args.seed)
    network, training, mean, std, keep = train_model(data, args)
    normalized = normalize_numpy(data.x[good_train], mean, std, keep)
    centers = np.stack([normalized[data.y[good_train] == index].mean(axis=0) for index in range(len(data.classes))])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output / 'temporal.pt')
    np.savez_compressed(output / 'baseline.npz', mean=mean, std=std, keep=keep, centers=centers)
    validation = data.x[val_rows]
    y, quality = data.y[val_rows], data.quality[val_rows]
    policies = {
        'temporal': scores_and_policy(predict(network, validation), y, quality, len(data.classes)),
        'centroid': scores_and_policy(centroid_scores(validation, mean, std, keep, centers), y, quality, len(data.classes)),
    }
    write_json(output / 'freeze.json', {'manifest_sha256': digest(manifest), 'classes': data.classes,
        'planned_classes': planned, 'usable_fit_signers_by_label': eligibility,
        'extractor_sha256': data.metadata['extractor_sha256'],
        'model_sha256': {name: digest(output / name) for name in ('temporal.pt', 'baseline.npz')},
        'training': training, 'train_usable': int(good_train.sum()), 'train_total': int(fit_rows.sum()),
        'train_usable_by_label': dict(Counter(data.raw_label[good_train])), 'policies': policies})


def score(data, output, report_path, manifest, rows):
    freeze = json.loads((output / 'freeze.json').read_bytes())
    if (freeze['manifest_sha256'] != digest(manifest) or freeze['planned_classes'] != data.classes
            or freeze['extractor_sha256'] != data.metadata['extractor_sha256']):
        raise ValueError('The evaluation plan changed after fitting')
    dropped = set(data.classes) - set(freeze['classes'])
    excluded_held = [str(identity) for identity, label, target in zip(data.source_id, data.raw_label, data.y, strict=True)
                     if target >= 0 and label in dropped]
    data = active_classes(data, freeze['classes'])
    for name, expected in freeze['model_sha256'].items():
        if digest(output / name) != expected:
            raise ValueError('The model changed after fitting')
    with np.load(output / 'baseline.npz', allow_pickle=False) as baseline:
        mean, std, keep, centers = (baseline[name] for name in ('mean', 'std', 'keep', 'centers'))
    seed_everything(42)
    network = TemporalClassifier(mean, std, len(data.classes), keep=keep)
    network.load_state_dict(torch.load(output / 'temporal.pt', map_location='cpu', weights_only=True))
    results = {}
    for name, logits in {'temporal': predict(network, data.x),
                         'centroid': centroid_scores(data.x, mean, std, keep, centers)}.items():
        policy = freeze['policies'][name]
        probabilities = softmax(logits, policy['temperature'])
        predicted, accepted, confidence = decisions(probabilities, np.where(data.quality >= .6, data.quality, 0),
                                                   len(data.classes), policy['confidence_threshold'], policy['min_quality'])
        results[name] = {'metrics': classification_metrics(data.y, predicted, accepted, len(data.classes)),
                         'counts': {'correct_top1': int(((data.y >= 0) & (data.y == predicted)).sum()),
                                    'correct_accepted': int((accepted & (data.y == predicted)).sum()),
                                    'wrong_accepted': int((accepted & (data.y >= 0) & (data.y != predicted)).sum()),
                                    'off_vocabulary_accepted': int((accepted & (data.y < 0)).sum())},
                         'rows': [{'id': str(identity), 'signer': str(signer), 'reference': str(label),
                                   'known': bool(target >= 0), 'prediction': data.classes[guess],
                                   'accepted': bool(take), 'score': float(probability), 'quality': float(quality)}
                                  for identity, signer, label, target, guess, take, probability, quality
                                  in zip(data.source_id, data.signer, data.raw_label, data.y, predicted, accepted,
                                         confidence, data.quality, strict=True)]}
    split = {role: {'signers': sorted({row['signer'] for row in rows if row['role'] == role}),
                    'clips': sum(row['role'] == role for row in rows)} for role in ('train', 'validation', 'test')}
    report = {'scope': 'Small studio MP4 pilot with whole-clip Greek transcript targets, split by documented signer ID. '
                       'Greek text is weak sign-level supervision; this does not verify shared gloss identity or webcam accuracy.',
              'classes': data.classes, 'split': split, 'manifest_sha256': freeze['manifest_sha256'],
              'extractor_sha256': freeze['extractor_sha256'], 'freeze_sha256': digest(output / 'freeze.json'),
              'excluded_labels_by_fit_coverage': sorted(dropped), 'excluded_held_clip_ids': excluded_held,
              'usable_fit_signers_by_label': freeze['usable_fit_signers_by_label'],
              'fit_usable': freeze['train_usable'], 'policies': freeze['policies'], 'held': results}
    write_json(report_path, report)
    print(json.dumps({name: value['metrics'] for name, value in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('features', type=Path)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/recognition-scale-pilot'))
    parser.add_argument('--report', type=Path, default=Path('reports/recognition-scale-pilot/benchmark.json'))
    args = parser.parse_args()
    samples, labels = load_inputs(args.manifest)
    if args.phase == 'fit':
        fit(arrays(samples, labels, args.features, args.manifest, {'train', 'validation'}), args.artifacts, args.manifest)
    else:
        score(arrays(samples, labels, args.features, args.manifest, {'test'}), args.artifacts, args.report, args.manifest, samples)
