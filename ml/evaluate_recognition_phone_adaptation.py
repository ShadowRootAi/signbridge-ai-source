"""Retrain V2 with documented phone recordings and evaluate held handset footage."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from evaluate_recognition_scale_pilot import arrays, digest, reference, text_key, write_json
from evaluate_recognition_scale_v2 import MIN_TRACKING, ROOT, evaluate_arms, fit_arrays, restore


STUDIO = ROOT / '.runtime-tmp/recognition-scale-v2'
V2 = ROOT / 'artifacts/recognition-scale-v2'
ARTIFACTS = ROOT / 'artifacts/recognition-phone-adaptation'
REPORT = ROOT / 'reports/recognition-phone-adaptation/benchmark.json'


def read_manifest(path):
    value = json.loads(path.read_bytes())
    if not value['complete']:
        raise ValueError(f'Incomplete recording cohort: {path}')
    return value['samples']


def verify_split(studio, phone, labels):
    possible_roles = ('train', 'validation', 'test')
    if any(row['role'] not in possible_roles for row in phone) or not any(row['role'] == 'test' for row in phone):
        raise ValueError('Phone data requires an explicit held test')
    if any(row['is_known'] and row['label'] not in labels for row in phone):
        raise ValueError('Phone positive does not name exactly one V2 target')
    for key in ('id', 'sha256', 'original_sha256'):
        identifiers = [row[key] for row in studio + phone]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f'Duplicate source recordings: {key}')
    people, known_words, unknown_words = set(), set(map(text_key, labels)), set()
    for role in possible_roles:
        group = [row for row in studio + phone if row['role'] == role]
        role_people = {row['signer'] for row in group}
        role_unknown = {text_key(reference(row)) for row in group if not row['is_known']}
        if role_people & people or role_unknown & (known_words | unknown_words):
            raise ValueError('Cohorts share signers or out-of-scope labels across roles')
        people |= role_people
        unknown_words |= role_unknown
    phone_glosses = set()
    for role in possible_roles:
        role_glosses = {text_key(row['gloss']) for row in phone if row['role'] == role and not row['is_known'] and row.get('gloss')}
        if role_glosses & (known_words | phone_glosses):
            raise ValueError('Phone OTHER glosses must be disjoint across roles and from known words')
        phone_glosses |= role_glosses


def combine(studio, phone):
    if studio.metadata != phone.metadata or studio.classes != phone.classes:
        raise ValueError('Phone and studio features must share the exact preprocessing contract')
    return replace(studio, **{name: np.concatenate([getattr(studio, name), getattr(phone, name)])
                             for name in ('x', 'y', 'signer', 'source_id', 'split', 'raw_label', 'quality')})


def check_model(directory):
    frozen = json.loads((directory / 'freeze.json').read_bytes())
    for name, expected in frozen['model_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'Unexpected checkpoint bytes: {directory / name}')
    return frozen


def freeze_binding(args, v2_model):
    return {'studio_manifest_sha256': digest(args.studio / 'manifest.json'),
            'studio_features_sha256': digest(args.studio / 'features.jsonl'),
            'phone_manifest_sha256': digest(args.phone_manifest), 'phone_features_sha256': digest(args.phone_features),
            'base_checkpoint_sha256': v2_model['model_sha256']['model.pt']}


def count_roles(rows):
    result = {}
    for role in ('train', 'validation', 'test'):
        group = [row for row in rows if row['role'] == role]
        result[role] = {'clips': len(group), 'known': sum(row['is_known'] for row in group),
                        'signers': sorted({row['signer'] for row in group}),
                        'words': sorted({reference(row) for row in group})}
    return result


def fit(args, studio, phone, labels, binding):
    development = {'train', 'validation'}
    if not any(row['role'] == 'train' for row in phone):
        print('No verified phone training recordings. The frozen V2 can still be evaluated.')
        return
    studio_data = arrays(studio, labels, args.studio / 'features.jsonl', args.studio / 'manifest.json', development)
    phone_data = arrays(phone, labels, args.phone_features, args.phone_manifest, development)
    usable = phone_data.mask('train') & (phone_data.quality >= MIN_TRACKING)
    if not usable.any():
        print('Phone training recordings failed V2 tracking/motion checks. No adapted model was fitted.')
        return
    fit_arrays(combine(studio_data, phone_data), args.artifacts,
               {**binding, 'phone_fit_used': int(usable.sum()), 'phone_fit_known_used': int((usable & (phone_data.y >= 0)).sum()),
                'phone_fit_signers': sorted(set(phone_data.signer[usable])), 'phone_source': count_roles(phone)})


def evaluate(args, studio, phone, labels, binding, v2_model):
    cohorts = {'phone_held': arrays(phone, labels, args.phone_features, args.phone_manifest, {'test'}),
               'observed_studio_regression': arrays(studio, labels, args.studio / 'features.jsonl', args.studio / 'manifest.json', {'test'})}
    if args.external_manifest:
        extra = read_manifest(args.external_manifest)
        cohorts['external_non_phone_known_diagnostic'] = arrays(extra, labels, args.external_features, args.external_manifest, {'test'})
    if args.window_manifest:
        windows = read_manifest(args.window_manifest)
        held = {row['id']: row['sha256'] for row in phone if row['role'] == 'test'}
        if {row['parent_id']: row['parent_mp4_sha256'] for row in windows} != held:
            raise ValueError('Phone windows must replay the same protected held parents')
        cohorts['phone_window_diagnostic'] = arrays(windows, labels, args.window_features, args.window_manifest, {'test'})
    frozen = {'v2': v2_model}
    directories = {'v2': args.v2}
    if (args.artifacts / 'freeze.json').is_file():
        adapted = check_model(args.artifacts)
        if any(adapted[key] != value for key, value in binding.items()):
            raise ValueError('Adapted model was frozen on a different cohort')
        if adapted['classes'] != labels:
            raise ValueError('Adapted candidate lost a V2 class')
        frozen['adapted'] = adapted
        directories['adapted'] = args.artifacts
    output = {'scope': 'Phone scores apply only to documented handheld-device corpus recordings. '
                       'Separate external camera and previously used studio cohorts are diagnostic. '
                       'Published corpora use documented participant codes; no cross-corpus identity crosswalk was provided.',
              'provenance': binding, 'classes': labels, 'phone_cohort': count_roles(phone), 'models': {}, 'results': {}}
    if args.window_manifest:
        output['phone_window_provenance'] = {'manifest_sha256': digest(args.window_manifest),
            'features_sha256': digest(args.window_features),
            'scope': 'The same held source videos after interpreting the annotation bounds as seconds. '
                     'This reused set is a tracking diagnostic with both original policies held fixed.'}
    if args.external_manifest:
        output['external_camera_manifest_sha256'] = digest(args.external_manifest)
    for name, state in frozen.items():
        temporal, prototype = restore(directories[name])
        output['models'][name] = {'selected_on_validation': state['selected'],
                                  'phone_train_clips': state.get('phone_fit_used', 0),
                                  'phone_train_known_clips': state.get('phone_fit_known_used', 0)}
        for cohort, data in cohorts.items():
            if data.metadata['extractor_sha256'] != state['extractor_sha256']:
                raise ValueError('Mismatched feature extractor during scoring')
            results = evaluate_arms(data, temporal, prototype, state)[state['selected']]
            usable = data.quality >= MIN_TRACKING
            results['quality_gate_eligible'] = int(usable.sum())
            results['unknown_quality_gate_eligible'] = int((usable & (data.y < 0)).sum())
            for row, quality in zip(results['rows'], data.quality, strict=True):
                row['tracking_after_motion_check'] = float(quality)
            output['results'].setdefault(cohort, {})[name] = results
    write_json(args.report, output)
    print(json.dumps({cohort: {name: result['counts'] for name, result in values.items()}
                      for cohort, values in output['results'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate'))
    parser.add_argument('phone_manifest', type=Path)
    parser.add_argument('phone_features', type=Path)
    parser.add_argument('--studio', type=Path, default=STUDIO)
    parser.add_argument('--v2', type=Path, default=V2)
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--external-manifest', type=Path)
    parser.add_argument('--external-features', type=Path)
    parser.add_argument('--window-manifest', type=Path)
    parser.add_argument('--window-features', type=Path)
    args = parser.parse_args()
    if bool(args.external_manifest) != bool(args.external_features):
        parser.error('External recordings require both a manifest and its extracted features')
    if bool(args.window_manifest) != bool(args.window_features):
        parser.error('Phone windows require both a manifest and its extracted features')
    v2_model = check_model(args.v2)
    labels = v2_model['classes']
    studio = read_manifest(args.studio / 'manifest.json')
    phone = read_manifest(args.phone_manifest)
    verify_split(studio, phone, labels)
    binding = freeze_binding(args, v2_model)
    if v2_model['manifest_sha256'] != binding['studio_manifest_sha256']:
        parser.error('The frozen V2 and supplied studio cohort disagree')
    if args.phase == 'fit':
        fit(args, studio, phone, labels, binding)
    else:
        evaluate(args, studio, phone, labels, binding, v2_model)
