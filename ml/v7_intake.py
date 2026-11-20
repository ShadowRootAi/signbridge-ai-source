"""Validate local future-recording manifests; never train, upload or verify identity.

CLI output contains only aggregate counts, digests and indexed error codes, not
participant IDs or local paths. A pass proves this contract, not GSL correctness,
consent authenticity, new identity, or independence from all prior recordings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KNOWN_LABELS = frozenset(('ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ'))
SPLITS = frozenset(('train', 'calibration', 'final'))
SHA256 = re.compile(r'[0-9a-f]{64}')
TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
SIGNER = re.compile(r'P[0-9]{3,6}')
CATEGORY = re.compile(r'u_[a-z0-9][a-z0-9_-]{0,47}')
VIDEO_SUFFIXES = frozenset(('.mp4', '.webm', '.mov', '.m4v', '.avi', '.mkv'))
GSL_REFERENCES = tuple(ROOT / name for name in (
    'data/manifest.json', 'data/open-set-v2/manifest.json', 'data/v3/manifest.json',
    'data/clips/mapping.json', 'data/open-set-v2/clips/mapping.json',
    'web/public/library/catalog.json',
))
REQUIRED_SAMPLE = frozenset((
    'clip_id', 'signer_id', 'split', 'consent_local_ml', 'human_recorded',
    'new_participant_declared', 'label', 'source_path', 'source_sha256',
    'clip_path', 'clip_sha256',
))
OPTIONAL_SAMPLE = frozenset(('session_id', 'condition_id'))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key')
        value[key] = item
    return value


def _reject_constant(_):
    raise ValueError('Nonfinite JSON')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_keys,
                      parse_constant=_reject_constant)


def _matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def reference_digests(paths):
    """Read existing GSL manifests/mappings. Missing references fail closed."""
    blocked, evidence = set(), []
    for path in paths:
        path = Path(path)
        document = read_json(path)
        if not isinstance(document, dict):
            raise ValueError('Invalid reference')
        if isinstance(document.get('samples'), list):
            records = document['samples']
        elif isinstance(document.get('clips'), dict):
            records = list(document['clips'].values())
        elif isinstance(document.get('entries'), list):
            records = document['entries']
        else:
            raise ValueError('Unsupported reference schema')
        found = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError('Invalid reference record')
            for key in ('sha256', 'mp4_sha256', 'source_frame_sequence_sha256'):
                value = record.get(key)
                if _matches(SHA256, value):
                    found.add(value)
            for key in ('frame_sha256', 'source_frame_sha256'):
                for value in record.get(key, []):
                    if not _matches(SHA256, value):
                        raise ValueError('Invalid reference digest')
                    found.add(value)
        if not found:
            raise ValueError('Reference has no digests')
        blocked.update(found)
        evidence.append(digest(path))
    if not blocked:
        raise ValueError('No reference digests')
    return blocked, evidence


def validate_manifest(manifest_path: Path, reference_paths=None) -> dict:
    """Check declarations, paths, content hashes and split invariants only.

    reference_paths is an explicit registry injection for unit tests/other local
    callers. The CLI always checks every built-in GSL reference above.
    """
    issues = []
    result = {
        'schema_version': 1, 'valid': False, 'status': 'manifest_checks_failed',
        'sample_count': 0, 'split_counts': {}, 'signer_count': 0,
        'verified_media_file_count': 0, 'issues': issues,
        'assurance': 'declarations_and_exact_file_checks_only',
        'human_identity_or_consent_authenticity_verified': False,
        'linguistic_labels_verified': False, 'independent_evaluation_verified': False,
    }

    def error(code, index=None):
        issue = {'code': code}
        if index is not None:
            issue['sample_index'] = index
        issues.append(issue)

    try:
        path = Path(manifest_path).resolve(strict=True)
        document = read_json(path)
        result['manifest_sha256'] = digest(path)
    except (OSError, ValueError, UnicodeError, RuntimeError, RecursionError):
        error('manifest_unreadable_or_invalid_json')
        return result
    try:
        blocked, evidence = reference_digests(GSL_REFERENCES if reference_paths is None else reference_paths)
        result['reference_file_sha256'] = evidence
        result['blocked_gsl_digest_count'] = len(blocked)
    except (OSError, ValueError, TypeError, UnicodeError, RuntimeError, RecursionError):
        error('gsl_reference_registry_unavailable_or_invalid')
        return result
    top_keys = {'schema_version', 'novel_final_unknown_categories', 'samples'}
    if not isinstance(document, dict) or set(document) != top_keys or type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        error('manifest_schema_invalid')
        return result
    novel = document['novel_final_unknown_categories']
    if not isinstance(novel, list) or not all(_matches(CATEGORY, c) for c in novel) or len(set(novel)) != len(novel):
        error('novel_category_registry_invalid')
        return result
    samples = document['samples']
    if not isinstance(samples, list) or not samples:
        error('samples_must_be_nonempty_list')
        return result
    result['sample_count'] = len(samples)
    seen_ids, signer_splits, category_splits, hash_owners = set(), {}, {}, {}
    split_counts, observed_final = Counter(), set()
    file_cache = {}

    def media(sample, field, index):
        name, expected = sample[field + '_path'], sample[field + '_sha256']
        if not isinstance(name, str) or not name or not _matches(SHA256, expected):
            error(field + '_path_or_digest_invalid', index)
            return None
        relative = Path(name.replace('\\', '/'))
        # Both path separators are accepted, but absolute/escaping paths are not.
        if relative.is_absolute() or relative.drive or ':' in name or relative.suffix.lower() not in VIDEO_SUFFIXES:
            error(field + '_path_invalid', index)
            return None
        try:
            resolved = (path.parent / relative).resolve(strict=True)
            if not resolved.is_relative_to(path.parent) or not resolved.is_file():
                error(field + '_path_outside_collection', index)
                return None
            if resolved not in file_cache:
                if resolved.stat().st_size == 0:
                    error(field + '_file_empty', index)
                    return None
                file_cache[resolved] = digest(resolved)
            actual = file_cache[resolved]
        except (OSError, ValueError, RuntimeError):
            error(field + '_file_unreadable', index)
            return None
        if actual != expected:
            error(field + '_digest_mismatch', index)
        if actual in blocked or expected in blocked:
            error('existing_gsl_source_detected', index)
        owner = hash_owners.setdefault(actual, index)
        if owner != index:
            error('duplicate_clip_or_original_source', index)
        return actual

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not REQUIRED_SAMPLE.issubset(sample) or set(sample) - REQUIRED_SAMPLE - OPTIONAL_SAMPLE:
            error('sample_schema_invalid', index)
            continue
        clip_id, signer, split = sample['clip_id'], sample['signer_id'], sample['split']
        if not _matches(TOKEN, clip_id):
            error('clip_id_invalid', index)
        elif clip_id in seen_ids:
            error('duplicate_clip_id', index)
        else:
            seen_ids.add(clip_id)
        for field in OPTIONAL_SAMPLE:
            if field in sample and not _matches(TOKEN, sample[field]):
                error(field + '_invalid', index)
        if not _matches(SIGNER, signer):
            error('signer_pseudonym_invalid', index)
        if not isinstance(split, str) or split not in SPLITS:
            error('split_invalid', index)
        else:
            split_counts[split] += 1
            if _matches(SIGNER, signer) and signer_splits.setdefault(signer, split) != split:
                error('signer_crosses_splits', index)
        for declaration in ('consent_local_ml', 'human_recorded', 'new_participant_declared'):
            if sample[declaration] is not True:
                error(declaration + '_required', index)
        label = sample['label']
        if isinstance(label, dict) and label.get('kind') == 'known':
            if set(label) != {'kind', 'id'} or not isinstance(label.get('id'), str) or label['id'] not in KNOWN_LABELS:
                error('known_label_invalid', index)
        elif isinstance(label, dict) and label.get('kind') == 'unknown':
            category = label.get('category_id')
            if set(label) != {'kind', 'category_id'} or not _matches(CATEGORY, category):
                error('unknown_category_invalid', index)
            elif isinstance(split, str) and split in SPLITS:
                category_splits.setdefault(category, set()).add(split)
                if split == 'final':
                    observed_final.add(category)
                    if category not in novel:
                        error('final_unknown_not_declared_novel', index)
                elif category in novel:
                    error('novel_final_category_in_development', index)
        else:
            error('label_kind_invalid', index)
        media(sample, 'source', index)
        media(sample, 'clip', index)
    if set(novel) != observed_final:
        error('novel_registry_does_not_match_final_samples')
    if any('final' in roles and len(roles) > 1 for roles in category_splits.values()):
        error('unknown_category_crosses_final_boundary')
    result.update(split_counts=dict(split_counts), signer_count=len(signer_splits),
                  verified_media_file_count=len(file_cache))
    result['valid'] = not issues
    if result['valid']:
        result['status'] = 'manifest_checks_passed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    return 0 if report['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
