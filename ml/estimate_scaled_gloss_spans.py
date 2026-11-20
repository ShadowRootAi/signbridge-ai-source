"""Estimate train-only CERTH gloss windows by matching same-take isolated poses."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from train_continuous_hard_timing import (
    ISOLATE_PATH, MAX_DISTANCE, MIN_REVERSE_GAIN, ROOT, SENTENCE_PATH, SPLIT, SUPPLEMENTARY,
    align_windows, frame_feature_array, grouped_labels, read_annotations, read_labels, verify_feature_math,
)


COHORT = ROOT / 'data/continuous-scale'
CACHE = ROOT / 'research/cache/openhands-gsl'
POSE_FILES = (CACHE / 'official-train-prototypes-poses.npz', CACHE / 'official-train-expanded-poses.npz')
SIGNERS = {1, 2, 4, 5}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def candidate_ids(rows):
    canonical, _ = read_annotations(ROOT / SPLIT, ROOT / SUPPLEMENTARY)
    sd = grouped_labels(canonical.items(), SENTENCE_PATH)
    with ZipFile(ROOT / SPLIT) as archive:
        prefix = 'GSL_split/GSL_continuous/gsl_split_SI_'
        si = grouped_labels(((path.split('/', 1)[1], tuple(labels.split())) for path, labels in
                             read_labels(archive, [f'{prefix}{split}.csv' for split in ('train', 'dev', 'test')])),
                            SENTENCE_PATH)
        iso_labels = dict(read_labels(archive, ['GSL_split/GSL_isolated/train_greek_iso.csv']))
    isolates = grouped_labels(iso_labels.items(), ISOLATE_PATH)
    verified = {}
    for session, sentences in si.items():
        if sentences != sd.get(session) or sorted(sentences) != list(range(len(sentences))):
            continue
        expected = tuple(gloss for _, phrase in sorted(sentences.items()) for gloss in phrase)
        observed = isolates.get(session, {})
        if tuple(observed.get(i) for i in range(len(observed))) == expected:
            verified[session] = sentences
    candidates = {}
    for identifier, row in rows.items():
        source = SENTENCE_PATH.fullmatch(row['source_path'].split('/', 1)[1])
        if source is None or int(source['signer']) != row['signer'] or row['signer'] not in SIGNERS or row['fps'] != 30:
            raise ValueError(f'Unexpected training source: {identifier}')
        session, index = source['session'], int(source['index'])
        sentences = verified.get(session)
        if sentences is None:
            continue
        glosses = tuple(row['glosses'])
        if sentences[index] != glosses:
            raise ValueError(f'Manifest glosses differ from the official annotations: {identifier}')
        start = sum(len(sentences[i]) for i in range(index))
        candidates[identifier] = tuple(f'{session}_glosses/glosses{i:04d}' for i in range(start, start + len(glosses)))
    return candidates, iso_labels


def pose_bank(candidates, labels):
    wanted = {item for ids in candidates.values() for item in ids}
    bank = {}
    for path in POSE_FILES:
        with np.load(path, allow_pickle=False) as archive:
            poses, ids, offsets, glosses = (archive[name] for name in ('keypoints', 'ids', 'offsets', 'glosses'))
            if (len(ids) != len(set(ids)) or len(ids) != len(glosses) or len(offsets) != len(ids) + 1
                    or offsets[0] != 0 or offsets[-1] != len(poses) or np.any(np.diff(offsets) <= 0)):
                raise ValueError(f'Malformed pose cache: {path}')
            for index, (identity, gloss) in enumerate(zip(ids, glosses, strict=True)):
                if identity not in wanted:
                    continue
                if labels[identity] != gloss:
                    raise ValueError(f'Isolated label differs: {identity}')
                segment = poses[offsets[index]:offsets[index + 1]]
                if not np.isfinite(segment).all():
                    raise ValueError(f'Nonfinite isolated segment: {identity}')
                if identity in bank:
                    if not np.array_equal(bank[identity], segment):
                        raise ValueError(f'Isolated source changed across caches: {identity}')
                else:
                    bank[str(identity)] = segment.copy()
    return bank


def build(manifest, features):
    manifest_hash, feature_hash = digest(manifest), digest(features)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    training = [row for row in data['samples'] if row['split'] == 'train']
    rows = {row['id']: row for row in training}
    if not data['complete'] or len(rows) != len(training):
        raise ValueError('Expected a complete acquisition with distinct training sources')
    candidates, labels = candidate_ids(rows)
    bank = pose_bank(candidates, labels)
    available = {identity: ids for identity, ids in candidates.items() if all(item in bank for item in ids)}
    retained = {item for ids in available.values() for item in ids}
    lengths = np.cumsum([0, *(len(bank[item]) for item in sorted(retained))])
    raw = np.concatenate([bank[item] for item in sorted(retained)])
    pose_features = frame_feature_array(raw)
    verify_feature_math(raw, pose_features)
    converted = {item: pose_features[start:end] for item, start, end in
                 zip(sorted(retained), lengths[:-1], lengths[1:], strict=True)}

    spans, evidence, compared, failures = {}, {}, set(), Counter()
    with features.open(encoding='utf-8') as stream:
        for line in stream:
            feature = json.loads(line)
            identity = feature['id']
            if identity not in available:
                continue
            row, isolates = rows[identity], available[identity]
            source_key = sha256(json.dumps([row['fps'], row['frame_paths']], ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest()
            sequence = np.asarray(feature['features'], dtype=np.float32)
            if (identity in compared or source_key != feature['source_key'] or feature['fps'] != 15
                    or sequence.shape != ((row['frame_count'] + 1) // 2, 224) or not np.isfinite(sequence).all()):
                raise ValueError(f'Browser feature source/schema mismatch: {identity}')
            compared.add(identity)
            segments = [converted[item] for item in isolates]
            forward, windows = align_windows(sequence, segments)
            reverse, _ = align_windows(sequence[::-1], segments)
            if not np.isfinite(forward) or not np.isfinite(reverse):
                failures['no_finite_forward_and_reverse'] += 1
                continue
            if forward > MAX_DISTANCE or reverse - forward < MIN_REVERSE_GAIN:
                failures['failed_forward_distance_or_reverse_gain'] += 1
                continue
            spans[identity] = [[2 * start, min(2 * end, row['frame_count']), gloss]
                               for (start, end), gloss in zip(windows, row['glosses'], strict=True)]
            evidence[identity] = {'source_path': row['source_path'], 'source_rgb_sha256': row['sha256'],
                                  'browser_source_key': source_key, 'extractor_sha256': feature['extractor_sha256'],
                                  'isolated_source_ids': isolates, 'source_frame_count': row['frame_count'],
                                  'forward_distance': forward, 'reverse_distance': reverse}
    if set(available) != compared or manifest_hash != digest(manifest) or feature_hash != digest(features):
        raise ValueError('Inputs are incomplete or changed during alignment')
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / SPLIT, ROOT / SUPPLEMENTARY, *POSE_FILES)}
    counts = {'training_clips': len(rows), 'exact_metadata_order': len(candidates), 'compared': len(compared),
              'kept': len(spans), 'segments': sum(map(len, spans.values())), **failures,
              'kept_by_signer': dict(Counter(rows[identity]['signer'] for identity in spans))}
    return {'scope': 'Train-only visually estimated same-recording windows. No official per-gloss timestamps.',
            'manifest_sha256': manifest_hash, 'features_sha256': feature_hash, 'source_sha256': hashes,
            'span_format': '30 Hz original RGB [start_inclusive, end_exclusive, gloss]',
            'distance_max': MAX_DISTANCE, 'reverse_gain_min': MIN_REVERSE_GAIN, 'counts': counts,
            'spans': spans, 'evidence': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=COHORT / 'manifest.json')
    parser.add_argument('--features', type=Path, default=COHORT / 'features.fit.jsonl')
    parser.add_argument('--output', type=Path, default=COHORT / 'estimated-gloss-spans.json')
    parser.add_argument('--min-kept', type=int, default=30)
    args = parser.parse_args()
    if args.min_kept < 1:
        parser.error('--min-kept must be positive')
    result = build(args.manifest, args.features)
    print(json.dumps(result['counts'], ensure_ascii=False), flush=True)
    if result['counts']['kept'] >= args.min_kept:
        with args.output.open('x', encoding='utf-8') as target:
            json.dump(result, target, ensure_ascii=False, indent=2)
            target.write('\n')


if __name__ == '__main__':
    main()
