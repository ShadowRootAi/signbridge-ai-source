"""Acquire new official CERTH training templates with same-take pose references."""

import argparse
from collections import Counter, defaultdict
import sys
from zipfile import ZipFile

from acquire_continuous_scale import (
    DATA, LIBRARY, METADATA, ROOT, SOURCE, SOURCE_ID, SPLIT, SUPPLEMENTARY, SYNTHETIC, acquire,
    archive_metadata, cached_sizes, digest, normalize_greek, order, overlaps, prior_sources, read_annotations,
    read_json, source_id, write_manifest,
)
from v3_data_diagnosis import annotation_timestamps

sys.path.insert(0, str(ROOT / 'ml'))
from estimate_scaled_gloss_spans import SIGNERS, POSE_FILES, candidate_ids, pose_bank


OUTPUT = DATA / 'continuous-span-expansion'
FRESH_PLAN = DATA / 'continuous-scale-novel/manifest.planned.json'
CAP, TARGET = 540_000_000, 96


def historical_keys(canonical, greek):
    used, previous, held, sources = prior_sources(canonical, greek, excluded=(OUTPUT,))
    sources.extend(DATA / folder / 'manifest.json' for folder in ('continuous-scale', 'continuous-scale-tranche1'))
    sources.append(FRESH_PLAN)
    for path in sources[-3:]:
        for row in read_json(path)['samples']:
            identity = source_id(row['source_path'])
            if identity not in canonical:
                continue
            used.add(identity)
            ref = row['template'], canonical[identity], tuple(normalize_greek(greek[identity]).split())
            previous.add(ref)
            if row['split'] != 'train':
                held.add(ref)
    return used, previous, held, sources


def build_plan():
    canonical, greek = read_annotations(SPLIT, SUPPLEMENTARY)
    with ZipFile(SUPPLEMENTARY) as archive:
        timestamps, _ = annotation_timestamps(archive)
    used, previous, held, sources = historical_keys(canonical, greek)
    templates, gloss_sequences, texts = map(set, zip(*previous))
    meta = archive_metadata()
    rows = {}
    for identity, labels in canonical.items():
        if identity in used or identity not in timestamps or not 2 <= len(labels) <= 9:
            continue
        match = SOURCE_ID.fullmatch(identity)
        scene, signer, repetition = match['scenario'], int(match['signer']), int(match['rep'])
        if signer not in SIGNERS:
            continue
        template = f"{scene}/{match['sentence']}"
        text, timing = greek.get(identity, ''), timestamps[identity]
        normalized = normalize_greek(text)
        words = tuple(normalized.split())
        if (template in templates or labels in gloss_sequences or words in texts or not 3 <= len(words) <= 9
                or normalized != normalize_greek(timing['translation'])
                or any(overlaps((template, labels, words), reference) for reference in held)):
            continue
        archive = scene + '.zip'
        identifier = identity.replace('/', '__')
        rows[identifier] = {'id': identifier, 'source_path': f'{scene}/{identity}', 'source_archive': archive,
                            'source_url': meta[archive]['links']['self'], 'signer': signer, 'repetition': repetition,
                            'take_id': identity.split('/', 1)[0].removesuffix('_sentences'), 'split': 'train',
                            'composition': 'known_template', 'category': 'train', 'text': text, 'glosses': list(labels),
                            'template': template, 'fps': 30, 'start_ms': timing['start_ms'], 'end_ms': timing['end_ms'],
                            'annotated_frame_count': timing['implied_inclusive_frame_count'],
                            'annotation_evidence': {'file': timing['annotation_file'], 'timestamp_file': timing['timestamp_file'],
                                                    'line_1based': timing['line_1based']}}
    candidates, labels = candidate_ids(rows)
    bank = pose_bank(candidates, labels)
    available = defaultdict(dict)
    for identifier, pose_ids in candidates.items():
        if all(item in bank for item in pose_ids):
            row = rows[identifier]
            pose_frames = sum(len(bank[item]) for item in pose_ids)
            if row['annotated_frame_count'] >= 40 and pose_frames <= row['annotated_frame_count']:
                available[row['source_archive']][source_id(row['source_path'])] = {
                    **row, 'isolated_source_ids': list(pose_ids), 'isolated_pose_frames': pose_frames}
    grouped = defaultdict(list)
    for name, wanted in sorted(available.items()):
        sizes = cached_sizes(name, meta[name], {identity: row['annotated_frame_count'] for identity, row in wanted.items()})
        for identity, size in sizes.items():
            row = wanted[identity]
            grouped[row['template']].append({**row, 'preflight_rgb_bytes': size})
    selected, signer_counts, used_bytes = [], Counter(), 0
    for _ in range(max(map(len, grouped.values()), default=0)):
        for template in sorted(grouped, key=order):
            choices = [row for row in grouped[template]
                       if row not in selected and used_bytes + row['preflight_rgb_bytes'] <= CAP - 25_000_000]
            if not choices:
                continue
            chosen_signers = Counter(old['signer'] for old in selected if old['template'] == template)
            row = min(choices, key=lambda row: (chosen_signers[row['signer']], signer_counts[row['signer']],
                                                row['preflight_rgb_bytes'], order(row['id'])))
            selected.append(row)
            used_bytes += row['preflight_rgb_bytes']
            signer_counts[row['signer']] += 1
            if len(selected) == TARGET:
                break
        if len(selected) == TARGET:
            break
    if not selected:
        raise ValueError('No exact-order pose-linked new training templates')
    counts = {'unused_source_candidates': len(rows), 'same_take_label_order': len(candidates),
              'eligible_pose_and_rgb_sources': sum(map(len, grouped.values())), 'eligible_templates': len(grouped),
              'selected_templates': len({row['template'] for row in selected}), 'selected_signers': dict(signer_counts),
              'selected_glosses': len({label for row in selected for label in row['glosses']}), 'range_forecast': used_bytes}
    return {'schema': 1, 'source': SOURCE, 'license': 'CC-BY-4.0', 'planned_samples': len(selected), 'counts': counts,
            'scope': 'Training-only new CERTH utterance templates, no historical exact template, Greek text or gloss sequence, and no one-edit match to held references.',
            'source_hashes': {p.relative_to(ROOT).as_posix(): digest(p) for p in
                              (SPLIT, SUPPLEMENTARY, METADATA, LIBRARY, SYNTHETIC, *POSE_FILES, *sources)},
            'samples': selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'acquire'))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'manifest.planned.json'
    if plan_path.exists():
        plan = read_json(plan_path)
        if any(digest(ROOT / path) != value for path, value in plan['source_hashes'].items()):
            raise ValueError('A frozen source changed')
    else:
        plan = build_plan()
        write_manifest(plan_path, plan)
    print(plan['planned_samples'], plan['counts'], flush=True)
    if args.phase == 'acquire':
        manifest = OUTPUT / 'manifest.json'
        previous = read_json(manifest) if manifest.exists() else None
        if previous is None or not previous['complete']:
            acquire(plan, OUTPUT, CAP, False, previous, max_workers=2)
        actual = read_json(manifest)
        if not actual['complete'] or any(row['frame_count'] != row['annotated_frame_count'] for row in actual['samples']):
            raise ValueError('Acquired clips do not match the planned author frame counts')


if __name__ == '__main__':
    main()
