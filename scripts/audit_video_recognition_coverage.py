"""Audit candidate training coverage without running inference or downloading media."""

from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED

from acquire_elementary_library_expansion import ARCHIVES, MAX_MEMBER, METADATA, WORKBOOK, official, transcript
from package_video_library import COHORTS, ROOT, SEED, lookup_key
from prepare_elementary_catalog_plan import RECORD_URL, workbook_rows


DIST = ROOT / 'publication-local-assistant/dist'
CATALOG = DIST / 'library/catalog.json'
POLICY = DIST / '__signbridge/openhands/policy.json'
ISOLATED_RECORDS = ROOT / 'data/gsl_isolated_records.json'
OUTPUT = ROOT / 'reports/video-library/recognition-coverage.json'


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def local_elementary_ids(entries, manifests):
    local = {Path(entry['sourceUtterance']).stem for entry in entries
             if entry['sourceUrl'] == RECORD_URL and (DIST / entry['videoUrl'].lstrip('/')).is_file()}
    for path in manifests:
        for clip in read(path)['clips']:
            if (ROOT / clip['path']).is_file():
                local.add(clip.get('workbook_video_id') or clip['source_id'])
    return local


def indexed_members(files, indexes):
    indexed = {}
    for archive, path in indexes.items():
        data = read(path)
        spec = files[archive]
        if data['spec'] != spec:
            raise ValueError(f'Archive metadata changed: {archive}')
        indexed[archive] = {identity for identity, member in data['members'].items()
                            if (not member['flags'] & 1 and member['method'] in (ZIP_DEFLATED, ZIP_STORED)
                                and 0 < member['compressed'] <= member['size'] <= MAX_MEMBER
                                and member['offset'] + member['compressed'] + 1024 <= spec['size'])}
    return indexed


def count_signers(rows):
    return len({row['signer'] for row in rows})


def cohort_summary(entries, candidates, active_text):
    rows = [candidates[lookup_key(entry['phrase'])] for entry in entries]
    available = [[row for row in group if row['indexed']] for group in rows]
    cached = [[row for row in group if row['local_reference']] for group in rows]
    return {
        'entries': len(entries),
        'explicit_gloss_annotation': sum(bool(entry.get('annotationLabel')) for entry in entries),
        'exact_active_display_text_matches': sum(lookup_key(entry['phrase']) in active_text for entry in entries),
        'workbook_signer_distribution': dict(sorted(Counter(map(count_signers, rows)).items())),
        'indexed_signer_distribution': dict(sorted(Counter(map(count_signers, available)).items())),
        'local_reference_signer_distribution': dict(sorted(Counter(map(count_signers, cached)).items())),
        'workbook_at_least_three_signers': sum(count_signers(group) >= 3 for group in rows),
        'indexed_at_least_three_signers': sum(count_signers(group) >= 3 for group in available),
        'local_reference_at_least_three_signers': sum(count_signers(group) >= 3 for group in cached),
    }


def main():
    files, _ = official()
    entries = read(CATALOG)['entries']
    policy = read(POLICY)
    texts = {lookup_key(entry['phrase']) for entry in entries}
    if len(texts) != len(entries):
        raise ValueError('Catalog text keys are not unique')
    active = set(policy['supportedLabels'])
    active_text = {lookup_key(policy['displayLabels'][label]) for label in active}
    manifests = [SEED, *sorted(COHORTS.glob('*/manifest.json'))]
    local = local_elementary_ids(entries, manifests)
    indexes = {archive: COHORTS / 'indexes' / (Path(archive).stem + '.json') for archive in ARCHIVES.values()}
    indexed = indexed_members(files, indexes)

    candidates = defaultdict(dict)
    for number, identity, signer, text in workbook_rows(WORKBOOK):
        meaning, kind = transcript(text)
        key = lookup_key(meaning) if kind else None
        if key not in texts:
            continue
        archive = ARCHIVES.get(identity.split('_', 1)[0])
        row = {'source_id': identity, 'signer': signer, 'workbook_row': number, 'archive': archive,
               'indexed': identity in indexed.get(archive, ()), 'local_reference': identity in local}
        previous = candidates[key].setdefault(identity, row)
        if previous['signer'] != signer:
            raise ValueError(f'Conflicting signer for {identity}')
    by_text = defaultdict(list, {key: sorted(rows.values(), key=lambda row: (row['signer'], row['source_id']))
                                for key, rows in candidates.items()})
    cohorts = {kind: cohort_summary([entry for entry in entries if entry['contentKind'] == kind],
                                    by_text, active_text) for kind in ('word', 'phrase')}
    shortlisted = []
    for entry in sorted(entries, key=lambda item: (item['contentKind'], lookup_key(item['phrase']))):
        key = lookup_key(entry['phrase'])
        rows = by_text[key]
        if count_signers(rows) < 3:
            continue
        shortlist = [row for row in rows if row['indexed'] or row['local_reference']]
        shortlisted.append({'catalog_id': entry['id'], 'text': entry['phrase'], 'kind': entry['contentKind'],
                            'active_label_text_overlap': key in active_text, 'workbook_signers': count_signers(rows),
                            'indexed_signers': count_signers([row for row in shortlist if row['indexed']]),
                            'local_reference_signers': count_signers([row for row in shortlist if row['local_reference']]),
                            'shortlist': shortlist})

    isolated = read(ISOLATED_RECORDS)
    gsl_signers = defaultdict(set)
    for record in isolated:
        gsl_signers[record['gloss']].add(record['signer'])
    sources = [CATALOG, POLICY, WORKBOOK, METADATA, ISOLATED_RECORDS, *indexes.values(), *manifests]
    report = {
        'scope': 'Catalog and source metadata inventory. No accuracy, independent webcam or visual lexical validation measured.',
        'limits': [
            'The catalog pairs one video with each distinct normalized Greek text. A Greek transcript is not a GSL gloss.',
            'Identical normalized Greek texts across clips can correspond to different GSL variants or complete utterances.',
            'Signer IDs are the identities documented within each corpus; there is no cross-corpus identity mapping.',
            'Indexed means a qualifying FLV archive member exists in the locally cached index. The member was not newly fetched or decoded.',
            'Local references are present in the package or catalog acquisition manifests. Other research caches are excluded from this count.',
            'An exact text overlap with an active recognition label is lexical only; it is not a video prediction.',
        ],
        'inputs_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        'catalog': {'entries': len(entries), 'normalized_unique_texts': len(texts),
                    'by_source': dict(Counter(entry['sourceUrl'] for entry in entries)), 'by_kind': cohorts},
        'active_isolated_recognition': {'glosses': len(active), 'normalized_display_texts': len(active_text)},
        'certh_isolated_annotations': {'clips': len(isolated), 'glosses': len(gsl_signers),
                                      'signer_distribution': dict(sorted(Counter(map(len, gsl_signers.values())).items())),
                                      'glosses_with_three_or_more_signers': sum(len(group) >= 3 for group in gsl_signers.values())},
        'elementary_multi_signer_candidates': shortlisted,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': OUTPUT.relative_to(ROOT).as_posix(), 'catalog': report['catalog'],
                      'active_isolated_recognition': report['active_isolated_recognition']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
