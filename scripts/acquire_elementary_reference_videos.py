"""Acquire a bounded set of additional exact-text Elementary23 reference clips."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5, sha256
import json
from pathlib import Path
from subprocess import TimeoutExpired
from zipfile import ZipFile

from build_elementary_library import VIDEO_DIR, stage_clip
from dataset_fetch import RemoteZipFile
from prepare_elementary_catalog_plan import ARCHIVE, RECORD_URL, ROOT, SOURCE_DIR, VIDEO_ID, single_word, word_key, workbook_rows


OUTPUT = ROOT / 'data/sign-reference-videos'
PREFERRED = tuple(map(word_key, ('δευτέρα', 'τρίτη', 'τετάρτη', 'πέμπτη', 'παρασκευή', 'σάββατο', 'κυριακή',
                                'μεσημέρι', 'ημερομηνία', 'ώρα', 'ραντεβού', 'εξέταση', 'νοσοκομείο', 'γιατρός',
                                'παρακαλώ', 'βοήθεια', 'πρόβλημα', 'συγγνώμη', 'όνομα', 'ποτέ')))


def write_json(path, value):
    staging = path.with_suffix('.pending.json')
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    staging.replace(path)


def candidates(workbook):
    library_paths = [ROOT / path for path in ('web/public/library/catalog.json', 'publication-library-100/library/catalog.json')]
    existing = {word_key(phrase) for path in library_paths if path.is_file()
                for entry in json.loads(path.read_text(encoding='utf-8'))['entries']
                for phrase in (entry['phrase'], *entry.get('aliases', []))}
    rows = list(workbook_rows(workbook))
    counts = Counter(identity for _, identity, _, _ in rows)
    terms = defaultdict(list)
    for row, identity, signer, text in rows:
        if counts[identity] != 1 or not identity.startswith('glossa_') or not VIDEO_ID.fullmatch(identity):
            continue
        word = single_word(text)
        if word and word_key(word) not in existing:
            terms[word_key(word)].append({'phrase': word, 'source_id': identity, 'signer': signer,
                                           'workbook_row': row, 'transcription': text})
    return terms


def run(count, workers):
    meta = json.loads((SOURCE_DIR / 'elementary23-zenodo-metadata.json').read_text(encoding='utf-8'))
    details = meta['metadata']
    if (meta['id'] != 7847052 or details['access_right'] != 'open' or details['license']['id'] != 'cc-by-4.0'):
        raise ValueError('Expected the official open Elementary23 dataset under CC BY 4.0')
    files = {row['key']: row for row in meta['files']}
    workbook = SOURCE_DIR / 'All_Elementary.xlsx'
    workbook_bytes = workbook.read_bytes()
    if files[workbook.name]['checksum'] != 'md5:' + md5(workbook_bytes).hexdigest():
        raise ValueError('The annotation workbook failed its official checksum')
    spec = files[ARCHIVE]
    source = {'archiveUrl': spec['links']['self'], 'archiveBytes': spec['size'], 'recordUrl': RECORD_URL,
              'title': details['title'], 'creators': [row['name'] for row in details['creators']],
              'license': {'id': 'CC-BY-4.0', 'url': 'https://creativecommons.org/licenses/by/4.0/'}}
    terms = candidates(workbook)
    with RemoteZipFile(source['archiveUrl'], source['archiveBytes']) as remote, ZipFile(remote) as archive:
        infos = sorted(archive.infolist(), key=lambda item: item.header_offset)
        members = {item.filename: item for item in infos}
        if len(members) != len(infos):
            raise ValueError('Duplicate path in the official video archive')
        ends = dict(zip((item.header_offset for item in infos),
                        [item.header_offset for item in infos[1:]] + [archive.start_dir], strict=True))
        initial = remote.bytes_received
    total = initial
    selected, signers = [], Counter()
    priority = {key: index for index, key in enumerate(PREFERRED)}
    for key in sorted(terms, key=lambda value: (priority.get(value, len(priority)),
                                              -len({row['signer'] for row in terms[value]}), value)):
        options = []
        for row in terms[key]:
            info = members.get(f'{VIDEO_DIR}/{row["source_id"]}.flv')
            if info and 0 < info.file_size <= 16 * 1024**2 and not info.flag_bits & 1:
                length = ends[info.header_offset] - info.header_offset
                if length > 0 and total + length <= 90_000_000:
                    options.append((row, info, length))
        if options:
            row, info, length = min(options, key=lambda item: (signers[item[0]['signer']], item[2], item[0]['source_id']))
            signers[row['signer']] += 1
            selected.append((row, info))
            total += length
        if len(selected) == count:
            break
    if not selected:
        raise ValueError('The official workbook and archive have no usable new clips')
    print(json.dumps({'planned_clips': len(selected), 'bounded_transfer_bytes': total}), flush=True)

    media = OUTPUT / 'media'
    media.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / 'manifest.json'
    workbook_hash = sha256(workbook_bytes).hexdigest()
    previous = {}
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved.get('workbook_sha256') != workbook_hash:
            raise ValueError('Existing reference videos were verified against a different workbook')
        previous = {item['source_id']: item for item in saved['clips']}

    def acquire(item):
        row, info = item
        cached = previous.get(row['source_id'])
        if cached and cached['workbook_row'] == row['workbook_row'] and cached['greek_text'] == row['phrase']:
            local = (ROOT / cached['path']).resolve()
            if local.parent == media.resolve() and local.is_file() and sha256(local.read_bytes()).hexdigest() == cached['sha256']:
                return cached, None, 0
        try:
            entry, review, transferred = stage_clip(source, row, info, ends[info.header_offset], media, verify_timestamps=True)
            path = media / f'{entry["id"]}.mp4'
            return {'id': entry['id'], 'greek_text': row['phrase'], 'source_url': RECORD_URL,
                    'source_path': review['source_path'], 'source_id': row['source_id'], 'signer': row['signer'],
                    'workbook_row': row['workbook_row'], 'original_transcription': row['transcription'],
                    'license': 'CC-BY-4.0', 'reference_type': 'greek_translation', 'path': path.relative_to(ROOT).as_posix(),
                    'sha256': review['mp4_sha256'], 'original_sha256': review['original_sha256'],
                    'archive_crc32': review['archive_member_crc32'], 'frames': review['frames'],
                    'duration_seconds': review['duration_seconds'], 'modifications': entry['modifications'],
                    'attribution': entry['attribution']}, None, transferred
        except (OSError, ValueError, TimeoutExpired) as error:
            return None, {'source_id': row['source_id'], 'error': str(error)}, 0

    clips, failures, transferred = [], [], initial
    seen_content = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (clip, error, size) in enumerate(pool.map(acquire, selected), 1):
            transferred += size
            if clip and clip['original_sha256'] not in seen_content:
                clips.append(clip)
                seen_content.add(clip['original_sha256'])
            elif clip:
                failures.append({'source_id': clip['source_id'], 'error': 'duplicate source video bytes'})
            else:
                failures.append(error)
            if index % 20 == 0 or index == len(selected):
                print(json.dumps({'processed': index, 'verified': len(clips)}), flush=True)
    manifest = {'schema': 1, 'source': RECORD_URL, 'annotation': 'Official Greek text references for whole clips.',
                'workbook_sha256': workbook_hash, 'planned': len(selected),
                'transfer_bytes_upper_bound': total, 'successful_transfer_bytes': transferred,
                'clips': clips, 'failures': failures}
    write_json(manifest_path, manifest)
    if not clips:
        raise ValueError('No verified playable reference clip could be acquired')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=160)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.count <= 200 or not 1 <= args.workers <= 8:
        parser.error('Use 1–200 new reference clips and 1–8 workers')
    run(args.count, args.workers)


if __name__ == '__main__':
    main()
