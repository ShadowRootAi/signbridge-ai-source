"""Verify and sample already-local Elementary23 lexical warm-up videos."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5, sha256
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg

from prepare_elementary_catalog_plan import RECORD_URL, ROOT, SOURCE_DIR, workbook_rows
from prepare_elementary_translation_pilot import words


OUTPUT = ROOT / 'data/native-greek-paired-supplement'
REFERENCE = ROOT / 'data/sign-reference-videos/manifest.json'
CATALOG = ROOT / 'web/public/library/catalog.json'
EXCLUDE = tuple(ROOT / 'data' / name / 'manifest.json' for name in (
    'elementary-translation-pilot', 'appointment-external-lexical', 'appointment-month-lexical',
))
FIT_SIGNERS = frozenset('ABCE')
FPS = 15
FRAME_LIMIT = 1500
GROUPS = {
    'time': 'χτες αύριο προχτές μεσημέρι απόγευμα πρωί αύγουστος σάββατο',
    'clinical_general': 'γιατρός θερμόμετρο μύτη νιώθω μάσκα νερό',
    'pronouns_and_negation': 'εμείς εσείς εσύ ποτέ ποιός',
    'communication': 'ζητώ επικοινωνώ μιλώ καληνύχτα όνομα οδηγίες είπα',
    'movement': 'πηγαίνω όρθιος ανοίγω ψάχνω',
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def annotated_rows(workbook):
    annotations = defaultdict(list)
    for number, identity, signer, text in workbook_rows(workbook):
        annotations[identity].append((number, signer, text))
    return annotations


def local_sources(reference, catalog):
    for clip in reference['clips']:
        if clip['source_url'] == RECORD_URL and clip['license'] == 'CC-BY-4.0':
            yield (clip['source_id'], clip['source_path'], clip['original_transcription'],
                   ROOT / clip['path'], clip['sha256'], clip['original_sha256'],
                   clip['duration_seconds'], REFERENCE)
    for clip in catalog['entries']:
        if clip['sourceUrl'] == RECORD_URL and clip['license']['id'] == 'CC-BY-4.0':
            source = clip['sourceUtterance']
            yield (Path(source).stem, source, clip['phrase'], CATALOG.parent / Path(clip['videoUrl']).name,
                   clip['sha256'], None, clip['durationSeconds'], CATALOG)


def choose(annotations, reference, catalog, protected):
    blocked_ids = {Path(row['source_path']).stem for row in protected}
    blocked_text = {tuple(words(row['text'])) for row in protected}
    blocked_hashes = {row['source_sha256'] for row in protected}
    groups = {target: group for group, phrase in GROUPS.items() for target in phrase.split()}
    candidates = defaultdict(list)
    for identity, member, recorded_text, local, expected, original, duration, index in local_sources(reference, catalog):
        label = tuple(words(recorded_text))
        if len(label) != 1 or label[0] not in groups or label in blocked_text:
            continue
        source_annotations = annotations.get(identity, [])
        if len(source_annotations) != 1 or identity in blocked_ids or original in blocked_hashes:
            continue
        number, signer, text = source_annotations[0]
        if signer not in FIT_SIGNERS or tuple(words(text)) != label:
            continue
        if not local.is_file() or digest(local) != expected:
            raise ValueError(f'Missing or changed audited local source {local}')
        row = {'id': f'elementary-{identity}', 'source_id': identity, 'source_path': member,
               'signer': signer, 'split': 'train', 'text': label[0], 'original_text': text,
               'workbook_row': number, 'source_index': index.relative_to(ROOT).as_posix(),
               'local_video': local.relative_to(ROOT).as_posix(), 'local_video_sha256': expected,
               'archive_video_sha256': original, 'duration_seconds': duration,
               'label_type': 'whole_clip_greek_word', 'selection_group': groups[label[0]]}
        candidates[label[0]].append(row)
    selected = [min(candidates[target], key=lambda row: (row['duration_seconds'], row['id']))
                for target in groups if candidates[target]]
    if len(selected) < 25 or len({row['local_video_sha256'] for row in selected}) != len(selected):
        raise ValueError('Insufficient distinct new local word-video evidence')
    return selected


def sample_frames(row):
    directory = OUTPUT / 'rgb' / row['id']
    directory.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror', '-y',
               '-threads', '1', '-i', str(ROOT / row['local_video']), '-map', '0:v:0', '-an', '-vf', f'fps={FPS}',
               '-threads', '1', '-q:v', '3', str(directory / 'frame-%05d.jpg')]
    subprocess.run(command, check=True, capture_output=True, timeout=45)
    frames = sorted(directory.glob('frame-*.jpg'))
    if len(frames) < 6 or abs(len(frames) - FPS * row['duration_seconds']) > 2:
        raise ValueError(f"Unexpected 15 Hz frame count {len(frames)} for {row['id']}")
    return {**row, 'fps': FPS, 'frame_paths': [path.relative_to(ROOT).as_posix() for path in frames],
            'frame_sha256': [digest(path) for path in frames]}


def main():
    workbook = SOURCE_DIR / 'All_Elementary.xlsx'
    metadata = read_json(SOURCE_DIR / 'elementary23-zenodo-metadata.json')
    official = {row['key']: row['checksum'] for row in metadata['files']}
    workbook_bytes = workbook.read_bytes()
    if (metadata['id'] != 7847052 or metadata['metadata']['license']['id'] != 'cc-by-4.0'
            or official[workbook.name] != f'md5:{md5(workbook_bytes).hexdigest()}'):
        raise ValueError('Unexpected official Elementary23 workbook or license')
    reference, catalog = read_json(REFERENCE), read_json(CATALOG)
    workbook_hash = sha256(workbook_bytes).hexdigest()
    if reference['workbook_sha256'] != workbook_hash:
        raise ValueError('Reference clips were audited against a different workbook')
    protected = [row for manifest in EXCLUDE for row in read_json(manifest)['samples']]
    selected = choose(annotated_rows(workbook), reference, catalog, protected)
    with ThreadPoolExecutor(max_workers=4) as pool:
        samples = list(pool.map(sample_frames, selected))
    frames = sum(len(row['frame_paths']) for row in samples)
    if frames > FRAME_LIMIT:
        raise ValueError(f'Supplement exceeds the {FRAME_LIMIT} frame budget ({frames})')
    manifest = {'source': RECORD_URL, 'license': 'CC-BY-4.0', 'workbook_sha256': workbook_hash,
                'creators': metadata['metadata']['creators'], 'complete': True, 'sample_fps': FPS,
                'scope': 'Previously verified local, whole-clip one-word Greek translations for lexical warm-up. '
                         'No Greek phrases, gloss labels, or within-clip alignment.',
                'input_sha256': {path.relative_to(ROOT).as_posix(): digest(path)
                                 for path in (REFERENCE, CATALOG, *EXCLUDE)},
                'samples': samples, 'sampled_frames': frames}
    pending = OUTPUT / 'manifest.pending.json'
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(OUTPUT / 'manifest.json')
    print(json.dumps({'samples': len(samples), 'frames': frames,
                      'signers': Counter(row['signer'] for row in samples)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
