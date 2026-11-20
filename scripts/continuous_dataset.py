"""Acquire a bounded, signer-disjoint GSL sentence cohort from Zenodo RGB frames."""

import argparse
import collections
import hashlib
import io
import json
import re
import zipfile
import zlib

from PIL import Image

from dataset_fetch import DATA, ROOT, get_archive


ARCHIVE = 'police1.zip'
COHORT = DATA / 'continuous-v1'
HOLDOUT_COHORT = DATA / 'continuous-v1-holdout'
EXTRA_COHORT = DATA / 'continuous-v1-extra'
COMPOSITION_COHORT = DATA / 'continuous-composition-pilot'
SPLIT_SIGNERS = {'train': (1, 2, 4, 5), 'val': (3,), 'test': (6, 7)}
DEFAULT_TEMPLATES = (2, 4, 9, 14)
DEFAULT_HOLDOUT_TEMPLATES = (3, 5)
DEFAULT_EXTRA_TEMPLATES = (0, 1)
COMPOSITION_TEMPLATES = (6, 15)
COMPOSITION_DISTRACTOR_TEMPLATES = (4,)
COMPOSITION_HOLDOUT_TEMPLATES = (14,)
MAX_CLIP_BYTES = 16 * 1024 * 1024
FRAME_NAME = re.compile(r'frame_(\d+)\.jpg')


def read_annotation_maps(scenario):
    annotations = DATA / 'original_annotations'
    prefix = 'GSL_split/GSL_continuous/gsl_split_SI_'
    glosses = {}
    with zipfile.ZipFile(annotations / 'GSL_split.zip') as splits:
        for split in ('train', 'dev', 'test'):
            for line in splits.read(prefix + split + '.csv').decode('utf-8-sig').splitlines():
                path, raw_glosses = line.split('|', 1)
                if path.startswith(f'{scenario}/'):
                    if path in glosses:
                        raise ValueError(f'Duplicate official gloss annotation: {path}')
                    glosses[path] = raw_glosses.split()
    texts = {}
    with zipfile.ZipFile(annotations / 'supplementary.zip') as extra:
        for line in extra.read('merged_continuous.csv').decode('utf-8-sig').splitlines():
            path, _, content = line.partition('|')
            if path.startswith(f'{scenario}_'):
                text, _, _ = content.partition('|')
                if not text.strip() or path in texts:
                    raise ValueError(f'Missing or duplicate official Greek translation: {path}')
                texts[path] = text.strip()
    return glosses, texts


def build_plan(templates, repetitions, split_signers, holdout, scenario='police1'):
    glosses, texts = read_annotation_maps(scenario)
    for template_number in templates:
        template = f'sentences{template_number:04}'
        for split, signers in split_signers.items():
            for signer in signers:
                for repetition in repetitions:
                    session = f'{scenario}_signer{signer}_rep{repetition}_sentences'
                    path = f'{session}/{template}'
                    source_path = f'{scenario}/{path}'
                    tokens = glosses[source_path]
                    if len(tokens) < 2:
                        raise ValueError(f'Expected a multi-gloss sentence: {source_path}')
                    sample = {
                        'id': f'{session}__{template}',
                        'source_path': source_path,
                        'signer': signer,
                        'split': split,
                        'glosses': tokens,
                        'text': texts[path],
                        'fps': 30,
                        'template': f'{scenario}/{template}',
                    }
                    if holdout:
                        sample['composition'] = 'unseen_template'
                    yield sample


def write_manifest(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def template_names(output, default_templates):
    names = {f'police1/sentences{number:04}' for number in default_templates}
    path = output / 'manifest.planned.json'
    if path.exists():
        plan = json.loads(path.read_text(encoding='utf-8'))
        names.update(item['template'] for item in plan['samples'])
    return names


def verify_rgb(data, info):
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f'Invalid size or CRC: {info.filename}')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or image.mode != 'RGB':
            raise ValueError(f'Expected a GSL RGB JPEG frame: {info.filename}')
        image.load()


def acquire(samples, output, split_signers, max_transfer_bytes, archive_name=ARCHIVE):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    archive_path = 'https://zenodo.org/records/4756317'
    manifest = {
        'schema': 1,
        'source': archive_path,
        'license': 'CC-BY-4.0',
        'rgb_archive': archive_name,
        'annotations': {'glosses': 'GSL_split.zip/GSL_continuous/gsl_split_SI_*.csv',
                        'text': 'supplementary.zip/merged_continuous.csv'},
        'split_signers': split_signers,
        'planned_samples': len(samples),
        'samples': [],
        'complete': False,
    }
    if path.exists():
        manifest['samples'] = json.loads(path.read_text(encoding='utf-8'))['samples']
    completed = {sample['id']: sample for sample in manifest['samples']}
    expected = {sample['id'] for sample in samples}
    if completed.keys() - expected:
        raise ValueError('Requested cohort omits earlier clips. Include the previous templates and repetitions.')
    write_manifest(output / 'manifest.planned.json', {**manifest, 'samples': samples})

    remote, archive = get_archive(archive_name)
    with archive:
        metadata_cache = list(remote.cache)
        by_path = collections.defaultdict(list)
        for info in archive.infolist():
            if info.filename.lower().endswith('.jpg'):
                by_path[info.filename.rsplit('/', 1)[0]].append(info)

        for sample in samples:
            members = sorted(by_path[sample['source_path'].split('/', 1)[1]], key=lambda i: i.filename)
            if not members:
                raise ValueError(f"Missing source frames: {sample['source_path']}")
            clip_dir = output / 'rgb' / sample['id']
            clip_dir.mkdir(parents=True, exist_ok=True)
            frames = [(info, clip_dir / info.filename.rsplit('/', 1)[1]) for info in members]
            numbers = [int(FRAME_NAME.fullmatch(target.name)[1]) for _, target in frames]
            if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
                raise ValueError(f"Nonconsecutive source frames: {sample['source_path']}")
            missing = [(info, target) for info, target in frames if not target.exists()]
            if missing:
                start = min(info.header_offset for info, _ in missing)
                end = min(remote.size, max(info.header_offset + info.compress_size + 1024 for info, _ in missing))
                if end - start > MAX_CLIP_BYTES or remote.bytes_received + end - start > max_transfer_bytes:
                    raise ValueError('Clip range would exceed the requested transfer budget')
                remote.seek(start)
                remote.read(end - start)
            clip_hash = hashlib.sha256()
            for info, target in frames:
                if target.exists():
                    data = target.read_bytes()
                    verify_rgb(data, info)
                else:
                    data = archive.read(info)
                    verify_rgb(data, info)
                    temporary = target.with_suffix('.tmp')
                    temporary.write_bytes(data)
                    temporary.replace(target)
                clip_hash.update(hashlib.sha256(data).digest())
            digest = clip_hash.hexdigest()
            if any(item['sha256'] == digest for key, item in completed.items() if key != sample['id']):
                raise ValueError(f"Duplicate RGB sequence: {sample['source_path']}")
            completed[sample['id']] = {
                **sample,
                'frame_paths': [target.relative_to(ROOT).as_posix() for _, target in frames],
                'frame_count': len(frames),
                'sha256': digest,
            }
            manifest['samples'] = [completed[item['id']] for item in samples if item['id'] in completed]
            manifest['complete'] = len(completed) == len(samples)
            manifest['last_run_transfer_bytes'] = remote.bytes_received
            write_manifest(path, manifest)
            print(f"{len(completed)}/{len(samples)} {sample['id']} {len(frames)} frames "
                  f'({remote.bytes_received:,} B transferred)', flush=True)
            remote.cache = list(metadata_cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates', type=int, nargs='+',
                        help='police1 sentence numbers, e.g. 2 4 9 14')
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument('--holdout', action='store_true', help='Acquire unseen templates for test signers only')
    cohort.add_argument('--extra', action='store_true', help='Acquire additional templates in a separate cohort')
    cohort.add_argument('--composition', action='store_true', help='Acquire the police2 unseen-composition pilot')
    parser.add_argument('--repetitions', type=int, nargs='+', choices=range(1, 6), default=(1, 2))
    parser.add_argument('--max-transfer-mib', type=int, help='Defaults to 120 MiB, or 145 MiB for composition')
    parser.add_argument('--plan', action='store_true', help='Print metadata without downloading or writing files')
    args = parser.parse_args()
    max_transfer_mib = args.max_transfer_mib
    if max_transfer_mib is None:
        max_transfer_mib = 145 if args.composition else 120
    max_transfer_bytes = max_transfer_mib * 1024 * 1024
    if len(set(args.repetitions)) != len(args.repetitions):
        parser.error('Repetitions must be unique')
    if max_transfer_mib < 7:
        parser.error('The transfer budget must allow at least 7 MiB for the ZIP directory')
    if args.composition:
        if args.templates:
            parser.error('The composition pilot has a fixed template selection')
        known = list(build_plan(COMPOSITION_TEMPLATES, args.repetitions, SPLIT_SIGNERS, False, 'police2'))
        distractor = list(build_plan(COMPOSITION_DISTRACTOR_TEMPLATES, args.repetitions,
                                     SPLIT_SIGNERS, False, 'police2'))
        test_only = {'test': SPLIT_SIGNERS['test']}
        unseen = list(build_plan(COMPOSITION_HOLDOUT_TEMPLATES, args.repetitions, test_only, True, 'police2'))
        train_sequences = {tuple(s['glosses']) for s in known if s['split'] == 'train'}
        vocabulary = set().union(*train_sequences)
        for sample in unseen:
            glosses = tuple(sample['glosses'])
            if glosses in train_sequences or set(glosses) - vocabulary:
                parser.error('An unseen composition must be new and use only trained glosses')
        samples = known + unseen + distractor
        if args.plan:
            print(json.dumps(samples, ensure_ascii=True, indent=2))
        else:
            acquire(samples, COMPOSITION_COHORT, SPLIT_SIGNERS, max_transfer_bytes, 'police2.zip')
        return
    output = COHORT
    split_signers = SPLIT_SIGNERS
    default_templates = DEFAULT_TEMPLATES
    if args.holdout:
        output = HOLDOUT_COHORT
        split_signers = {'test': SPLIT_SIGNERS['test']}
        default_templates = DEFAULT_HOLDOUT_TEMPLATES
    elif args.extra:
        output = EXTRA_COHORT
        default_templates = DEFAULT_EXTRA_TEMPLATES
    templates = args.templates or default_templates
    if len(set(templates)) != len(templates):
        parser.error('Templates must be unique')
    other_templates = set()
    for path, defaults in ((COHORT, DEFAULT_TEMPLATES), (EXTRA_COHORT, DEFAULT_EXTRA_TEMPLATES),
                           (HOLDOUT_COHORT, DEFAULT_HOLDOUT_TEMPLATES)):
        if path != output:
            other_templates.update(template_names(path, defaults))
    if any(f'police1/sentences{number:04}' in other_templates for number in templates):
        parser.error('Templates must be distinct across cohorts')
    samples = list(build_plan(templates, args.repetitions, split_signers, args.holdout))
    if args.plan:
        print(json.dumps(samples, ensure_ascii=True, indent=2))
    else:
        acquire(samples, output, split_signers, max_transfer_bytes)


if __name__ == '__main__':
    main()
