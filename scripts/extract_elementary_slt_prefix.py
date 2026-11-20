"""Extract numeric Elementary23-SLT rows from an official gzip byte prefix.

Pickle opcodes are read as data. This script never invokes pickle loaders,
resolves serialized globals, or executes a serialized callable.
"""

from collections import Counter, defaultdict
from hashlib import md5, sha256
from pathlib import Path
import gzip
import io
import json
import pickletools
import re
import struct
import unicodedata

import numpy as np

from prepare_elementary_catalog_plan import ROOT, SOURCE_DIR, workbook_rows


DATA = ROOT / 'data/elementary-slt-prefix'
FIGSHARE = ROOT / 'research/external-gsl-metadata/elementary23-slt-figshare.json'
WIDTH = 250
MARKER = object()
GLOBAL_NAMES = {('torch.storage', '_load_from_bytes'),
                ('torch._utils', '_rebuild_tensor_v2'), ('collections', 'OrderedDict')}
VALUE_OPS = {'BINUNICODE', 'SHORT_BINUNICODE', 'BINBYTES', 'SHORT_BINBYTES',
             'BININT', 'BININT1', 'BININT2'}
SAMPLE_FIELDS = {'name', 'signer', 'gloss', 'text', 'sign'}


def hash_file(path):
    result = sha256()
    with path.open('rb') as source:
        while block := source.read(1024**2):
            result.update(block)
    return result.hexdigest()


class BoundedGzip:
    def __init__(self, source):
        self.source = source

    def read(self, count):
        if not 0 <= count <= 16 * 1024**2 or self.tell() + count > 768 * 1024**2:
            raise ValueError('Unexpected size in Elementary23 training prefix')
        return self.source.read(count)

    def tell(self):
        return self.source.tell()


def legacy_storage(payload):
    source = io.BytesIO(payload)
    headers = []
    for _ in range(5):
        headers.append([(op.name, value) for op, value, _ in pickletools.genops(source)
                        if op.name not in {'PROTO', 'BINPUT', 'LONG_BINPUT', 'STOP'}])
    magic, version, system, storage, indexes = headers
    if magic != [('LONG1', 119547037146038801333356)] or version != [('BININT2', 1001)]:
        raise ValueError('Unexpected tensor storage version')
    if [('BINUNICODE', 'little_endian'), ('NEWTRUE', None)] != system[4:6]:
        raise ValueError('Tensor storage is not little endian')
    expected = ['MARK', 'BINUNICODE', 'GLOBAL', 'BINUNICODE', 'BINUNICODE', None, 'NONE', 'TUPLE', 'BINPERSID']
    if (len(storage) != len(expected)
            or any(name is not None and storage[i][0] != name for i, name in enumerate(expected))
            or storage[1][1] != 'storage' or storage[2][1] != 'torch DoubleStorage' or storage[4][1] != 'cpu'
            or storage[5][0] not in {'BININT', 'BININT1', 'BININT2'}
            or indexes != [('EMPTY_LIST', None), ('BINUNICODE', storage[3][1]), ('APPEND', None)]):
        raise ValueError('Unsupported tensor storage format')
    count = storage[5][1]
    offset = source.tell()
    if (not 0 < count <= 2_000_000 or len(payload) != offset + 8 + count * 8
            or struct.unpack_from('<Q', payload, offset)[0] != count):
        raise ValueError('Tensor storage length does not match its numeric payload')
    return np.frombuffer(payload, dtype='<f8', count=count, offset=offset + 8)


def numeric_reduce(global_name, args):
    if global_name == ('torch.storage', '_load_from_bytes'):
        if len(args) == 1 and isinstance(args[0], bytes):
            return legacy_storage(args[0])
    elif global_name == ('collections', 'OrderedDict'):
        if args == ():
            return {}
    elif global_name == ('torch._utils', '_rebuild_tensor_v2') and len(args) == 6:
        values, offset, shape, strides, requires_grad, extras = args
        if (isinstance(values, np.ndarray) and offset == 0 and isinstance(shape, tuple)
                and len(shape) == 2 and all(type(i) is int and i > 0 for i in shape)
                and shape[1] == WIDTH and shape[0] * WIDTH == len(values)
                and strides in {(1, shape[0]), (WIDTH, 1)}
                and requires_grad is False and extras == {}):
            return values.reshape(shape, order='F' if strides[0] == 1 else 'C')
    raise ValueError(f'Unsupported data construction in source: {global_name}')


def take_mark(stack):
    index = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is MARKER)
    values = stack[index + 1:]
    del stack[index:]
    return values


def read_records(source):
    memo, stack = [], []
    operations = pickletools.genops(BoundedGzip(source))
    while True:
        try:
            op, value, offset = next(operations)
        except EOFError:
            # Expected at the exact HTTP gzip byte cap. Only complete rows
            # emitted by SETITEMS before this point are retained.
            return
        name = op.name
        if name == 'PROTO':
            if offset != 0 or value != 4:
                raise ValueError('Unexpected Elementary23 pickle protocol')
        elif name == 'FRAME':
            continue
        elif name in VALUE_OPS:
            stack.append(value)
        elif name in {'BINGET', 'LONG_BINGET'}:
            stack.append(memo[value])
        elif name == 'MEMOIZE':
            memo.append(stack[-1])
        elif name == 'MARK':
            stack.append(MARKER)
        elif name == 'EMPTY_LIST':
            if stack:
                raise ValueError('Unexpected nested list in Elementary23 data')
            stack.append([])
        elif name == 'EMPTY_DICT':
            stack.append({})
        elif name == 'EMPTY_TUPLE':
            stack.append(())
        elif name == 'TUPLE':
            stack.append(tuple(take_mark(stack)))
        elif name == 'TUPLE1':
            stack.append((stack.pop(),))
        elif name == 'TUPLE2':
            second, first = stack.pop(), stack.pop()
            stack.append((first, second))
        elif name == 'NEWFALSE':
            stack.append(False)
        elif name == 'STACK_GLOBAL':
            member, module = stack.pop(), stack.pop()
            global_name = module, member
            if global_name not in GLOBAL_NAMES:
                raise ValueError(f'Unexpected source global at offset {offset}')
            stack.append(global_name)
        elif name == 'REDUCE':
            args, global_name = stack.pop(), stack.pop()
            if not isinstance(args, tuple) or global_name not in GLOBAL_NAMES:
                raise ValueError(f'Invalid data constructor at offset {offset}')
            stack.append(numeric_reduce(global_name, args))
        elif name == 'SETITEMS':
            fields = take_mark(stack)
            record = dict(zip(fields[::2], fields[1::2], strict=True))
            if len(fields) != 10 or set(record) != SAMPLE_FIELDS or stack[-1] != {}:
                raise ValueError(f'Unexpected Elementary23 row at offset {offset}')
            if not isinstance(record['sign'], np.ndarray) or not all(
                    isinstance(record[key], str) for key in SAMPLE_FIELDS - {'sign'}):
                raise ValueError(f'Unexpected data type in Elementary23 row at offset {offset}')
            stack[-1].update(record)
            yield offset, record
        elif name == 'APPENDS':
            take_mark(stack)
        elif name == 'STOP':
            raise ValueError('Expected a byte prefix of the official full training file')
        else:
            raise ValueError(f'Unsupported pickle opcode {name} at offset {offset}')


def text_key(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


def prior_roles():
    sources = ('elementary-translation-pilot', 'native-greek-paired-supplement',
               'appointment-external-lexical', 'appointment-month-lexical')
    return {Path(row['source_path']).stem: f'{source}:{row["split"]}'
            for source in sources
            if (path := ROOT / 'data' / source / 'manifest.json').is_file()
            for row in json.loads(path.read_bytes())['samples']}


def main():
    prefix = DATA / 'train-prefix.gz'
    metadata = json.loads(FIGSHARE.read_bytes())
    evidence = json.loads((DATA / 'download-evidence.json').read_bytes())
    entry = next(item for item in metadata['files'] if item['name'] == 'Elementary23_SLT.train')
    if (metadata['id'] != 22262953 or metadata['doi'] != evidence['record_doi']
            or metadata['license']['name'] != 'CC BY 4.0' or entry != evidence['source_file']
            or not 0 < prefix.stat().st_size == evidence['received_bytes'] <= 128 * 1024**2
            or evidence['http_content_range'] != f'bytes 0-{prefix.stat().st_size-1}/{entry["size"]}'
            or hash_file(prefix) != evidence['sha256']):
        raise ValueError('Figshare training prefix does not match the audited source')
    workbook = SOURCE_DIR / 'All_Elementary.xlsx'
    original = json.loads((SOURCE_DIR / 'elementary23-zenodo-metadata.json').read_bytes())
    expected = next(item['checksum'] for item in original['files'] if item['key'] == workbook.name)
    if original['id'] != 7847052 or expected != f'md5:{md5(workbook.read_bytes()).hexdigest()}':
        raise ValueError('Elementary23 workbook failed source verification')
    official = defaultdict(list)
    for row, identity, signer, text in workbook_rows(workbook):
        official[identity].append((row, signer, text))
    prior = prior_roles()
    with gzip.open(prefix, 'rb') as file:
        all_rows = list(read_records(file))
    if not all_rows:
        raise ValueError('No complete paired rows found in the training prefix')
    counts = Counter(row['name'] for _, row in all_rows)
    source_signers = Counter(row['signer'] for _, row in all_rows)
    skipped, samples, frames, offsets = Counter(), [], [], [0]
    for position, item in all_rows:
        name, text, pose = item['name'], item['text'], item['sign']
        if (counts[name] != 1 or not re.fullmatch(r'[a-z0-9_]+', name) or len(official[name]) != 1):
            skipped['identity_unmatched_or_ambiguous'] += 1
            continue
        row, signer, original_text = official[name][0]
        if not re.fullmatch(r'[A-Z]', signer) or item['signer'] not in {'Unknown', signer}:
            skipped['signer_unmatched_or_ambiguous'] += 1
            continue
        if not text_key(text) or text_key(text) != text_key(original_text):
            skipped['text_differs_from_official_workbook'] += 1
            continue
        if len(pose) < 2 or not np.isfinite(pose).all() or np.abs(pose).max() > 100_000:
            skipped['invalid_numeric_pose'] += 1
            continue
        frame_count = len(pose)
        samples.append({'id': name, 'text': original_text, 'figshare_text': text, 'signer': signer,
                        'figshare_signer': item['signer'], 'split': 'train', 'workbook_row': row,
                        'pose_start': offsets[-1], 'pose_end': offsets[-1] + frame_count,
                        'source_pickle_row_end_offset': position + 1, 'prior_use': prior.get(name)})
        offsets.append(offsets[-1] + frame_count)
        frames.append(pose.astype(np.float32))
    if not samples:
        raise ValueError('No unambiguous signer and text matches in the source prefix')
    result = DATA / 'poses.npz'
    pending = DATA / 'poses.pending.npz'
    np.savez_compressed(pending, features=np.ascontiguousarray(np.concatenate(frames)),
                        offsets=np.array(offsets, dtype=np.int64),
                        ids=np.array([row['id'] for row in samples]),
                        texts=np.array([row['text'] for row in samples]),
                        signers=np.array([row['signer'] for row in samples]))
    pending.replace(result)
    manifest = {
        'source': metadata['figshare_url'], 'record_doi': metadata['doi'], 'license': metadata['license'],
        'authors': metadata['authors'], 'download': evidence, 'split': 'figshare_train_prefix_only',
        'scope': 'Complete records recoverable from a bounded compressed training prefix. '
                 'The published dev and test files are excluded. Source prefix was not randomly sampled.',
        'feature_schema': 'Elementary23-SLT author OpenPose frame vectors, 250 numeric columns in original order',
        'coordinate_scope': 'Paper identifies OpenPose 2D upper body, hands and face. Exact 250-column '
                            'landmark ordering and frame clock are unverified for these serialized tensors.',
        'paper': 'https://arxiv.org/abs/2310.04753',
        'source_dtype': 'float64 little endian', 'output_dtype': 'float32',
        'workbook_record': 'https://zenodo.org/records/7847052', 'workbook_sha256': hash_file(workbook),
        'archive_metadata_sha256': hash_file(FIGSHARE), 'pose_file': result.name, 'npz_sha256': hash_file(result),
        'source_rows_recovered': len(all_rows), 'source_signer_field': source_signers,
        'retained_rows': len(samples), 'skipped': skipped, 'retained_frames': offsets[-1],
        'resolved_signers': Counter(row['signer'] for row in samples),
        'prior_source_usage': Counter(row['prior_use'] or 'none_detected' for row in samples), 'samples': samples,
    }
    pending_manifest = DATA / 'manifest.pending.json'
    pending_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending_manifest.replace(DATA / 'manifest.json')
    print({key: manifest[key] for key in ('source_rows_recovered', 'source_signer_field', 'retained_rows',
                                         'retained_frames', 'skipped', 'resolved_signers', 'prior_source_usage')})


if __name__ == '__main__':
    main()
