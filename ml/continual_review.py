"""Validate and seal reviewed landmark bundles for local continual training."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import torch
from filelock import FileLock, Timeout

from preprocess import FEATURE_COUNT, FEATURE_SCHEMA
from train_continuous import Utterance, edit_counts


TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
HASH = re.compile(r'[0-9a-f]{64}')
BATCH = re.compile(r'[0-9a-f]{32}')
REVIEW_SOURCE = re.compile(r'reviewed/([0-9a-f]{64})/(\d+)-(\d+)')
MANIFEST_LIMIT = 256 * 1024
FEATURES_LIMIT = 48 * 1024 * 1024
APPROVAL_LIMIT = 4096
MAX_BATCH_SAMPLES = 24
MAX_APPROVED_SAMPLES = 240
MAX_APPROVED_BATCHES = 64
FPS = 15
MIN_TRACKING_QUALITY = .70
SAMPLE_KEYS = {'id', 'source_sha256', 'start_ms', 'end_ms', 'signer', 'session', 'consent_local_ml', 'glosses', 'text'}
FEATURE_KEYS = {'id', 'feature_schema', 'features', 'fps', 'quality'}
FEATURE_OPTIONAL = {'elapsed_ms', 'extractor_sha256', 'source_key', 'source_frames', 'stride'}
APPROVAL_KEYS = {'schema_version', 'reviewer', 'manifest_sha256', 'features_sha256'}
FILES = {'manifest.json': MANIFEST_LIMIT, 'features.jsonl': FEATURES_LIMIT, 'approval.json': APPROVAL_LIMIT}


def _matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _read(path, limit):
    if not path.is_file():
        raise ValueError('A reviewed bundle input is missing or is not a file')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError('A reviewed bundle input is empty or exceeds its size limit')
    return data


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError('Nonfinite JSON value')


def _parse(value):
    try:
        return json.loads(value.decode('utf-8-sig'), object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError('Invalid reviewed bundle JSON') from error


def _digest(data):
    return sha256(data).hexdigest()


def _frame_digest(frames):
    values = frames.detach().cpu().numpy().astype('<f4', copy=True)
    values[values == 0] = 0  # The model treats positive and negative zero identically.
    return _digest(values.tobytes())


def _text_key(text):
    decomposed = unicodedata.normalize('NFKD', text.casefold())
    no_accents = ''.join(char for char in decomposed if not unicodedata.combining(char))
    return ' '.join(''.join(char if char.isalnum() else ' ' for char in no_accents).split())


def _source_details(row):
    source = getattr(row, 'source_sha256', None)
    if source is not None and not _matches(HASH, source):
        raise ValueError('Invalid source digest in trusted baseline or protected rows')
    match = REVIEW_SOURCE.fullmatch(row.source)
    if match:
        if source is not None and source != match[1]:
            raise ValueError('Inconsistent source digests in trusted rows')
        return match[1], (int(match[2]), int(match[3]))
    return source, None


class _Exclusions:
    def __init__(self, base_rows, protected_rows, labels):
        base_rows, protected_rows = list(base_rows), list(protected_rows)
        self.ids = {row.id for row in base_rows + protected_rows}
        self.frames = {_frame_digest(row.frames) for row in base_rows + protected_rows}
        self.labels = {label for label in labels if isinstance(label, str) and label}
        self.signers = {str(row.signer).casefold() for row in protected_rows}
        self.glosses = {tuple(row.glosses) for row in protected_rows if row.glosses}
        self.texts = {_text_key(row.text) for row in protected_rows if row.text}
        self.whole_sources = {source for row in protected_rows if (source := _source_details(row)[0])}
        self.intervals = defaultdict(list)
        self.source_owners = {}
        for row in base_rows:
            source, span = _source_details(row)
            if source and span is None:
                self.whole_sources.add(source)
            elif source:
                self.intervals[source].append(span)
                self.source_owners[source] = (str(row.signer).casefold(), getattr(row, 'session', None))

    def register(self, row, source, span, session):
        if row.id in self.ids:
            raise ValueError('A reviewed sample ID duplicates an existing sample')
        signer = row.signer.casefold()
        if signer in self.signers:
            raise ValueError('A reviewed signer is reserved for evaluation')
        if _text_key(row.text) in self.texts or any(sum(edit_counts(glosses, row.glosses)) <= 1 for glosses in self.glosses):
            raise ValueError('A reviewed sample matches or is one gloss edit from a protected template')
        if source in self.whole_sources:
            raise ValueError('A reviewed source digest is already reserved or used in full')
        owner = self.source_owners.get(source)
        if owner is not None and (owner[0] != signer or (owner[1] is not None and owner[1] != session)):
            raise ValueError('One source digest cannot belong to different signers or sessions')
        if any(span[0] < end and start < span[1] for start, end in self.intervals[source]):
            raise ValueError('Reviewed ranges overlap within the same source')
        fingerprint = _frame_digest(row.frames)
        if fingerprint in self.frames:
            raise ValueError('A reviewed sample duplicates exact landmark frames')
        self.ids.add(row.id)
        self.frames.add(fingerprint)
        self.intervals[source].append(span)
        self.source_owners[source] = (signer, session)


def _parse_features(blob):
    records = {}
    for line in blob.splitlines():
        if not line.strip():
            continue
        if len(records) >= MAX_BATCH_SAMPLES:
            raise ValueError('Too many reviewed feature rows')
        row = _parse(line)
        if not isinstance(row, dict) or not FEATURE_KEYS <= row.keys() <= FEATURE_KEYS | FEATURE_OPTIONAL:
            raise ValueError('Unexpected reviewed feature fields')
        identity = row['id']
        if not _matches(TOKEN, identity) or identity in records:
            raise ValueError('Invalid or duplicate reviewed feature ID')
        records[identity] = row
    return records


def _feature_values(record, duration_ms, minimum_frames):
    if record['feature_schema'] != FEATURE_SCHEMA or type(record['fps']) not in (int, float) or record['fps'] != FPS:
        raise ValueError('Reviewed features must use the current schema at 15 FPS')
    if any(not _matches(HASH, record[key]) for key in ('extractor_sha256', 'source_key') if key in record):
        raise ValueError('Invalid optional feature provenance digest')
    elapsed = record.get('elapsed_ms')
    if elapsed is not None and (type(elapsed) not in (int, float) or not 0 <= elapsed <= 900_000):
        raise ValueError('Invalid reviewed feature extraction time')
    rows = record['features']
    if not isinstance(rows, list) or not max(6, minimum_frames) <= len(rows) <= 182 or abs(len(rows) - duration_ms * FPS / 1000) > 2:
        raise ValueError('Reviewed frame count cannot represent the range and glosses')
    if 'source_frames' in record or 'stride' in record:
        source_frames, stride = record.get('source_frames'), record.get('stride')
        if (type(source_frames) is not int or type(stride) is not int or not 1 <= stride <= 8
                or (source_frames + stride - 1) // stride != len(rows)):
            raise ValueError('Reviewed frame extraction counts disagree')
    tracked = 0
    for frame in rows:
        if not isinstance(frame, list) or len(frame) != FEATURE_COUNT or any(
                type(value) not in (int, float) or not -8 <= value <= 8 for value in frame):
            raise ValueError('Invalid reviewed landmark feature values or shape')
        left, right, pose, face = frame[-4:]
        if any(mask not in (0, 1) for mask in (left, right, pose, face)):
            raise ValueError('Reviewed landmark tracking masks must be binary')
        tracked += int(bool(pose and (left or right)))
    quality = record['quality']
    if (type(quality) not in (int, float) or not MIN_TRACKING_QUALITY <= quality <= 1
            or abs(quality - tracked / len(rows)) > 1e-6 or tracked < minimum_frames):
        raise ValueError('Reviewed tracking quality is too low or inconsistent with the landmarks')
    return torch.tensor(rows, dtype=torch.float32), float(quality), elapsed


def _samples(manifest, features, exclusions):
    content = _parse(manifest)
    if (not isinstance(content, dict) or set(content) != {'samples'} or not isinstance(content['samples'], list)
            or not 1 <= len(content['samples']) <= MAX_BATCH_SAMPLES):
        raise ValueError('A reviewed manifest must contain 1–24 samples')
    feature_rows = _parse_features(features)
    rows = []
    for sample in content['samples']:
        if not isinstance(sample, dict) or set(sample) != SAMPLE_KEYS or sample['consent_local_ml'] is not True:
            raise ValueError('Invalid reviewed sample schema or missing explicit local ML consent')
        identity, signer, session = (sample[key] for key in ('id', 'signer', 'session'))
        source = sample['source_sha256']
        if not all(_matches(TOKEN, value) for value in (identity, signer, session)) or not _matches(HASH, source):
            raise ValueError('Invalid reviewed identity, session, or source digest')
        start, end = sample['start_ms'], sample['end_ms']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= 90_000 or not 400 <= end - start <= 12_000:
            raise ValueError('A reviewed range must be 0.4–12 seconds within a 90-second source')
        glosses = sample['glosses']
        if (not isinstance(glosses, list) or not 1 <= len(glosses) <= 24
                or any(not isinstance(gloss, str) or gloss not in exclusions.labels for gloss in glosses)):
            raise ValueError('A reviewed gloss is missing or outside the frozen vocabulary')
        minimum_frames = len(glosses) + sum(a == b for a, b in zip(glosses, glosses[1:]))
        text = sample['text']
        if not isinstance(text, str) or not 1 <= len(text) <= 300 or any(unicodedata.category(char).startswith('C') for char in text):
            raise ValueError('Invalid reviewed Greek reference text')
        text = unicodedata.normalize('NFC', text.strip())
        if not _text_key(text):
            raise ValueError('A reviewed Greek reference needs words')
        if identity not in feature_rows:
            raise ValueError('A reviewed sample has missing or duplicate landmark data')
        frames, quality, elapsed = _feature_values(feature_rows.pop(identity), end - start, minimum_frames)
        row = Utterance(identity, f'reviewed/{source}/{start}-{end}', signer, 'train', tuple(glosses), text,
                        frames, float(FPS), quality, elapsed)
        exclusions.register(row, source, (start, end), session.casefold())
        rows.append(row)
    if feature_rows:
        raise ValueError('The reviewed feature file contains an unmatched row')
    return rows


def _inside(path, parent):
    if path.is_symlink() or path.is_junction() or path.resolve().parent != parent.resolve():
        raise ValueError('A reviewed storage path must stay inside its expected directory')


def _sealed_batches(root, exclusions):
    folder = root / 'approved'
    _inside(folder, root)
    if not folder.exists():
        return [], {}
    if not folder.is_dir():
        raise ValueError('The approved bundle folder is invalid')
    paths = sorted(folder.iterdir())
    if len(paths) > MAX_APPROVED_BATCHES:
        raise ValueError('Too many approved review bundles')
    rows, digests = [], {}
    for path in paths:
        _inside(path, folder)
        if not _matches(BATCH, path.name) or not path.is_dir() or {child.name for child in path.iterdir()} != FILES.keys():
            raise ValueError('An approved bundle has an unexpected directory or file layout')
        for name in FILES:
            _inside(path / name, path)
        seal = _read(path / 'approval.json', APPROVAL_LIMIT)
        digest = _digest(seal)
        approval = _parse(seal)
        if (path.name != digest[:32] or not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS
                or type(approval['schema_version']) is not int or approval['schema_version'] != 1
                or not _matches(TOKEN, approval['reviewer'])):
            raise ValueError('An approved bundle seal is invalid or changed')
        manifest = _read(path / 'manifest.json', MANIFEST_LIMIT)
        features = _read(path / 'features.jsonl', FEATURES_LIMIT)
        if _digest(manifest) != approval['manifest_sha256'] or _digest(features) != approval['features_sha256']:
            raise ValueError('An approved bundle file has changed since review')
        rows.extend(_samples(manifest, features, exclusions))
        if len(rows) > MAX_APPROVED_SAMPLES:
            raise ValueError('Too many approved reviewed samples')
        digests[path.name] = digest
    return rows, digests


def load_approved(state_dir, base_rows, protected_rows, labels):
    """Read verified train rows and the immutable approval digests for a run."""
    return _sealed_batches(Path(state_dir).resolve(), _Exclusions(base_rows, protected_rows, labels))


def approve(state_dir, manifest_path, features_path, reviewer, base_rows, protected_rows, labels):
    """Record an explicit local operator approval after validating submitted bytes."""
    if not _matches(TOKEN, reviewer):
        raise ValueError('A reviewer ID of up to 64 technical characters is required')
    manifest = _read(Path(manifest_path), MANIFEST_LIMIT)
    features = _read(Path(features_path), FEATURES_LIMIT)
    approval = {'schema_version': 1, 'reviewer': reviewer, 'manifest_sha256': _digest(manifest),
                'features_sha256': _digest(features)}
    seal = (json.dumps(approval, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    digest = _digest(seal)
    batch_id = digest[:32]
    root = Path(state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / '.approval.lock'
    _inside(lock_path, root)
    lock = FileLock(lock_path)
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise ValueError('Another local review approval already holds the lock') from error
    try:
        exclusions = _Exclusions(base_rows, protected_rows, labels)
        existing, digests = _sealed_batches(root, exclusions)
        if batch_id in digests:
            if digests[batch_id] != digest:
                raise ValueError('A reviewed batch ID collides with an existing approval')
            return {'batch_id': batch_id, 'sha256': digest, 'samples': len(_parse(manifest)['samples']), 'already_approved': True}
        rows = _samples(manifest, features, exclusions)
        if len(existing) + len(rows) > MAX_APPROVED_SAMPLES or len(digests) >= MAX_APPROVED_BATCHES:
            raise ValueError('The local approved training collection reached its size limit')
        folder = root / 'approved'
        folder.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix='.review-', dir=root) as directory:
            stage = Path(directory) / 'bundle'
            stage.mkdir()
            for name, data in zip(FILES, (manifest, features, seal), strict=True):
                path = stage / name
                with path.open('wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _digest(_read(path, FILES[name])) != _digest(data):
                    raise ValueError('A reviewed bundle copy changed before publication')
            stage.rename(folder / batch_id)
    finally:
        lock.release()
    return {'batch_id': batch_id, 'sha256': digest, 'samples': len(rows), 'already_approved': False}
