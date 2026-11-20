"""Search verified GSL references and attach source evidence to a trained recognizer."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import torch

from evaluate_isolated_exemplar_retrieval import load_frozen_exemplar_fusion
from train_isolated_large_vocabulary import infer, load_checkpoint, prepare


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from plan_continuous_composition import normalize_greek


def index(path):
    if not path.is_file():
        raise ValueError('Build the verified library index with scripts/build_gsl_provenance_index.py')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def search(db, text, limit):
    term = normalize_greek(text)
    if not term:
        raise ValueError('Search for a Greek word, phrase or source gloss')
    groups = db.execute('''
        SELECT a.kind AS annotation_kind, a.value, COUNT(DISTINCT u.key) AS source_records
        FROM annotation a JOIN unit u ON u.key = a.unit_key
        WHERE a.lookup = ? GROUP BY a.kind, a.value ORDER BY source_records DESC, a.value LIMIT ?
    ''', (term, limit))
    videos = db.execute('''
        SELECT DISTINCT r.greek_text, r.local_path, r.asset_kind, c.source_url, c.license, u.source_id
        FROM reference_library r JOIN unit u ON u.key = r.unit_key JOIN corpus c ON c.id = u.corpus
        WHERE r.lookup = ? AND r.asset_kind IN ('public_video', 'reference_video', 'source_video')
        ORDER BY r.asset_kind, r.greek_text, u.source_id LIMIT ?
    ''', (term, limit))
    return {'query': text, 'matching_annotations': [dict(row) for row in groups],
            'verified_video_references': [{**dict(row), 'local_path': str(ROOT / row['local_path'])} for row in videos]}


def source_pose(db, identity):
    row = db.execute('''
        SELECT a.local_path, a.locator, a.payload_sha256
        FROM unit u JOIN asset a ON a.unit_key = u.key
        WHERE u.corpus = 'certh' AND u.kind = 'isolated' AND u.source_id = ?
          AND a.kind IN ('native_pose', 'native_pose_expanded')
        ORDER BY a.kind DESC LIMIT 1
    ''', (identity,)).fetchone()
    if row is None:
        raise ValueError('The library has no local pose for that official source ID')
    path = (ROOT / row['local_path']).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('An indexed pose asset is outside the project')
    location = int(row['locator'])
    with np.load(path, allow_pickle=False) as archive:
        ids, offsets = archive['ids'], archive['offsets']
        if location < 0 or location >= len(ids) or ids[location] != identity:
            raise ValueError('The indexed pose row differs from the source cache')
        raw = archive['keypoints'][offsets[location]:offsets[location + 1]]
    if sha256(raw.astype('<f4', copy=False).tobytes()).hexdigest() != row['payload_sha256']:
        raise ValueError('The source pose differs from its verified library digest')
    return raw


def checked_pose(path):
    if path.suffix.lower() != '.npy' or path.stat().st_size > 3_000_000:
        raise ValueError('Use a bounded native T × 75 × 3 .npy pose file')
    return np.load(path, allow_pickle=False)


def training_examples(db, gloss, model_manifest, cache, supporting=None):
    local = db.execute('SELECT sha256 FROM artifact WHERE path = ?', (cache,)).fetchone()
    if not local or local['sha256'] != model_manifest['pose_cache_sha256']:
        raise ValueError('The model training cache is missing from the verified library version')
    people = [str(value) for value in model_manifest['train_signers']]
    placeholders = ', '.join('?' for _ in people)
    rows = db.execute(f'''
        SELECT u.key AS lexical_unit, u.source_id, u.signer, u.scenario, a.frames
        FROM unit u JOIN annotation n ON n.unit_key = u.key JOIN asset a ON a.unit_key = u.key
        JOIN membership m ON m.unit_key = u.key
        WHERE n.kind = 'gloss' AND n.value = ? AND u.kind = 'isolated' AND a.local_path = ?
          AND m.policy = 'official-isolated' AND m.role = 'train' AND u.signer IN ({placeholders})
        ORDER BY u.signer, u.source_id
    ''', (gloss, cache, *people)).fetchall()
    examples = {}
    matched = None
    for row in rows:
        examples.setdefault(row['signer'], dict(row))
        if row['lexical_unit'] == supporting:
            matched = dict(row)
    if supporting is not None and matched is None:
        raise ValueError('The matched exemplar is outside the verified model training subset')
    return {'training_clips': len(rows), 'training_signers': sorted(examples),
            'verified_examples': list(examples.values()), 'matched_example': matched}


def recognize(db, raw, arm, limit):
    if (raw.ndim != 3 or raw.shape[1:] != (75, 3) or not 2 <= len(raw) <= 1500
            or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all()):
        raise ValueError('Expected a finite native T × 75 × 3 pose with 2–1500 frames')
    model, labels, freeze = load_checkpoint(arm)
    frames = prepare(raw, np.array([0, len(raw)], dtype=np.int64))
    logits = infer(model, frames)
    fusion = load_frozen_exemplar_fusion(arm).predict(raw, logits[0].numpy(), labels)
    filename = 'official-train-expanded-poses.npz' if arm == 'expanded' else 'official-train-prototypes-poses.npz'
    cache = f'research/cache/openhands-gsl/{filename}'
    candidates = []
    for candidate in fusion['top5'][:limit]:
        gloss = candidate['gloss']
        supporting = fusion['supporting_lexical_unit'] if gloss == fusion['gloss'] else None
        support = training_examples(db, gloss, freeze, cache, supporting)
        if not support['training_clips']:
            raise ValueError(f'The model label is missing its audited training evidence: {gloss}')
        candidates.append({'gloss': gloss, 'ranking_score': candidate['score'], **support})
    return {'experiment': arm, 'vocabulary_size': len(labels), 'model_sha256': freeze['checkpoint_sha256'],
            'accepted_gloss': fusion['gloss'] if fusion['accepted'] else None, 'candidates': candidates,
            'scope': 'Experimental isolated-sign inference with training exemplars on native pose sequences.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / 'data/gsl-provenance.sqlite')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('stats')
    lookup = commands.add_parser('search')
    lookup.add_argument('text')
    lookup.add_argument('--limit', type=int, default=8)
    predict = commands.add_parser('recognize')
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--pose', type=Path)
    inputs.add_argument('--source-id')
    predict.add_argument('--arm', choices=('expanded', 'full'), default='expanded')
    predict.add_argument('--limit', type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    if hasattr(args, 'limit') and not 1 <= args.limit <= 25:
        parser.error('Use a result limit between 1 and 25')
    path = args.index.resolve()
    pose_path = args.pose.resolve() if args.command == 'recognize' and args.pose else None
    os.chdir(ROOT)
    torch.set_num_threads(2)
    with closing(index(path)) as db:
        if args.command == 'stats':
            result = {row['key']: json.loads(row['value_json']) for row in db.execute('SELECT * FROM index_info')}
        elif args.command == 'search':
            result = search(db, args.text, args.limit)
        else:
            raw = checked_pose(pose_path) if pose_path else source_pose(db, args.source_id)
            result = recognize(db, raw, args.arm, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
