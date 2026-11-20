"""Replay a recovery-only local hand-motion floor on real and artificial inputs."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_openhands_agreement import CACHE, REPORTS, load_rgb, probabilities
from evaluate_openhands_verifier import counts, make_session, sha256, write_json
from openhands_gsl import load_openhands_gsl, prepare_pose
from openhands_gsl_stgcn import prepare_stgcn_pose
from openset_core import motion_metrics
from preprocess import preprocess


SEED_COUNT = 24
MIN_TRACKING = .6
MIN_BODY_MOTION = .004
MIN_RECOVERY_LOCAL_MOTION = .005
RAW_GLOBS = ('data/features/*.jsonl', 'data/open-set-v2/features/*.jsonl', 'data/v3/features/*.jsonl')
CONTROLS = ('missing_person', 'frozen', 'camera_pan', 'tiny_wrist_jitter', 'artificial_reach')


def raw_records(patterns):
    paths = sorted({path for pattern in patterns for path in Path('.').glob(pattern)})
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('raw'):
                    yield row['id'], row['raw']


def first_tracked_frames():
    seeds = {}
    for identity, frames in raw_records(RAW_GLOBS[:1]):
        if identity in seeds or not {'t', 'width', 'height'} <= frames[0].keys():
            continue
        for frame in frames:
            pose = frame['pose']
            if (len(pose) == 33 and min(pose[i].get('visibility', 1) for i in (11, 12)) >= .5
                    and (len(frame['left']) == 21 or len(frame['right']) == 21)):
                seeds[identity] = frame
                break
        if len(seeds) == SEED_COUNT:
            return seeds
    raise ValueError('Insufficient distinct tracked seeds')


def perturb(source, kind):
    frames = []
    for i in range(48):
        frame = deepcopy(source)
        frame['t'] = i * 1000 / 30
        wave = np.sin(i * np.pi / 47)
        if kind == 'missing_person':
            for part in ('left', 'right', 'pose', 'face'):
                frame[part] = []
        elif kind == 'camera_pan':
            for part in ('left', 'right', 'pose', 'face'):
                for point in frame[part]:
                    point['x'] += .04 * wave
        elif kind in ('tiny_wrist_jitter', 'artificial_reach'):
            hand = frame['left'] or frame['right']
            dy = .0002 * np.sin(i * 1.8) if kind == 'tiny_wrist_jitter' else .30 * wave
            for point in hand:
                point['y'] += dy
        frames.append(frame)
    return frames


def motion(frames):
    x, coverage = preprocess(frames)
    values = motion_metrics(x[None, :, :])
    return coverage, float(values['motionBodyRms'][0]), float(values['motionLocalRms'][0])


def pose_array(frames):
    points = np.zeros((len(frames), 75, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        for part, start, size in (('pose', 0, 33), ('left', 33, 21), ('right', 54, 21)):
            if len(frame[part]) >= size:
                points[i, start:start + size, :2] = [(point['x'], point['y']) for point in frame[part][:size]]
    return points


def decisions(primary, secondary, eligible, local, roster, policy):
    winner = primary.argmax(axis=1)
    rows = np.arange(len(winner))
    confidence = probabilities(primary)[rows, winner]
    evidence = primary[rows, winner]
    alternate_score = probabilities(secondary)[rows, winner]
    rank = (secondary > secondary[rows, winner, None]).sum(axis=1) + 1
    gate, agreement = policy['decisionGate'], policy['agreementGate']
    strong = eligible & roster[winner] & (evidence >= gate['minimumEvidence'])
    baseline = strong & (confidence >= gate['minimumConfidence'])
    recovery = strong & ~baseline & (confidence >= agreement['minimumConfidence'])
    recovery &= (alternate_score >= agreement['minimumSecondaryProbability']) & (rank <= agreement['maximumSecondaryRank'])
    return winner, baseline, baseline | recovery, baseline | (recovery & (local >= MIN_RECOVERY_LOCAL_MOTION))


def main():
    torch.set_num_threads(1)
    model, labels = load_openhands_gsl(CACHE)
    label_index = {label: i for i, label in enumerate(labels)}
    session, _, _ = make_session(CACHE / 'stgcn.onnx', None)
    policy_path = REPORTS / 'stgcn-gate-agreement-candidate.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    roster = np.isin(labels, policy['supportedLabels'])
    ids, glosses, primary, secondary, tracked = load_rgb(session, model, label_index)
    refs = np.asarray([label_index[label] for label in glosses])
    rgb_frames = {}
    expected = set(ids)
    for identity, frames in raw_records(RAW_GLOBS):
        if identity in expected and identity not in rgb_frames:
            rgb_frames[identity] = frames
    coverage, body, local = np.asarray([motion(rgb_frames[identity]) for identity in ids]).T
    if not np.array_equal(coverage >= MIN_TRACKING, tracked):
        raise ValueError('Motion replay did not find the same raw frames as RGB inference')
    eligible = tracked & (body >= MIN_BODY_MOTION)
    winner, base, dual, checked = decisions(primary, secondary, eligible, local, roster, policy)
    removed = dual & ~checked
    recovered_motion = local[dual & ~base]
    real = {
        'bodyAndTrackingEligible': int(eligible.sum()),
        **{name: counts(mask, refs, winner, roster) for name, mask in
           (('primary', base), ('dual', dual), ('dualWithLocalMotion', checked))},
        'rescueLocalMotionRange': [float(recovered_motion.min()), float(recovered_motion.max())]
        if len(recovered_motion) else None,
        'withheldRescues': [{'id': str(ids[i]), 'expected': labels[refs[i]], 'winner': labels[winner[i]],
                             'localMotion': float(local[i])} for i in np.flatnonzero(removed)],
    }
    seeds = first_tracked_frames()
    artificial = {}
    for control in CONTROLS:
        rows = []
        for identity, frame in seeds.items():
            frames = perturb(frame, control)
            coverage, body, local = motion(frames)
            row = {'seedClip': identity, 'bodyMotion': body, 'localMotion': local}
            if coverage < MIN_TRACKING or body < MIN_BODY_MOTION:
                row['blockedBeforeClassifier'] = 'tracking' if coverage < MIN_TRACKING else 'no_motion'
            else:
                points = pose_array(frames)
                with torch.inference_mode():
                    a = session.run(['logits'], {'landmarks': prepare_stgcn_pose(points).numpy()})[0]
                    b = model(prepare_pose(points).unsqueeze(0)).numpy()
                winner, primary, dual, checked = decisions(a, b, True, np.array([local]), roster, policy)
                row.update(winner=labels[winner[0]], primaryFalseAccept=bool(primary[0]),
                           dualFalseAccept=bool(dual[0]), localMotionFalseAccept=bool(checked[0]))
            rows.append(row)
        artificial[control] = {
            'clips': len(rows), 'preClassifierRejections': dict(Counter(row['blockedBeforeClassifier'] for row in rows
                                                                     if 'blockedBeforeClassifier' in row)),
            'primaryFalseAccepts': sum(row.get('primaryFalseAccept', False) for row in rows),
            'dualFalseAccepts': sum(row.get('dualFalseAccept', False) for row in rows),
            'dualWithLocalMotionFalseAccepts': sum(row.get('localMotionFalseAccept', False) for row in rows),
            'rows': rows,
        }
    report = {'scope': 'Cached MediaPipe frames from the source corpus and artificial frame perturbations. '
                       'No independent signer or human everyday-motion performance claim. Recovery is enabled for this diagnostic.',
              'policySha256': sha256(policy_path), 'minimumBodyMotion': MIN_BODY_MOTION,
              'minimumLocalMotionOnRescue': MIN_RECOVERY_LOCAL_MOTION,
              'realRgb': real, 'artificial': artificial}
    path = REPORTS / 'stgcn-agreement-motion-controls.json'
    write_json(path, report)
    print(path, flush=True)
    print(json.dumps({'realRgb': real, 'artificial': {name: {k: v for k, v in row.items() if k != 'rows'}
                                                    for name, row in artificial.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
