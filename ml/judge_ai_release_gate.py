"""Decide whether a frozen isolated-sign candidate merits release review."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'publication-judge-demo'
CANDIDATE = ROOT / 'artifacts/judge-final-motion'
EVIDENCE = ROOT / 'reports/judge-final'
OUTPUT = ROOT / 'reports/judge-ai-development/release-gate.json'
FIELDS = ('correct_accepted', 'wrong_known_accepted', 'outside_accepted', 'abstained')


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def validate_outcomes(result):
    counts = (*FIELDS, 'known', 'outside_roster', 'known_top1_correct')
    if any(type(result[key]) is not int or result[key] < 0 for key in counts):
        raise ValueError('Invalid counts in paired evidence')
    if (result['correct_accepted'] + result['wrong_known_accepted'] > result['known']
            or not result['correct_accepted'] <= result['known_top1_correct'] <= result['known']
            or result['outside_accepted'] > result['outside_roster']
            or sum(result[key] for key in FIELDS) != result['known'] + result['outside_roster']):
        raise ValueError('Paired counts do not account for every video')


def outcome(reference, predicted, classes):
    if reference not in classes:
        return 'outside_accepted'
    return 'correct_accepted' if reference == predicted else 'wrong_known_accepted'


def paired_evidence(report, freeze_hash, classes, references):
    if report['freeze_sha256'] != freeze_hash:
        raise ValueError('Evaluation belongs to a different candidate')
    base, new = report['baseline'], report['hybrid']
    validate_outcomes(base)
    validate_outcomes(new)
    known = sum(word in classes for word in references.values())
    if (base['known'], base['outside_roster']) != (known, len(references) - known):
        raise ValueError('Evaluation counts differ from the input cohort')
    if (base['known'], base['outside_roster'], base['known_top1_correct']) != (new['known'], new['outside_roster'], new['known_top1_correct']):
        raise ValueError('Compared models must share the same examples and proposed labels')
    moves, identities = Counter(), set()
    regressions = []
    for row in report['changed']:
        identity = row['id']
        before, after = row['baseline_accepts'], row['hybrid_accepts']
        if (identity in identities or references.get(identity) != row['reference']
                or type(before) is not bool or type(after) is not bool or before == after
                or row['candidate'] not in classes):
            raise ValueError('Invalid or duplicated decision change')
        identities.add(identity)
        field = outcome(row['reference'], row['candidate'], classes)
        if (field == 'correct_accepted') == before:
            regressions.append({'id': identity, 'outcome': field})
        moves[field] += int(after) - int(before)
        moves['abstained'] += int(before) - int(after)
    delta = {key: new[key] - base[key] for key in FIELDS}
    if any(delta[key] != moves[key] for key in FIELDS):
        raise ValueError('Decision rows do not match the metric changes')
    return {'examples': base['known'] + base['outside_roster'], 'incumbent': base, 'candidate': new,
            'delta': delta, 'changed_count': len(identities), 'regressions': regressions, 'scope': report['scope']}


def assess(args):
    artifact = args.candidate
    proof = args.evidence
    package = args.package
    audit = read(package / 'library-audit.json')
    paths = ('__signbridge/openhands/policy.json', '__signbridge/openhands/stgcn.onnx', '__signbridge/openhands/lstm.onnx')
    observed = {name: digest(package / 'dist' / name) for name in paths}
    if any(observed[name] != audit['runtime_files'][name] for name in paths):
        raise ValueError('Incumbent package is different from the audited build')
    policy = read(package / 'dist' / paths[0])
    freeze_path = artifact / 'freeze.json'
    freeze = read(freeze_path)
    freeze_hash = digest(freeze_path)
    classes = set(policy['supportedLabels'])
    if classes != set(freeze['classes']) or freeze['baseline_policy_sha256'] != observed[paths[0]]:
        raise ValueError('Candidate does not belong to the active vocabulary and policy')
    model_hash = digest(artifact / 'motion.onnx')
    if model_hash != freeze['files']['motion.onnx'] or digest(artifact / 'motion.pt') != freeze['files']['motion.pt']:
        raise ValueError('Model checkpoint differs from the frozen candidate')

    parity_path = proof / 'browser-motion-parity.json'
    parity = read(parity_path)
    if (parity['passed'] is not True or parity['actualChrome'] is not True
            or parity['provenance']['modelSha256'] != model_hash or parity['provenance']['freezeSha256'] != freeze_hash
            or parity['summary']['cases'] <= 0):
        raise ValueError('Browser parity was not proved for this model')
    native_report = proof / 'native-comparison.json'
    video_report = proof / 'rgb-comparison.json'
    native, video = read(native_report), read(video_report)
    if native['held_sha256'] != digest(args.held) or video['browser_export_sha256'] != digest(args.rgb):
        raise ValueError('The held evidence changed')
    with np.load(args.held, allow_pickle=False) as poses:
        held_rows = list(zip(poses['ids'].tolist(), poses['glosses'].tolist(), strict=True))
    video_rows = [json.loads(line) for line in args.rgb.read_text(encoding='utf-8').splitlines()]
    held_refs = dict(held_rows)
    video_refs = {row['id']: row['gloss'] for row in video_rows}
    held_signers = {int(re.search(r'_signer(\d+)_', identity)[1]) for identity in held_refs}
    if (len(held_refs) != len(held_rows) or len(video_refs) != len(video_rows)
            or held_signers != set(native['held_signers'])
            or held_signers & (set(freeze['train']['signers']) | set(freeze['validation']['signers']))):
        raise ValueError('The held evidence changed or leaks into training')
    tests = {'native_poses': paired_evidence(native, freeze_hash, classes, held_refs),
             'rgb_videos': paired_evidence(video, freeze_hash, classes, video_refs)}
    reasons = []
    for key, item in tests.items():
        if item['regressions']:
            reasons.append(f'{key}: {len(item["regressions"])} clips lost a correct accept or gained a false accept')
    if tests['rgb_videos']['delta']['correct_accepted'] == 0:
        reasons.append('rgb_videos: no additional correct accepts on the paired footage')
    evidence_hashes = {path.name: digest(path) for path in (native_report, video_report, parity_path)}
    identity = sha256(json.dumps([observed, freeze_hash, evidence_hashes], sort_keys=True).encode()).hexdigest()
    return {'assessment_id': identity, 'decision': 'eligible_for_review' if not reasons else 'keep_incumbent',
            'reasons': reasons, 'incumbent_asset_sha256': observed, 'candidate_onnx_sha256': model_hash,
            'frozen_candidate_sha256': freeze_hash, 'evidence_sha256': evidence_hashes,
            'active_classes': len(classes), 'browser_model_parity_cases': parity['summary']['cases'], 'tests': tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=BASE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--held', type=Path, default=ROOT / 'research/cache/openhands-gsl/judge-final-held-poses.npz')
    parser.add_argument('--rgb', type=Path, default=ROOT / '.runtime-tmp/judge-final-rgb/poses.jsonl')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = assess(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix('.tmp')
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(args.output)
    print(json.dumps({'decision': result['decision'], 'reasons': result['reasons'], 'report': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
