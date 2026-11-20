"""Validate local paired V8 exports and print a bounded, pseudonymous summary.

Read-only: no fitting, inference, uploads, video decoding or identity verification.
An internally consistent report is not proof of genuine recording or GSL truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from v7_intake import GSL_REFERENCES, digest, read_json, reference_digests

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ')
GROUPS = ('known', 'close_unknown', 'daily', 'tracking')
HASHES = {
    'model': 'beaed67bc63103370754cdbf25eda012f4464c66af74a11a7923a127254c2a45',
    'baselineConfig': '817c389851b13c61b0b7e0feb68e62b848082d049b8abfd7bb5adeb36ab1b7d6',
    'v8Config': 'bc0b931319d25986f3c14b9b9aba699ffe448d200f6c147b4b62cf4791f21bc7',
    'handModel': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
}
THRESHOLDS = (0.9998770457578423, .9, .9)
CLIP_ID = re.compile(r'V8_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}')
SIGNER = re.compile(r'P[0-9]{3,6}')
SESSION = re.compile(r'[A-Za-z0-9_-]{1,24}')
CATEGORY = re.compile(r'u_[a-z0-9][a-z0-9_]{0,37}')
SHA = re.compile(r'[0-9a-f]{64}')
TRACKING_MESSAGES = (
    'Χρειάζεται επανάληψη: καταγράφηκαν πολύ λίγα καρέ.',
    'Χρειάζεται επανάληψη: η λήψη έχει μη έγκυρα καρέ ή χρονική σειρά.',
    'Χρειάζεται επανάληψη: κράτησε και τα δύο χέρια και το πάνω μέρος του σώματος ορατά σε όλη τη λήψη.',
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def keys(value, names, code):
    require(isinstance(value, dict) and set(value) == set(names.split()), code)


def number(value, low=0, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def integer(value, low=0, high=2**53):
    return type(value) is int and low <= value <= high


def matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def near(actual, expected):
    return number(actual, -math.inf) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def tracking(value, frames):
    keys(value, 'policyVersion source frameCount poseCompleteness leftHandCompleteness '
         'rightHandCompleteness facePresence eitherHandDetection requiredDataCompleteness '
         'allRequiredFramesPresent usable reasons minRequiredCoverage minimumFrames '
         'invalidMetadataFrames timestampsStrictlyIncreasing missingFrameCounts longestMissingRuns '
         'validMotionPairFraction validMotionPairs totalMotionPairs', 'tracking_schema_invalid')
    require(value['policyVersion'] == 'signbridge-raw-both-hands-v4-20261108'
            and value['source'] == 'raw_landmark_frames' and value['frameCount'] == frames
            and type(value['frameCount']) is int and value['minRequiredCoverage'] == .9
            and type(value['minimumFrames']) is int and value['minimumFrames'] == 4,
            'tracking_policy_mismatch')
    for key in ('allRequiredFramesPresent', 'usable', 'timestampsStrictlyIncreasing'):
        require(type(value[key]) is bool, 'tracking_boolean_invalid')
    require(integer(value['invalidMetadataFrames'], high=frames), 'tracking_metadata_count_invalid')
    for key in ('poseCompleteness', 'leftHandCompleteness', 'rightHandCompleteness',
                'facePresence', 'eitherHandDetection', 'requiredDataCompleteness', 'validMotionPairFraction'):
        require(number(value[key], high=1), 'tracking_fraction_invalid')
    for key in ('facePresence', 'eitherHandDetection'):
        require(near(value[key] * frames, round(value[key] * frames)), 'tracking_nonintegral_frame_fraction')
    for key in ('missingFrameCounts', 'longestMissingRuns'):
        keys(value[key], 'pose leftHand rightHand required', 'tracking_counts_schema_invalid')
        require(all(integer(v, high=frames) for v in value[key].values()), 'tracking_count_invalid')
    missing = value['missingFrameCounts']
    for key, fraction in (('pose', 'poseCompleteness'), ('leftHand', 'leftHandCompleteness'),
                          ('rightHand', 'rightHandCompleteness'), ('required', 'requiredDataCompleteness')):
        require(near(value[fraction], (frames - missing[key]) / frames), 'tracking_fraction_count_mismatch')
        run = value['longestMissingRuns'][key]
        require((run == 0 if missing[key] == 0 else 1 <= run <= missing[key]), 'tracking_missing_run_invalid')
    required = frames - missing['required']
    present = [frames - missing[key] for key in ('pose', 'leftHand', 'rightHand')]
    require(max(0, sum(present) - 2 * frames) <= required <= min(present), 'tracking_joint_count_invalid')
    require(value['requiredDataCompleteness'] <= value['eitherHandDetection'] + 1e-10
            and value['eitherHandDetection'] <= value['poseCompleteness'] + 1e-10,
            'tracking_either_hand_invalid')
    require(value['allRequiredFramesPresent'] == (required == frames), 'tracking_all_frames_mismatch')
    pairs = value['validMotionPairs']
    require(integer(pairs, high=max(0, required - 1)) and integer(value['totalMotionPairs'])
            and value['totalMotionPairs'] == frames - 1
            and near(value['validMotionPairFraction'], pairs / (frames - 1)), 'tracking_pairs_invalid')
    if value['timestampsStrictlyIncreasing']:
        require(pairs >= max(0, 2 * required - frames - 1), 'tracking_pair_count_impossible')
    reasons = []
    if frames < 4:
        reasons.append('too_few_frames')
    if value['invalidMetadataFrames'] or not value['timestampsStrictlyIncreasing']:
        reasons.append('invalid_frame_metadata')
    if required / frames < .9:
        reasons.append('incomplete_required_landmarks')
    require(value['reasons'] == reasons and value['usable'] == (not reasons), 'tracking_verdict_mismatch')
    return reasons


def comparison(value, duration_ms):
    keys(value, 'researchOnly source modelVersion baselineModelVersion baseline v8 logits probabilities '
         'classifierExecuted originalTracking fusedTracking quality diagnostics frames recoveredFrames timings hashes',
         'comparison_schema_invalid')
    require(value['researchOnly'] is True and value['source'] == 'video', 'not_exported_video_comparison')
    require(value['hashes'] == HASHES
            and value['modelVersion'] == 'signbridge-v8-v4-confidence-pilot'
            and value['baselineModelVersion'] == 'signbridge-open-set-v2-B_tcn_diverse_other',
            'frozen_version_mismatch')
    frames = value['frames']
    expected_frames = min(48, max(2, math.floor(duration_ms * .03 + .5)))
    require(integer(frames, 2, 48) and frames == expected_frames
            and integer(value['recoveredFrames'], high=frames), 'comparison_frame_count_invalid')
    tracking(value['originalTracking'], frames)
    reasons = tracking(value['fusedTracking'], frames)
    original, fused = value['originalTracking'], value['fusedTracking']
    unchanged = ('poseCompleteness', 'leftHandCompleteness', 'facePresence',
                 'invalidMetadataFrames', 'timestampsStrictlyIncreasing')
    require(all(original[k] == fused[k] for k in unchanged)
            and all(original[part][k] == fused[part][k] for part in ('missingFrameCounts', 'longestMissingRuns')
                    for k in ('pose', 'leftHand')), 'recovery_changed_preserved_stream')
    recovered = value['recoveredFrames']
    required_gain = original['missingFrameCounts']['required'] - fused['missingFrameCounts']['required']
    require(original['missingFrameCounts']['rightHand'] - fused['missingFrameCounts']['rightHand'] == recovered
            and 0 <= required_gain <= recovered and (not original['usable'] or fused['usable'])
            and original['eitherHandDetection'] <= fused['eitherHandDetection'] + 1e-10,
            'recovery_counts_impossible')
    require(type(value['classifierExecuted']) is bool and value['classifierExecuted'] == (not reasons),
            'inference_tracking_mismatch')
    keys(value['timings'], 'modelLoadMs decodeMs recoveryMs recoveryWorkerMs recoveryInitMs preprocessMs '
         'inferenceRoundTripMs decisionMs totalMs', 'timings_schema_invalid')
    require(all(number(v) for v in value['timings'].values()), 'timings_invalid')
    for name in ('baseline', 'v8'):
        keys(value[name], 'accepted label predicted classificationScore threshold reasons', 'decision_schema_invalid')
        require(type(value[name]['accepted']) is bool, 'decision_boolean_invalid')
    if reasons:
        message = TRACKING_MESSAGES[0 if 'too_few_frames' in reasons else 1 if 'invalid_frame_metadata' in reasons else 2]
        expected = {'accepted': False, 'label': None, 'predicted': None, 'classificationScore': None,
                    'threshold': None, 'reasons': [message]}
        require(all(value[name] == expected for name in ('baseline', 'v8'))
                and all(value[key] is None for key in ('logits', 'probabilities', 'quality', 'diagnostics')),
                'preinference_decision_invalid')
        return
    logits = value['logits']
    require(isinstance(logits, list) and len(logits) == 4
            and all(number(v, -1e6, 1e6) for v in logits), 'logits_invalid')
    exponents = [math.exp((v - max(logits)) / 2) for v in logits]
    probs = [v / sum(exponents) for v in exponents]
    reported = value['probabilities']
    require(isinstance(reported, list) and len(reported) == 4
            and all(near(a, b) for a, b in zip(reported, probs)), 'probabilities_logits_mismatch')
    require(number(value['quality'], high=1)
            and near(value['quality'], value['fusedTracking']['eitherHandDetection']), 'quality_tracking_mismatch')
    diagnostics = value['diagnostics']
    keys(diagnostics, 'motionBodyRms motionLocalRms validTrackedPairFraction meanPresence', 'diagnostics_schema_invalid')
    require(number(diagnostics['motionBodyRms']) and number(diagnostics['motionLocalRms'])
            and number(diagnostics['validTrackedPairFraction'], high=1)
            and isinstance(diagnostics['meanPresence'], list) and len(diagnostics['meanPresence']) == 4
            and all(number(v, high=1 + 1e-10) for v in diagnostics['meanPresence']), 'diagnostics_invalid')
    predicted = max(range(3), key=probs.__getitem__)
    score = probs[predicted]
    base_reasons = []
    if value['quality'] < .25:
        base_reasons.append('quality')
    if score < .9:
        base_reasons.append('confidence')
    if max(range(4), key=probs.__getitem__) == 3:
        base_reasons.append('background')
    for name, threshold in (('baseline', .9), ('v8', THRESHOLDS[predicted])):
        decision = value[name]
        expected_reasons = base_reasons + (['v8_confidence_veto'] if name == 'v8' and score < threshold else [])
        accepted = not expected_reasons
        require(integer(decision['predicted'], high=2) and decision['predicted'] == predicted
                and near(decision['classificationScore'], score) and decision['threshold'] == threshold
                and decision['accepted'] == accepted and decision['label'] == (LABELS[predicted] if accepted else None)
                and decision['reasons'] == expected_reasons, 'paired_verdict_mismatch')


def collection(value):
    if value is None:
        return
    keys(value, 'signer_id session_id condition_id trial_group truth consent_local_ml human_recorded new_participant_declared',
         'collection_schema_invalid')
    require(matches(SIGNER, value['signer_id']) and matches(SESSION, value['session_id'])
            and value['condition_id'] in ('A', 'B') and value['trial_group'] in GROUPS,
            'collection_metadata_invalid')
    require(all(value[key] is True for key in ('consent_local_ml', 'human_recorded', 'new_participant_declared')),
            'human_declarations_required')
    truth = value['truth']
    if value['trial_group'] == 'known':
        keys(truth, 'kind id', 'known_truth_invalid')
        require(truth['kind'] == 'known' and truth['id'] in LABELS, 'known_truth_invalid')
    else:
        keys(truth, 'kind category_id', 'unknown_truth_invalid')
        require(truth['kind'] == 'unknown' and matches(CATEGORY, truth['category_id']), 'unknown_truth_invalid')


def validate_report(value):
    keys(value, 'schema clip_id source media collection comparison camera', 'report_schema_invalid')
    require(value['schema'] == 'signbridge-v8-pilot-v1', 'report_schema_invalid')
    require(matches(CLIP_ID, value['clip_id']), 'clip_id_invalid')
    require(value['source'] in ('file', 'webcam'), 'source_invalid')
    media = value['media']
    keys(media, 'sha256 mime_type byte_length duration_ms width height', 'media_schema_invalid')
    require(matches(SHA, media['sha256']) and integer(media['byte_length'], 1)
            and number(media['duration_ms'], 1e-6, 15000)
            and integer(media['width'], 1, 16384) and integer(media['height'], 1, 16384), 'media_metadata_invalid')
    require(isinstance(media['mime_type'], str)
            and re.fullmatch(r'video/(mp4|webm)(;[A-Za-z0-9=,. _"-]+)?', media['mime_type']), 'media_mime_invalid')
    camera = value['camera']
    if value['source'] == 'file':
        require(camera is None, 'file_camera_metadata_invalid')
    else:
        keys(camera, 'preview_mirrored width height frame_rate facing_mode', 'camera_schema_invalid')
        require(type(camera['preview_mirrored']) is bool
                and all(camera[k] is None or integer(camera[k], 1, 16384) for k in ('width', 'height'))
                and (camera['frame_rate'] is None or number(camera['frame_rate'], 0, 1000))
                and camera['facing_mode'] in (None, '', 'user', 'environment', 'left', 'right'), 'camera_metadata_invalid')
    collection(value['collection'])
    comparison(value['comparison'], media['duration_ms'])


def counts(rows):
    result = {'trials': len(rows), 'known_trials': 0, 'unknown_trials': 0, 'tracking_rejected_before_inference': 0,
              'classifier_executed': 0, 'baseline': Counter(), 'v8': Counter(),
              'paired': {'wrong_acceptances_removed': 0, 'correct_acceptances_lost': 0, 'unchanged': 0}}
    for name in ('baseline', 'v8'):
        result[name].update({key: 0 for key in ('accepted', 'rejected', 'correct_known_accepted',
                                               'wrong_known_accepted', 'unknown_false_accepted')})
    for row in rows:
        truth, pair = row['collection']['truth'], row['comparison']
        known = truth['kind'] == 'known'
        result['known_trials' if known else 'unknown_trials'] += 1
        result['classifier_executed' if pair['classifierExecuted'] else 'tracking_rejected_before_inference'] += 1
        for name in ('baseline', 'v8'):
            decision = pair[name]
            result[name]['accepted' if decision['accepted'] else 'rejected'] += 1
            if decision['accepted']:
                key = ('correct_known_accepted' if decision['label'] == truth['id'] else 'wrong_known_accepted') if known else 'unknown_false_accepted'
                result[name][key] += 1
        if pair['baseline']['accepted'] and not pair['v8']['accepted']:
            correct = known and pair['baseline']['label'] == truth['id']
            result['paired']['correct_acceptances_lost' if correct else 'wrong_acceptances_removed'] += 1
        else:
            result['paired']['unchanged'] += 1
    return result


def summarize(rows):
    signers = sorted({r['collection']['signer_id'] for r in rows})
    categories = {g: sorted({r['collection']['truth']['category_id'] for r in rows
                            if r['collection']['trial_group'] == g}) for g in GROUPS[1:]}
    cells = Counter((r['collection']['signer_id'], r['collection']['condition_id'], r['collection']['trial_group'],
                     r['collection']['truth'].get('id', r['collection']['truth'].get('category_id'))) for r in rows)
    balance_issues = []
    if len(signers) < 2:
        balance_issues.append({'code': 'fewer_than_two_declared_signers'})
    for group, values in categories.items():
        if len(values) != 3:
            balance_issues.append({'code': 'expected_three_categories', 'group': group, 'observed': len(values)})
    for signer in signers:
        for condition in ('A', 'B'):
            for group in GROUPS:
                for label in LABELS if group == 'known' else categories[group]:
                    expected = 3 if group == 'known' else 1 if group == 'tracking' else 2
                    actual = cells[(signer, condition, group, label)]
                    if actual != expected:
                        balance_issues.append({'code': 'cell_count_mismatch', 'signer_id': signer,
                                               'condition_id': condition, 'group': group, 'label': label,
                                               'expected': expected, 'observed': actual})
    strata = []
    for signer, condition, group, label in sorted(cells):
        subset = [r for r in rows if r['collection']['signer_id'] == signer
                  and r['collection']['condition_id'] == condition and r['collection']['trial_group'] == group
                  and r['collection']['truth'].get('id', r['collection']['truth'].get('category_id')) == label]
        strata.append({'signer_id': signer, 'condition_id': condition, 'group': group, 'label': label, **counts(subset)})
    total = counts(rows)
    groups = {g: counts([r for r in rows if r['collection']['trial_group'] == g]) for g in GROUPS}
    complete = not balance_issues and len(rows) == 48 * len(signers)
    no_negative_increase = all(groups[g]['v8']['unknown_false_accepted'] <= groups[g]['baseline']['unknown_false_accepted'] for g in GROUPS[1:])
    nonvacuous = total['baseline']['correct_known_accepted'] > 0
    criterion = (nonvacuous and total['paired']['wrong_acceptances_removed'] > 0
                 and total['paired']['correct_acceptances_lost'] == 0 and no_negative_increase)
    return {
        'totals': total, 'by_group': groups, 'strata': strata,
        'balance': {'complete': complete, 'minimum_target': 96, 'expected_for_declared_signers': 48 * len(signers),
                    'declared_signer_count': len(signers), 'category_ids': categories, 'issues': balance_issues},
        'pilot_criterion': {'assessed': complete, 'met': criterion if complete else None,
                            'some_known_correctly_accepted': nonvacuous,
                            'status': 'partial_or_unbalanced' if not complete else 'promising_pilot_only' if criterion else 'criterion_not_met',
                            'release_authorized': False},
    }


def audit_folder(folder: Path, reference_paths=None):
    """The CLI always uses the full historical registry; injection is for tests."""
    result = {'schema': 'signbridge-v8-pilot-summary-v1', 'valid': False, 'status': 'validation_failed',
              'input_report_count': 0, 'excluded_quick_test_count': 0, 'collection_trial_count': 0,
              'frozen_hashes': HASHES, 'issues': [], 'assurance': {
                  'media_bytes_and_report_consistency_only': True, 'identity_or_consent_verified': False,
                  'genuine_recording_verified': False, 'linguistic_truth_or_novel_categories_verified': False,
                  'video_decoded_by_summary': False, 'independent_final_evaluation': False}}
    def error(code, index=None):
        result['issues'].append({'code': code, **({'report_index': index} if index is not None else {})})
    try:
        root = Path(folder).resolve(strict=True)
        require(root.is_dir(), 'collection_not_directory')
        paths = sorted(root.glob('*.json'))
        require(0 < len(paths) <= 10000, 'report_count_invalid')
        source = (ROOT / 'web/src/v8-pilot-core.ts').read_text(encoding='utf-8')
        require(all(re.search(r'\b' + name + r":\s*'" + sha + "'", source) for name, sha in HASHES.items()),
                'browser_frozen_hashes_changed')
        blocked, references = reference_digests(GSL_REFERENCES if reference_paths is None else reference_paths)
        if reference_paths is None:
            # These newly encoded QA fixtures are also old GSL, not new people.
            qa_path = ROOT / 'reports/v8/pilot-qa/fixtures.json'
            qa = read_json(qa_path)
            require(qa.get('noNewHumanRecording') is True and isinstance(qa.get('clips'), list)
                    and len(qa['clips']) > 0, 'qa_registry_invalid')
            for clip in qa['clips']:
                require(isinstance(clip, dict) and matches(SHA, clip.get('sha256')), 'qa_registry_invalid')
                blocked.add(clip['sha256'])
            references.append(digest(qa_path))
        result['historical_registry_sha256'] = references
    except (OSError, ValueError, TypeError, UnicodeError, RuntimeError, RecursionError) as exc:
        error(str(exc) if isinstance(exc, ValueError) and str(exc) in ('collection_not_directory', 'report_count_invalid', 'browser_frozen_hashes_changed')
              else 'collection_or_reference_registry_unavailable')
        return result
    result['input_report_count'] = len(paths)
    ids, media_hashes, used_media, category_groups, rows = set(), set(), set(), {}, []
    for index, path in enumerate(paths):
        try:
            require(path.resolve().is_relative_to(root) and path.stat().st_size <= 1_000_000, 'report_path_or_size_invalid')
            try:
                row = read_json(path)
            except (ValueError, UnicodeError, RecursionError):
                raise ValueError('report_json_invalid') from None
            validate_report(row)
            clip_id, media = row['clip_id'], row['media']
            require(clip_id not in ids, 'duplicate_clip_id')
            ids.add(clip_id)
            require(path.stem == clip_id, 'report_filename_id_mismatch')
            require(media['sha256'] not in media_hashes, 'duplicate_media_sha256')
            media_hashes.add(media['sha256'])
            candidates = [root / (clip_id + suffix) for suffix in ('.mp4', '.webm') if (root / (clip_id + suffix)).is_file()]
            require(len(candidates) == 1, 'paired_media_missing_or_ambiguous')
            video = candidates[0].resolve(strict=True)
            require(video.is_relative_to(root), 'media_path_outside_collection')
            require(media['mime_type'].split(';')[0] == 'video/' + video.suffix[1:], 'media_extension_mime_mismatch')
            require(video.stat().st_size == media['byte_length'], 'media_byte_length_mismatch')
            require(digest(video) == media['sha256'], 'media_sha256_mismatch')
            used_media.add(video)
            if row['collection'] is None:
                result['excluded_quick_test_count'] += 1
                continue
            require(media['sha256'] not in blocked, 'historical_gsl_media_not_new_human_evidence')
            meta = row['collection']
            if meta['truth']['kind'] == 'unknown':
                category_id = meta['truth']['category_id']
                require(category_groups.setdefault(category_id, meta['trial_group']) == meta['trial_group'], 'unknown_category_crosses_groups')
            rows.append(row)
        except ValueError as exc:
            # Only validator-owned fixed codes are exposed, never paths or JSON text.
            code = str(exc)
            error(code if re.fullmatch(r'[a-z][a-z0-9_]{1,90}', code) else 'report_invalid', index)
        except (OSError, TypeError, KeyError, UnicodeError, RuntimeError, RecursionError, OverflowError):
            error('report_or_media_unreadable_or_invalid', index)
    try:
        present_media = {p.resolve(strict=True) for p in root.iterdir() if p.suffix.lower() in ('.mp4', '.webm') and p.is_file()}
        if present_media - used_media:
            error('unpaired_media_files_present')
    except OSError:
        error('media_inventory_unreadable')
    result['collection_trial_count'] = len(rows)
    if result['issues']:
        return result
    result.update(valid=True, **summarize(rows))
    result['status'] = 'complete_balanced_pilot' if result['balance']['complete'] else 'partial_or_unbalanced_pilot' if rows else 'quick_tests_only'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection_folder', type=Path)
    args = parser.parse_args()
    report = audit_folder(args.collection_folder)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    return 0 if report['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
