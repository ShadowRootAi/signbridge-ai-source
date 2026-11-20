"""Compare frozen OpenHands acceptance gates on cached native and RGB landmarks."""

import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from calibrate_openhands_gsl import read_metadata
from evaluate_openhands_verifier import (
    EMBEDDING_WIDTH, MODEL_SHA256, cached_rgb_poses, counts, evidence, fit_signal,
    infer, infer_native, make_session, sha256, train_centroids, write_json,
)


FOCUS = ('ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ', 'ΕΤΟΙΜΟ', 'ΦΥΣΙΚΑ', 'ΟΚ', 'ΝΑΙ', 'ΟΧΙ')
MIN_CORRECT_CANDIDATES = 6
MIN_WRONG_CANDIDATES = 3
REGULARIZATION_C = 0.2
FEATURE_NAMES = ('confidenceLogOdds', 'winningLogit', 'topTwoLogitGap', 'prototypeSimilarity', 'prototypeMargin')


def cached_scores(cache, stem, nlabels):
    poses = cache / f'{stem}.npz'
    scores = cache / f'{stem}-verifier-scores.npz'
    pose_hash = sha256(poses)
    with np.load(poses, allow_pickle=False) as dataset:
        ids, labels = dataset['ids'], dataset['glosses']
    with np.load(scores, allow_pickle=False) as dataset:
        if dataset['sourceHash'].item() != pose_hash or dataset['modelHash'].item() != MODEL_SHA256:
            raise ValueError(f'Stale cache: {scores.name}')
        logits, vectors = dataset['logits'], dataset['vectors']
    if (len(set(ids)) != len(ids) or len(labels) != len(ids) or logits.shape != (len(ids), nlabels)
            or vectors.shape != (len(ids), EMBEDDING_WIDTH) or not np.isfinite(logits).all()
            or not np.isfinite(vectors).all()):
        raise ValueError(f'Malformed cache: {scores.name}')
    return ids, labels, logits, vectors, {poses.name: pose_hash, scores.name: sha256(scores)}


def inputs(logits, vectors, bias, weights, centers, center_counts):
    winners, confidence, _, signals = evidence(logits, vectors, bias, weights, centers, center_counts)
    probability = np.clip(confidence, 1e-15, 1. - 1e-15)
    ordered = np.partition(logits, -2, axis=1)[:, -2:]
    features = np.column_stack((
        np.log(probability) - np.log1p(-probability), signals['winningLogit'], ordered[:, 1] - ordered[:, 0],
        np.maximum(signals['prototypeSimilarity'], -1.), np.maximum(signals['prototypeMargin'], -2.),
    ))
    return winners, confidence, signals['winningLogit'], features


def global_accept(winner, confidence, logit, roster, threshold):
    return (roster[winner] & (confidence >= threshold['minimumConfidence'])
            & (logit >= threshold['minimumEvidence']))


def per_winner_floors(reference, winners, confidence, logit, roster, threshold):
    baseline = global_accept(winners, confidence, logit, roster, threshold)
    correct = reference == winners
    evidence_pass = logit >= threshold['minimumEvidence']
    floors = {}
    for label in np.flatnonzero(roster):
        candidates = winners == label
        right, wrong = candidates & correct, candidates & ~correct
        if right.sum() < MIN_CORRECT_CANDIDATES or wrong.sum() < MIN_WRONG_CANDIDATES:
            continue
        additional_wrong = confidence[wrong & evidence_pass & ~baseline]
        lowest_safe = float(np.nextafter(additional_wrong.max(), np.inf)) if len(additional_wrong) else 0.
        recoverable = confidence[right & evidence_pass & ~baseline & (confidence >= lowest_safe)]
        if len(recoverable):
            floors[int(label)] = float(recoverable.min())
    return floors


def class_accept(winner, confidence, logit, roster, threshold, floors):
    minimum = np.full(len(roster), threshold['minimumConfidence'])
    for label, floor in floors.items():
        minimum[label] = floor
    return roster[winner] & (confidence >= minimum[winner]) & (logit >= threshold['minimumEvidence'])


def fit_logistic(features, winners, correct, label_count, use_offsets):
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(C=REGULARIZATION_C, max_iter=500, solver='lbfgs')

    def transform(features, winners):
        standardized = scaler.transform(features)
        if use_offsets:
            offsets = np.eye(label_count)[winners]
            return np.column_stack((standardized, offsets))
        return standardized

    classifier.fit(transform(features, winners), correct)

    def predict(features, winners):
        return classifier.decision_function(transform(features, winners))

    return predict


def fit_correctness(features, winners, correct, groups, label_count, use_offsets):
    oof_scores = np.empty(len(features))
    for training, scoring in GroupKFold(n_splits=5).split(features, correct, groups):
        predict = fit_logistic(features[training], winners[training], correct[training], label_count, use_offsets)
        oof_scores[scoring] = predict(features[scoring], winners[scoring])
    return oof_scores, fit_logistic(features, winners, correct, label_count, use_offsets)


def budget_floor(scores, wrong, outside, wrong_budget, outside_budget):
    floor = float(np.nextafter(scores.min(), -np.inf))
    for negative, budget in ((wrong, wrong_budget), (outside, outside_budget)):
        forbidden = np.sort(scores[negative])[::-1]
        if len(forbidden) > budget:
            floor = max(floor, float(np.nextafter(forbidden[budget], np.inf)))
    return floor


def summary(accepted, ref, winners, roster, labels):
    report = counts(accepted, ref, winners, roster)
    known = roster[ref]
    correct = accepted & (ref == winners)
    values = []
    for label in np.flatnonzero(roster):
        is_label = ref == label
        if is_label.any():
            values.append((correct & is_label).sum() / is_label.sum())
    report['rawTop1OnKnown'] = int((known & (ref == winners)).sum())
    report['precisionAmongAccepted'] = report['correct'] / int(accepted.sum()) if accepted.any() else None
    report['macroCorrectAcceptance'] = float(np.mean(values)) if values else None
    report['focus'] = {}
    for label in FOCUS:
        index = labels.index(label)
        is_label = ref == index
        is_winner = winners == index
        report['focus'][label] = {
            'clips': int(is_label.sum()), 'rawTop1': int((is_label & is_winner).sum()),
            'correct': int((is_label & correct).sum()),
            'acceptedWrongAs': dict(Counter(labels[i] for i in ref[accepted & is_winner & ~is_label])),
        }
    return report


def score_shift(ref, winners, features, roster, target):
    eligible = roster[winners]
    correct = winners == ref
    report = {'eligibleClips': int(eligible.sum()), 'scores': {}}
    for feature, name in enumerate(FEATURE_NAMES):
        values = features[:, feature]
        slices = {'correct': eligible & correct, 'wrong': eligible & ~correct,
                  'entaxeiCorrect': (winners == target) & correct, 'entaxeiWrong': (winners == target) & ~correct}
        report['scores'][name] = {
            'rocAucCorrectness': float(roc_auc_score(correct[eligible], values[eligible]))
            if np.unique(correct[eligible]).size == 2 else None,
            'quantiles': {key: np.percentile(values[mask], [5, 25, 50, 75, 95]).tolist() if mask.any() else None
                          for key, mask in slices.items()},
            'cohortSizes': {key: int(mask.sum()) for key, mask in slices.items()},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('research/cache/openhands-gsl'))
    parser.add_argument('--reports', type=Path, default=Path('reports/pretrained-gsl'))
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--official-test', action='store_true', help='Also replay already frozen gates on the native official test cache')
    args = parser.parse_args()
    cache = args.cache
    labels, supported, metadata = read_metadata(cache / 'gsl_metadata.zip')
    deployed_path = args.reports / 'stgcn-gate-expanded-candidate.json'
    deployed = json.loads(deployed_path.read_text(encoding='utf-8'))
    if deployed['supportedLabels'] != supported or deployed['runtimeModelSha256'] != MODEL_SHA256:
        raise ValueError('Deployed policy differs from the fixed model or roster')
    ids, glosses, logits, vectors, val_hashes = cached_scores(cache, 'official-val-poses', len(labels))
    train_ids, train_glosses, _, train_vectors, train_hashes = cached_scores(
        cache, 'official-train-prototypes-poses', len(labels))
    if dict(zip(ids, glosses, strict=True)) != metadata:
        raise ValueError('Validation IDs do not match the published native split')
    label_index = {label: index for index, label in enumerate(labels)}
    roster = np.isin(labels, supported)
    references = np.asarray([label_index[label] for label in glosses])
    rep1, rep2 = (np.char.find(ids, f'_signer3_rep{rep}_glosses/') >= 0 for rep in (1, 2))
    if not np.all(rep1 ^ rep2):
        raise ValueError('Expected only signer 3 repetitions 1 and 2')
    centers, center_counts = train_centroids(train_ids, [label_index[label] for label in train_glosses], train_vectors, ids)
    session, bias, weights = make_session(cache / 'stgcn.onnx', None)
    winners, confidence, logit, features = inputs(logits, vectors, bias, weights, centers, center_counts)
    rgb = cached_rgb_poses(args.data, label_index)
    if not rgb:
        raise ValueError('No locally cached RGB landmark replay cohort')
    rgb_labels, rgb_poses = zip(*rgb.values())
    rgb_ref = np.asarray([label_index[label] for label in rgb_labels])
    rgb_winner, rgb_conf, rgb_logit, rgb_features = inputs(
        *infer(session, rgb_poses), bias, weights, centers, center_counts)

    def assess(native_accept, rgb_accept):
        return {
            'calibration': summary(native_accept[rep1], references[rep1], winners[rep1], roster, labels),
            'diagnosticRep2': summary(native_accept[rep2], references[rep2], winners[rep2], roster, labels),
            'diagnosticCachedRgb': summary(rgb_accept, rgb_ref, rgb_winner, roster, labels),
        }

    deployed_accept = global_accept(winners, confidence, logit, roster, deployed['decisionGate'])
    deployed_rgb = global_accept(rgb_winner, rgb_conf, rgb_logit, roster, deployed['decisionGate'])
    baseline = assess(deployed_accept, deployed_rgb)
    baseline_counts = baseline['calibration']
    budgets = {'deployedCalibration': (baseline_counts['wrong'], baseline_counts['outsideFalseAccepts']),
               'zeroCalibrationErrors': (0, 0)}
    eligible = roster[winners]
    calibration = rep1 & eligible
    is_correct = winners[calibration] == references[calibration]
    groups = np.asarray([identity.split('_signer', 1)[0] for identity in ids[calibration]])
    known_wrong = ~is_correct & roster[references[calibration]]
    outside = ~roster[references[calibration]]
    logistic = {name: fit_correctness(features[calibration], winners[calibration], is_correct, groups, len(labels), offsets)
                for name, offsets in (('logisticEvidence', False), ('logisticEvidenceWithOffsets', True))}
    candidates = {}
    for budget_name, (max_wrong, max_outside) in budgets.items():
        _, threshold = fit_signal(confidence[rep1], logit[rep1], eligible[rep1], references[rep1], winners[rep1], roster,
                                  max_wrong=max_wrong, max_outside=max_outside)
        floors = per_winner_floors(references[rep1], winners[rep1], confidence[rep1], logit[rep1], roster, threshold)
        contenders = {
            'globalWinningLogit': {**assess(global_accept(winners, confidence, logit, roster, threshold),
                                           global_accept(rgb_winner, rgb_conf, rgb_logit, roster, threshold)),
                                   'thresholds': threshold},
            'perWinnerConfidenceFloor': {**assess(class_accept(winners, confidence, logit, roster, threshold, floors),
                                                 class_accept(rgb_winner, rgb_conf, rgb_logit, roster, threshold, floors)),
                                         'thresholds': threshold,
                                         'lowerConfidenceByWinner': {labels[i]: floor for i, floor in floors.items()}},
        }
        for method, (oof_scores, predict) in logistic.items():
            full_scores = predict(features, winners)
            rgb_scores = predict(rgb_features, rgb_winner)
            floor = max(budget_floor(oof_scores, known_wrong, outside, max_wrong, max_outside),
                        budget_floor(full_scores[calibration], known_wrong, outside, max_wrong, max_outside))
            oof_accept = np.zeros(len(rep1), dtype=bool)
            oof_accept[calibration] = oof_scores >= floor
            contenders[method] = {**assess(eligible & (full_scores >= floor), roster[rgb_winner] & (rgb_scores >= floor)),
                                  'minimumCorrectnessLogit': floor,
                                  'groupedOofCalibration': summary(oof_accept[rep1], references[rep1], winners[rep1], roster, labels)}
        for results in contenders.values():
            metrics = results['calibration']
            if metrics['wrong'] > max_wrong or metrics['outsideFalseAccepts'] > max_outside:
                raise RuntimeError('A calibration candidate exceeded the stated budget')
        candidates[budget_name] = {'wrongAtMost': max_wrong, 'outsideFalseAcceptsAtMost': max_outside,
                                   'methods': contenders}

    report = {
        'scope': 'Experimental. Repetition 2 is a repeated diagnostic, not an independent validation. Native repetitions 1 and 2 '
                 'are the same signer with largely repeated scripts, and OpenHands authors used this signer in checkpoint selection. '
                 'Cached RGB comes from the same source corpus and may overlap upstream training; it contains no real webcam '
                 'trials or verified no-sign controls. Outside-roster glosses are known to the 310-output pretrained model.',
        'modelSha256': MODEL_SHA256, 'deployedPolicySha256': sha256(deployed_path),
        'cacheChecksums': {**val_hashes, **train_hashes},
        'development': {'nativeCalibrationClips': int(rep1.sum()), 'nativeDiagnosticClips': int(rep2.sum()),
                        'prototypeTrainingSigners': [1, 2, 4], 'prototypeTrainingClips': len(train_ids),
                        'prototypeTrainingLabels': int(np.count_nonzero(center_counts)),
                        'cachedRgbReplayIds': list(rgb), 'logisticFeatures': FEATURE_NAMES,
                        'logisticRegularizationC': REGULARIZATION_C, 'logisticCalibrationScenarioGroups': sorted(set(groups)),
                        'perWinnerFloorMinimumCorrectCandidates': MIN_CORRECT_CANDIDATES,
                        'perWinnerFloorMinimumWrongCandidates': MIN_WRONG_CANDIDATES,
                        'focusCalibrationCandidates': {
                            label: {'correct': int((calibration & (winners == label_index[label]) & (winners == references)).sum()),
                                    'wrong': int((calibration & (winners == label_index[label]) & (winners != references)).sum())}
                            for label in FOCUS}},
        'deployed': {'thresholds': deployed['decisionGate'], **baseline}, 'candidateBudgets': candidates,
    }
    target = label_index['ΕΝΤΑΞΕΙ']
    report['scoreShift'] = {
        'scope': 'Score order and 5th, 25th, 50th, 75th and 95th percentiles among roster winners. '
                 'Cached RGB has a small different label mix and is descriptive only.',
        'calibration': score_shift(references[rep1], winners[rep1], features[rep1], roster, target),
        'diagnosticRep2': score_shift(references[rep2], winners[rep2], features[rep2], roster, target),
        'diagnosticCachedRgb': score_shift(rgb_ref, rgb_winner, rgb_features, roster, target),
    }
    if args.official_test:
        # Open the additional data only after the calibrators and thresholds are frozen.
        if not (cache / 'official-test-poses-verifier-scores.npz').is_file():
            infer_native(session, cache / 'official-test-poses.npz', cache)
        test_ids, test_glosses, test_logits, test_vectors, test_hashes = cached_scores(cache, 'official-test-poses', len(labels))
        with ZipFile(cache / 'gsl_metadata.zip') as archive:
            lines = archive.read('GSL_split/GSL_isolated/test_greek_iso.csv').decode('utf-8').splitlines()
        expected = dict(line.split('|', 1) for line in lines if line.strip())
        if dict(zip(test_ids, test_glosses, strict=True)) != expected or set(test_ids) & (set(ids) | set(train_ids)):
            raise ValueError('Official test cache differs from the published split or overlaps fitting data')
        test_ref = np.asarray([label_index[label] for label in test_glosses])
        test_winner, test_conf, test_logit, test_features = inputs(test_logits, test_vectors, bias, weights, centers, center_counts)
        test_eligible = roster[test_winner]
        report['diagnosticOfficialTestScope'] = 'Official test native landmarks, signer 3 repetitions 3–5. '
        report['diagnosticOfficialTestScope'] += 'The checkpoint authors used this split for model selection. Frozen local gates only.'
        report['cacheChecksums'].update(test_hashes)
        report['scoreShift']['diagnosticOfficialTest'] = score_shift(test_ref, test_winner, test_features, roster, target)
        test_accept = global_accept(test_winner, test_conf, test_logit, roster, deployed['decisionGate'])
        report['deployed']['diagnosticOfficialTest'] = summary(test_accept, test_ref, test_winner, roster, labels)
        for budget_results in candidates.values():
            for method, results in budget_results['methods'].items():
                if method in logistic:
                    predict = logistic[method][1]
                    test_accept = test_eligible & (predict(test_features, test_winner) >= results['minimumCorrectnessLogit'])
                else:
                    floors = {label_index[label]: floor for label, floor in results.get('lowerConfidenceByWinner', {}).items()}
                    test_accept = class_accept(test_winner, test_conf, test_logit, roster, results['thresholds'], floors)
                results['diagnosticOfficialTest'] = summary(test_accept, test_ref, test_winner, roster, labels)
    path = args.reports / 'stgcn-recalibration-experiment.json'
    write_json(path, report)
    print(path)
    for name, results in [('deployed', report['deployed']), *(
        (f'{budget}/{method}', results) for budget, budget_results in candidates.items()
        for method, results in budget_results['methods'].items()
    )]:
        print(name, [(split, item['correct'], item['wrong'], item['outsideFalseAccepts'], item['focus']['ΕΝΤΑΞΕΙ']['correct'])
                     for split, item in results.items() if isinstance(item, dict) and 'focus' in item])


if __name__ == '__main__':
    main()
