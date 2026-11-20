"""Train a small cross-signer supervised metric, with an inner signer holdout.

Only replay the previously exposed signer3 diagnostic after a signer4 win.
This is an isolated-pose research experiment, not a production recognizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_isolated_exemplar_retrieval import (
    annotations, open_source, descriptors, motion, align_scores, probabilities,
    mix, scores, select_gate, sha256, write_json,
)
from evaluate_isolated_balanced_prototypes import load_sources, summary

CACHE = Path('research/cache/openhands-gsl')
SOURCE = Path('reports/isolated-exemplar-retrieval-expanded')
CLASSIFIER = Path('reports/isolated-large-vocabulary')
REPORT = Path('reports/isolated-cross-signer-metric')
ARTIFACT = Path('artifacts/isolated-cross-signer-metric')
SEED = 2301
CHECKPOINTS = (100, 200, 400, 600)
WEIGHTS = (0., .25, .5, 1., 2.)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalizer(sequences):
    return {'location': sequences.mean((0, 1)), 'scale': sequences.std((0, 1)).clip(min=.15),
            'motion_scale': motion(sequences).std((0, 1)).clip(min=.12)}


def convert(sequences, norm):
    return descriptors(sequences, norm['location'], norm['scale'], norm['motion_scale'])


class MetricNetwork(nn.Module):
    def __init__(self, input_size, labels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Dropout(.1), nn.Linear(input_size, 128), nn.LayerNorm(128),
                                     nn.GELU(), nn.Linear(128, 64))
        self.proxies = nn.Parameter(torch.randn(labels, 64) * .05)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=1)
        return z @ F.normalize(self.proxies, dim=1).T, z


def contrastive(z, target, signer):
    positive = (target[:, None] == target[None, :]) & (signer[:, None] != signer[None, :])
    eligible = (target[:, None] != target[None, :]) | positive
    similarity = (z @ z.T / .1).masked_fill(~eligible, -1e4)
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return z.sum() * 0.
    terms = -(F.log_softmax(similarity, dim=1) * positive).sum(1) / counts.clamp_min(1)
    return terms[valid].mean()


def batch_groups(target, signers):
    groups = {}
    for label in np.unique(target):
        groups[int(label)] = [torch.from_numpy(np.flatnonzero((target == label) & (signers == signer)))
                              for signer in np.unique(signers[target == label])]
    return groups


def draw(groups, generator):
    labels = list(groups)
    chosen = torch.randperm(len(labels), generator=generator)[:min(32, len(labels))]
    result = []
    for item in chosen.tolist():
        for indices in groups[labels[item]]:
            result.append(int(indices[torch.randint(len(indices), (1,), generator=generator)]))
    return torch.tensor(result)


@torch.inference_mode()
def infer(model, x):
    model.eval()
    return torch.cat([model(chunk)[0] for chunk in torch.as_tensor(x).split(256)]).numpy()


def raw_metrics(values, target, available):
    masked = values.copy()
    masked[:, ~available] = -1e4
    known = available[target]
    correct = masked.argmax(1) == target
    present = np.unique(target[known])
    return {'clips': int(known.sum()), 'excluded_labels_without_inner_training': int((~known).sum()),
            'correct': int(correct[known].sum()),
            'macro': float(np.mean([correct[target == label].mean() for label in present]))}


def train(x, target, signers, label_count, steps, monitor=None):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = MetricNetwork(x.shape[1], label_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(CHECKPOINTS), eta_min=.0001)
    source, labels, actors = torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(signers)
    groups = batch_groups(target, signers)
    available = np.zeros(label_count, dtype=bool)
    available[np.unique(target)] = True
    available_tensor = torch.from_numpy(available)
    history, losses = [], []
    for step in range(1, steps + 1):
        model.train()
        selected = draw(groups, generator)
        cosine, embedded = model(source[selected])
        logits = 16. * (cosine - .1 * F.one_hot(labels[selected], label_count))
        logits = logits.masked_fill(~available_tensor, -1e4)
        loss = F.cross_entropy(logits, labels[selected]) + .25 * contrastive(embedded, labels[selected], actors[selected])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite supervised metric training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if step in CHECKPOINTS or step == steps:
            row = {'step': step, 'mean_loss': float(np.mean(losses))}
            losses.clear()
            if monitor is not None:
                row.update(raw_metrics(infer(model, monitor[0]), monitor[1], available))
            history.append(row)
            print(json.dumps({'training': 'inner' if monitor is not None else 'final', **row}), flush=True)
    return model, history


def source_hashes():
    paths = [Path(__file__), Path('ml/evaluate_isolated_exemplar_retrieval.py'),
             Path('ml/evaluate_isolated_balanced_prototypes.py'), Path('ml/evaluate_synthetic_ctc_transfer.py'),
             SOURCE / 'freeze.json', SOURCE / 'fusion-expanded-freeze.json',
             CLASSIFIER / 'expanded-freeze.json', CLASSIFIER / 'expanded-dev-scores.npz',
             CACHE / 'official-train-expanded-poses.npz', CACHE / 'official-train-prototypes-poses.npz',
             CACHE / 'gsl_metadata.zip']
    return {path.as_posix(): sha256(path) for path in paths}


def fit():
    if (REPORT / 'protocol.json').exists():
        raise ValueError('This experiment is already locked; use a separately reviewed new run')
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    original, retrieval_freeze, bank = load_sources()
    protocol = {'seed': SEED, 'inner_fit_signers': [1, 2], 'inner_selection_signer': [5],
                'final_fit_signers': [1, 2, 5], 'final_calibration_signer': [4],
                'checkpoints': CHECKPOINTS, 'weights': WEIGHTS,
                'architecture': '6048 ->128 LayerNorm GELU ->64 normalized; normalized class proxies',
                'loss': 'Class-balanced cosine cross entropy (scale16, margin.1) +.25 cross-signer supervised contrastive (temperature.1)',
                'inner_selection': 'Maximum macro recall, then raw correct, then fewer steps, only on signer5',
                'final_selection': 'Frozen metric training duration; compare5 mixture weights once on signer4',
                'held_replay_gate': 'Strictly more development top1 than existing fusion, no macro decline, at least as many correct accepts under same1% wrong-known budget',
                'upstream_pretrained_model_used': False, 'sources': source_hashes()}
    write_json(REPORT / 'protocol.json', protocol)
    official = annotations(CACHE / 'gsl_metadata.zip')
    seq, ids, refs, signers = open_source(CACHE / 'official-train-expanded-poses.npz', official['train'], role='expanded')
    mapping = {label: i for i, label in enumerate(bank['labels'])}
    target = np.array([mapping[label] for label in refs], dtype=np.int64)
    inside = np.isin(signers, [1, 2])
    inner_norm = normalizer(seq[inside])
    inner_x = convert(seq, inner_norm)
    _, inner_history = train(inner_x[inside], target[inside], signers[inside], len(mapping), max(CHECKPOINTS),
                             (inner_x[~inside], target[~inside]))
    duration = max(inner_history, key=lambda row: (row['macro'], row['correct'], -row['step']))['step']
    del inner_x
    norm = normalizer(seq)
    x = convert(seq, norm)
    model, final_history = train(x, target, signers, len(mapping), duration)
    del x, seq
    # Signer4 is first accessed after duration selection and final weight training.
    dev, dev_ids, dev_refs, dev_signers = open_source(CACHE / 'official-train-prototypes-poses.npz', official['train'], role='train')
    mask = dev_signers == 4
    dev, dev_ids, dev_refs = dev[mask], dev_ids[mask], dev_refs[mask]
    if set(ids) & set(dev_ids) or set(ids) != set(bank['ids']):
        raise ValueError('Training/calibration identity overlap or source mismatch')
    local_logits = align_scores(CLASSIFIER / 'expanded-dev-scores.npz', dev_ids, dev_refs, bank['labels'])
    retrieval = scores(bank, dev)[0][retrieval_freeze['variants']['all_verified_labels']['selected']]
    anchor = mix(local_logits, retrieval, original['weight'])
    anchor_summary = summary(probabilities(anchor), bank['labels'], dev_refs, original['weighted_policy'])[0]
    expected = original['development']['weighted']
    for key in ('top1_correct', 'accepted_correct', 'accepted_wrong_known'):
        if anchor_summary[key] != expected[key]:
            raise ValueError('Existing fusion development reference did not reproduce')
    metric = infer(model, convert(dev, norm))
    candidates = []
    for weight in WEIGHTS:
        values = probabilities(mix(anchor, metric, weight))
        policy = select_gate(values, bank['labels'], dev_refs)
        measured = summary(values, bank['labels'], dev_refs, policy)[0]
        candidates.append({'weight': weight, 'policy': policy, 'summary': measured})
    selected = max(candidates, key=lambda row: (row['summary']['top1_correct'], row['summary']['macro_top1'], -row['weight']))
    result = selected['summary']
    eligible = (result['top1_correct'] > expected['top1_correct'] and result['macro_top1'] >= expected['macro_top1']
                and result['accepted_correct'] >= expected['accepted_correct'])
    checkpoint = ARTIFACT / 'model.pt'
    torch.save({'state_dict': model.state_dict(), 'labels': bank['labels'].tolist(),
                'normalizer': {key: torch.from_numpy(value) for key, value in norm.items()},
                'input_size': len(bank['location']) * 16 * 2, 'steps': duration}, checkpoint)
    np.savez_compressed(REPORT / 'development-scores.npz', ids=dev_ids, glosses=dev_refs,
                        labels=bank['labels'], metric_cosines=metric, anchor_logits=anchor)
    write_json(REPORT / 'freeze.json', {
        'protocol_sha256': sha256(REPORT / 'protocol.json'), 'checkpoint_sha256': sha256(checkpoint),
        'source_hashes': protocol['sources'], 'inner_history': inner_history, 'selected_steps': duration,
        'final_history': final_history, 'development_grid': candidates, 'selected': selected,
        'existing_fusion_development': anchor_summary, 'eligible_for_retrospective_replay': eligible,
        'training_ids': ids.tolist(), 'calibration_ids': dev_ids.tolist(),
        'elapsed_seconds': time.perf_counter() - started,
        'limits': ['Signer4 and3 were exposed in earlier research; this is adaptive development, not untouched final validation.',
                   'No true unknown signs or no-sign gestures in this all-label cohort.',
                   'Native pre-segmented landmarks; no webcam or continuous-recognition claims.']})
    print(json.dumps({'phase': 'frozen', 'steps': duration, 'eligible': eligible, 'selected': selected}, ensure_ascii=False), flush=True)


def evaluate():
    target = REPORT / 'benchmark.json'
    if target.exists():
        raise ValueError('Retrospective replay already exists')
    frozen = read(REPORT / 'freeze.json')
    if not frozen['eligible_for_retrospective_replay']:
        raise ValueError('Development gate failed: held poses will not be opened')
    checkpoint = ARTIFACT / 'model.pt'
    if (source_hashes() != frozen['source_hashes'] or sha256(checkpoint) != frozen['checkpoint_sha256']
            or sha256(REPORT / 'protocol.json') != frozen['protocol_sha256']):
        raise ValueError('Frozen experiment changed')
    original, retrieval_freeze, bank = load_sources()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['labels'] != bank['labels'].tolist():
        raise ValueError('Checkpoint class order changed')
    model = MetricNetwork(saved['input_size'], len(saved['labels']))
    model.load_state_dict(saved['state_dict'])
    norm = {key: value.numpy() for key, value in saved['normalizer'].items()}
    official = annotations(CACHE / 'gsl_metadata.zip')
    seq, ids, refs, signers = open_source(CACHE / 'official-test-poses.npz', official['test'], role='test')
    if set(ids) & (set(frozen['training_ids']) | set(frozen['calibration_ids'])):
        raise ValueError('Source identity leakage')
    logits = align_scores(CLASSIFIER / 'expanded-scores.npz', ids, refs, bank['labels'])
    retrieval = scores(bank, seq)[0][retrieval_freeze['variants']['all_verified_labels']['selected']]
    anchor = mix(logits, retrieval, original['weight'])
    metric = infer(model, convert(seq, norm))
    selected = frozen['selected']
    values = probabilities(mix(anchor, metric, selected['weight']))
    measured, ranks, accepted = summary(values, bank['labels'], refs, selected['policy'])
    baseline = summary(probabilities(anchor), bank['labels'], refs, original['weighted_policy'])[0]
    expected = read(SOURCE / 'fusion-expanded-benchmark.json')['held']['weighted']
    for key in ('top1_correct', 'accepted_correct', 'accepted_wrong_known'):
        if baseline[key] != expected[key]:
            raise ValueError('Historical held baseline did not reproduce')
    rows = [{'id': str(i), 'reference': str(r), 'prediction': str(bank['labels'][ranks[n, 0]]), 'accepted': bool(accepted[n])}
            for n, (i, r) in enumerate(zip(ids, refs, strict=True))]
    write_json(target, {'scope': 'Retrospective exposed signer3; not new independent final evidence',
                       'freeze_sha256': sha256(REPORT / 'freeze.json'), 'new_metric_fusion': measured,
                       'existing_fusion': baseline, 'rows': rows, 'limits': frozen['limits']})
    print(json.dumps({'phase': 'evaluated', 'new_metric_fusion': measured, 'existing_fusion': baseline}), flush=True)


def self_test():
    torch.manual_seed(1)
    y = torch.tensor([0, 0, 1, 1])
    s = torch.tensor([1, 2, 1, 2])
    good = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    bad = good.detach()[torch.tensor([0, 2, 1, 3])]
    assert contrastive(good, y, s) < contrastive(bad, y, s)
    contrastive(good, y, s).backward()
    assert torch.isfinite(good.grad).all()
    assert contrastive(good, y, torch.ones(4)).item() == 0.
    groups = batch_groups(y.numpy(), s.numpy())
    sampled = draw(groups, torch.Generator().manual_seed(1))
    assert len(sampled) == 4 and len(set(sampled.tolist())) == 4
    model = MetricNetwork(8, 2)
    cosine, embedded = model(torch.randn(3, 8))
    torch.testing.assert_close(embedded.norm(dim=1), torch.ones(3))
    assert cosine.shape == (3, 2) and cosine.abs().max() <= 1.00001
    print('Five contrastive/sampling/normalization invariants passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('self-test', 'fit', 'evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {'self-test': self_test, 'fit': fit, 'evaluate': evaluate}[args.phase]()
