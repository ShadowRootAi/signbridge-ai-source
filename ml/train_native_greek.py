"""Learn Greek subtitles from complete, timed Greek Sign Language recordings."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from continuous_selective import tracked_hand_motion
from preprocess import FEATURE_COUNT, FEATURE_SCHEMA
from train_continuous import edit_counts
from train_greek_translation_pilot import ALPHABET, elementary_sets, greek, sha
from train_grounded_greek import load_visual, write


BASE = Path('data/sartinas-dialogue-fit')
REPORT = Path('reports/native-greek')
ARTIFACTS = Path('artifacts/native-greek')
ENCODER_REPORT = Path('reports/grounded-greek/appointment-unseen-freeze.json')
FPS, SEED, WIDTH = 10, 491, 96
MODEL_VERSION = 2
WORD_START = '▁'
HELD_SIGNERS = {'n1', 'n2', 'n6', 'n7', 'n8'}
FOCUS = ('ραντεβού', 'μήνα', 'φορά το μήνα')


@dataclass
class Stream:
    id: str
    signer: str
    scenario: str
    split: str
    frames: torch.Tensor
    fps: float
    turns: tuple[tuple[float, float, str], ...]
    glosses: tuple[tuple[float, float, str], ...] = ()
    source_key: str = ''


def load_streams(folder, roles):
    manifest_path, feature_path = folder / 'manifest.json', folder / 'features.jsonl'
    manifest = json.loads(manifest_path.read_bytes())
    selected = [row for row in manifest['dialogues'] if row['split'] in roles]
    dialogues = {row['id']: row for row in selected}
    annotations = {row['id']: row for row in manifest['samples']}
    if len(annotations) != len(manifest['samples']) or len(dialogues) != len(selected) or not dialogues:
        raise ValueError(f'Missing or duplicate annotation IDs in {folder}')
    video_path = folder / 'video-manifest.json'
    videos = {}
    if video_path.exists():
        video = json.loads(video_path.read_bytes())
        if video['source_manifest_sha256'] != sha(manifest_path):
            raise ValueError(f'Video manifest differs from annotations in {folder}')
        videos = {row['id']: row for row in video['samples']}
    streams = []
    with feature_path.open(encoding='utf-8') as records:
        for line in records:
            row = json.loads(line)
            if row['id'] not in dialogues:
                continue
            dialogue = dialogues[row['id']]
            frames = torch.tensor(row['features'], dtype=torch.float32)
            fps = float(row['fps'])
            if (fps not in (10, 15) or frames.ndim != 2 or frames.shape[1] != FEATURE_COUNT
                    or len(frames) < 2 or not frames.isfinite().all() or row['feature_schema'] != FEATURE_SCHEMA):
                raise ValueError(f'Invalid feature stream: {row["id"]}')
            if row['id'] in videos:
                source = videos[row['id']]
                key = sha256(json.dumps([source['fps'], source['frame_paths']], ensure_ascii=False,
                                        separators=(',', ':')).encode()).hexdigest()
                if (row['source_key'] != key or not math.isclose(fps, source['fps'] / row['stride'])
                        or len(frames) != len(source['frame_paths'][::row['stride']])):
                    raise ValueError(f'Unmatched RGB source and landmarks: {row["id"]}')
            turns = []
            for key in dialogue['turn_ids']:
                item = annotations[key]
                start, end, text = item['start_ms'] / 1000, item['end_ms'] / 1000, greek(item['text'])
                if (not text or not 0 <= start < end <= len(frames) / fps + .1
                        or (turns and start < turns[-1][1]) or item['split'] != dialogue['split']
                        or str(item['signer']) != str(dialogue['signer'])):
                    raise ValueError(f'Invalid timed Greek annotation: {key}')
                turns.append((start, end, text))
            glosses = tuple((item['start_ms'] / 1000, item['end_ms'] / 1000,
                             item.get('label', item.get('gloss'))) for item in dialogue.get('gloss_intervals', ()))
            if not turns or any(not label or not 0 <= start < end <= len(frames) / fps + .1
                                for start, end, label in glosses):
                raise ValueError(f'Missing Greek or invalid gloss timing: {row["id"]}')
            streams.append(Stream(row['id'], str(dialogue['signer']), dialogue['scenario'], dialogue['split'],
                                  frames, fps, tuple(turns), glosses, row['source_key']))
    if len(streams) != len(dialogues) or {row.id for row in streams} != set(dialogues):
        raise ValueError(f'Missing or duplicate landmark recordings in {folder}')
    if len({row.source_key for row in streams}) != len(streams):
        raise ValueError(f'Repeated video source in {folder}')
    return streams, {'manifest_sha256': sha(manifest_path), 'features_sha256': sha(feature_path),
                     'video_manifest_sha256': sha(video_path) if video_path.exists() else None}


def check_fit_split(streams, *, include_n2=False):
    training = [row for row in streams if row.split == 'train']
    dev = [row for row in streams if row.split == 'val']
    if (not training or not dev or {row.signer for row in training} & {row.signer for row in dev}
            or any(is_reserved(row.signer, row.scenario, include_n2=include_n2 and row.split == 'train')
                   for row in streams)):
        raise ValueError('Native fitting needs disjoint signers and excludes the S6 and S10 holdouts')
    return training, dev


def is_reserved(signer, scenario, *, include_n2=False):
    name = str(signer).removeprefix('sartinas-')
    scene = scenario.casefold().split('-')
    if include_n2 and name == 'n2':
        return not any(key in scene for key in ('s12', 's18'))
    return name in HELD_SIGNERS or any(key in scene for key in ('s6', 's10'))


class Pieces:
    """Small Greek subword vocabulary with a character fallback for any Greek word."""

    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.symbols = tuple(dict.fromkeys(('', WORD_START, *ALPHABET.strip(), *(a + b for a, b in self.merges))))
        self.ids = {part: i for i, part in enumerate(self.symbols)}
        self.bos, self.eos = len(self.symbols), len(self.symbols) + 1

    @staticmethod
    def merge(word, pair):
        result, index = [], 0
        while index < len(word):
            if word[index:index + 2] == pair:
                result.append(''.join(pair))
                index += 2
            else:
                result.append(word[index])
                index += 1
        return tuple(result)

    @classmethod
    def fit(cls, texts, limit=185):
        words = Counter(tuple(WORD_START + word) for text in texts for word in greek(text).split())
        merges = []
        while len(merges) < limit:
            counts = Counter()
            for word, frequency in words.items():
                for pair in zip(word, word[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            merges.append(pair)
            reduced = Counter()
            for word, count in words.items():
                reduced[cls.merge(word, pair)] += count
            words = reduced
        return cls(merges)

    def encode(self, text):
        pieces = []
        for word in greek(text).split():
            row = tuple(WORD_START + word)
            for pair in self.merges:
                row = self.merge(row, pair)
            pieces.extend(self.ids[part] for part in row)
        return tuple(pieces)

    def decode(self, ids):
        return greek(''.join(self.symbols[i] for i in ids if 0 < i < self.bos).replace(WORD_START, ' '))


def resample(frames, source_fps, target_fps, count=None):
    if count is None:
        count = math.ceil(len(frames) * target_fps / source_fps - 1e-7)
    indices = (torch.arange(count) * (source_fps / target_fps)).round().long().clamp(max=len(frames) - 1)
    return frames[indices]


@torch.inference_mode()
def embed_stream(visual, stream):
    at_15 = resample(stream.frames, stream.fps, 15)
    # Fixed physical windows keep the pretrained short-clip encoder in its
    # training regime. The learned timeline network still receives every frame.
    embedded = []
    for start in range(0, len(at_15), 96):
        end = min(start + 96, len(at_15))
        left, right = max(0, start - 24), min(len(at_15), end + 24)
        _, memory, _ = visual.encode(at_15[None, left:right], torch.tensor([right - left]))
        embedded.append(memory[0, start - left:end - left])
    raw = resample(stream.frames, stream.fps, FPS)
    pretrained = resample(torch.cat(embedded), 15, FPS, len(raw))
    return torch.cat((raw, pretrained), dim=-1).clone()


class SubtitleModel(nn.Module):
    def __init__(self, inputs, vocabulary, glosses):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, WIDTH), nn.GELU())
        self.temporal = nn.ModuleList(nn.Sequential(
            nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d), nn.GELU(), nn.Dropout(.12))
            for d in (1, 2, 4, 8))
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in self.temporal)
        self.ctc = nn.Linear(WIDTH, vocabulary)
        self.active = nn.Linear(WIDTH, 1)
        self.gloss = nn.Linear(WIDTH, glosses)
        self.words = nn.Embedding(vocabulary + 2, WIDTH, padding_idx=0)
        self.decoder = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.query, self.key = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)
        self.text = nn.Linear(WIDTH * 2, vocabulary + 2)

    def encode(self, data, lengths):
        mask = torch.arange(data.shape[1])[None, :, None] < lengths[:, None, None]
        value = self.project(data) * mask
        for temporal, norm in zip(self.temporal, self.norms, strict=True):
            value = norm(value + temporal(value.transpose(1, 2)).transpose(1, 2)) * mask
        return value

    def attend(self, memory, mask, query, keys):
        attention = (self.query(query) @ keys.transpose(1, 2) / math.sqrt(WIDTH)).masked_fill(~mask[:, None], -torch.inf)
        context = attention.softmax(-1) @ memory
        return self.text(torch.cat((query, context), dim=-1))

    def translate(self, memory, mask, tokens):
        query, _ = self.decoder(self.words(tokens), self.initial_context(memory, mask))
        return self.attend(memory, mask, query, self.key(memory))

    @staticmethod
    def initial_context(memory, mask):
        return ((memory * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).tanh().unsqueeze(0)


def memory_batch(model, features):
    lengths = torch.tensor([len(row) for row in features])
    return model.encode(pad_sequence(features, batch_first=True), lengths), lengths


def frame_spans(turns, count):
    return [(min(count - 1, round(start * FPS)), min(count, max(round(start * FPS) + 1, round(end * FPS))), text)
            for start, end, text in turns]


def pack_intervals(memory, intervals):
    rows = [memory[index, start:end] for index, start, end in intervals]
    lengths = torch.tensor([len(row) for row in rows])
    return pad_sequence(rows, batch_first=True), torch.arange(max(lengths))[None] < lengths[:, None]


def ctc_loss(scores, targets, lengths):
    keep = [i for i, (length, target) in enumerate(zip(lengths, targets, strict=True))
            if len(target) + sum(a == b for a, b in zip(target, target[1:])) <= length]
    if not keep:
        return scores.sum() * 0
    active = [targets[i] for i in keep]
    return F.ctc_loss(scores[keep].log_softmax(-1).transpose(0, 1),
                      torch.tensor([token for row in active for token in row]),
                      torch.tensor([lengths[i] for i in keep]), torch.tensor(list(map(len, active))))


def align_takes(paired, mask, intervals, targets):
    owners = torch.tensor([index for index, _, _ in intervals])
    cross_take = owners[:, None] != owners[None, :]
    same_text = torch.tensor([[left == right for right in targets] for left in targets])
    matches = cross_take & same_text
    valid = matches.any(1)
    if not valid.any():
        return paired.sum() * 0
    embedded = F.normalize((paired * mask[..., None]).sum(1), dim=-1)
    similarities = (embedded @ embedded.T)[valid] / .12
    positives = similarities.masked_fill(~matches[valid], -torch.inf).logsumexp(1)
    all_takes = similarities.masked_fill(~cross_take[valid], -torch.inf).logsumexp(1)
    return (all_takes - positives).mean()


def loss_for(model, codec, streams, features, gloss_ids, *, timed):
    memory, lengths = memory_batch(model, features)
    intervals, targets, texts = [], [], []
    activity, annotated_gloss = torch.zeros(memory.shape[:2]), torch.full(memory.shape[:2], -100, dtype=torch.long)
    for index, (stream, length) in enumerate(zip(streams, lengths.tolist(), strict=True)):
        for start, end, text in frame_spans(stream.turns, length):
            intervals.append((index, start, end))
            targets.append(codec.encode(text))
            texts.append((codec.bos, *targets[-1], codec.eos))
            activity[index, start:end] = 1
        for start, end, label in frame_spans(stream.glosses, length):
            if label in gloss_ids:
                annotated_gloss[index, start:end] = gloss_ids[label]
    paired, mask = pack_intervals(memory, intervals)
    tokens = pad_sequence([torch.tensor(row) for row in texts], batch_first=True)
    predicted = model.translate(paired, mask, tokens[:, :-1])
    label = tokens[:, 1:]
    translation = F.cross_entropy(predicted.transpose(1, 2), label, ignore_index=0, label_smoothing=.05)
    if len(targets) > 1:
        different = torch.tensor([row != targets[index - 1] for index, row in enumerate(targets)])
        if different.any():
            wrong = model.translate(paired.roll(1, 0), mask.roll(1, 0), tokens[:, :-1])
            target_lengths = (label != 0).sum(1)
            positive = F.cross_entropy(predicted.transpose(1, 2), label, ignore_index=0, reduction='none').sum(1) / target_lengths
            negative = F.cross_entropy(wrong.transpose(1, 2), label, ignore_index=0, reduction='none').sum(1) / target_lengths
            translation = translation + .20 * F.relu(.22 + positive[different] - negative[different]).mean()
    local_ctc = ctc_loss(model.ctc(paired), targets, mask.sum(1).tolist())
    if not timed:
        return translation + .4 * local_ctc
    valid = torch.arange(memory.shape[1])[None] < lengths[:, None]
    activity_loss = F.binary_cross_entropy_with_logits(model.active(memory).squeeze(-1)[valid], activity[valid],
                                                        pos_weight=torch.tensor(1.5))
    silent = valid & (activity == 0) & (annotated_gloss == -100)
    scores = model.ctc(memory)
    silence_loss = -scores[silent].log_softmax(-1)[:, 0].mean() if silent.any() else scores.sum() * 0
    whole_targets = [codec.encode(' '.join(text for _, _, text in row.turns)) for row in streams]
    global_ctc = ctc_loss(scores, whole_targets, lengths.tolist())
    gloss_loss = (F.cross_entropy(model.gloss(memory).flatten(0, 1), annotated_gloss.flatten())
                  if (annotated_gloss != -100).any() else scores.sum() * 0)
    alignment = align_takes(paired, mask, intervals, targets) if len(streams) > 1 else scores.sum() * 0
    return (translation + .35 * local_ctc + .20 * global_ctc + .25 * activity_loss + .12 * silence_loss
            + .16 * gloss_loss + .20 * alignment)


def visual_checkpoint():
    record = json.loads(ENCODER_REPORT.read_bytes())['arms']['isolated_transfer']
    return Path(record['vision_path']), record['vision_sha256']


def paired_sources():
    from train_continuous_library import datasets
    from train_targeted_greek import BASE as OLD, TARGET, compatible_training, split_rows

    historical, _, provenance = datasets(OLD / 'manifest.json', OLD / 'features.jsonl')
    manifest, extra, _, added_hash = split_rows(TARGET / 'manifest.json', TARGET / 'features.jsonl')
    certh = compatible_training(historical, extra, [row for row in manifest['samples'] if row['split'] != 'train'])
    elementary = elementary_sets()['train']
    rows = [(row, 'certh', row.fps) for row in certh] + [(row, 'elementary23', 15) for row in elementary]
    pairs = [Stream(row.id, f'{source}-{row.signer}', source, 'train', row.frames, fps,
                    ((0., len(row.frames) / fps, greek(row.text)),)) for row, source, fps in rows]
    return pairs, {'certh': provenance, 'extra_certh_features': added_hash,
                   'elementary_train_ids': sorted(row.id for row in elementary),
                   'elementary_features_sha256': sha(Path('data/elementary-translation-pilot/features.jsonl'))}


def extra_pairs(folder):
    manifest, features = folder / 'manifest.json', folder / 'features.jsonl'
    source = json.loads(manifest.read_bytes())
    rows = {row['id']: row for row in source['samples']}
    if len(rows) != len(source['samples']) or any(row['split'] != 'train' for row in rows.values()):
        raise ValueError(f'Expected only authentic paired warm-up training sources: {folder}')
    result = []
    with features.open(encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            row = rows[item['id']]
            source_key = sha256(json.dumps([row['fps'], row['frame_paths']], ensure_ascii=False,
                                          separators=(',', ':')).encode()).hexdigest()
            frames = torch.tensor(item['features'], dtype=torch.float32)
            text = greek(row['text'])
            if (item['feature_schema'] != FEATURE_SCHEMA or item['fps'] != 15 or not text
                    or item['source_key'] != source_key or not frames.isfinite().all()
                    or frames.shape != (len(row['frame_paths'][::item['stride']]), FEATURE_COUNT)):
                raise ValueError(f'Invalid supplementary paired evidence: {item["id"]}')
            result.append(Stream(item['id'], f'elementary23-{row["signer"]}', 'elementary23', 'train', frames, 15,
                                 ((0., len(frames) / 15, text),), source_key=source_key))
    if len(result) != len(rows) or {row.id for row in result} != set(rows):
        raise ValueError(f'Incomplete or duplicate supplementary features: {folder}')
    return result, {'manifest_sha256': sha(manifest), 'features_sha256': sha(features), 'pairs': len(result)}


def prepare_pairs(path, supplement):
    paired, provenance = paired_sources()
    evidence = {}
    for folder in supplement:
        additional, record = extra_pairs(folder)
        paired.extend(additional)
        evidence[folder.as_posix()] = record
    if len({row.id for row in paired}) != len(paired):
        raise ValueError('Duplicate paired warm-up recording')
    visual_path, visual_hash = visual_checkpoint()
    visual, _ = load_visual(visual_path, visual_hash)
    features = [embed_stream(visual, row) for row in paired]
    target = {'visual_sha256': visual_hash, 'provenance': {**provenance, 'supplements': evidence},
              'rows': [{'id': row.id, 'signer': row.signer, 'scenario': row.scenario, 'turns': row.turns,
                        'features': values} for row, values in zip(paired, features, strict=True)]}
    save_checkpoint(path, target)
    print(f'Cached {len(features)} verified paired warm-up clips', flush=True)


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.pt')
    torch.save(data, pending)
    pending.replace(path)


def prepare_native_model(args, codec, gloss_ids, warmup, *, initial_state=None):
    vocab = len(codec.symbols)
    model = SubtitleModel(warmup[0]['features'].shape[1], vocab, max(1, len(gloss_ids)))
    if initial_state is not None:
        shared = {name: value for name, value in initial_state.items() if not name.startswith('project.')}
        status = model.load_state_dict(shared, strict=False)
        if status.unexpected_keys or any(not name.startswith('project.') for name in status.missing_keys):
            raise ValueError('Incompatible cross-corpus model weights')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = random.Random(SEED)
    pairs, warm_features = warmup_pairs(warmup)
    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        order = rng.sample(range(len(pairs)), len(pairs))
        losses = []
        for start in range(0, len(order), 20):
            group = order[start:start + 20]
            loss = loss_for(model, codec, [pairs[i] for i in group], [warm_features[i] for i in group], gloss_ids, timed=False)
            step(optimizer, model, loss)
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0:
            print(f'Paired warm-up {epoch}: loss {sum(losses) / len(losses):.3f}', flush=True)
    return model


def warmup_pairs(warmup):
    return ([Stream(row['id'], row['signer'], row['scenario'], 'train', torch.empty(0), FPS, tuple(row['turns']))
             for row in warmup], [row['features'] for row in warmup])


def preparation(native_text, native_glosses, rows):
    codec = Pieces.fit(native_text + [text for row in rows for _, _, text in row['turns']])
    gloss_ids = {word: index for index, word in enumerate(sorted(set(native_glosses)))}
    return codec, gloss_ids


def pretrain(args):
    plan = json.loads((args.data / 'manifest.planned.json').read_bytes())
    rows = [row for row in plan['dialogues'] if row['split'] == 'train']
    if not rows or any(is_reserved(row['signer'], row['scenario'], include_n2=args.cross_take) for row in rows):
        raise ValueError('Invalid timing plan for vocabulary preparation')
    annotations = {row['id']: row for row in plan['samples'] if row['split'] == 'train'}
    native_text = [greek(annotations[key]['text']) for row in rows for key in row['turn_ids']]
    native_glosses = [item.get('label', item.get('gloss')) for row in rows for item in row.get('gloss_intervals', ())]
    cache = torch.load(args.pairs, weights_only=True)
    codec, gloss_ids = preparation(native_text, native_glosses, cache['rows'])
    torch.manual_seed(SEED)
    model = prepare_native_model(args, codec, gloss_ids, cache['rows'])
    save_checkpoint(args.warmup, {'model_version': MODEL_VERSION, 'state': model.state_dict(),
                                 'merges': codec.merges, 'gloss_ids': gloss_ids,
                                 'pairs_sha256': sha(args.pairs), 'warmup_epochs': args.warmup_epochs})


def step(optimizer, model, loss):
    if not loss.isfinite():
        raise ValueError('Subtitle loss became non-finite')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()


def warp_timeline(row, features, rng):
    rate = rng.uniform(.88, 1.12)
    altered = resample(features, FPS / rate, FPS)
    if rng.random() < .5:
        missing = torch.rand(len(altered)) < .02
        altered[missing] = 0.
    return replace(row, turns=tuple((start * rate, end * rate, text) for start, end, text in row.turns),
                   glosses=tuple((start * rate, end * rate, label) for start, end, label in row.glosses)), altered


def activity_spans(logits, floor, gap):
    probability = F.avg_pool1d(logits.sigmoid()[None, None], 5, stride=1, padding=2).flatten().tolist()
    spans, start, last = [], None, None
    for index, score in enumerate(probability):
        if score >= floor:
            if start is None:
                start = index
            last = index
        if start is not None and (index - last > gap or index == len(probability) - 1):
            if last - start >= 3:
                while last + 1 - start > 12 * FPS:
                    cut = min(range(start + 5 * FPS, start + 10 * FPS), key=probability.__getitem__)
                    spans.append((start, cut))
                    start = cut
                spans.append((start, last + 1))
            start = last = None
    return spans


def ctc_tokens(logits):
    score, index = logits.softmax(-1).max(-1)
    ids, scores, last = [], [], 0
    for token, probability in zip(index.tolist(), score.tolist(), strict=True):
        if token and token != last:
            ids.append(token)
            scores.append(probability)
        elif token:
            scores[-1] = max(scores[-1], probability)
        last = token
    return ids, sum(scores) / len(scores) if scores else 0.


@torch.inference_mode()
def generate(model, codec, memory, spans):
    if not spans:
        return []
    packed, mask = pack_intervals(memory, [(0, start, end) for start, end in spans])
    keys, hidden = model.key(packed), model.initial_context(packed, mask)
    tokens = torch.full((len(spans), 1), codec.bos, dtype=torch.long)
    outputs, confidences = [[] for _ in spans], [[] for _ in spans]
    complete = [False] * len(spans)
    for _ in range(38):
        query, hidden = model.decoder(model.words(tokens), hidden)
        logits = model.attend(packed, mask, query, keys)[:, 0]
        logits[:, [0, codec.bos]] = -torch.inf
        values, ids = logits.softmax(-1).max(-1)
        for index, token in enumerate(ids.tolist()):
            if complete[index]:
                continue
            if token == codec.eos:
                complete[index] = True
            else:
                outputs[index].append(token)
                confidences[index].append(float(values[index]))
        if all(complete):
            break
        tokens = ids[:, None]
    return [(codec.decode(row), sum(scores) / len(scores) if scores else 0., ended)
            for row, scores, ended in zip(outputs, confidences, complete, strict=True)]


@torch.inference_mode()
def decode_memory(model, codec, features, memory, logits, activity, policy, spans=None):
    if spans is None:
        spans = activity_spans(activity, policy['activity'], policy['gap'])
    generated = generate(model, codec, memory, spans)
    output = []
    for (start, end), (text, score, ended) in zip(spans, generated, strict=True):
        ctc, ctc_score = ctc_tokens(logits[start:end])
        candidate = codec.decode(ctc)
        raw = features[start:end, :FEATURE_COUNT]
        tracked = ((raw[:, 222] >= .75) & (raw[:, 220:222] >= .75).any(1)).float().mean().item()
        motion = tracked_hand_motion(raw)
        visual_signal = tracked >= .45 and motion >= .003
        if policy['decoder'] == 'ctc':
            accepted = candidate if visual_signal and ctc_score >= policy['confidence'] else ''
        else:
            accepted = text if visual_signal and ended and score >= policy['confidence'] else ''
        output.append({'start_s': start / FPS, 'end_s': end / FPS, 'text': accepted,
                       'ar_text': text, 'ar_score': score, 'ar_completed': ended,
                       'ctc_text': candidate, 'ctc_score': ctc_score, 'visual_signal': visual_signal,
                       'tracking_fraction': tracked, 'hand_motion': motion})
    return output


@torch.inference_mode()
def predictions(model, codec, features, policy, *, spans=None):
    memory, _ = memory_batch(model, [features])
    return decode_memory(model, codec, features, memory, model.ctc(memory)[0], model.active(memory)[0, :, 0], policy, spans)


def measure(reference, outputs, observed_training_text):
    expected = [text for _, _, text in reference]
    target = ' '.join(expected).split()
    actual = ' '.join(row['text'] for row in outputs).split()
    substitutions, deletions, insertions = edit_counts(target, actual)
    edits = substitutions + deletions + insertions
    matched = set()
    novel = {i for i, text in enumerate(expected) if text not in observed_training_text}
    for guess in outputs:
        candidates = [i for i, (start, end, text) in enumerate(reference) if i not in matched and text == guess['text']
                      and min(end, guess['end_s']) - max(start, guess['start_s'])
                      >= .5 * max(end - start, guess['end_s'] - guess['start_s'])]
        if candidates:
            matched.add(candidates[0])
    return {'reference_words': len(target), 'word_edits': edits, 'wer': edits / len(target),
            'aligned_correct_words': len(target) - substitutions - deletions,
            'reference_turns': len(reference), 'exact_timed_turns': len(matched),
            'novel_reference_turns': len(novel), 'exact_novel_timed_turns': len(matched & novel),
            'emitted_turns': sum(bool(row['text']) for row in outputs),
            'exact_dialogue': target == actual, 'predicted_words': len(actual)}


def focus_evidence(reference, predictions):
    matches = []
    for start, end, text in reference:
        terms = [term for term in FOCUS if f' {term} ' in f' {text} ']
        if not terms:
            continue
        overlapping = [row for row in predictions if row['start_s'] < end and row['end_s'] > start]
        accepted = ' '.join(row['text'] for row in overlapping)
        matches.append({'start_s': start, 'end_s': end, 'reference': text, 'terms': terms,
                        'recognized_terms': [term for term in terms if f' {term} ' in f' {accepted} '],
                        'overlapping_predictions': overlapping})
    return matches


def causal_spans(detector, row, feature_count):
    edges = [(round(start * FPS / row.fps), min(feature_count, round(end * FPS / row.fps)))
             for start, end in detector.predict(row.frames, fps=row.fps)]
    return [(left, right) for left, right in edges if right > left]


@torch.inference_mode()
def dev_select(model, codec, dev, features, observed, causal=None):
    memories = [memory_batch(model, [value])[0] for value in features]
    signals = [(value, memory, model.ctc(memory)[0], model.active(memory)[0, :, 0])
               for value, memory in zip(features, memories, strict=True)]
    best = None
    options = [('joint', activity, gap) for activity in (.40, .55, .70) for gap in (2, 5)]
    if causal is not None:
        options.append(('causal', None, None))
    for detector, activity, gap in options:
        reference_policy = {'boundary': detector, 'activity': activity, 'gap': gap, 'decoder': 'ar', 'confidence': 0.}
        edges = causal if detector == 'causal' else [None] * len(signals)
        predictions_by_stream = [decode_memory(model, codec, *signal, reference_policy, spans)
                                 for signal, spans in zip(signals, edges, strict=True)]
        for decoder in ('ar', 'ctc'):
            for confidence in (.0, .35, .55):
                result = []
                for source, outputs in zip(dev, predictions_by_stream, strict=True):
                    rows = []
                    for row in outputs:
                        eligible = row['visual_signal'] and (decoder == 'ctc' or row['ar_completed'])
                        text = row[f'{decoder}_text'] if eligible and row[f'{decoder}_score'] >= confidence else ''
                        rows.append({**row, 'text': text})
                    result.append(measure(source.turns, rows, observed))
                edits = sum(row['word_edits'] for row in result)
                exact = sum(row['exact_timed_turns'] for row in result)
                key = (edits, -exact, -confidence)
                if best is None or key < best[0]:
                    policy = {**reference_policy, 'decoder': decoder, 'confidence': confidence}
                    best = key, policy, result
    return best


def train(args):
    if args.freeze.exists():
        raise ValueError(f'Existing frozen model: {args.freeze}')
    streams, provenance = load_streams(args.data, ('train', 'val'))
    fit, dev = check_fit_split(streams, include_n2=args.cross_take)
    visual_path, visual_hash = visual_checkpoint()
    cache = torch.load(args.pairs, weights_only=True)
    if cache['visual_sha256'] != visual_hash:
        raise ValueError('The warm-up features use a different visual model')
    visual, _ = load_visual(visual_path, visual_hash)
    fit_features, dev_features = ([embed_stream(visual, row) for row in group] for group in (fit, dev))
    causal = None
    if args.boundary_freeze:
        from train_native_dialogue_boundaries import load_native_boundary_detector

        boundary = load_native_boundary_detector(args.boundary_freeze)
        causal = [causal_spans(boundary, row, len(features)) for row, features in zip(dev, dev_features, strict=True)]
    native_text = [text for row in fit for _, _, text in row.turns]
    observed = set(native_text)
    observed.update(text for row in cache['rows'] for _, _, text in row['turns'])
    codec, gloss_ids = preparation(native_text, [label for row in fit for _, _, label in row.glosses], cache['rows'])
    torch.manual_seed(SEED)
    additional_pretraining = None
    if args.warmup.exists():
        initial = torch.load(args.warmup, weights_only=True)
        if (initial.get('model_version') != MODEL_VERSION or initial['merges'] != codec.merges or initial['gloss_ids'] != gloss_ids
                or initial['pairs_sha256'] != sha(args.pairs) or initial['warmup_epochs'] != args.warmup_epochs):
            raise ValueError('Warm-up weights were prepared with a different vocabulary or paired data')
        model = SubtitleModel(fit_features[0].shape[1], len(codec.symbols), max(1, len(gloss_ids)))
        model.load_state_dict(initial['state'])
        additional_pretraining = initial.get('additional_pretraining')
        if additional_pretraining:
            observed.update(additional_pretraining['training_text'])
    else:
        model = prepare_native_model(args, codec, gloss_ids, cache['rows'])
    pairs, warm_features = warmup_pairs(cache['rows'])
    torch.manual_seed(SEED + 1)
    rng = random.Random(SEED + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0013, weight_decay=.015)
    groups = [[i] for i in range(len(fit))]
    if args.cross_take:
        groups = [[i for i, row in enumerate(fit) if row.scenario == scene]
                  for scene in sorted({row.scenario for row in fit})]
        if any(len({fit[i].signer for i in group}) < 2 for group in groups):
            raise ValueError('Cross-take training needs multiple signers for every training script')
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for group in rng.sample(groups, len(groups)):
            timed = [warp_timeline(fit[index], fit_features[index], rng) if epoch % 3
                     else (fit[index], fit_features[index]) for index in group]
            rows, values = map(list, zip(*timed, strict=True))
            loss = loss_for(model, codec, rows, values, gloss_ids, timed=True)
            if epoch % 2:
                keep = rng.sample(range(len(pairs)), 12)
                loss = loss + .15 * loss_for(model, codec, [pairs[i] for i in keep],
                                             [warm_features[i] for i in keep], gloss_ids, timed=False)
            step(optimizer, model, loss)
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            key, policy, results = dev_select(model, codec, dev, dev_features, observed, causal)
            record = {'epoch': epoch, 'loss': sum(losses) / len(losses), 'policy': policy, 'validation': results}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if best is None or key < best[0]:
                best = key, record, {name: value.detach().clone() for name, value in model.state_dict().items()}
    _, picked, state = best
    save_checkpoint(args.checkpoint, {'model_version': MODEL_VERSION, 'state': state,
                                     'merges': codec.merges, 'inputs': fit_features[0].shape[1],
                                     'gloss_classes': max(1, len(gloss_ids)), 'observed_training_text': sorted(observed)})
    write(args.freeze, {'scope': 'Native whole-video timeline training with timed Greek supervision and open subword generation.',
                        'data': provenance, 'warmup_sha256': sha(args.pairs), 'checkpoint_sha256': sha(args.checkpoint),
                        'warmup_model_sha256': sha(args.warmup) if args.warmup.exists() else None,
                        'additional_pretraining': ({key: value for key, value in additional_pretraining.items()
                                                    if key != 'training_text'} if additional_pretraining else None),
                        'visual_path': visual_path.as_posix(), 'visual_sha256': visual_hash, 'model_fps': FPS,
                        'warmup_pairs': len(pairs), 'warmup_epochs': args.warmup_epochs, 'native_epochs': args.epochs,
                        'native_fit': [describe(row) for row in fit], 'native_dev': [describe(row) for row in dev],
                        'shared_development_scenarios': sorted({row.scenario for row in fit} & {row.scenario for row in dev}),
                        'cross_take_alignment': args.cross_take,
                        'causal_boundary_freeze_sha256': sha(args.boundary_freeze) if args.boundary_freeze else None,
                        'greek_subwords': len(codec.symbols) - 1, 'timed_gloss_classes': len(gloss_ids),
                        'selected': picked, 'development': history})


def describe(row):
    return {'id': row.id, 'signer': row.signer, 'scenario': row.scenario, 'source_key': row.source_key,
            'seconds': len(row.frames) / row.fps, 'turns': len(row.turns), 'glosses': len(row.glosses)}


def evaluate(args):
    frozen = json.loads(args.freeze.read_bytes())
    if sha(args.checkpoint) != frozen['checkpoint_sha256']:
        raise ValueError('The Greek subtitle checkpoint changed since validation')
    held, provenance = load_streams(args.data, ('test',))
    used = frozen['native_fit'] + frozen['native_dev']
    for name in ('id', 'signer', 'scenario', 'source_key'):
        if {row[name] for row in used} & {getattr(row, name) for row in held}:
            raise ValueError(f'The independent evaluation overlaps fit or development in {name}')
    artifact = torch.load(args.checkpoint, weights_only=True)
    if artifact['model_version'] != MODEL_VERSION:
        raise ValueError('Incompatible subtitle checkpoint version')
    visual, _ = load_visual(Path(frozen['visual_path']), frozen['visual_sha256'])
    codec = Pieces(artifact['merges'])
    model = SubtitleModel(artifact['inputs'], len(codec.symbols), artifact['gloss_classes'])
    model.load_state_dict(artifact['state'])
    model.eval()
    policy, observed = frozen['selected']['policy'], set(artifact['observed_training_text'])
    warmup_only = None
    if frozen['warmup_model_sha256'] and args.warmup.exists():
        if sha(args.warmup) != frozen['warmup_model_sha256']:
            raise ValueError('The comparison warm-up weights changed after freezing')
        initial = torch.load(args.warmup, weights_only=True)
        warmup_only = SubtitleModel(artifact['inputs'], len(codec.symbols), artifact['gloss_classes'])
        warmup_only.load_state_dict(initial['state'])
        warmup_only.eval()
    boundary = None
    if args.boundary_freeze:
        from train_native_dialogue_boundaries import load_native_boundary_detector

        boundary = load_native_boundary_detector(args.boundary_freeze)
    selected_causal = policy.get('boundary') == 'causal'
    if selected_causal and (not boundary or sha(args.boundary_freeze) != frozen['causal_boundary_freeze_sha256']):
        raise ValueError('The selected causal detector must be the same frozen detector used on development')
    output = []
    for row in held:
        start = perf_counter()
        features = embed_stream(visual, row)
        spans = causal_spans(boundary, row, len(features)) if boundary else None
        rendered = predictions(model, codec, features, policy, spans=spans if selected_causal else None)
        elapsed = perf_counter() - start
        results = measure(row.turns, rendered, observed)
        record = {**describe(row), 'metrics': results, 'decode_seconds': elapsed,
                  'reference': row.turns, 'prediction': rendered, 'appointment_evidence': focus_evidence(row.turns, rendered)}
        if warmup_only:
            same_spans = [(round(item['start_s'] * FPS), round(item['end_s'] * FPS)) for item in rendered]
            comparison = predictions(warmup_only, codec, features, policy, spans=same_spans)
            record['warmup_only_same_intervals'] = {'metrics': measure(row.turns, comparison, observed),
                                                   'prediction': comparison}
        if boundary and not selected_causal:
            automatic = predictions(model, codec, features, policy, spans=spans)
            record['causal_detector_comparison'] = {'metrics': measure(row.turns, automatic, observed), 'prediction': automatic,
                                                     'appointment_evidence': focus_evidence(row.turns, automatic)}
        output.append(record)
        print(json.dumps({'id': row.id, **results}, ensure_ascii=False), flush=True)
    write(args.output, {'scope': 'Predictions from full native recordings without reference timing, text or turn counts. '
                               'References are used after inference for scoring.',
                        'freeze_sha256': sha(args.freeze), 'data': provenance, 'policy': policy, 'recordings': output,
                        'causal_boundary_freeze_sha256': sha(args.boundary_freeze) if boundary else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('cache-pairs', 'warmup', 'train', 'evaluate'))
    parser.add_argument('--data', type=Path, default=BASE)
    parser.add_argument('--pairs', type=Path, default=ARTIFACTS / 'warmup-features.pt')
    parser.add_argument('--supplement', type=Path, action='append', default=[],
                        help='Add an independently verified paired training directory to the cache')
    parser.add_argument('--warmup', type=Path, default=ARTIFACTS / 'warmup.pt')
    parser.add_argument('--checkpoint', type=Path, default=ARTIFACTS / 'subtitle.pt')
    parser.add_argument('--freeze', type=Path, default=REPORT / 'freeze.json')
    parser.add_argument('--output', type=Path, default=REPORT / 'held-benchmark.json')
    parser.add_argument('--boundary-freeze', type=Path, help='Compare an independently frozen automatic interval detector')
    parser.add_argument('--cross-take', action='store_true', help='Align same-reference turns across original takes during fit')
    parser.add_argument('--warmup-epochs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=45)
    args = parser.parse_args()
    if args.epochs < 1 or args.warmup_epochs < 1:
        parser.error('Training epochs must be positive')
    torch.set_num_threads(3)
    if args.phase == 'cache-pairs':
        prepare_pairs(args.pairs, args.supplement)
    elif args.phase == 'warmup':
        pretrain(args)
    elif args.phase == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
