"""Small train-only IBM1 gloss-to-Greek lexical baseline on the frozen official split.

Outputs are partial lexical suggestions, not validated fluent translations. The
11 prior test scripts are retrospective; no native/video/fresh test is loaded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from train_continuous import edit_counts
from train_greek_translation_pilot import greek, sha

ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path('reports/greek-translation-pilot-context/official-gloss-split.json')
PREVIOUS = Path('reports/greek-translation-pilot-context/official-gloss-to-greek.json')
PREFIX = Path('reports/timed-gloss')
NULL = '<NULL>'
ITERATIONS = 40
SUPPORTS = (1, 2, 3)
CONFIDENCES = (0., .25, .5, .75)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fit_ibm1(rows):
    cooccurs = defaultdict(set)
    for row in rows:
        for gloss in [NULL, *row['glosses']]:
            cooccurs[gloss].update(greek(row['text']).split())
    table = {gloss: {word: 1 / len(words) for word in sorted(words)}
             for gloss, words in sorted(cooccurs.items())}
    trace = []
    for _ in range(ITERATIONS):
        counts = defaultdict(Counter)
        likelihood = 0.
        for row in rows:
            source = [NULL, *row['glosses']]
            for word in greek(row['text']).split():
                total = sum(table[gloss].get(word, 0.) for gloss in source)
                if total <= 0:
                    raise ValueError('Zero IBM1 mass on a training word')
                likelihood += math.log(total / len(source))
                for gloss in source:
                    counts[gloss][word] += table[gloss].get(word, 0.) / total
        trace.append(likelihood)
        table = {gloss: {word: value / sum(words.values()) for word, value in words.items()}
                 for gloss, words in sorted(counts.items())}
    if any(b + 1e-8 < a for a, b in zip(trace, trace[1:])):
        raise AssertionError('IBM1 likelihood must not decrease')
    return table, trace


def keys(glosses, position):
    gloss = glosses[position]
    left = glosses[position - 1] if position else '<START>'
    right = glosses[position + 1] if position + 1 < len(glosses) else '<END>'
    return (('both', gloss, left, right), ('left', gloss, left), ('right', gloss, right), ('gloss', gloss))


def fit_phrases(rows, table):
    lexicon = defaultdict(dict)
    statistics = Counter()
    for row in rows:
        glosses, words = row['glosses'], greek(row['text']).split()
        source = [NULL, *glosses]
        aligned = []
        for word in words:
            scores = [table[gloss].get(word, 0.) for gloss in source]
            winner = max(range(len(scores)), key=lambda index: scores[index])
            aligned.append((winner, scores[winner] / sum(scores)))
            statistics['training_words'] += 1
            statistics['null_aligned_words'] += winner == 0
        start = 0
        while start < len(words):
            owner = aligned[start][0]
            end = start + 1
            while end < len(words) and aligned[end][0] == owner:
                end += 1
            if owner and end - start <= 3:
                phrase = tuple(words[start:end])
                weight = sum(value for _, value in aligned[start:end]) / (end - start)
                for key in keys(glosses, owner - 1):
                    candidate = lexicon[key].setdefault(phrase, {'weight': 0., 'templates': set()})
                    candidate['weight'] += weight
                    candidate['templates'].add(row['template'])
                statistics['retained_short_chunks'] += 1
            elif owner:
                statistics['discarded_chunks_longer_than_three_words'] += 1
            start = end
    return dict(lexicon), dict(statistics)


def predict(glosses, lexicon, policy):
    emitted, evidence = [], []
    for position, gloss in enumerate(glosses):
        chosen = None
        for key in keys(glosses, position):
            candidates = lexicon.get(key)
            if not candidates:
                continue
            documents = set().union(*(row['templates'] for row in candidates.values()))
            if key[0] != 'gloss' and len(documents) < 2:
                continue
            phrase, candidate = min(candidates.items(), key=lambda pair: (
                -pair[1]['weight'], -len(pair[1]['templates']), pair[0]))
            confidence = candidate['weight'] / sum(item['weight'] for item in candidates.values())
            support = len(candidate['templates'])
            if support >= policy['minimum_support'] and confidence >= policy['minimum_vote_share']:
                chosen = {'gloss_index': position, 'gloss': gloss, 'context_key': key,
                          'phrase': phrase, 'vote_share': confidence, 'support': support,
                          'training_templates': sorted(candidate['templates'])}
                break
        if chosen:
            emitted.extend(chosen['phrase'])
            evidence.append(chosen)
    return {'prediction': ' '.join(emitted), 'evidence': evidence,
            'supported_glosses': len(evidence), 'total_glosses': len(glosses),
            'partial': len(evidence) < len(glosses)}


def evaluate(rows, predictions):
    details = []
    for row, output in zip(rows, predictions, strict=True):
        reference = greek(row['text'])
        sub, delete, insert = edit_counts(reference.split(), output['prediction'].split())
        correct = len(reference.split()) - sub - delete
        details.append({'template': row['template'], 'glosses': row['glosses'], 'reference': reference,
                        **output, 'substitutions': sub, 'deletions': delete, 'insertions': insert,
                        'word_edits': sub + delete + insert,
                        'aligned_correct_emitted_words': correct})
    reference_words = sum(len(row['reference'].split()) for row in details)
    predicted_words = sum(len(row['prediction'].split()) for row in details)
    summary = {'templates': len(details), 'reference_words': reference_words,
               'predicted_words': predicted_words,
               'exact': sum(row['reference'] == row['prediction'] for row in details),
               'abstentions': sum(not row['prediction'] for row in details),
               'partial_outputs': sum(bool(row['prediction']) and row.get('partial', False) for row in details)}
    for key in ('word_edits', 'substitutions', 'deletions', 'insertions', 'aligned_correct_emitted_words'):
        summary[key] = sum(row[key] for row in details)
    summary['wer'] = summary['word_edits'] / reference_words
    summary['aligned_word_precision'] = summary['aligned_correct_emitted_words'] / predicted_words if predicted_words else None
    summary['aligned_word_recall'] = summary['aligned_correct_emitted_words'] / reference_words
    return {'summary': summary, 'rows': details}


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the project root')
    started = perf_counter()
    split, previous = json.loads(SPLIT.read_bytes()), json.loads(PREVIOUS.read_bytes())
    if sha(SPLIT) != previous['split_sha256']:
        raise ValueError('The frozen official split changed')
    roles = split['roles']
    if {name: len(rows) for name, rows in roles.items()} != {'train': 133, 'development': 9, 'test': 11}:
        raise ValueError('Expected exactly the archived133/9/11 script split')
    if any(NULL in row['glosses'] for rows in roles.values() for row in rows):
        raise ValueError('Reserved NULL token collision')
    protected = {path.as_posix(): sha(path) for path in (SPLIT, PREVIOUS, Path(__file__).relative_to(ROOT))}
    protocol = {'scope': 'A new lexical method on a previously evaluated gold-gloss split; prior11 tests are retrospective.',
                'source_sha256': protected, 'iterations': ITERATIONS, 'maximum_phrase_words': 3,
                'training': 'IBM1 EM with NULL, uniform source-position alignment and train-only word cooccurrence support. No literal-gloss spelling shortcut.',
                'phrases': 'Train-only Viterbi alignment; keep observed contiguous1–3-word chunks. Statistical alignments, not expert word-level labels.',
                'context': 'Try both neighbours, left, right, unigram; context keys require at least2 distinct training templates. Back off if selected candidate fails policy.',
                'grid': [{'minimum_support': support, 'minimum_vote_share': confidence}
                         for support in SUPPORTS for confidence in CONFIDENCES],
                'selection': ['minimum development word edits', 'fewer insertions', 'more aligned correct words',
                              'higher minimum vote share', 'higher minimum support'],
                'unknown_glosses': 'Abstain per gloss; explicitly mark partial emitted suggestions.',
                'no_native_or_fresh_n8_data_loaded': True}
    protocol_path = PREFIX / 'lexicon-protocol.json'
    write(protocol_path, protocol)
    table, likelihood = fit_ibm1(roles['train'])
    lexicon, induction = fit_phrases(roles['train'], table)
    development = []
    for policy in protocol['grid']:
        result = evaluate(roles['development'], [predict(row['glosses'], lexicon, policy) for row in roles['development']])
        development.append({'policy': policy, **result['summary']})
    chosen = min(development, key=lambda row: (row['word_edits'], row['insertions'],
                 -row['aligned_correct_emitted_words'], -row['policy']['minimum_vote_share'], -row['policy']['minimum_support']))
    model = {'protocol_sha256': sha(protocol_path), 'split_sha256': sha(SPLIT),
             'selected_on_development_only': chosen, 'development_grid': development,
             'training_templates': [row['template'] for row in roles['train']],
             'training_log_likelihood_before_each_em_step': likelihood, 'induction': induction,
             'translation_probabilities': table,
             'lexicon': [{'key': key, 'candidates': [{'phrase': phrase, 'weight': value['weight'],
                          'training_templates': sorted(value['templates'])} for phrase, value in sorted(candidates.items())]}
                         for key, candidates in sorted(lexicon.items())]}
    model_path = PREFIX / 'lexicon-model.json'
    write(model_path, model)
    frozen_hash = sha(model_path)
    # Test predictions begin only after the model and development-selected policy
    # have been written and hashed. The already-exposed test cannot select them.
    reports = {}
    for name, rows in roles.items():
        old_rows = previous['reports'][name]['learned_character_decoder']['rows']
        old_map = {row['template']: row for row in old_rows}
        if set(old_map) != {row['template'] for row in rows}:
            raise ValueError('Prior comparison does not use the same templates')
        if any(old_map[row['template']]['reference'] != greek(row['text']) for row in rows):
            raise ValueError('Prior Greek references differ')
        reports[name] = {
            'lexicon': evaluate(rows, [predict(row['glosses'], lexicon, chosen['policy']) for row in rows]),
            'blank': evaluate(rows, [{'prediction': ''} for _ in rows]),
            'previous_character_decoder': evaluate(rows, [{'prediction': old_map[row['template']]['prediction']} for row in rows])}
    if sha(model_path) != frozen_hash or any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Frozen model or source changed during scoring')
    report = {'scope': protocol['scope'], 'protocol_sha256': sha(protocol_path), 'model_sha256': frozen_hash,
              'source_sha256': protected, 'selected_policy': chosen['policy'], 'development_selection': chosen,
              'protected_sources_unchanged': True, 'training_only': True,
              'elapsed_seconds': perf_counter() - started,
              'metrics_note': 'Exact-word Levenshtein alignment counts, summed per unique script. Correct emitted words = reference words - substitutions - deletions; not a human semantic or fluency judgement.',
              'limitations': ['Gold glosses bypass vision; no whole-video or webcam improvement is established.',
                             'Nine development scripts from one scene make policy selection fragile;11 prior test scripts are retrospective.',
                             'Word/short-phrase alignments are learned statistical hypotheses, not expert linguistic annotations.',
                             'Source-order concatenation does not solve Greek grammar, morphological agreement or context.',
                             'Lower error than the old hallucinating decoder can mostly reflect omission; compare blank, recall and precision explicitly.',
                             'No prediction, threshold or lexicon is deployed; no fresh n8 evidence is accessed.'],
              'reports': reports}
    output = PREFIX / 'lexicon-results.json'
    write(output, report)
    print(json.dumps({'output': str(output), 'selected_policy': chosen['policy'],
                     'metrics': {role: {method: item['summary'] for method, item in value.items()}
                                 for role, value in reports.items()}, 'seconds': report['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
