"""Replay train-only Greek decoders with annotated and frozen CTC glosses."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unicodedata

from train_continuous import edit_counts, reference_lookup
from train_greek_translation_pilot import certh_sets, contextual_sets, greek, sha


CTC = Path('reports/continuous-contrastive/context-matched-comparison.json')
REPORT = Path('reports/greek-translation-pilot-context/gloss-oracle.json')
METHODS = ('exact_lookup', 'nearest_lookup', 'literal_boundary_deletion')


def literal_anchor(text):
    # Uppercase CERTH glosses commonly omit Greek word stress.
    return unicodedata.normalize('NFD', greek(text)).replace('\u0301', '')


def decode(glosses, lookup, method):
    if not glosses:
        return '', 'empty_input'
    if glosses in lookup:
        return lookup[glosses], 'exact_lookup'
    if method == 'exact_lookup':
        return '', 'unseen_order'
    if method == 'nearest_lookup':
        neighbors = [(sum(edit_counts(source, glosses)), text) for source, text in lookup.items()]
        best_distance = min(distance for distance, _ in neighbors)
        candidates = {text for distance, text in neighbors if distance == best_distance}
    else:
        candidates = set()
        for source, text in lookup.items():
            if len(source) != len(glosses) + 1:
                continue
            words = text.split()
            if len(words) < 2:
                continue
            if source[1:] == glosses and literal_anchor(source[0]) == literal_anchor(words[0]):
                candidates.add(' '.join(words[1:]))
            if source[:-1] == glosses and literal_anchor(source[-1]) == literal_anchor(words[-1]):
                candidates.add(' '.join(words[:-1]))
    if len(candidates) == 1:
        return next(iter(candidates)), method
    return '', 'conflicting_candidates' if candidates else 'no_supported_edit'


def evaluate(rows, lookup, method, input_key, score_text):
    outputs = []
    for row in rows:
        query = tuple(row[input_key])
        text, reason = decode(query, lookup, method)
        reference = greek(row['reference_text']) if score_text else ''
        errors = sum(edit_counts(reference.split(), text.split())) if score_text else None
        outputs.append({'id': row['id'], 'glosses': query, 'reference': reference or None,
                        'prediction': text or None, 'decision': reason, 'word_edits': errors})
    emitted = sum(row['prediction'] is not None for row in outputs)
    summary = {'clips': len(outputs), 'emitted': emitted, 'abstained': len(outputs) - emitted,
               'decisions': dict(Counter(row['decision'] for row in outputs))}
    if score_text:
        exact = sum(row['prediction'] == row['reference'] for row in outputs)
        summary.update(exact=exact, non_exact_emitted=emitted - exact,
                       word_edits=sum(row['word_edits'] for row in outputs),
                       reference_words=sum(len(row['reference'].split()) for row in outputs),
                       unique_reference_phrases=len({row['reference'] for row in outputs}))
        summary['word_error_rate_counting_abstention'] = summary['word_edits'] / summary['reference_words']
    return {'summary': summary, 'rows': outputs}


def main():
    base = certh_sets()
    context = contextual_sets()
    train = base['train'] + context['train']
    lookup = reference_lookup(train)
    if len(train) != 144 or len({row.id for row in train}) != len(train):
        raise ValueError('Expected the frozen 144 unique CERTH training recordings')
    if {row.glosses for row in train} & {row.glosses for row in context['test']}:
        raise ValueError('The novel context gloss order is already in training')

    frozen = json.loads(CTC.read_text(encoding='utf-8'))
    names = frozen['ranked_on_existing_dev_only']
    arms = {'overall_dev_first': names[0], 'expanded_dev_first': next(name for name in names if name.startswith('expanded_'))}
    expanded = frozen['arms'][arms['expanded_dev_first']]
    actual = {row['id']: (tuple(row['reference_glosses']), greek(row['reference_text']))
              for row in expanded['context_novel']['rows']}
    if actual != {row.id: (row.glosses, row.text) for row in context['test']}:
        raise ValueError('The previously scored held IDs or references changed')
    groups = {'original_development_gold': ([{'id': row.id, 'reference_glosses': row.glosses, 'reference_text': row.text}
                                             for row in base['val']], 'reference_glosses', True)}
    for cohort in ('context_development_diagnostic', 'context_novel', 'compact_known', 'compact_novel'):
        groups[cohort + '_gold'] = (expanded[cohort]['rows'], 'reference_glosses', True)
        for tag, name in arms.items():
            groups[cohort + '_' + tag] = (frozen['arms'][name][cohort]['rows'], 'predicted_glosses', True)
    groups['outside_scope_expanded_dev_first'] = (expanded['archived_outside_scope']['rows'], 'predicted_glosses', False)

    report = {
        'scope': 'Retrospective diagnosis of previously viewed CERTH clips. Gold glosses are human annotation. '
                 'Predicted glosses are frozen CTC outputs. These text baselines do not use visual confidence gates.',
        'comparison': 'Use training references only. Exact lookup abstains on unseen orders. Nearest lookup abstains '
                      'on conflicting ties. The boundary rule permits a single removal when a literal gloss anchors the '
                      'removed Greek word at the same sentence edge, ignoring stress, and every proposal agrees. The known near-template '
                      'failure motivated this rule, so its retrospective result requires new validation.',
        'ctc_report_sha256': sha(CTC),
        'frozen_arms': arms,
        'training': {'clips': len(train), 'unambiguous_orders': len(lookup), 'signers': sorted({row.signer for row in train})},
        'cohorts': {group: {method: evaluate(rows, lookup, method, input_key, score_text) for method in METHODS}
                    for group, (rows, input_key, score_text) in groups.items()},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for group in ('original_development_gold', 'context_novel_gold', 'context_novel_expanded_dev_first',
                  'compact_novel_expanded_dev_first', 'outside_scope_expanded_dev_first'):
        print(group, {key: result['summary'] for key, result in report['cohorts'][group].items()})


if __name__ == '__main__':
    main()
