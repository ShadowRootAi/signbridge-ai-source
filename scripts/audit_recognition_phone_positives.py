"""Audit documented GSLW whole-clip targets against the frozen V2 vocabulary."""

import csv
import json
from pathlib import Path
import re

from prepare_elementary_catalog_plan import word_key


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data/gslw-external-novel/official.csv'
MODEL = ROOT / 'artifacts/recognition-scale-v2/freeze.json'
OUTPUT = ROOT / 'reports/recognition-phone-adaptation/source-audit.json'


def key(value):
    return ' '.join(word_key(value).split()).rstrip(' ?!.,;;·…')


def gloss_key(value):
    # Conventional numbered variants are retained as leads for sign review.
    return key(re.sub(r'\(\d+\)$', '', value))


def main():
    from hashlib import sha256

    known = json.loads(MODEL.read_bytes())['classes']
    by_key = {key(label): label for label in known}
    matches = {label: {'exact_whole_greek': [], 'single_gloss_or_numbered_variant': [],
                       'longer_utterance_gloss_occurrences': 0} for label in known}
    seen = 0
    with CSV.open(encoding='utf-8-sig', newline='') as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            source, gloss, _, _, signer, _, text, _ = row
            seen += 1
            evidence = {'line': line, 'source_member': source, 'signer': signer, 'gloss': gloss, 'greek': text}
            label = by_key.get(key(text))
            if label is not None:
                matches[label]['exact_whole_greek'].append(evidence)
            tokens = gloss.split()
            for token in tokens:
                label = by_key.get(gloss_key(token))
                if label is None:
                    continue
                if len(tokens) == 1:
                    matches[label]['single_gloss_or_numbered_variant'].append(evidence)
                else:
                    matches[label]['longer_utterance_gloss_occurrences'] += 1
    result = {'source': 'https://zenodo.org/records/8047231', 'annotation_rows': seen,
              'annotation_sha256': sha256(CSV.read_bytes()).hexdigest(),
              'v2_freeze_sha256': sha256(MODEL.read_bytes()).hexdigest(), 'requested_words': matches,
              'scope': 'Text and gloss audit of complete official annotation CSV. A sign embedded in a longer utterance '
                       'needs independent token-level timing and sign review before it can be used as an isolated positive.'}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows': seen, 'whole_greek_matches': sum(len(row['exact_whole_greek']) for row in matches.values()),
                      'single_gloss_matches': sum(len(row['single_gloss_or_numbered_variant']) for row in matches.values()),
                      'longer_utterance_gloss_occurrences': sum(row['longer_utterance_gloss_occurrences'] for row in matches.values())}))


if __name__ == '__main__':
    main()
