"""Retrieve official GSL metadata and inspect its original annotation manifests."""
from pathlib import Path
import collections
import hashlib
import json
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
BASE = 'https://zenodo.org/records/4756317/files/'

def fetch(url, target):
    if not target.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            target.write_bytes(response.read())
    return target.read_bytes()

def decode(raw):
    try:
        return raw.decode('utf-8-sig'), 'utf-8-sig'
    except UnicodeDecodeError:
        return raw.decode('cp1253'), 'cp1253'

def main():
    originals = DATA / 'original_annotations'
    originals.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch('https://zenodo.org/api/records/4756317', originals / 'zenodo_metadata.json'))
    for name in ['GSL_split.zip', 'supplementary.zip']:
        fetch(BASE + name + '?download=1', originals / name)
    archive = zipfile.ZipFile(originals / 'GSL_split.zip')
    classes = collections.defaultdict(lambda: collections.Counter())
    records = []
    encodings = {}
    for split in ['train', 'val', 'test']:
        name = f'GSL_split/GSL_isolated/{split}_greek_iso.csv'
        content, encoding = decode(archive.read(name))
        encodings[name] = encoding
        for line in content.splitlines():
            if not line.strip():
                continue
            source, gloss = line.split('|', 1)
            signer_match = re.search(r'signer(\d+)', source)
            signer = int(signer_match.group(1)) if signer_match else None
            records.append({'source_path':source, 'gloss':gloss, 'signer':signer, 'official_split':split})
            classes[gloss][signer] += 1
    (DATA / 'gsl_isolated_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'source':'https://zenodo.org/records/4756317', 'license':metadata['metadata'].get('license'),
              'encodings':encodings,'count':len(records),'class_count':len(classes),
              'classes':{k:dict(v) for k,v in sorted(classes.items())},
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in originals.iterdir()}}
    source_counts=collections.Counter(row['source_path'] for row in records)
    report.update({'published_isolated_count':40785,'observed_isolated_manifest_rows':len(records),
                   'observed_isolated_unique_paths':len(source_counts),
                   'observed_isolated_split_rows':dict(collections.Counter(row['official_split'] for row in records)),
                   'isolated_duplicate_source_paths':sum(n>1 for n in source_counts.values()),
                   'note':'Pilot uses single-gloss whole utterances from continuous annotation manifests, not the isolated benchmark.'})
    (REPORTS / 'dataset_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'count':len(records),'class_count':len(classes),'encodings':encodings,'license':report['license'],
                      'split_rows':report['observed_isolated_split_rows'],
                      'sample':records[:2],'classes_with_all_seven_signers':sum(len(v)==7 for v in classes.values())},ensure_ascii=True))

if __name__ == '__main__':
    main()
