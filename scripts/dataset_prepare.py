"""Create a provenance-preserving pilot subset of officially single-gloss utterances.

This is not the official GSL isolated benchmark. Entire sentence clips whose
official annotation is one gloss are selected, without inventing sign boundaries.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import zipfile
from dataset_fetch import get_archive, ROOT, DATA, REPORTS

KNOWN = ['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ']
SIGNERS = {'train':[1,2,4,5], 'val':[6], 'test':[3,7]}
UNKNOWN = {'train':'ΦΥΣΙΚΑ', 'val':'ΟΧΙ', 'test':'ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ'}
SOURCES = {'ΓΕΙΑ':{'kep4','police2'}, 'ΕΥΧΑΡΙΣΤΩ':{'kep4','police1'},
           'ΕΝΤΑΞΕΙ':{'kep4','police1'}, 'ΦΥΣΙΚΑ':{'kep4'},
           'ΟΧΙ':{'police2'}, 'ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ':{'police2'}}

def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def plan():
    split_zip = zipfile.ZipFile(DATA/'original_annotations/GSL_split.zip')
    supplementary = zipfile.ZipFile(DATA/'original_annotations/supplementary.zip')
    merged = {}
    for line in supplementary.read('merged_continuous.csv').decode('utf-8-sig').splitlines():
        if len(line.split('|',2)) != 3:
            continue # An unrelated police4 row contains an embedded line break.
        path, text, glosses = line.split('|',2)
        merged[path] = (text,glosses)
    gloss_names = split_zip.read('GSL_split/GSL_isolated/iso_classes.csv').decode('utf-8-sig').splitlines()
    selected = []
    excluded = []
    for original_split in ['train','val','test']:
        content = split_zip.read(f'GSL_split/GSL_continuous/GSL-SD-{original_split}.txt').decode('utf-8-sig')
        for line in content.splitlines():
            if not line: continue
            path, label = line.split('|',1)
            if label not in SOURCES: continue
            scenario=path.split('_')[0]
            if scenario not in SOURCES[label]: continue
            signer=int(re.search(r'signer(\d+)',path).group(1))
            split=next(s for s,people in SIGNERS.items() if signer in people)
            is_known=label in KNOWN
            if not is_known and UNKNOWN[split] != label: continue
            text,raw_glosses=merged[path]
            if raw_glosses.rstrip(';') != label:
                excluded.append({'path':path,'cleaned':label,'raw':raw_glosses})
                continue
            sample_id=path.replace('/','__')
            selected.append({'id':sample_id,'raw_label':label,'label':label if is_known else '__unknown__',
                 'class_id':KNOWN.index(label) if is_known else -1,
                 'gloss_id':gloss_names.index(label),'gloss_id_basis':'zero-based line of official iso_classes.csv',
                 'signer':signer,'signer_id':signer,'split':split,'is_known':is_known,
                 'scenario':scenario,'session':path.split('/')[0],
                 'source_path':path,'source_archive':scenario+'.zip',
                 'source_url':f'https://zenodo.org/records/4756317/files/{scenario}.zip',
                 'original_split':original_split,'annotation_source':'GSL-SD-'+original_split+'.txt',
                 'utterance_text':text,'annotation_raw_glosses':raw_glosses,
                 'fps':30,'lexical_validation':'original dataset annotation; no new expert validation',
                 'frame_paths':[],'status':'pending'})
    selected.sort(key=lambda s:(['police1','kep4','police2'].index(s['scenario']),s['signer'],s['source_path']))
    assert len({s['id'] for s in selected}) == len(selected)
    manifest={'schema_version':1,'dataset':'CERTH GSL','dataset_version':'Zenodo 4756317 v2',
              'subset_type':'whole utterances with exactly one officially annotated gloss',
              'not_official_isolated_benchmark':True,'license':'CC-BY-4.0',
              'license_url':'https://creativecommons.org/licenses/by/4.0/',
              'source':'https://zenodo.org/records/4756317','classes':KNOWN,
              'split_signers':SIGNERS,'unknown_classes':UNKNOWN,
              'planned_samples':len(selected),'complete':False,'exclusions':excluded,'samples':selected}
    write_json(DATA/'manifest.pending.json',manifest)
    print(json.dumps({'planned_samples':len(selected),'counts':dict(collections.Counter(s['split'] for s in selected)),
                      'labels':dict(collections.Counter(s['raw_label'] for s in selected)),'excluded':excluded},ensure_ascii=True),flush=True)
    return manifest

def acquire(manifest,limit=None):
    completed=[]
    transfers=[]
    raw_root=DATA/'raw'
    raw_root.mkdir(exist_ok=True)
    for scenario in ['police1','kep4','police2']:
        samples=[s for s in manifest['samples'] if s['scenario']==scenario]
        if not samples: continue
        remote,archive=get_archive(scenario+'.zip')
        metadata_cache=list(remote.cache)
        grouped=collections.defaultdict(list)
        for info in archive.infolist():
            if info.filename.lower().endswith('.jpg'):
                grouped[str(Path(info.filename).parent).replace('\\','/')].append(info)
        for sample in samples:
            frames=sorted(grouped[sample['source_path']],key=lambda i:i.filename)
            if not frames:raise ValueError('No RGB frames for '+sample['source_path'])
            sample_dir=raw_root/sample['id']
            sample_dir.mkdir(exist_ok=True)
            missing=[i for i in frames if not (sample_dir/Path(i.filename).name).exists()]
            if missing:
                start=min(i.header_offset for i in frames)
                end=max(i.header_offset+i.compress_size+1024 for i in frames)
                if end-start > 16*1024*1024:raise ValueError('Unexpected large clip span')
                remote.seek(start)
                remote.read(end-start)
            frame_hashes=[]
            for info in frames:
                target=sample_dir/Path(info.filename).name
                if not target.exists():
                    content=archive.read(info) # zipfile validates the original CRC32.
                    if not content.startswith(b'\xff\xd8'):raise ValueError('Non-JPEG '+info.filename)
                    target.write_bytes(content)
                content=target.read_bytes()
                frame_hashes.append(hashlib.sha256(content).hexdigest())
            sample['frame_paths']=[(sample_dir/Path(i.filename).name).relative_to(ROOT).as_posix() for i in frames]
            sample['frame_sha256']=frame_hashes
            sample['sha256']=hashlib.sha256(''.join(frame_hashes).encode('ascii')).hexdigest()
            sample['frame_count']=len(frames)
            sample['duration_seconds']=len(frames)/30
            sample['status']='downloaded_crc_verified'
            completed.append(sample)
            progress={**manifest,'samples':completed,'downloaded_samples':len(completed)}
            write_json(DATA/'manifest.json',progress)
            if len(completed)==1 or len(completed)%10==0:
                print(json.dumps({'downloaded_samples':len(completed),'id':sample['id'],
                                  'frames':len(frames),'first_path':sample['frame_paths'][0]},ensure_ascii=True),flush=True)
            remote.cache=list(metadata_cache)
            if limit and len(completed)>=limit:return
        transfers.append({'archive':scenario+'.zip','bytes_received':remote.bytes_received,'requests':remote.requests})
    duplicate_hashes=collections.defaultdict(list)
    for sample in completed:duplicate_hashes[sample['sha256']].append(sample['id'])
    duplicates=[v for v in duplicate_hashes.values() if len(v)>1]
    if duplicates:raise ValueError('Duplicate full-clip hashes: '+repr(duplicates))
    manifest['samples']=completed
    manifest['downloaded_samples']=len(completed)
    manifest['complete']=True
    manifest['transfers']=transfers
    manifest['duplicate_clip_hashes']=duplicates
    write_json(DATA/'manifest.json',manifest)
    report={'samples':len(completed),'frames':sum(s['frame_count'] for s in completed),
            'split_counts':dict(collections.Counter(s['split'] for s in completed)),
            'known_split_counts':dict(collections.Counter(s['split'] for s in completed if s['is_known'])),
            'unknown_split_counts':dict(collections.Counter(s['split'] for s in completed if not s['is_known'])),
            'signers':SIGNERS,'classes':KNOWN,'unknown_classes':UNKNOWN,'transfers':transfers,
            'duplicate_clip_hashes':duplicates,'excluded_annotation_mismatches':manifest['exclusions']}
    write_json(REPORTS/'dataset_subset.json',report)
    print(json.dumps(report,ensure_ascii=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    manifest=plan()
    if not args.plan_only:acquire(manifest,args.limit)
