"""Plan and acquire a separate open-set dataset without modifying prior artifacts.

Planning reads official annotations only. Acquisition requires an exact plan
hash matching the protocol lock written by the task owner.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import io
import queue
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/open-set-v2'
REPORTS=ROOT/'reports/open-set-v2'
KNOWN=['ΓΕΙΑ','ΕΥΧΑΡΙΣΤΩ','ΕΝΤΑΞΕΙ']
GROUPS={'train':[1,2,4,5],'val':[6],'final_test':[3,7]}
UNKNOWN={'train':['ΦΥΣΙΚΑ','ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ','ΟΚ','ΤΕΛΕΙΑ'],
         'val':['ΟΧΙ','ΚΑΛΟ'],
         'final_test':['ΝΑΙ','ΚΑΤΑΛΑΒΑΙΝΩ','ΗΜΕΡΑ_ΤΡΕΙΣ']}
ALIASES={'ΜΕΡΕΣ_ΤΡΕΙΣ':'ΗΜΕΡΑ_ΤΡΕΙΣ'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==99:raise
            time.sleep(0.1)

def load_locked():
    lock=json.loads((REPORTS/'protocol-lock.json').read_text(encoding='utf-8-sig'))
    actual=digest(DATA/'plan.locked.json')
    if actual!=lock['dataset_plan_sha256']:
        raise ValueError('Locked dataset plan hash does not match protocol-lock.json')
    plan=json.loads((DATA/'plan.locked.json').read_text(encoding='utf-8'))
    for source,name in [('GSL_split.zip','GSL_split.zip'),('supplementary.zip','supplementary.zip')]:
        assert digest(ROOT/'data/original_annotations'/source)==plan['source_sha256'][name]
    assert digest(ROOT/'data/manifest.json')==plan['source_sha256']['old_manifest.json']
    return plan,actual

def publish(plan,completed,plan_hash,transfers):
    base={k:v for k,v in plan.items() if k not in ['samples','status']}
    base.update({'status':'locked_acquisition','dataset_plan_sha256':plan_hash,
                 'updated_at_utc':datetime.now(timezone.utc).isoformat(),
                 'transfers':transfers,'unknown_classes':UNKNOWN})
    for filename,predicate in [('manifest.json',lambda r:True),
                               ('manifest.new.json',lambda r:not r['previously_seen']),
                               ('manifest.new.development.json',lambda r:not r['previously_seen'] and r['split']!='final_test'),
                               ('manifest.new.final.json',lambda r:not r['previously_seen'] and r['split']=='final_test')]:
        planned=[r for r in plan['samples'] if predicate(r)]
        ready=[completed[r['id']] for r in planned if r['id'] in completed]
        write(DATA/filename,{**base,'planned_samples':len(planned),'downloaded_samples':len(ready),
                             'complete':len(ready)==len(planned),'samples':ready})

def acquire_group(archive_name,samples,events):
    from dataset_fetch import get_archive
    from PIL import Image
    remote,archive=get_archive(archive_name)
    metadata_cache=list(remote.cache)
    grouped=collections.defaultdict(list)
    for member in archive.infolist():
        if member.filename.lower().endswith('.jpg'):
            grouped[str(Path(member.filename).parent).replace('\\','/')].append(member)
    for original in samples:
        sample=dict(original)
        frames=sorted(grouped[sample['source_path']],key=lambda r:r.filename)
        if not frames:raise ValueError('No RGB frames: '+sample['source_path'])
        directory=DATA/'raw'/sample['id']
        directory.mkdir(parents=True,exist_ok=True)
        missing=[r for r in frames if not (directory/Path(r.filename).name).exists()]
        if missing:
            start=min(r.header_offset for r in frames)
            end=max(r.header_offset+r.compress_size+1024 for r in frames)
            if end-start>16*1024*1024:raise ValueError('Unexpected clip larger than 16 MiB')
            remote.seek(start);remote.read(end-start)
        paths=[];hashes=[];numbers=[]
        for member in frames:
            target=directory/Path(member.filename).name
            if target.exists():
                content=target.read_bytes()
                if zlib.crc32(content)&0xffffffff!=member.CRC:
                    raise ValueError('Existing frame CRC mismatch: '+str(target))
            else:
                content=archive.read(member)
                if not content.startswith(b'\xff\xd8'):raise ValueError('Non-JPEG member')
                for attempt in range(100):
                    try:
                        target.write_bytes(content)
                        break
                    except PermissionError:
                        if attempt==99:raise
                        time.sleep(0.1)
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                if image.mode!='RGB' or image.size!=(648,480):
                    raise ValueError('Unexpected RGB frame dimensions/mode: '+str(target))
            paths.append(target.relative_to(ROOT).as_posix())
            hashes.append(hashlib.sha256(content).hexdigest())
            numbers.append(int(re.search(r'(\d+)\.jpg$',target.name).group(1)))
        assert numbers==list(range(len(numbers))),sample['id']
        sample.update({'frame_paths':paths,'frame_sha256':hashes,
                       'sha256':hashlib.sha256(''.join(hashes).encode('ascii')).hexdigest(),
                       'frame_count':len(frames),'duration_seconds':len(frames)/30,
                       'signer':sample['signer_id'],'label':sample['raw_label'] if sample['is_known'] else '__unknown__',
                       'lexical_validation':'original dataset annotation; no new expert validation',
                       'acquisition_status':'downloaded_crc_and_decode_verified'})
        remote.cache=list(metadata_cache)
        events.put(('sample',sample))
    archive.close()
    events.put(('transfer',{'archive':archive_name,'bytes_received':remote.bytes_received,
                            'requests':remote.requests,'clips':len(samples)}))

def acquire():
    plan,plan_hash=load_locked()
    completed={}
    transfers=[]
    if (DATA/'manifest.json').exists():
        previous=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        assert previous['dataset_plan_sha256']==plan_hash
        completed={r['id']:r for r in previous['samples']}
        transfers=previous.get('transfers',[])
    for original in plan['samples']:
        if original['previously_seen']:
            sample=dict(original)
            sample['signer']=sample['signer_id']
            sample['label']=sample['raw_label'] if sample['is_known'] else '__unknown__'
            sample['lexical_validation']='original dataset annotation; no new expert validation'
            completed[sample['id']]=sample
    publish(plan,completed,plan_hash,transfers)
    print(json.dumps({'event':'acquisition_started','completed':len(completed),'planned':len(plan['samples']),
                      'remaining':len(plan['samples'])-len(completed),'workers':3,'plan_sha256':plan_hash}),flush=True)
    for phase in ['development','final_test']:
        pending=[r for r in plan['samples'] if r['id'] not in completed and
                 ((r['split']=='final_test')==(phase=='final_test'))]
        grouped=collections.defaultdict(list)
        for row in pending:grouped[row['source_archive']].append(row)
        events=queue.Queue()
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(acquire_group,name,rows,events) for name,rows in sorted(grouped.items())]
            while not all(job.done() for job in jobs) or not events.empty():
                try:kind,value=events.get(timeout=1)
                except queue.Empty:continue
                if kind=='sample':completed[value['id']]=value
                else:transfers.append({'phase':phase,**value})
                publish(plan,completed,plan_hash,transfers)
                if kind=='transfer' or len(completed)%10==0:
                    print(json.dumps({'event':kind,'phase':phase,'completed':len(completed),
                                      'new_completed':sum(not r['previously_seen'] for r in completed.values()),
                                      'detail':value if kind=='transfer' else value['id']},ensure_ascii=True),flush=True)
            for job in jobs:job.result()
        print(json.dumps({'event':'phase_complete','phase':phase,'completed':len(completed)}),flush=True)
    verify()

def verify():
    from PIL import Image
    plan,plan_hash=load_locked()
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['dataset_plan_sha256']==plan_hash
    expected={r['id']:r for r in plan['samples']}
    assert len(manifest['samples'])==len(expected)==478
    old=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    old_ids={r['id'] for r in old['samples']}
    old_test_sessions={r['session'] for r in old['samples'] if r['split']=='test'}
    old_clips={r['sha256'] for r in old['samples']}
    old_frames={h for r in old['samples'] for h in r['frame_sha256']}
    clip_groups=collections.defaultdict(list);frame_groups=collections.defaultdict(set)
    sizes=collections.Counter();byte_count=0;fresh_old_frame_matches=[];new_old_frame_matches=[]
    for row in manifest['samples']:
        original=expected[row['id']]
        for field in ['source_path','raw_label','signer_id','split','is_known','class_id','scenario','session']:
            assert row[field]==original[field],(row['id'],field)
        assert row['signer_id'] in GROUPS[row['split']]
        assert len(row['frame_paths'])==len(row['frame_sha256'])==row['frame_count']>0
        hashes=[]
        for relative,expected_hash in zip(row['frame_paths'],row['frame_sha256']):
            path=(ROOT/relative).resolve();path.relative_to((ROOT/'data').resolve())
            content=path.read_bytes();actual=hashlib.sha256(content).hexdigest()
            assert actual==expected_hash,(row['id'],relative,'hash mismatch')
            hashes.append(actual);frame_groups[actual].add(row['split']);byte_count+=len(content)
            with Image.open(io.BytesIO(content)) as image:
                image.load();assert image.mode=='RGB' and image.size==(648,480)
                sizes[str(image.size)]+=1
            if row['split']=='final_test' and actual in old_frames:fresh_old_frame_matches.append(relative)
            if not row['previously_seen'] and actual in old_frames:new_old_frame_matches.append(relative)
        assert hashlib.sha256(''.join(hashes).encode('ascii')).hexdigest()==row['sha256']
        clip_groups[row['sha256']].append(row['id'])
        if row['split']=='final_test':
            assert row['id'] not in old_ids and row['sha256'] not in old_clips
    duplicate_clips=[ids for ids in clip_groups.values() if len(ids)>1]
    duplicate_split_frames={h:sorted(splits) for h,splits in frame_groups.items() if len(splits)>1}
    final_old_test_sessions=sorted({r['session'] for r in manifest['samples'] if r['split']=='final_test'} & old_test_sessions)
    assert not duplicate_clips and not duplicate_split_frames and not fresh_old_frame_matches and not new_old_frame_matches
    assert not final_old_test_sessions
    report={'verified':True,'dataset_plan_sha256':plan_hash,'samples':len(manifest['samples']),
            'new_samples':sum(not r['previously_seen'] for r in manifest['samples']),
            'reused_samples':sum(r['previously_seen'] for r in manifest['samples']),
            'split_counts':dict(collections.Counter(r['split'] for r in manifest['samples'])),
            'known_split_counts':dict(collections.Counter(r['split'] for r in manifest['samples'] if r['is_known'])),
            'unknown_split_counts':dict(collections.Counter(r['split'] for r in manifest['samples'] if not r['is_known'])),
            'decoded_jpeg_frames':sum(sizes.values()),'dimensions':dict(sizes),'raw_bytes':byte_count,
            'identical_full_clips':len(duplicate_clips),'identical_frames_across_splits':len(duplicate_split_frames),
            'final_frame_matches_any_old_243':len(fresh_old_frame_matches),
            'new_frame_matches_any_old_243':len(new_old_frame_matches),
            'fresh_final_clip_overlap_old_243':0,'hashes_recomputed':True,
            'final_session_identifiers_overlapping_old_test':final_old_test_sessions,
            'session_caveat':'Scenario/signer/repetition identifiers; not independently verified recording dates.',
            'new_rgb_frames_preserve_original_bytes':True,'signer_groups_disjoint':True,
            'unknown_categories_disjoint':True,'final_identity_caveat':plan['final_identity_caveat'],
            'transfers':manifest['transfers'],
            'transfer_accounting_note':'Per-archive counters omit unrecorded workers from the interrupted first run; see acquisition_notes.json. Raw disk bytes are fully verified.',
            'manifest_sha256':digest(DATA/'manifest.json'),
            'label_signer_counts':[{'split':split,'raw_label':label,'signer_id':signer,'count':count}
                 for (split,label,signer),count in sorted(collections.Counter((r['split'],r['raw_label'],r['signer_id']) for r in manifest['samples']).items())]}
    write(REPORTS/'dataset_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['label_signer_counts','transfers']},ensure_ascii=True),flush=True)

def inventory():
    old=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    seen={s['source_path']:s for s in old['samples']}
    zs=zipfile.ZipFile(ROOT/'data/original_annotations/GSL_split.zip')
    supplementary=zipfile.ZipFile(ROOT/'data/original_annotations/supplementary.zip')
    merged={}
    for line in supplementary.read('merged_continuous.csv').decode('utf-8-sig').splitlines():
        parts=line.split('|',2)
        if len(parts)==3:merged[parts[0]]=(parts[1],parts[2])
    rows=[];excluded=[]
    for official_split in ['train','val','test']:
        for line in zs.read(f'GSL_split/GSL_continuous/GSL-SD-{official_split}.txt').decode('utf-8-sig').splitlines():
            if not line:continue
            path,label=line.split('|',1)
            if len(label.split())!=1:continue
            text,raw=merged.get(path,('',''))
            cleaned_raw=raw.strip().rstrip(';').strip()
            alias=ALIASES.get(cleaned_raw)
            if cleaned_raw!=label and alias!=label:
                excluded.append({'source_path':path,'canonical_label':label,'raw_annotation':raw})
                continue
            signer=int(re.search(r'signer(\d+)',path).group(1))
            rows.append({'id':path.replace('/','__'),'source_path':path,'raw_label':label,
                         'annotation_raw_glosses':raw,'utterance_text':text,'signer_id':signer,
                         'scenario':path.split('_')[0],'session':path.split('/')[0],
                         'official_split':official_split,'previously_seen':path in seen,
                         'annotation_alias_applied':alias if cleaned_raw!=label else None,
                         'is_known':label in KNOWN,'class_id':KNOWN.index(label) if label in KNOWN else -1})
    assert len({r['source_path'] for r in rows})==len(rows)
    counts={}
    for label in sorted({r['raw_label'] for r in rows}):
        subset=[r for r in rows if r['raw_label']==label]
        counts[label]={'all_by_signer':dict(collections.Counter(r['signer_id'] for r in subset)),
                       'unused_by_signer':dict(collections.Counter(r['signer_id'] for r in subset if not r['previously_seen'])),
                       'by_scenario':{scenario:dict(collections.Counter(r['signer_id'] for r in subset if r['scenario']==scenario)) for scenario in sorted({r['scenario'] for r in subset})}}
    report={'source':'https://zenodo.org/records/4756317','old_seen_examples':len(seen),
            'eligible_single_gloss_examples':len(rows),'classes':counts,
            'exact_official_aliases':ALIASES,'excluded_annotation_disagreements':excluded,
            'alias_evidence':{raw:{'official_canonical_label':canonical,
                                   'matching_examples':sum(r['annotation_alias_applied']==canonical for r in rows),
                                   'source_split_archive':'GSL_split.zip / GSL_continuous / GSL-SD-*.txt',
                                   'source_raw_archive':'supplementary.zip / merged_continuous.csv',
                                   'sample_source_paths':[r['source_path'] for r in rows if r['annotation_alias_applied']==canonical][:3]}
                              for raw,canonical in ALIASES.items()},
            'known_final_available':{label:dict(collections.Counter(r['signer_id'] for r in rows if r['raw_label']==label and r['signer_id'] in GROUPS['final_test'] and not r['previously_seen'])) for label in KNOWN},
            'limitations':['Whole single-gloss utterances offer only nine well-covered unknown labels, not 10-20 development plus five final categories.',
                           'Previous test signers 3 and 7 have already been viewed; new final clips are unseen examples from previously viewed identities.',
                           'ΟΚ and ΕΝΤΑΞΕΙ may be semantically close; official labels remain separate and no project-specific expert equivalence review has occurred.'],
            'source_sha256':{'GSL_split.zip':digest(ROOT/'data/original_annotations/GSL_split.zip'),
                             'supplementary.zip':digest(ROOT/'data/original_annotations/supplementary.zip'),
                             'old_manifest.json':digest(ROOT/'data/manifest.json')}}
    write(REPORTS/'dataset_inventory.json',report)
    return rows,seen,report

def propose():
    rows,seen,report=inventory()
    selected=[]
    def add(row,split,reason):
        record=dict(row)
        record['split']=split
        record['selection_reason']=reason
        record['source_archive']=row['scenario']+'.zip'
        record['source_url']=f'https://zenodo.org/records/4756317/files/{row["scenario"]}.zip'
        record['fps']=30
        if row['previously_seen']:
            original=seen[row['source_path']]
            record['original_evaluation_split']=original['split']
            for key in ['frame_paths','frame_sha256','sha256','frame_count','duration_seconds']:
                record[key]=original[key]
            record['acquisition_status']='reuse_existing_verified_frames'
        else:
            record['frame_paths']=[]
            record['acquisition_status']='pending_lock_and_download'
        selected.append(record)
    def pick(split,label,count,scenarios,unused=True):
        for signer in GROUPS[split]:
            candidates=[r for r in rows if r['signer_id']==signer and r['raw_label']==label and r['scenario'] in scenarios and (not unused or not r['previously_seen'])]
            pools={s:sorted([r for r in candidates if r['scenario']==s],key=lambda r:r['source_path']) for s in scenarios}
            chosen=[]
            while any(pools.values()) and (count is None or len(chosen)<count):
                for s in scenarios:
                    if pools[s] and (count is None or len(chosen)<count):chosen.append(pools[s].pop(0))
            if count is not None and len(chosen)<count:raise ValueError(f'Insufficient {split}/{label}/{signer}: {len(chosen)}<{count}')
            for r in chosen:add(r,split,'unused; round-robin scenarios with lexical source-path order')
    for row in rows:
        old=seen.get(row['source_path'])
        if not old:continue
        if old['split'] in ['train','val'] and (row['is_known'] or row['raw_label'] in UNKNOWN[old['split']]):
            add(row,old['split'],'reuse previously seen development example; never fresh final')
    for split in ['train','val']:
        pick(split,'ΓΕΙΑ',None,['kep1'])
        pick(split,'ΕΥΧΑΡΙΣΤΩ',5,['health4','health5','kep2'])
        pick(split,'ΕΝΤΑΞΕΙ',5,['health4','health5','kep2'])
    pick('train','ΦΥΣΙΚΑ',5,['health4','health5','kep5'])
    pick('train','ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ',5,['police2'])
    pick('train','ΟΚ',5,['kep5'])
    pick('train','ΤΕΛΕΙΑ',5,['health5'])
    pick('val','ΟΧΙ',10,['health1','health2','health3'])
    pick('val','ΚΑΛΟ',15,['health3','health4','kep5'])
    pick('final_test','ΓΕΙΑ',None,['kep1'])
    pick('final_test','ΕΥΧΑΡΙΣΤΩ',15,['health4','health5','kep2'])
    pick('final_test','ΕΝΤΑΞΕΙ',20,['health4','health5','kep2','kep5'])
    pick('final_test','ΝΑΙ',15,['health1','health3','kep1'])
    pick('final_test','ΚΑΤΑΛΑΒΑΙΝΩ',5,['health4'])
    pick('final_test','ΗΜΕΡΑ_ΤΡΕΙΣ',5,['health2'])
    ids=[r['source_path'] for r in selected]
    assert len(ids)==len(set(ids))
    assert not any(r['previously_seen'] for r in selected if r['split']=='final_test')
    assert not any(r['signer_id'] in [3,7] for r in selected if r['split']!='final_test')
    assert all(not(set(UNKNOWN[a])&set(UNKNOWN[b])) for a,b in [('train','val'),('train','final_test'),('val','final_test')])
    summary={split:{'total':sum(r['split']==split for r in selected),
                    'known':sum(r['split']==split and r['is_known'] for r in selected),
                    'unknown':sum(r['split']==split and not r['is_known'] for r in selected),
                    'reuse':sum(r['split']==split and r['previously_seen'] for r in selected),
                    'new':sum(r['split']==split and not r['previously_seen'] for r in selected),
                    'labels':dict(collections.Counter(r['raw_label'] for r in selected if r['split']==split)),
                    'scenarios':dict(collections.Counter(r['scenario'] for r in selected if r['split']==split))} for split in GROUPS}
    plan={'schema_version':1,'status':'proposed_not_locked','dataset':'CERTH GSL',
          'dataset_version':'Zenodo 4756317 v2','source':'https://zenodo.org/records/4756317',
          'license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
          'subset_type':'whole utterances with exactly one officially annotated gloss',
          'not_official_isolated_benchmark':True,'classes':KNOWN,'split_signers':GROUPS,
          'unknown_categories':UNKNOWN,'exact_official_aliases':ALIASES,
          'selection_uses_model_results':False,'old_243_are_all_previously_seen':True,
          'old_test_policy':'diagnostic only; no old test clip is selected for training, calibration or new final evaluation',
          'final_identity_caveat':'New final examples/sessions from previously viewed test people 3 and 7, not unseen identities.',
          'counts':summary,'total_examples':len(selected),
          'new_examples':sum(not r['previously_seen'] for r in selected),
          'reused_examples':sum(r['previously_seen'] for r in selected),
          'required_archives':sorted({r['source_archive'] for r in selected if not r['previously_seen']}),
          'source_sha256':report['source_sha256'],'samples':selected}
    write(DATA/'plan.proposed.json',plan)
    write(REPORTS/'dataset_plan_summary.json',{k:v for k,v in plan.items() if k!='samples'})
    print(json.dumps({'plan':str(DATA/'plan.proposed.json'),'sha256':digest(DATA/'plan.proposed.json'),
                      'total_examples':len(selected),'new_examples':plan['new_examples'],'counts':summary,'required_archives':plan['required_archives']},ensure_ascii=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--acquire',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.acquire:acquire()
    elif args.verify:verify()
    elif args.inventory:
        _,_,r=inventory()
        print(json.dumps({'eligible_rows':r['eligible_single_gloss_examples'],'known_final_available':r['known_final_available']},ensure_ascii=True))
    else:propose()
