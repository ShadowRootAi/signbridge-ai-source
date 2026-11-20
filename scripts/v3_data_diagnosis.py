"""V3 annotation diagnosis and explicitly locked development-only acquisition.

Diagnosis reads local files only; the separate acquisition command fetches just
the authorized selected RGB ranges. No training or relabeling is performed.
Contact-sheet thumbnails are declared visualization derivatives of real frames.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse
import hashlib
import json
import re
import zipfile
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v3'
PLAN=ROOT/'data/v3'
KNOWN=['ΓΕΙΑ','ΕΥΧΑΡΙΣΤΩ','ΕΝΤΑΞΕΙ']
GROUPS={'train':[1,2,4,5],'val':[6],'prior_final_people':[3,7]}
ALIAS={'ΜΕΡΕΣ_ΤΡΕΙΣ':'ΗΜΕΡΑ_ΤΡΕΙΣ'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temporary=path.with_name(path.name+'.tmp')
    for attempt in range(100):
        try:
            temporary.write_text(text,encoding='utf-8');temporary.replace(path);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.1)

def source_data():
    old=read(ROOT/'data/manifest.json');v2=read(ROOT/'data/open-set-v2/manifest.json')
    seen={s['source_path']:s for s in old['samples']+v2['samples']}
    supplement=zipfile.ZipFile(ROOT/'data/original_annotations/supplementary.zip')
    merged={}
    for line in supplement.read('merged_continuous.csv').decode('utf-8-sig').splitlines():
        pieces=line.split('|',2)
        if len(pieces)==3:merged[pieces[0]]={'translation':pieces[1],'raw_gloss':pieces[2]}
    splits={}
    archive=zipfile.ZipFile(ROOT/'data/original_annotations/GSL_split.zip')
    for split in ['train','val','test']:
        for number,line in enumerate(archive.read(f'GSL_split/GSL_continuous/GSL-SD-{split}.txt').decode('utf-8-sig').splitlines(),1):
            if not line:continue
            path,label=line.split('|',1)
            assert path not in splits
            splits[path]={'canonical':label,'official_file':f'GSL-SD-{split}.txt','official_line':number}
    return old,v2,seen,supplement,merged,splits

def inventory(context):
    old,v2,seen,supplement,merged,splits=context
    eligible=[];disagreements=[]
    for path,label in splits.items():
        if len(label['canonical'].split())!=1:continue
        raw=merged.get(path,{}).get('raw_gloss','').strip().rstrip(';').strip()
        if raw!=label['canonical'] and ALIAS.get(raw)!=label['canonical']:
            disagreements.append({'source_path':path,'canonical':label['canonical'],'raw':raw});continue
        signer=int(re.search(r'signer(\d+)',path).group(1))
        eligible.append({'source_path':path,'raw_label':label['canonical'],'annotation_raw_gloss':merged[path]['raw_gloss'],
                         'translation':merged[path]['translation'],'signer_id':signer,'scenario':path.split('_')[0],
                         'session':path.split('/')[0],'seen':path in seen,'official_split':label['official_file']})
    previously_seen_labels={s['raw_label'] for s in seen.values()}
    counts={}
    for label in sorted({s['raw_label'] for s in eligible}):
        all_rows=[s for s in eligible if s['raw_label']==label]
        fresh=[s for s in all_rows if not s['seen']]
        counts[label]={'total_eligible':len(all_rows),'previously_seen':len(all_rows)-len(fresh),'unused':len(fresh),
            'unused_by_person':{str(i):sum(s['signer_id']==i for s in fresh) for i in range(1,8)},
            'unused_by_role':{role:sum(s['signer_id'] in people for s in fresh) for role,people in GROUPS.items()},
            'unused_by_scenario':dict(Counter(s['scenario'] for s in fresh)),
            'category_previously_exposed':label in previously_seen_labels}
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'old_examples':len(old['samples']),
            'v2_examples':len(v2['samples']),'old_v2_overlap':len({s['source_path'] for s in old['samples']}&{s['source_path'] for s in v2['samples']}),
            'previously_seen_union':len(seen),'eligible_single_gloss_examples':len(eligible),
            'unused_eligible_examples':sum(not s['seen'] for s in eligible),'labels':counts,
            'never_exposed_eligible_labels':sorted(set(counts)-previously_seen_labels),
            'excluded_annotation_disagreements':disagreements,
            'three_known_class_fresh_final_possible_with_prior_person_split':all(counts[label]['unused_by_role']['prior_final_people']>0 for label in KNOWN),
            'limitations':['Unused means absent from the union of the original243 and V2/478, not new people or a new collection.',
              'All previously evaluated examples and final categories are now exposed. Reusing them in V3 requires an explicit diagnostic/repeated-evaluation label.',
              'No additional boundaries or semantic aliases were invented. Multi-gloss utterances are not counted as isolated signs.',
              'This inventory excludes unresolved official-cleaned/raw annotation disagreements.'],
            'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/manifest.json',ROOT/'data/open-set-v2/manifest.json',ROOT/'data/original_annotations/GSL_split.zip',ROOT/'data/original_annotations/supplementary.zip']}}
    write(OUT/'unused_data_inventory.json',report)
    write(PLAN/'unused_single_gloss_catalog.json',{'status':'inventory_only_not_a_locked_training_plan','source_hashes':report['source_hashes'],'samples':[s for s in eligible if not s['seen']]})
    print(json.dumps({'event':'inventory','seen_union':len(seen),'eligible':len(eligible),'unused':report['unused_eligible_examples'],
        'known':{label:counts[label] for label in KNOWN},'target_unknown':counts['ΚΑΤΑΛΑΒΑΙΝΩ'],
        'never_exposed_eligible_labels':{label:counts[label] for label in report['never_exposed_eligible_labels']}},ensure_ascii=True),flush=True)
    return report

def annotation_timestamps(supplement):
    result={};metadata={}
    for filename in supplement.namelist():
        if not filename.startswith('extracted_annotations_'):continue
        scenario=filename.removeprefix('extracted_annotations_').removesuffix('.csv')
        times_file='extracted_timestamps'+scenario+'.csv'
        rows=supplement.read(filename).decode('utf-8-sig').splitlines()
        times=supplement.read(times_file).decode('utf-8-sig').splitlines()
        if len(rows)!=len(times):
            metadata[scenario]={'annotation_rows':len(rows),'timestamp_rows':len(times),'positional_pairing_used':False};continue
        metadata[scenario]={'annotation_rows':len(rows),'timestamp_rows':len(times),'positional_pairing_used':True}
        for index,(line,time) in enumerate(zip(rows,times)):
            parts=line.split('|',2)
            if len(parts)!=3:continue
            vals=[float(v) for v in time.split(',')]
            if len(vals)!=6:continue
            neighbors=[]
            for j in [index-1,index+1]:
                if 0<=j<len(rows):
                    other=rows[j].split('|',2)
                    if len(other)==3 and other[0].split('/')[0]==parts[0].split('/')[0]:
                        neighbors.append({'source_path':other[0],'translation':other[1],'raw_gloss':other[2]})
            result[parts[0]]={'annotation_file':filename,'timestamp_file':times_file,'line_1based':index+1,
                'translation':parts[1],'raw_gloss':parts[2],'timestamp_raw':time,'start_ms':vals[0],'end_ms':vals[1],
                'frame_start_in_concatenated_sentence_sequence':int(vals[3]),'frame_end_in_concatenated_sentence_sequence':int(vals[4]),
                'implied_inclusive_frame_count':int(vals[4]-vals[3]+1),'adjacent_sentence_annotations':neighbors}
    return result,metadata

def frame_audit(sample,timestamps,merged,splits):
    source=sample['source_path'];paths=[ROOT/p for p in sample['frame_paths']]
    numbers=[];hashes=[];pixels=[];orientations=[]
    for p in paths:
        numbers.append(int(re.search(r'(\d+)\.jpg$',p.name).group(1)));hashes.append(sha(p))
        with Image.open(p) as image:
            image.load();assert image.size==(648,480) and image.mode=='RGB'
            orientations.append(image.getexif().get(274))
            pixels.append(np.asarray(image.convert('L'),dtype=np.float32))
    assert numbers==list(range(len(paths))) and hashes==sample['frame_sha256']
    assert hashlib.sha256(''.join(hashes).encode('ascii')).hexdigest()==sample['sha256']
    annotation=timestamps[source]
    assert annotation['implied_inclusive_frame_count']==len(paths)
    assert merged[source]['raw_gloss']==sample['annotation_raw_glosses']==annotation['raw_gloss']
    assert splits[source]['canonical']==sample['raw_label']
    assert merged[source]['translation']==sample['utterance_text']==annotation['translation']
    changes=[float(np.mean(np.abs(b-a))) for a,b in zip(pixels,pixels[1:])]
    return {'id':sample['id'],'source_path':source,'label':sample['raw_label'],'signer_id':sample['signer_id'],
            'session':sample['session'],'scenario':sample['scenario'],'official_cleaned_split':splits[source],
            'supplementary':annotation,'frame_count':len(paths),'fps':sample['fps'],
            'playback_duration_seconds':len(paths)/30,'first_last_observation_span_seconds':(len(paths)-1)/30,
            'source_timestamp_duration_seconds':(annotation['end_ms']-annotation['start_ms'])/1000,
            'source_timestamp_vs_playback_duration_error_seconds':abs((annotation['end_ms']-annotation['start_ms'])/1000-len(paths)/30),
            'frame_numbers_contiguous':True,'original_manifest_frame_hashes_match':True,'identical_adjacent_frame_hashes':sum(a==b for a,b in zip(hashes,hashes[1:])),
            'all_frames_decoded_rgb_648x480':True,'exif_orientation_values':list(dict.fromkeys(orientations)),
            'adjacent_gray_pixel_mean_absolute_differences_0_255':changes,
            'source_frame_sequence_sha256':sample['sha256'],
            'boundary_claim':'All supplied sentence-clip frames retained; matching official sentence timestamps/frame count. This does not independently validate a linguistically complete onset/offset.'}

def sheet(samples,title,path):
    thumb=(324,240);columns=5;left=18;top=64;row_height=306
    canvas=Image.new('RGB',(left*2+columns*thumb[0],top+row_height*len(samples)+52),'#f8fafc')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    big=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',25)
    draw.text((left,12),title,fill='#101828',font=big)
    mapping=[]
    for row,sample in enumerate(samples):
        n=sample['frame_count'];indices=[int(round(i*(n-1)/4)) for i in range(5)]
        y=top+row*row_height
        draw.text((left,y),sample['source_path']+' | '+sample['raw_label']+' | N='+str(n),fill='#101828',font=font)
        for column,index in enumerate(indices):
            p=ROOT/sample['frame_paths'][index]
            with Image.open(p) as original:image=original.resize(thumb,Image.Resampling.LANCZOS)
            x=left+column*thumb[0];canvas.paste(image,(x,y+28))
            draw.text((x+5,y+271),f'frame_{index:04d} | t={index/30:.3f}s',fill='#344054',font=small)
        mapping.append({'id':sample['id'],'source_path':sample['source_path'],'frame_indices':indices,
                        'source_frame_paths':[sample['frame_paths'][i] for i in indices]})
    draw.text((left,canvas.height-39),'Real source frames; thumbnails resized only. No mirroring, relabeling or synthetic frames.',fill='#344054',font=small)
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'dimensions':list(canvas.size),'rows':mapping,
            'transformation':'Original RGB frames resized to324x240 thumbnails and laid out with source/frame labels; no flips/crops/reordering.'}

def diagnose(context):
    old,v2,seen,supplement,merged,splits=context
    ts,ts_metadata=annotation_timestamps(supplement)
    selected=sorted([s for s in v2['samples'] if s['split']=='final_test' and s['raw_label']=='ΚΑΤΑΛΑΒΑΙΝΩ'],key=lambda s:s['source_path'])
    assert len(selected)==10
    matching=[]
    for sample in selected:
        rep=int(re.search(r'_rep(\d+)',sample['source_path']).group(1))
        candidates=[s for s in seen.values() if s['raw_label']=='ΓΕΙΑ' and s['signer_id']==sample['signer_id'] and s['scenario']=='kep4' and int(re.search(r'_rep(\d+)',s['source_path']).group(1))==rep]
        assert len(candidates)==1;matching.append(candidates[0])
    greetings=sorted([s for s in seen.values() if s['raw_label']=='ΓΕΙΑ' and s['signer_id'] in [3,7] and s['scenario']=='kep1'],key=lambda s:s['source_path'])
    audited=[frame_audit(s,ts,merged,splits) for s in selected+matching+greetings]
    alignment=[]
    for sample in seen.values():
        if sample['source_path'] in ts:
            alignment.append(sample['frame_count']==ts[sample['source_path']]['implied_inclusive_frame_count'])
    sheets=[]
    for person in [3,7]:
        for group,rows,heading in [('katalavaino',selected,'ΚΑΤΑΛΑΒΑΙΝΩ / official text: κατάλαβα'),('geia_farewell_matched',matching,'ΓΕΙΑ / matched person and repetition; KEP4'),('geia_greeting_context',greetings,'ΓΕΙΑ / additional greeting context; KEP1')]:
            subset=[s for s in rows if s['signer_id']==person]
            sheets.append(sheet(subset,f'Person {person}: {heading}',OUT/'contact-sheets'/f'{group}_signer{person}.png'))
    description=read(ROOT/'data/original_annotations/zenodo_metadata.json')['metadata']['description']
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'focus_count':10,'matched_geia_count':10,'additional_greeting_context_count':len(greetings),
        'matching_rule':'Same signer and repetition number; KEP4 for all primary comparisons. This is a different scenario, not a same-session pair. Extra KEP1 greeting examples show contextual variation.',
        'selection_uses_model_outcomes':False,'official_annotation_agreement_for_every_audited_clip':True,
        'timestamp_pairing':'Same-row pairing of each scenario annotation and timestamp CSV; equal line counts required; implied inclusive frame counts verified against acquired images.',
        'timestamp_scenario_counts':ts_metadata,'timestamp_count_matches_all_locally_acquired_eligible_rows':{'checked':len(alignment),'matched':sum(alignment)},
        'fps_evidence':{'declared_manifest_fps':30,'official_zenodo_metadata_mentions_30fps':'30 fps' in description,
            'original_sensor_resolution_in_metadata':'848x480','acquired_jpeg_resolution':'648x480',
            'resolution_caveat':'The source JPEGs already have648x480 pixels. Local acquisition did not resize them; original sensor and released-frame resolution differ.'},
        'mirror_evidence':{'local_source_frames_equal_previously_verified_original_hashes':True,
            'source_images_flipped_by_this_script':False,'contact_sheets_flipped':False,
            'upstream_capture_mirroring':'Not established by these files; no claim about anatomical handedness or linguistic mirror invariance.',
            'project_source_mirroring':'No mirroring is applied by dataset acquisition; source JPEG bytes are preserved. Browser UI/pipeline behavior is audited separately.'},
        'clips':audited,'contact_sheets':sheets,
        'linguistic_limit':'Original official research labels retained; no project-specific expert relabeling or semantic-equivalence claim.',
        'visual_inspection_status':'Contact sheets generated; observations recorded separately after actual image inspection.'}
    write(OUT/'annotation_boundary_audit.json',report)
    write(OUT/'contact-sheets/index.json',{'source':'https://zenodo.org/records/4756317','license':'CC-BY-4.0',
        'attribution':'See data/open-set-v2/ATTRIBUTION.md; thumbnails are resized visual derivatives of the original research images.',
        'sheets':sheets})
    print(json.dumps({'event':'diagnosis','audited_clips':len(audited),'focus':len(selected),'sheets':[s['path'] for s in sheets],
                      'timestamp_counts':report['timestamp_count_matches_all_locally_acquired_eligible_rows'],
                      'katalavaino_frame_counts':[s['frame_count'] for s in selected]},ensure_ascii=True),flush=True)

def lock_targeted_development():
    context=source_data();_,_,seen,supplement,merged,splits=context
    catalog=read(PLAN/'unused_single_gloss_catalog.json')
    ts,_=annotation_timestamps(supplement)
    chosen=[]
    for label in ['ΚΑΤΑΛΑΒΑΙΝΩ','ΗΜΕΡΑ_ΤΡΕΙΣ','ΝΑΙ']:
        for person in [1,2,4,5,6]:
            candidates=sorted([r for r in catalog['samples'] if r['raw_label']==label and r['signer_id']==person],key=lambda r:r['source_path'])
            if label!='ΝΑΙ':assert len(candidates)==5,(label,person,len(candidates))
            assert len(candidates)>=5
            for row in candidates[:5]:
                source=row['source_path'];assert source not in seen
                raw=merged[source]['raw_gloss'];canonical=splits[source]['canonical']
                assert canonical==label and (raw.strip().rstrip(';').strip()==label or ALIAS.get(raw.strip().rstrip(';').strip())==label)
                chosen.append({'id':source.replace('/','__'),'source_path':source,'raw_label':label,
                    'annotation_raw_glosses':raw,'utterance_text':merged[source]['translation'],
                    'annotation_source':splits[source]['official_file'],'annotation_line_1based':splits[source]['official_line'],
                    'annotation_alias_applied':label if raw.strip().rstrip(';').strip()!=label else None,
                    'class_id':-1,'label':'__unknown__','is_known':False,'signer_id':person,'signer':person,
                    'split':'val' if person==6 else 'train','scenario':row['scenario'],'session':row['session'],
                    'source_archive':row['scenario']+'.zip','source_url':f'https://zenodo.org/records/4756317/files/{row["scenario"]}.zip',
                    'fps':30,'expected_frame_count':ts[source]['implied_inclusive_frame_count'],
                    'official_timestamp_row':ts[source]['timestamp_raw'],'frame_paths':[],
                    'previously_seen':False,'category_exposed_in_v2_final':True,
                    'lexical_validation':'original official annotation; no new expert review',
                    'selection_rule':'first5 lexicographic source_path per person per label; all5 for KATALAVAINO/HMERA_TREIS'})
    assert len(chosen)==len({r['source_path'] for r in chosen})==75
    files=[ROOT/'data/manifest.json',ROOT/'data/open-set-v2/manifest.json',
           ROOT/'data/original_annotations/GSL_split.zip',ROOT/'data/original_annotations/supplementary.zip',
           PLAN/'unused_single_gloss_catalog.json']
    document={'schema_version':1,'status':'locked_authorized_targeted_development_only',
        'locked_at_utc':datetime.now(timezone.utc).isoformat(),'dataset':'CERTH GSL','dataset_version':'Zenodo4756317v2',
        'source':'https://zenodo.org/records/4756317','license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
        'classes':KNOWN,'split_signers':{'train':[1,2,4,5],'val':[6]},'planned_samples':75,
        'counts':{'train':60,'val':15},'unknown_labels':['ΚΑΤΑΛΑΒΑΙΝΩ','ΗΜΕΡΑ_ΤΡΕΙΣ','ΝΑΙ'],
        'adaptive_development':True,'category_exposure':'All3 categories were exposed in V2 final evaluation. These new clips are adaptive development data, not an unseen-class final test.',
        'final_test_samples':0,'people3_7_excluded':True,'training_not_authorized_by_this_script':True,
        'subset_type':'complete officially single-gloss sentence clips, not the official isolated benchmark',
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in files},'samples':chosen}
    path=PLAN/'targeted-development.plan.locked.json'
    if path.exists():
        existing=read(path);document['locked_at_utc']=existing['locked_at_utc'];assert existing==document,'Refusing to change existing targeted plan'
    else:write(path,document)
    digest=sha(path)
    write(OUT/'targeted-development-lock.json',{'locked_at_utc':document['locked_at_utc'],'plan_path':str(path.relative_to(ROOT)),
        'plan_sha256':digest,'selection_method':'Metadata only, exactly authorized3labels x5people x5clips',
        'created_before_network_acquisition':True,'adaptive_development':True})
    print(json.dumps({'event':'targeted_plan_locked','plan_sha256':digest,'clips':75,'train':60,'val':15,
        'archives':sorted({s['source_archive'] for s in chosen}),'expected_frames':sum(s['expected_frame_count'] for s in chosen)},ensure_ascii=True),flush=True)
    return document,digest

def acquire_targeted_development():
    # Reuse the tested bounded HTTP Range/CRC/decode helper with this run's output
    # directory. The imported V2 module and every old manifest stay unchanged.
    import openset_dataset as acquisition
    document=read(PLAN/'targeted-development.plan.locked.json')
    lock=read(OUT/'targeted-development-lock.json');plan_sha=sha(PLAN/'targeted-development.plan.locked.json')
    assert lock['plan_sha256']==plan_sha
    for path,expected in document['source_hashes'].items():assert sha(ROOT/path)==expected,path
    acquisition.DATA=PLAN
    manifest_path=PLAN/'manifest.json';completed={};transfers=[]
    if manifest_path.exists():
        previous=read(manifest_path);assert previous['dataset_plan_sha256']==plan_sha
        completed={s['id']:s for s in previous['samples']};transfers=previous.get('transfers',[])
    def publish():
        samples=[completed[s['id']] for s in document['samples'] if s['id'] in completed]
        write(manifest_path,{**{k:v for k,v in document.items() if k!='samples'},'dataset_plan_sha256':plan_sha,
            'complete':len(samples)==75,'downloaded_samples':len(samples),'samples':samples,'transfers':transfers})
    publish();events=queue.Queue();groups=defaultdict(list)
    for row in document['samples']:
        if row['id'] not in completed:groups[row['source_archive']].append(row)
    print(json.dumps({'event':'targeted_acquisition_started','ready':len(completed),'remaining':75-len(completed),'plan_sha256':plan_sha}),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(acquisition.acquire_group,name,rows,events) for name,rows in sorted(groups.items())]
        while not all(job.done() for job in jobs) or not events.empty():
            try:kind,row=events.get(timeout=1)
            except queue.Empty:continue
            if kind=='sample':
                assert row['frame_count']==row['expected_frame_count'];completed[row['id']]=row
            else:transfers.append(row)
            publish()
            if kind=='transfer' or len(completed)%10==0:
                print(json.dumps({'event':kind,'ready':len(completed),'detail':row if kind=='transfer' else row['id']},ensure_ascii=True),flush=True)
        for job in jobs:job.result()
    assert len(completed)==75
    old,v2,seen,supplement,merged,splits=source_data();timestamps,_=annotation_timestamps(supplement)
    all_old_frames={h for s in seen.values() for h in s['frame_sha256']}
    clip_hashes=set();frame_splits=defaultdict(set);checks=[];byte_count=0
    for sample in completed.values():
        assert sample['source_path'] not in seen and sample['signer_id'] in [1,2,4,5,6]
        assert sample['split']==('val' if sample['signer_id']==6 else 'train') and sample['class_id']==-1
        checks.append(frame_audit(sample,timestamps,merged,splits))
        assert sample['sha256'] not in clip_hashes;clip_hashes.add(sample['sha256'])
        for path,h in zip(sample['frame_paths'],sample['frame_sha256']):
            assert h not in all_old_frames;frame_splits[h].add(sample['split']);byte_count+=(ROOT/path).stat().st_size
    assert all(len(splits)==1 for splits in frame_splits.values())
    assert all(sha(ROOT/path)==expected for path,expected in document['source_hashes'].items())
    result={'verified':True,'plan_sha256':plan_sha,'manifest_sha256':sha(manifest_path),'samples':75,
        'split_counts':dict(Counter(s['split'] for s in completed.values())),
        'label_counts':dict(Counter(s['raw_label'] for s in completed.values())),
        'frames':sum(s['frame_count'] for s in completed.values()),'raw_bytes':byte_count,
        'all_original_frame_hashes_and_official_frame_counts_verified':True,
        'all_frames_rgb_648x480':True,'source_fps':30,'duplicate_full_clips':0,'cross_split_frame_duplicates':0,
        'new_frame_hash_matches_old547':0,'old_artifact_hashes_unchanged':True,
        'adaptive_development_not_fresh_final':True,'people3_7_included':False,
        'transfers':transfers,'clips':checks}
    write(OUT/'targeted-development-acquisition.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='clips'},ensure_ascii=True),flush=True)

def tracking_overlay():
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=read(ROOT/'data/open-set-v2/manifest.json')
    samples={s['id']:s for s in manifest['samples']}
    focal={s['id'] for s in manifest['samples'] if s['split']=='final_test' and s['raw_label'] in ['ΓΕΙΑ','ΚΑΤΑΛΑΒΑΙΝΩ']}
    retained={};inputs=[]
    for path in sorted((ROOT/'data/open-set-v2/features').glob('final-*.jsonl')):
        inputs.append(path)
        for line in path.open(encoding='utf-8-sig'):
            if not line.strip():continue
            record=json.loads(line)
            if record['id'] in focal and record.get('raw'):
                retained[record['id']]={'record':record,'jsonl':path}
    ordered=sorted(retained,key=lambda identity:(['ΓΕΙΑ','ΚΑΤΑΛΑΒΑΙΝΩ'].index(samples[identity]['raw_label']),samples[identity]['source_path']))
    input_hashes_before={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    assert len(ordered)==6
    persons=sorted({samples[i]['signer_id'] for i in ordered})
    assert persons==[3]
    edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
           (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    colors={'left':'#00f0ff','right':'#ff8a00'}
    fig,axes=plt.subplots(2,3,figsize=(14.6,10.4),dpi=180)
    rows=[]
    def draw_landmarks(axis,frame,width,height):
        for hand,color in colors.items():
            points=frame.get(hand,[])
            if len(points)!=21:continue
            xy=np.asarray([[p['x']*width,p['y']*height] for p in points])
            for first,last in edges:axis.plot(xy[[first,last],0],xy[[first,last],1],color=color,linewidth=.75,alpha=.95)
            axis.scatter(xy[:,0],xy[:,1],s=7,c=color,edgecolors='#101010',linewidths=.25,zorder=4)
            axis.scatter(xy[[0,8],0],xy[[0,8],1],s=16,facecolors='none',edgecolors='white',linewidths=.8,zorder=5)
    for ax,identity in zip(axes.ravel(),ordered):
        sample=samples[identity];item=retained[identity];record=item['record'];raw=record['raw'];index=len(raw)//2
        assert len(raw)==sample['frame_count']==record['frames']
        frame=raw[index];source=ROOT/sample['frame_paths'][index]
        assert sha(source)==sample['frame_sha256'][index]
        with Image.open(source) as opened:rgb=np.asarray(opened.convert('RGB'))
        height,width=rgb.shape[:2]
        assert (width,height)==(frame['width'],frame['height'])==(648,480)
        assert abs(frame['t']-index*1000/30)<1e-6
        ax.imshow(rgb,origin='upper');draw_landmarks(ax,frame,width,height)
        ax.set_xlim(0,width);ax.set_ylim(height,0);ax.set_xticks([]);ax.set_yticks([])
        rep=int(re.search(r'_rep(\d+)',identity).group(1))
        ax.set_title(f"{sample['raw_label']} | person3 rep{rep} | frame{index:04d} | {frame['t']/1000:.3f}s",fontsize=10,pad=7)
        ax.set_xlabel(sample['source_path'],fontsize=7.5,labelpad=4)
        # A declared detail view follows the reported active/right-hand bounds.
        # The full unflipped source image above remains visible for context.
        hand=frame['right'];detail=None
        if len(hand)==21:
            xy=np.asarray([[p['x']*width,p['y']*height] for p in hand])
            xmin=max(0,float(xy[:,0].min()-20));xmax=min(width,float(xy[:,0].max()+20))
            ymin=max(0,float(xy[:,1].min()-20));ymax=min(height,float(xy[:,1].max()+20))
            inset=ax.inset_axes([.015,.015,.35,.35])
            inset.imshow(rgb,origin='upper');draw_landmarks(inset,frame,width,height)
            inset.set_xlim(xmin,xmax);inset.set_ylim(ymax,ymin);inset.set_xticks([]);inset.set_yticks([])
            for spine in inset.spines.values():spine.set_color('#ffffff');spine.set_linewidth(1)
            detail={'x_min':xmin,'x_max':xmax,'y_min':ymin,'y_max':ymax,'basis':'reported right-hand bounding box plus20pixels; source image not mirrored'}
        else:
            ax.text(.025,.055,'Reported right hand: MISSING',transform=ax.transAxes,color='#ffb54d',fontsize=9,
                    bbox={'facecolor':'#101010','alpha':.85,'edgecolor':'none','pad':5})
        def valid(points,count):
            return len(points)>=count and all(np.isfinite(p['x']) and np.isfinite(p['y']) for p in points[:count])
        quality_flags=[valid(f['pose'],25) and f['pose'][11].get('visibility',1)>=.5 and f['pose'][12].get('visibility',1)>=.5
                       and (valid(f['left'],21) or valid(f['right'],21)) for f in raw]
        quality_recomputed=float(np.mean(quality_flags));assert abs(quality_recomputed-record['quality'])<1e-12
        row={'id':identity,'source_path':sample['source_path'],'label':sample['raw_label'],'person':sample['signer_id'],
             'source_jsonl':item['jsonl'].relative_to(ROOT).as_posix(),'source_jsonl_sha256':sha(item['jsonl']),
             'source_image':source.relative_to(ROOT).as_posix(),'source_image_sha256':sha(source),
             'displayed_frame_index':index,'raw_frame_timestamp_ms':frame['t'],'raw_frame_count':len(raw),
             'clip_quality':record['quality'],'displayed_hand_landmark_counts':{hand:len(frame.get(hand,[])) for hand in colors},
             'quality_recomputed_from_actual_presence_formula':quality_recomputed,
             'missing_hand_frame_indices':{hand:[i for i,f in enumerate(raw) if len(f.get(hand,[]))!=21] for hand in colors},
             'coordinate_mapping':'x_pixel=raw_x*648; y_pixel=raw_y*480; image origin upper-left; no flip/rotation',
             'raw_hand_coordinates_normalized':{hand:frame[hand] for hand in colors},'detail_view_pixels':detail}
        rows.append(row)
    fig.suptitle('Actual RGB + retained raw hand landmarks\nPerson3 only: all6 focal raw fixtures; middle frame per clip',fontsize=15,y=.985)
    fig.text(.5,.025,'Cyan = reported left | Orange = reported right | White rings = wrist/index tip\nAll6 clips have quality=1: pose + EITHER hand suffices, so an active-hand dropout can be hidden. Person7 raw was not retained.',ha='center',va='center',fontsize=9)
    fig.subplots_adjust(top=.90,bottom=.09,left=.025,right=.985,wspace=.09,hspace=.18)
    output=OUT/'tracking-overlay.png';fig.savefig(output,dpi=180,facecolor='white');plt.close(fig)
    assert input_hashes_before=={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'figure':output.relative_to(ROOT).as_posix(),
        'figure_sha256':sha(output),'selection':'All six retained focal raw fixtures; chronological middle index. No outcome-based selection.',
        'labels':['ΓΕΙΑ','ΚΑΤΑΛΑΒΑΙΝΩ'],'people_available':persons,'person7_retained_focal_raw_available':False,
        'no_new_extraction_or_model_inference':True,'original_images_and_jsonl_unchanged':True,
        'input_hashes':input_hashes_before,'panels':rows,
        'quality_definition':'mean over raw frames of poseValid AND(leftValid OR rightValid); shoulders visibility>=0.5 and finite required points. Both hands are not required.',
        'limitations':['Only person3 is covered by retained focal raw data; no claim about person7 tracking accuracy.',
            'Reported left/right names are retained from the detector, not anatomically relabeled.',
            'Overlay position can reveal gross visible displacement but there are no manual ground-truth landmark coordinates or per-joint accuracy scores.',
            'The clip quality value is an aggregate presence measure, not proof of anatomically correct finger joints.',
            'The raw hand visibility fields can be zero even when landmarks exist; they are not interpreted as calibrated landmark confidence here.',
            'Middle-frame snapshots and detail insets cannot establish full-sequence tracking accuracy or the cause of model confusion.'],
        'inspection_status':'Generated; observations recorded after viewing the actual output.'}
    write(OUT/'tracking-overlay.json',report)
    print(json.dumps({'event':'tracking_overlay','figure':str(output),'panels':len(rows),'persons':persons,
        'missing_hands':{r['id']:r['missing_hand_frame_indices'] for r in rows}},ensure_ascii=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--lock-targeted-development',action='store_true');parser.add_argument('--acquire-targeted-development',action='store_true')
    parser.add_argument('--tracking-overlay',action='store_true')
    args=parser.parse_args()
    if args.tracking_overlay:tracking_overlay()
    elif args.lock_targeted_development:lock_targeted_development()
    elif args.acquire_targeted_development:acquire_targeted_development()
    else:
        context=source_data();inventory(context)
        if not args.inventory_only:diagnose(context)
