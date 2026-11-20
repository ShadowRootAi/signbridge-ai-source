"""Encode 12 metadata-selected final RGB clips as local browser fixtures.

Selection is persisted before encoding and never reads model predictions.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import imageio_ffmpeg
from make_demo_clips import NOTICE

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/open-set-v2'
OUT=DATA/'clips'
REPORTS=ROOT/'reports/open-set-v2'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete']
    labels=manifest['classes']+manifest['unknown_categories']['final_test']
    assert len(labels)==6 and len(set(labels))==6
    chosen=[]
    for label in labels:
        for signer in [3,7]:
            candidates=[s for s in manifest['samples'] if s['split']=='final_test' and
                        s['signer_id']==signer and s['raw_label']==label]
            sample=min(candidates,key=lambda s:s['source_path'])
            assert not sample['previously_seen']
            chosen.append(sample)
    assert len({s['id'] for s in chosen})==12
    selection={'schema_version':1,
               'selection':'Lexicographically first source_path for each of the six final labels and each final signer 3 and 7',
               'selection_inputs':'Dataset metadata only; no predictions, features, logits, acceptance or model outcomes read',
               'created_at_utc':datetime.now(timezone.utc).isoformat(),
               'dataset_manifest_sha256':digest(DATA/'manifest.json'),
               'dataset_plan_sha256':manifest['dataset_plan_sha256'],
               'samples':[{'id':s['id'],'source_path':s['source_path'],'raw_label':s['raw_label'],
                           'signer_id':s['signer_id'],'is_known':s['is_known'],
                           'frame_count':s['frame_count'],'source_frame_sequence_sha256':s['sha256']}
                          for s in chosen]}
    selection_path=REPORTS/'browserfixture-selection.json'
    if selection_path.exists():
        previous=json.loads(selection_path.read_text(encoding='utf-8'))
        selection['created_at_utc']=previous['created_at_utc']
        assert previous==selection,'Existing fixture selection differs; refusing to replace prespecification'
    else:
        selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    selection_hash=digest(selection_path)
    (REPORTS/'browserfixture-selection.sha256').write_text(selection_hash+'\n',encoding='ascii')
    print(json.dumps({'event':'selection_saved_before_encoding','selection_sha256':selection_hash,
                      'clips':len(chosen),'created_at_utc':selection['created_at_utc']}),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    executable=imageio_ffmpeg.get_ffmpeg_exe()
    clips={}
    for sample in chosen:
        paths=[ROOT/p for p in sample['frame_paths']]
        assert all(p.parent==paths[0].parent for p in paths)
        assert [p.name for p in paths]==[f'frame_{n:04d}.jpg' for n in range(len(paths))]
        source_hashes=[digest(p) for p in paths]
        assert source_hashes==sample['frame_sha256']
        assert hashlib.sha256(''.join(source_hashes).encode('ascii')).hexdigest()==sample['sha256']
        output=OUT/(sample['id']+'.mp4')
        command=[executable,'-hide_banner','-loglevel','error','-y','-framerate','30',
                 '-start_number','0','-i',str(paths[0].parent/'frame_%04d.jpg'),
                 '-frames:v',str(len(paths)),'-c:v','libx264','-preset','medium','-crf','18',
                 '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)]
        subprocess.run(command,check=True,capture_output=True)
        reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24')
        metadata=next(reader)
        decoded=sum(1 for _ in reader)
        assert decoded==len(paths),(sample['id'],decoded,len(paths))
        assert tuple(metadata['size'])==(648,480) and abs(metadata['fps']-30)<0.01,metadata
        assert metadata['codec']=='h264',metadata
        assert abs(metadata['duration']-len(paths)/30)<0.05,metadata
        clip={'path':output.relative_to(ROOT).as_posix(),'raw_label':sample['raw_label'],
              'true_class':sample['label'],'class_id':sample['class_id'],'is_known':sample['is_known'],
              'split':sample['split'],'signer_id':sample['signer_id'],
              'utterance_text':sample['utterance_text'],'annotation_raw_glosses':sample['annotation_raw_glosses'],
              'source_path':sample['source_path'],'source_archive':sample['source_archive'],
              'source_url':sample['source_url'],'source_frame_sequence_sha256':sample['sha256'],
              'source_frame_sha256':source_hashes,'mp4_sha256':digest(output),'bytes':output.stat().st_size,
              'codec':metadata['codec'],'pixel_format':'yuv420p','width':648,'height':480,'fps':30,
              'frame_count':decoded,'duration_seconds':len(paths)/30,'verified_decodable':True}
        clips[sample['id']]=clip
        print(json.dumps({'event':'encoded_verified','id':sample['id'],'frames':decoded,
                          'bytes':clip['bytes']},ensure_ascii=True),flush=True)
    mapping={'schema_version':1,'selection':selection['selection'],
             'selection_report':'reports/open-set-v2/browserfixture-selection.json',
             'selection_sha256':selection_hash,'selection_before_encoding':True,
             'dataset_manifest_sha256':selection['dataset_manifest_sha256'],
             'license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
             'dataset_source':'https://zenodo.org/records/4756317',
             'transformation':'JPEG sequence to lossy H.264/yuv420p MP4 at 30fps, CRF18; no resizing, trimming or added frames',
             'linguistic_validation':'Original official annotations only; no new expert review',
             'final_identity_caveat':manifest['final_identity_caveat'],'clips':clips}
    (OUT/'mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'NOTICE.txt').write_text(NOTICE+'\nOpen-set fixtures: 12 metadata-selected new final examples from previously viewed people 3 and 7. The prespecified selection is hashed in mapping.json. Do not use these clips for fitting or threshold tuning.\n',encoding='utf-8')
    print(json.dumps({'event':'complete','clips':len(clips),'total_bytes':sum(c['bytes'] for c in clips.values()),
                      'all_decoded':True,'mapping_sha256':digest(OUT/'mapping.json')}),flush=True)

if __name__=='__main__':main()
