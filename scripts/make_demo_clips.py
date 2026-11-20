"""Encode eight preselected held-out utterances as local H.264 browser fixtures.

The first manifest sample of every known/unknown label for test signers 3 and 7
is used. Selection does not depend on model predictions or correctness.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/clips'

NOTICE='''SignBridge AI local browser test fixtures

Source: The Greek Sign Language (GSL) Dataset, version 2.
https://doi.org/10.5281/zenodo.4756317
https://zenodo.org/records/4756317

Creators: Ilias Papastratis; Nikolas Adaloglou; Andreas Stergioulas;
Theocharis Chatzis; Georgios Th. Papadopoulos; Vassia Zacharopoulou;
George J. Xydopoulos; Klimnis Atzakas; Dimitris Papazachariou;
Kosmas Dimitropoulos; Petros Daras.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

Changes: selected entire officially single-gloss utterances; original RGB
JPEG frame sequences transcoded to lossy H.264 MP4, yuv420p, CRF 18, 30 fps.
Original 648 x 480 resolution, frame count and chronological order retained.
No cropping, added frames, repeated playback, audio or resizing.
The videos are local fixtures; this script does not publish or upload them.

The files are a custom subset of continuous utterance data, not the official
isolated-sign benchmark. Original source mappings and hashes are in mapping.json.
These are test signers and must not be used for training or tuning thresholds.

Dataset annotations provide the ground truth. No additional expert linguistic
review has been performed for SignBridge AI. No endorsement by dataset creators
is implied. The application only supports the declared small vocabulary.
'''

def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete']
    chosen={}
    for sample in manifest['samples']:
        if sample['split']!='test' or sample['signer'] not in (3,7):continue
        chosen.setdefault((sample['signer'],sample['raw_label']),sample)
    assert len(chosen)==8,len(chosen)
    OUT.mkdir(parents=True,exist_ok=True)
    executable=imageio_ffmpeg.get_ffmpeg_exe()
    clips={}
    for sample in chosen.values():
        paths=[ROOT/p for p in sample['frame_paths']]
        assert all(p.parent==paths[0].parent for p in paths)
        assert [p.name for p in paths]==[f'frame_{n:04d}.jpg' for n in range(len(paths))]
        output=OUT/(sample['id']+'.mp4')
        args=[executable,'-hide_banner','-loglevel','error','-y',
              '-framerate','30','-start_number','0','-i',str(paths[0].parent/'frame_%04d.jpg'),
              '-frames:v',str(len(paths)),'-c:v','libx264','-preset','medium','-crf','18',
              '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)]
        subprocess.run(args,check=True,capture_output=True)
        reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24')
        metadata=next(reader)
        decoded_frames=sum(1 for _ in reader)
        assert decoded_frames==len(paths),(sample['id'],decoded_frames,len(paths))
        assert tuple(metadata['size'])==(648,480),metadata
        assert abs(metadata['fps']-30)<0.01,metadata
        expected_duration=len(paths)/30
        assert abs(metadata['duration']-expected_duration)<0.05,metadata
        clip={
            'path':output.relative_to(ROOT).as_posix(),
            'raw_label':sample['raw_label'],
            'true_class':sample['label'],
            'class_id':sample['class_id'],
            'is_known':sample['is_known'],
            'split':sample['split'],
            'signer_id':sample['signer'],
            'utterance_text':sample['utterance_text'],
            'annotation_raw_glosses':sample['annotation_raw_glosses'],
            'source_path':sample['source_path'],
            'source_archive':sample['source_archive'],
            'source_url':sample['source_url'],
            'source_frame_sequence_sha256':sample['sha256'],
            'mp4_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'bytes':output.stat().st_size,
            'codec':metadata['codec'],'pixel_format':'yuv420p',
            'width':648,'height':480,'fps':30,'frame_count':decoded_frames,
            'duration_seconds':expected_duration,
            'verified_decodable':True,
        }
        clips[sample['id']]=clip
        print(json.dumps({'id':sample['id'],'frames':decoded_frames,'bytes':clip['bytes'],'codec':clip['codec']},ensure_ascii=True),flush=True)
    mapping={
        'schema_version':1,'selection':'First manifest sample of each label for test signers 3 and 7; independent of predictions',
        'license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
        'dataset_source':'https://zenodo.org/records/4756317',
        'transformation':'JPEG sequence to lossy H.264/yuv420p MP4 at 30fps, CRF18; no resizing, trimming or added frames',
        'linguistic_validation':'Original official annotations only; no new expert review',
        'clips':clips,
    }
    (OUT/'mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'NOTICE.txt').write_text(NOTICE,encoding='utf-8')
    print(json.dumps({'clips':len(clips),'total_bytes':sum(c['bytes'] for c in clips.values()),'all_decoded':True}))

if __name__=='__main__':main()
