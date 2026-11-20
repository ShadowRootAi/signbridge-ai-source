"""Verify the downloaded pilot data, hashes, signer split and decoded RGB frames."""
from pathlib import Path
import collections
import hashlib
import json
import re
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'], 'Acquisition has not completed'
    samples=manifest['samples']
    assert len(samples)==manifest['planned_samples']
    person_sets={name:set(people) for name,people in manifest['split_signers'].items()}
    assert not (person_sets['train'] & person_sets['val'] or person_sets['train'] & person_sets['test'] or person_sets['val'] & person_sets['test'])
    all_clip_hashes=set()
    frame_splits=collections.defaultdict(set)
    dimensions=collections.Counter()
    byte_count=0
    rows=collections.Counter()
    for sample in samples:
        assert sample['signer'] in person_sets[sample['split']]
        assert len(sample['frame_paths'])==sample['frame_count']
        assert sample['frame_paths'], sample['id']
        hashes=[]
        numbers=[]
        for relative,expected in zip(sample['frame_paths'],sample['frame_sha256']):
            path=(ROOT/relative).resolve()
            path.relative_to((ROOT/'data/raw').resolve())
            content=path.read_bytes()
            digest=hashlib.sha256(content).hexdigest()
            assert digest==expected,(sample['id'],relative,'hash mismatch')
            hashes.append(digest)
            frame_splits[digest].add(sample['split'])
            byte_count+=len(content)
            with Image.open(path) as image:
                image.load()
                assert image.mode=='RGB',(relative,image.mode)
                dimensions[str(image.size)]+=1
            numbers.append(int(re.search(r'(\d+)\.jpg$',path.name).group(1)))
        assert numbers==list(range(len(numbers))),(sample['id'],'noncontiguous frames',numbers)
        clip_hash=hashlib.sha256(''.join(hashes).encode('ascii')).hexdigest()
        assert clip_hash==sample['sha256']
        assert clip_hash not in all_clip_hashes,'duplicate clip'
        all_clip_hashes.add(clip_hash)
        rows[(sample['split'],sample['raw_label'],sample['signer'])]+=1
    duplicates={digest:sorted(splits) for digest,splits in frame_splits.items() if len(splits)>1}
    assert not duplicates,'Identical frames in different splits'
    summary={'verified':True,'samples':len(samples),'decoded_jpeg_frames':sum(dimensions.values()),
             'dimensions':dict(dimensions),'raw_bytes':byte_count,'signer_overlap':False,
             'identical_full_clips':0,'identical_frames_across_splits':0,
             'frame_numbering':'zero-based contiguous for every clip','hashes_recomputed':True,
             'label_signer_counts':[{'split':split,'raw_label':label,'signer':signer,'count':count} for (split,label,signer),count in sorted(rows.items())]}
    (ROOT/'reports/dataset_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='label_signer_counts'},ensure_ascii=True))

if __name__=='__main__':main()
