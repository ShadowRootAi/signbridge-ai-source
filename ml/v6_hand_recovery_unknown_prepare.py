"""Separate bounded transfer plan: same locked fallback, all45 person6 OTHER."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];out=R/'reports/v6'
base_bytes=(out/'hand-recovery-protocol.json').read_bytes();base=json.loads(base_bytes)
val_path=R/'data/v3/val.npz';a=np.load(val_path,allow_pickle=False)
ids=[str(s) for s,y,p in zip(a['source_id'],a['y'],a['signer']) if y<0 and str(p)=='6']
assert len(ids)==45
index={}
for filename in ['data/open-set-v2/manifest.json','data/v3/manifest.json']:
 for sample in json.loads((R/filename).read_text(encoding='utf-8'))['samples']:
  if sample['source_path'] in ids:index[sample['source_path']]=sample
assert set(index)==set(ids)
samples=[]
for sid in sorted(ids):
 s=index[sid];assert s['split']=='val' and s['signer_id']==6 and not s['is_known']
 samples.append({'id':s['id'],'source_id':sid,'label':s['label'],'split':'val','signer':6,'paths':s['frame_paths'],'sourceSha256':s['sha256']})
p={'version':'v6-dedicated-hand-unknown-transfer-1','scope':'All45 real person6 OTHER clips in data/v3/val.npz, real/original pixels only. Same previously frozen recovery rule, no parameter changes, classifier or old129 access.',
 'parentProtocolSha256':hashlib.sha256(base_bytes).hexdigest(),'sourceNpzSha256':hashlib.sha256(val_path.read_bytes()).hexdigest(),
 'rule':base['rule'],'model':base['model'],'sampling':'Every original JPEG frame in manifest order at30fps. No transform, crop preprocessing except the locked fallback ROI, temporal trim, or repetition.',
 'samples':samples,'output':'Exact original/fused raw guard and x/quality only for streams whose exact raw guard passes. Full arrays transient in browser, not retained in this transfer output.'}
dest=out/'hand-recovery-unknown-protocol.json';dest.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'samples':len(samples),'frames':sum(len(s['paths']) for s in samples),'maxFrames':max(len(s['paths']) for s in samples),'protocolSha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))
