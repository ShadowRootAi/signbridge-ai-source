"""Lock all424 development RGB sequences, validate bytes, preserve V6 fallback."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];out=R/'data/v7-recovery';out.mkdir(parents=True,exist_ok=True);(out/'records').mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
v6_bytes=(R/'reports/v6/hand-recovery-protocol.json').read_bytes();v6=json.loads(v6_bytes)
index={}
for p in ['data/open-set-v2/manifest.json','data/v3/manifest.json']:
 for s in json.loads((R/p).read_text(encoding='utf-8'))['samples']:index[s['source_path']]=s
samples=[];npz_hashes={}
for split,n in [('val',86),('train',338)]:
 path=R/f'data/v3/{split}.npz';npz_hashes[split]=sha(path.read_bytes());a=np.load(path,allow_pickle=False);assert len(a['source_id'])==n
 for source_id,y,person,label in sorted(zip(a['source_id'],a['y'],a['signer'],a['raw_label']),key=lambda t:str(t[0])):
  sid=str(source_id);s=index[sid];assert s['split']==split and int(s['signer_id'])==int(person) and int(person) not in (3,7)
  hashes=[sha((R/p).read_bytes()) for p in s['frame_paths']]
  assert hashes==s['frame_sha256'],f'RGB byte hash mismatch: {sid}'
  samples.append({'id':s['id'],'source_id':sid,'split':split,'signer':int(person),'y':int(y),'raw_label':str(label),
   'paths':s['frame_paths'],'sourceSha256':s['sha256'],'sourceContentSha256':sha(('\n'.join(hashes)).encode()),'frameSha256':hashes})
files=['web/src/v6_hand_recovery.worker.ts','web/src/v6_hand_detector.worker.ts','web/src/features.ts','web/src/tracking.ts','web/v6_hand_recovery.html','web/scripts/v7_recovery_extract.mjs','web/package-lock.json','web/public/models/holistic_landmarker.task',v6['model']['path']]
file_hashes={p:sha((R/p).read_bytes()) for p in files}
pipeline_hash=sha(json.dumps(file_hashes,sort_keys=True,separators=(',',':')).encode())
protocol={'schemaVersion':1,'version':'v7-full-development-recovery-1','scope':'Exactly338 train and86 val from V3 development arrays; no person3/7, old129 or final-test data.',
 'v6ProtocolSha256':sha(v6_bytes),'sourceNpzSha256':npz_hashes,'pipelineSha256':pipeline_hash,'sourceFilesSha256':file_hashes,
 'rule':v6['rule'],'model':v6['model'],'sampling':'All original JPEG frames in metadata order, timestamps i*1000/30. Maximum39 frames in this collection. No additional subsampling, mirroring or transforms.',
 'output':'Both original/fused features for every clip, even when exact V4 raw guard fails. Guards and diagnostics are independent outputs; no classifier.',
 'sampleCount':len(samples),'frameCount':sum(len(s['paths']) for s in samples),'samples':samples}
assert len(samples)==424 and len({s['id'] for s in samples})==424
payload=json.dumps(protocol,ensure_ascii=False,indent=2).encode();dest=out/'extraction-protocol.json'
if dest.exists():assert dest.read_bytes()==payload,'Existing V7 protocol differs; do not silently invalidate a locked experiment.'
else:dest.write_bytes(payload)
print(json.dumps({'samples':len(samples),'frames':protocol['frameCount'],'pipelineSha256':pipeline_hash,'protocolSha256':sha(payload)}))
