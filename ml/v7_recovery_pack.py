"""Pack verified atomic V7 records; never omit a failed-guard example."""
import gzip,json,hashlib,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];out=R/'data/v7-recovery';raw=(out/'extraction-protocol.json').read_bytes();p=json.loads(raw);ph=hashlib.sha256(raw).hexdigest()
report={'protocolSha256':ph,'pipelineSha256':p['pipelineSha256'],'splits':{}}
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['val','train','both'],default='both');args=parser.parse_args()
for split,n in [('val',86),('train',338)]:
 if args.split!='both' and args.split!=split:continue
 samples=[s for s in p['samples'] if s['split']==split];assert len(samples)==n;rows=[];hashes={}
 for s in samples:
  f=out/'records'/f"{s['id']}.json.gz";r=json.loads(gzip.decompress(f.read_bytes()));assert r['protocolSha256']==ph and r['pipelineSha256']==p['pipelineSha256'] and r['sourceContentSha256']==s['sourceContentSha256'] and r['preservationPassed'];assert all(r[k]==s[k] for k in ['id','source_id','split','y','signer']);rows.append(r);hashes[s['id']]=hashlib.sha256(f.read_bytes()).hexdigest()
 arrays={}
 for stream in ['original','fused']:
  x=np.asarray([r[f'{stream}Features']['x'] for r in rows],dtype=np.float32);assert x.shape==(n,48,224) and np.isfinite(x).all()
  arrays[f'x_{stream}']=x;arrays[f'quality_{stream}']=np.asarray([r[f'{stream}Features']['quality'] for r in rows],dtype=np.float32)
  assert np.isfinite(arrays[f'quality_{stream}']).all()
  arrays[f'guard_{stream}']=np.asarray([r[f'{stream}Guard']['usable'] for r in rows],dtype=np.bool_)
 arrays.update(y=np.asarray([r['y'] for r in rows],dtype=np.int64),signer=np.asarray([r['signer'] for r in rows],dtype=np.int64),
  source_id=np.asarray([r['source_id'] for r in rows]),raw_label=np.asarray([r['raw_label'] for r in rows]),split=np.asarray([split]*n),classes=np.load(R/f'data/v3/{split}.npz',allow_pickle=False)['classes'],
  frame_count=np.asarray([r['frameCount'] for r in rows],dtype=np.int32),recovered_frames=np.asarray([r['recoveredFrames'] for r in rows],dtype=np.int32),
  metadata_json=np.asarray(json.dumps({'protocolSha256':ph,'pipelineSha256':p['pipelineSha256'],'recordSha256':hashes,'guardFailuresRetained':True})))
 dest=out/f'{split}.npz';np.savez_compressed(dest,**arrays)
 report['splits'][split]={'clips':n,'frames':int(arrays['frame_count'].sum()),'recoveredFrames':int(arrays['recovered_frames'].sum()),'originalGuardPass':int(arrays['guard_original'].sum()),'fusedGuardPass':int(arrays['guard_fused'].sum()),'npzSha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
(out/'pack-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
