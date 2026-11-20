"""Read-only compatibility audit against prior features and independent Python."""
import gzip,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'ml'))
from preprocess import preprocess
from v4_tracking import measure_raw_tracking
out=R/'data/v7-recovery';archive=np.load(out/'val.npz',allow_pickle=False);b={k:archive[k] for k in archive.files};idx={str(s):i for i,s in enumerate(b['source_id'])}
archive=np.load(R/'data/v3/val.npz',allow_pickle=False);a={k:archive[k] for k in archive.files};oldidx={str(s):i for i,s in enumerate(a['source_id'])}
baseline=[float(np.max(np.abs(b['x_original'][j]-a['x'][oldidx[str(s)]]))) for j,s in enumerate(b['source_id'])]
assert len(baseline)==86 and max(baseline)==0
unknown=json.loads((R/'reports/v6/hand-recovery-unknown-results.json').read_text(encoding='utf-8'));unknown_cases=[]
for row in unknown['rows']:
 j=idx[row['source_id']]
 for stream in ['original','fused']:
  assert bool(b[f'guard_{stream}'][j])==row[f'{stream}Guard']['usable']
  old=row[f'{stream}Features']
  if old is not None:
   delta=float(np.max(np.abs(b[f'x_{stream}'][j]-np.asarray(old['x'],dtype=np.float32))));assert delta==0
   unknown_cases.append({'source_id':row['source_id'],'stream':stream,'maxAbsDifference':delta})
protocol=json.loads((out/'extraction-protocol.json').read_text(encoding='utf-8'));sid={s['id']:s['source_id'] for s in protocol['samples']}
v6=json.loads((R/'reports/v6/hand-recovery-results.json').read_text(encoding='utf-8'));known_cases=[]
for row in v6['rows']:
 if row['split']!='val' or row['control']!='none':continue
 j=idx[sid[row['id']]]
 for stream,key in [('original','original'),('fused','fused')]:
  frames=[f[key] for f in row['frames']];x,q=preprocess(frames);tracking=measure_raw_tracking(frames)
  delta=float(np.max(np.abs(b[f'x_{stream}'][j]-x)));assert delta==0
  assert bool(b[f'guard_{stream}'][j])==tracking['usable']
  assert abs(float(b[f'quality_{stream}'][j])-q)<1e-6
  known_cases.append({'source_id':sid[row['id']],'stream':stream,'maxAbsDifference':delta,'pythonGuard':tracking['usable']})
report={'passed':True,'originalV3Validation':{'cases':86,'maximumDifference':max(baseline)},'priorV6UnknownGuardPassingFeatures':unknown_cases,'independentPythonPriorV6Gamma':known_cases,
 'scope':'Existing exposed validation data only. Compatibility/reproducibility checks, no accuracy measurement or threshold choice.'}
if (out/'train.npz').exists():
 old=np.load(R/'data/v3/train.npz',allow_pickle=False);new=np.load(out/'train.npz',allow_pickle=False);ix={str(s):i for i,s in enumerate(old['source_id'])};old_x=old['x'];new_x=new['x_original'];sources=new['source_id']
 differences=[float(np.max(np.abs(new_x[j]-old_x[ix[str(s)]]))) for j,s in enumerate(sources)]
 assert len(differences)==338 and max(differences)<=1e-6
 report['originalV3Train']={'cases':338,'maximumDifference':max(differences),'bitwiseEqualCases':sum(v==0 for v in differences),'withinAbsoluteToleranceCases':sum(v<=1e-6 for v in differences),'absoluteTolerance':1e-6,'nonExact':[{'source_id':str(s),'maximumDifference':v} for s,v in zip(sources,differences) if v!=0]}
 report['scope']='Current338 train plus86 exposed validation only. Compatibility/reproducibility checks, no accuracy measurement or threshold choice.'
(out/'compatibility-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'baselineCases':86,'v6UnknownStreams':len(unknown_cases),'pythonGammaStreams':len(known_cases)}))
