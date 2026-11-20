"""Tracking summary only for fixed45-OTHER transfer; no semantic scoring."""
import json,hashlib,statistics,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];out=R/'reports/v6'
result_path=out/'hand-recovery-unknown-results.json';r=json.loads(result_path.read_text(encoding='utf-8'))
assert r['completed'] and len(r['rows'])==45
a=np.load(R/'data/v3/val.npz',allow_pickle=False);label_by_source={str(s):str(v) for s,v in zip(a['source_id'],a['raw_label'])}
rows=r['rows'];groups=collections.defaultdict(list)
for row in rows:
 assert row['preservationPassed']
 for key,guard in [('originalFeatures','originalGuard'),('fusedFeatures','fusedGuard')]:
  f=row[key];assert (f is not None)==row[guard]['usable']
  if f is not None:assert np.asarray(f['x']).shape==(48,224) and np.isfinite(f['x']).all() and np.isfinite(f['quality'])
 groups[label_by_source[row['source_id']]].append(row)
def summarize(rr):
 times=[t for x in rr for t in x['frameTiming']];calls=[t['handMs'] for t in times if t['handMs']>0]
 return {'clips':len(rr),'frames':sum(x['frameCount'] for x in rr),'recoveredFrames':sum(x['recoveredFrames'] for x in rr),
  'fallbackCalls':sum(x['fallbackCalls'] for x in rr),'guardPassBefore':sum(x['originalGuard']['usable'] for x in rr),
  'guardPassAfter':sum(x['fusedGuard']['usable'] for x in rr),'newGuardPassingClips':sum(not x['originalGuard']['usable'] and x['fusedGuard']['usable'] for x in rr),
  'fallbackCallMsMedian':statistics.median(calls) if calls else None,'totalFrameMsMedian':statistics.median(t['frameMs'] for t in times),
  'clipWallMsMedianIncludingWorkers':statistics.median(x['totalMs'] for x in rr)}
s={'completed':True,'protocolSha256':r['protocolSha256'],'resultsSha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
 'timingScope':{'fallbackCallMs':'Child-worker message roundtrip and dedicated palm/hand detection, excluding model initialization.','totalFrameMs':'RGB fetch/decode, Holistic and optional crop/fallback inside primary worker.','clipWallMsMedianIncludingWorkers':'Legacy field name: primary-worker job processing interval including both task initialization and child-worker startup, excluding primary-worker bootstrap, result serialization and browser orchestration. Not full UI latency.'},
 'browserVersion':r['browserVersion'],'total':summarize(rows),'byRawLabel':{key:summarize(value) for key,value in groups.items()},
 'sourceLabels':'Joined by exact NPZ source_id from data/v3/val.npz.raw_label; manifest label may be __unknown__. Frozen protocol/results left unchanged.',
 'featuresChecked':True,'originalLandmarksPreserved':True,'noClassifier':True,
 'interpretation':'More OTHER clips passing the completeness guard is not recognition success. Frozen semantic classifiers must be evaluated separately to measure any additional false acceptance.',
 'limitations':['Same person6 exposed validation data; not independent new-user evaluation.','Only original RGB in this transfer run; no human webcam or natural non-sign controls.','No extra parameter fitting or changes to locked ROI/handedness/geometry rule.']}
(out/'hand-recovery-unknown-tracking-summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(s['total']))
