"""Tracking-only report and inspectable overlays for every recovered frame."""
from pathlib import Path
import json,hashlib,statistics,math,collections
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];out=R/'reports/v6'
p=json.loads((out/'hand-recovery-protocol.json').read_text(encoding='utf-8'))
r=json.loads((out/'hand-recovery-results.json').read_text(encoding='utf-8'))
assert r.get('completed') and len(r['rows'])==42 and r['preflightPassed']
samples={s['id']:s for s in p['samples']};rows=r['rows']
preserved=True
for row in rows:
 for f in row['frames']:
  for k in ['t','width','height','left','pose','face']:
   assert f['original'][k]==f['fused'][k],(row['id'],k)
  if f['original']['right']:
   assert f['original']['right']==f['fused']['right'] and not f['recovered']
aggregates=[]
for split in ['train','val']:
 for control in p['conditions']:
  rr=[x for x in rows if x['split']==split and x['control']==control];ff=[f for x in rr for f in x['frames']]
  attempts=[f for f in ff if f['handMs']>0]
  candidates=[c for f in ff for c in f['candidates']]
  a={'split':split,'control':control,'clips':len(rr),'frames':len(ff),
    'originalRightPresent':sum(len(f['original']['right'])>=21 for f in ff),
    'fusedRightPresent':sum(len(f['fused']['right'])>=21 for f in ff),
    'recoveredFrames':sum(f['recovered'] for f in ff),'fallbackCalls':len(attempts),
    'exactV4GuardPassBefore':sum(x['originalGuard']['usable'] for x in rr),
    'exactV4GuardPassAfter':sum(x['fusedGuard']['usable'] for x in rr),
    'candidateCount':len(candidates),'candidateRejectionReasons':dict(collections.Counter(reason for c in candidates for reason in c['reasons'])),
    'holisticMsMedian':statistics.median(f['holisticMs'] for f in ff),
    'fallbackCallMsMedian':statistics.median(f['handMs'] for f in attempts) if attempts else None,
    'totalFrameMsMedian':statistics.median(f['frameMs'] for f in ff),
    'clipWallMsMedianIncludingTwoWorkers':statistics.median(x['totalMs'] for x in rr),
    'initMsMedian':statistics.median(x['initMs'] for x in rr)}
  aggregates.append(a)
comparisons=[{'id':x['id'],'split':x['split'],'frames':len(x['frames']),
 'originalRightPresent':sum(bool(f['original']['right']) for f in x['frames']),
 'fusedRightPresent':sum(bool(f['fused']['right']) for f in x['frames']),
 'originalGuard':x['originalGuard'],'fusedGuard':x['fusedGuard']} for x in rows if x['control']=='none']
v5=json.loads((R/'reports/v5/tracking-results.json').read_text(encoding='utf-8'))
baseline_diffs=[]
for old in v5['rows']:
 if old['control']!='none' or old['variant']!='original':continue
 new=next(x for x in rows if x['id']==old['id'] and x['control']=='none');delta=0.
 for a,b in zip(old['raw'],new['frames']):
  for k in ['left','right','pose']:
   assert len(a[k])==len(b['original'][k])
   for u,v in zip(a[k],b['original'][k]):
    delta=max(delta,*[abs(u[c]-v[c]) for c in ['x','y','z']])
 baseline_diffs.append({'id':old['id'],'maximumCoordinateDifference':delta})

connections=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
overlay_dir=out/'hand-recovery-overlays';overlay_dir.mkdir(exist_ok=True)
overlay_files=[];overview=[]
for row in rows:
 indices=[i for i,f in enumerate(row['frames']) if f['recovered']]
 if not indices:continue
 sheet=Image.new('RGB',(4*256,math.ceil(len(indices)/4)*294),'white');sd=ImageDraw.Draw(sheet)
 for j,i in enumerate(indices):
  f=row['frames'][i];source=Image.open(R/samples[row['id']]['paths'][i]).convert('RGB')
  d=ImageDraw.Draw(source)
  if f['occlusion']:d.rectangle(f['occlusion'],fill='black')
  x,y,w,h=f['roi'];crop=source.crop((x,y,x+w,y+h)).resize((256,256));cd=ImageDraw.Draw(crop)
  xy=[((v['x']*source.width-x)*256/w,(v['y']*source.height-y)*256/h) for v in f['fused']['right']]
  for a,b in connections:cd.line([xy[a],xy[b]],fill='#00ff44',width=2)
  for a,b in xy:cd.ellipse([a-2,b-2,a+2,b+2],fill='#00ff44')
  wrist=f['original']['pose'][16];wx=(wrist['x']*source.width-x)*256/w;wy=(wrist['y']*source.height-y)*256/h
  cd.line([wx-6,wy,wx+6,wy],fill='#ff2233',width=2);cd.line([wx,wy-6,wx,wy+6],fill='#ff2233',width=2)
  left=j%4*256;top=j//4*294;sheet.paste(crop,(left,top+38))
  candidate=next(c for c in f['candidates'] if c['plausible'])
  sd.text((left+4,top+3),f"frame{i}: Right={candidate['score']:.3f}",fill='black')
  sd.text((left+4,top+18),f"wrist distance={candidate['targetDistance']:.3f} shoulders",fill='black')
  if j==0:
   full=source.copy();dd=ImageDraw.Draw(full);dd.rectangle([x,y,x+w,y+h],outline='#3377ff',width=2)
   for a,b in connections:
    u=f['fused']['right'][a];v=f['fused']['right'][b];dd.line([u['x']*648,u['y']*480,v['x']*648,v['y']*480],fill='#00ff44',width=2)
   overview.append((row['id'],row['control'],i,full))
 filename=f"{row['id']}--{row['control']}.png";sheet.save(overlay_dir/filename);overlay_files.append(str((overlay_dir/filename).relative_to(out)).replace('\\','/'))
sheet=Image.new('RGB',(1296,math.ceil(len(overview)/2)*512),'white');d=ImageDraw.Draw(sheet)
for j,(sid,control,i,img) in enumerate(overview):
 x=j%2*648;y=j//2*512;d.text((x+5,y+4),f"{sid} frame{i} / {control}",fill='black');sheet.paste(img,(x,y+32))
sheet.save(out/'hand-recovery-overview.png')
sources=['ml/v6_hand_recovery_prepare.py','ml/v6_hand_recovery_report.py','web/src/v6_hand_recovery.worker.ts','web/src/v6_hand_detector.worker.ts','web/scripts/v6_hand_recovery_run.mjs','web/v6_hand_recovery.html','web/src/tracking.ts','web/public/models/holistic_landmarker.task']
summary={'version':p['version'],'completed':True,'browserVersion':r['browserVersion'],'protocolSha256':r['protocolSha256'],'model':p['model'],
 'timingScope':{'holisticMs':'Synchronous Holistic detect only.','fallbackCallMs':'Bitmap transfer, child-worker message roundtrip and dedicated palm/hand detection; excludes model initialization.','totalFrameMs':'Primary-worker per-frame RGB fetch/decode, Holistic, optional crop and fallback.','clipWallMsMedianIncludingTwoWorkers':'Legacy field name: actually primary-worker job processing interval including both task initialization and child-worker startup, EXCLUDING primary-worker bootstrap, result serialization and browser orchestration. Not full UI latency.'},
 'scope':{'clips':14,'validationClips':11,'trainControls':3,'sourceFrames':214,'conditions':3,'conditionSequences':42,'processedFrames':642,'preflightFramesExcludedFromTotals':3,'noClassifier':True,'noOld129':True,'noHumanWebcam':True},
 'preflightPassed':r['preflightPassed'],'preflightCategories':[{'id':x['id'],'candidates':[{k:c[k] for k in ['category','score','targetDistance','oppositeDistance','reasons']} for f in x['frames'] for c in f['candidates']]} for x in r['preflight']],
 'originalLandmarksPreservedExactly':preserved,'baselineComparisonWithV5':baseline_diffs,'aggregates':aggregates,'comparisons':comparisons,
 'sourceSha256':{x:hashlib.sha256((R/x).read_bytes()).hexdigest() for x in sources},'rawResultsSha256':hashlib.sha256((out/'hand-recovery-results.json').read_bytes()).hexdigest(),
 'overlayFiles':overlay_files,'overview':'hand-recovery-overview.png',
 'limitations':['All validation Gamma clips belong to signer6; these are exposed research examples, not new-user generalization.',
 'Handedness score is a left/right classifier score, not landmark localization confidence.',
 'Exact V4 guard pass establishes required raw-point presence only; no semantic classifier was executed.',
 'Occlusion and all-black are derived controls, not natural ordinary hand motions or human webcam.',
 'No frame has expert anatomical landmark ground truth; zero control recoveries cannot establish zero false matches on real scenes.',
 'Fallback latency includes child-worker messaging and palm+landmark detection. Full frame latency includes RGB decode, Holistic, crop and fallback; clip time includes task initialization.',
 'MediaPipe ESM requires separate worker realms for the two tasks in this pinned runtime; initial preflight initialization error was fixed without changing protocol.'],
}
(out/'hand-recovery-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'aggregates':aggregates,'overlays':len(overlay_files),'baselineMaxDifference':max(x['maximumCoordinateDifference'] for x in baseline_diffs)}))
