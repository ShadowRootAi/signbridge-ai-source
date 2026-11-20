"""Freeze a dedicated-hand fallback pilot before observing outcomes."""
import json, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
m=json.loads((R/'data/open-set-v2/manifest.json').read_text(encoding='utf-8'))
samples=[s for s in m['samples'] if s['split']=='val' and s['class_id']==0]
old=json.loads((R/'reports/v5/tracking-input.json').read_text(encoding='utf-8'))
train=[s for s in old['samples'] if s['split']=='train']
samples=[{'id':s['id'],'split':s['split'],'signer':s['signer_id'],'paths':s['frame_paths'],'sourceSha256':s['sha256']} for s in samples]
samples=[{'id':s['id'],'split':s['split'],'signer':s['signer'],'paths':s['paths'],'sourceSha256':s['sourceSha256']} for s in train]+sorted(samples,key=lambda s:s['id'])
assert len(samples)==14 and sum(s['split']=='val' for s in samples)==11
model=R/'research/cache/mediapipe/hand_landmarker-float16-1.task'
protocol={
 'version':'v6-dedicated-hand-fallback-1','sampleCount':len(samples),'originalFrameCount':sum(len(s['paths']) for s in samples),
 'selection':'All11 existing validation Gamma clips plus the exact3 train Gamma controls from V5. No old129/final-test, classifier, fit, or semantic evaluation.',
 'model':{'path':'research/cache/mediapipe/hand_landmarker-float16-1.task','source':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task','documentation':'https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker','sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'bytes':model.stat().st_size},
 'conditions':['none','target_wrist_occlusion','all_black'],
 'rule':{'onlyWhenOriginalRightMissing':True,'preserveOriginalPoints':True,'poseWristIndex':16,'oppositeWristIndex':15,'minimumPoseVisibility':.5,
   'roiSideShoulderWidths':1.6,'roiSideMinimumPixels':96,'roiSideMaximumPixels':384,
   'candidateWristDistanceMaximumShoulderWidths':.45,'oppositeWristDistanceMinimumShoulderWidths':.5,
   'oppositeDistanceMustExceedTargetDistanceBy':1.5,'requiredHandedness':'Right','minimumHandednessScore':.8,
   'minimumLandmarks':21,'requireAllPointsWithinOriginalImage':True,'ambiguityPolicy':'Reject unless exactly one plausible candidate.',
   'taskThresholds':{'palm':.5,'presence':.5,'tracking':.5},'runningMode':'IMAGE','delegate':'CPU'},
 'preflight':'Before validation, detect first-frame wrist ROI on all3 originally tracked train controls. Require at least2 geometry-consistent observations and all such handedness predictions Right; otherwise stop, no silent inversion.',
 'occlusion':'Same V5 square: centered at original pose right wrist; half-size=max(35px,0.45*shoulder width), expanded to original right-hand bounds+12px. Fallback uses the NEW occluded-frame pose, not original pose. Occlusion may remove adjacent face.',
 'guard':'Read-only import of exact production measureRawTracking and TRACKING_POLICY. Guard is completeness, not semantic accuracy.',
 'samples':samples,
}
out=R/'reports/v6';out.mkdir(parents=True,exist_ok=True)
p=out/'hand-recovery-protocol.json';p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'samples':len(samples),'frames':protocol['originalFrameCount'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
