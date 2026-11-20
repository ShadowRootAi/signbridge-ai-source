"""Summarize the fixed tracking-only pilot. No fitting or classifier decisions."""
from pathlib import Path
import hashlib
import json
import statistics
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'reports/v5'
protocol = json.loads((out / 'tracking-input.json').read_text(encoding='utf-8'))
result = json.loads((out / 'tracking-results.json').read_text(encoding='utf-8'))
assert result.get('completed') and len(result['rows']) == 54
rows = result['rows']
variants = protocol['variants']
sample_by_id = {s['id']: s for s in protocol['samples']}
comparisons = []
for sample in protocol['samples']:
    same = {r['variant']: r for r in rows if r['id'] == sample['id'] and r['control'] == 'none'}
    original = same['original']['raw']
    comparisons.append({'id': sample['id'], 'split': sample['split'], 'frames': len(original),
        'rightPresence': {v: same[v]['summary']['rightPresent'] for v in variants},
        'requiredPresence': {v: same[v]['summary']['requiredPresent'] for v in variants},
        'passes90PercentPresenceOnly': {v: same[v]['summary']['passes90PercentPresenceOnly'] for v in variants},
        'pairedCandidateFrames': {v: {
            'newPresenceOnOriginallyMissingFrames': sum(len(a['right']) != 21 and len(b['right']) == 21 for a, b in zip(original, same[v]['raw'])),
            'lostPresenceOnOriginallyPresentFrames': sum(len(a['right']) == 21 and len(b['right']) != 21 for a, b in zip(original, same[v]['raw'])),
        } for v in variants[1:]},
    })
aggregates = []
for control in ['none', *protocol['controls']]:
    for variant in variants:
        rr = [r for r in rows if r['control'] == control and r['variant'] == variant]
        aggregate = {'control': control, 'variant': variant,
            **{key: sum(r['summary'][key] for r in rr) for key in ['frames','leftPresent','rightPresent','posePresent','requiredPresent','rightWristInOccludedArea']},
            'clipsMeeting90PercentPresenceOnly': sum(r['summary']['passes90PercentPresenceOnly'] for r in rr),
            'frameMsMedian': statistics.median(f['elapsedMs'] for r in rr for f in r['raw']),
            'clipWallMsMedianIncludingWorkerWarmup': statistics.median(r['elapsedMs'] for r in rr),
        }
        aggregates.append(aggregate)

# The baseline should reproduce the prior corrected extraction, not silently change detector behavior.
max_difference = 0.0
for sample in protocol['samples']:
    original = next(r for r in rows if r['id'] == sample['id'] and r['control'] == 'none' and r['variant'] == 'original')['raw']
    for old, new in zip(sample['reference'], original):
        for kind in ['pose','right']:
            assert len(old[kind]) == len(new[kind])
            for a,b in zip(old[kind],new[kind]):
                max_difference = max(max_difference, *(abs(a[k]-b[k]) for k in ['x','y','z']))

# Prespecified diagnostic selection after measurements: first gained frame per validation clip,
# plus every residual right output in the wrist-occlusion control. No sample is relabeled.
panels = []
for sample in protocol['samples']:
    if sample['split'] != 'val':
        continue
    base = next(r for r in rows if r['id'] == sample['id'] and r['control'] == 'none' and r['variant'] == 'original')
    crop = next(r for r in rows if r['id'] == sample['id'] and r['control'] == 'none' and r['variant'] == 'pose_upper_crop')
    first = next((i for i,(a,b) in enumerate(zip(base['raw'], crop['raw'])) if not a['right'] and b['right']), None)
    if first is not None:
        panels.append((sample,crop,first))
for row in rows:
    if row['control'] == 'target_wrist_occlusion':
        panels.extend((sample_by_id[row['id']], row, i) for i,f in enumerate(row['raw']) if f['right'])
sheet = Image.new('RGB', (1296, len(panels)*530), 'white')
draw = ImageDraw.Draw(sheet)
for j,(sample,row,i) in enumerate(panels):
    source = Image.open(ROOT / sample['paths'][i]).convert('RGB')
    transformed = source.copy(); d=ImageDraw.Draw(transformed)
    frame=row['raw'][i];g=frame['geometry']
    if g['occlusion']:
        d.rectangle(g['occlusion'], fill='black')
    d.rectangle([g['ox'],g['oy'],g['ox']+g['sx'],g['oy']+g['sy']], outline='#3377ff',width=2)
    for p in frame['right']:
        x,y=p['x']*648,p['y']*480
        d.ellipse([x-3,y-3,x+3,y+3],fill='#00ff44')
    wrist=sample['reference'][i]['pose'][16]
    x,y=wrist['x']*648,wrist['y']*480
    d.line([x-8,y,x+8,y],fill='#ff2233',width=3);d.line([x,y-8,x,y+8],fill='#ff2233',width=3)
    sheet.paste(source,(0,j*530+50));sheet.paste(transformed,(648,j*530+50))
    draw.text((8,j*530+5),f"{sample['id']} frame {i}: original pixels",fill='black')
    draw.text((656,j*530+5),f"{row['variant']} / {row['control']}",fill='black')
    draw.text((656,j*530+23),'Green=detector-right points; red=original pose wrist; blue=crop',fill='black')
sheet.save(out/'tracking-overlays.png')
source_paths=['ml/v5_tracking_prepare.py','ml/v5_tracking_report.py','web/src/v5_tracking.worker.ts','web/v5_tracking.html','web/scripts/v5_tracking_run.mjs','web/public/models/holistic_landmarker.task']
summary={
 'version':'v5-tracking-probe-1','completed':True,'browserVersion':result['browserVersion'],
 'protocolSha256':result['protocolSha256'],'sourceSha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
 'scope':'Six raw-retained original train/validation Gamma clips, 100 frames, 54 independent worker sequences. Tracking only; no semantic classifier, no old129/final-test, no human webcam.',
 'baselineMaxCoordinateDifferenceFromPriorExtraction':max_difference,
 'comparisons':comparisons,'aggregates':aggregates,
 'roiApiProbe':{'acceptedAll54':all(r['roiApi']=='accepted' for r in rows),'note':'Runtime accepted ImageProcessingOptions.regionOfInterest after each measured sequence. Measured variants use explicit canvas crops, not direct ROI API. No independent hand landmarker asset was installed.'},
 'conclusion':'Neither padding nor pose-guided crop restores >=90% detector-right/required presence in any of the three validation clips. Cropping gains some frames but also loses prior positives. Do not promote this experiment to production.',
 'limitations':['Detector presence does not establish anatomical landmark accuracy, handedness, or semantic meaning.', 'Wrist-occlusion crop produces one residual right detection outside the erased wrist region; this is a suspicious/control residual, not established accuracy or a perfectly labeled false positive.', 'All-black controls are easy negatives and do not validate real-world ordinary movements.', 'Timing includes image fetch/decode/transforms/detection per frame; clip wall includes fresh worker/task initialization; no classifier timing.', 'Diagnostic overlays are derived from existing real RGB frames and are not live camera observations.'],
 'artifacts':['tracking-input.json','tracking-results.json','tracking-overlays.png'],
}
(out/'tracking-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'baselineMaxDifference':max_difference,'aggregates':aggregates},ensure_ascii=False))
