"""Bounded tracking probe; development/validation only, no classifier or fitting."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
eligible = {s['id']: s for s in manifest['samples'] if s['split'] in ('train', 'val') and s['class_id'] == 0}
raw = {}
for path in sorted((ROOT / 'data/features').glob('extracted-isolated-*.jsonl')):
    for line in path.open(encoding='utf-8'):
        # Only existing train/val Gamma rows with retained raw landmarks are eligible.
        row = json.loads(line)
        if row['id'] in eligible and 'raw' in row:
            raw[row['id']] = row['raw']
samples = []
for sid in sorted(raw):
    sample = eligible[sid]
    frames = raw[sid]
    assert len(frames) == len(sample['frame_paths'])
    samples.append({'id': sid, 'split': sample['split'], 'signer': sample['signer_id'],
                    'label': sample['label'], 'paths': sample['frame_paths'],
                    'sourceSha256': sample['sha256'],
                    'reference': [{'pose': f['pose'], 'right': f['right']} for f in frames]})
protocol = {
  'version': 'v5-tracking-probe-1',
  'selection': 'All original train/val class_id=0 (Gamma) clips with raw frames retained in corrected isolated extraction shards; metadata and prior tracking availability only. No old129/final-test samples.',
  'sampleCount': len(samples), 'noClassifier': True, 'noThresholdFit': True,
  'thresholds': {'face': .5, 'pose': .5, 'hand': .5},
  'variants': ['original', 'pad10_black', 'pose_upper_crop'],
  'controls': ['target_wrist_occlusion', 'all_black'],
  'geometry': {
    'padding': '10% of original width and height on every side, black, no pixel rescaling.',
    'crop': 'Bounding box of original pose indices0..24 with visibility>=0.5, expanded20% of box size each side and clamped to original image. Original pixel resolution retained; Holistic performs its own resizing.',
    'occlusion': 'Erase axis-aligned square around original pose detector-right wrist(index16), half-size=max(35px,0.45*shoulder distance), also enclosing originally detected right-hand points plus12px. Black fill; may remove adjacent face. Diagnostic, not a hand-labeled ground truth.',
    'mapping': 'Every output landmark mapped back into original image coordinates. No swapping/filling/interpolation of missing hands.'},
  'caveats': ['Presence is not landmark accuracy or correct handedness.', 'Derived pixel transforms are not human webcam tests.', 'No tuning or selection after observing these outputs.'],
  'samples': samples,
}
out = ROOT / 'reports/v5/tracking-input.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'samples': len(samples), 'protocolSha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
