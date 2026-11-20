"""Stage fixed author-timed Sartinas clips for a local non-phone V2 diagnostic."""

import json

from prepare_isolated_sartinas_rgb import encode
from prepare_sartinas_appointment_holdout import FRAME_RATE, PAGE, PAPER, RECORDINGS, ROOT, digest, save
from prepare_sartinas_dialogue_fit import annotation_tiers


OUTPUT = ROOT / '.runtime-tmp/recognition-phone-studio-diagnostic'
V2_MANIFEST = ROOT / '.runtime-tmp/recognition-scale-v2/manifest.json'
SELECTION = (('n1', 'a28', 70_937, 71_355), ('n6', 'a108', 123_075, 123_405), ('n7', 'a103', 123_490, 123_936))


def stage(signer, annotation_id, start_ms, end_ms):
    recording = RECORDINGS[signer]
    annotation = recording.media / f'{recording.name}.eaf'
    source = recording.media / f'{recording.name}.avi'
    video_manifest = recording.data / 'video-manifest.json'
    [video] = json.loads(video_manifest.read_bytes())['samples']
    if (video['id'] != recording.id or video['fps'] != FRAME_RATE
            or video['source_video_bytes'] != source.stat().st_size):
        raise ValueError(f'AVI provenance differs for {recording.id}')
    if not recording.avi_sha256 == video['source_video_sha256'] == digest(source):
        raise ValueError(f'Original AVI checksum differs for {recording.id}')
    if not recording.eaf_sha256 == video['elan_sha256'] == digest(annotation):
        raise ValueError(f'Original ELAN checksum differs for {recording.id}')
    _, tier, _, spans = annotation_tiers(annotation.read_bytes(), recording.name)
    index, span = next((i, span) for i, span in enumerate(spans) if span['elan_annotation_id'] == annotation_id)
    if (span['label'], span['start_ms'], span['end_ms']) != ('ΕΓΩ', start_ms, end_ms):
        raise ValueError(f'Exact Greek gloss or annotated interval changed for {recording.id}/{annotation_id}')
    if ((index and spans[index - 1]['end_ms'] > start_ms)
            or (index + 1 < len(spans) and spans[index + 1]['start_ms'] < end_ms)):
        raise ValueError(f'The annotated gloss overlaps another sign for {recording.id}/{annotation_id}')

    first = (start_ms * FRAME_RATE + 999) // 1000
    stop = (end_ms * FRAME_RATE + 999) // 1000
    paths = video['frame_paths'][first:stop]
    expected = [recording.media / 'frames' / f'frame-{index + 1:06}.jpg' for index in range(first, stop)]
    if stop - first < 10 or [ROOT / path for path in paths] != expected or any(not path.is_file() for path in expected):
        raise ValueError(f'Canonical decoded frame range differs for {recording.id}/{annotation_id}')

    identity = f'{recording.id}-{annotation_id}'
    target = OUTPUT / f'{identity}.mp4'
    sample = encode({'id': identity, 'path': target, 'first_frame_zero_based': first,
                     'frames': len(paths), 'frame_paths': paths})
    return {**sample, 'path': target.relative_to(ROOT).as_posix(), 'signer': f'sartinas:{signer}',
            'label': 'εγώ', 'original_text': span['label'], 'is_known': True, 'role': 'test', 'fps': FRAME_RATE,
            'source_video_sha256': video['source_video_sha256'], 'source_video': source.relative_to(ROOT).as_posix(),
            'video_manifest_sha256': digest(video_manifest), 'elan_sha256': video['elan_sha256'],
            'elan_tier': tier, 'annotation_id': annotation_id, 'start_ms': start_ms, 'end_ms': end_ms,
            'frame_sha256': [digest(path) for path in expected]}


def main():
    v2 = json.loads(V2_MANIFEST.read_bytes())
    if v2.get('complete') is not True:
        raise ValueError('The V2 acquisition manifest must be complete')
    v2_rows = [*v2['samples'], *v2.get('previously_used_diagnostic', [])]
    v2_ids = {sample['id'] for sample in v2_rows}
    v2_hashes = {sample['sha256'] for sample in v2_rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = [stage(*selection) for selection in SELECTION]
    if any(sample['id'] in v2_ids or sample['sha256'] in v2_hashes for sample in samples):
        raise ValueError('The diagnostic overlaps the V2 media by ID or checksum')
    if len({sample['sha256'] for sample in samples}) != len(samples):
        raise ValueError('Diagnostic MP4s are not unique')
    save(OUTPUT / 'manifest.json', {
        'complete': True, 'source': PAGE, 'paper': PAPER, 'v2_manifest_sha256': digest(V2_MANIFEST),
        'scope': 'One-class cross-corpus diagnostic with three exact author-timed ΕΓΩ spans from scripted Sartinas Ximea '
                 'camera recordings. Source camera records at 60 fps; this derivative uses the verified cached 30 fps '
                 'decoded grid. Parent recordings were used in other SignBridge probes. Cross-corpus personal identity '
                 'is unverified. The clips are short contextual intervals, used only as a held V2 diagnostic.',
        'samples': samples,
    })
    print(f'Fingerprinted {len(samples)} verified author-annotated Ximea clips in {OUTPUT}')


if __name__ == '__main__':
    main()
