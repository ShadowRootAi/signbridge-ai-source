"""Prepare diagnostic windows from the already observed held GSLW clips."""

import json
import subprocess

import imageio_ffmpeg

from acquire_fresh_continuous import ROOT, digest, write_manifest


PHONE = ROOT / '.runtime-tmp/recognition-phone-negatives/manifest.json'
OUTPUT = ROOT / '.runtime-tmp/recognition-phone-windows'


def main():
    phone = json.loads(PHONE.read_bytes())
    if not phone['complete']:
        raise ValueError('The full phone cohort has not been verified')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in phone['samples']:
        if row['role'] != 'test':
            continue
        path = ROOT / row['path']
        start, end = row['annotation_start_seconds'], row['annotation_end_seconds']
        if digest(path) != row['sha256'] or not 0 <= start < end <= row['duration_seconds']:
            raise ValueError(f'Invalid parent video or source interval: {row["id"]}')
        identity = row['id'] + '-window'
        out = OUTPUT / (identity + '.mp4')
        pending = out.with_suffix('.part.mp4')
        try:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
                            '-y', '-ss', str(start), '-i', str(path), '-t', str(end - start), '-map', '0:v:0',
                            '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
                            '-movflags', '+faststart', str(pending)], check=True, timeout=30)
            frames, duration = imageio_ffmpeg.count_frames_and_secs(str(pending))
            if frames < 6 or abs(duration - (end - start)) > .15:
                raise ValueError(f'Unexpected diagnostic duration for {identity}')
            pending.replace(out)
        finally:
            pending.unlink(missing_ok=True)
        samples.append({**row, 'id': identity, 'parent_id': row['id'], 'parent_mp4_sha256': row['sha256'],
                        'path': out.relative_to(ROOT).as_posix(), 'sha256': digest(out), 'frames': frames,
                        'duration_seconds': duration})
    write_manifest(OUTPUT / 'manifest.json', {'complete': True, 'parent_sha256': digest(PHONE), 'samples': samples,
                   'scope': 'Supplemental replay of the same held phone sources, interpreting the official '
                            'numeric annotation bounds as seconds. The fields are described inconsistently in the '
                            'published materials, so this does not verify precise token boundaries. No fit or '
                            'calibration uses these excerpts.'})
    print(f'{len(samples)} phone diagnostic windows')


if __name__ == '__main__':
    main()
