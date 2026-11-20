"""Build whole-exchange browser inputs from the verified consecutive CERTH RGB cuts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'data/appointment-dialogues/manifest.json'
OUTPUT = ROOT / '.runtime-tmp/appointment-video'
FPS = 30


def video(dialogue, by_id, destination):
    identity = dialogue['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', identity):
        raise ValueError(f'Invalid dialogue identity: {identity}')
    turns = [by_id[key] for key in dialogue['turn_ids']]
    expected = [(dialogue['split'], str(dialogue['signer']), int(dialogue['repetition']))] * len(turns)
    actual = [(row['split'], str(row['signer']), int(row['repetition'])) for row in turns]
    indices = [int(row['template'].rsplit('sentences', 1)[1]) for row in turns]
    if (len(turns) < 3 or actual != expected or indices != list(range(indices[0], indices[0] + len(turns)))
            or {row['template'].split('/')[0] for row in turns} != {dialogue['scenario']}):
        raise ValueError(f'Dialogue must contain consecutive cuts from one original take: {identity}')

    offsets, paths, jpeg_parts = [0], [], []
    for row in turns:
        if row['fps'] != FPS or len(row['frame_paths']) != row['frame_count'] or not row['frame_count']:
            raise ValueError(f'Invalid RGB frame count or FPS: {row["id"]}')
        digest = sha256()
        for name in row['frame_paths']:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('An RGB source path escaped the project')
            jpeg = path.read_bytes()
            jpeg_parts.append(jpeg)
            digest.update(sha256(jpeg).digest())
        if digest.hexdigest() != row['sha256']:
            raise ValueError(f'RGB sequence digest differs: {row["id"]}')
        paths.extend(row['frame_paths'])
        offsets.append(len(paths))

    target = destination / f'{identity}.mp4'
    staging = target.with_suffix('.pending.mp4')
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror', '-y',
        '-threads', '1', '-f', 'image2pipe', '-framerate', str(FPS), '-c:v', 'mjpeg', '-i', 'pipe:0',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1', '-an',
        '-movflags', '+faststart', str(staging),
    ], input=b''.join(jpeg_parts), check=True, capture_output=True)
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(staging))
    if frames != len(paths) or abs(seconds - frames / FPS) > .011:
        raise ValueError(f'Encoded MP4 changed the number or duration of source frames: {identity}')
    staging.replace(target)
    print(json.dumps({'dialogue': identity, 'frames': frames, 'seconds': round(seconds, 2)}, ensure_ascii=False), flush=True)
    return {'id': identity, 'fps': FPS, 'frame_paths': paths, 'frame_count': frames,
            'reference_offsets_30fps': offsets, 'mp4': target.relative_to(ROOT).as_posix(),
            'mp4_sha256': sha256(target.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--include-training', action='store_true', help='Also assemble original chronological training takes')
    mode.add_argument('--test-only', action='store_true', help='Assemble only the held chronological takes')
    args = parser.parse_args()
    source = args.manifest.read_bytes()
    manifest = json.loads(source.decode('utf-8-sig'))
    if not manifest.get('complete'):
        raise ValueError('Require a complete verified appointment acquisition')
    by_id = {row['id']: row for row in manifest['samples']}
    roles = {'test'}
    if not args.test_only:
        roles.add('val')
    if args.include_training:
        roles.add('train')
    groups = [row for row in manifest['dialogues'] if row['split'] in roles]
    if (not groups or {row['split'] for row in groups} != roles
            or len({row['id'] for row in groups}) != len(groups)):
        raise ValueError('Missing distinct dialogue takes for the requested roles')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Use a project-local output path')
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        samples = list(pool.map(lambda item: video(item, by_id, output), groups))
    payload = {'source_manifest_sha256': sha256(source).hexdigest(), 'fps': FPS,
               'scope': 'Official consecutive RGB cuts joined without inserting frames. Original gaps are not reconstructed.',
               'samples': samples}
    target = output / 'manifest.json'
    staging = target.with_suffix('.pending.json')
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    staging.replace(target)


if __name__ == '__main__':
    main()
