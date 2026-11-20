"""Fetch the official landmark model and copy pinned npm runtime assets locally."""
from pathlib import Path
import hashlib
import json
import shutil
import requests

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
URL = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    target = WEB / 'public/models/holistic_landmarker.task'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with requests.get(URL, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with target.with_suffix('.partial').open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        target.with_suffix('.partial').replace(target)
    copied = []
    for source in (WEB / 'node_modules/@mediapipe/tasks-vision/wasm').glob('*'):
        dest = WEB / 'public/vendor/mediapipe' / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(dest)
    for name in ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']:
        source = WEB / 'node_modules/onnxruntime-web/dist' / name
        dest = WEB / 'public/vendor/onnxruntime' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(dest)
    manifest = {'modelSource': URL, 'files': [
        {'path': str(p.relative_to(WEB)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in [target, *copied]
    ]}
    path = ROOT / 'artifacts/runtime_assets.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
