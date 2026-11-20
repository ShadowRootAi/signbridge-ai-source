"""Build and verify the static Cloudflare Pages package. No network deployment."""

import argparse
from hashlib import sha256
import json
from os import environ
from pathlib import Path
import re
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
MAX_ASSET = 25 * 1024 * 1024
STATIC_ASSETS = (
    'models/model.onnx', 'models/model_config.json', 'models/holistic_landmarker.task',
    'vendor/mediapipe/vision_wasm_internal.js', 'vendor/mediapipe/vision_wasm_internal.wasm',
    'vendor/mediapipe/vision_wasm_module_internal.js', 'vendor/mediapipe/vision_wasm_module_internal.wasm',
    'vendor/onnxruntime/ort-wasm-simd-threaded.mjs', 'vendor/onnxruntime/ort-wasm-simd-threaded.wasm',
)
LICENSES = (
    'GSL-CC-BY-4.0.txt', 'mediapipe-APACHE-2.0.txt', 'mediapipe-model-NOTICE.txt',
    'onnxruntime-MIT.txt', 'onnxruntime-ThirdPartyNotices.txt',
)
APP_LICENSES = ('react-LICENSE.txt', 'react-dom-LICENSE.txt', 'scheduler-LICENSE.txt')
OPENHANDS_LICENSES = {
    'openhands-APACHE-2.0.txt': 'd32ce31913494b79882c5b2571e23ffd491fa52d76db7a5075dd47a8cf2a90a8',
    'openhands-model-NOTICE.txt': '0fea2a5abe08f04c8fd9ece353f1e9a50f2f602a70ec7a4fee5319d933860525',
}
OPENHANDS_PATH = '__signbridge/openhands'
OPENHANDS_ASSETS = {
    'stgcn.onnx': ('research/cache/openhands-gsl/stgcn.onnx', '8ace9360ed396ecd6df2670a9c34d1bab271d6384808f2972456355c577fbd79'),
    'stgcn-onnx.json': ('research/cache/openhands-gsl/stgcn-onnx.json', '4b312e59995c35e62edae6412f1c7ffed1013d3a18eafb40b5bf9d8e258aa1ed'),
    'policy.json': ('reports/pretrained-gsl/stgcn-gate-config.json', '4d1f2f390877c962c30df8545041d8a3739dd62e243fb860fea6ef5d7bf69836'),
}
EXPANDED_POLICY = (
    'reports/pretrained-gsl/stgcn-gate-expanded-candidate.json',
    '1d7aedb2c11525f0ac7669ea58045506000f8c9348ce703f0764f3bb8bb43efb',
)
AGREEMENT_POLICY = (
    'reports/pretrained-gsl/stgcn-gate-agreement-candidate.json',
    'd5de91810997d3cd9d42658519a7656d12c5935864cb35131ff89aa64b68beb0',
)
ROSTER_200_POLICY = (
    'reports/pretrained-gsl/stgcn-gate-roster-200.json',
    '0c42091df6eca76d41127fc1ef62d05649ea2e082bed5ca03d90f249bb098a6a',
)
LSTM_ASSET = ('research/cache/openhands-gsl/lstm.onnx',
              '78cec3333c33c1880de3da421d0ae4c8b914e7536827662f4db7d3aa630426d1')
CONTINUOUS_PATH = '__signbridge/continuous'
CONTINUOUS_SOURCE = ROOT / 'artifacts/continuous-v1-expanded/browser'
CONTINUOUS_ASSETS = {
    'sequence.onnx': '8deb68dabb1ead76797663097280c1c6329671ff3141af50e4aa0b3072e0a238',
    'sequence-config.json': '10d32e03d4e22fde4072309c069e61d8195a490527e7b36766f559e9b9b1c040',
}
LIBRARY_IDS = ('geia', 'efcharisto', 'entaxei')
LIBRARY_CATALOG = 'library/catalog.json'
LIBRARY_NOTICE = 'library/NOTICE.txt'
LIBRARY_SNAPSHOTS = {
    'legacy': WEB / 'deployment/legacy-library',
    'expanded': WEB / 'deployment/expanded-library',
}
LIBRARY_REVIEWS = {
    'legacy': ROOT / 'reports/v4/text-to-sign-rights-review.json',
    'expanded': ROOT / 'reports/library-expansion/rights-review.json',
    'library100': ROOT / 'reports/library-100/rights-review.json',
}
ELEMENTARY_RECORD = 'https://zenodo.org/records/7847052'
BUILD_ASSET = re.compile(r'assets/[A-Za-z0-9_.-]+\.(?:js|mjs|css|wasm|svg)')
FROZEN_MODELS = {
    'models/model.onnx': 'beaed67bc63103370754cdbf25eda012f4464c66af74a11a7923a127254c2a45',
    'models/model_config.json': '817c389851b13c61b0b7e0feb68e62b848082d049b8abfd7bb5adeb36ab1b7d6',
    'models/holistic_landmarker.task': 'e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8',
    'hand': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    'config': 'bc0b931319d25986f3c14b9b9aba699ffe448d200f6c147b4b62cf4791f21bc7',
}


def digest(data):
    return sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def static_files(directory):
    found = {}
    for path in sorted(directory.rglob('*')):
        check(not path.is_symlink() and not path.is_junction(), f'Unexpected link: {path.name}')
        if path.is_file():
            found[path.relative_to(directory).as_posix()] = path
    return found


def library_assets(directory, edition, check_media=False, catalog_path=None):
    catalog = read_json(catalog_path or directory / LIBRARY_CATALOG)
    review = LIBRARY_REVIEWS[edition]
    check(catalog['reviewSha256'] == digest(review.read_bytes()), 'Library rights review changed')
    entries = {entry['id']: entry for entry in catalog['entries']}
    check(len(catalog['entries']) == len(entries), 'Duplicate library IDs')
    review_data = read_json(review)
    previous_entries = {}
    if edition == 'library100':
        elementary = review_data['elementary_source']
        check(elementary['record_url'] == ELEMENTARY_RECORD and elementary['license']['id'] == 'cc-by-4.0'
              and elementary['access_right'] == 'open', 'Unexpected Elementary library source license')
        previous_catalog = read_json(LIBRARY_SNAPSHOTS['expanded'] / 'catalog.json')
        check(previous_catalog['reviewSha256'] == digest(LIBRARY_REVIEWS['expanded'].read_bytes()),
              'Previous expanded library rights review changed')
        previous_entries = {entry['id']: entry for entry in previous_catalog['entries']}
        check(len(previous_entries) == 18 and all(entries.get(name) == entry for name, entry in previous_entries.items()),
              'Previous expanded library entries changed')
        check(len(entries) - len(previous_entries) > 100, 'Expected more than 100 new Elementary library words')
    if edition != 'legacy':
        clips = {clip['id']: clip for clip in review_data['clips']}
        check(review_data['source']['license']['id'] == 'cc-by-4.0' and review_data['source']['access_right'] == 'open',
              'Unexpected expanded library source license')
        check(len(clips) == len(review_data['clips']) and set(clips) == set(entries)
              and set(LIBRARY_IDS) <= set(entries) and len(clips) >= 12, 'Unexpected expanded library catalog')
    else:
        check(set(entries) == set(LIBRARY_IDS), 'Unexpected library catalog')
    for name, entry in entries.items():
        check(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name) is not None, f'Invalid library ID: {name}')
        video = f'library/{name}.mp4'
        check(entry['videoUrl'] == '/' + video and entry['rights']['redistributionConfirmed'] is True
              and entry['license']['id'] == 'CC-BY-4.0', f'Unapproved public library entry: {name}')
        if edition != 'legacy':
            approved = clips[name]
            check(entry['phrase'] == approved['phrase'] and entry.get('annotationLabel') == approved.get('annotation_label')
                  and entry['sourceUtterance'] == approved['source_path'] and entry['sha256'] == approved['mp4_sha256'],
                  f'Library entry differs from review: {name}')
            if edition == 'library100' and name not in previous_entries:
                video_id = approved['workbook_video_id']
                check(name == 'elementary-' + video_id.replace('_', '-') and entry['sourceUrl'] == ELEMENTARY_RECORD
                      and entry['sourceArchiveUrl'] == elementary['archive_url']
                      and approved['source_record_url'] == ELEMENTARY_RECORD
                      and Path(approved['source_path']).name == f'{video_id}.flv'
                      and entry['phrase'] == approved['workbook_transcription'].strip()
                      and approved['verified_decodable'] is True
                      and entry.get('annotationLabel') is None and entry['contentKind'] == 'word'
                      and entry['verification']['status'] == 'dataset-annotated', f'Unapproved Elementary clip: {name}')
        if check_media:
            check(entry['sha256'] == digest((directory / video).read_bytes()), f'Library video changed: {name}')
    return (LIBRARY_CATALOG, LIBRARY_NOTICE, *(f'library/{name}.mp4' for name in entries))


def add_openhands_assets(staged, expanded=False, agreement=False, roster_200=False):
    target = staged / OPENHANDS_PATH
    target.mkdir(parents=True)
    model_config = {}
    assets = dict(OPENHANDS_ASSETS)
    if agreement:
        assets.update({'lstm.onnx': LSTM_ASSET, 'policy.json': ROSTER_200_POLICY if roster_200 else AGREEMENT_POLICY})
    elif expanded:
        assets['policy.json'] = EXPANDED_POLICY
    for name, (source, expected) in assets.items():
        content = (ROOT / source).read_bytes()
        check(digest(content) == expected, f'Unexpected OpenHands asset: {name}')
        if name == 'stgcn-onnx.json':
            # Retain inference metadata and leave the parity dataset IDs out.
            model_config = json.loads(content)
            keys = ('labels', 'input', 'input_shape', 'output', 'source_checkpoint_sha256', 'model_sha256')
            content = (json.dumps({key: model_config[key] for key in keys}, ensure_ascii=False) + '\n').encode('utf-8')
        elif name == 'policy.json':
            policy = json.loads(content)
            check(policy['runtimeModelSha256'] == OPENHANDS_ASSETS['stgcn.onnx'][1]
                  and policy['sourceModelSha256'] == model_config['source_checkpoint_sha256']
                  and len(policy['supportedLabels']) == (200 if roster_200 else 120)
                  and set(policy['supportedLabels'] + policy['unknownLabels']) == set(model_config['labels']),
                  'OpenHands policy does not match the model and roster')
            if agreement:
                check(policy['agreementGate']['modelSha256'] == LSTM_ASSET[1],
                      'OpenHands agreement policy does not match the secondary model')
        (target / name).write_bytes(content)


def add_static_assets(staged, full_app, openhands, edition, agreement=False, sequence=False, roster_200=False):
    snapshot = LIBRARY_SNAPSHOTS.get(edition)
    library_sources = {name: snapshot / Path(name).name for name in (LIBRARY_CATALOG, LIBRARY_NOTICE)} if snapshot else {}
    library = library_assets(WEB / 'public', edition, catalog_path=library_sources.get(LIBRARY_CATALOG)) if full_app else ()
    names = STATIC_ASSETS + library
    for name in names:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(library_sources.get(name, WEB / 'public' / name), target)
        if name in FROZEN_MODELS:
            check(digest(target.read_bytes()) == FROZEN_MODELS[name], f'Unexpected model bytes: {name}')

    licenses = WEB / 'deployment/licenses'
    provenance = {row['file']: row['sha256'] for row in read_json(licenses / 'provenance.json')['records']}
    (staged / 'licenses').mkdir()
    for name in LICENSES + (APP_LICENSES if full_app else ()):
        source = licenses / name
        check(digest(source.read_bytes()) == provenance[name], f'Upstream notice changed: {name}')
        copyfile(source, staged / 'licenses' / name)

    notice = (WEB / 'public/NOTICE.txt').read_text(encoding='utf-8')
    if openhands:
        add_openhands_assets(staged, edition != 'legacy', agreement, roster_200)
        for name, expected in OPENHANDS_LICENSES.items():
            source = licenses / name
            check(digest(source.read_bytes()) == expected, f'OpenHands license or notice changed: {name}')
            copyfile(source, staged / 'licenses' / name)
        notice += ('\n\nOPENHANDS GREEK ST-GCN\n'
                   'See /licenses/openhands-model-NOTICE.txt and /licenses/openhands-APACHE-2.0.txt '
                   'for the trained model, source data, license and SignBridge changes.\n')
        if agreement:
            notice += ('\nOPENHANDS GREEK LSTM VERIFIER\n'
                       'Copyright 2021, the AI4Bharat team. Apache License 2.0.\n'
                       'https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/gsl_lstm.zip\n'
                       'Checkpoint ZIP SHA-256: eea777414207393db225f24e6e6377746114e019f6471d73998c6ae564c440ff\n'
                       f'ONNX SHA-256: {LSTM_ASSET[1]}\n'
                       'SignBridge converted the official checkpoint to ONNX for local verification of uncertain predictions.\n'
                       'See /licenses/openhands-model-NOTICE.txt for the shared citation and datasets.\n')
    if sequence:
        target = staged / CONTINUOUS_PATH
        target.mkdir(parents=True)
        for name, expected in CONTINUOUS_ASSETS.items():
            content = (CONTINUOUS_SOURCE / name).read_bytes()
            check(digest(content) == expected, f'Unexpected continuous recognition artifact: {name}')
            (target / name).write_bytes(content)
        notice += ('\n\nSIGNBRIDGE CONTINUOUS VIDEO PILOT\n'
                   'SignBridge GRU/CTC trained on 48 original utterances from the Greek Sign Language (GSL) Dataset.\n'
                   'Source and author attribution: /library/NOTICE.txt. Dataset: https://doi.org/10.5281/zenodo.4756317.\n'
                   'The recognizer offers exact matches to six training references.\n')
    if full_app:
        library_assets(staged, edition, check_media=True)
        notice = notice.replace('V4 PUBLIC VIDEO LIBRARY', 'PUBLIC VIDEO LIBRARY')
        if edition == 'library100':
            notice = notice.replace('Three selected GSL utterance video derivatives',
                                    'Selected GSL utterance video derivatives and Elementary dataset word clips')
            notice += f'\nElementary source record: {ELEMENTARY_RECORD} (CC BY 4.0).\n'
        elif edition == 'expanded':
            notice = notice.replace('Three selected GSL utterance video derivatives', 'Selected GSL utterance video derivatives')
        (staged / '404.html').write_text(
            '<!doctype html><html lang="el"><meta charset="utf-8"><title>SignBridge AI</title>'
            '<main><h1>Η σελίδα δεν βρέθηκε.</h1><a href="/">Επιστροφή στο SignBridge</a></main></html>\n',
            encoding='utf-8',
        )
    else:
        notice = notice.split('\nREACT\n', 1)[0]
    notice += (
        '\n\nMEDIAPIPE HAND LANDMARKER\n'
        'Official Google asset, float16 version 1, unmodified.\n'
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n'
        f'SHA-256: {FROZEN_MODELS["hand"]}\n'
        'See /licenses/mediapipe-model-NOTICE.txt for the upstream model-license statement.\n'
    )
    notice += ('\nOnly the attributed library excerpts are included. Evaluation footage, webcam recordings and reports are excluded.\n'
               if full_app else '\nThis build contains no source or derived GSL videos, webcam recordings or evaluation reports.\n')
    (staged / 'NOTICE.txt').write_text(notice, encoding='utf-8')
    headers = read_json(WEB / 'deployment/headers.json')
    rules = '/*\n' + ''.join(f'  {name}: {value}\n' for name, value in headers.items())
    if openhands:
        rules += f'/{OPENHANDS_PATH}/stgcn.onnx\n  Content-Type: application/octet-stream\n'
    if agreement:
        rules += f'/{OPENHANDS_PATH}/lstm.onnx\n  Content-Type: application/octet-stream\n'
    if sequence:
        rules += f'/{CONTINUOUS_PATH}/sequence.onnx\n  Content-Type: application/octet-stream\n'
    (staged / '_headers').write_text(rules, encoding='utf-8', newline='\n')


def inventory(staged, full_app, openhands, edition, agreement=False, sequence=False):
    exact = {'index.html', '_headers', 'NOTICE.txt', *STATIC_ASSETS, *('licenses/' + name for name in LICENSES)}
    if full_app:
        exact.update(('404.html', *library_assets(staged, edition), *('licenses/' + name for name in APP_LICENSES)))
        logos = list(staged.glob('assets/signbridge-logo-*.png'))
        logo_source = WEB / 'src/assets/signbridge-logo.png'
        check(len(logos) == 1 and digest(logos[0].read_bytes()) == digest(logo_source.read_bytes()),
              'Invalid bundled SignBridge logo')
        exact.add(logos[0].relative_to(staged).as_posix())
    if openhands:
        exact.update(f'{OPENHANDS_PATH}/{name}' for name in OPENHANDS_ASSETS)
        exact.update(f'licenses/{name}' for name in OPENHANDS_LICENSES)
    if agreement:
        exact.add(f'{OPENHANDS_PATH}/lstm.onnx')
    if sequence:
        exact.update(f'{CONTINUOUS_PATH}/{name}' for name in CONTINUOUS_ASSETS)
    imports = {'hand': 'assets/hand_landmarker-float16-1-*.task', 'config': 'assets/model-config-*.json'}
    for kind, glob in imports.items():
        paths = list(staged.glob(glob))
        if openhands and not paths:
            continue
        check(len(paths) == 1 and digest(paths[0].read_bytes()) == FROZEN_MODELS[kind], f'Invalid bundled {kind} bytes')
        exact.add(paths[0].relative_to(staged).as_posix())

    paths = static_files(staged)
    check(exact.issubset(paths), 'Release files are missing')
    check(len(paths) <= 1000, 'Cloudflare dashboard file-count limit exceeded')
    result = []
    for name, path in paths.items():
        check(name in exact or BUILD_ASSET.fullmatch(name), f'Unexpected release asset: {name}')
        content = path.read_bytes()
        check(len(content) <= MAX_ASSET, f'Cloudflare asset limit exceeded: {name}')
        if path.suffix in {'.html', '.js', '.mjs', '.json', '.css', '.txt'}:
            check(not re.search(rb'[a-zA-Z]:[\\/]+Users[\\/]+|/@fs/', content), f'Local path leaked: {name}')
        result.append({'path': name, 'bytes': len(content), 'sha256': digest(content)})
    html = (staged / 'index.html').read_text(encoding='utf-8')
    check('/assets/' in html and '/src/' not in html and '/@vite/' not in html, 'Expected a production HTML entry')
    check(not re.search(r'<style\b|<script(?![^>]*\bsrc=)[^>]*>', html, re.IGNORECASE), 'Inline code violates release CSP')
    return result


def make_archive(staged, target, assets):
    with ZipFile(target, 'w', ZIP_DEFLATED, compresslevel=6) as bundle:
        for item in assets:
            bundle.write(staged / item['path'], item['path'])
    with ZipFile(target) as bundle:
        check(bundle.testzip() is None, 'Archive CRC check failed')
        check(bundle.namelist() == [item['path'] for item in assets], 'Archive entries differ from the audited build')
        for item in assets:
            check(digest(bundle.read(item['path'])) == item['sha256'], f'Archive differs: {item["path"]}')


def main(full_app, openhands, edition, agreement=False, sequence=False, roster_200=False, frame_offset=False):
    release_name = 'recognition-200' if roster_200 else 'sequence-pilot' if sequence else 'recognition-agreement' if agreement else {'expanded': 'expanded', 'library100': 'library-100'}.get(edition)
    if release_name is None:
        release_name = 'openhands' if openhands else 'app' if full_app else 'v8'
    if frame_offset:
        release_name += '-updated'
    release = ROOT / f'publication-{release_name}'
    archive_path = ROOT / f'signbridge-{release_name}-pages.zip'
    runtime = ROOT / '.runtime-tmp'
    runtime.mkdir(exist_ok=True)
    check(runtime.resolve().parent == ROOT and release.resolve().parent == ROOT, 'Release directories must stay in the project')
    with TemporaryDirectory(prefix='v8-pages-', dir=runtime) as temp:
        staging = Path(temp).resolve()
        check(staging.parent == runtime.resolve(), 'Invalid staging directory')
        staged = staging / 'dist'
        subprocess.run([
            'node', 'node_modules/vite/bin/vite.js', 'build', '--config', 'vite.v8-public.config.ts',
            '--outDir', str(staged),
            *(['--mode', 'signbridge-openhands' if openhands else 'signbridge-app'] if full_app else []),
        ], cwd=WEB, check=True, env={**environ, 'VITE_SIGNBRIDGE_CONTINUOUS': '1' if sequence else '0',
                                     'VITE_SIGNBRIDGE_JUDGE_DEMO': '0',
                                     'VITE_SIGNBRIDGE_OPENHANDS_FRAME_OFFSET': '1' if frame_offset else '0'})
        if not full_app:
            (staged / 'v8-pilot.html').rename(staged / 'index.html')
        add_static_assets(staged, full_app, openhands, edition, agreement, sequence, roster_200)
        assets = inventory(staged, full_app, openhands, edition, agreement, sequence)
        archive = staging / archive_path.name
        make_archive(staged, archive, assets)

        release.mkdir(exist_ok=True)
        target = release / 'dist'
        check(target.resolve() == target, 'Release output must not be a symlink or junction')
        previous = staging / 'previous-dist'
        if target.exists():
            target.rename(previous)
        try:
            staged.rename(target)
        except OSError:
            if previous.exists():
                previous.rename(target)
            raise
        archive.replace(archive_path)
        source_paths = ('index.html', 'src/main.tsx', 'src/App.tsx', 'src/styles.css', 'src/assets/signbridge-logo.png',
                        *(p.relative_to(WEB).as_posix() for p in sorted((WEB / 'src/ui').glob('*')) if p.is_file())) if full_app else (
                            'v8-pilot.html', 'src/v8-pilot.ts', 'src/v8-pilot.css',
                        )
        if openhands:
            source_paths += ('src/recognition-release.ts', 'src/pipeline.ts', 'src/openhands-pipeline.ts',
                             'src/openhands-classifier.worker.ts', 'src/openhands-recognizer.ts',
                             'src/use-openhands-vocabulary.ts')
        if sequence:
            source_paths += ('src/continuous-inference.ts', 'src/continuous-pipeline.ts', 'src/continuous.worker.ts',
                             'src/personal-gestures.ts', 'src/local-video.ts')
        result = {
            'status': 'prepared_not_published', 'files': assets,
            'product': f'signbridge-{release_name}' if full_app else 'v8-pilot',
            'openhands_frame_boundary_offset_ms': 2 if frame_offset else 0,
            'totalBytes': sum(item['bytes'] for item in assets),
            'largestAssetBytes': max(item['bytes'] for item in assets),
            'archive': {'file': archive_path.name, 'bytes': archive_path.stat().st_size, 'sha256': digest(archive_path.read_bytes())},
            'sources': {name: digest((WEB / name).read_bytes()) for name in (
                *source_paths, 'src/v8-pilot-core.ts', 'vite.v8-public.config.ts',
            )},
        }
        (release / 'publication-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'Prepared {archive_path.name} with {len(assets)} verified files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', action='store_true', help='Package the complete public app and its licensed library')
    parser.add_argument('--openhands', action='store_true', help='Package the full app with the verified OpenHands model')
    parser.add_argument('--expanded', action='store_true', help='Package OpenHands and the expanded reviewed library')
    parser.add_argument('--library100', action='store_true', help='Package OpenHands and the library with 100+ new annotated words')
    parser.add_argument('--recognition', action='store_true', help='Package the 131-entry library with the OpenHands agreement verifier')
    parser.add_argument('--recognition-200', action='store_true', help='Package the expanded isolated-sign policy for preview')
    parser.add_argument('--sequence', action='store_true', help='Package the continuous video and Teach Me pilots')
    parser.add_argument('--frame-offset', action='store_true', help='Build the recognition update with later MP4 frame seeks')
    args = parser.parse_args()
    if args.recognition_200 and args.sequence:
        parser.error('The 200-label package is a separate recognition release')
    if args.frame_offset and not args.recognition_200:
        parser.error('--frame-offset requires --recognition-200')
    edition = 'library100' if args.library100 or args.recognition or args.recognition_200 or args.sequence else 'expanded' if args.expanded else 'legacy'
    openhands = args.openhands or edition != 'legacy'
    main(args.app or openhands, openhands, edition, args.recognition or args.recognition_200 or args.sequence,
         args.sequence, args.recognition_200, args.frame_offset)
