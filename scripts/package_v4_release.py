"""Archive the audited static release only, verifying every shipped file."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'publication-v4'
ARCHIVE = ROOT / 'signbridge-v4-static-preview.zip'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    audit = json.loads((RELEASE / 'publication-audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'prepared_not_published' and not audit['published']
    qa = json.loads((ROOT / 'reports/v4/product-qa/report.json').read_text(encoding='utf-8'))
    assert qa['passed'], 'A passing production product check is required.'
    assert qa['publicationAuditSha256'] == sha((RELEASE / 'publication-audit.json').read_bytes()), 'QA belongs to a different release.'
    access_path = ROOT / 'reports/v4/product-qa/accessibility.json'
    access = json.loads(access_path.read_text(encoding='utf-8'))
    assert access['passed'] and access['publicationAuditSha256'] == qa['publicationAuditSha256']
    expected = {item['path']: item for item in audit['files']}
    actual = {p.relative_to(RELEASE / 'dist').as_posix() for p in (RELEASE / 'dist').rglob('*') if p.is_file()}
    assert actual == set(expected), 'Static files changed after publication audit.'
    for name, item in expected.items():
        data = (RELEASE / 'dist' / name).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256'], name
    outer = ['README.md', 'vercel.json', 'publication-audit.json']
    names = sorted(['dist/' + name for name in expected] + outer)
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in names:
            source = RELEASE / name
            assert source.is_file() and not source.is_symlink()
            bundle.write(source, name)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == set(names)
        for name in names:
            assert sha(bundle.read(name)) == sha((RELEASE / name).read_bytes())
    result = {'archive': ARCHIVE.name, 'bytes': ARCHIVE.stat().st_size,
              'sha256': sha(ARCHIVE.read_bytes()), 'entries': len(names),
              'crcAndAllFilesVerified': True, 'published': False,
              'productQaSha256': sha((ROOT / 'reports/v4/product-qa/report.json').read_bytes()),
              'accessibilityQaSha256': sha(access_path.read_bytes())}
    (ROOT / 'reports/v4/release-package.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
