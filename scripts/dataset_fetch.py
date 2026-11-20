"""Fetch only selected members from public GSL archives using HTTP byte ranges."""
from pathlib import Path
import argparse
import collections
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'

class RemoteZipFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.bytes_received = 0
        self.requests = 0
        self.cache = []
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position
    def read(self, count=-1):
        start = self.position
        end = self.size if count < 0 else min(self.size, start + count)
        if end <= start: return b''
        if end - start > 64 * 1024 * 1024:
            raise ValueError('Refusing a single read larger than 64 MiB')
        for offset, block in self.cache:
            if offset <= start and offset + len(block) >= end:
                self.position = end
                return block[start-offset:end-offset]
        req = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{end-1}', 'Accept-Encoding':'identity'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206:
                        raise ValueError(f'Server ignored Range: HTTP {response.status}')
                    expected_prefix = f'bytes {start}-{end-1}/'
                    if not response.headers.get('Content-Range','').startswith(expected_prefix):
                        raise ValueError('Unexpected Content-Range ' + response.headers.get('Content-Range',''))
                    block = response.read(end-start+1)
                    if len(block) != end-start: raise ValueError('Incomplete range')
                break
            except Exception:
                if attempt == 3: raise
                time.sleep(1+attempt)
        self.requests += 1
        self.bytes_received += len(block)
        if len(block) > 100000 or start >= self.size-100000:
            self.cache.append((start,block))
        self.position = end
        return block

def get_archive(name):
    metadata = json.loads((DATA/'original_annotations/zenodo_metadata.json').read_text(encoding='utf-8'))
    entry = next(item for item in metadata['files'] if item['key']==name)
    remote = RemoteZipFile(entry['links']['self'], entry['size'])
    return remote, zipfile.ZipFile(remote)

def inspect(name):
    remote, archive = get_archive(name)
    members = archive.infolist()
    inventory = [{'path':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size,'offset':x.header_offset,'crc':x.CRC} for x in members]
    (DATA/'archive_indexes').mkdir(exist_ok=True)
    (DATA/'archive_indexes'/f'{name}.json').write_text(json.dumps(inventory,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'archive':name,'member_count':len(members),'first':inventory[:12], 'last':inventory[-3:],
                      'types':dict(collections.Counter(Path(x.filename).suffix for x in members)),
                      'sample_glosses':[x.filename for x in members if 'gloss' in x.filename][:5],
                      'bytes_received':remote.bytes_received, 'requests':remote.requests},ensure_ascii=True),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inspect',default='police1.zip')
    args=parser.parse_args()
    inspect(args.inspect)
