"""Fetch and safely decode an official Greek OpenHands evaluation pose split."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import pickletools
import re
import struct
import zipfile

import numpy as np

from dataset_fetch import ROOT, RemoteZipFile


URL = "https://zenodo.org/api/records/6674324/files/GSL.zip/content"
ARCHIVE_BYTES = 725750274
METADATA_SHA256 = "c25babb27efe114b92832f49b110763d8e641aca2cf6432c98ff5369ce787597"
ROOT_CACHE = ROOT / "research/cache/openhands-gsl"


class MemberView(io.RawIOBase):
    """A seekable view of a STORED child archive, with absolute HTTP ranges."""
    def __init__(self, parent, item):
        if item.compress_type != zipfile.ZIP_STORED or item.compress_size != item.file_size:
            raise ValueError("Nested ZIP must be stored without compression")
        self.parent, self.size, self.position = parent, item.file_size, 0
        parent.seek(item.header_offset)
        fields = struct.unpack("<4s5H3I2H", parent.read(30))
        if fields[0] != b"PK\x03\x04":
            raise ValueError("Incorrect local ZIP header")
        self.start = item.header_offset + 30 + sum(fields[-2:])
        if self.start + self.size > parent.size:
            raise ValueError("Child archive goes beyond parent")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Unknown whence")
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise ValueError("Invalid negative seek")
        return self.position

    def read(self, count=-1):
        remaining = max(0, self.size - self.position)
        amount = remaining if count < 0 else min(remaining, count)
        if not amount:
            return b""
        self.parent.seek(self.start + self.position)
        data = self.parent.read(amount)
        self.position += len(data)
        return data


def decode_pose(payload):
    """Read the two published little-endian arrays as bytes; execute no pickle opcodes."""
    ops = [(op.name, arg, pos) for op, arg, pos in pickletools.genops(payload)]
    if ops[-1] != ("STOP", None, len(payload) - 1) or ops[0][:2] != ("PROTO", 4):
        raise ValueError("Unknown pose serialization")
    names = [arg for op, arg, _ in ops if op == "SHORT_BINUNICODE"]
    if names != ["keypoints", "numpy.core.multiarray", "_reconstruct", "numpy", "ndarray", "dtype", "f8", "<", "confidences"]:
        raise ValueError("Unexpected pose key or dtype")
    blobs = [(i, arg) for i, (op, arg, _) in enumerate(ops) if op in ("BINBYTES", "BINBYTES8")]
    if len(blobs) != 2 or any(ops[i-1][0] != "NEWFALSE" for i, _ in blobs):
        raise ValueError("Expected two C-order pose arrays")
    points, confidence = [data for _, data in blobs]
    frames, remainder = divmod(len(points), 75 * 3 * 8)
    if remainder or not 0 < frames <= 1500 or len(confidence) != frames * 75 * 8:
        raise ValueError("Invalid pose array dimensions")
    shape_ops = [arg for name, arg, _ in ops if name in ("BININT1", "BININT2", "BININT")]
    if shape_ops != [0, 1, frames, 75, 3, 3, -1, -1, 0, 0, 1, frames, 75]:
        raise ValueError("Unexpected serialized pose dimensions")
    points = np.frombuffer(points, dtype="<f8").reshape(frames, 75, 3).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite keypoints")
    return points


def metadata(path, split, limit):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != METADATA_SHA256:
        raise ValueError("Metadata does not match the published OpenHands archive")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        def read(filename):
            member = [name for name in archive.namelist() if name.endswith("/" + filename)]
            if len(member) != 1:
                raise ValueError(f"Ambiguous annotation: {filename}")
            return archive.read(member[0]).decode("utf-8-sig").splitlines()
        classes = read("iso_classes.csv")
        test = [line.split("|") for line in read("test_greek_iso.csv")]
        train = [line.split("|") for line in read("train_greek_iso.csv")]
        development = [line.split("|") for line in read("val_greek_iso.csv")]
    if split == "train-expanded":
        selected = [row for row in train if re.search(r"_signer[125]_rep[123]_glosses/", row[0])]
        other = test + development
    elif split == "train-prototypes":
        # A fixed, bounded pool from the original training signers, without
        # looking at any validation or test predictions.
        selected = [row for row in train if re.search(r"_signer[124]_rep1_glosses/", row[0])]
        other = test + development
    else:
        selected = development if split == "val" else test
        other = train + (test if split == "val" else development)
    ids, glosses = map(list, zip(*selected))
    previous = {identity for identity, _ in other}
    if len(test) != 3500 or len(development) != 2331 or len(classes) != 310 or len(set(classes)) != 310 or len(set(ids)) != len(selected) or set(ids) & previous:
        raise ValueError("Unexpected count or overlapping OpenHands split")
    if not set(glosses) <= set(classes):
        raise ValueError("Unmapped gloss labels")
    return selected[:limit]


def reusable_training_poses(metadata_path, selected):
    path = ROOT_CACHE / "official-train-prototypes-poses.npz"
    if not path.is_file():
        return {}
    expected = dict(metadata(metadata_path, "train-prototypes", 3500))
    with np.load(path, allow_pickle=False) as archive:
        points, ids, glosses, offsets = (archive[key] for key in ("keypoints", "ids", "glosses", "offsets"))
    if (len(ids) != len(set(ids)) or dict(zip(ids, glosses, strict=True)) != expected
            or points.ndim != 3 or points.shape[1:] != (75, 3) or not np.isfinite(points).all()
            or offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(points)
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("The reusable pose cache does not match the verified native training subset")
    return {identity: points[offsets[index]:offsets[index + 1]] for index, identity in enumerate(ids)
            if identity in selected}


def fetch(args):
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    selected = metadata(args.metadata, args.split, args.limit)
    results = reusable_training_poses(args.metadata, dict(selected)) if args.split == "train-expanded" else {}
    reused = len(results)
    remote = RemoteZipFile(URL, ARCHIVE_BYTES)
    with zipfile.ZipFile(remote) as outer:
        nested = MemberView(remote, outer.getinfo("GSL/GSL_pose.zip"))
    with zipfile.ZipFile(nested) as poses:
        groups = defaultdict(list)
        for identity, gloss in selected:
            if identity in results:
                continue
            info = poses.getinfo(f"GSL_pose/{identity}.pkl")
            if info.file_size > 4_000_000:
                raise ValueError(f"Oversized pose member: {identity}")
            groups[identity.split("/")[0]].append((identity, gloss, info))
        all_members = sorted(poses.infolist(), key=lambda info: info.header_offset)
        end_offsets = {info.filename: successor.header_offset for info, successor in zip(all_members, all_members[1:])}
        end_offsets[all_members[-1].filename] = poses.start_dir
        planned = []
        for group in groups.values():
            start = min(info.header_offset for _, _, info in group)
            end = max(end_offsets[info.filename] for _, _, info in group)
            planned.append((start, end, group))
        estimate = remote.bytes_received + sum(end - start for start, end, _ in planned)
        if estimate > args.max_transfer_mb * 1_000_000:
            raise ValueError(f"Planned {estimate} bytes exceeds transfer cap")
        print(json.dumps({"split": args.split, "clips": len(selected), "class_coverage": len(set(gloss for _, gloss in selected)),
                          "reused": reused, "groups": len(planned), "planned_bytes": estimate}), flush=True)
        for start, end, group in sorted(planned):
            if remote.bytes_received + end - start > args.max_transfer_mb * 1_000_000:
                raise ValueError("Transfer cap reached")
            nested.seek(start)
            previous = remote.cache
            data = nested.read(end - start)
            if len(data) != end - start:
                raise ValueError("Incomplete clip group")
            # Present the verified range to ZIP's own CRC-checked member reader.
            remote.cache = [(nested.start + start, data), *previous]
            for identity, _, info in group:
                results[identity] = decode_pose(poses.read(info))
            remote.cache = previous
            if len(results) == len(selected) or len(results) // 500 != (len(results) - len(group)) // 500:
                print(json.dumps({"verified_clips": len(results), "bytes": remote.bytes_received}), flush=True)
    sequences = [results[identity] for identity, _ in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, keypoints=np.concatenate(sequences), split=np.array(args.split),
                            offsets=np.cumsum([0, *(len(seq) for seq in sequences)]),
                            ids=np.array([identity for identity, _ in selected]),
                            glosses=np.array([gloss for _, gloss in selected]))
    print(json.dumps({"output": str(args.output), "clips": len(results), "reused": reused,
                      "bytes_received": remote.bytes_received, "requests": remote.requests}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT_CACHE / "gsl_metadata.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("val", "test", "train-prototypes", "train-expanded"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-transfer-mb", type=int)
    args = parser.parse_args()
    args.output = args.output or ROOT_CACHE / f"official-{args.split}-poses.npz"
    max_clips, max_mb = (12000, 230) if args.split == "train-expanded" else (3500, 96)
    args.limit = max_clips if args.limit is None else args.limit
    args.max_transfer_mb = max_mb if args.max_transfer_mb is None else args.max_transfer_mb
    if not 1 <= args.limit <= max_clips or not 1 <= args.max_transfer_mb <= max_mb:
        parser.error(f"Choose 1–{max_clips} clips and at most {max_mb} MB")
    fetch(args)


if __name__ == "__main__":
    main()
