"""Acquire official Elementary23 Greek references relevant to appointment language."""

import argparse
from collections import defaultdict
from hashlib import md5, sha256
import json
from pathlib import Path
import zipfile

from dataset_fetch import RemoteZipFile
from prepare_elementary_catalog_plan import RECORD_URL, ROOT, SOURCE_DIR, workbook_rows
from prepare_elementary_translation_pilot import materialize, words


OUTPUT = ROOT / "data/appointment-external-lexical"
MAX_BYTES = 24_000_000
TARGETS = {
    "Greek Language.zip": {
        "glossa_erg_b2_172": (18474, "C", "μεθαύριο"),
        "glossa_erg_b2_182": (18484, "C", "μεθαύριο"),
        "glossa_b3_0145": (12982, "E", "μεθαύριο"),
    },
    "Anthology of Greek Literacy.zip": {"anth_cd_1315": (2730, "A", "αίμα")},
    "Environmental Studies.zip": {"mp_c_0250": (7028, "A", "αίμα")},
}
MONTH_TARGETS = {
    "Environmental Studies.zip": {"mp_b_1604": (6680, "B", "μήνας")},
    "Mathematics.zip": {"math_c_1038": (23038, "A", "μήνας")},
}


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", action="store_true", help="Acquire only exact whole-video Greek references for μήνας")
    args = parser.parse_args()
    targets_by_archive = MONTH_TARGETS if args.months else TARGETS
    output = ROOT / "data/appointment-month-lexical" if args.months else OUTPUT
    metadata = json.loads((SOURCE_DIR / "elementary23-zenodo-metadata.json").read_text(encoding="utf-8"))
    workbook = SOURCE_DIR / "All_Elementary.xlsx"
    workbook_bytes = workbook.read_bytes()
    files = {row["key"]: row for row in metadata["files"]}
    if (metadata["id"] != 7847052 or metadata["metadata"]["license"]["id"] != "cc-by-4.0"
            or files[workbook.name]["checksum"] != f"md5:{md5(workbook_bytes).hexdigest()}"):
        raise ValueError("The workbook or its official license metadata changed")
    annotations = defaultdict(list)
    for row, identity, signer, text in workbook_rows(workbook):
        annotations[identity].append((row, signer, text))

    pending, received = [], 0
    for name, targets in targets_by_archive.items():
        file = files[name]
        remote = {"archiveUrl": file["links"]["self"], "archiveBytes": file["size"]}
        with RemoteZipFile(remote["archiveUrl"], remote["archiveBytes"]) as stream, zipfile.ZipFile(stream) as archive:
            members = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
            bounds = [entry.header_offset for entry in members[1:]] + [archive.start_dir]
            by_stem = defaultdict(list)
            for entry, end in zip(members, bounds, strict=True):
                if Path(entry.filename).suffix.lower() == ".flv":
                    by_stem[Path(entry.filename).stem].append((entry, end))
            received += stream.bytes_received
        for identity, (row, signer, target) in targets.items():
            [(actual_row, actual_signer, text)] = annotations[identity]
            tokens = words(text)
            if ((actual_row, actual_signer) != (row, signer) or target not in tokens
                    or (args.months and tokens != [target])):
                raise ValueError(f"Unexpected workbook reference for {identity}")
            [(member, end)] = by_stem[identity]
            row_data = {"id": f"elementary-{identity}", "source_path": member.filename, "source_archive": name,
                        "source_url": remote["archiveUrl"], "signer": signer, "split": "train", "text": " ".join(words(text)),
                        "original_text": text.strip(), "workbook_row": row, "target_greek_word": target,
                        "whole_translation_is_target": words(text) == [target], "label_type": "greek_translation_only"}
            pending.append((row_data, remote, member, end))
            received += end - member.header_offset
    if received > MAX_BYTES:
        raise ValueError(f"Required byte ranges {received} exceed the acquisition cap {MAX_BYTES}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {row["id"]: row for row in json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]}
    samples = [materialize(row, archive, member, end, previous.get(row["id"]), output=output)
               for row, archive, member, end in pending]
    manifest = {"source": RECORD_URL, "license": "CC-BY-4.0", "workbook_sha256": sha256(workbook_bytes).hexdigest(),
                "creators": metadata["metadata"]["creators"], "complete": True, "planned_samples": len(samples),
                "scope": "External Greek translation supervision only. No CERTH gloss or per-word timing annotations.",
                "maximum_payload_bytes": received, "samples": samples}
    save(manifest_path, manifest)
    print(json.dumps({"clips": len(samples), "maximum_payload_bytes": received,
                      "sampled_frames": sum(len(row["frame_paths"]) for row in samples)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
