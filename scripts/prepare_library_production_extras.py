"""Verify and transcode previously acquired Elementary23 clips for the public library."""

from pathlib import Path
from shutil import copyfile
import zlib

from acquire_elementary_library_expansion import ARCHIVES, OUTPUT, WORKBOOK, acquire, archive_index, load, official, sha, transcript
from acquire_fresh_continuous import write_manifest
from package_video_library import lookup_key
from prepare_elementary_catalog_plan import ROOT, workbook_rows


COHORT = OUTPUT / "production-extras"
EXTRAS = (
    ("appointment-external-lexical", "glossa_erg_b2_172", "μεθαύριο"),
    ("appointment-month-lexical", "mp_b_1604", "μήνας"),
    ("appointment-external-lexical", "anth_cd_1315", "φωτιά και αίμα"),
)


def main():
    files, official_source = official()
    originals = {identity: (number, signer, text) for number, identity, signer, text in workbook_rows(WORKBOOK)}
    catalog = load(ROOT / "publication-video-library/dist/library/catalog.json")["entries"]
    existing = {lookup_key(word) for item in catalog if item["id"] not in
                {"elementary-" + identity.replace("_", "-") for _, identity, _ in EXTRAS}
                for word in (item["phrase"], *item["aliases"])}
    rows, raw = [], {}
    for group, identity, expected in EXTRAS:
        root = ROOT / "data" / group
        source_id = f"elementary-{identity}"
        source = next(item for item in load(root / "manifest.json")["samples"] if item["id"] == source_id)
        number, signer, annotation = originals[identity]
        greek, kind = transcript(annotation)
        if (greek != expected or lookup_key(greek) in existing or source["workbook_row"] != number
                or source["signer"] != signer or source["original_text"] != annotation):
            raise ValueError(f"Changed workbook annotation or duplicated text for {identity}")
        archive = ARCHIVES[identity.split("_", 1)[0]]
        members, transferred = archive_index(files[archive])
        member = members[identity]
        frozen_raw = root / "rgb" / source_id / "source.flv"
        payload = frozen_raw.read_bytes()
        if (transferred or Path(source["source_path"]).stem != identity or len(payload) != member["size"]
                or zlib.crc32(payload) != member["crc"] or sha(frozen_raw) != source["source_sha256"]):
            raise ValueError(f"Cached original differs from official archive member {identity}")
        existing.add(lookup_key(greek))
        raw[identity] = frozen_raw
        rows.append({"source_id": identity, "signer": signer, "workbook_row": number,
                     "phrase": greek, "transcription": annotation, "kind": kind, "archive": archive,
                     "range_bytes": member["compressed"] + 1024, "member": member})

    media = COHORT / "media"
    media.mkdir(parents=True, exist_ok=True)
    planned = COHORT / "manifest.planned.json"
    plan = {"source": official_source, "workbook_sha256": sha(WORKBOOK), "source_hashes":
            {path.relative_to(ROOT).as_posix(): sha(path) for path in raw.values()}, "rows": rows}
    if planned.exists() and load(planned) != plan:
        raise ValueError("Production library provenance changed")
    write_manifest(planned, plan)
    clips = []
    for row in rows:
        identity = row["source_id"]
        copyfile(raw[identity], media / f"{identity}.flv")
        clip = acquire(row, files, official_source, media)
        if clip["range_bytes_this_run"]:
            raise ValueError("Expected only local video sources")
        clips.append(clip)
    write_manifest(COHORT / "manifest.json", {"source": official_source, "plan_sha256": sha(planned),
                   "complete": True, "requested": len(rows), "clips": clips})
    print(f"Verified {len(clips)} new public library clips from locally cached official source files")


if __name__ == "__main__":
    main()
