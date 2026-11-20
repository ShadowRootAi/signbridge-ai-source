"""Verify exact GSL annotations and prepare a library of complete RGB utterances."""

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "web/public/library"
REVIEW = ROOT / "reports/library-expansion/rights-review.json"
ORIGINAL_REVIEW = ROOT / "reports/v4/text-to-sign-rights-review.json"
ANNOTATIONS = ROOT / "data/original_annotations"
MANIFESTS = (
    ROOT / "data/manifest.json",
    ROOT / "data/open-set-v2/manifest.json",
    ROOT / "data/v3/manifest.json",
)
ORIGINAL_IDS = ("geia", "efcharisto", "entaxei")
ADDITIONS = (
    ("nai", "ναι", "ΝΑΙ"),
    ("ochi", "όχι", "ΟΧΙ"),
    ("oraia", "ωραία", "ΚΑΛΟ"),
    ("vevaios", "βεβαίως", "ΦΥΣΙΚΑ"),
    ("katalava", "κατάλαβα", "ΚΑΤΑΛΑΒΑΙΝΩ"),
    ("teleia", "τέλεια", "ΤΕΛΕΙΑ"),
    ("antio", "αντίο", "ΓΕΙΑ"),
    ("geia-sas", "γεια σας", "ΓΕΙΑ"),
    ("kalimera", "καλημέρα", "ΓΕΙΑ"),
    ("malista", "μάλιστα", "ΟΚ"),
    ("prin-ligo", "πριν λίγο", "ΠΑΡΕΛΘΟΝ_ΠΡΟΣΦΑΤΟ"),
)
PHRASES = (
    ("kalo-apogevma", "καλό απόγευμα", "ΚΑΛΟ ΑΠΟΓΕΥΜΑ"),
    ("poio-provlima", "ποιο είναι το πρόβλημα", "ΠΡΟΒΛΗΜΑ ΤΙ"),
    ("den-egine-tipota", "δεν έγινε τίποτα", "ΑΠΟΤΕΛΕΣΜΑ ΤΙΠΟΤΑ"),
    ("thelo-kataggelia", "θέλω να κάνω καταγγελία", "ΕΓΩ(1) ΘΕΛΩ ΠΑΡΑΠΟΝΟ ΚΑΤΑΓΓΕΛΙΑ"),
)
CONTINUOUS = ROOT / "data/continuous-v1-expanded/manifest.json"
VERIFICATION = (
    "Απόδοση σύμφωνα με τις επισημάνσεις του GSL. Το ελληνικό κείμενο και το νόημα "
    "ελέγχθηκαν στα αρχεία της πηγής. Δεν έγινε νέα αξιολόγηση από ειδικό ΕΝΓ."
)
CHANGES = (
    "Original released JPEG sequence encoded as H.264/yuv420p MP4 at 30 fps and CRF 18. "
    "Frame sequence, dimensions and orientation preserved. No audio added."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def annotations(archive_path, accept_member):
    rows = {}
    with ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not accept_member(member):
                continue
            for number, line in enumerate(archive.read(member).decode("utf-8-sig").splitlines(), 1):
                source, *_ = line.split("|")
                rows.setdefault(source.strip(), []).append({
                    "member": member, "line_number_1_based": number, "exact_line": line,
                })
    return rows


def verified_annotations(sample, phrase, gloss, supplement, split):
    source = sample["source_path"]
    text_rows = supplement.get(source, [])
    split_rows = split.get(source, [])
    def normalize_gloss(value):
        return " ".join(value.replace(";", " ").split())

    return (
        len(text_rows) == 2 and len(split_rows) == 1
        and {row["member"] == "merged_continuous.csv" for row in text_rows} == {True, False}
        and all(row["exact_line"].split("|")[1].strip() == phrase
                and normalize_gloss(row["exact_line"].split("|")[2]) == gloss for row in text_rows)
        and split_rows[0]["exact_line"].split("|")[1].strip() == gloss
    )


def encode(sample, destination):
    paths = [ROOT / path for path in sample["frame_paths"]]
    frame_hashes = [sha(path) for path in paths]
    # The continuous manifest uses concatenated binary digests; the isolated
    # manifests use concatenated ASCII hex. Recompute the recorded convention.
    digests = ("".join(frame_hashes).encode("ascii") if "frame_sha256" in sample
               else b"".join(bytes.fromhex(value) for value in frame_hashes))
    sequence_hash = hashlib.sha256(digests).hexdigest()
    require(frame_hashes == sample.get("frame_sha256", frame_hashes) and sequence_hash == sample["sha256"],
            f"Source frame hash mismatch: {sample['id']}")
    require(len(paths) == sample["frame_count"] and sample["fps"] == 30
            and all(path.parent == paths[0].parent for path in paths)
            and [path.name for path in paths] == [f"frame_{n:04d}.jpg" for n in range(len(paths))],
            f"Incomplete frame sequence: {sample['id']}")
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "30", "-start_number", "0", "-i", str(paths[0].parent / "frame_%04d.jpg"),
        "-frames:v", str(len(paths)), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(destination),
    ], check=True)
    frames = imageio_ffmpeg.read_frames(str(destination), pix_fmt="rgb24")
    metadata = next(frames)
    decoded = sum(1 for _ in frames)
    require(decoded == len(paths) and metadata["codec"] == "h264"
            and tuple(metadata["size"]) == (648, 480) and abs(metadata["fps"] - 30) < 0.01
            and abs(metadata["duration"] - decoded / 30) < 0.05,
            f"Could not decode every original frame: {sample['id']}")
    return {
        "source_frame_count": len(paths),
        "source_frame_sequence_sha256": sequence_hash,
        "mp4_sha256": sha(destination),
        "decoded_frame_count": decoded,
        "duration_seconds": decoded / 30,
        "verified_decodable": True,
    }


def main():
    original_review = read_json(ORIGINAL_REVIEW)
    source = original_review["source"]
    require(source["license"]["id"] == "cc-by-4.0" and source["access_right"] == "open",
            "Expected the original open CC BY 4.0 dataset record")
    old = read_json(LIBRARY / "catalog.json")
    by_id = {entry["id"]: entry for entry in old["entries"]}
    originals = {clip["source_path"]: clip for clip in original_review["clips"]}
    entries = []
    clips = []
    for identifier in ORIGINAL_IDS:
        entry = by_id[identifier]
        reviewed = originals[entry["sourceUtterance"]]
        require(sha(LIBRARY / f"{identifier}.mp4") == entry["sha256"] == reviewed["mp4_sha256"]
                and entry["phrase"] == reviewed["utterance_text"].strip()
                and entry["annotationLabel"] == reviewed["annotation_raw_glosses"],
                f"Original library entry changed: {identifier}")
        entries.append(entry)
        clips.append({
            "id": identifier, "phrase": entry["phrase"], "annotation_label": entry["annotationLabel"],
            "source_path": entry["sourceUtterance"], "mp4_sha256": entry["sha256"],
            "verified_decodable": reviewed["verified_decodable"],
            "official_supplementary_rows": reviewed["official_supplementary_rows"],
            "official_cleaned_split_rows": reviewed["official_cleaned_split_rows"],
        })

    supplement = annotations(
        ANNOTATIONS / "supplementary.zip",
        lambda member: member == "merged_continuous.csv" or member.startswith("extracted_annotations_"),
    )
    split = annotations(
        ANNOTATIONS / "GSL_split.zip",
        lambda member: "GSL-SD-" in member and member.endswith((".txt", ".csv")),
    )
    samples = {}
    for manifest in MANIFESTS:
        for sample in read_json(manifest)["samples"]:
            if sample.get("frame_sha256") and all((ROOT / frame).is_file() for frame in sample["frame_paths"]):
                samples[sample["source_path"]] = sample
    for sample in read_json(CONTINUOUS)["samples"]:
        if all((ROOT / frame).is_file() for frame in sample["frame_paths"]):
            scenario, source_path = sample["source_path"].split("/", 1)
            archive_name = f"{scenario}.zip"
            samples[source_path] = {
                **sample, "source_path": source_path, "raw_label": " ".join(sample["glosses"]),
                "utterance_text": sample["text"], "source_archive": archive_name,
                "source_url": f"{source['record_url']}/files/{archive_name}",
            }
    selected = []
    for identifier, phrase, gloss in ADDITIONS + PHRASES:
        candidates = [
            sample for sample in samples.values()
            if sample["raw_label"] == gloss and sample["utterance_text"].strip() == phrase
            and verified_annotations(sample, phrase, gloss, supplement, split)
        ]
        require(candidates, f"No complete exact annotated utterance for {phrase}")
        sample = min(candidates, key=lambda item: (-item["frame_count"], item["source_path"]))
        selected.append((identifier, phrase, gloss, sample, "phrase" if " " in gloss else "word"))

    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="library-", dir=REVIEW.parent) as temporary:
        staging = Path(temporary)
        for identifier, phrase, gloss, sample, content_kind in selected:
            filename = f"{identifier}.mp4"
            evidence = encode(sample, staging / filename)
            source_path = sample["source_path"]
            clips.append({
                "id": identifier, "phrase": phrase, "annotation_label": gloss, "source_path": source_path,
                "content_kind": content_kind,
                "source_archive": sample["source_archive"], "source_url": sample["source_url"],
                "source_sample_id": sample["id"], "official_supplementary_rows": supplement[source_path],
                "official_cleaned_split_rows": split[source_path], **evidence,
            })
            entries.append({
                "id": identifier, "phrase": phrase, "aliases": [], "annotationLabel": gloss,
                "contentKind": content_kind,
                "videoUrl": f"/library/{filename}", "durationSeconds": evidence["duration_seconds"],
                "sourceUrl": source["record_url"], "sourceArchiveUrl": sample["source_url"],
                "sourceUtterance": source_path, "sha256": evidence["mp4_sha256"],
                "attribution": entries[0]["attribution"], "license": entries[0]["license"],
                "modifications": CHANGES, "rights": entries[0]["rights"],
                "verification": {
                    "status": "dataset-annotated", "basis": VERIFICATION, "newExpertReview": False,
                },
                "contentNote": "Ολόκληρο επισημασμένο απόσπασμα του GSL, στην αρχική σειρά και ταχύτητα.",
            })

        report = {
            "schema_version": 1,
            "status": original_review["status"],
            "source": source,
            "rights": original_review["rights"],
            "citation": original_review["citation"],
            "original_review_sha256": sha(ORIGINAL_REVIEW),
            "annotation_evidence": {
                "supplementary_zip_sha256": sha(ANNOTATIONS / "supplementary.zip"),
                "official_split_zip_sha256": sha(ANNOTATIONS / "GSL_split.zip"),
                "selection": "Exact Greek text and gloss sequence in both supplementary files, and same gloss sequence in the SD split. Only annotation punctuation is normalized. Prefer longer complete cached clips, with source path to break ties.",
                "new_expert_review_performed": False,
            },
            "clips": clips,
        }
        staged_review = staging / REVIEW.name
        write_json(staged_review, report)
        catalog = {
            **old, "scope": "Exact lookup of complete GSL utterances with original dataset annotations.",
            "reviewSha256": sha(staged_review), "entries": entries,
        }
        write_json(staging / "catalog.json", catalog)
        notice = (
            "SIGNBRIDGE GSL VIDEO LIBRARY\n\n" + entries[0]["attribution"] + "\n"
            "Source: https://zenodo.org/records/4756317\n"
            "DOI: https://doi.org/10.5281/zenodo.4756317\n"
            "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n"
            "Full terms: https://creativecommons.org/licenses/by/4.0/legalcode.en\n\n"
            + CHANGES + "\n"
            "Exact source annotations are used. No new expert review or endorsement is claimed.\n"
            "Supplied without warranties.\n\n"
        )
        notice += "\n\n".join(
            f"{entry['videoUrl']}: {entry['phrase']} / {entry['annotationLabel']}\n"
            f"Source archive: {entry['sourceArchiveUrl']}\n"
            f"Source utterance: {entry['sourceUtterance']}\nSHA256: {entry['sha256']}"
            for entry in entries
        )
        (staging / "NOTICE.txt").write_text(notice + "\n", encoding="utf-8", newline="\n")
        for identifier, *_ in selected:
            name = f"{identifier}.mp4"
            (staging / name).replace(LIBRARY / name)
        staged_review.replace(REVIEW)
        (staging / "NOTICE.txt").replace(LIBRARY / "NOTICE.txt")
        (staging / "catalog.json").replace(LIBRARY / "catalog.json")
    print(json.dumps({"entries": len(entries), "phrases": [entry["phrase"] for entry in entries],
                      "review_sha256": sha(REVIEW)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
