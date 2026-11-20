"""Build the library UI with the static model assets of the audited 200-class preview."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from os import environ
from pathlib import Path
import re
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory
import unicodedata

from acquire_elementary_library_expansion import ARCHIVES, WORKBOOK, official, transcript
from prepare_elementary_catalog_plan import RECORD_URL, ROOT, workbook_rows


BASE = ROOT / "publication-recognition-200"
OUTPUT = ROOT / "publication-video-library"
PUBLIC_OUTPUT = ROOT / "publication-public-library-200"
SEED = ROOT / "data/sign-reference-videos/manifest.json"
COHORTS = ROOT / "data/sign-library-expansion"
WEB = ROOT / "web"
CATALOG = "library/catalog.json"
NOTICE = "library/NOTICE.txt"
JUDGE_OUTPUT = ROOT / "publication-judge-demo"
ASSISTANT_OUTPUT = ROOT / "publication-local-assistant"
JUDGE_MANIFEST = ROOT / "reports/judge-robustness/manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lookup_key(text):
    plain = "".join(char for char in unicodedata.normalize("NFD", text)
                    if not unicodedata.category(char).startswith("M"))
    return " ".join(plain.lower().replace("ς", "σ").strip().rstrip(".!?;;·…").split())


def source_row(rows, number, identity, signer, original, phrase):
    require(rows.get(number) == (identity, signer, original), f"Workbook mismatch for {identity}")
    meaning, kind = transcript(original)
    require(meaning == phrase and kind, f"Reference text mismatch for {identity}")
    return kind


def source_archive(archives, identity, member):
    archive = ARCHIVES[identity.split("_", 1)[0]]
    require(member.startswith(archive.removesuffix(".zip") + "/") and Path(member).stem == identity,
            f"Archive member mismatch for {identity}")
    return archives[archive]["links"]["self"]


def seed_entries(rows, archives, source, workbook_hash):
    manifest = read(SEED)
    require(manifest["source"] == RECORD_URL and manifest["workbook_sha256"] == workbook_hash,
            "Seed library provenance changed")
    attribution = f"{source['title']} v3 — {'; '.join(source['creators'])}."
    for clip in manifest["clips"]:
        kind = source_row(rows, clip["workbook_row"], clip["source_id"], clip["signer"],
                          clip["original_transcription"], clip["greek_text"])
        require(clip["license"] == "CC-BY-4.0" and clip["source_url"] == RECORD_URL
                and clip["frames"] > 0 and clip["duration_seconds"] > 0
                and clip["attribution"] == attribution, f"Incomplete seed evidence for {clip['id']}")
        yield {
            "id": clip["id"], "phrase": clip["greek_text"], "aliases": [], "contentKind": kind,
            "videoUrl": f"/library/{clip['id']}.mp4", "durationSeconds": clip["duration_seconds"],
            "sourceUrl": RECORD_URL,
            "sourceArchiveUrl": source_archive(archives, clip["source_id"], clip["source_path"]),
            "sourceUtterance": clip["source_path"], "sha256": clip["sha256"], "attribution": attribution,
            "license": {**source["license"], "name": "Creative Commons Attribution 4.0 International"},
            "modifications": clip["modifications"],
            "rights": {"redistributionConfirmed": True, "basis": "Official open Zenodo record 7847052, CC BY 4.0."},
            "verification": {"status": "dataset-annotated", "newExpertReview": False,
                             "basis": "Ακριβής επίσημη ελληνική μεταγραφή για αυτό το πλήρες βίντεο."},
            "contentNote": "Πλήρες βίντεο της πηγής με τη σειρά και την ταχύτητα του πρωτοτύπου.",
        }, ROOT / clip["path"]


def cohort_entries(name, rows, archives, source, workbook_hash):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", name), f"Invalid cohort {name}")
    directory = COHORTS / name
    path = directory / "manifest.json"
    manifest = read(path)
    plan_path = directory / "manifest.planned.json"
    plan = read(plan_path)
    original_workbook = plan.get("workbook_sha256")
    if original_workbook is None:
        original_workbook = next((fingerprint for relative, fingerprint in plan["source_hashes"].items()
                                  if (ROOT / relative).resolve() == WORKBOOK.resolve()), None)
    require(manifest["source"] == source and manifest["plan_sha256"] == digest(plan_path)
            and original_workbook == workbook_hash, f"Cohort provenance changed: {name}")
    attribution = f"{source['title']} v3 — {'; '.join(source['creators'])}."
    for clip in manifest["clips"]:
        entry = clip["entry"]
        identity = clip["workbook_video_id"]
        kind = source_row(rows, clip["workbook_row"], identity, clip["signer_id"],
                          clip["workbook_transcription"], clip["phrase"])
        media = directory / "media" / (entry["id"] + ".mp4")
        require((ROOT / clip["path"]).resolve() == media.resolve()
                and clip["verified_decodable"] and clip["frames"] > 0 and clip["duration_seconds"] > 0
                and entry["id"] == clip["id"] and entry["phrase"] == clip["phrase"]
                and entry["contentKind"] == clip["kind"] == kind and entry["sourceUrl"] == RECORD_URL
                and entry["sourceArchiveUrl"] == source_archive(archives, identity, clip["source_path"])
                and entry["sourceUtterance"] == clip["source_path"] and entry["attribution"] == attribution
                and entry["sha256"] == clip["mp4_sha256"] and media.stat().st_size == clip["mp4_bytes"],
                f"Incomplete acquisition evidence for {identity}")
        yield entry, media


def add_entry(entry, path, catalog, library, texts, ids, sources, hashes):
    identity = entry["id"]
    name = identity + ".mp4"
    text = lookup_key(entry["phrase"])
    terms = {lookup_key(term) for term in [entry["phrase"], *entry["aliases"]]}
    source = (entry["sourceUrl"], entry["sourceUtterance"])
    fingerprint = entry["sha256"]
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", identity) and entry["videoUrl"] == "/library/" + name
            and path.resolve().is_relative_to(ROOT) and entry["license"]["id"] == "CC-BY-4.0"
            and entry["rights"]["redistributionConfirmed"] is True and entry["attribution"]
            and entry["verification"]["status"] in {"dataset-annotated", "expert-reviewed"}
            and re.fullmatch(r"[a-f0-9]{64}", fingerprint) and digest(path) == fingerprint,
            f"Invalid library entry or changed media for {identity}")
    require("" not in terms and not texts.intersection(terms) and identity not in ids
            and source not in sources and fingerprint not in hashes,
            f"Duplicate library text, source, id or video for {identity}")
    texts.update(terms)
    ids.add(identity)
    sources.add(source)
    hashes.add(fingerprint)
    copy2(path, library / name)
    catalog.append({**entry, "contentKind": entry.get("contentKind", "phrase" if " " in text else "word")})


def judge_assets(dist):
    manifest = read(JUDGE_MANIFEST)
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    target = dist / "__signbridge/judge"
    target.mkdir(parents=True)
    notices = []
    for identity, name in (("frozen", "frozen.mp4"), ("no-tracking", "no-tracking.mp4"), ("unknown", "unsupported.mp4")):
        clip = clips[identity]
        original = (JUDGE_MANIFEST.parent / clip["path"]).resolve()
        require(original.is_relative_to(ROOT / "reports") and digest(original) == clip["sha256"],
                f"Judge fixture failed verification: {identity}")
        copy2(original, target / name)
        notices.append({"file": name, **{key: clip[key] for key in
                        ("sha256", "source_id", "attribution", "source", "license")},
                        "changes": clip.get("intervention", "Original annotated JPEG sequence encoded as MP4.")})
    write(target / "sources.json", {"scope": manifest["scope"], "clips": notices})


def assemble(stage, cohorts, minimum, judge, frame_offset, assistant, public):
    archives, source = official()
    workbook_hash = digest(WORKBOOK)
    rows = {number: (identity, signer, text) for number, identity, signer, text in workbook_rows(WORKBOOK)}
    audited = read(BASE / "publication-audit.json")["files"]
    base_dist = BASE / "dist"
    dist = stage / "dist"
    subprocess.run([
        "node", "node_modules/vite/bin/vite.js", "build", "--config", "vite.v8-public.config.ts",
        "--mode", "signbridge-local-assistant" if assistant else "signbridge-openhands", "--outDir", str(dist),
    ], cwd=WEB, env={**environ, "VITE_SIGNBRIDGE_CONTINUOUS": "0",
                     "VITE_SIGNBRIDGE_JUDGE_DEMO": "1" if judge else "0",
                     "VITE_SIGNBRIDGE_OPENHANDS_FRAME_OFFSET": "1" if frame_offset else "0"}, check=True)
    library = dist / "library"
    library.mkdir(parents=True)
    for asset in audited:
        relative = asset["path"]
        path = base_dist / relative
        require(path.resolve().is_relative_to(base_dist) and path.stat().st_size == asset["bytes"]
                and digest(path) == asset["sha256"], f"Preview asset failed verification: {relative}")
        if not relative.startswith(("library/", "assets/")) and relative != "index.html":
            destination = dist / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(path, destination)
    if judge:
        judge_assets(dist)

    catalog = []
    texts, ids, sources, hashes = set(), set(), set(), set()
    counts = {}

    def add_group(name, entries):
        before = len(catalog)
        for entry, path in entries:
            add_entry(entry, path, catalog, library, texts, ids, sources, hashes)
        counts[name] = len(catalog) - before

    base_entries = read(base_dist / CATALOG)["entries"]
    add_group("preview", ((entry, base_dist / entry["videoUrl"].lstrip("/")) for entry in base_entries))
    add_group("reference-videos", seed_entries(rows, archives, source, workbook_hash))
    for name in cohorts:
        add_group(name, cohort_entries(name, rows, archives, source, workbook_hash))
    require(len(catalog) >= minimum, f"Only {len(catalog)} verified videos, expected at least {minimum}")

    write(dist / CATALOG, {"schemaVersion": 1, "scope": "Exact whole-video Greek Sign Language lookup",
                           "entries": catalog})
    notice = (base_dist / NOTICE).read_text(encoding="utf-8")
    additions = ["\n\nEXPANDED VIDEO REFERENCES", "Source: " + RECORD_URL,
                 "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/"]
    for entry in catalog[len(base_entries):]:
        additions.extend([f"\n{entry['videoUrl']}: {entry['phrase']}", entry["attribution"],
                          entry["sourceArchiveUrl"], entry["sourceUtterance"], entry["modifications"],
                          f"SHA256: {entry['sha256']}"])
    (dist / NOTICE).write_text(notice.rstrip() + "\n".join(additions) + "\n", encoding="utf-8")
    published_files = sorted(path for path in dist.rglob("*") if path.is_file())
    if public:
        require(len(published_files) <= 20_000 and all(path.stat().st_size <= 25 * 1024**2 for path in published_files),
                "The public build exceeds Cloudflare Pages upload limits")
        require(len(read(dist / "__signbridge/openhands/policy.json")["supportedLabels"]) == 200,
                "The public library must use the verified 200-label recognition policy")
    audit = {
        "status": ("public_static_candidate" if public else "local_assistant" if assistant
                   else "local_judge_demo" if judge else "local_verified_catalog"),
        "openhands_frame_boundary_offset_ms": 2 if frame_offset else 0,
        "entries": len(catalog),
        "counts_by_kind": dict(Counter(entry["contentKind"] for entry in catalog)),
        "counts_by_source": dict(Counter(entry["sourceUrl"] for entry in catalog)),
        "counts_by_group": counts,
        "catalog_sha256": digest(dist / CATALOG),
        "runtime_audit_sha256": digest(BASE / "publication-audit.json"),
        "runtime_files": {path.relative_to(dist).as_posix(): digest(path)
                          for path in published_files if library not in path.parents},
        "workbook_sha256": workbook_hash,
        "media_bytes": sum(path.stat().st_size for path in library.glob("*.mp4")),
        "static_files": len(published_files),
    }
    write(stage / "library-audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts", nargs="*", help="Acquisition cohort directory names")
    parser.add_argument("--minimum", type=int, default=288)
    edition = parser.add_mutually_exclusive_group()
    edition.add_argument("--judge", action="store_true", help="Build the separate local judge edition")
    edition.add_argument("--assistant", action="store_true", help="Build the local Ollama edition")
    edition.add_argument("--public", action="store_true", help="Build the static Cloudflare Pages edition")
    parser.add_argument("--frame-offset", action="store_true", help="Use later frame-boundary video seeks")
    args = parser.parse_args()
    require(args.minimum >= 288 and len(args.cohorts) == len(set(args.cohorts)), "Invalid library build request")
    require(not args.frame_offset or args.judge or args.assistant or args.public,
            "Frame offset requires an explicit edition")
    with TemporaryDirectory(prefix="video-library-", dir=ROOT / ".runtime-tmp") as temp:
        work = Path(temp).resolve()
        require(work.parent == ROOT / ".runtime-tmp", "Library staging must stay in the project")
        stage = work / "new"
        audit = assemble(stage, args.cohorts, args.minimum, args.judge, args.frame_offset, args.assistant, args.public)
        output = PUBLIC_OUTPUT if args.public else ASSISTANT_OUTPUT if args.assistant else JUDGE_OUTPUT if args.judge else OUTPUT
        previous = work / "previous"
        if output.exists():
            require(output.is_dir() and output.resolve() == output, "Unexpected existing library path")
            output.rename(previous)
        try:
            stage.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise
    print(json.dumps({key: value for key, value in audit.items() if key != "runtime_files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
