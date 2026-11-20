"""Index verified local GSL assets and official metadata in one SQLite database."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import file_digest, md5, sha256
import json
from math import isfinite
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np

from plan_continuous_composition import ROOT, normalize_greek, read_annotations
from prepare_elementary_catalog_plan import SOURCE_DIR, workbook_rows
from prepare_synthetic_continuous_train_plan import ISOLATE_PATH, SENTENCE_PATH, grouped_labels


DATA = ROOT / "data"
SPLIT = DATA / "original_annotations/GSL_split.zip"
SUPPLEMENT = DATA / "original_annotations/supplementary.zip"
POSES = ROOT / "research/cache/openhands-gsl"
CATALOG = ROOT / "web/public/library/catalog.json"
ELEMENTARY = DATA / "elementary-translation-pilot/manifest.json"
REFERENCES = DATA / "sign-reference-videos/manifest.json"
OUTPUT = DATA / "gsl-provenance.sqlite"
ROLES = {"train": "train", "val": "development", "test": "held", "final_test": "held"}

SCHEMA = """
CREATE TABLE corpus (id TEXT PRIMARY KEY, source_url TEXT NOT NULL, license TEXT NOT NULL);
CREATE TABLE artifact (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL);
CREATE TABLE unit (
  key TEXT PRIMARY KEY, corpus TEXT NOT NULL REFERENCES corpus, kind TEXT NOT NULL,
  source_id TEXT NOT NULL, signer TEXT NOT NULL, scenario TEXT,
  UNIQUE(corpus, kind, source_id)
);
CREATE TABLE annotation (
  unit_key TEXT REFERENCES unit, kind TEXT NOT NULL, ordinal INTEGER NOT NULL,
  value TEXT NOT NULL, lookup TEXT NOT NULL, evidence TEXT NOT NULL REFERENCES artifact, locator TEXT NOT NULL,
  PRIMARY KEY(unit_key, kind, ordinal)
);
CREATE TABLE membership (
  unit_key TEXT REFERENCES unit, policy TEXT NOT NULL, role TEXT NOT NULL, evidence TEXT NOT NULL REFERENCES artifact,
  PRIMARY KEY(unit_key, policy)
);
CREATE TABLE asset (
  unit_key TEXT REFERENCES unit, kind TEXT NOT NULL, local_path TEXT NOT NULL, locator TEXT NOT NULL,
  frames INTEGER, fps REAL, payload_sha256 TEXT, evidence TEXT NOT NULL REFERENCES artifact,
  PRIMARY KEY(unit_key, kind, local_path, locator)
);
CREATE TABLE containment (
  parent TEXT REFERENCES unit, ordinal INTEGER, child TEXT UNIQUE REFERENCES unit,
  PRIMARY KEY(parent, ordinal)
);
CREATE TABLE catalog_entry (
  id TEXT PRIMARY KEY, unit_key TEXT REFERENCES unit, display_text TEXT NOT NULL,
  lookup TEXT NOT NULL, evidence TEXT NOT NULL REFERENCES artifact
);
CREATE TABLE annotation_conflict (
  unit_key TEXT REFERENCES unit, evidence TEXT REFERENCES artifact, kind TEXT NOT NULL,
  expected_json TEXT NOT NULL, observed_json TEXT NOT NULL,
  PRIMARY KEY(unit_key, evidence, kind)
);
CREATE TABLE index_info (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE INDEX annotation_lookup ON annotation(kind, lookup);
CREATE INDEX membership_policy ON membership(policy, role);
CREATE INDEX asset_kind ON asset(kind, payload_sha256);
CREATE INDEX asset_payload ON asset(payload_sha256) WHERE payload_sha256 IS NOT NULL;
CREATE INDEX unit_signer ON unit(corpus, signer, kind);
CREATE INDEX catalog_lookup ON catalog_entry(lookup);
CREATE VIEW reference_library AS
SELECT u.key AS unit_key, u.corpus, n.value AS greek_text, n.lookup,
       a.kind AS asset_kind, a.local_path, a.locator
FROM unit u JOIN annotation n ON n.unit_key = u.key JOIN asset a ON a.unit_key = u.key
WHERE n.kind = 'greek_translation' AND a.kind IN ('public_video', 'reference_video');
CREATE VIEW fit_pool AS
SELECT u.key AS unit_key, u.corpus, u.kind, u.signer
FROM unit u
WHERE ((
  u.kind = 'isolated' AND u.signer IN ('1', '2', '4', '5')
  AND EXISTS (SELECT 1 FROM asset a WHERE a.unit_key = u.key AND a.kind IN ('native_pose', 'native_pose_expanded'))
  AND EXISTS (SELECT 1 FROM membership m WHERE m.unit_key = u.key
              AND m.policy = 'official-isolated' AND m.role = 'train')
  AND NOT EXISTS (SELECT 1 FROM containment c JOIN membership m ON m.unit_key = c.parent
                  WHERE c.child = u.key AND m.policy LIKE 'local:%' AND m.role != 'train')
) OR (
  u.kind IN ('continuous', 'translation_reference')
  AND EXISTS (SELECT 1 FROM asset a WHERE a.unit_key = u.key AND a.kind = 'mediapipe')
  AND EXISTS (SELECT 1 FROM membership m WHERE m.unit_key = u.key
              AND m.policy LIKE 'local:%' AND m.role = 'train')
  AND NOT EXISTS (SELECT 1 FROM membership m WHERE m.unit_key = u.key
                  AND m.policy LIKE 'local:%' AND m.role != 'train')
))
AND NOT EXISTS (SELECT 1 FROM annotation_conflict c WHERE c.unit_key = u.key)
AND NOT EXISTS (
  SELECT 1 FROM asset a JOIN asset b ON a.payload_sha256 = b.payload_sha256 AND a.unit_key != b.unit_key
  JOIN membership m ON m.unit_key = b.unit_key
  WHERE a.unit_key = u.key AND m.role != 'train'
    AND (m.policy = 'official-isolated' OR m.policy LIKE 'local:%')
);
"""


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=None)
def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def register(db, path):
    name = relative(path)
    db.execute("INSERT OR IGNORE INTO artifact VALUES (?, ?, ?)", (name, digest(path), path.stat().st_size))
    return name


def key(corpus, kind, identity):
    return f"certh:{kind}:{identity}" if corpus == "certh" else f"elementary23:{identity}"


def annotation(db, identity, kind, rank, value, evidence, locator):
    db.execute("INSERT INTO annotation VALUES (?, ?, ?, ?, ?, ?, ?)",
               (identity, kind, rank, value, normalize_greek(value), evidence, locator))


def official_sources(db):
    annotations = register(db, SPLIT)
    supplement = register(db, SUPPLEMENT)
    canonical, greek = read_annotations(SPLIT, SUPPLEMENT)
    isolates = {}
    with ZipFile(SPLIT) as archive:
        for kind, pattern, policy in (
            ("isolated", ISOLATE_PATH, "official-isolated"),
            ("continuous", SENTENCE_PATH, "official-continuous-sd"),
        ):
            for role in ("train", "val", "test"):
                member = (f"GSL_split/GSL_isolated/{role}_greek_iso.csv" if kind == "isolated"
                          else f"GSL_split/GSL_continuous/GSL-SD-{role}.txt")
                lines = archive.read(member).decode("utf-8-sig").splitlines()
                for number, line in enumerate(lines, 1):
                    if not line:
                        continue
                    source, label = line.split("|", 1)
                    match = pattern.fullmatch(source)
                    if not match:
                        raise ValueError(f"Invalid official source ID: {source}")
                    identity = key("certh", kind, source)
                    scene = match["session"].split("_", 1)[0]
                    db.execute("INSERT INTO unit VALUES (?, ?, ?, ?, ?, ?)",
                               (identity, "certh", kind, source, match["signer"], scene))
                    db.execute("INSERT INTO membership VALUES (?, ?, ?, ?)",
                               (identity, policy, ROLES[role], annotations))
                    if kind == "isolated":
                        isolates[source] = (label, ROLES[role])
                        tokens = (label,)
                    else:
                        tokens = canonical[source]
                        if text := greek.get(source):
                            annotation(db, identity, "greek_translation", 0, text, supplement,
                                       f"merged_continuous.csv#{source}")
                    for rank, token in enumerate(tokens):
                        annotation(db, identity, "gloss", rank, token, annotations, f"{member}:{number}")
    sentence_sessions = grouped_labels(canonical.items(), SENTENCE_PATH)
    isolate_sessions = grouped_labels(((source, entry[0]) for source, entry in isolates.items()), ISOLATE_PATH)
    for session, sentences in sorted(sentence_sessions.items()):
        segments = isolate_sessions[session]
        if sorted(sentences) != list(range(len(sentences))) or sorted(segments) != list(range(len(segments))):
            continue
        ordered = [token for index in sorted(sentences) for token in sentences[index]]
        if ordered != [segments[index] for index in sorted(segments)]:
            continue
        cursor = 0
        for sentence, glosses in sorted(sentences.items()):
            parent = key("certh", "continuous", f"{session}_sentences/sentences{sentence:04d}")
            for rank in range(len(glosses)):
                child = key("certh", "isolated", f"{session}_glosses/glosses{cursor:04d}")
                db.execute("INSERT INTO containment VALUES (?, ?, ?)", (parent, rank, child))
                cursor += 1
    return canonical, greek, isolates


def elementary_sources(db):
    workbook = SOURCE_DIR / "All_Elementary.xlsx"
    metadata = SOURCE_DIR / "elementary23-zenodo-metadata.json"
    register(db, metadata)
    official = load_json(metadata)
    expected = next(item["checksum"] for item in official["files"] if item["key"] == workbook.name)
    if official["id"] != 7847052 or expected != f"md5:{md5(workbook.read_bytes()).hexdigest()}":
        raise ValueError("Elementary workbook differs from the official release")
    evidence = register(db, workbook)
    labels = {}
    for row, source, signer, text in workbook_rows(workbook):
        identity = key("elementary23", "translation_reference", source)
        db.execute("INSERT INTO unit VALUES (?, ?, ?, ?, ?, ?)",
                   (identity, "elementary23", "translation_reference", source, signer, None))
        annotation(db, identity, "greek_translation", 0, text, evidence, f"sheet1:{row}")
        labels[source] = (signer, text, row)
    return labels


def add_asset(db, unit, kind, path, locator, count, fps, checksum, evidence):
    db.execute("INSERT OR IGNORE INTO asset VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (unit, kind, relative(path), str(locator), count, fps, checksum, evidence))


def native_poses(db, labels):
    known_payloads = {}
    for name, role, fit_signers, reps in (
        ("train-prototypes", "train", {"1", "2", "4"}, {"1"}),
        ("val", "development", set(), set()),
        ("test", "held", set(), set()),
        ("train-expanded", "train", {"1", "2", "5"}, {"1", "2", "3"}),
    ):
        path = POSES / f"official-{name}-poses.npz"
        if name == "train-expanded" and not path.is_file():
            continue
        evidence = register(db, path)
        with np.load(path, allow_pickle=False) as cache:
            ids, glosses, offsets, points = (cache[item] for item in ("ids", "glosses", "offsets", "keypoints"))
            if (cache["split"].item() != name or len(set(ids)) != len(ids) or len(ids) != len(glosses)
                    or len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(points)
                    or points.shape[1:] != (75, 3) or not np.isfinite(points).all() or np.any(np.diff(offsets) <= 0)):
                raise ValueError(f"Unexpected pose archive: {path.name}")
            for row, (source, gloss, frames) in enumerate(zip(ids, glosses, np.diff(offsets), strict=True)):
                if labels[str(source)] != (gloss, role):
                    raise ValueError(f"Pose source changed official split or label: {source}")
                match = ISOLATE_PATH.fullmatch(str(source))
                if fit_signers and (match["signer"] not in fit_signers or match["repetition"] not in reps):
                    raise ValueError(f"Unexpected native fit take: {source}")
                checksum = sha256(points[offsets[row]:offsets[row + 1]].tobytes()).hexdigest()
                if known_payloads.setdefault(source, checksum) != checksum:
                    raise ValueError(f"Two pose caches differ for the same official clip: {source}")
                representation = "native_pose_expanded" if name == "train-expanded" else "native_pose"
                add_asset(db, key("certh", "isolated", source), representation, path, row,
                          int(frames), None, checksum, evidence)


def source_identity(row, elementary):
    if elementary:
        source = Path(row["source_path"]).stem
        return key("elementary23", "translation_reference", source), source
    source = row["source_path"]
    if not SENTENCE_PATH.fullmatch(source):
        source = source.split("/", 1)[1]
    if not SENTENCE_PATH.fullmatch(source):
        raise ValueError(f"Invalid local CERTH source path: {source}")
    return key("certh", "continuous", source), source


def local_recordings(db, canonical, greek, elementary_labels):
    manifests = sorted({*DATA.glob("continuous-*/manifest.json"), *DATA.glob("appointment-dialogues*/manifest.json"),
                        DATA / "manifest.json", DATA / "open-set-v2/manifest.json", DATA / "v3/manifest.json", ELEMENTARY})
    for path in manifests:
        evidence = register(db, path)
        samples = load_json(path)["samples"]
        is_elementary = path == ELEMENTARY
        active = {}
        for row in samples:
            unit, source = source_identity(row, is_elementary)
            active[row["id"]] = (unit, row)
            signer = str(row.get("signer_id", row.get("signer")))
            if is_elementary:
                if elementary_labels[source][:2] != (signer, row["original_text"]):
                    raise ValueError(f"Elementary source identity changed: {source}")
            else:
                if source not in canonical or SENTENCE_PATH.fullmatch(source)["signer"] != signer:
                    raise ValueError(f"CERTH source identity changed: {source}")
                differences = [
                    ("gloss", list(canonical[source]), row.get("glosses")),
                    ("greek_translation", greek.get(source), row.get("text")),
                ]
                for kind, expected, observed in differences:
                    if observed is not None and observed != expected:
                        db.execute("INSERT INTO annotation_conflict VALUES (?, ?, ?, ?, ?)",
                                   (unit, evidence, kind, json.dumps(expected, ensure_ascii=False),
                                    json.dumps(observed, ensure_ascii=False)))
            db.execute("INSERT INTO membership VALUES (?, ?, ?, ?)",
                       (unit, f"local:{path.parent.name}", ROLES[row["split"]], evidence))
            frames = [ROOT / name for name in row.get("frame_paths", [])]
            if frames and all(frame.is_file() for frame in frames):
                frame_digests = [digest(frame) for frame in frames]
                if row.get("frame_sha256") is not None and row["frame_sha256"] != frame_digests:
                    raise ValueError(f"RGB frame digest changed: {source}")
                binary = sha256(b"".join(bytes.fromhex(value) for value in frame_digests)).hexdigest()
                ascii_hex = sha256("".join(frame_digests).encode("ascii")).hexdigest()
                if row.get("sha256") and row["sha256"] not in {binary, ascii_hex}:
                    raise ValueError(f"RGB clip digest changed: {source}")
                add_asset(db, unit, "rgb_frames", frames[0].parent, row["id"], len(frames), row["fps"], binary, evidence)
                if is_elementary:
                    original = frames[0].parent / "source.flv"
                    if not original.is_file() or digest(original) != row["source_sha256"]:
                        raise ValueError(f"Original Elementary video changed: {source}")
                    add_asset(db, unit, "source_video", original, row["id"], None, None, row["source_sha256"], evidence)
        feature_path = path.parent / "features.jsonl"
        if not feature_path.is_file():
            continue
        feature_evidence = register(db, feature_path)
        seen = set()
        with feature_path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                feature = json.loads(line)
                identity = feature["id"]
                unit, row = active[identity]
                matrix = feature["features"]
                if (identity in seen or feature["feature_schema"] != "signbridge-landmarks-xy-v1"
                        or feature["fps"] != 15 or not matrix
                        or any(len(frame) != 224 or not all(isfinite(value) for value in frame) for frame in matrix)):
                    raise ValueError(f"Malformed feature row: {identity}")
                if "source_key" in feature:
                    source_ref = json.dumps([row["fps"], row["frame_paths"]], ensure_ascii=False, separators=(",", ":"))
                    if sha256(source_ref.encode()).hexdigest() != feature["source_key"]:
                        raise ValueError(f"Feature source provenance changed: {identity}")
                seen.add(identity)
                add_asset(db, unit, "mediapipe", feature_path, identity, len(matrix), 15, None, feature_evidence)


def public_assets(db):
    evidence = register(db, CATALOG)
    for row in load_json(CATALOG)["entries"]:
        source = row["sourceUtterance"]
        if row["sourceUrl"].endswith("/4756317"):
            unit = key("certh", "continuous", source)
        elif row["sourceUrl"].endswith("/7847052"):
            unit = key("elementary23", "translation_reference", Path(source).stem)
        else:
            raise ValueError(f"Unverified catalog corpus: {row['id']}")
        path = ROOT / "web/public" / row["videoUrl"].lstrip("/")
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"Public catalog video changed: {row['id']}")
        add_asset(db, unit, "public_video", path, row["id"], None, None, row["sha256"], evidence)
        db.execute("INSERT INTO catalog_entry VALUES (?, ?, ?, ?, ?)",
                   (row["id"], unit, row["phrase"], normalize_greek(row["phrase"]), evidence))


def additional_references(db, elementary_labels):
    if not REFERENCES.is_file():
        return
    evidence = register(db, REFERENCES)
    rows = load_json(REFERENCES)["clips"]
    for row in rows:
        source = row["source_id"]
        signer, original, sheet_row = elementary_labels[source]
        if (str(row["signer"]) != signer or row["workbook_row"] != sheet_row
                or normalize_greek(row["greek_text"]) != normalize_greek(original)
                or row["reference_type"] != "greek_translation"):
            raise ValueError(f"Reference video differs from official Elementary annotation: {source}")
        path = ROOT / row["path"]
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"Elementary reference video changed: {source}")
        add_asset(db, key("elementary23", "translation_reference", source), "reference_video",
                  path, row["id"], row["frames"], None, row["sha256"], evidence)


def summarize(db):
    def counts(sql):
        return {":".join(map(str, row[:-1])): row[-1] for row in db.execute(sql)}

    return {
        "indexed_sources": counts("SELECT corpus, kind, COUNT(*) FROM unit GROUP BY corpus, kind"),
        "official_gloss_types": counts("SELECT u.kind, COUNT(DISTINCT a.value) FROM unit u JOIN annotation a"
                                       " ON a.unit_key=u.key WHERE a.kind='gloss' GROUP BY u.kind"),
        "local_sources": counts("SELECT u.corpus, u.kind, COUNT(DISTINCT u.key) FROM unit u JOIN asset a"
                                " ON a.unit_key=u.key GROUP BY u.corpus, u.kind"),
        "local_asset_sources": counts("SELECT kind, COUNT(DISTINCT unit_key) FROM asset GROUP BY kind"),
        "fit_candidate_sources": counts("SELECT corpus, kind, COUNT(*) FROM fit_pool GROUP BY corpus, kind"),
        "official_pose_role_sources": counts("SELECT m.role, COUNT(DISTINCT m.unit_key) FROM membership m"
                                             " JOIN asset a ON a.unit_key=m.unit_key"
                                             " WHERE m.policy='official-isolated' AND a.kind IN ('native_pose','native_pose_expanded')"
                                             " GROUP BY m.role"),
        "official_pose_role_signers": counts("SELECT m.role, u.signer, COUNT(DISTINCT m.unit_key) FROM membership m"
                                             " JOIN unit u ON u.key=m.unit_key JOIN asset a ON a.unit_key=m.unit_key"
                                             " WHERE m.policy='official-isolated' AND a.kind IN ('native_pose','native_pose_expanded')"
                                             " GROUP BY m.role,u.signer"),
        "cross_local_role_sources_excluded": db.execute("SELECT COUNT(*) FROM (SELECT unit_key FROM membership"
            " WHERE policy LIKE 'local:%' GROUP BY unit_key HAVING SUM(role='train')>0 AND SUM(role!='train')>0)").fetchone()[0],
        "annotation_conflict_sources_excluded": db.execute("SELECT COUNT(DISTINCT unit_key) FROM annotation_conflict").fetchone()[0],
        "ordered_isolate_links": db.execute("SELECT COUNT(*) FROM containment").fetchone()[0],
        "public_entries": db.execute("SELECT COUNT(*) FROM catalog_entry").fetchone()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(DATA) or output.suffix != ".sqlite":
        parser.error("Use an SQLite output inside the project's data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix="gsl-index-", suffix=".sqlite", dir=output.parent, delete=False) as file:
        staging = Path(file.name)
    try:
        db = sqlite3.connect(staging)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            db.executemany("INSERT INTO corpus VALUES (?, ?, 'CC-BY-4.0')",
                           [("certh", "https://zenodo.org/records/4756317"),
                            ("elementary23", "https://zenodo.org/records/7847052")])
            canonical, greek, isolates = official_sources(db)
            elementary = elementary_sources(db)
            native_poses(db, isolates)
            local_recordings(db, canonical, greek, elementary)
            public_assets(db)
            additional_references(db, elementary)
            report = summarize(db)
            db.executemany("INSERT INTO index_info VALUES (?, ?)", (
                ("version", "1"),
                ("annotation_semantics", json.dumps("Official gloss and Greek translation are separate. Containment links verify label order, not timing.")),
                ("fit_pool_semantics", json.dumps("Available fit candidates. A specific experiment must still enforce held signer, class and scenario exclusions.")),
                ("counts", json.dumps(report, sort_keys=True)),
            ))
            db.commit()
            db.execute("VACUUM")
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("The provenance index did not pass SQLite integrity validation")
        finally:
            db.close()
        staging.replace(output)
    finally:
        staging.unlink(missing_ok=True)
    print(json.dumps({"index": relative(output), **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
