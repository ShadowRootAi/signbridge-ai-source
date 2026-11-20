"""Plan an exact-word video catalog from the official Elementary v3 workbook."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "research/external-gsl-metadata"
RECORD_URL = "https://zenodo.org/records/7847052"
ARCHIVE = "Greek Language.zip"
BASELINE_CATALOG = ROOT / "web/deployment/expanded-library/catalog.json"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
GREEK_WORD = re.compile(r"[α-ωάέήίόύώϊϋΐΰ]+")
VIDEO_ID = re.compile(r"[a-z][a-z0-9_]+")

# Ordered Greek lookup words, chosen for practical vocabulary. A word is
# eligible only when an original Greek Language clip has that whole annotation.
WORDS = """
καληνύχτα εγώ εσύ εμείς εσείς πρωί απόγευμα αύριο χτες αρχή τέλος λάθος χαρά
άνθρωπος παιδί κορίτσι μαμά γιαγιά θεία θείος γείτονας δασκάλα
μιλώ επικοινωνώ διαβάζω γράφω μαθαίνω παίζω τραγουδώ γελώ χορεύω σκέφτομαι
φοβάμαι ψάχνω ανοίγω κρυώνω μπαίνω βγαίνω δουλεύω ταξιδεύω χαίρομαι ζωγραφίζω
πηγαίνω σηκώνω ζητώ προσέχω ξεχνώ μυρίζω αγγίζω
σπίτι σχολείο τάξη βιβλίο μολύβι ψαλίδι γράμμα κλειδί τσάντα οδοντόβουρτσα
καπέλο πόρτα ψυγείο φλιτζάνι δώρο τηλεόραση πετσέτα παπούτσι γάντια κατσαρόλα
βιβλιοθήκη κήπος φωτογραφία ταχυδρομείο πλατεία πόλη γειτονιά δρόμος αυλή
νερό γάλα μήλο ψωμί τυρί πατάτα σαλάτα ζάχαρη σοκολάτα αυγό βούτυρο αλεύρι
μακαρόνια σταφύλι πεπόνι πίτα καραμέλα
αυτοκίνητο τρένο ποδήλατο λεωφορείο πλοίο αεροπλάνο φορτηγό μηχανάκι βάρκα
θάλασσα νησί βουνό λίμνη ποτάμι βροχή ήλιος αέρας φως ουρανός φεγγάρι άμμος
καλοκαίρι χειμώνας άνοιξη φθινόπωρο κρύο λουλούδι χώμα
σκύλος γάτα ψάρι πουλί χελώνα αλεπού ελέφαντας πεταλούδα μέλισσα
άσπρος μπλε όμορφος γρήγορος θόρυβος ήχος χρώμα κασέλα φτυάρι
""".split()


def word_key(text):
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def single_word(transcription):
    word = unicodedata.normalize("NFC", transcription.strip().rstrip("!?;.,·…").strip())
    return word.lower() if GREEK_WORD.fullmatch(word.lower()) else None


def workbook_rows(path):
    with zipfile.ZipFile(path) as workbook:
        shared = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = ["".join(node.text or "" for node in item.findall(".//s:t", NS))
                   for item in shared]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    for row in sheet.findall(".//s:row", NS):
        cells = {}
        for cell in row.findall("s:c", NS):
            column = cell.attrib["r"].rstrip("0123456789")
            index = cell.findtext("s:v", namespaces=NS)
            if cell.get("t") == "s" and index is not None:
                cells[column] = strings[int(index)]
        if row.attrib["r"] == "1":
            if tuple(cells.get(column) for column in "ABC") != (
                "Video_ID", "Signer ID", "Transcription"
            ):
                raise ValueError("Unexpected Elementary workbook columns")
        elif all(cells.get(column, "").strip() for column in "ABC"):
            yield int(row.attrib["r"]), cells["A"].strip(), cells["B"].strip(), cells["C"]


def build_plan(workbook, metadata, existing_catalog, limit):
    source = json.loads(metadata.read_text(encoding="utf-8"))
    details = source["metadata"]
    if source["id"] != 7847052 or details["access_right"] != "open" or details["license"]["id"] != "cc-by-4.0":
        raise ValueError("Expected official open Elementary v3 record under CC BY 4.0")

    files = {file["key"]: file for file in source["files"]}
    workbook_bytes = workbook.read_bytes()
    workbook_md5 = hashlib.md5(workbook_bytes).hexdigest()
    if files[workbook.name]["checksum"] != f"md5:{workbook_md5}":
        raise ValueError("Workbook does not match the checksum in the official record")

    catalog = json.loads(existing_catalog.read_text(encoding="utf-8"))
    excluded = {word_key(phrase) for entry in catalog["entries"]
                for phrase in [entry["phrase"], *entry.get("aliases", [])]}
    by_word = defaultdict(list)
    all_words = Counter()
    greek_language_rows = 0
    seen_source_ids = set()
    duplicate_source_ids = set()
    row_count = 0
    for row, source_id, signer, transcription in workbook_rows(workbook):
        row_count += 1
        if source_id in seen_source_ids:
            duplicate_source_ids.add(source_id)
        seen_source_ids.add(source_id)
        word = single_word(transcription)
        if word:
            all_words[word] += 1
        if not source_id.startswith("glossa_"):
            continue
        greek_language_rows += 1
        if word and VIDEO_ID.fullmatch(source_id):
            by_word[word].append({
                "id": f"elementary-{source_id}",
                "phrase": word,
                "source_id": source_id,
                "signer": signer,
                "workbook_row": row,
                "transcription": transcription,
                "archive": ARCHIVE,
            })

    entries = []
    unavailable_words = []
    for word in WORDS:
        key = word_key(word)
        if key in excluded:
            continue
        matches = [candidate for candidate in by_word[word]
                   if candidate["source_id"] not in duplicate_source_ids]
        if not matches:
            unavailable_words.append(word)
            continue
        matches.sort(key=lambda candidate: (
            candidate["transcription"].strip().lower() != word, candidate["workbook_row"]
        ))
        selected, *alternatives = matches
        if alternatives:
            selected["alternate_sources"] = alternatives
        entries.append(selected)
        excluded.add(key)
        if len(entries) == limit:
            break

    if len(entries) != limit:
        raise ValueError(f"Only {len(entries)} of {limit} requested entries available: {unavailable_words}")

    archive = files[ARCHIVE]
    return {
        "schemaVersion": 1,
        "scope": "Planning manifest of original clips with exact single-word Greek transcriptions. Media and independent GSL review are pending.",
        "source": {
            "title": details["title"],
            "recordUrl": RECORD_URL,
            "doi": details["doi"],
            "creators": [creator["name"] for creator in details["creators"]],
            "license": {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "workbook": workbook.name,
            "workbookUrl": files[workbook.name]["links"]["self"],
            "workbookSha256": hashlib.sha256(workbook_bytes).hexdigest(),
            "archive": ARCHIVE,
            "archiveUrl": archive["links"]["self"],
            "archiveBytes": archive["size"],
            "archiveMd5": archive["checksum"].removeprefix("md5:"),
        },
        "statistics": {
            "completeWorkbookRows": row_count,
            "singleWordClips": sum(all_words.values()),
            "singleWordForms": len(all_words),
            "greekLanguageRows": greek_language_rows,
            "greekLanguageSingleWordForms": len(by_word),
            "selectedForms": len(entries),
            "selectedSigners": dict(sorted(Counter(entry["signer"] for entry in entries).items())),
        },
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=SOURCE_DIR / "All_Elementary.xlsx")
    parser.add_argument("--metadata", type=Path, default=SOURCE_DIR / "elementary23-zenodo-metadata.json")
    parser.add_argument("--existing-catalog", type=Path, default=BASELINE_CATALOG)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/library-100/elementary-plan.json")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    plan = build_plan(args.workbook, args.metadata, args.existing_catalog, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan["statistics"], ensure_ascii=True))


if __name__ == "__main__":
    main()
