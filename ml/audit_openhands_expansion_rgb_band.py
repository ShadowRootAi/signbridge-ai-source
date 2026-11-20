"""Identify recorded real-video cases that could exercise frozen expansion recovery."""

from pathlib import Path

from evaluate_openhands_expansion_recovery import CACHE, FREEZE, POLICY, ROOT, digest, read, write_json


REPORTS = ROOT / "reports/pretrained-gsl"
SOURCES = (
    ("rgb-final-browser", ROOT / ".runtime-tmp/recognition-rgb-audit"),
    ("roster-200-certh-rgb", ROOT / "data/isolated-expansion-rgb"),
    ("roster-200-sartinas-rgb", ROOT / "data/isolated-expansion-sartinas"),
)
OUTPUT = ROOT / "reports/expansion-recovery-rgb"


def main():
    policy = read(POLICY)
    frozen = read(FREEZE)
    if frozen["policySha256"] != digest(POLICY):
        raise ValueError("Frozen fit differs from this policy")
    labels = set(read(CACHE / "stgcn-onnx.json")["labels"])
    supported = set(policy["supportedLabels"])
    added = set(policy["expansionGate"]["labels"])
    low, high = frozen["selectedThresholds"]["minimumPrimary"], policy["expansionGate"]["minimumConfidence"]
    relevant, evaluations = [], {}
    for name, media in SOURCES:
        path = REPORTS / f"{name}.json"
        rows = read(path)["rows"]
        matches = []
        for observation in rows:
            worker = observation["diagnostic"]
            if not worker["classifierExecuted"]:
                continue
            best, *_ = worker["topCandidates"]
            if best["gloss"] not in added or not low <= best["score"] < high:
                continue
            original = (media / Path(observation["path"]).name).resolve()
            if not original.is_relative_to(media) or digest(original) != observation["sha256"]:
                raise ValueError(f"Raw video differs from browser run: {original.name}")
            identity, reference = observation["id"], observation["gloss"]
            if reference not in labels:
                raise ValueError(f"Unmapped reference: {identity}")
            matches.append({"id": identity, "signer": observation["signer"], "reference": reference,
                            "primary": best["gloss"], "score": best["score"],
                            "workerReason": worker["rejectionCode"], "rawCorrect": reference == best["gloss"]})
            relative = "../../" + original.relative_to(ROOT).as_posix()
            relevant.append({"id": identity, "gloss": reference, "on_roster": reference in supported,
                             "group": name, "path": relative, "sha256": observation["sha256"]})
        evaluations[name] = {"clips": len(rows), "sourceReportSha256": digest(path), "matches": matches}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Upper bound from previously recorded RGB browser evidence where an unchanged primary winner meets the candidate score band. "
                 "No claim that new recovery was executed or passed these cases. Other gates, motion and video domain still apply.",
        "freezeSha256": digest(FREEZE), "scoreBand": {"inclusive": low, "exclusive": high},
        "cohorts": evaluations, "totalBandMatches": len(relevant),
        "knownWrongBandMatches": sum(not match["rawCorrect"] and match["reference"] in supported
                                     for evaluation in evaluations.values() for match in evaluation["matches"]),
        "outsideBandMatches": sum(match["reference"] not in supported
                                  for evaluation in evaluations.values() for match in evaluation["matches"]),
    }
    write_json(OUTPUT / "source-score-band.json", report)
    write_json(OUTPUT / "near-miss-fixtures.json", {
        "scope": "All minimum-to-primary-floor expansion outputs in three previously evaluated browser cohorts. "
                 "Diagnostic mechanistic samples selected with earlier scores, not final effectiveness evidence.",
        "clips": relevant,
    })
    print(f"Frozen expansion score band: {len(relevant)} candidate clips across {len(evaluations)} existing browser cohorts")


if __name__ == "__main__":
    main()
