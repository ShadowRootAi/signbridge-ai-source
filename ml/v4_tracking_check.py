"""Export independent guard fixtures and a bounded, already-seen raw-frame diagnosis."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from test_v4_tracking import frames
from v4_tracking import POLICY, measure_raw_tracking

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/"reports/v4"; output.mkdir(parents=True,exist_ok=True)
    locked=json.loads((output/"tracking-policy.locked.json").read_text(encoding="utf-8"))
    assert locked["policy"]==POLICY
    cases=[]
    def add(name, raw, kind="structural_unit_fixture"):
        cases.append({"id":name,"kind":kind,"raw":raw,"expected":measure_raw_tracking(raw)})
    add("complete_static_four_frames",frames(4));add("empty",[]);add("too_short",frames(3))
    raw=frames();raw[5]["right"]=[];add("exact90_percent_boundary",raw)
    raw=deepcopy(raw);raw[6]["right"]=[];add("below90_percent",raw)
    raw=frames(15)
    for i in (0,1,2,3,7,10):raw[i]["right"]=[]
    add("resting_hand_cannot_hide_dropout",raw)
    raw=frames(5);raw[2]["left"]=[];add("missing_pair_endpoints",raw)
    raw=frames();raw[1]["t"]=raw[0]["t"];add("duplicate_timestamps",raw)
    raw=frames();raw[0]["width"]=0;add("invalid_dimensions",raw)
    raw=frames();raw[0]["right"][0]["x"]=None;raw[1]["right"]=[];add("non_numeric_coordinate",raw)
    raw=frames()
    for f in raw:
        for hand in ("left","right"):
            for p in f[hand]:p["visibility"]=0
    add("hand_visibility_not_confidence",raw)
    sources=[]
    targets={"kep1_signer3_rep1_sentences__sentences0000", "kep1_signer3_rep2_sentences__sentences0000", "kep1_signer3_rep3_sentences__sentences0000",
             "health4_signer3_rep1_sentences__sentences0021", "health4_signer3_rep2_sentences__sentences0021", "health4_signer3_rep3_sentences__sentences0021"}
    for path in sorted((ROOT/"data/open-set-v2/features").glob("final-*.jsonl")):
        sources.append({"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            if r["id"] in targets:
                assert r.get("raw"),r["id"]
                add(r["id"],r["raw"],"already_seen_real_raw_fixture")
    assert sum(c["kind"]=="already_seen_real_raw_fixture" for c in cases)==6
    document={"policy":POLICY,"policy_lock_sha256":hashlib.sha256((output/"tracking-policy.locked.json").read_bytes()).hexdigest(),
              "structural_fixtures_are_not_signing_examples":True,"no_fit_or_threshold_selection":True,"sources":sources,"cases":cases}
    (output/"tracking-parity-fixtures.json").write_text(json.dumps(document,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    report={"scope":"Six retained, already-exposed V2 person3 raw fixtures; not independent evaluation",
            "policy":POLICY,"cases":[{"id":c["id"],"tracking":c["expected"]} for c in cases if c["kind"]=="already_seen_real_raw_fixture"],
            "feature_schema_unchanged":True,"model_and_thresholds_unchanged":True,"python_browser_parity":"pending separate browser run"}
    (output/"tracking-raw-diagnosis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"fixture_cases":len(cases),"real_raw":report["cases"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
