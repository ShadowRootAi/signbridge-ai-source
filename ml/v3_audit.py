"""Independent saved-artifact audit; no imports from fitting or scoring modules."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def normalize(x, model):
    return np.maximum(-8, np.minimum(8, (x-model["mean"])/model["std"])) * model["keep"]


def logits(x, model):
    z = normalize(x, model).reshape(len(x), -1).astype(np.float64)
    d = np.asarray([np.mean((z-c.astype(np.float64))**2, axis=1) for c in model["centroids"]]).T
    return -np.column_stack((d[:, :3], d[:, 3:].min(axis=1)))


def motion(x):
    x = x.astype(float); total = np.zeros(len(x)); pairs = np.zeros(len(x))
    for offset, flag in ((0, 220), (84, 221)):
        good = (x[:, :, flag] >= 1-1e-6) & (x[:, :, 222] >= 1-1e-6)
        usable = good[:, 1:] & good[:, :-1]
        d = x[:, 1:, offset:offset+42] - x[:, :-1, offset:offset+42]
        total += (d*d*usable[:, :, None]).sum(axis=(1, 2)); pairs += usable.sum(axis=1)*21
    return np.sqrt(np.divide(total, pairs, out=np.zeros(len(x)), where=pairs>0))


def probability(l, temperature):
    l = l/temperature; p = np.exp(l-l.max(axis=1, keepdims=True))
    return p/p.sum(axis=1, keepdims=True)


def score(p, quality, movement, gate):
    predicted = p[:, :3].argmax(axis=1); confidence = p[np.arange(len(p)), predicted]
    reasons = {"quality": quality < max(.25, gate["minQuality"]),
               "motion": movement < gate["minMotionBodyRms"],
               "confidence": confidence < gate["confidenceThreshold"], "background": p.argmax(axis=1) == 3}
    accepted = ~np.logical_or.reduce(list(reasons.values()))
    return predicted, confidence, accepted, reasons


def basic(y, pred, accepted):
    known = y>=0; correct = y==pred; ak = accepted & known; un = ~known
    cm = np.zeros((3, 3), int)
    for a, b in zip(y[known], pred[known]): cm[a,b] += 1
    f1 = np.divide(2*np.diag(cm), cm.sum(0)+cm.sum(1), out=np.zeros(3), where=(cm.sum(0)+cm.sum(1))>0)
    ratio = lambda a,b: float(a/b) if b else None
    return {"n":len(y), "known_n":int(known.sum()), "unknown_n":int(un.sum()),
            "closed_set_accuracy":ratio((correct&known).sum(), known.sum()), "closed_set_macro_f1":float(f1.mean()),
            "closed_set_confusion_matrix":cm.tolist(), "per_class_f1":f1.tolist(),
            "accepted_known_n":int(ak.sum()), "accepted_known_accuracy":ratio((correct&ak).sum(), ak.sum()),
            "known_coverage":ratio(ak.sum(), known.sum()), "known_rejection_rate":ratio((known&~accepted).sum(), known.sum()),
            "correct_known_acceptance_rate":ratio((correct&ak).sum(), known.sum()),
            "unknown_false_acceptance_n":int((un&accepted).sum()), "unknown_false_acceptance_rate":ratio((un&accepted).sum(), un.sum()),
            "unknown_rejection_rate":ratio((un&~accepted).sum(), un.sum()), "all_accepted_n":int(accepted.sum()),
            "all_accepted_accuracy":ratio((correct&ak).sum(), accepted.sum())}


def compare(actual, expected, path="metrics"):
    if isinstance(actual, dict):
        for k,v in actual.items(): compare(v, expected[k], path+"."+k)
    elif isinstance(actual, (list, tuple)):
        assert len(actual)==len(expected), path
        for i,v in enumerate(actual): compare(v, expected[i], path+f"[{i}]")
    elif actual is None or isinstance(actual, str): assert actual==expected, path
    elif isinstance(actual, (int, np.integer)): assert actual==expected, (path,actual,expected)
    else: assert np.isclose(actual, expected, atol=1e-7, rtol=1e-7), (path,actual,expected)


def check_rows(data, model, config, saved):
    classes = data["classes"].tolist(); y = data["y"]
    pred, confidence, accepted, reasons = score(probability(logits(data["x"], model), config["rejection"]["temperature"]),
                                               data["quality"], motion(data["x"]), config["rejection"])
    report = basic(y,pred,accepted)
    report["gate_reasons_nonexclusive"] = {k:int(v.sum()) for k,v in reasons.items()}
    report["open_set_decision_accuracy"] = float(((accepted&(y==pred)&(y>=0)) | (~accepted&(y<0))).mean())
    report["per_known_sign"] = {label:{"n":int((y==k).sum()), "top1_correct_n":int(((y==k)&(pred==y)).sum()),
        "accepted_n":int(((y==k)&accepted).sum()), "accepted_correct_n":int(((y==k)&accepted&(y==pred)).sum()),
        "rejected_n":int(((y==k)&~accepted).sum()),
        "rejection_reasons_nonexclusive":{r:int(((y==k)&v).sum()) for r,v in reasons.items()}} for k,label in enumerate(classes)}
    report["per_signer"] = {person:basic(y[data["signer"]==person],pred[data["signer"]==person],accepted[data["signer"]==person]) for person in np.unique(data["signer"])}
    report["per_unknown_category"] = {}
    for label in np.unique(data["raw_label"][y<0]):
        mask = (y<0)&(data["raw_label"]==label); n=int(mask.sum()); f=int((mask&accepted).sum())
        report["per_unknown_category"][label] = {"n":n,"false_acceptance_n":f,"false_acceptance_rate":f/n,
                                                "accepted_as":dict(Counter(classes[p] for p in pred[mask&accepted]))}
    compare(report,saved["metrics"])
    errors=[]
    assert len(saved["predictions"])==len(y)
    for i,row in enumerate(saved["predictions"]):
        assert row["id"]==data["source_id"][i] and row["person"]==data["signer"][i] and row["label"]==data["raw_label"][i]
        assert row["predicted"]==classes[pred[i]] and row["accepted"]==bool(accepted[i])
        errors.append(abs(row["confidence"]-confidence[i]))
    assert max(errors)<1e-6
    return {"rows":len(y), "max_confidence_error":max(errors), "prediction_or_acceptance_mismatches":0,
            "aggregate_per_sign_person_unknown_metrics_match":True}


def eligible(m):
    if m["known_coverage"]<.6 or m["accepted_known_accuracy"] is None or m["accepted_known_accuracy"]<.9: return None
    return (-m["unknown_false_acceptance_rate"],m["correct_known_acceptance_rate"],m["accepted_known_accuracy"],m["known_coverage"])


def main():
    directory=ROOT/"artifacts/v3-centroid"; reportdir=ROOT/"reports/v3"
    lock=read(directory/"protocol.locked.json"); fit=read(directory/"fit.locked.json")
    for name,h in lock["file_hashes"].items(): assert sha(ROOT/name)==h, name
    assert sha(directory/"protocol.locked.json")==fit["protocol_sha256"]
    assert sha(ROOT/"data/v3/packing-audit.json")==fit["packing_audit_sha256"]
    packing=read(ROOT/"data/v3/packing-audit.json")
    for role in ("train","val"): assert sha(ROOT/f"data/v3/{role}.npz")==packing["outputs"][role]["sha256"]
    train=npz(ROOT/"data/v3/train.npz"); val=npz(ROOT/"data/v3/val.npz"); exposed=npz(ROOT/"data/open-set-v2/final_test.npz")
    assert not (set(train["source_id"])&set(val["source_id"]) or set(train["signer"])&set(val["signer"]))
    assert not (set(exposed["source_id"])&(set(train["source_id"])|set(val["source_id"])))
    for role,data in (("train",train),("val",val)):
        previous=npz(ROOT/f"data/open-set-v2/{role}.npz")
        lookup={v:i for i,v in enumerate(data["source_id"])}
        for i,source in enumerate(previous["source_id"]):
            j=lookup[source]
            for key in ("x","y","signer","raw_label","quality"): assert np.array_equal(data[key][j],previous[key][i]), (role,key,source)
    original=npz(ROOT/"artifacts/open-set-v2/reference-centroid.npz")
    z=normalize(train["x"],original).reshape(len(train["y"]),-1); unknown=train["y"]<0
    expected_pool=z[unknown].mean(0); labels=sorted(set(train["raw_label"][unknown]))
    expected_separate=np.asarray([z[unknown&(train["raw_label"]==label)].mean(0) for label in labels])
    development=read(reportdir/"centroid-development-results.json"); regression=read(reportdir/"centroid-exposed-regression.json")
    assert regression["fit_lock_sha256"]==sha(directory/"fit.locked.json")
    assert regression["data_sha256"]==sha(ROOT/"data/open-set-v2/final_test.npz")
    result={}; selected=None; best=None; movement=motion(val["x"]); training_motion=motion(train["x"])
    targets=np.where(val["y"]<0,3,val["y"])
    for name in lock["protocol"]["methods"]:
        path=directory/name; model=npz(path/"centroids.npz"); config=read(path/"config.json")
        assert sha(path/"centroids.npz")==fit["candidates"][name]["model_sha256"]
        assert sha(path/"config.json")==fit["candidates"][name]["config_sha256"]
        for key in ("mean","std","keep"): assert np.array_equal(model[key],original[key]), (name,key)
        assert np.array_equal(model["centroids"][:3],original["centroids"][:3])
        expected=expected_pool[None,:] if name=="augmented_pool" else expected_separate if name=="separate_negative_prototypes" else original["centroids"][3:]
        assert np.array_equal(expected,model["centroids"][3:]), name
        l=logits(val["x"],model); temperatures=(.5,.75,1,1.5,2,3,5)
        losses=[-np.log(np.clip(probability(l,t)[np.arange(len(l)),targets],1e-12,1)).mean() for t in temperatures]
        temperature=temperatures[int(np.argmin(losses))]; assert temperature==config["rejection"]["temperature"]
        p=probability(l,temperature); method_best=None; chosen=None
        for confidence in (0,.25,.35,.45,.55,.65,.75,.85,.90,.95,.975,.99):
            for quality in (.25,.5,.75):
                for cutoff in (0,float(np.quantile(training_motion[train["y"]>=0],.01)),float(np.quantile(training_motion[train["y"]>=0],.05))):
                    gate={"confidenceThreshold":confidence,"minQuality":quality,"minMotionBodyRms":cutoff}
                    pred,_,accepted,_=score(p,val["quality"],movement,gate); s=eligible(basic(val["y"],pred,accepted))
                    if s is not None and (method_best is None or s>method_best): method_best,chosen=s,gate
        if chosen is not None:
            for key,value in chosen.items(): assert config["rejection"][key]==value,(name,key)
        if method_best is not None and (best is None or method_best>best): best,selected=method_best,name
        result[name]={"fixed_normalization_and_known_centers_bitwise_equal":True,"negative_centers_train_only_bitwise_reproduced":True,
                      "validation_temperature_and_grid_reproduced":True,
                      "training":check_rows(train,model,config,development["methods"][name]["training_in_sample_only"]),
                      "validation":check_rows(val,model,config,development["methods"][name]["validation"]),
                      "exposed_regression":check_rows(exposed,model,config,regression["methods"][name])}
    assert selected==fit["selected_development_candidate"]=="augmented_pool"
    reference_config=read(ROOT/"artifacts/open-set-v2/reference-centroid-config.json")
    reference={"validation":check_rows(val,original,reference_config,development["frozen_reference_validation"]),
               "regression":check_rows(exposed,original,reference_config,regression["methods"]["frozen_reference"])}
    report={"status":"pass","independence":"NumPy implementation without importing fitting/scoring/metric modules; saved parameters only; no refitting or threshold changes",
            "protocol_hashes_and_split_provenance_pass":True,"original_development_rows_reused_unchanged":True,
            "train_validation_source_signer_disjoint":True,"exposed_regression_clips_absent_from_development":True,
            "selected_candidate_reproduced":selected,"methods":result,"frozen_reference":reference,
            "rows_verified":sum(v[k]["rows"] for v in result.values() for k in ("training","validation","exposed_regression"))+sum(v["rows"] for v in reference.values()),
            "critical_limitation":"Selected augmented_pool accepts zero of eleven validation ΓΕΙΑ clips. Aggregate eligibility does not require per-sign coverage. The exposed129 regression is adaptive evidence, not fresh confirmation; no deployment recommendation.",
            "checked_files":{"protocol":sha(directory/"protocol.locked.json"),"fit":sha(directory/"fit.locked.json"),
                             "development":sha(reportdir/"centroid-development-results.json"),"regression":sha(reportdir/"centroid-exposed-regression.json")}}
    output=reportdir/"centroid-independent-audit.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"pass","rows_verified":report["rows_verified"],"report":str(output),"sha256":sha(output)},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
