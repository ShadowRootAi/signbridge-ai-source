"""Expose concise, attributed evidence without local paths or inflated claims."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'web/public/reports'


def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value,str):
        value=value.replace(str(ROOT),'[project]').replace(ROOT.as_posix(),'[project]')
        return re.sub(r'[A-Za-z]:[\\/][^\n]*','[local-path-redacted]',value)
    if isinstance(value,list): return [sanitize(v) for v in value]
    if isinstance(value,dict): return {k:sanitize(v) for k,v in value.items()}
    return value


def main():
    config=read('web/public/models/model_config.json')
    final=read('artifacts/open-set-v2/final-evaluation.json')
    metric=final['candidates'][config['candidate']]
    evidence={
        'schemaVersion':1, 'release':'SignBridge AI V4', 'modelVersion':config['modelVersion'],
        'modelSha256':sha(ROOT/'web/public/models/model.onnx'), 'labels':config['labels'],
        'historicalV2':{
            'knownN':metric['known_n'], 'acceptedKnownCorrect':sum(v['accepted_correct_n'] for v in metric['per_known_sign'].values()),
            'unknownN':metric['unknown_n'], 'falseAcceptedUnknown':metric['unknown_false_acceptance_n'],
            'closedSetMacroF1':metric['closed_set_macro_f1'], 'acceptedPredictionPrecision':metric['all_accepted_accuracy'],
            'perKnownSign':[{'label':k,'n':v['n'],'acceptedCorrect':v['accepted_correct_n']} for k,v in metric['per_known_sign'].items()],
            'scope':'Historical V2 evaluation before the V4 input-completeness guard. These129 clips are now exposed.',
            'people':[3,7], 'fullyNewHumanEvaluation':False,
        },
        'currentV4':{
            'status':'Research prototype; input checks are engineering safeguards, not validated open-set reliability.',
            'trackingPolicyVersion':'signbridge-tracking-v4',
            'independentHumanEvaluation':False, 'humanWebcamTested':False,
            'modelSelection':'Frozen validation-selected B retained; no replacement selected from the exposed129 examples.',
        },
        'links':{'technicalReport':'/reports/technical-report.json','notice':'/NOTICE.txt'},
        'sources':[{'title':'GSL Dataset / CERTH','url':'https://vcl.iti.gr/dataset/gsl/'},
                   {'title':'GSL dataset v2 / Zenodo','url':'https://zenodo.org/records/4756317'}],
        'supportedScope':'Three isolated supported signs; no continuous GSL translation or word-by-word sentence generation.',
        'cameraPrivacy':'Frames are processed in browser memory, not uploaded or recorded by this app. Speech is a separate opt-in device/browser service.',
    }
    policy=ROOT/'reports/v4/tracking-policy.locked.json'
    if policy.exists():
        evidence['currentV4']['trackingPolicy']=sanitize(json.loads(policy.read_text(encoding='utf-8-sig')))
        evidence['currentV4']['trackingPolicyVersion']=evidence['currentV4']['trackingPolicy']['policy']['version']
    PUBLIC.mkdir(parents=True,exist_ok=True)
    (PUBLIC/'app-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    old=read('reports/open-set-v2/technical-report.json')
    old=sanitize(old)
    old['publicReportContext']={'historical':True,'v4TrackingGuardApplied':False,
                                'note':'Historical V2 model evidence; not a current V4 application or independent-new-person evaluation.'}
    (PUBLIC/'technical-report.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'app_evidence':'web/public/reports/app-evidence.json','historical_report_sanitized':True}))


if __name__=='__main__': main()
