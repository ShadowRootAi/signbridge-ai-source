"""Copy only three explicitly reviewed, licensed complete-utterance video derivatives."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'web/public/library'
REVIEW=ROOT/'reports/v4/text-to-sign-rights-review.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=json.loads(REVIEW.read_text(encoding='utf-8-sig'))
    assert review['source']['license']['id']=='cc-by-4.0'
    assert review['source']['access_right']=='open'
    assert review['status']=='copyright_redistribution_supported_by_official_CC_BY_4_0_record_with_attribution'
    exact={'γεια':('geia','ΓΕΙΑ'),'ευχαριστώ':('efcharisto','ΕΥΧΑΡΙΣΤΩ'),'εντάξει':('entaxei','ΕΝΤΑΞΕΙ')}
    assert {clip['utterance_text'].strip() for clip in review['clips']}==set(exact)
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[]
    creators='; '.join(review['source']['creators'])
    attribution='The Greek Sign Language (GSL) Dataset v2 — '+creators+'.'
    changes='Original released JPEG sequence encoded as H.264/yuv420p MP4, CRF18, 30fps; no trimming, resizing, mirroring, added frames or added audio.'
    for clip in review['clips']:
        phrase=clip['utterance_text'].strip(); identifier,label=exact[phrase]
        assert clip['annotation_raw_glosses']==label and clip['verified_decodable']
        source=ROOT/clip['path'];assert sha(source)==clip['mp4_sha256']
        destination=OUT/(identifier+'.mp4');shutil.copyfile(source,destination)
        assert sha(destination)==clip['mp4_sha256']
        entries.append({'id':identifier,'phrase':phrase,'aliases':[],'annotationLabel':label,
            'videoUrl':'/library/'+destination.name,'durationSeconds':clip['duration_seconds'],
            'sourceUrl':review['source']['record_url'],'sourceArchiveUrl':clip['source_url'],
            'sourceUtterance':clip['source_path'],'sha256':clip['mp4_sha256'],
            'attribution':attribution,'license':{'id':'CC-BY-4.0','name':'Creative Commons Attribution 4.0 International','url':review['rights']['license_url']},
            'modifications':changes,'rights':{'redistributionConfirmed':True,'basis':'Official open Zenodo record4756317, CC BY4.0; attribution and change notices retained.',
              'scope':'Dataset copyright license; no endorsement or independently inspected participant releases claimed.'},
            'verification':{'status':'dataset-annotated','basis':'Απόδοση σύμφωνα με τις επισημάνσεις του GSL. Ελέγχθηκε η ακριβής αντιστοίχιση του αποσπάσματος με το αρχικό ελληνικό κείμενο και το gloss. Δεν έγινε νέα αξιολόγηση από ειδικό ΕΝΓ.',
              'newExpertReview':False},
            'contentNote':'Πολύ σύντομο, ολόκληρο επισημασμένο απόσπασμα του dataset. Αναπαράγεται με την αρχική σειρά και ταχύτητα.'})
    catalog={'schemaVersion':1,'scope':'Exact three-phrase lookup, not automatic Greek-to-GSL sentence translation.',
             'reviewSha256':sha(REVIEW),'entries':entries,'generatedPredictions':False,'wordConcatenation':False,
             'licenseNotice':'Supplied without warranties. No endorsement by dataset creators or depicted signers.'}
    (OUT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    notice='SIGNBRIDGE V4 — THREE GSL VIDEO DERIVATIVES\n\n'+attribution+'\nSource: '+review['source']['record_url']+'\nDOI: https://doi.org/10.5281/zenodo.4756317\nLicense: CC BY 4.0, '+review['rights']['license_url']+'\nFull terms: '+review['rights']['legalcode_url']+'\n\nChanges: '+changes+'\n\nOriginal dataset annotations are preserved. No new GSL expert review, endorsement\nor general translation capability is claimed. Supplied without warranties.\n\n'
    notice+='\n'.join(f"{e['videoUrl']}: {e['phrase']} / {e['annotationLabel']}\nSource utterance: {e['sourceUtterance']}\nSHA256: {e['sha256']}" for e in entries)+'\n'
    (OUT/'NOTICE.txt').write_text(notice,encoding='utf-8')
    root_notice=ROOT/'web/public/NOTICE.txt'
    original=root_notice.read_text(encoding='utf-8').split('\nV4 PUBLIC VIDEO LIBRARY\n',1)[0]
    original=original.replace('SIGNBRIDGE AI — LOCAL SCIENTIFIC PROOF OF CONCEPT','SIGNBRIDGE AI — LIMITED RESEARCH WEB APPLICATION')
    root_notice.write_text(original.rstrip()+'\n\nV4 PUBLIC VIDEO LIBRARY\n\nThree selected GSL utterance video derivatives are included under /library/.\nTheir individual annotations, source identities, licenses, modifications and\nhashes are in /library/catalog.json and /library/NOTICE.txt. Redistribution\nis based on the official open CC BY4.0 dataset record; no endorsement or new\nGSL expert review is claimed. No full dataset archive or webcam recordings\nare included. Full runtime license texts are provided under /licenses/.\n',encoding='utf-8')
    print(json.dumps({'entries':len(entries),'bytes':sum((OUT/(e['id']+'.mp4')).stat().st_size for e in entries),'catalog_sha256':sha(OUT/'catalog.json')},ensure_ascii=False))


if __name__=='__main__':main()
