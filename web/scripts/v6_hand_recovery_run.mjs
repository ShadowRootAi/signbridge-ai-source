import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),base=process.argv[2]??'http://127.0.0.1:5178';
const raw=await readFile(path.join(root,'reports/v6/hand-recovery-protocol.json')),protocol=JSON.parse(raw);
const report={createdAt:new Date().toISOString(),protocolSha256:createHash('sha256').update(raw).digest('hex'),completed:false,noClassifier:true,noHumanWebcam:true,preflight:[],rows:[]};
const browser=await chromium.launch({channel:'chrome',headless:true});report.browserVersion=browser.version();
const page=await browser.newPage();
const url=p=>'/@fs/'+path.join(root,p).replaceAll('\\','/');
const save=()=>writeFile(path.join(root,'reports/v6/hand-recovery-results.json'),JSON.stringify(report,null,2));
const invoke=(sample,control,reference,probe=false)=>page.evaluate(data=>window.v6HandRecovery(data),{paths:(probe?sample.paths.slice(0,1):sample.paths).map(url),control,reference,probe,rule:protocol.rule,handModelUrl:url(protocol.model.path)});
try{
 await page.goto(base+'/v6_hand_recovery.html');await page.waitForFunction(()=>typeof window.v6HandRecovery==='function');
 for(const sample of protocol.samples.filter(s=>s.split==='train')){const r=await invoke(sample,'none',null,true);report.preflight.push({id:sample.id,...r});}
 const geometry=report.preflight.flatMap(r=>r.frames.flatMap(f=>f.candidates.filter(c=>c.reasons.every(reason=>reason==='handedness'))));
 report.preflightPassed=geometry.length>=2&&geometry.every(c=>c.category===protocol.rule.requiredHandedness);
 await save();if(!report.preflightPassed)throw new Error('Handedness preflight failed; stopped before validation.');
 for(const sample of protocol.samples){let reference;
  for(const control of protocol.conditions){const r=await invoke(sample,control,reference);if(control==='none')reference=r.frames.map(f=>f.original);
   const row={id:sample.id,split:sample.split,control,...r};report.rows.push(row);
   console.log(JSON.stringify({id:row.id,control,n:r.frames.length,recovered:r.frames.filter(f=>f.recovered).length,before:r.originalGuard.requiredDataCompleteness,after:r.fusedGuard.requiredDataCompleteness,guardBefore:r.originalGuard.usable,guardAfter:r.fusedGuard.usable}));await save();
  }
 }
 report.completed=true;await save();
}finally{await browser.close();}
