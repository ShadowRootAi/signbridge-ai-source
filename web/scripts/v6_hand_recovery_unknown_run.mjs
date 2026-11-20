import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),base=process.argv[2]??'http://127.0.0.1:5178';
const bytes=await readFile(path.join(root,'reports/v6/hand-recovery-unknown-protocol.json')),p=JSON.parse(bytes);
const report={createdAt:new Date().toISOString(),protocolSha256:createHash('sha256').update(bytes).digest('hex'),completed:false,noClassifier:true,noHumanWebcam:true,rows:[]};
const browser=await chromium.launch({channel:'chrome',headless:true});report.browserVersion=browser.version();const page=await browser.newPage();
const url=v=>'/@fs/'+path.join(root,v).replaceAll('\\','/');
const save=()=>writeFile(path.join(root,'reports/v6/hand-recovery-unknown-results.json'),JSON.stringify(report,null,2));
try{
 await page.goto(base+'/v6_hand_recovery.html');await page.waitForFunction(()=>typeof window.v6HandRecovery==='function');
 for(const sample of p.samples){const row=await page.evaluate(async data=>{
  const {preprocess}=await import('/src/features.ts');
  const r=await window.v6HandRecovery(data);
  const originalFeatures=r.originalGuard.usable?preprocess(r.frames.map(f=>f.original)):null;
  const fusedFeatures=r.fusedGuard.usable?preprocess(r.frames.map(f=>f.fused)):null;
  const fallback=r.frames.filter(f=>f.handMs>0);
  return {originalGuard:r.originalGuard,fusedGuard:r.fusedGuard,originalFeatures,fusedFeatures,guardPolicy:r.guardPolicy,
   frameCount:r.frames.length,recoveredFrames:r.frames.filter(f=>f.recovered).length,fallbackCalls:fallback.length,
   frameTiming:r.frames.map(f=>({holisticMs:f.holisticMs,handMs:f.handMs,frameMs:f.frameMs})),initMs:r.initMs,totalMs:r.totalMs,
   preservationPassed:r.frames.every(f=>JSON.stringify(f.original.left)===JSON.stringify(f.fused.left)&&JSON.stringify(f.original.pose)===JSON.stringify(f.fused.pose)&&JSON.stringify(f.original.face)===JSON.stringify(f.fused.face)&&(!f.original.right.length||JSON.stringify(f.original.right)===JSON.stringify(f.fused.right)))};
 },{paths:sample.paths.map(url),control:'none',rule:p.rule,handModelUrl:url(p.model.path)});
 report.rows.push({id:sample.id,source_id:sample.source_id,label:sample.label,...row});await save();console.log(JSON.stringify({id:sample.id,n:row.frameCount,recovered:row.recoveredFrames,guardBefore:row.originalGuard.usable,guardAfter:row.fusedGuard.usable}));
 }
 report.completed=true;await save();
}finally{await browser.close();}
