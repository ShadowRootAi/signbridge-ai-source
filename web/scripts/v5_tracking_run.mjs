import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const base=process.argv[2]??'http://127.0.0.1:5178';
const bytes=await readFile(path.join(root,'reports/v5/tracking-input.json'));
const protocol=JSON.parse(bytes);const rows=[];
const browser=await chromium.launch({channel:'chrome',headless:true});
const context=await browser.newContext();const page=await context.newPage();
const report={createdAt:new Date().toISOString(),browserVersion:browser.version(),protocolSha256:createHash('sha256').update(bytes).digest('hex'),noClassifier:true,noHumanWebcam:true,rows};
function summarise(frames){const n=frames.length;const right=frames.filter(f=>f.right.length===21),both=frames.filter(f=>f.right.length===21&&f.left.length===21&&f.pose.length===33);return {frames:n,leftPresent:frames.filter(f=>f.left.length===21).length,rightPresent:right.length,posePresent:frames.filter(f=>f.pose.length===33).length,requiredPresent:both.length,requiredCoverage:both.length/n,passes90PercentPresenceOnly:both.length/n>=.9,rightWristInOccludedArea:right.filter(f=>{const b=f.geometry.occlusion,p=f.right[0];return b&&p.x*f.width>=b[0]&&p.x*f.width<=b[2]&&p.y*f.height>=b[1]&&p.y*f.height<=b[3];}).length,frameMsMean:frames.reduce((s,f)=>s+f.elapsedMs,0)/n};}
try{
 await page.goto(base+'/v5_tracking.html');await page.waitForFunction(()=>typeof window.v5TrackingRun==='function');
 for(const sample of protocol.samples){for(const control of ['none',...protocol.controls]){for(const variant of protocol.variants){
   const started=performance.now();const result=await page.evaluate(data=>window.v5TrackingRun(data),{paths:sample.paths.map(p=>'/@fs/'+path.join(root,p).replaceAll('\\','/')),reference:sample.reference,control,variant});
   const row={id:sample.id,split:sample.split,signer:sample.signer,label:sample.label,control,variant,summary:summarise(result.frames),roiApi:result.roiApi,elapsedMs:performance.now()-started,raw:result.frames};rows.push(row);
   console.log(JSON.stringify({id:row.id,control,variant,...row.summary}));
   await writeFile(path.join(root,'reports/v5/tracking-results.json'),JSON.stringify(report,null,2));
 }}}
 report.completed=true;await writeFile(path.join(root,'reports/v5/tracking-results.json'),JSON.stringify(report,null,2));
}finally{await browser.close();}
