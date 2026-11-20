/** Resume-safe all-development extraction. No fitting/classifier; both streams always retained. */
import {chromium} from 'playwright';
import {readFile,writeFile,rename,mkdir} from 'node:fs/promises';
import {gzipSync,gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),base=process.argv[2]??'http://127.0.0.1:5178',parallel=Number(process.argv[3]??3);
if(!Number.isSafeInteger(parallel)||parallel<1||parallel>4)throw new Error('Parallelism must be1..4');
const out=path.join(root,'data/v7-recovery'),bytes=await readFile(path.join(out,'extraction-protocol.json')),p=JSON.parse(bytes);
const sha=b=>createHash('sha256').update(b).digest('hex'),protocolSha256=sha(bytes),url=v=>'/@fs/'+path.join(root,v).replaceAll('\\','/');
const cacheKey=s=>sha(JSON.stringify({schema:1,protocolSha256,pipelineSha256:p.pipelineSha256,source:s.sourceContentSha256,id:s.id}));
const recordPath=s=>path.join(out,'records',s.id+'.json.gz');await mkdir(path.join(out,'records'),{recursive:true});
async function verifySources(){for(const [name,hash]of Object.entries(p.sourceFilesSha256)){if(sha(await readFile(path.join(root,name)))!==hash)throw new Error('Frozen pipeline asset changed: '+name);}}
await verifySources();
function valid(r,s){return r.schemaVersion===1&&r.cacheKey===cacheKey(s)&&r.id===s.id&&r.source_id===s.source_id&&r.protocolSha256===protocolSha256&&r.preservationPassed&&['originalFeatures','fusedFeatures'].every(k=>r[k]?.x?.length===48&&r[k].x.every(row=>row.length===224&&row.every(Number.isFinite))&&Number.isFinite(r[k].quality))&&r.frameCount===s.paths.length;}
async function readCached(s){try{const r=JSON.parse(gunzipSync(await readFile(recordPath(s))).toString('utf8'));return valid(r,s)?r:null;}catch{return null;}}
const report={schemaVersion:1,createdAt:new Date().toISOString(),protocolSha256,pipelineSha256:p.pipelineSha256,parallelism:parallel,completed:false,total:424,reused:0,extracted:0,failed:[],splitCounts:{val:0,train:0},rows:[]};
const browser=await chromium.launch({channel:'chrome',headless:true});report.browserVersion=browser.version();const pages=[];
let writeQueue=Promise.resolve();
function progress(){const snapshot=JSON.stringify(report,null,2);writeQueue=writeQueue.then(async()=>{const temp=path.join(out,'extraction-progress.json.tmp');await writeFile(temp,snapshot);await rename(temp,path.join(out,'extraction-progress.json'));});return writeQueue;}
function compact(r,reused){report[reused?'reused':'extracted']++;report.splitCounts[r.split]++;report.rows.push({id:r.id,split:r.split,reused,frameCount:r.frameCount,recoveredFrames:r.recoveredFrames,originalUsable:r.originalGuard.usable,fusedUsable:r.fusedGuard.usable,totalMs:r.totalMs});}
try{
 for(let i=0;i<parallel;i++){const page=await browser.newPage();await page.goto(base+'/v6_hand_recovery.html');await page.waitForFunction(()=>typeof window.v6HandRecovery==='function');await page.evaluate(()=>import('/src/features.ts'));pages.push(page);}
 // Complete the validation split before starting any training job.
 for(const split of ['val','train']){
  const queue=[];for(const s of p.samples.filter(s=>s.split===split)){const cached=await readCached(s);if(cached)compact(cached,true);else queue.push(s);}
  await progress();let cursor=0;
  await Promise.all(pages.map(async(page,workerIndex)=>{
   while(cursor<queue.length){const s=queue[cursor++];let lastError;
    for(let attempt=0;attempt<2;attempt++){
     try{
      await verifySources();
      const row=await page.evaluate(async data=>{
       const {preprocess}=await import('/src/features.ts');const r=await window.v6HandRecovery(data);
       const originalFeatures=preprocess(r.frames.map(f=>f.original)),fusedFeatures=preprocess(r.frames.map(f=>f.fused));
       return {originalGuard:r.originalGuard,fusedGuard:r.fusedGuard,originalFeatures,fusedFeatures,guardPolicy:r.guardPolicy,
        frameCount:r.frames.length,recoveredFrames:r.frames.filter(f=>f.recovered).length,fallbackCalls:r.frames.filter(f=>f.handMs>0).length,
        frameTiming:r.frames.map(f=>({holisticMs:f.holisticMs,handMs:f.handMs,frameMs:f.frameMs})),initMs:r.initMs,totalMs:r.totalMs,
        originalFramePresence:r.frames.map(f=>({left:f.original.left.length,right:f.original.right.length,pose:f.original.pose.length,face:f.original.face.length})),
        fusedFramePresence:r.frames.map(f=>({left:f.fused.left.length,right:f.fused.right.length,pose:f.fused.pose.length,face:f.fused.face.length})),
        recoveryAudit:r.frames.map(f=>({recovered:f.recovered,skipReason:f.skipReason,roi:f.roi,candidates:f.candidates.map(c=>({category:c.category,score:c.score,targetDistance:c.targetDistance,oppositeDistance:c.oppositeDistance,reasons:c.reasons,plausible:c.plausible}))})),
        preservationPassed:r.frames.every(f=>JSON.stringify(f.original.left)===JSON.stringify(f.fused.left)&&JSON.stringify(f.original.pose)===JSON.stringify(f.fused.pose)&&JSON.stringify(f.original.face)===JSON.stringify(f.fused.face)&&(!f.original.right.length||JSON.stringify(f.original.right)===JSON.stringify(f.fused.right)))};
      },{paths:s.paths.map(url),control:'none',rule:p.rule,handModelUrl:url(p.model.path)});
      const record={schemaVersion:1,cacheKey:cacheKey(s),protocolSha256,pipelineSha256:p.pipelineSha256,sourceSha256:s.sourceSha256,sourceContentSha256:s.sourceContentSha256,modelSha256:p.model.sha256,
       id:s.id,source_id:s.source_id,split:s.split,raw_label:s.raw_label,y:s.y,signer:s.signer,browserVersion:report.browserVersion,...row};
      if(!valid(record,s))throw new Error('Record validation failed: '+s.id);
      const dest=recordPath(s),temp=dest+`.tmp-${process.pid}-${workerIndex}`;await writeFile(temp,gzipSync(JSON.stringify(record),{level:6}));await rename(temp,dest);
      compact(record,false);await progress();console.log(JSON.stringify({split,done:report.splitCounts[split],id:s.id,frames:row.frameCount,recovered:row.recoveredFrames,guardOriginal:row.originalGuard.usable,guardFused:row.fusedGuard.usable}));lastError=null;break;
     }catch(e){lastError=e;await page.goto(base+'/v6_hand_recovery.html');await page.waitForFunction(()=>typeof window.v6HandRecovery==='function');}
    }
    if(lastError){report.failed.push({id:s.id,error:String(lastError)});await progress();throw lastError;}
   }
  }));
 }
 await verifySources();report.completed=report.rows.length===424&&report.failed.length===0;report.completedAt=new Date().toISOString();await progress();await writeQueue;
}finally{await writeQueue;await browser.close();}
