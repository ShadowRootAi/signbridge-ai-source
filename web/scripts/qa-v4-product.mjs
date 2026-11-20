/** Product QA. Real MP4/model tests are separate from explicitly emulated API flow tests. */
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),root=path.resolve(web,'..');
const base=(process.argv[2]??'http://127.0.0.1:5176').replace(/\/$/,'');
const out=path.join(root,'reports/v4/product-qa');await mkdir(out,{recursive:true});
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const catalog=JSON.parse(await readFile(path.join(web,'public/library/catalog.json'),'utf8'));
const mapping=JSON.parse(await readFile(path.join(root,'data/open-set-v2/clips/mapping.json'),'utf8'));
const report={createdAt:new Date().toISOString(),base,passed:false,humanWebcamTested:false,actualSpeechAudioVerified:false,
  realMp4Runs:[],checks:[],emulatedApiChecks:[],screenshots:[],errors:[],requests:[]};
report.publicationAuditSha256=hash(await readFile(path.join(root,'publication-v4/publication-audit.json')));
const requests=new Set();let failure;
const browser=await chromium.launch({channel:'chrome',headless:true});
report.browserVersion=browser.version();
const context=await browser.newContext({viewport:{width:1440,height:1080}});
context.on('request',r=>{if(/^https?:/.test(r.url()))requests.add(r.url());});
await context.addInitScript(()=>{
  window.__v4qa={results:[],speechCalls:0};
  const OriginalWorker=window.Worker;
  window.Worker=class extends OriginalWorker{
    starts=new Map();
    postMessage(...args){if(args[0]?.id!==undefined)this.starts.set(args[0].id,performance.now());return super.postMessage(...args);}
    constructor(...args){super(...args);this.addEventListener('message',event=>{
      const value=event.data?.value;
      const started=this.starts.get(event.data?.id);this.starts.delete(event.data?.id);
      if(value&&typeof value.accepted==='boolean'&&value.tracking)window.__v4qa.results.push({accepted:value.accepted,label:value.label,classificationScore:value.classificationScore,classifierExecuted:value.classifierExecuted,tracking:value.tracking,modelVersion:value.modelVersion,
        elapsedMs:Number.isFinite(value.elapsedMs)?value.elapsedMs:null,
        workerRequestElapsedMs:started===undefined?null:performance.now()-started,
        timingScope:'Worker inferClip request/reply only: raw guard, preprocessing and classifier if executed. Excludes preceding video decode/landmark extraction. Full UI execution is reported separately as wallMs.'});
    });}
  };
  if(window.speechSynthesis){const original=window.speechSynthesis.speak.bind(window.speechSynthesis);window.speechSynthesis.speak=(value)=>{window.__v4qa.speechCalls++;return original(value);};}
});
const page=await context.newPage();page.setDefaultTimeout(120000);
page.on('pageerror',error=>report.errors.push(error.message));
async function shot(name,target=page){
  await target.evaluate(()=>window.scrollTo({top:0,left:0,behavior:'instant'}));
  await target.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await target.screenshot({path:path.join(out,name),fullPage:true,animations:'disabled'});report.screenshots.push(name);
}
async function route(name,target=page){await target.goto(`${base}/#/${name==='home'?'':name}`);await target.locator('main h1').waitFor();}
async function resultAfterRun(source){
  console.log('Real MP4: '+source);
  const previous=await page.evaluate(()=>window.__v4qa.results.length);
  const started=Date.now();await page.getByRole('button',{name:'Ανάλυση βίντεο',exact:true}).click();
  await page.waitForFunction(n=>window.__v4qa.results.length>n,previous);
  await page.locator('.result-title').waitFor();
  const result=await page.evaluate(()=>window.__v4qa.results.at(-1));
  const visible=await page.locator('.result-panel').innerText();
  report.realMp4Runs.push({source,result,uiText:visible,wallMs:Date.now()-started});return result;
}
try{
  const response=await page.goto(base);assert.ok(response.ok());
  report.servedIndexSha256=hash(await response.body());
  const publicationAudit=JSON.parse(await readFile(path.join(root,'publication-v4/publication-audit.json'),'utf8'));
  assert.equal(report.servedIndexSha256,publicationAudit.files.find(item=>item.path==='index.html')?.sha256,'Served HTML differs from reviewed artifact.');
  const headers=response.headers();
  assert.match(headers['content-security-policy'],/wasm-unsafe-eval/);assert.ok(!headers['content-security-policy'].includes("'unsafe-eval'"));
  assert.equal(headers['cross-origin-opener-policy'],'same-origin');assert.equal(headers['cross-origin-embedder-policy'],'require-corp');
  assert.equal(headers['permissions-policy'],'camera=(self), microphone=(), geolocation=()');
  assert.ok(!(await response.text()).includes('/@vite/client'));
  assert.match((await context.request.get(base+'/vendor/onnxruntime/ort-wasm-simd-threaded.mjs')).headers()['content-type'],/javascript/);
  const servedModel=await context.request.get(base+'/models/model.onnx');
  assert.equal(hash(await servedModel.body()),hash(await readFile(path.join(web,'public/models/model.onnx'))));
  const evidence=await (await context.request.get(base+'/reports/app-evidence.json')).json();
  assert.equal(evidence.currentV4.independentHumanEvaluation,false);assert.equal(evidence.historicalV2.falseAcceptedUnknown,24);
  report.modelVersion=evidence.modelVersion;report.trackingPolicyVersion=evidence.currentV4.trackingPolicyVersion;
  for(const forbidden of ['/extract.html','/webcam-check.html','/src/App.tsx','/data/manifest.json','/__signbridge/webcam-capture/status','/@fs/C:/'])assert.equal((await context.request.get(base+forbidden)).status(),404);
  assert.equal((await context.request.post(base+'/__signbridge/webcam-capture',{data:{}})).status(),405);
  report.checks.push('Production HTML/model identity/headers pass; scientific and capture endpoints excluded.');
  for(const name of ['home','sign','library','try','about']){
    await route(name);assert.equal(await page.locator('main h1').count(),1);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`Desktop overflow: ${name}`);
  }
  await route('home');await shot('desktop-home.png');
  await route('about');assert.match(await page.locator('main').innerText(),/24\s*\/\s*50/);
  await shot('desktop-about.png');report.checks.push('All five desktop routes render; About discloses historical unknown failures.');
  await route('library');await page.getByRole('button',{name:'γεια',exact:true}).waitFor();
  for(const entry of catalog.entries){
    await page.getByRole('button',{name:entry.phrase,exact:true}).click();
    const video=page.locator('.library-video video');
    await video.evaluate(async element=>{await element.play();});
    await page.waitForFunction(()=>{const v=document.querySelector('.library-video video');return v&&v.videoWidth===648&&v.currentTime>0;});
    const actual=await video.evaluate(v=>({duration:v.duration,currentSrc:v.currentSrc,paused:v.paused}));
    assert.ok(Math.abs(actual.duration-entry.durationSeconds)<.04);
    const media=await context.request.get(base+entry.videoUrl);assert.equal(hash(await media.body()),entry.sha256);
    const range=await context.request.get(base+entry.videoUrl,{headers:{Range:'bytes=0-99'}});assert.equal(range.status(),206);assert.equal((await range.body()).length,100);
  }
  await page.locator('#phrase').fill('ΕΥΧΑΡΙΣΤΩ');await page.getByRole('button',{name:'Αναζήτηση φράσης'}).click();
  await page.getByRole('heading',{name:'ευχαριστώ',exact:true}).waitFor();
  await page.locator('.library-video video').evaluate(async video=>{await video.play();});
  await page.waitForFunction(()=>{const video=document.querySelector('.library-video video');return video&&video.readyState>=2&&video.currentTime>0;});
  await page.locator('.library-video video').evaluate(video=>video.pause());await shot('desktop-library.png');
  await page.locator('#phrase').fill('γεια ευχαριστώ');await page.getByRole('button',{name:'Αναζήτηση φράσης'}).click();
  await page.getByRole('heading',{name:'Αυτή η φράση δεν υποστηρίζεται ακόμη.',exact:true}).waitFor();
  assert.equal(await page.locator('.library-video video').count(),0);report.checks.push('Three real licensed videos decode/seek; normalized exact lookup works; concatenated unsupported phrase shows no invented video.');
  await route('try');await page.getByRole('button',{name:'γεια',exact:true}).waitFor();
  console.log('Library checks passed; starting real model runs.');
  for(const entry of catalog.entries){
    await page.getByRole('button',{name:entry.phrase,exact:true}).click();
    await page.getByRole('button',{name:'Ανάλυση βίντεο',exact:true}).waitFor({state:'visible'});
    await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
    // Selecting a source must clear the old result, not produce a canned prediction.
    assert.equal(await page.locator('.result-title').count(),0);
    const predicted=await resultAfterRun(`catalog:${entry.id}`);
    if(predicted.classifierExecuted)assert.equal(predicted.modelVersion,evidence.modelVersion);
    else{
      assert.equal(predicted.modelVersion,'not-executed-incomplete-input');
      assert.equal(predicted.accepted,false);assert.equal(predicted.label,null);assert.equal(predicted.classificationScore,null);
      assert.equal(predicted.tracking.usable,false);
    }
  }
  // Same-example reselect must leave Analyze usable, even if no metadata event fires.
  await page.getByRole('button',{name:catalog.entries.at(-1).phrase,exact:true}).click();
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
  const known=catalog.entries.find(e=>e.id==='geia');
  await page.locator('#video-file').setInputFiles({name:'different-label-entaxei.mp4',mimeType:'video/mp4',buffer:await readFile(path.join(web,'public',known.videoUrl))});
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
  const renamed=await resultAfterRun('same-geia-bytes-renamed-entaxei');assert.equal(renamed.label,'ΓΕΙΑ');assert.equal(renamed.classifierExecuted,true);
  assert.equal(await page.evaluate(()=>window.__v4qa.speechCalls),0);await shot('desktop-real-result.png');
  const voices=await page.evaluate(()=>({supported:'speechSynthesis'in window,greek:window.speechSynthesis?.getVoices().filter(v=>/^el(?:[-_]|$)/i.test(v.lang)).map(v=>({lang:v.lang,local:v.localService}))??[]}));
  report.actualVoiceAvailability=voices;
  const incomplete=mapping.clips['kep1_signer7_rep3_sentences__sentences0000'];
  await page.locator('#video-file').setInputFiles(path.join(root,incomplete.path));
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
  const rejected=await resultAfterRun('real-mp4-incomplete-hands');assert.equal(rejected.accepted,false);assert.equal(rejected.classifierExecuted,false);assert.equal(rejected.classificationScore,null);
  assert.ok(rejected.tracking.requiredDataCompleteness<.9);await shot('desktop-tracking-retry.png');
  const unsupported=mapping.clips['health4_signer3_rep1_sentences__sentences0021'];
  await page.locator('#video-file').setInputFiles(path.join(root,unsupported.path));
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
  const unknown=await resultAfterRun('known-regression-flaw-unsupported-katalavaino');
  report.remainingSemanticFailure={trueLabel:'ΚΑΤΑΛΑΒΑΙΝΩ',accepted:unknown.accepted,predicted:unknown.label};
  // This records the real remaining failure rather than forcing a favorable UI result.
  assert.equal(await page.evaluate(()=>window.__v4qa.speechCalls),0);
  await page.locator('#video-file').setInputFiles({name:'broken.mp4',mimeType:'video/mp4',buffer:Buffer.from('not a video')});
  await page.getByRole('alert').waitFor();assert.equal(await page.getByRole('button',{name:'Ανάλυση βίντεο',exact:true}).isDisabled(),true);
  report.checks.push('Actual MP4 inference, renamed-file invariance, incomplete-input retry, malformed-file error and no automatic speech pass.');
  const mobile=await context.newPage();await mobile.setViewportSize({width:390,height:844});
  for(const name of ['home','sign','library','try','about']){
    await route(name,mobile);assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`Mobile overflow: ${name}`);
  }
  await route('home',mobile);await shot('mobile-home.png',mobile);await route('sign',mobile);await shot('mobile-camera.png',mobile);
  await mobile.close();report.checks.push('Five mobile routes have no horizontal overflow at390px.');

  // Explicit API-flow emulation: not real camera hardware, sign-language performance or actual speech audio.
  const emulated=await browser.newContext({viewport:{width:1280,height:1000}});
  console.log('Real MP4 and responsive checks passed; starting emulated API-flow checks.');
  await emulated.addInitScript(()=>{
    window.__flow={calls:0,streams:[],speech:[],constraints:[],deny:true};
    const synth=new EventTarget();synth.getVoices=()=>[{name:'QA Greek voice',lang:'el-GR',localService:true}];
    synth.cancel=()=>{};synth.speak=utterance=>{window.__flow.speech.push({text:utterance.text,lang:utterance.lang});utterance.onstart?.();setTimeout(()=>utterance.onend?.(),30);};
    Object.defineProperty(window,'speechSynthesis',{value:synth});window.SpeechSynthesisUtterance=class{constructor(text){this.text=text;}};
    navigator.mediaDevices.getUserMedia=async constraints=>{
      window.__flow.calls++;window.__flow.constraints.push(constraints);
      if(window.__flow.deny)throw new DOMException('QA permission denied','NotAllowedError');
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;const ctx=canvas.getContext('2d');let n=0;
      const timer=setInterval(()=>{ctx.fillStyle=`rgb(${n++%20},${n%20},${n%20})`;ctx.fillRect(0,0,640,480);},33);
      const stream=canvas.captureStream(30);for(const track of stream.getTracks()){const stop=track.stop.bind(track);track.stop=()=>{clearInterval(timer);stop();};}
      window.__flow.streams.push(stream);return stream;
    };
  });
  const flow=await emulated.newPage();flow.setDefaultTimeout(120000);flow.on('pageerror',e=>report.errors.push(e.message));
  await route('sign',flow);assert.equal(await flow.evaluate(()=>window.__flow.calls),0);
  await flow.getByRole('button',{name:'Άνοιγμα κάμερας',exact:true}).click();await flow.getByRole('alert').waitFor();
  assert.equal(await flow.evaluate(()=>window.__flow.calls),1);
  await flow.evaluate(()=>{window.__flow.deny=false;});await flow.getByRole('button',{name:'Άνοιγμα κάμερας',exact:true}).click();
  await flow.getByRole('button',{name:'Κλείσιμο κάμερας',exact:true}).waitFor();
  await flow.locator('#capture-duration').selectOption('600');await flow.getByRole('button',{name:'Έναρξη λήψης',exact:true}).click();
  await flow.locator('.countdown-overlay strong').filter({hasText:'3'}).waitFor();
  await flow.locator('.result.rejected').waitFor();assert.equal(await flow.locator('.countdown-overlay').count(),0);
  await flow.getByRole('button',{name:'Κλείσιμο κάμερας',exact:true}).click();
  assert.equal(await flow.evaluate(()=>window.__flow.streams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))),true);
  assert.equal(await flow.evaluate(()=>window.__flow.constraints.every(c=>c.audio===false)),true);
  report.emulatedApiChecks.push('Permission denied/retry; explicit opening;3s countdown;600ms auto-stop; actual MediaPipe rejects generated blank canvas; camera close stops tracks; no microphone. No human camera was used.');
  await flow.getByRole('button',{name:'Άνοιγμα κάμερας',exact:true}).click();await flow.getByRole('button',{name:'Κλείσιμο κάμερας',exact:true}).waitFor();
  await flow.getByRole('button',{name:'Έναρξη λήψης',exact:true}).click();await flow.locator('.countdown-overlay').waitFor();
  await flow.getByRole('button',{name:'Ακύρωση',exact:true}).click();await flow.getByRole('button',{name:'Έναρξη λήψης',exact:true}).waitFor();
  await flow.evaluate(()=>{location.hash='#/try';});await flow.locator('#video-file').waitFor();
  assert.equal(await flow.evaluate(()=>window.__flow.streams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))),true);
  await flow.getByRole('button',{name:'γεια',exact:true}).click();
  await flow.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
  await flow.getByRole('button',{name:'Ανάλυση βίντεο',exact:true}).click();await flow.locator('.result.accepted').waitFor();
  assert.equal(await flow.evaluate(()=>window.__flow.speech.length),0);
  await flow.getByRole('button',{name:'Επιβεβαίωση και εκφώνηση',exact:true}).click();
  const spoken=await flow.evaluate(()=>window.__flow.speech);assert.equal(spoken.length,1);assert.equal(spoken[0].lang,'el-GR');
  report.emulatedApiChecks.push('Countdown cancellation and route changes stop resources. With emulated Greek speech API, speak is called only after explicit confirmation; actual speech audio is not claimed.');
  await emulated.close();
  report.requests=[...requests];assert.ok([...requests].every(url=>new URL(url).origin===new URL(base).origin));
  assert.deepEqual(report.errors,[]);report.passed=true;
}catch(error){failure=error;report.failure=String(error?.stack??error);await shot('failure.png').catch(()=>{});}
finally{await browser.close();await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));}
console.log(JSON.stringify({passed:report.passed,realMp4Runs:report.realMp4Runs.length,checks:report.checks.length,emulatedApiChecks:report.emulatedApiChecks.length,report:path.join(out,'report.json'),failure:report.failure}));
if(failure)throw failure;
