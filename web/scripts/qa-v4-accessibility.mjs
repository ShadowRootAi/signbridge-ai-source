/** Bounded DOM/keyboard checks of the real production artifact. Not WCAG certification. */
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const base=(process.argv[2]??'http://127.0.0.1:5176').replace(/\/$/,'');
const output=path.join(root,'reports/v4/product-qa');await mkdir(output,{recursive:true});
const hash=value=>createHash('sha256').update(value).digest('hex');
const auditBytes=await readFile(path.join(root,'publication-v4/publication-audit.json'));
const audit=JSON.parse(auditBytes.toString('utf8'));
const report={scope:'Bounded real-browser keyboard, landmarks, form-label and responsive checks; no WCAG certification or screen-reader/human usability claim',
  createdAt:new Date().toISOString(),base,publicationAuditSha256:hash(auditBytes),passed:false,
  humanWebcamTested:false,actualSpeechAudioVerified:false,voiceAvailabilityEmulated:'Only an empty voice list, to test the disabled speech control. Predictions are real model executions.',
  checks:[],failures:[],pageErrors:[],screenshots:[]};
const browser=await chromium.launch({channel:'chrome',headless:true});report.browserVersion=browser.version();

async function check(viewport,name,run){
  console.log(`Checking ${viewport}px: ${name}`);
  try{const detail=await run();report.checks.push({viewport,name,passed:true,...(detail?{detail}: {})});}
  catch(error){report.failures.push({viewport,name,error:String(error?.stack??error)});console.error(`Failed ${viewport}px: ${name}: ${error.message}`);}
}
async function navigate(page,route){
  await page.goto(`${base}/#/${route==='home'?'':route}`);
  await page.locator(`main.route-${route}`).waitFor();
}
async function screen(page,name){
  await page.screenshot({path:path.join(output,name),fullPage:true});report.screenshots.push(name);
}
async function formsAndStructure(page,route){
  assert.equal(await page.getByRole('main').count(),1);
  assert.equal(await page.getByRole('heading',{level:1}).count(),1);
  const expected={home:'Αρχική',sign:'Νόημα σε κείμενο',library:'Κείμενο σε νόημα',try:'Δοκίμασε το AI',about:'Το AI μας'};
  await page.waitForFunction(title=>document.title===title,`${expected[route]} · SignBridge AI`);
  assert.equal(await page.title(),`${expected[route]} · SignBridge AI`);
  const details=await page.evaluate(()=>{
    const visible=element=>{const style=getComputedStyle(element),rect=element.getBoundingClientRect();return style.display!=='none'&&style.visibility!=='hidden'&&rect.width>0&&rect.height>0;};
    const name=element=>{
      if(element.getAttribute('aria-label')?.trim())return element.getAttribute('aria-label').trim();
      if(element.getAttribute('aria-labelledby'))return element.getAttribute('aria-labelledby').split(/\s+/).map(id=>document.getElementById(id)?.textContent??'').join(' ').trim();
      if(element.labels?.length)return [...element.labels].map(label=>label.textContent??'').join(' ').trim();
      if(element.tagName==='BUTTON'){const clone=element.cloneNode(true);clone.querySelectorAll('[aria-hidden="true"],svg').forEach(item=>item.remove());return clone.textContent.trim();}
      return '';
    };
    const controls=[...document.querySelectorAll('button,input:not([type="hidden"]),select,textarea')].filter(visible).map(element=>({tag:element.tagName,id:element.id,name:name(element)}));
    const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);
    const invalidReferences=[...document.querySelectorAll('[aria-labelledby],[aria-controls]')].flatMap(e=>['aria-labelledby','aria-controls'].flatMap(attribute=>(e.getAttribute(attribute)??'').split(/\s+/).filter(Boolean).filter(id=>!document.getElementById(id)).map(id=>({attribute,id}))));
    return {lang:document.documentElement.lang,overflow:document.documentElement.scrollWidth>innerWidth+1,
      unnamedControls:controls.filter(control=>!control.name),controls,duplicateIds:ids.filter((id,i)=>ids.indexOf(id)!==i),invalidReferences};
  });
  assert.equal(details.lang,'el');assert.equal(details.overflow,false);
  assert.deepEqual(details.unnamedControls,[]);assert.deepEqual(details.duplicateIds,[]);assert.deepEqual(details.invalidReferences,[]);
  if(route==='sign')assert.equal(await page.getByLabel(/^Διάρκεια λήψης/).count(),1);
  if(route==='library')assert.equal(await page.getByLabel('Ελληνική λέξη ή φράση',{exact:true}).count(),1);
  if(route==='try')assert.equal(await page.getByLabel(/Επίλεξε αρχείο/).count(),1);
  return details;
}

try{
  for(const width of [390,1440]){
    const context=await browser.newContext({viewport:{width,height:width===390?844:1080}});
    await context.addInitScript(()=>{
      window.__a11ySpeechCalls=0;
      if(window.speechSynthesis){
        window.speechSynthesis.getVoices=()=>[];
        const original=window.speechSynthesis.speak.bind(window.speechSynthesis);
        window.speechSynthesis.speak=utterance=>{window.__a11ySpeechCalls++;return original(utterance);};
      }
    });
    const page=await context.newPage();page.setDefaultTimeout(120000);page.on('pageerror',error=>report.pageErrors.push({width,message:error.message}));
    const response=await page.goto(base);assert.ok(response?.ok());
    assert.equal(hash(await response.body()),audit.files.find(item=>item.path==='index.html')?.sha256);
    assert.ok(!(await response.text()).includes('/@vite/client'));
    await page.locator('main.route-home h1').waitFor();
    await check(width,'Skip link is first keyboard stop and focuses main',async()=>{
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(()=>document.activeElement?.classList.contains('skip-link')),true);
      const rectangle=await page.locator('.skip-link').boundingBox();assert.ok(rectangle&&rectangle.y+rectangle.height>0&&rectangle.y<page.viewportSize().height);
      await screen(page,`accessibility-${width}-skip-focus.png`);
      await page.keyboard.press('Enter');await page.waitForFunction(()=>document.activeElement?.id==='main-content');
      return {focusedMain:true};
    });
    if(width===390){
      await check(width,'Mobile menu keyboard activation, tab navigation and closure',async()=>{
        const toggle=page.getByRole('button',{name:'Άνοιγμα μενού',exact:true});assert.equal(await toggle.isVisible(),true);
        assert.equal(await toggle.getAttribute('aria-expanded'),'false');await toggle.focus();await page.keyboard.press('Enter');
        assert.equal(await page.getByRole('button',{name:'Κλείσιμο μενού',exact:true}).getAttribute('aria-expanded'),'true');
        await screen(page,'accessibility-390-menu-open.png');
        let reached=false;
        for(let i=0;i<10;i++){
          await page.keyboard.press('Tab');
          if(await page.evaluate(()=>document.activeElement?.closest('#main-navigation')&&document.activeElement?.getAttribute('href')==='#/library')){reached=true;break;}
        }
        assert.ok(reached,'Library navigation link must be reachable with Tab from the menu toggle.');
        await page.keyboard.press('Enter');await page.locator('main.route-library').waitFor();
        await page.waitForFunction(()=>document.activeElement?.id==='main-content');
        assert.equal(await page.getByRole('button',{name:'Άνοιγμα μενού',exact:true}).getAttribute('aria-expanded'),'false');
        assert.equal(await page.locator('#main-navigation').isVisible(),false);
        return {linkReachedByTab:true,mainFocusedAfterNavigation:true,menuClosed:true};
      });
    }
    for(const route of ['home','sign','library','try','about']){
      await navigate(page,route);
      await check(width,`Structure, labels and viewport: ${route}`,()=>formsAndStructure(page,route));
    }
    await check(width,'Library form submits by Enter; unsupported phrase stays explicit',async()=>{
      await navigate(page,'library');await page.getByRole('button',{name:'ευχαριστώ',exact:true}).waitFor();
      const input=page.getByLabel('Ελληνική λέξη ή φράση',{exact:true});await input.fill('ΕΥΧΑΡΙΣΤΩ');await input.press('Enter');
      await page.getByRole('heading',{name:'ευχαριστώ',exact:true}).waitFor();
      const media=page.getByLabel('Απόδοση της φράσης ευχαριστώ στη νοηματική',{exact:true});assert.equal(await media.count(),1);
      assert.equal(await media.evaluate(video=>video.controls),true);
      await input.fill('γεια ευχαριστώ');await input.press('Enter');
      await page.getByRole('heading',{name:'Αυτή η φράση δεν υποστηρίζεται ακόμη.',exact:true}).waitFor();
      assert.equal(await page.locator('.library-video video').count(),0);
    });
    if(width===390){
      await check(width,'Real accepted result keeps speech disabled when Greek voice inventory is empty',async()=>{
        await navigate(page,'try');await page.getByRole('button',{name:'γεια',exact:true}).click();
        const analyze=page.getByRole('button',{name:'Ανάλυση βίντεο',exact:true});
        await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Ανάλυση βίντεο')&&!b.disabled));
        await analyze.focus();await page.keyboard.press('Enter');await page.locator('.result.accepted').waitFor();
        const speak=page.getByRole('button',{name:'Επιβεβαίωση και εκφώνηση',exact:true});assert.equal(await speak.isDisabled(),true);
        await page.getByText('Δεν βρέθηκε ελληνική φωνή σε αυτόν τον browser.',{exact:true}).waitFor();
        assert.equal(await page.evaluate(()=>window.__a11ySpeechCalls),0);
        await screen(page,'accessibility-390-no-greek-voice.png');
        return {realMp4:'licensed geia fixture, already exposed; integration only',voiceList:'explicitly emulated empty',automaticSpeechCalls:0};
      });
    }
    await context.close();
  }
  assert.equal(hash(await readFile(path.join(root,'publication-v4/publication-audit.json'))),report.publicationAuditSha256,'Publication changed during checks.');
  report.passed=report.failures.length===0&&report.pageErrors.length===0;
}catch(error){report.failures.push({name:'Harness or publication identity',error:String(error?.stack??error)});}
finally{await browser.close();await writeFile(path.join(output,'accessibility.json'),JSON.stringify(report,null,2));}
console.log(JSON.stringify({passed:report.passed,checks:report.checks.length,failures:report.failures,pageErrors:report.pageErrors,report:path.join(output,'accessibility.json')}));
if(!report.passed)process.exitCode=1;
