/** Prepare a reviewed V4 static artifact. No cloud registration/deployment. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const project=path.resolve(web,'..'), output=path.join(project,'publication-v4'), dist=path.join(output,'dist');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const json=async filename=>JSON.parse(await readFile(filename,'utf8'));
function containsLocalUserPath(value) {
  if(typeof value==='string')return /[A-Za-z]:[\\/]+Users[\\/]+/i.test(value);
  if(Array.isArray(value))return value.some(containsLocalUserPath);
  if(value&&typeof value==='object')return Object.entries(value).some(([key,item])=>containsLocalUserPath(key)||containsLocalUserPath(item));
  return false;
}
async function files(directory,prefix='') {
  const found=[];
  for(const item of await readdir(directory,{withFileTypes:true})) {
    assert.ok(!item.isSymbolicLink(),'Symlinks are not publication inputs.');
    const name=prefix+item.name;
    if(item.isDirectory()) found.push(...await files(path.join(directory,item.name),name+'/'));
    else found.push(name);
  }
  return found.sort();
}
const source=path.join(web,'dist');
const config=await json(path.join(source,'models/model_config.json'));
const model=await readFile(path.join(source,'models/model.onnx'));
const locked=await json(path.join(project,'artifacts/open-set-v2/selection.locked.json'));
assert.equal(config.candidate,locked.selected_candidate);
assert.equal(hash(model),locked.candidates[locked.selected_candidate].export.model_sha256);
assert.equal(hash(await readFile(path.join(source,'models/model_config.json'))),locked.candidates[locked.selected_candidate].export.config_sha256);
const catalog=await json(path.join(source,'library/catalog.json'));
const reviewBytes=await readFile(path.join(project,'reports/v4/text-to-sign-rights-review.json'));
const rightsReview=JSON.parse(reviewBytes.toString('utf8'));
assert.equal(catalog.reviewSha256,hash(reviewBytes),'Catalog does not match the completed rights/source review.');
assert.equal(rightsReview.clips.length,3,'This release reviews exactly three clips.');
assert.equal(catalog.entries.length,3,'This release publishes exactly the three reviewed phrases.');
const reviewedBySource=new Map(rightsReview.clips.map(clip=>[clip.source_path,clip]));
assert.equal(reviewedBySource.size,3,'Duplicate source in rights review.');
assert.equal(new Set(catalog.entries.map(entry=>entry.sourceUtterance)).size,3,'Catalog repeats a source.');
const licensed=new Map(catalog.entries.map(entry=>[entry.videoUrl.replace(/^\//,''),entry]));
assert.equal(licensed.size,3,'Catalog repeats a video path.');
for(const entry of catalog.entries) {
  const reviewed=reviewedBySource.get(entry.sourceUtterance);
  assert.ok(reviewed,'A catalog source was not part of the completed review.');
  assert.match(entry.videoUrl,/^\/library\/[a-z0-9-]+\.mp4$/);
  assert.equal(entry.sha256,reviewed.mp4_sha256,'Video differs from reviewed MP4.');
  assert.equal(entry.annotationLabel,reviewed.raw_label);
  assert.equal(entry.phrase,reviewed.utterance_text.trim(),'Phrase differs from reviewed annotation.');
  assert.deepEqual(entry.aliases,[],'No additional phrase aliases were reviewed.');
  assert.equal(entry.sourceUrl,rightsReview.source.record_url);
  assert.equal(entry.sourceArchiveUrl,reviewed.source_url);
  assert.ok(Math.abs(entry.durationSeconds-reviewed.duration_seconds)<1e-9);
  assert.equal(entry.license.id,'CC-BY-4.0');
  assert.equal(entry.license.url,rightsReview.rights.license_url);
  assert.equal(entry.rights.redistributionConfirmed,true);
  assert.equal(entry.verification.status,'dataset-annotated');
  assert.ok(entry.sha256 && entry.attribution && entry.modifications);
  assert.equal(hash(await readFile(path.join(source,entry.videoUrl))),entry.sha256);
}
const allowed=/^(index\.html|NOTICE\.txt|assets\/[A-Za-z0-9_.-]+\.(?:js|css|wasm|svg)|models\/(?:model\.onnx|model_config\.json|holistic_landmarker\.task)|reports\/(?:technical-report|app-evidence)\.json|library\/(?:catalog\.json|NOTICE\.txt|[a-z0-9-]+\.mp4)|vendor\/(?:mediapipe|onnxruntime)\/[A-Za-z0-9_.-]+)$/;
const sourceFiles=await files(source);
assert.ok(sourceFiles.every(file=>allowed.test(file)),`Unexpected public asset: ${sourceFiles.filter(file=>!allowed.test(file))}`);
for(const name of sourceFiles) {
  if(name.endsWith('.mp4')) assert.ok(licensed.has(name),`Video lacks explicit catalog rights: ${name}`);
}
await mkdir(dist,{recursive:true});
for(const name of sourceFiles) {
  const target=path.join(dist,name);await mkdir(path.dirname(target),{recursive:true});
  await copyFile(path.join(source,name),target);
}
const licenseRoot=path.join(web,'deployment/licenses'), licenseFiles=await files(licenseRoot);
assert.ok(licenseFiles.includes('provenance.json'));
for(const name of licenseFiles) {
  const target=path.join(dist,'licenses',name);await mkdir(path.dirname(target),{recursive:true});
  await copyFile(path.join(licenseRoot,name),target);
}
const headers=await json(path.join(web,'deployment/headers.json'));
await writeFile(path.join(dist,'_headers'),'/*\n'+Object.entries(headers).map(([key,value])=>`  ${key}: ${value}`).join('\n')+'\n');
const staticVercel={version:2,framework:null,buildCommand:'',outputDirectory:'dist',headers:[{source:'/(.*)',headers:Object.entries(headers).map(([key,value])=>({key,value}))}]};
await writeFile(path.join(output,'vercel.json'),JSON.stringify(staticVercel,null,2)+'\n');
const expected=new Set([...sourceFiles,...licenseFiles.map(name=>'licenses/'+name),'_headers']);
const inventory=[];
for(const name of await files(dist)) {
  assert.ok(expected.has(name),`Stale/unexpected output requires review: ${name}`);
  assert.ok(!/(^|\/)(?:data|src|node_modules|\.git|\.env)(\/|$)|webcam-check|extract\.html|\.map$|\.(?:jpg|jpeg|webm)$/i.test(name));
  const bytes=await readFile(path.join(dist,name));
  assert.ok(bytes.length<=25*1024*1024,`Asset exceeds reference static limit: ${name}`);
  if(/\.(?:json|txt|html|js|css)$/.test(name)) {
    const text=bytes.toString('utf8');
    // Raw scan catches doubled JS/JSON backslashes; parsing also catches JSON unicode escapes.
    assert.ok(!containsLocalUserPath(text),`Local path leaked: ${name}`);
    if(name.endsWith('.json'))assert.ok(!containsLocalUserPath(JSON.parse(text)),`Escaped local path leaked: ${name}`);
  }
  inventory.push({path:name,bytes:bytes.length,sha256:hash(bytes)});
}
const evidence=await json(path.join(dist,'reports/app-evidence.json'));
const audit={status:'prepared_not_published',createdAt:new Date().toISOString(),release:'V4',
  modelVersion:config.modelVersion,modelSha256:hash(model),inputPolicyVersion:evidence.currentV4.trackingPolicyVersion,
  files:inventory,totalBytes:inventory.reduce((sum,item)=>sum+item.bytes,0),licensedVideoCount:licensed.size,
  rightsReviewSha256:hash(reviewBytes),reviewedPhraseSources:catalog.entries.map(entry=>({phrase:entry.phrase,source:entry.sourceUtterance,mp4Sha256:entry.sha256})),
  datasetArchivesIncluded:false,developmentEndpointsIncluded:false,humanWebcamTested:false,independentNewSignerEvaluation:false,
  historicalV2:evidence.historicalV2,sourceReportScope:'Historical metrics precede V4 input guard.',
  sourceModelUnchanged:true,cloudRegistrationPerformed:false,published:false};
await writeFile(path.join(output,'publication-audit.json'),JSON.stringify(audit,null,2)+'\n');
await copyFile(path.join(web,'deployment/V4_DEPLOYMENT.md'),path.join(output,'README.md'));
console.log(JSON.stringify({directory:output,status:audit.status,files:inventory.length,totalBytes:audit.totalBytes,licensedVideos:licensed.size}));
