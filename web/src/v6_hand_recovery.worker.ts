/// <reference lib="webworker" />
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision';
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { measureRawTracking, TRACKING_POLICY } from './tracking';
import type { LandmarkFrame, Point } from './features';
const dist=(a:Point,b:Point,w:number,h:number)=>Math.hypot((a.x-b.x)*w,(a.y-b.y)*h);
self.onmessage=async({data})=>{
 try{
  const started=performance.now(),rule=data.rule;
  const files=await FilesetResolver.forVisionTasks('/vendor/mediapipe',true);
  const holistic=await HolisticLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/holistic_landmarker.task',delegate:'CPU'},runningMode:'IMAGE',outputPoseSegmentationMasks:false,outputFaceBlendshapes:false,minFaceDetectionConfidence:.5,minFacePresenceConfidence:.5,minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minHandLandmarksConfidence:.5});
  // Each ESM task needs its own module realm in this pinned runtime.
  const handWorker=new Worker(new URL('./v6_hand_detector.worker.ts',import.meta.url),{type:'module'});
  const handCall=(payload:unknown,transfer:Transferable[]=[])=>new Promise<HandLandmarkerResult>((resolve,reject)=>{
   handWorker.onmessage=({data})=>data.error?reject(new Error(data.error)):resolve(data);
   handWorker.onerror=e=>reject(new Error(e.message));handWorker.postMessage(payload,transfer);
  });
  await handCall({kind:'init',modelUrl:data.handModelUrl});
  const initMs=performance.now()-started,frames=[];
  for(let i=0;i<data.paths.length;i++){
   const startFrame=performance.now(),image=await createImageBitmap(await(await fetch(data.paths[i])).blob()),w=image.width,h=image.height;
   const canvas=new OffscreenCanvas(w,h),ctx=canvas.getContext('2d')!;ctx.drawImage(image,0,0);image.close();let occlusion:number[]|null=null;
   if(data.control==='all_black'){ctx.fillStyle='black';ctx.fillRect(0,0,w,h);}
   if(data.control==='target_wrist_occlusion'){
    const ref=data.reference[i];if(ref.pose.length>=25){const wrist=ref.pose[16],s=dist(ref.pose[11],ref.pose[12],w,h),rad=Math.max(35,.45*s);
     let x0=wrist.x*w-rad,y0=wrist.y*h-rad,x1=wrist.x*w+rad,y1=wrist.y*h+rad;
     for(const p of ref.right){x0=Math.min(x0,p.x*w-12);y0=Math.min(y0,p.y*h-12);x1=Math.max(x1,p.x*w+12);y1=Math.max(y1,p.y*h+12);}
     occlusion=[Math.max(0,x0),Math.max(0,y0),Math.min(w,x1),Math.min(h,y1)];ctx.fillStyle='black';ctx.fillRect(occlusion[0],occlusion[1],occlusion[2]-occlusion[0],occlusion[3]-occlusion[1]);}}
   const detectStart=performance.now(),r=holistic.detect(canvas),holisticMs=performance.now()-detectStart;
   const original:LandmarkFrame={t:i*1000/30,width:w,height:h,left:r.leftHandLandmarks[0]??[],right:r.rightHandLandmarks[0]??[],pose:r.poseLandmarks[0]??[],face:r.faceLandmarks[0]??[]};
   const candidates:{points:Point[];category:string;score:number;targetDistance:number;oppositeDistance:number;reasons:string[];plausible:boolean}[]=[];
   let roi:number[]|null=null,handMs=0,recovered=false,skipReason='original_right_present';let fused=original;
   if(original.right.length<21||data.probe){
    skipReason='pose_ineligible';const pose=original.pose;
    if(pose.length>=25&&[11,12,15,16].every(j=>Number.isFinite(pose[j].x)&&Number.isFinite(pose[j].y)&&(pose[j].visibility??1)>=rule.minimumPoseVisibility)){
     const s=dist(pose[11],pose[12],w,h),side=Math.round(Math.min(rule.roiSideMaximumPixels,Math.max(rule.roiSideMinimumPixels,rule.roiSideShoulderWidths*s)));
     const x=Math.round(pose[16].x*w-side/2),y=Math.round(pose[16].y*h-side/2);roi=[x,y,side,side];
     const crop=new OffscreenCanvas(side,side),c=crop.getContext('2d')!;c.fillStyle='black';c.fillRect(0,0,side,side);c.drawImage(canvas,-x,-y);
     const hs=performance.now(),bitmap=crop.transferToImageBitmap(),hr=await handCall({kind:'detect',bitmap},[bitmap]);handMs=performance.now()-hs;skipReason='no_plausible_candidate';
     for(let k=0;k<hr.landmarks.length;k++){
      const points=hr.landmarks[k].map(p=>({...p,x:(p.x*side+x)/w,y:(p.y*side+y)/h,z:p.z*side/w}));
      const category=hr.handedness[k]?.[0]?.categoryName??'',score=hr.handedness[k]?.[0]?.score??0;
      const targetDistance=dist(points[0],pose[16],w,h)/s,oppositeDistance=dist(points[0],pose[15],w,h)/s,reasons:string[]=[];
      if(points.length!==21||points.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)||!Number.isFinite(p.z)||p.x<0||p.x>1||p.y<0||p.y>1))reasons.push('invalid_points');
      if(category!==rule.requiredHandedness||score<rule.minimumHandednessScore)reasons.push('handedness');
      if(!(targetDistance<=rule.candidateWristDistanceMaximumShoulderWidths))reasons.push('far_from_target_wrist');
      if(!(oppositeDistance>=rule.oppositeWristDistanceMinimumShoulderWidths&&oppositeDistance>targetDistance*rule.oppositeDistanceMustExceedTargetDistanceBy))reasons.push('opposite_hand_ambiguity');
      candidates.push({points,category,score,targetDistance,oppositeDistance,reasons,plausible:reasons.length===0});
     }
     const plausible=candidates.filter(c=>c.plausible);
     if(plausible.length===1){skipReason=data.probe?'preflight_candidate':'recovered';if(!data.probe&&original.right.length<21){fused={...original,right:plausible[0].points};recovered=true;}}
     if(plausible.length>1)skipReason='ambiguous_candidates';
    }
   }
   frames.push({original,fused,recovered,skipReason,candidates,roi,occlusion,holisticMs,handMs,frameMs:performance.now()-startFrame});
  }
  holistic.close();handWorker.terminate();self.postMessage({frames,initMs,totalMs:performance.now()-started,guardPolicy:TRACKING_POLICY,originalGuard:measureRawTracking(frames.map(f=>f.original)),fusedGuard:measureRawTracking(frames.map(f=>f.fused))});
 }catch(e){self.postMessage({error:e instanceof Error?e.stack:String(e)});}
};
