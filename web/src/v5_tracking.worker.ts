/// <reference lib="webworker" />
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision';
type P = {x:number;y:number;z:number;visibility?:number};
self.onmessage = async ({data}) => {
  try {
    const detector = await HolisticLandmarker.createFromOptions(await FilesetResolver.forVisionTasks('/vendor/mediapipe', true), {
      baseOptions:{modelAssetPath:'/models/holistic_landmarker.task',delegate:'CPU'},runningMode:'IMAGE',
      outputPoseSegmentationMasks:false,outputFaceBlendshapes:false,minFaceDetectionConfidence:.5,
      minFacePresenceConfidence:.5,minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minHandLandmarksConfidence:.5,
    });
    const frames=[]; let roiApi='not-probed';
    for(let i=0;i<data.paths.length;i++) {
      const started=performance.now();
      const image=await createImageBitmap(await (await fetch(data.paths[i])).blob());
      const w=image.width,h=image.height,ref=data.reference[i];
      const base=new OffscreenCanvas(w,h),ctx=base.getContext('2d')!;ctx.drawImage(image,0,0);image.close();
      let occlusion:number[]|null=null;
      if(data.control==='all_black'){ctx.fillStyle='black';ctx.fillRect(0,0,w,h);}
      if(data.control==='target_wrist_occlusion'&&ref.pose.length===33){
        const wrist=ref.pose[16],shoulder=Math.hypot((ref.pose[11].x-ref.pose[12].x)*w,(ref.pose[11].y-ref.pose[12].y)*h),radius=Math.max(35,.45*shoulder);
        let x0=wrist.x*w-radius,y0=wrist.y*h-radius,x1=wrist.x*w+radius,y1=wrist.y*h+radius;
        for(const p of ref.right){x0=Math.min(x0,p.x*w-12);y0=Math.min(y0,p.y*h-12);x1=Math.max(x1,p.x*w+12);y1=Math.max(y1,p.y*h+12);}
        occlusion=[Math.max(0,x0),Math.max(0,y0),Math.min(w,x1),Math.min(h,y1)];
        ctx.fillStyle='black';ctx.fillRect(occlusion[0],occlusion[1],occlusion[2]-occlusion[0],occlusion[3]-occlusion[1]);
      }
      let input=base,ox=0,oy=0,sx=w,sy=h;
      if(data.variant==='pad10_black'){
        const px=Math.round(w*.1),py=Math.round(h*.1);input=new OffscreenCanvas(w+2*px,h+2*py);const c=input.getContext('2d')!;
        c.fillStyle='black';c.fillRect(0,0,input.width,input.height);c.drawImage(base,px,py);ox=-px;oy=-py;sx=input.width;sy=input.height;
      }else if(data.variant==='pose_upper_crop'){
        const points=ref.pose.slice(0,25).filter((p:P)=>(p.visibility??1)>=.5&&Number.isFinite(p.x)&&Number.isFinite(p.y));
        if(points.length>=8){
          const xs=points.map((p:P)=>p.x*w),ys=points.map((p:P)=>p.y*h);const dx=Math.max(...xs)-Math.min(...xs),dy=Math.max(...ys)-Math.min(...ys);
          ox=Math.max(0,Math.floor(Math.min(...xs)-dx*.2));oy=Math.max(0,Math.floor(Math.min(...ys)-dy*.2));
          sx=Math.min(w,Math.ceil(Math.max(...xs)+dx*.2))-ox;sy=Math.min(h,Math.ceil(Math.max(...ys)+dy*.2))-oy;
          if(sx>20&&sy>20){input=new OffscreenCanvas(sx,sy);input.getContext('2d')!.drawImage(base,ox,oy,sx,sy,0,0,sx,sy);}else{ox=oy=0;sx=w;sy=h;}
        }
      }
      // Probe after the measured sequence: don't alter graph history before observations.
      const result=detector.detect(input);
      const map=(p:P)=>({...p,x:(p.x*sx+ox)/w,y:(p.y*sy+oy)/h,z:p.z*sx/w});
      frames.push({t:i*1000/30,width:w,height:h,left:(result.leftHandLandmarks[0]??[]).map(map),right:(result.rightHandLandmarks[0]??[]).map(map),pose:(result.poseLandmarks[0]??[]).map(map),faceCount:result.faceLandmarks[0]?.length??0,geometry:{ox,oy,sx,sy,occlusion},elapsedMs:performance.now()-started});
      if(i===data.paths.length-1){try{detector.detect(input,{regionOfInterest:{left:.1,top:.1,right:.9,bottom:.9}});roiApi='accepted';}catch(e){roiApi=e instanceof Error?e.message:String(e);}}
    }
    detector.close();self.postMessage({frames,roiApi});
  }catch(e){self.postMessage({error:e instanceof Error?e.stack:String(e)});}
};
