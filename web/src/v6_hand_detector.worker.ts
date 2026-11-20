/// <reference lib="webworker" />
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
let detector:HandLandmarker|undefined;
self.onmessage=async({data})=>{
 try{
  if(data.kind==='init'){
   detector=await HandLandmarker.createFromOptions(await FilesetResolver.forVisionTasks('/vendor/mediapipe',true),{
    baseOptions:{modelAssetPath:data.modelUrl,delegate:'CPU'},runningMode:'IMAGE',numHands:2,
    minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5,
   });self.postMessage({ready:true});
  }else{
   const bitmap:ImageBitmap=data.bitmap;
   try{if(!detector)throw new Error('Hand detector not initialized');const r=detector.detect(bitmap);self.postMessage({landmarks:r.landmarks,worldLandmarks:r.worldLandmarks,handedness:r.handedness});}
   finally{bitmap.close();}
  }
 }catch(e){self.postMessage({error:e instanceof Error?e.stack:String(e)});}
};
