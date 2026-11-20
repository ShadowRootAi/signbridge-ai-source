/// <reference lib="webworker" />
import { loadOpenHandsRecognizer } from './openhands-recognizer';
import type { LandmarkFrame } from './features';

const path = '/__signbridge/openhands/';
let loading: ReturnType<typeof loadOpenHandsRecognizer> | undefined;

self.onmessage = async ({ data }: MessageEvent<{ id: number; frames: LandmarkFrame[]; allowRecovery: boolean }>) => {
  try {
    loading ??= loadOpenHandsRecognizer({
      modelUrl: path + 'stgcn.onnx',
      configUrl: path + 'stgcn-onnx.json',
      policyUrl: path + 'policy.json',
      secondaryModelUrl: path + 'lstm.onnx',
    }).catch(error => { loading = undefined; throw error; });
    const recognizer = await loading;
    self.postMessage({ id: data.id, result: await recognizer.classify(data.frames, data.allowRecovery) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) });
  }
};
