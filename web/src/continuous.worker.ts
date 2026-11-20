/// <reference lib="webworker" />
import { loadContinuousRecognizer } from './continuous-inference';
import type { SequenceTimeline } from './continuous-inference';

const base = '/__signbridge/continuous/';
let loading: ReturnType<typeof loadContinuousRecognizer> | undefined;

self.onmessage = async ({ data }: MessageEvent<{ features: number[][]; timeline: SequenceTimeline }>) => {
  try {
    loading ??= loadContinuousRecognizer(base + 'sequence-config.json', base + 'sequence.onnx');
    const predict = await loading;
    self.postMessage({ result: await predict(data.features, data.timeline) });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
};
