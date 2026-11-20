/** Shared interface between the scientific pipeline and the local proof UI. */
import type { TrackingDiagnostics } from './tracking';
export interface ModelStatus {
  ready: boolean;
  message: string;
  modelVersion?: string;
  labels?: string[];
  /** Only true when rejection thresholds have actually been calibrated. */
  rejectionCalibrated?: boolean;
}

export interface PipelineProgress {
  stage: string;
  message: string;
  /** Actual completed progress between 0 and 1; omit for indeterminate work. */
  fraction?: number;
  frames?: number;
}

export interface RecognitionResult {
  accepted: boolean;
  label: string | null;
  /** Model score, not a probability that the translation is correct. */
  confidence: number | null;
  /** Frozen model softmax score; null when raw completeness blocks inference. Not correctness probability. */
  classificationScore: number | null;
  classifierExecuted?: boolean;
  reason: string;
  frames: number;
  elapsedMs: number;
  trackingQuality?: number;
  tracking?: TrackingDiagnostics;
  modelVersion: string;
}

export type ProgressListener = (progress: PipelineProgress) => void;

// Implemented by pipeline.ts:
// getModelStatus(): Promise<ModelStatus>
// analyzeVideo(video: HTMLVideoElement, progress?: ProgressListener,
//              signal?: AbortSignal): Promise<RecognitionResult>
// analyzeCamera(video: HTMLVideoElement, durationMs: number,
//               progress?: ProgressListener, signal?: AbortSignal): Promise<RecognitionResult>
