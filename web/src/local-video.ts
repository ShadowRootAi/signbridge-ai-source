function waitForVideo(video: HTMLVideoElement, event: 'loadeddata' | 'seeked', signal: AbortSignal, start: () => void) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const ready = () => finish();
    const error = () => finish(new Error('Το αρχείο βίντεο δεν ανοίγει. Δοκίμασε MP4 ή WebM.'));
    const abort = () => finish(signal.reason);
    const timeout = setTimeout(() => finish(new Error('Η φόρτωση του βίντεο καθυστέρησε.')), 15_000);
    function finish(failure?: unknown) {
      clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
      video.removeEventListener('error', error);
      video.removeEventListener(event, ready);
      if (failure) reject(failure);
      else resolve();
    }
    video.addEventListener(event, ready, { once: true });
    video.addEventListener('error', error, { once: true });
    signal.addEventListener('abort', abort, { once: true });
    try { start(); } catch (failure) { finish(failure); }
  });
}

export function localVideoUrl(blob: Blob) {
  if (!blob.size || blob.size > 40_000_000) throw new Error('Επίλεξε αρχείο βίντεο έως 40 MB.');
  return URL.createObjectURL(blob);
}

export async function loadLocalVideo(video: HTMLVideoElement, url: string, signal: AbortSignal) {
  await waitForVideo(video, 'loadeddata', signal, () => { video.src = url; video.load(); });
  if (!Number.isFinite(video.duration)) {
    await waitForVideo(video, 'seeked', signal, () => { video.currentTime = 1e8; });
    await waitForVideo(video, 'seeked', signal, () => { video.currentTime = 0; });
  }
  if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth || !video.videoHeight) {
    throw new Error('Επίλεξε ένα αναγνώσιμο αρχείο βίντεο.');
  }
}

