import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readdir, writeFile } from 'node:fs/promises';
import { resolve, relative, sep } from 'node:path';
import type { Plugin } from 'vite';

const TRIALS = ['idle', 'ordinary_hand_motion', 'hands_out_of_view', 'position_or_light_change'] as const;
const MAX_BODY = 8 * 1024 * 1024;
const MAX_VIDEO = 5 * 1024 * 1024;

export function webcamCaptureMiddleware(projectRoot: string): Plugin {
  const root = resolve(projectRoot);
  const directory = resolve(root, 'data/open-set-v2/webcam');
  const child = relative(root, directory);
  if (child.startsWith(`..${sep}`) || child === '..') throw new Error('Invalid capture directory.');
  return {
    name: 'signbridge-manual-local-webcam-capture',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;
        if (!pathname.startsWith('/__signbridge/webcam-capture')) return next();
        const reply = (code: number, value: unknown) => {
          res.statusCode = code;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.setHeader('Cache-Control', 'no-store');
          res.end(JSON.stringify(value));
        };
        const peer = req.socket.remoteAddress ?? '';
        if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(peer)) return reply(403, { error: 'Local connections only.' });
        const host = new URL(`http://${req.headers.host ?? 'invalid'}`).hostname;
        if (!['127.0.0.1', 'localhost', '[::1]'].includes(host)) return reply(403, { error: 'Loopback host required.' });
        if (pathname.endsWith('/status') && req.method === 'GET') {
          let savedCount = 0;
          try { savedCount = (await readdir(directory)).filter((name) => name.endsWith('.json')).length; }
          catch (error: any) { if (error.code !== 'ENOENT') return reply(500, { error: 'Cannot inspect local capture directory.' }); }
          return reply(200, { ready: true, localOnly: true, savedCount, directory: 'data/open-set-v2/webcam', trials: TRIALS });
        }
        if (pathname !== '/__signbridge/webcam-capture' || req.method !== 'POST') return reply(405, { error: 'POST required.' });
        const origin = req.headers.origin;
        if (!origin || origin !== `http://${req.headers.host}`) return reply(403, { error: 'Same-origin local request required.' });
        if (!(req.headers['content-type'] ?? '').startsWith('application/json')) return reply(415, { error: 'JSON required.' });
        if (Number(req.headers['content-length'] ?? 0) > MAX_BODY) { req.resume(); return reply(413, { error: 'Capture exceeds the local size limit.' }); }
        try {
          const chunks: Buffer[] = [];
          let length = 0;
          for await (const chunk of req) {
            const bytes = Buffer.from(chunk);
            length += bytes.length;
            if (length > MAX_BODY) return reply(413, { error: 'Capture exceeds the local size limit.' });
            chunks.push(bytes);
          }
          const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          const metadata = body.metadata;
          if (!metadata || metadata.consentLocalRecording !== true || metadata.labelConfirmed !== true
            || metadata.noIntentionalSignLanguage !== true || !TRIALS.includes(metadata.trialKind)) {
            return reply(400, { error: 'Explicit local consent and a confirmed observational category are required.' });
          }
          if (metadata.requestedDurationMs !== 1500 || !Number.isFinite(metadata.actualDurationMs)
            || metadata.actualDurationMs < 200 || metadata.actualDurationMs > 10000) return reply(400, { error: 'Invalid recording duration.' });
          if (typeof body.mimeType !== 'string' || !body.mimeType.startsWith('video/webm')) return reply(415, { error: 'WebM video required.' });
          if (typeof body.videoBase64 !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(body.videoBase64)) return reply(400, { error: 'Invalid video encoding.' });
          const video = Buffer.from(body.videoBase64, 'base64');
          if (video.length > MAX_VIDEO) return reply(413, { error: 'Video exceeds 5 MiB.' });
          if (video.length < 128 || video.subarray(0, 4).toString('hex') !== '1a45dfa3') return reply(400, { error: 'Readable WebM container required.' });
          const timestamps: unknown[] = metadata.observedFrameTimestampsMs ?? [];
          if (!Array.isArray(timestamps) || timestamps.length > 600 || timestamps.some((time) => typeof time !== 'number' || !Number.isFinite(time) || time < 0)) {
            return reply(400, { error: 'Invalid observed frame timestamps.' });
          }
          const id = `webcam-${Date.now()}-${randomUUID()}`;
          const videoName = `${id}.webm`;
          const jsonName = `${id}.json`;
          const settings = metadata.trackSettings ?? {};
          const stored = {
            schemaVersion: 1, id, savedAtUtc: new Date().toISOString(),
            source: 'manual_user_webcam', sourceDeclaration: 'user_confirmed', synthetic: false,
            intendedUse: 'held_out_observational_negative_only', useForThresholdFitting: false,
            label: metadata.trialKind, labelConfirmed: true, noIntentionalSignLanguage: true,
            fluentGslClaim: false, consentLocalRecording: true, audioRecorded: false,
            requestedDurationMs: 1500, actualDurationMs: metadata.actualDurationMs,
            captureStartedAtUtc: String(metadata.captureStartedAtUtc ?? '').slice(0, 40),
            captureFinishedAtUtc: String(metadata.captureFinishedAtUtc ?? '').slice(0, 40),
            observedFrameTimestampsMs: timestamps,
            observedFrameCount: timestamps.length,
            observedFps: typeof metadata.observedFps === 'number' && Number.isFinite(metadata.observedFps) ? metadata.observedFps : null,
            timestampSource: 'HTMLVideoElement presentation callbacks; independent of decoded WebM timestamps',
            trackSettings: { width: settings.width, height: settings.height, frameRate: settings.frameRate, facingMode: settings.facingMode },
            navigator: { userAgent: String(metadata.navigator?.userAgent ?? '').slice(0, 500), platform: String(metadata.navigator?.platform ?? '').slice(0, 80) },
            previewMirrored: false, mimeType: body.mimeType, bytes: video.length,
            videoSha256: createHash('sha256').update(video).digest('hex'),
            videoPath: `data/open-set-v2/webcam/${videoName}`,
            predictionsShownDuringCollection: false,
          };
          await mkdir(directory, { recursive: true });
          await writeFile(resolve(directory, videoName), video, { flag: 'wx' });
          await writeFile(resolve(directory, jsonName), JSON.stringify(stored, null, 2), { flag: 'wx' });
          return reply(201, { saved: true, id, videoPath: stored.videoPath, metadataPath: `data/open-set-v2/webcam/${jsonName}`, videoSha256: stored.videoSha256 });
        } catch (error) {
          return reply(400, { error: error instanceof SyntaxError ? 'Invalid JSON.' : 'The local recording could not be saved.' });
        }
      });
    },
  };
}
