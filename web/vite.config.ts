import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { webcamCaptureMiddleware } from './scripts/webcam-capture-middleware';

// MediaPipe/ORT import their own pinned WASM glue dynamically. Serve these files
// exactly as in production, including Vite's ?import query, without transforming
// third-party Emscripten glue as application source.
const runtimeAssets = {
  name: 'signbridge-local-runtime-assets',
  configureServer(server: any) {
    server.middlewares.use((req: any, res: any, next: () => void) => {
      const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;
      if (!/^\/vendor\/(mediapipe|onnxruntime)\/[a-zA-Z0-9_.-]+$/.test(pathname)) return next();
      const file = resolve('public', pathname.slice(1));
      if (!existsSync(file)) return next();
      res.setHeader('Content-Type', pathname.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
      createReadStream(file).pipe(res);
    });
  },
};

const modelAssets: Record<string, [string, string]> = {
  'openhands/stgcn.onnx': ['research/cache/openhands-gsl/stgcn.onnx', 'application/octet-stream'],
  'openhands/lstm.onnx': ['research/cache/openhands-gsl/lstm.onnx', 'application/octet-stream'],
  'openhands/stgcn-onnx.json': ['research/cache/openhands-gsl/stgcn-onnx.json', 'application/json'],
  'openhands/policy.json': ['reports/pretrained-gsl/stgcn-gate-agreement-candidate.json', 'application/json'],
  'continuous/sequence.onnx': ['artifacts/continuous-v1-expanded/browser/sequence.onnx', 'application/octet-stream'],
  'continuous/sequence-config.json': ['artifacts/continuous-v1-expanded/browser/sequence-config.json', 'application/json'],
};
const localModels = {
  name: 'signbridge-local-models',
  configureServer(server: import('vite').ViteDevServer) {
    server.middlewares.use('/__signbridge', (req, res) => {
      const name = new URL(req.url ?? '/', 'http://localhost').pathname.slice(1);
      const asset = Object.hasOwn(modelAssets, name) ? modelAssets[name] : undefined;
      if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress ?? '')) {
        res.statusCode = 403;
        res.end();
        return;
      }
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.statusCode = 405;
        res.setHeader('Allow', 'GET, HEAD');
        res.end();
        return;
      }
      const file = asset ? resolve('..', asset[0]) : undefined;
      const stats = file ? statSync(file, { throwIfNoEntry: false }) : undefined;
      if (!asset || !file || !stats?.isFile()) {
        res.statusCode = 404;
        res.end();
        return;
      }
      res.setHeader('Content-Type', asset[1]);
      res.setHeader('Content-Length', stats.size);
      res.setHeader('Cache-Control', 'no-store');
      if (req.method === 'HEAD') { res.end(); return; }
      const source = createReadStream(file);
      source.on('error', error => res.destroy(error));
      res.on('close', () => source.destroy());
      source.pipe(res);
    });
  },
};

export default defineConfig({
  plugins: [runtimeAssets, localModels, webcamCaptureMiddleware(resolve('..')), react()],
  server: {
    fs: { allow: ['..'] },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      // This response policy also covers module workers, where MediaPipe runs.
      // Keep local blobs and the Vite loopback reload socket available.
      'Content-Security-Policy': "connect-src 'self' blob: ws://127.0.0.1:* ws://localhost:*",
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
