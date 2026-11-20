import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import headers from './deployment/headers.json' with { type: 'json' };
import { localAssistantPreview } from './local-assistant-preview.ts';

export default defineConfig(({ mode }) => {
  const assistant = mode === 'signbridge-local-assistant';
  const openHands = mode === 'signbridge-openhands' || assistant;
  const fullApp = mode === 'signbridge-app' || openHands;
  const openHandsPreview = {
    name: 'signbridge-openhands-preview-mime',
    configurePreviewServer(server: import('vite').PreviewServer) {
      for (const model of ['stgcn.onnx', 'lstm.onnx']) {
        server.middlewares.use(`/__signbridge/openhands/${model}`, (_req, res, next) => {
          res.setHeader('Content-Type', 'application/octet-stream');
          next();
        });
      }
    },
  };
  return {
    plugins: fullApp ? [react(), ...(openHands ? [openHandsPreview] : []), ...(assistant ? [localAssistantPreview()] : [])] : [],
    appType: 'mpa',
    publicDir: false,
    define: { 'import.meta.env.VITE_SIGNBRIDGE_PUBLIC_OPENHANDS': JSON.stringify(openHands ? '1' : '0'),
      'import.meta.env.VITE_SIGNBRIDGE_LOCAL_ASSISTANT': JSON.stringify(assistant ? '1' : '0') },
    worker: { format: 'es' },
    build: {
      outDir: assistant ? '../publication-local-assistant/dist' : openHands ? '../publication-openhands/dist'
        : fullApp ? '../publication-app/dist' : '../publication-v8/dist',
      emptyOutDir: false,
      assetsInlineLimit: 0,
      rolldownOptions: {
        input: fileURLToPath(new URL(fullApp ? './index.html' : './v8-pilot.html', import.meta.url)),
      },
    },
    preview: { host: '127.0.0.1', headers },
  };
});
