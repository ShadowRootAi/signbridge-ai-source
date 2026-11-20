import { execFile } from 'node:child_process';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

const endpoint = '/__signbridge/local-assistant';
const python = fileURLToPath(new URL('../.venv/Scripts/python.exe', import.meta.url));
const assistant = fileURLToPath(new URL('../ml/local_assistant.py', import.meta.url));
const maxBytes = 16_384;

function send(response: ServerResponse, status: number, body: unknown) {
  if (response.destroyed || response.writableEnded) return;
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(body));
}

function localHost(request: IncomingMessage) {
  const port = request.socket.localPort;
  const host = request.headers.host;
  return host === `127.0.0.1:${port}` || host === `localhost:${port}`;
}

export function localAssistantPreview(): Plugin {
  return {
    name: 'signbridge-local-assistant',
    configurePreviewServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = request.url?.split('?')[0];
        if (path !== `${endpoint}/chat` && path !== `${endpoint}/status`) return next();
        if (!localHost(request)) return send(response, 403, { error: 'Ο βοηθός δέχεται μόνο τοπικά αιτήματα.' });
        if (path === `${endpoint}/status` && request.method === 'GET') {
          return send(response, 200, { service: 'signbridge-local-assistant', model: 'qwen3:4b' });
        }
        if (path !== `${endpoint}/chat` || request.method !== 'POST') {
          return send(response, 405, { error: 'Η μέθοδος δεν υποστηρίζεται.' });
        }
        if (request.headers.origin !== `http://${request.headers.host}`
            || (request.headers['sec-fetch-site'] && request.headers['sec-fetch-site'] !== 'same-origin')) {
          return send(response, 403, { error: 'Το αίτημα πρέπει να προέρχεται από την τοπική εφαρμογή.' });
        }
        if (request.headers['content-type']?.split(';')[0].trim() !== 'application/json') {
          return send(response, 415, { error: 'Αναμενόταν JSON.' });
        }
        const chunks: Buffer[] = [];
        let size = 0;
        request.on('data', (chunk: Buffer) => {
          size += chunk.length;
          if (size <= maxBytes) chunks.push(chunk);
          else send(response, 413, { error: 'Το αίτημα είναι πολύ μεγάλο.' });
        });
        request.on('end', () => {
          if (size > maxBytes || response.destroyed) return;
          const child = execFile(python, ['-B', '-X', 'utf8', assistant], { windowsHide: true, timeout: 115_000,
            maxBuffer: 65_536 }, (error, stdout) => {
            if (!error) {
              try {
                const reply = JSON.parse(stdout) as { status: number; body: unknown };
                if ([200, 400, 500, 503].includes(reply.status)) return send(response, reply.status, reply.body);
              } catch { /* Use the same client error for a failed helper. */ }
            }
            send(response, 503, { error: 'Ο βοηθός δεν είναι διαθέσιμος. Έλεγξε το τοπικό Ollama και δοκίμασε ξανά.' });
          });
          child.stdin?.on('error', () => child.kill());
          child.stdin?.end(Buffer.concat(chunks));
          response.once('close', () => { if (!response.writableEnded) child.kill(); });
        });
      });
    },
  };
}
