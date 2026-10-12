// extensions/trainer/main.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Neutralino extension — bridges GUI to Brymar College CLI training loop.
// Spawns `node index.js train` with options from GUI, pipes stdout as events.
// The training core is never modified — this extension is a thin pipe.

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { spawn } = require('child_process');
const WebSocket  = require('ws');
const path       = require('path');
const { fileURLToPath } = require('url');

let NL_PORT;
let NL_TOKEN;
let NL_CONNECTTOKEN;
let NL_EXTID = 'org.consciousnode.brymar.trainer';

// Root of the Brymar College project (two levels up from extensions/trainer/)
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const fs = require('fs');

let ws;
let trainingProcess = null;

// ── Neutralino event sender ───────────────────────────────────────────────────

function sendEvent(event, data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      id:          Date.now().toString(),
      method:      'app.broadcast',
      accessToken: NL_TOKEN,
      data:        { event, data }
    }));
  }
}

// ── WebSocket connection to Neutralino server ─────────────────────────────────

function connect() {
  // Parse argv for production builds
  for (const arg of process.argv) {
    const val = arg.split('=')[1];
    if (!val || arg.includes('${NL_')) continue;

    if (arg.startsWith('--nl-port=')) {
      NL_PORT = val;
    } else if (arg.startsWith('--nl-token=')) {
      NL_TOKEN = val;
    } else if (arg.startsWith('--nl-extension-id=')) {
      NL_EXTID = val;
    } else if (arg.startsWith('--nl-connect-token=')) {
      NL_CONNECTTOKEN = val;
    }
  }

  if (!NL_PORT || !NL_TOKEN || !NL_EXTID || !NL_CONNECTTOKEN) {
    // Dev mode: not running inside Neutralino — skip connection, log to stdout
    console.log('[trainer-ext] Not running inside Neutralino. Dev mode: events logged to stdout.');
    return;
  }

  ws = new WebSocket(`ws://localhost:${NL_PORT}?extensionId=${NL_EXTID}&connectToken=${NL_CONNECTTOKEN}`);

  ws.on('open', () => {
    sendEvent('trainer.ready', { status: 'extension connected' });
  });

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    if (!msg.event) return;

    if (msg.event === 'app.broadcast') {
      const payload = typeof msg.data === 'string' ? JSON.parse(msg.data) : msg.data;
      if (payload.event === 'trainer.start') {
        startTraining(payload.data ?? {});
      } else if (payload.event === 'trainer.eval') {
        startEval(payload.data ?? {});
      } else if (payload.event === 'trainer.stop') {
        stopTraining();
      }
    } else if (msg.event === 'trainer.start') {
      startTraining(msg.data ?? {});
    } else if (msg.event === 'trainer.eval') {
      startEval(msg.data ?? {});
    } else if (msg.event === 'trainer.stop') {
      stopTraining();
    }
  });

  ws.on('error', (err) => {
    console.error('[trainer-ext] WebSocket error:', err.message);
  });

  ws.on('close', () => {
    // Reconnect after 1 s on unexpected drop
    setTimeout(connect, 1000);
  });
}

// ── Training process management ───────────────────────────────────────────────

function startTraining(opts) {
  if (trainingProcess) {
    sendEvent('trainer.error', { msg: 'Training already running' });
    return;
  }

  // Determine CLI sub-command
  const mode = opts._mode ?? 'train';

  // Build args: node index.js <mode> [--flag value ...]
  const args = ['index.js', mode];
  if (mode === 'train') {
    if (opts.model)    args.push('--model',    String(opts.model));
    if (opts.rank)     args.push('--rank',     String(opts.rank));
    if (opts.steps)    args.push('--steps',    String(opts.steps));
    if (opts.corpus)   args.push('--corpus',   String(opts.corpus));
    if (opts.resume)   args.push('--resume',   String(opts.resume));
    if (opts.mode)     args.push('--mode',     String(opts.mode));
    if (opts.schedule) args.push('--schedule', String(opts.schedule));
    if (opts.userName) args.push('--user-name', String(opts.userName));
    if (opts.aiName)   args.push('--ai-name',   String(opts.aiName));
    if (opts.fristonAlpha !== undefined) args.push('--friston-alpha', String(opts.fristonAlpha));
    if (opts.backend)  args.push('--backend',   String(opts.backend));
  } else if (mode === 'export') {
    if (opts.checkpoint) args.push('--checkpoint', String(opts.checkpoint));
    if (opts.output)     args.push('--output',     String(opts.output));
    if (opts.format)     args.push('--format',     String(opts.format));
  }

  trainingProcess = spawn('node', args, {
    cwd: PROJECT_ROOT,
    env: { ...process.env }
  });

  sendEvent('trainer.started', { args, cwd: PROJECT_ROOT });

  trainingProcess.stdout.on('data', (data) => {
    const lines = data.toString().split('\n');
    for (const line of lines) {
      if (line.trim()) sendEvent('trainer.log', { line });
    }
  });

  trainingProcess.stderr.on('data', (data) => {
    const lines = data.toString().split('\n');
    for (const line of lines) {
      if (line.trim()) sendEvent('trainer.log', { line, level: 'error' });
    }
  });

  trainingProcess.on('close', (code) => {
    sendEvent('trainer.done', { exitCode: code ?? 0 });
    trainingProcess = null;
  });

  trainingProcess.on('error', (err) => {
    sendEvent('trainer.error', { msg: `Failed to spawn node: ${err.message}` });
    trainingProcess = null;
  });
}

function startEval(opts) {
  if (trainingProcess) {
    sendEvent('trainer.error', { msg: 'A process is already running' });
    return;
  }

  const args = ['index.js', 'eval'];
  if (opts.checkpoint) args.push('--checkpoint', String(opts.checkpoint));
  if (opts.prompt)     args.push('--prompt',     String(opts.prompt));
  if (opts.maxTokens)  args.push('--max-tokens', String(opts.maxTokens));
  if (opts.temperature)args.push('--temperature',String(opts.temperature));

  trainingProcess = spawn('node', args, {
    cwd: PROJECT_ROOT,
    env: { ...process.env }
  });

  sendEvent('trainer.started', { args, cwd: PROJECT_ROOT, type: 'eval' });

  trainingProcess.stdout.on('data', (data) => {
    // Stream raw tokens as they arrive for live generation
    sendEvent('trainer.stream', { text: data.toString() });
  });

  trainingProcess.stderr.on('data', (data) => {
    const lines = data.toString().split('\n');
    for (const line of lines) {
      if (line.trim()) sendEvent('trainer.log', { line, level: 'error' });
    }
  });

  trainingProcess.on('close', (code) => {
    sendEvent('trainer.done', { exitCode: code ?? 0, type: 'eval' });
    trainingProcess = null;
  });

  trainingProcess.on('error', (err) => {
    sendEvent('trainer.error', { msg: `Failed to spawn eval node: ${err.message}` });
    trainingProcess = null;
  });
}

function stopTraining() {
  if (trainingProcess) {
    trainingProcess.kill('SIGTERM');
    trainingProcess = null;
    sendEvent('trainer.stopped', {});
  } else {
    sendEvent('trainer.error', { msg: 'No training process is running' });
  }
}

// ── Entry point ───────────────────────────────────────────────────────────────

(async () => {
  // Wait for Neutralino to write auth_info.json in Dev Mode (up to 5 seconds)
  if (!NL_PORT || !NL_CONNECTTOKEN) {
    const authInfoPath = path.resolve(PROJECT_ROOT, '.tmp', 'auth_info.json');
    let retries = 50;
    while (retries > 0) {
      if (fs.existsSync(authInfoPath)) {
        try {
          const content = fs.readFileSync(authInfoPath, 'utf8');
          const authData = JSON.parse(content);
          if (authData.nlPort && authData.nlConnectToken) {
            NL_PORT = authData.nlPort;
            NL_TOKEN = authData.nlToken;
            NL_CONNECTTOKEN = authData.nlConnectToken;
            break;
          }
        } catch (e) {
          // File might be partially written, ignore and retry
        }
      }
      await new Promise(r => setTimeout(r, 100));
      retries--;
    }
  }

  connect();
})();
