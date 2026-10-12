// gui/app.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Brymar College GUI — Neutralino frontend logic

Neutralino.init();

// ── DOM refs ──────────────────────────────────────
const el = id => document.getElementById(id);

const modelName     = el('modelName');
const trainingMode  = el('trainingMode');
const loraRank      = el('loraRank');
const loraRankField = el('loraRankField');
const trainSteps    = el('trainSteps');
const lrSchedule    = el('lrSchedule');
const corpusPath    = el('corpusPath');
const resumePath    = el('resumePath');
const exportFormat  = el('exportFormat');
const startBtn      = el('startBtn');
const stopBtn       = el('stopBtn');
const exportBtn     = el('exportBtn');
const statusDot     = el('statusDot');
const statusText    = el('statusText');
const meters        = el('meters');
const meterLoss     = el('meterLoss');
const meterSpeed    = el('meterSpeed');
const meterStep     = el('meterStep');
const logOutput     = el('logOutput');
const clearLog      = el('clearLog');
const scrollLock    = el('scrollLock');
const helpBtn       = el('helpBtn');
const helpOverlay   = el('helpOverlay');
const helpClose     = el('helpClose');
const helpBody      = el('helpBody');
const browseCorpus  = el('browseCorpus');
const browseResume  = el('browseResume');
const progressWrap  = el('progressWrap');
const progressBar   = el('progressBar');
const progressLabel = el('progressLabel');
const modeHint      = el('modeHint');
const formatHint    = el('formatHint');
const lossCanvas    = el('lossCanvas');

// New UI components
const fileList      = el('fileList');
const refreshCorpusBtn = el('refreshCorpusBtn');
const openDirBtn  = el('openDirBtn');
const browseModel = el('browseModel');
const refreshBrowserBtn = el('refreshBrowserBtn');
const menuItems = document.querySelectorAll('.menu-item');
const actionNewProject = el('actionNewProject');
const actionOpenCorpusDir = el('actionOpenCorpusDir');
const actionOpenCheckpoint = el('actionOpenCheckpoint');
const actionOpenModel = el('actionOpenModel');
const actionExit = el('actionExit');
const actionClearLog = el('actionClearLog');
const actionPreferences = el('actionPreferences');
const actionHelp = el('actionHelp');
const actionCheckUpdates = el('actionCheckUpdates');
const actionAbout = el('actionAbout');
const actionInference = el('actionInference');

// Inference modal
const inferenceOverlay = el('inferenceOverlay');
const inferenceClose = el('inferenceClose');
const infCheckpoint = el('infCheckpoint');
const infBrowseCkpt = el('infBrowseCkpt');
const infMaxTokens = el('infMaxTokens');
const infTemperature = el('infTemperature');
const infPrompt = el('infPrompt');
const infOutput = el('infOutput');
const infStatus = el('infStatus');
const infRunBtn = el('infRunBtn');
const infCancelBtn = el('infCancelBtn');

// Preferences modal
const prefOverlay = el('prefOverlay');
const prefClose = el('prefClose');
const prefCancelBtn = el('prefCancelBtn');
const prefSaveBtn = el('prefSaveBtn');
const prefUserName = el('prefUserName');
const prefAiName = el('prefAiName');
const prefFriston = el('prefFriston');
const prefFristonVal = el('prefFristonVal');

// ── State ─────────────────────────────────────────
let running     = false;
let totalSteps  = 0;
let currentStep = 0;

// Loss history for the graph
const lossHistory = [];
const MAX_LOSS_POINTS = 200;

// ── Format descriptions ───────────────────────────
const FORMAT_HINTS = {
  rofl:     'Recurrent Omnimodal Friston Language model — full FP32 precision',
  lmao:     'Language Model, Active-inference, Omnimodal — Q4 (4-bit) quantized',
  fuctrump: 'Fristonian Underpinning Capabilities Transformer-agnostic Recurrent Unconcerned with Model Parameters — Q2 ternary',
  json:     'Legacy JSON format — backward compatible with Phase 3',
};

const MODE_HINTS = {
  lora: 'Fine-tune adapter matrices only (fast, small checkpoints)',
  full: 'Train ALL parameters end-to-end (slower, full model checkpoints)',
};

// ── Dynamic UI updates ────────────────────────────

trainingMode.addEventListener('change', () => {
  const mode = trainingMode.value;
  modeHint.textContent = MODE_HINTS[mode] || '';
  // Show/hide LoRA rank field
  loraRankField.style.display = mode === 'full' ? 'none' : 'flex';
});

exportFormat.addEventListener('change', () => {
  formatHint.textContent = FORMAT_HINTS[exportFormat.value] || '';
});

// ── Severity detection ────────────────────────────

// Match CLI terminal output tags → CSS classes
const SEVERITY_MAP = [
  { pat: /\[OK\s*\]/i,   cls: 'ok'    },
  { pat: /\[WARN\s*\]/i, cls: 'warn'  },
  { pat: /\[ERR\s*\]/i,  cls: 'error' },
  { pat: /\[MEM\s*\]/i,  cls: 'mem'   },
  { pat: /\[STEP\s*\]/i, cls: 'train' },
  { pat: /\[INFO\s*\]/i, cls: 'info'  },
  { pat: /step\s+\d+.*loss/i,           cls: 'train' },
  { pat: /^(■|▶|●)/,                    cls: 'ok'    },
];

function detectSeverity(line) {
  for (const { pat, cls } of SEVERITY_MAP) {
    if (pat.test(line)) return cls;
  }
  return '';
}

// ── Log rendering ─────────────────────────────────

const MAX_LOG_LINES = 2000;

function appendLog(line, levelHint = '') {
  // Trim log to prevent unbounded growth
  while (logOutput.children.length >= MAX_LOG_LINES) {
    logOutput.removeChild(logOutput.firstChild);
  }

  const div = document.createElement('div');
  div.className = 'log-line';

  const cls = levelHint || detectSeverity(line);
  if (cls) div.classList.add(cls);

  div.classList.add('flash-in');
  div.textContent = line;
  logOutput.appendChild(div);

  // Auto-scroll if enabled
  if (scrollLock.checked) {
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  // Parse training metrics from STEP lines
  if (cls === 'train' || /step\s+\d+.*loss/i.test(line)) {
    const lossM  = line.match(/loss[:\s]+([\d.]+)/i);
    const speedM = line.match(/([\d.]+)\s*tok\/s/i);
    const stepM  = line.match(/step[=\s]+(\d+)/i);

    if (lossM) {
      const lossVal = parseFloat(lossM[1]);
      flashMeter(meterLoss, lossVal.toFixed(4));
      recordLoss(lossVal);
    }
    if (speedM) flashMeter(meterSpeed, Math.round(parseFloat(speedM[1])).toString());
    if (stepM) {
      currentStep = parseInt(stepM[1]);
      flashMeter(meterStep, currentStep.toString());
      updateProgress();
    }
  }
}

function flashMeter(el, value) {
  el.textContent = value;
  el.classList.remove('flash');
  void el.offsetWidth; // reflow
  el.classList.add('flash');
}

// ── Loss graph ────────────────────────────────────

function recordLoss(value) {
  lossHistory.push(value);
  if (lossHistory.length > MAX_LOSS_POINTS) lossHistory.shift();
  drawLossGraph();
}

function drawLossGraph() {
  const ctx = lossCanvas.getContext('2d');
  const W = lossCanvas.width;
  const H = lossCanvas.height;

  ctx.clearRect(0, 0, W, H);

  if (lossHistory.length < 2) return;

  // Find range
  let minL = Infinity, maxL = -Infinity;
  for (const v of lossHistory) {
    if (v < minL) minL = v;
    if (v > maxL) maxL = v;
  }
  const range = maxL - minL || 1;
  const pad = range * 0.1;

  // Grid lines
  ctx.strokeStyle = 'rgba(37, 37, 37, 0.8)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 4; i++) {
    const y = H * (i / 3);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  // Loss curve
  ctx.strokeStyle = '#00e5a0';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = 'rgba(0, 229, 160, 0.4)';
  ctx.shadowBlur = 4;
  ctx.beginPath();

  const n = lossHistory.length;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H - ((lossHistory[i] - minL + pad) / (range + 2 * pad)) * H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Labels
  ctx.fillStyle = 'rgba(85, 85, 70, 0.9)';
  ctx.font = '8px IBM Plex Mono, monospace';
  ctx.fillText(maxL.toFixed(2), 2, 10);
  ctx.fillText(minL.toFixed(2), 2, H - 3);
}

// ── Progress bar ──────────────────────────────────

function updateProgress() {
  if (totalSteps <= 0) return;
  const pct = Math.min(100, Math.round((currentStep / totalSteps) * 100));
  progressBar.style.setProperty('--progress', pct + '%');
  progressLabel.textContent = pct + '%';
}

// ── Training control ──────────────────────────────

function setRunning(state) {
  running = state;
  startBtn.disabled = state;
  stopBtn.disabled  = !state;

  statusDot.className  = 'status-dot' + (state ? ' running' : '');
  statusText.textContent = state ? 'Training…' : 'Ready';

  if (state) {
    meters.classList.add('visible');
    progressWrap.style.display = 'flex';
    totalSteps  = parseInt(trainSteps.value) || 0;
    currentStep = 0;
    updateProgress();
  } else {
    // Keep meters visible after run so user can read final values
    progressWrap.style.display = 'none';
  }
}

startBtn.addEventListener('click', () => {
  if (running) return;
  setRunning(true);

  const mode = trainingMode.value;
  appendLog(`▶ Starting ${mode === 'full' ? 'FULL WEIGHT' : 'LoRA'} training session…`, 'ok');

  // Clear loss history for new run
  lossHistory.length = 0;
  drawLossGraph();

  Neutralino.events.broadcast('trainer.start', {
    model:    modelName.value.trim(),
    rank:     loraRank.value,
    steps:    trainSteps.value,
    corpus:   corpusPath.value.trim(),
    resume:   resumePath.value.trim() || undefined,
    mode:     mode,
    schedule: lrSchedule.value,
    userName: prefUserName.value.trim() || 'USER',
    aiName:   prefAiName.value.trim() || 'CLAUDE',
    fristonAlpha: parseFloat(prefFriston.value) || 0.0,
    backend:  document.getElementById('computeBackend').value,
  });
});

stopBtn.addEventListener('click', () => {
  Neutralino.events.broadcast('trainer.stop', {});
  appendLog('■ Stop requested…', 'warn');
});

// ── Extension events ──────────────────────────────

Neutralino.events.on('trainer.log', e => {
  const lines = (e.detail.line || '').split('\n');
  const hint  = e.detail.level === 'error' ? 'error' : '';
  for (const l of lines) {
    if (l.trim()) appendLog(l, hint);
  }
});

Neutralino.events.on('trainer.done', e => {
  const code = e.detail.exitCode ?? 0;
  if (e.detail.type === 'eval') {
    inferenceRunning = false;
    infRunBtn.style.display = 'inline-block';
    infCancelBtn.style.display = 'none';
    infStatus.textContent = code === 0 ? 'Generation complete.' : `Exited with error (${code})`;
    infStatus.style.color = code === 0 ? 'var(--info)' : 'var(--error)';
    return;
  }
  setRunning(false);
  statusDot.className = 'status-dot ' + (code === 0 ? 'done' : 'error');
  appendLog(`■ Training complete (exit ${code})`, code === 0 ? 'ok' : 'error');
});

Neutralino.events.on('trainer.stopped', () => {
  if (inferenceRunning) {
    inferenceRunning = false;
    infRunBtn.style.display = 'inline-block';
    infCancelBtn.style.display = 'none';
    infStatus.textContent = 'Generation stopped.';
    return;
  }
  setRunning(false);
  statusDot.className = 'status-dot';
  appendLog('■ Process forcefully stopped', 'error');
});

Neutralino.events.on('trainer.stream', e => {
  // Raw token stream from eval
  if (e.detail.text) {
    infOutput.value += e.detail.text;
    infOutput.scrollTop = infOutput.scrollHeight;
  }
});

Neutralino.events.on('trainer.error', e => {
  appendLog(`[ERR ] ${e.detail.msg}`, 'error');
  if (inferenceRunning) {
    inferenceRunning = false;
    infRunBtn.style.display = 'inline-block';
    infCancelBtn.style.display = 'none';
    infStatus.textContent = 'Error: ' + e.detail.msg;
    infStatus.style.color = 'var(--error)';
  }
  setRunning(false);
  statusDot.className = 'status-dot error';
});

Neutralino.events.on('trainer.ready', () => {
  appendLog('[OK  ] Extension connected.', 'ok');
});

actionAbout.addEventListener('click', () => {
  appendLog('---', 'info');
  appendLog('Brymar College v3.3.3', 'ok');
  appendLog('ConsciousNode SoftWorks', 'info');
  appendLog('---', 'info');
});

// ── Preferences Logic ─────────────────────────────
actionPreferences.addEventListener('click', () => {
  prefOverlay.style.display = 'flex';
});

const closePrefModal = () => { prefOverlay.style.display = 'none'; };
prefClose.addEventListener('click', closePrefModal);
prefCancelBtn.addEventListener('click', closePrefModal);

prefFriston.addEventListener('input', () => {
  prefFristonVal.textContent = prefFriston.value;
});

prefSaveBtn.addEventListener('click', () => {
  localStorage.setItem('brymar_userName', prefUserName.value.trim());
  localStorage.setItem('brymar_aiName', prefAiName.value.trim());
  localStorage.setItem('brymar_fristonAlpha', prefFriston.value);
  appendLog('[INFO] Preferences saved.', 'ok');
  closePrefModal();
});

// Load preferences on start
if (localStorage.getItem('brymar_userName')) {
  prefUserName.value = localStorage.getItem('brymar_userName');
}
if (localStorage.getItem('brymar_aiName')) {
  prefAiName.value = localStorage.getItem('brymar_aiName');
}
if (localStorage.getItem('brymar_fristonAlpha')) {
  prefFriston.value = localStorage.getItem('brymar_fristonAlpha');
  prefFristonVal.textContent = prefFriston.value;
}

// ── New Project Logic ─────────────────────────────
actionNewProject.addEventListener('click', () => {
  if (confirm('Reset workspace for a new project? Unsaved training data will be lost.')) {
    modelName.value = '';
    trainingMode.value = 'lora';
    loraRank.value = '4';
    trainSteps.value = '2000';
    lrSchedule.value = 'cosine';
    corpusPath.value = '';
    resumePath.value = '';
    exportFormat.value = 'rofl';
    appendLog('[INFO] Workspace reset for new project.', 'info');
  }
});

Neutralino.events.on('trainer.started', e => {
  appendLog(`[INFO] Spawned: node ${e.detail.args.slice(1).join(' ')}`, 'info');
});

// ── File/folder browse (Config panel) ─────────────────
// These are handled by the unified promptOpenDirectory and promptOpenCheckpoint


// ── Export adapter ────────────────────────────────

exportBtn.addEventListener('click', async () => {
  let checkpoint = '';
  try {
    const files = await Neutralino.os.showOpenDialog('Select Checkpoint to Export', {
      filters: [{ name: 'Checkpoint JSON', extensions: ['json'] }]
    });
    if (!files || !files[0]) return;
    checkpoint = files[0];
  } catch (_) { return; }

  const fmt = exportFormat.value;
  appendLog(`[INFO] Exporting as .${fmt}: ${checkpoint}`, 'info');
  Neutralino.events.broadcast('trainer.start', {
    _mode:      'export',
    checkpoint: checkpoint,
    output:     './exports/',
    format:     fmt,
  });
});

// ── Clear log ─────────────────────────────────────

clearLog.addEventListener('click', () => { logOutput.innerHTML = ''; });

// ── Help modal ────────────────────────────────────

let helpLoaded = false;

async function openHelp() {
  if (!helpLoaded) {
    try {
      const res  = await fetch('help.html');
      helpBody.innerHTML = await res.text();
      helpLoaded = true;
    } catch (_) {
      helpBody.innerHTML = '<p>Help file not found.</p>';
    }
  }
  helpOverlay.style.display = 'flex';
}

function closeHelp() {
  helpOverlay.style.display = 'none';
}

helpBtn.addEventListener('click', openHelp);
helpClose.addEventListener('click', closeHelp);
helpOverlay.addEventListener('click', e => {
  if (e.target === helpOverlay) closeHelp();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeHelp();
});

// ── Window close handler ──────────────────────────

Neutralino.events.on('windowClose', () => {
  Neutralino.app.exit();
});

// ── Startup ───────────────────────────────────────

appendLog('■■■  BRYMAR COLLEGE  ■■■   ConsciousNode SoftWorks', 'ok');
appendLog('[INFO] GUI shell ready. Configure and press Start Training.', 'info');
appendLog('[INFO] Training modes: LoRA (adapters only) or Full Weight (all parameters)', 'info');
appendLog('[INFO] Export formats: .rofl (FP32) | .lmao (Q4) | .fuctrump (Q2)', 'info');
appendLog('[INFO] Corpus supported formats: .txt, .md, .rst, .csv, .tsv, .json, .jsonl, .xml, .html, .htm, .yaml, .yml, .toml, .log, .rtf, .tex, .org, .adoc, .png, .jpg, .jpeg, .webp, .tiff, .tif, .gif, .avif, .svg, .bmp, .wav, .mp3, .flac, .ogg, .aac, .m4a, .opus, .wma, .mp4, .mkv, .avi, .mov, .webm, .flv, .wmv, .m4v', 'dim');
appendLog('[INFO] CLI: node index.js train --help', 'dim');

// Initialize canvas size
if (lossCanvas) {
  const wrap = lossCanvas.parentElement;
  lossCanvas.width = wrap.clientWidth - 16;
  lossCanvas.height = wrap.clientHeight - 8;
}

// ── Menu Bar Logic ────────────────────────────────

// Toggle menus
menuItems.forEach(item => {
  item.addEventListener('click', (e) => {
    e.stopPropagation();
    const isActive = item.classList.contains('active');
    menuItems.forEach(m => m.classList.remove('active'));
    if (!isActive) item.classList.add('active');
  });
});

// Close menus when clicking outside
document.addEventListener('click', () => {
  menuItems.forEach(m => m.classList.remove('active'));
});

actionExit.addEventListener('click', () => Neutralino.app.exit());
actionClearLog.addEventListener('click', () => { logOutput.innerHTML = ''; });
actionHelp.addEventListener('click', openHelp);

// ── System Browser Logic ───────────────────────────

let currentBrowserPath = null;

async function renderBrowser(targetPath) {
  if (!window.Neutralino) return;
  
  if (!targetPath) {
    if (currentBrowserPath) {
      targetPath = currentBrowserPath;
    } else {
      targetPath = window.NL_CWD || '/';
    }
  }
  
  currentBrowserPath = targetPath;
  browserPath.textContent = targetPath;
  browserList.innerHTML = '<div class="file-empty">Loading...</div>';
  
  try {
    const entries = await Neutralino.filesystem.readDirectory(targetPath);
    browserList.innerHTML = '';
    
    // Parent directory link if not root
    if (targetPath !== '/' && targetPath !== 'C:\\') {
      const up = document.createElement('div');
      up.className = 'file-item';
      up.style.cursor = 'pointer';
      up.innerHTML = '<span class="file-name">📁 ..</span>';
      up.onclick = () => {
        let parent = targetPath.substring(0, targetPath.lastIndexOf('/'));
        if (!parent) parent = '/';
        renderBrowser(parent);
      };
      browserList.appendChild(up);
    }
    
    // Sort directories first, then files
    entries.sort((a, b) => {
      if (a.type === b.type) return a.entry.localeCompare(b.entry);
      return a.type === 'DIRECTORY' ? -1 : 1;
    });
    
    for (const f of entries) {
      if (f.entry === '.' || f.entry === '..') continue;
      
      const item = document.createElement('div');
      item.className = 'file-item';
      item.style.cursor = f.type === 'DIRECTORY' ? 'pointer' : 'default';
      
      const name = document.createElement('span');
      name.className = 'file-name';
      name.textContent = (f.type === 'DIRECTORY' ? '📁 ' : '📄 ') + f.entry;
      name.title = f.entry;
      item.appendChild(name);
      
      const fullPath = targetPath.endsWith('/') ? targetPath + f.entry : targetPath + '/' + f.entry;
      
      if (f.type === 'DIRECTORY') {
        item.onclick = () => renderBrowser(fullPath);
        
        const setBtn = document.createElement('button');
        setBtn.className = 'file-delete';
        setBtn.style.color = 'var(--accent)';
        setBtn.textContent = 'Set Corpus';
        setBtn.title = 'Set as Active Corpus';
        setBtn.onclick = (e) => {
          e.stopPropagation();
          corpusPath.value = fullPath;
          appendLog(`[INFO] Corpus set to ${fullPath}`, 'ok');
        };
        item.appendChild(setBtn);
      } else {
        // It's a file
        // Allow setting as Base Model if it looks like a model
        if (f.entry.endsWith('.pth') || f.entry.endsWith('.bin') || f.entry.endsWith('.safetensors') || f.entry.endsWith('.rofl') || f.entry.endsWith('.lmao') || f.entry.endsWith('.fuctrump')) {
          const setBtn = document.createElement('button');
          setBtn.className = 'file-delete';
          setBtn.style.color = 'var(--info)';
          setBtn.textContent = 'Set Model';
          setBtn.title = 'Set as Base Model';
          setBtn.onclick = (e) => {
            e.stopPropagation();
            modelName.value = fullPath;
            appendLog(`[INFO] Base model set to ${f.entry}`, 'ok');
          };
          item.appendChild(setBtn);
        } else if (f.entry.endsWith('.json')) {
          const setBtn = document.createElement('button');
          setBtn.className = 'file-delete';
          setBtn.style.color = 'var(--warn)';
          setBtn.textContent = 'Set Checkpoint';
          setBtn.title = 'Set as Resume Checkpoint';
          setBtn.onclick = (e) => {
            e.stopPropagation();
            resumePath.value = fullPath;
            appendLog(`[INFO] Checkpoint set to ${f.entry}`, 'ok');
          };
          item.appendChild(setBtn);
        }
      }
      
      browserList.appendChild(item);
    }
  } catch (err) {
    browserList.innerHTML = `<div class="file-empty" style="color:var(--error);">Error: ${err.message || err.code || 'Access denied'}</div>`;
    appendLog(`[ERR ] Browser Error: ${err.message || err.code || JSON.stringify(err)}`, 'error');
  }
}

// ── Inference Studio logic ──────────────────────────

actionInference.onclick = () => {
  inferenceOverlay.style.display = 'flex';
  if (!infCheckpoint.value && resumePath.value) {
    infCheckpoint.value = resumePath.value;
  }
};

inferenceClose.onclick = () => {
  inferenceOverlay.style.display = 'none';
};

infBrowseCkpt.onclick = async () => {
  try {
    let entries = await Neutralino.os.showOpenDialog('Select Checkpoint', {
      filters: [{ name: 'Checkpoints', extensions: ['rofl', 'lmao', 'fuctrump', 'json'] }]
    });
    if (entries.length > 0) infCheckpoint.value = entries[0];
  } catch (err) {
    appendLog('Browse cancelled', 'warn');
  }
};

let inferenceRunning = false;

infRunBtn.onclick = () => {
  if (inferenceRunning) return;
  const ckpt = infCheckpoint.value.trim();
  if (!ckpt) {
    infStatus.textContent = 'Error: No checkpoint selected';
    infStatus.style.color = 'var(--error)';
    return;
  }
  
  inferenceRunning = true;
  infRunBtn.style.display = 'none';
  infCancelBtn.style.display = 'inline-block';
  infOutput.value = '';
  infStatus.textContent = 'Generating...';
  infStatus.style.color = 'var(--text-hi)';

  Neutralino.events.broadcast('trainer.eval', {
    checkpoint: ckpt,
    prompt: infPrompt.value,
    maxTokens: parseInt(infMaxTokens.value),
    temperature: parseFloat(infTemperature.value)
  });
};

infCancelBtn.onclick = () => {
  Neutralino.events.broadcast('trainer.stop', {});
};

// ── Dropdowns & Globals ──────────────────────────────

async function promptOpenDirectory() {
  if (!window.Neutralino) return;
  try {
    const dir = await Neutralino.os.showFolderDialog('Select Corpus Directory');
    if (dir) corpusPath.value = dir;
  } catch (err) {
    appendLog(`[ERR ] Folder Dialog: ${err.message || err.code || err}`, 'error');
  }
}

async function promptOpenCheckpoint() {
  if (!window.Neutralino) return;
  try {
    const files = await Neutralino.os.showOpenDialog('Select Checkpoint', {
      filters: [{ name: 'Checkpoint JSON', extensions: ['json'] }]
    });
    if (files && files[0]) resumePath.value = files[0];
  } catch (err) {
    appendLog(`[ERR ] Open Dialog: ${err.message || err.code || err}`, 'error');
  }
}

async function promptOpenModel() {
  if (!window.Neutralino) return;
  try {
    const files = await Neutralino.os.showOpenDialog('Select Base Model', {
      filters: [{ name: 'Model Files', extensions: ['pth', 'bin', 'safetensors', 'rofl', 'lmao'] }]
    });
    if (files && files[0]) modelName.value = files[0];
  } catch (err) {
    appendLog(`[ERR ] Open Dialog: ${err.message || err.code || err}`, 'error');
  }
}

// ── Event Listeners ───────────────────────────────

actionCheckUpdates.addEventListener('click', () => {
  appendLog('[INFO] Brymar College is up to date (v3.3.3)', 'ok');
});

actionOpenCorpusDir.addEventListener('click', promptOpenDirectory);
actionOpenCheckpoint.addEventListener('click', promptOpenCheckpoint);
actionOpenModel.addEventListener('click', promptOpenModel);

refreshBrowserBtn.addEventListener('click', () => renderBrowser(currentBrowserPath));
browseCorpus.addEventListener('click', promptOpenDirectory);
browseResume.addEventListener('click', promptOpenCheckpoint);
browseModel.addEventListener('click', promptOpenModel);

// Initial load
Neutralino.events.on('ready', () => {
  setTimeout(() => renderBrowser(null), 500);
});
