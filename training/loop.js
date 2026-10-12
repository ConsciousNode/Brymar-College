// training/loop.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// TrainingLoop orchestrator — LoRA + Full-weight training

/**
 * TrainingLoop wires the corpus loader, tokenizer, model, OOMB trainer,
 * Muon optimizer, metrics, and checkpointing into a single run() call.
 *
 * Training modes:
 *   lora — freeze base model, train only LoRA A/B adapter matrices (default)
 *   full — unfreeze everything, train all model parameters end-to-end
 *
 * LR schedule options (mirror the HTMLNLM.html UI):
 *   manual  — constant LR throughout
 *   cosine  — linear warmup then cosine decay to minLR
 *   step    — multiply by decay every <stepSize> steps
 *   exp     — LR * decay^step
 */

import { TRAINING, MODEL, PATHS } from '../src/config.js';
import { Metrics } from '../src/ui/metrics.js';

export class TrainingLoop {
  /**
   * @param {Object} opts
   * @param {string}  [opts.model]        - Base model name (HuggingFace repo ID)
   * @param {string}  [opts.rank]         - LoRA rank override (string from CLI)
   * @param {string}  [opts.steps]        - Total training steps override
   * @param {string}  [opts.corpus]       - Path to corpus directory
   * @param {string}  [opts.resume]       - Checkpoint path to resume from
   * @param {string}  [opts.mode]         - 'lora' | 'full' training mode
   * @param {string}  [opts.schedule]     - 'manual'|'cosine'|'step'|'exp'
   * @param {number}  [opts.warmupSteps]  - LR warmup steps (cosine schedule)
   * @param {number}  [opts.minLr]        - Minimum LR floor
   * @param {number}  [opts.decayFactor]  - Step/exp decay factor
   * @param {number}  [opts.stepSize]     - Step-decay interval
   * @param {import('../src/ui/terminal.js').Terminal} [opts.terminal]
   */
  constructor(opts = {}) {
    this.opts     = opts;
    this.terminal = opts.terminal ?? null;

    // Parse numeric CLI strings
    this.totalSteps   = parseInt(opts.steps) || TRAINING.total_steps;   // (`?? ` never caught parseInt's NaN)
    this.rank         = parseInt(opts.rank) || MODEL.lora_rank;
    this.trainingMode = opts.mode                   ?? TRAINING.mode;
    this.schedule     = opts.schedule               ?? 'cosine';
    this.warmupSteps  = opts.warmupSteps            ?? TRAINING.warmup_steps;
    this.minLr        = opts.minLr                  ?? TRAINING.min_lr;
    this.decayFactor  = opts.decayFactor            ?? 0.5;
    this.stepSize     = opts.stepSize               ?? 500;

    this.baseLR       = TRAINING.lr;
    this.metrics      = new Metrics();
    this._stopRequested = false;
  }

  // ── Public entry point ─────────────────────────────────────────────────────

  /**
   * Request graceful shutdown.
   */
  stop() {
    this._stopRequested = true;
    if (this.terminal) {
      this.terminal.log('Stop requested, shutting down gracefully...', 'warn');
    } else {
      console.log('[WARN] Stop requested, shutting down gracefully...');
    }
  }

  /**
   * Execute the full training loop.
   *
   * Downloads/loads weights via Safetensors, ingests the corpus directory,
   * and runs OOMB chunk-recurrent training with the configured optimizer.
   */
  async run() {
    const t = this.terminal;
    const log = (msg, level = 'info') => t ? t.log(msg, level) : console.log(`[${level.toUpperCase()}] ${msg}`);
    const isFullMode = this.trainingMode === 'full';

    const { RWKV7 }             = await import('../model/rwkv7.js');
    const { loadRWKV7From }     = await import('../model/rwkv7_load.js');
    const { RWKVWorldTokenizer } = await import('../tokenizer/rwkv_world.js');
    const { AdamW }             = await import('./adamw.js');
    const { OOMBTrainer }       = await import('./oomb.js');
    const { saveCheckpoint, loadCheckpoint } = await import('../persistence/checkpoint.js');
    const { CorpusLoader }      = await import('../tokenizer/corpus.js');

    log(`Brymar College · real RWKV-7 engine · ${isFullMode ? 'FULL-WEIGHT' : 'LoRA'} mode`, 'ok');

    // ── Base model ─────────────────────────────────────────────────────────
    const source = this.opts.model ?? MODEL.base;
    let model, baseModel = source;
    if (source === 'scratch') {
      model = new RWKV7({ vocabSize: 256, dim: 256, numLayers: 4, headSize: 64, dimFFN: 1024, dDecay: 32, dAAA: 32, dMV: 16, dGate: 64 });
      baseModel = null;
      log('From scratch: byte-level RWKV-7 (dim 256, 4 layers). For persona work, start from a pretrained model.', 'warn');
    } else {
      try { ({ model } = await loadRWKV7From(source, log, { RWKV7 })); }
      catch (e) { log(`Could not load base model "${source}": ${e.message}`, 'error'); log('Pass --model path/to/rwkv7.pth (or .safetensors), or --model scratch', 'info'); return; }
    }
    const world = model.V >= 65530;
    const tokenizer = world ? new RWKVWorldTokenizer() : null;
    if (world) log('Tokenizer: RWKV World (the base model\'s own vocabulary)', 'ok');

    if (!isFullMode) {
      model.injectLoRA(this.rank || MODEL.lora_rank, MODEL.lora_alpha);
      const nLora = model.trainable().reduce((a, p) => a + p.weight.length, 0);
      log(`LoRA rank ${this.rank || MODEL.lora_rank} on receptance/key/value/output: ${nLora.toLocaleString()} trainable parameters (base frozen)`, 'info');
    }

    let startStep = 0;
    if (this.opts.resume) {
      try { startStep = await loadCheckpoint(this.opts.resume, model) ?? 0; log(`Resumed at step ${startStep}`, 'ok'); }
      catch (e) { log(`Could not resume: ${e.message}`, 'error'); return; }
    }

    // ── Corpus ─────────────────────────────────────────────────────────────
    let tokens;
    try {
      const loader = new CorpusLoader(this.opts.corpus ?? PATHS.corpus, t, this.opts);
      tokens = await loader.load(world ? { encode: s => tokenizer.encode(s), textOnly: true } : { byteLevel: true, textOnly: true });
    } catch (e) { log(`Corpus load failed: ${e.message}`, 'error'); return; }
    if (tokens.length < TRAINING.chunk_size + 2) { log(`Corpus too small: ${tokens.length} tokens (need > ${TRAINING.chunk_size + 1})`, 'error'); return; }
    const bad = tokens.find(x => x >= model.V); if (bad !== undefined) { log(`Token id ${bad} ≥ model vocab ${model.V}: tokenizer and model disagree`, 'error'); return; }

    // ── Optimizer + trainer ────────────────────────────────────────────────
    this.baseLR = isFullMode ? (TRAINING.lr_full ?? TRAINING.lr / 10) : TRAINING.lr;
    const optimizer = new AdamW(this.baseLR);
    const trainer = new OOMBTrainer(model, { chunkSize: TRAINING.chunk_size });
    const params = model.trainable();
    log(`AdamW · base LR ${this.baseLR} · ${this.schedule} schedule · ${this.totalSteps} steps × ${TRAINING.grad_accum_steps} sequences of ${TRAINING.chunk_size} tokens`, 'info');

    if (t) t.startProgress(this.totalSteps, isFullMode ? 'Full-weight fine-tuning' : 'LoRA fine-tuning');
    const tStart = Date.now(), seqLen = TRAINING.chunk_size + 1;
    let globalStep = startStep, accumLoss = 0, microStep = 0;
    model.zeroGrad();
    while (globalStep < this.totalSteps && !this._stopRequested) {
      const start = Math.floor(Math.random() * Math.max(1, tokens.length - seqLen));
      try { accumLoss += await trainer.trainSequence(tokens.slice(start, start + seqLen)); microStep++; }
      catch (e) { log(`Sequence training error: ${e.message}`, 'error'); break; }
      if (microStep < TRAINING.grad_accum_steps) continue;
      optimizer.lr = this._computeLR(globalStep);
      const gnorm = optimizer.step(params, 1 / TRAINING.grad_accum_steps);
      model.zeroGrad();
      globalStep++;
      const avgLoss = accumLoss / TRAINING.grad_accum_steps, elapsed = (Date.now() - tStart) / 1000;
      const tokPerSec = ((globalStep - startStep) * TRAINING.grad_accum_steps * TRAINING.chunk_size) / Math.max(elapsed, 1e-6);
      this.metrics.record(globalStep, avgLoss, tokPerSec);
      if (t) t.updateProgress(globalStep, avgLoss, tokPerSec);
      else console.log(`step=${globalStep}/${this.totalSteps} loss=${avgLoss.toFixed(4)} lr=${optimizer.lr.toExponential(2)} |g|=${gnorm.toFixed(3)} ${tokPerSec.toFixed(1)} tok/s`);
      if (globalStep % TRAINING.checkpoint_every === 0) {
        const p = await saveCheckpoint(model, globalStep, avgLoss, PATHS.checkpoints, this.trainingMode, { baseModel, tokenizer: world ? 'rwkv_world' : 'bytes' });
        log(`Checkpoint saved → ${p}`, 'ok');
      }
      accumLoss = 0; microStep = 0;
    }
    if (t) t.stopProgress();
    const finalLoss = this.metrics.getAverage(20);
    const finalCkpt = await saveCheckpoint(model, globalStep, finalLoss, PATHS.checkpoints, this.trainingMode, { baseModel, tokenizer: world ? 'rwkv_world' : 'bytes' });
    log(`Training complete. Final checkpoint → ${finalCkpt}`, 'ok');
    return { globalStep, metrics: this.metrics };
  }

  // ── LR Schedule ───────────────────────────────────────────────────────────

  /**
   * Compute scheduled learning rate for a given global step.
   *
   * @param {number} step
   * @returns {number}
   */
  _computeLR(step) {
    const base   = this.baseLR;
    const minLR  = this.minLr;
    const total  = this.totalSteps;
    const warmup = this.warmupSteps;

    switch (this.schedule) {
      case 'cosine': {
        if (step < warmup) return base * (step / Math.max(1, warmup));
        const prog = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
        return minLR + 0.5 * (base - minLR) * (1 + Math.cos(Math.PI * prog));
      }
      case 'step': {
        const factor   = this.decayFactor;
        const stepSize = this.stepSize;
        return Math.max(minLR, base * Math.pow(factor, Math.floor(step / stepSize)));
      }
      case 'exp': {
        const rate = this.decayFactor;
        return Math.max(minLR, base * Math.exp(-rate * step));
      }
      default:      // 'manual'
        return base;
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /**
   * Generate a synthetic byte-level token sequence for smoke-testing.
   *
   * @param {number} vocabSize
   * @param {number} length
   * @returns {number[]}
   */
  _makeSyntheticTokens(vocabSize, length) {
    const tokens = new Array(length);
    for (let i = 0; i < length; i++) {
      tokens[i] = Math.floor(Math.random() * vocabSize);
    }
    return tokens;
  }
}
