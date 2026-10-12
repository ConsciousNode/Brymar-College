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
    this.totalSteps   = parseInt(opts.steps)        ?? TRAINING.total_steps;
    this.rank         = parseInt(opts.rank)         ?? MODEL.lora_rank;
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
    const log = (msg, level = 'info') =>
      t ? t.log(msg, level) : console.log(`[${level.toUpperCase()}] ${msg}`);

    const isFullMode = this.trainingMode === 'full';

    log(`OOMB Training Loop starting (${isFullMode ? 'FULL WEIGHT' : 'LoRA'} mode)`, 'ok');
    log(`  Schedule: ${this.schedule} | Steps: ${this.totalSteps} | Base LR: ${this.baseLR}`, 'info');
    if (isFullMode) {
      log(`  Mode: Full-weight training — ALL parameters are trainable`, 'info');
    } else {
      log(`  LoRA rank: ${this.rank} | Chunk size: ${TRAINING.chunk_size} | Grad accum: ${TRAINING.grad_accum_steps}`, 'info');
    }

    // ── Dynamic imports (avoid circular deps) ───────────────────────────────
    const { RWKVModel }     = await import('../model/model.js');
    const { loadModel }     = await import('../model/loader.js');
    const { injectLoRA }    = await import('../model/lora.js');
    const { MuonOptimizer } = await import('./muon.js');
    const { OOMBTrainer }   = await import('./oomb.js');
    const { saveCheckpoint, loadCheckpoint } = await import('../persistence/checkpoint.js');
    const { CorpusLoader }  = await import('../tokenizer/corpus.js').catch(() => ({
      CorpusLoader: null
    }));

    // ── Build model ────────────────────────────────────────────────────────
    const modelCfg = {
      modelName: this.opts.model ?? MODEL.base,
      loraRank: this.rank,
      loraAlpha: MODEL.lora_alpha,
      trainingMode: this.trainingMode
    };
    
    // loadModel automatically handles downloading, Safetensors parsing,
    // auto-detecting dimensions (vocab, dim, layers), and applying weights
    const model = await loadModel(modelCfg, t);
    
    // Use the auto-detected dimensions from the loaded model
    const actualVocabSize = model.vocabSize;

    // ── Resume from checkpoint if requested ───────────────────────────────
    let startStep = 0;
    if (this.opts.resume) {
      log(`Resuming from ${this.opts.resume}`, 'info');
      try {
        startStep = await loadCheckpoint(this.opts.resume, model) ?? 0;
        log(`Resumed at step ${startStep}`, 'ok');
      } catch (e) {
        log(`Could not load checkpoint: ${e.message} — starting fresh`, 'warn');
      }
    }

    // ── Corpus ─────────────────────────────────────────────────────────────
    let tokens;
    if (CorpusLoader && this.opts.corpus) {
      try {
        const loader = new CorpusLoader(this.opts.corpus, t, this.opts);
        tokens = await loader.load({ vocabSize: actualVocabSize });
        log(`Corpus loaded: ${tokens.length} tokens from ${this.opts.corpus}`, 'ok');
      } catch (e) {
        log(`Corpus load failed (${e.message}) — using synthetic tokens`, 'warn');
      }
    }

    if (!tokens) {
      // Synthetic byte-level corpus for smoke testing
      tokens = this._makeSyntheticTokens(actualVocabSize, 1024);
      log('Using synthetic 1024-token corpus for smoke test', 'warn');
    }

    // ── Optimizer ─────────────────────────────────────────────────────────
    const trainableParams = isFullMode
      ? model.getAllParams()
      : this._extractLoRAParams(model);

    log(`Trainable parameters: ${trainableParams.length} matrices`, 'info');

    const optimizer = new MuonOptimizer(
      this.baseLR,
      TRAINING.muon_momentum,
      TRAINING.muon_wd,
      TRAINING.grad_accum_steps
    );

    // ── Trainer ───────────────────────────────────────────────────────────
    const trainer = new OOMBTrainer(model, optimizer, {
      chunkSize: TRAINING.chunk_size,
      trainingMode: this.trainingMode,
      fristonAlpha: this.opts.fristonAlpha,
    });
    trainer.step = startStep;

    // ── Progress bar ──────────────────────────────────────────────────────
    if (t) t.startProgress(this.totalSteps, 'LoRA fine-tuning');

    const tStart        = Date.now();
    let accumLoss       = 0;
    let microStep       = 0;
    let globalStep      = startStep;
    const seqLen        = TRAINING.chunk_size + 1;
    const corpusLen     = tokens.length;

    // ── Main loop ─────────────────────────────────────────────────────────
    while (globalStep < this.totalSteps && !this._stopRequested) {
      // Sample a random window from corpus
      const maxStart = Math.max(0, corpusLen - seqLen - 1);
      const start    = Math.floor(Math.random() * maxStart);
      const sequence = tokens.slice(start, start + seqLen);

      try {
        const loss = await trainer.trainSequence(sequence);
        accumLoss += loss;
        microStep++;
        if (t && !t._bar) {
          log(`  [micro] chunk ${microStep}/${TRAINING.grad_accum_steps} — loss ${loss.toFixed(4)}`, 'info');
        }
      } catch (e) {
        log(`Sequence training error: ${e.message}`, 'error');
        break;
      }

      // Optimizer step every grad_accum_steps micro-steps
      if (microStep >= TRAINING.grad_accum_steps) {
        const lr = this._computeLR(globalStep);
        optimizer.lr = lr;

        try {
          await optimizer.step(trainableParams);
          if (isFullMode) {
            model.zeroGradAll();
          } else {
            this._zeroLoRAGrads(trainableParams);
          }
        } catch (e) {
          log(`Optimizer step error: ${e.message}`, 'error');
          break;
        }

        globalStep++;
        const avgLoss = accumLoss / TRAINING.grad_accum_steps;
        const elapsed = (Date.now() - tStart) / 1000;
        const tokPerSec = (globalStep * seqLen) / Math.max(elapsed, 1e-6);

        this.metrics.record(globalStep, avgLoss, tokPerSec);

        if (t) {
          t.updateProgress(globalStep, avgLoss, tokPerSec);
        } else {
          if (globalStep % 20 === 0) {
            console.log(
              `step=${globalStep}/${this.totalSteps} loss=${avgLoss.toFixed(4)} lr=${lr.toExponential(3)} ${tokPerSec.toFixed(0)} tok/s`
            );
          }
        }

        // Auto-checkpoint
        if (globalStep % TRAINING.checkpoint_every === 0) {
          const ckptPath = await saveCheckpoint(model, globalStep, avgLoss, PATHS.checkpoints, this.trainingMode);
          log(`Checkpoint saved → ${ckptPath}`, 'ok');
        }

        accumLoss  = 0;
        microStep  = 0;
      }
    }

    if (t) t.stopProgress();

    // Final checkpoint
    const finalLoss = this.metrics.getAverage(20);
    const finalCkpt = await saveCheckpoint(model, globalStep, finalLoss, PATHS.checkpoints, this.trainingMode);
    log(`Training complete. Final checkpoint → ${finalCkpt}`, 'ok');
    log(`Total steps: ${globalStep} | Avg loss (last 20): ${finalLoss.toFixed(4)}`, 'ok');

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
   * Extract LoRA adapter weight/grad/mom param objects for the Muon optimizer.
   * Each entry matches the interface expected by MuonOptimizer.step().
   *
   * @param {import('../model/model.js').RWKVModel} model
   * @returns {Array<{ weight: Float32Array, grad: Float32Array, mom: Float32Array, rows: number, cols: number }>}
   */
  _extractLoRAParams(model) {
    const params = [];
    const adapters = model.getLoRALayers();
    for (const adapter of adapters) {
      // LoRA A matrix
      if (adapter.A) {
        if (!adapter.A_mom) adapter.A_mom = new Float32Array(adapter.A.length);
        params.push({
          weight: adapter.A,
          grad:   adapter.gradA ?? (adapter.gradA = new Float32Array(adapter.A.length)),
          mom:    adapter.A_mom,
          rows:   adapter.rank,
          cols:   adapter.inDim
        });
      }
      // LoRA B matrix
      if (adapter.B) {
        if (!adapter.B_mom) adapter.B_mom = new Float32Array(adapter.B.length);
        params.push({
          weight: adapter.B,
          grad:   adapter.gradB ?? (adapter.gradB = new Float32Array(adapter.B.length)),
          mom:    adapter.B_mom,
          rows:   adapter.outDim,
          cols:   adapter.rank
        });
      }
    }
    return params;
  }

  /**
   * Zero out gradients on all LoRA param objects.
   *
   * @param {Array<{ grad: Float32Array }>} params
   */
  _zeroLoRAGrads(params) {
    for (const p of params) p.grad.fill(0);
  }

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
