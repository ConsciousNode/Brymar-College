// training/oomb.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// OOMB chunk-recurrent BPTT trainer for RWKV-v7 LoRA and full-weight fine-tuning
//
// OOMB = Out-Of-Memory Budget
// Splits the input sequence into fixed-size chunks, saves a recurrent state
// checkpoint at every chunk boundary, then replays each chunk forward/backward
// to compute full BPTT gradients without ever materialising the entire sequence
// activation history at once.
//
// Reference: HTMLNLM.html — Trainer class (Kham / Kehai Interim / Ed Interim)

import { TRAINING } from '../src/config.js';

/**
 * OOMBTrainer: Chunk-recurrent BPTT for RWKV-v7 fine-tuning.
 *
 * Supports both LoRA-only and full-weight training modes via a configurable
 * backward function. Memory budget is controlled by TRAINING.chunk_size and
 * TRAINING.grad_accum_steps from src/config.js.
 */
export class OOMBTrainer {
  /**
   * @param {import('../model/model.js').RWKVModel} model
   * @param {import('./muon.js').MuonOptimizer} optimizer
   * @param {Object} [opts]
   * @param {number} [opts.chunkSize]     - Tokens per BPTT chunk (default TRAINING.chunk_size)
   * @param {string} [opts.trainingMode]  - 'lora' | 'full' (default 'lora')
   */
  constructor(model, optimizer, opts = {}) {
    this.model   = model;
    this.opt     = optimizer;
    this.step    = 0;
    this.chunkSize = opts.chunkSize ?? TRAINING.chunk_size;
    this.trainingMode = opts.trainingMode ?? 'lora';
    this.fristonAlpha = opts.fristonAlpha ?? 0.0;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Train on a single token sequence using chunk-recurrent BPTT.
   *
   * Gradients are accumulated into LoRA adapter weights (lora mode) or
   * all model parameters (full mode).
   * The caller is responsible for calling optimizer.step() every
   * TRAINING.grad_accum_steps invocations and then zeroGrad().
   *
   * @param {number[]} tokens - Token ID sequence (length ≥ 2)
   * @returns {Promise<number>} Average cross-entropy loss for this sequence
   */
  async trainSequence(tokens) {
    const len = tokens.length;
    if (len < 2) return 0;

    const chunkSize = this.chunkSize;

    // ── Phase 1: Forward scan — collect one state checkpoint per chunk ──────
    let state = this.model.getEmptyState();
    const checkpoints = [];    // [{ start, state }]

    for (let start = 0; start < len - 1; start += chunkSize) {
      // Snapshot state before chunk begins
      const stateCopy = this._cloneState(state);
      checkpoints.push({ start, state: stateCopy });

      const end = Math.min(start + chunkSize, len - 1);
      for (let t = start; t < end; t++) {
        const res = this.model.forward(tokens[t], state);
        state     = res.state;
      }

      // Yield to Node.js event loop between chunks
      await new Promise(resolve => setImmediate(resolve));
    }

    // ── Phase 2: Backward scan — replay each chunk and backprop ─────────────
    state = null;   // release forward-scan state
    let totalLoss = 0;

    // stateGrads propagated backward across chunk boundaries
    let stateGrads = this.model.blocks.map(() => ({
      num: new Float32Array(this.model.dim),
      den: new Float32Array(this.model.dim)
    }));

    // Select backward function based on training mode
    const backwardFn = this.trainingMode === 'full'
      ? this.model.backwardFull.bind(this.model)
      : this.model.backwardLoRAOnly.bind(this.model);

    for (let ci = checkpoints.length - 1; ci >= 0; ci--) {
      const { start, state: replayState } = checkpoints[ci];
      const end = Math.min(start + chunkSize, len - 1);

      // Re-run forward pass for this chunk, collecting activations
      const acts = [];
      let currentState = replayState;

      for (let t = start; t < end; t++) {
        const res = this.model.forward(tokens[t], currentState);
        acts.push({
          logits:    res.logits,
          caches:    res.caches,
          loraCaches: res.loraCaches,
          tokenId:   tokens[t]
        });
        currentState = res.state;
      }

      // Backward through timesteps in reverse (BPTT within chunk)
      for (let j = acts.length - 1; j >= 0; j--) {
        const targetId = tokens[start + j + 1];
        const { loss, grad } = this._lossAndGrad(acts[j].logits, targetId);
        totalLoss += loss;

        // Accumulate gradients (LoRA-only or full)
        const result = backwardFn(
          grad,
          acts[j].caches,
          acts[j].loraCaches,
          stateGrads
        );
        stateGrads = result.gradStatesOut;
      }

      await new Promise(resolve => setImmediate(resolve));
    }

    this.step++;
    return totalLoss / Math.max(1, len - 1);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * Compute softmax cross-entropy loss and gradient w.r.t. logits.
   *
   * @param {Float32Array} logits   - Raw logits [vocabSize]
   * @param {number}       targetId - Target token index
   * @returns {{ loss: number, grad: Float32Array }}
   */
  _lossAndGrad(logits, targetId) {
    const V = logits.length;

    // Numerically stable softmax
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];

    let sum = 0;
    const probs = new Float32Array(V);
    for (let i = 0; i < V; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    
    let entropy = 0;
    for (let i = 0; i < V; i++) {
      probs[i] /= sum;
      if (probs[i] > 1e-8) {
        entropy -= probs[i] * Math.log(probs[i]);
      }
    }

    // Standard Cross-Entropy
    let loss = -Math.log(probs[targetId] + 1e-8);

    // Fristonian Expected Surprise (Entropy Penalty)
    // We minimize Free Energy by minimizing uncertainty (entropy)
    if (this.fristonAlpha > 0) {
      loss += this.fristonAlpha * entropy;
    }

    // Gradient of softmax cross-entropy: p - one_hot(target)
    for (let i = 0; i < V; i++) {
      let grad_i = probs[i];
      if (i === targetId) grad_i -= 1.0;
      
      // Inject Friston gradient if enabled
      if (this.fristonAlpha > 0) {
        // dH/dz_i = -p_i * (log(p_i) - sum(p_j log(p_j)))
        // sum(p_j log(p_j)) = -H
        // dH/dz_i = -p_i * (log(p_i) + H)
        let log_p = probs[i] > 1e-8 ? Math.log(probs[i]) : -18.42;
        let fristonGrad = -probs[i] * (log_p + entropy);
        grad_i += this.fristonAlpha * fristonGrad;
      }
      
      probs[i] = grad_i;
    }

    return { loss, grad: probs };
  }

  /**
   * Deep-clone the per-block recurrent state.
   *
   * Each block state = { prevX, prevX2, num, den } (all Float32Array | null).
   *
   * @param {Array<Object>} state
   * @returns {Array<Object>}
   */
  _cloneState(state) {
    return state.map(s => ({
      prevX:  s.prevX  ? new Float32Array(s.prevX)  : null,
      prevX2: s.prevX2 ? new Float32Array(s.prevX2) : null,
      num:    s.num    ? new Float32Array(s.num)    : null,
      den:    s.den    ? new Float32Array(s.den)    : null
    }));
  }
}
