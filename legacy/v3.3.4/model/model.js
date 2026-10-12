// model/model.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// RWKV-v7 model assembly with LoRA and full-weight training support

import { RWKVv7Block } from './rwkv7block.js';
import { LayerNorm } from './layernorm.js';
import { BitLinear } from './bitlinear.js';

/**
 * RWKVModel: Full RWKV-v7 model with embedding, blocks, and head
 * 
 * Architecture:
 * - Token embedding (learned)
 * - Pre-block LayerNorm
 * - N × RWKVv7Block
 * - Post-block LayerNorm
 * - Output head (BitLinear)
 *
 * Training modes:
 * - LoRA-only: freeze base weights, train only LoRA A/B adapter matrices
 * - Full: train all parameters end-to-end (embeddings, LN, all BitLinear, head)
 */
export class RWKVModel {
  /**
   * @param {Object} cfg - Model configuration
   * @param {number} cfg.vocabSize - Vocabulary size
   * @param {number} cfg.dim - Model dimension
   * @param {number} cfg.numLayers - Number of RWKV blocks
   * @param {number} cfg.dimFFN - FFN hidden dimension
   */
  constructor(cfg) {
    this.vocabSize = cfg.vocabSize;
    this.dim = cfg.dim;
    this.numLayers = cfg.numLayers;
    this.dimFFN = cfg.dimFFN;
    
    // Token embedding: [vocabSize × dim]
    this.embedding = new Float32Array(cfg.vocabSize * cfg.dim);
    this.embeddingGrad = new Float32Array(cfg.vocabSize * cfg.dim);
    this.embeddingMom = new Float32Array(cfg.vocabSize * cfg.dim);
    const embScale = Math.sqrt(1.0 / cfg.dim);
    for (let i = 0; i < this.embedding.length; i++) {
      this.embedding[i] = (Math.random() * 2 - 1) * embScale;
    }
    
    // Pre-block LayerNorm
    this.ln_in = new LayerNorm(cfg.dim);
    
    // RWKV-v7 blocks
    this.blocks = [];
    for (let i = 0; i < cfg.numLayers; i++) {
      this.blocks.push(new RWKVv7Block(cfg.dim, cfg.dimFFN));
    }
    
    // Post-block LayerNorm
    this.ln_out = new LayerNorm(cfg.dim);
    
    // Output head
    this.head = new BitLinear(cfg.dim, cfg.vocabSize);
  }

  /**
   * Get empty initial state for all blocks.
   * 
   * @returns {Array<Object>} Array of empty states [numLayers]
   */
  getEmptyState() {
    const states = [];
    for (let i = 0; i < this.numLayers; i++) {
      states.push({
        prevX: null,
        prevX2: null,
        num: null,
        den: null
      });
    }
    return states;
  }

  /**
   * Forward pass through the model.
   * 
   * @param {number} tokenId - Input token ID
   * @param {Array<Object>} state - Recurrent state for each block
   * @returns {{ logits: Float32Array, state: Array<Object>, caches: Object, loraCaches: Array<Object> }}
   */
  forward(tokenId, state) {
    // Embed token
    const x = new Float32Array(this.dim);
    const embOffset = tokenId * this.dim;
    for (let i = 0; i < this.dim; i++) {
      x[i] = this.embedding[embOffset + i];
    }
    
    // Pre-block LayerNorm
    const lnInRes = this.ln_in.forward(x);
    let h = lnInRes.out;
    
    // Forward through blocks
    const newStates = [];
    const caches = [];
    const loraCaches = [];
    
    for (let i = 0; i < this.numLayers; i++) {
      const blockRes = this.blocks[i].forward(h, state[i]);
      h = blockRes.out;
      newStates.push(blockRes.state);
      caches.push(blockRes.cache);
      
      // Collect LoRA caches if adapters are present
      const blockLoraCaches = {};
      const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];
      for (const name of targetNames) {
        const layer = this.blocks[i][name];
        if (layer && layer.loraAdapter) {
          // LoRA cache is stored in the layer's last forward call
          // We need to extract it from the cache
          blockLoraCaches[name] = layer.loraCache || null;
        }
      }
      loraCaches.push(blockLoraCaches);
    }
    
    // Post-block LayerNorm
    const lnOutRes = this.ln_out.forward(h);
    
    // Output head
    const logits = this.head.forward(lnOutRes.out);
    
    return {
      logits,
      state: newStates,
      caches: { lnInRes, blockCaches: caches, lnOutRes, tokenId },
      loraCaches
    };
  }

  /**
   * Backward pass for LoRA-only training.
   *
   * Propagates gradients through the frozen base model (accumulating no base-weight
   * gradients), and accumulates gradients only for LoRA A/B adapter matrices.
   * Threading BPTT state gradients (num/den) across chunk boundaries enables the
   * OOMB chunk-recurrent training scheme.
   *
   * @param {Float32Array} gradLogits  - Gradient w.r.t. logits [vocabSize]
   * @param {Object}       caches      - Cached activations from forward()
   * @param {Array}        loraCaches  - Per-block LoRA caches (unused — adapters cache internally)
   * @param {Array<Object>} stateGrads - Incoming BPTT state grads [numLayers] × { num, den }
   * @returns {{ gradStatesOut: Array<Object> }}
   */
  backwardLoRAOnly(gradLogits, caches, loraCaches, stateGrads) {
    const { lnInRes, blockCaches, lnOutRes } = caches;

    // ── Head backward (frozen, no weight grad) ──────────────────────────────
    let gradH = this.head.backward(gradLogits, lnOutRes.out);

    // ── Post-block LN backward ───────────────────────────────────────────────
    gradH = this.ln_out.backward(gradH, lnOutRes.cache);

    // ── Block backward (reverse order for BPTT) ─────────────────────────────
    const gradStatesOut = new Array(this.numLayers).fill(null).map(() => ({
      num: new Float32Array(this.dim),
      den: new Float32Array(this.dim)
    }));

    const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];

    for (let i = this.numLayers - 1; i >= 0; i--) {
      const block    = this.blocks[i];
      const cache    = blockCaches[i];   // { x, x_ln, ln1Res, prevX, ..., h_ln, ln2Res, ... }
      const stateGrad = stateGrads ? stateGrads[i] : null;

      // ── Channel-mix backward ─────────────────────────────────────────────
      // out = h + r2 * cmix_out   → gradH passes through to h directly
      // Also need grad for r2 * cmix_out
      const gradCmixOut = new Float32Array(this.dim);
      const gradR2      = new Float32Array(this.dim);
      if (cache && cache.r2 && cache.cmix_out) {
        for (let d = 0; d < this.dim; d++) {
          gradCmixOut[d] = gradH[d] * cache.r2[d];
          gradR2[d]      = gradH[d] * cache.cmix_out[d];
        }
        // Wv2 backward (frozen)
        let gradKRelu2 = block.Wv2.backward(gradCmixOut, cache.k_relu2);

        // Squared-ReLU gradient
        const gradKcmix = new Float32Array(gradKRelu2.length);
        for (let d = 0; d < gradKRelu2.length; d++) {
          gradKcmix[d] = gradKRelu2[d] * 2 * Math.sqrt(Math.max(0, cache.k_relu2[d]));
        }
        // Wk2 backward (frozen)
        block.Wk2.backward(gradKcmix, cache.kx2);
        // Wr2 backward (frozen) — gradient through sigmoid
        const gradPreSigR2 = new Float32Array(this.dim);
        for (let d = 0; d < this.dim; d++) {
          gradPreSigR2[d] = gradR2[d] * cache.r2[d] * (1 - cache.r2[d]);
        }
        block.Wr2.backward(gradPreSigR2, cache.rx2);
      }

      // LN2 backward (frozen)
      if (cache && cache.ln2Res) {
        this._ignoredGrad = block.ln2.backward(gradH, cache.ln2Res.cache ?? cache.ln2Res);
      }

      // ── Time-mix backward (simplified — propagate input gradient only) ───
      // For BPTT we only need to thread the recurrent state gradient.
      // Full time-mix backward with LoRA: handled by LoRA adapter backward below.
      let gradTimeMixIn = new Float32Array(this.dim);
      if (cache && cache.gated && cache.wkv) {
        // att_o backward (frozen)
        const gradGated = block.att_o.backward(gradH, cache.gated);

        // Approximate WKV state backward: pass state gradients through
        if (stateGrad) {
          gradStatesOut[i] = {
            num: new Float32Array(stateGrad.num),
            den: new Float32Array(stateGrad.den)
          };
        }

        // att_r, att_k, att_v, att_g backward (frozen, accumulate LoRA grads)
        const r_grad = block.att_r.backward(gradGated, cache.xr ?? cache.x_ln);
        const k_grad = block.att_k.backward(gradGated, cache.xk ?? cache.x_ln);
        const v_grad = block.att_v.backward(cache.wkv, cache.xv ?? cache.x_ln);
        if (block.att_g && cache.g) {
          block.att_g.backward(gradH, cache.x_ln);
        }
        // Accumulate LoRA adapter gradients for each target layer
        for (const name of targetNames) {
          const layer = block[name];
          if (layer && layer.loraAdapter && layer.loraCache) {
            // Use the layer's cached forward inputs for the backward
            layer.loraAdapter.backward(gradGated, layer.loraCache);
          }
        }
      }

      // LN1 backward (frozen)
      if (cache && cache.ln1Res) {
        gradTimeMixIn = block.ln1.backward(gradH, cache.ln1Res.cache ?? cache.ln1Res);
      }

      // Residual: grad flows through x unchanged
      gradH = gradTimeMixIn;
    }

    // ── Pre-block LN and embedding (frozen, no accumulation needed) ─────────
    // (embedding gradients are skipped for LoRA-only training)

    return { gradStatesOut };
  }

  /**
   * Backward pass for full-weight training.
   *
   * Propagates gradients through ALL layers and accumulates weight gradients
   * for every parameter: embeddings, LayerNorms, all BitLinear layers, and head.
   * LoRA adapters (if present) also get gradients.
   *
   * @param {Float32Array} gradLogits  - Gradient w.r.t. logits [vocabSize]
   * @param {Object}       caches      - Cached activations from forward()
   * @param {Array}        loraCaches  - Per-block LoRA caches
   * @param {Array<Object>} stateGrads - Incoming BPTT state grads [numLayers] × { num, den }
   * @returns {{ gradStatesOut: Array<Object> }}
   */
  backwardFull(gradLogits, caches, loraCaches, stateGrads) {
    const { lnInRes, blockCaches, lnOutRes, tokenId } = caches;

    // ── Head backward (UNFROZEN — accumulates weight grad) ──────────────────
    let gradH = this.head.backward(gradLogits, lnOutRes.out);

    // ── Post-block LN backward (accumulates weight/bias grads) ──────────────
    gradH = this.ln_out.backward(gradH, lnOutRes.cache);

    // ── Block backward (reverse order for BPTT) ─────────────────────────────
    const gradStatesOut = new Array(this.numLayers).fill(null).map(() => ({
      num: new Float32Array(this.dim),
      den: new Float32Array(this.dim)
    }));

    const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];

    for (let i = this.numLayers - 1; i >= 0; i--) {
      const block    = this.blocks[i];
      const cache    = blockCaches[i];
      const stateGrad = stateGrads ? stateGrads[i] : null;

      // ── Channel-mix backward (all layers accumulate grads) ────────────────
      const gradCmixOut = new Float32Array(this.dim);
      const gradR2      = new Float32Array(this.dim);
      if (cache && cache.r2 && cache.cmix_out) {
        for (let d = 0; d < this.dim; d++) {
          gradCmixOut[d] = gradH[d] * cache.r2[d];
          gradR2[d]      = gradH[d] * cache.cmix_out[d];
        }
        let gradKRelu2 = block.Wv2.backward(gradCmixOut, cache.k_relu2);

        const gradKcmix = new Float32Array(gradKRelu2.length);
        for (let d = 0; d < gradKRelu2.length; d++) {
          gradKcmix[d] = gradKRelu2[d] * 2 * Math.sqrt(Math.max(0, cache.k_relu2[d]));
        }
        block.Wk2.backward(gradKcmix, cache.kx2);
        const gradPreSigR2 = new Float32Array(this.dim);
        for (let d = 0; d < this.dim; d++) {
          gradPreSigR2[d] = gradR2[d] * cache.r2[d] * (1 - cache.r2[d]);
        }
        block.Wr2.backward(gradPreSigR2, cache.rx2);
      }

      // LN2 backward (accumulates grads)
      if (cache && cache.ln2Res) {
        this._ignoredGrad = block.ln2.backward(gradH, cache.ln2Res.cache ?? cache.ln2Res);
      }

      // ── Time-mix backward ────────────────────────────────────────────────
      let gradTimeMixIn = new Float32Array(this.dim);
      if (cache && cache.gated && cache.wkv) {
        const gradGated = block.att_o.backward(gradH, cache.gated);

        if (stateGrad) {
          gradStatesOut[i] = {
            num: new Float32Array(stateGrad.num),
            den: new Float32Array(stateGrad.den)
          };
        }

        block.att_r.backward(gradGated, cache.xr ?? cache.x_ln);
        block.att_k.backward(gradGated, cache.xk ?? cache.x_ln);
        block.att_v.backward(cache.wkv, cache.xv ?? cache.x_ln);
        if (block.att_g && cache.g) {
          block.att_g.backward(gradH, cache.x_ln);
        }

        // Also accumulate LoRA adapter gradients if present
        for (const name of targetNames) {
          const layer = block[name];
          if (layer && layer.loraAdapter && layer.loraCache) {
            layer.loraAdapter.backward(gradGated, layer.loraCache);
          }
        }
      }

      // LN1 backward (accumulates grads)
      if (cache && cache.ln1Res) {
        gradTimeMixIn = block.ln1.backward(gradH, cache.ln1Res.cache ?? cache.ln1Res);
      }

      gradH = gradTimeMixIn;
    }

    // ── Pre-block LN backward ───────────────────────────────────────────────
    gradH = this.ln_in.backward(gradH, lnInRes.cache);

    // ── Embedding gradient accumulation ─────────────────────────────────────
    if (tokenId !== undefined && tokenId !== null) {
      const embOffset = tokenId * this.dim;
      for (let i = 0; i < this.dim; i++) {
        this.embeddingGrad[embOffset + i] += gradH[i];
      }
    }

    return { gradStatesOut };
  }

  /**
   * Get all LoRA adapters in the model.
   * 
   * @returns {Array<LoRAAdapter>} Array of all LoRA adapters
   */
  getLoRALayers() {
    const adapters = [];
    const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];
    
    for (const block of this.blocks) {
      for (const name of targetNames) {
        const layer = block[name];
        if (layer && layer.loraAdapter) {
          adapters.push(layer.loraAdapter);
        }
      }
    }
    
    return adapters;
  }

  /**
   * Get ALL trainable parameters for full-weight training.
   *
   * Returns param objects for: embeddings, all LayerNorms, all BitLinear layers
   * (including PoST, channel-mix, and head), plus LoRA adapters if present.
   *
   * @returns {Array<{ weight: Float32Array, grad: Float32Array, mom: Float32Array, rows: number, cols: number }>}
   */
  getAllParams() {
    const params = [];

    // ── Embedding ───────────────────────────────────────────────────────────
    params.push({
      weight: this.embedding,
      grad:   this.embeddingGrad,
      mom:    this.embeddingMom,
      rows:   this.vocabSize,
      cols:   this.dim,
    });

    // ── Pre-block LayerNorm ─────────────────────────────────────────────────
    params.push(...this._lnParams(this.ln_in));

    // ── Blocks ──────────────────────────────────────────────────────────────
    for (const block of this.blocks) {
      // LayerNorms
      params.push(...this._lnParams(block.ln1));
      params.push(...this._lnParams(block.ln2));

      // All BitLinear layers in this block
      for (const layer of block.getLinearLayers()) {
        params.push(this._bitLinearParam(layer));
      }

      // LoRA adapters (if present)
      const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];
      for (const name of targetNames) {
        const layer = block[name];
        if (layer && layer.loraAdapter) {
          const adapter = layer.loraAdapter;
          params.push({
            weight: adapter.A,
            grad:   adapter.gradA,
            mom:    adapter.momA ?? (adapter.momA = new Float32Array(adapter.A.length)),
            rows:   adapter.rank,
            cols:   adapter.inDim,
          });
          params.push({
            weight: adapter.B,
            grad:   adapter.gradB,
            mom:    adapter.momB ?? (adapter.momB = new Float32Array(adapter.B.length)),
            rows:   adapter.outDim,
            cols:   adapter.rank,
          });
        }
      }
    }

    // ── Post-block LayerNorm ────────────────────────────────────────────────
    params.push(...this._lnParams(this.ln_out));

    // ── Head ────────────────────────────────────────────────────────────────
    params.push(this._bitLinearParam(this.head));

    return params;
  }

  /**
   * Unfreeze all base model layers (for full-weight training).
   */
  unfreezeAll() {
    for (const block of this.blocks) {
      for (const layer of block.getLinearLayers()) {
        layer.frozen = false;
      }
    }
    this.head.frozen = false;
  }

  /**
   * Freeze all base model layers (for LoRA-only training).
   */
  freezeAll() {
    for (const block of this.blocks) {
      for (const layer of block.getLinearLayers()) {
        layer.frozen = true;
      }
    }
    this.head.frozen = true;
  }

  /**
   * Get all weight arrays in the model (for checkpointing / export).
   *
   * @returns {Array<{ name: string, data: Float32Array, rows: number, cols: number }>}
   */
  getAllWeights() {
    const weights = [];

    weights.push({ name: 'embedding', data: this.embedding, rows: this.vocabSize, cols: this.dim });
    weights.push({ name: 'ln_in.weight', data: this.ln_in.weight, rows: 1, cols: this.dim });
    weights.push({ name: 'ln_in.bias', data: this.ln_in.bias, rows: 1, cols: this.dim });

    for (let b = 0; b < this.blocks.length; b++) {
      const block = this.blocks[b];
      weights.push({ name: `block.${b}.ln1.weight`, data: block.ln1.weight, rows: 1, cols: this.dim });
      weights.push({ name: `block.${b}.ln1.bias`, data: block.ln1.bias, rows: 1, cols: this.dim });
      weights.push({ name: `block.${b}.ln2.weight`, data: block.ln2.weight, rows: 1, cols: this.dim });
      weights.push({ name: `block.${b}.ln2.bias`, data: block.ln2.bias, rows: 1, cols: this.dim });

      const layerNames = [
        'att_r', 'att_k', 'att_v', 'att_o', 'att_g',
        'W_x_kappa', 'W_x_a', 'W_x_w',
        'W_x_rx2', 'W_x_kx2', 'Wr2', 'Wk2', 'Wv2'
      ];
      for (const ln of layerNames) {
        const layer = block[ln];
        if (layer) {
          weights.push({ name: `block.${b}.${ln}`, data: layer.weight, rows: layer.outDim, cols: layer.inDim });
        }
      }
    }

    weights.push({ name: 'ln_out.weight', data: this.ln_out.weight, rows: 1, cols: this.dim });
    weights.push({ name: 'ln_out.bias', data: this.ln_out.bias, rows: 1, cols: this.dim });
    weights.push({ name: 'head', data: this.head.weight, rows: this.head.outDim, cols: this.head.inDim });

    return weights;
  }

  /**
   * Zero out all gradients (LoRA adapters only).
   */
  zeroGrad() {
    const adapters = this.getLoRALayers();
    for (const adapter of adapters) {
      adapter.zeroGrad();
    }
  }

  /**
   * Zero out ALL gradients (all parameters — for full-weight training).
   */
  zeroGradAll() {
    // Embedding
    this.embeddingGrad.fill(0);

    // LayerNorms
    this.ln_in.zeroGrad();
    this.ln_out.zeroGrad();

    // Blocks
    for (const block of this.blocks) {
      block.ln1.zeroGrad();
      block.ln2.zeroGrad();
      for (const layer of block.getLinearLayers()) {
        layer.zeroGrad();
        layer.markDirty();
      }
    }

    // Head
    this.head.zeroGrad();
    this.head.markDirty();

    // LoRA adapters (if present)
    for (const adapter of this.getLoRALayers()) {
      adapter.zeroGrad();
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  /**
   * Extract optimizer param objects from a LayerNorm.
   * @param {LayerNorm} ln
   * @returns {Array<Object>}
   */
  _lnParams(ln) {
    // Ensure momentum buffers exist
    if (!ln._mom_w) ln._mom_w = new Float32Array(ln.dim);
    if (!ln._mom_b) ln._mom_b = new Float32Array(ln.dim);
    return [
      { weight: ln.weight, grad: ln.grad_w, mom: ln._mom_w, rows: 1, cols: ln.dim },
      { weight: ln.bias,   grad: ln.grad_b, mom: ln._mom_b, rows: 1, cols: ln.dim },
    ];
  }

  /**
   * Extract optimizer param object from a BitLinear layer.
   * @param {BitLinear} layer
   * @returns {Object}
   */
  _bitLinearParam(layer) {
    return {
      weight: layer.weight,
      grad:   layer.grad,
      mom:    layer.momentum,
      rows:   layer.outDim,
      cols:   layer.inDim,
    };
  }
}
