// model/lora.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// LoRA (Low-Rank Adaptation) adapter system

/**
 * LoRAAdapter: Low-rank adapter for fine-tuning frozen layers
 * 
 * Implements ΔW = B @ A where:
 * - A: [rank × inDim] initialized with Kaiming normal
 * - B: [outDim × rank] initialized to zero
 * - scale: alpha / rank
 * 
 * Forward: out = baseOutput + scale * (B @ (A @ x))
 */
export class LoRAAdapter {
  /**
   * @param {number} inDim - Input dimension
   * @param {number} outDim - Output dimension
   * @param {number} rank - LoRA rank
   * @param {number} alpha - LoRA alpha (scaling factor)
   */
  constructor(inDim, outDim, rank, alpha) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.rank = rank;
    this.alpha = alpha;
    this.scale = alpha / rank;
    
    // A matrix: Kaiming normal initialization
    this.A = new Float32Array(rank * inDim);
    const stdA = Math.sqrt(2.0 / inDim);
    for (let i = 0; i < this.A.length; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      this.A[i] = stdA * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }
    
    // B matrix: zero initialization
    this.B = new Float32Array(outDim * rank).fill(0);
    
    // Gradients
    this.gradA = new Float32Array(rank * inDim);
    this.gradB = new Float32Array(outDim * rank);
    
    // Momentum buffers
    this.momA = new Float32Array(rank * inDim);
    this.momB = new Float32Array(outDim * rank);
  }

  /**
   * Forward pass: compute LoRA delta and add to base output
   * 
   * @param {Float32Array} x - Input vector [inDim]
   * @param {Float32Array} baseOutput - Output from frozen base layer [outDim]
   * @returns {{ out: Float32Array, cache: { x: Float32Array, tmp: Float32Array } }}
   */
  forward(x, baseOutput) {
    // tmp = A @ x  (rank-dimensional intermediate)
    const tmp = new Float32Array(this.rank);
    for (let r = 0; r < this.rank; r++) {
      let acc = 0;
      const rowOffset = r * this.inDim;
      for (let i = 0; i < this.inDim; i++) {
        acc += this.A[rowOffset + i] * x[i];
      }
      tmp[r] = acc;
    }
    
    // delta = B @ tmp  (outDim-dimensional delta)
    const delta = new Float32Array(this.outDim);
    for (let o = 0; o < this.outDim; o++) {
      let acc = 0;
      const rowOffset = o * this.rank;
      for (let r = 0; r < this.rank; r++) {
        acc += this.B[rowOffset + r] * tmp[r];
      }
      delta[o] = acc * this.scale;
    }
    
    // out = baseOutput + delta
    const out = new Float32Array(this.outDim);
    for (let o = 0; o < this.outDim; o++) {
      out[o] = baseOutput[o] + delta[o];
    }
    
    return { out, cache: { x, tmp } };
  }

  /**
   * Backward pass: accumulate gradients for A and B
   * 
   * @param {Float32Array} gradOut - Gradient w.r.t. output [outDim]
   * @param {{ x: Float32Array, tmp: Float32Array }} cache - Cached values from forward
   * @returns {Float32Array} Gradient w.r.t. input [inDim]
   */
  backward(gradOut, cache) {
    const { x, tmp } = cache;
    
    // Scale gradient by LoRA scale
    const scaledGradOut = new Float32Array(this.outDim);
    for (let o = 0; o < this.outDim; o++) {
      scaledGradOut[o] = gradOut[o] * this.scale;
    }
    
    // Accumulate gradB: gradB += scaledGradOut ⊗ tmp
    for (let o = 0; o < this.outDim; o++) {
      const rowOffset = o * this.rank;
      const go = scaledGradOut[o];
      for (let r = 0; r < this.rank; r++) {
        this.gradB[rowOffset + r] += go * tmp[r];
      }
    }
    
    // Compute gradient w.r.t. tmp: gradTmp = B^T @ scaledGradOut
    const gradTmp = new Float32Array(this.rank);
    for (let r = 0; r < this.rank; r++) {
      let acc = 0;
      for (let o = 0; o < this.outDim; o++) {
        acc += this.B[o * this.rank + r] * scaledGradOut[o];
      }
      gradTmp[r] = acc;
    }
    
    // Accumulate gradA: gradA += gradTmp ⊗ x
    for (let r = 0; r < this.rank; r++) {
      const rowOffset = r * this.inDim;
      const gt = gradTmp[r];
      for (let i = 0; i < this.inDim; i++) {
        this.gradA[rowOffset + i] += gt * x[i];
      }
    }
    
    // Compute gradient w.r.t. input: gradIn = A^T @ gradTmp
    const gradIn = new Float32Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) {
      let acc = 0;
      for (let r = 0; r < this.rank; r++) {
        acc += this.A[r * this.inDim + i] * gradTmp[r];
      }
      gradIn[i] = acc;
    }
    
    return gradIn;
  }

  /**
   * Zero out accumulated gradients.
   */
  zeroGrad() {
    this.gradA.fill(0);
    this.gradB.fill(0);
  }
}

/**
 * Inject LoRA adapters into model layers.
 * 
 * Wraps specified layers (att_r, att_k, att_v, att_o) with LoRA adapters.
 * 
 * @param {Object} model - RWKVModel instance
 * @param {number} rank - LoRA rank
 * @param {number} alpha - LoRA alpha
 */
export function injectLoRA(model, rankOrOpts, alpha) {
  // Accept either injectLoRA(model, rank, alpha) or injectLoRA(model, { rank, alpha })
  let rank, resolvedAlpha;
  if (typeof rankOrOpts === 'object' && rankOrOpts !== null) {
    rank          = rankOrOpts.rank;
    resolvedAlpha = rankOrOpts.alpha ?? rank * 2;
  } else {
    rank          = rankOrOpts;
    resolvedAlpha = alpha ?? rank * 2;
  }

  const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];

  for (const block of model.blocks) {
    for (const name of targetNames) {
      const layer = block[name];
      if (!layer) continue;

      // Create LoRA adapter
      const adapter = new LoRAAdapter(layer.inDim, layer.outDim, rank, resolvedAlpha);

      // Store original forward method (unquantized weight-level pass-through for BPTT)
      const originalForward = layer.forward.bind(layer);

      // Wrap forward method: stores last cache for backward access
      layer.forward = function(x) {
        const baseOut = originalForward(x);
        const loraResult = adapter.forward(x, baseOut);
        // Store cache so backwardLoRAOnly can retrieve it
        layer.loraCache = loraResult.cache;
        return loraResult.out;
      };

      // Attach adapter to layer for gradient access
      layer.loraAdapter = adapter;

      // Freeze base layer weights (BitLinear.backward respects this flag)
      layer.frozen = true;
    }
  }
}

/**
 * Extract LoRA parameters for optimizer.
 * 
 * @param {Object} model - RWKVModel instance
 * @returns {Array<{ weight: Float32Array, grad: Float32Array, mom: Float32Array, rows: number, cols: number }>}
 */
export function getLoRAParams(model) {
  const params = [];
  
  for (const block of model.blocks) {
    const targetNames = ['att_r', 'att_k', 'att_v', 'att_o'];
    
    for (const name of targetNames) {
      const layer = block[name];
      if (!layer || !layer.loraAdapter) continue;
      
      const adapter = layer.loraAdapter;
      
      // A matrix parameters
      params.push({
        weight: adapter.A,
        grad: adapter.gradA,
        mom: adapter.momA,
        rows: adapter.rank,
        cols: adapter.inDim
      });
      
      // B matrix parameters
      params.push({
        weight: adapter.B,
        grad: adapter.gradB,
        mom: adapter.momB,
        rows: adapter.outDim,
        cols: adapter.rank
      });
    }
  }
  
  return params;
}
