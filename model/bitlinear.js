// model/bitlinear.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// BitNet b1.58 linear layer with ternary quantization

import { TMAC } from './tmac.js';

/**
 * BitLinear: BitNet b1.58 linear layer
 * 
 * Uses ternary {-1, 0, 1} quantization via TMAC for forward pass.
 * Maintains full-precision Float32 weights and gradients for training.
 * 
 * Reference: HTMLNLM.html BitLinear class
 */
export class BitLinear {
  /**
   * @param {number} inDim - Input dimension
   * @param {number} outDim - Output dimension
   * @param {number} gradAccumSteps - Gradient accumulation steps (default 1)
   */
  constructor(inDim, outDim, gradAccumSteps = 1) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.gradAccumSteps = gradAccumSteps;
    
    // Full-precision weights and gradients
    this.weight = new Float32Array(outDim * inDim);
    this.grad = new Float32Array(outDim * inDim);
    this.momentum = new Float32Array(outDim * inDim);
    
    // Ternary quantized weights
    this.packed = new Uint8Array(Math.ceil((outDim * inDim) / 4));
    this.gamma = 1.0;
    this.dirty = true;
    
    // TMAC scratch buffer
    this.lutScratch = new Float32Array(Math.ceil(inDim / 4) * 81);
    
    // Frozen flag for base model weights (LoRA training)
    this.frozen = false;
    
    // Kaiming uniform initialization: U(-sqrt(3/fan_in), sqrt(3/fan_in))
    const scale = Math.sqrt(3.0 / inDim);
    for (let i = 0; i < this.weight.length; i++) {
      this.weight[i] = (Math.random() * 2 - 1) * scale;
    }
  }

  /**
   * Pack weights to ternary if dirty.
   */
  pack() {
    if (this.dirty) {
      this.gamma = TMAC.quantizeWeights(this.weight, this.packed);
      this.dirty = false;
    }
  }

  /**
   * Forward pass: y = W @ x (using ternary quantized weights)
   * 
   * @param {Float32Array} x - Input vector [inDim]
   * @returns {Float32Array} Output vector [outDim]
   */
  forward(x) {
    this.pack();
    const out = new Float32Array(this.outDim);
    TMAC.execute(this.packed, x, this.outDim, this.inDim, this.gamma, out, this.lutScratch);
    return out;
  }

  /**
   * Backward pass: accumulate gradients and return input gradient
   * 
   * @param {Float32Array} gradOut - Gradient w.r.t. output [outDim]
   * @param {Float32Array} x - Input from forward pass [inDim]
   * @returns {Float32Array} Gradient w.r.t. input [inDim]
   */
  backward(gradOut, x) {
    // Accumulate weight gradients (skip if frozen)
    if (!this.frozen) {
      for (let o = 0; o < this.outDim; o++) {
        const rowOffset = o * this.inDim;
        const go = gradOut[o];
        for (let i = 0; i < this.inDim; i++) {
          this.grad[rowOffset + i] += go * x[i];
        }
      }
    }
    
    // Compute input gradient: gradIn = W^T @ gradOut
    const gradIn = new Float32Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) {
      let acc = 0;
      for (let o = 0; o < this.outDim; o++) {
        acc += this.weight[o * this.inDim + i] * gradOut[o];
      }
      gradIn[i] = acc;
    }
    
    return gradIn;
  }

  /**
   * Zero out accumulated gradients.
   */
  zeroGrad() {
    this.grad.fill(0);
  }

  /**
   * Mark weights as dirty (need re-quantization).
   */
  markDirty() {
    this.dirty = true;
  }
}
