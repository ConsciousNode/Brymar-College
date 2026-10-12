// model/layernorm.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Layer normalization with learnable affine parameters

/**
 * LayerNorm: Layer normalization with learnable weight and bias
 * 
 * Normalizes input to mean=0, variance=1, then applies affine transformation.
 * 
 * Reference: HTMLNLM.html LayerNorm class
 */
export class LayerNorm {
  /**
   * @param {number} dim - Layer dimension
   */
  constructor(dim) {
    this.dim = dim;
    
    // Learnable affine parameters
    this.weight = new Float32Array(dim).fill(1.0);
    this.bias = new Float32Array(dim).fill(0.0);
    
    // Gradients
    this.grad_w = new Float32Array(dim);
    this.grad_b = new Float32Array(dim);
  }

  /**
   * Forward pass: normalize then apply affine transformation
   * 
   * @param {Float32Array} x - Input vector [dim]
   * @returns {{ out: Float32Array, cache: { x: Float32Array, mean: number, rstd: number } }}
   */
  forward(x) {
    // Compute mean
    let mean = 0;
    for (let i = 0; i < this.dim; i++) {
      mean += x[i];
    }
    mean /= this.dim;
    
    // Compute variance
    let v = 0;
    for (let i = 0; i < this.dim; i++) {
      v += (x[i] - mean) ** 2;
    }
    
    // Reciprocal standard deviation
    const rstd = 1.0 / Math.sqrt((v / this.dim) + 1e-5);
    
    // Normalize and apply affine transformation
    const out = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      out[i] = ((x[i] - mean) * rstd) * this.weight[i] + this.bias[i];
    }
    
    return { out, cache: { x, mean, rstd } };
  }

  /**
   * Backward pass: accumulate gradients for weight and bias
   * 
   * @param {Float32Array} gradOut - Gradient w.r.t. output [dim]
   * @param {{ x: Float32Array, mean: number, rstd: number }} cache - Cached values from forward
   * @returns {Float32Array} Gradient w.r.t. input [dim]
   */
  backward(gradOut, cache) {
    const { x, mean, rstd } = cache;
    const gradIn = new Float32Array(this.dim);
    
    // Accumulate weight and bias gradients
    let sumGradOut = 0;
    let sumGradOutNorm = 0;
    
    for (let i = 0; i < this.dim; i++) {
      const norm = (x[i] - mean) * rstd;
      this.grad_w[i] += gradOut[i] * norm;
      this.grad_b[i] += gradOut[i];
      
      sumGradOut += gradOut[i] * this.weight[i];
      sumGradOutNorm += gradOut[i] * this.weight[i] * norm;
    }
    
    // Compute input gradient
    const scale = rstd / this.dim;
    for (let i = 0; i < this.dim; i++) {
      const norm = (x[i] - mean) * rstd;
      gradIn[i] = scale * (this.dim * gradOut[i] * this.weight[i] - sumGradOut - norm * sumGradOutNorm);
    }
    
    return gradIn;
  }

  /**
   * Zero out accumulated gradients.
   */
  zeroGrad() {
    this.grad_w.fill(0);
    this.grad_b.fill(0);
  }

  /**
   * Apply gradient descent step.
   * 
   * @param {number} lr - Learning rate
   */
  step(lr) {
    for (let i = 0; i < this.dim; i++) {
      this.weight[i] -= lr * this.grad_w[i];
      this.bias[i] -= lr * this.grad_b[i];
    }
  }
}
