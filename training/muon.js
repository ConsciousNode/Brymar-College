// training/muon.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Muon optimizer with Newton-Schulz-5 orthogonalization

/**
 * MuonOptimizer: Momentum-based optimizer with Newton-Schulz orthogonalization
 * 
 * Applies Newton-Schulz iteration to orthogonalize momentum before weight update.
 * This provides better conditioning for low-rank updates like LoRA.
 * 
 * Reference: HTMLNLM.html MuonOptimizer class
 */
export class MuonOptimizer {
  /**
   * @param {number} lr - Learning rate
   * @param {number} momentum - Momentum coefficient (default 0.95)
   * @param {number} wd - Weight decay (default 0.01)
   * @param {number} gradAccum - Gradient accumulation steps (default 1)
   */
  constructor(lr, momentum = 0.95, wd = 0.01, gradAccum = 1) {
    this.lr = lr;
    this.momentum = momentum;
    this.wd = wd;
    this.gradAccum = gradAccum;
    this._buffers = {};
  }

  /**
   * Get or create scratch buffers for Newton-Schulz iteration.
   * 
   * @param {number} cols - Number of columns
   * @param {number} rows - Number of rows
   * @returns {Object} Scratch buffers
   */
  _getBuffers(cols, rows) {
    const key = `${cols}_${rows}`;
    if (!this._buffers[key]) {
      const n = cols * cols;
      this._buffers[key] = {
        S: new Float32Array(n),
        S2: new Float32Array(n),
        Poly: new Float32Array(n),
        Gc: new Float32Array(rows * cols),
        Gnew: new Float32Array(rows * cols)
      };
    }
    return this._buffers[key];
  }

  /**
   * Newton-Schulz-5 iteration for matrix orthogonalization.
   * 
   * Computes an approximate orthogonal matrix from gradient matrix G.
   * Uses 5 iterations of Newton-Schulz with polynomial coefficients.
   * 
   * @param {Float32Array} G - Gradient matrix [rows × cols]
   * @param {number} rows - Number of rows
   * @param {number} cols - Number of columns
   * @returns {Promise<Float32Array>} Orthogonalized matrix
   */
  async newtonSchulz5(G, rows, cols) {
    const n = rows * cols;
    
    // Normalize gradient
    let norm = 0;
    for (let i = 0; i < n; i++) {
      norm += G[i] * G[i];
    }
    norm = Math.sqrt(norm) + 1e-12;
    
    const bufs = this._getBuffers(cols, rows);
    const { S, S2, Poly, Gc, Gnew } = bufs;
    
    // Gc = G / norm
    for (let i = 0; i < n; i++) {
      Gc[i] = G[i] / norm;
    }

    // Newton-Schulz polynomial coefficients
    const a = 3.4445;
    const b = -4.7750;
    const c = 2.0315;

    // 5 iterations
    for (let iter = 0; iter < 5; iter++) {
      // S = Gc^T @ Gc  (Gram matrix)
      for (let x = 0; x < cols; x++) {
        for (let y = 0; y < cols; y++) {
          let dot = 0;
          for (let r = 0; r < rows; r++) {
            dot += Gc[r * cols + x] * Gc[r * cols + y];
          }
          S[x * cols + y] = dot;
        }
      }
      
      // Yield to event loop (replace yieldToMain with setImmediate)
      await new Promise(resolve => setImmediate(resolve));

      // S2 = S @ S
      for (let x = 0; x < cols; x++) {
        for (let y = 0; y < cols; y++) {
          let dot = 0;
          for (let k = 0; k < cols; k++) {
            dot += S[x * cols + k] * S[k * cols + y];
          }
          S2[x * cols + y] = dot;
        }
      }
      
      // Poly = a*I + b*S + c*S2
      for (let x = 0; x < cols; x++) {
        for (let y = 0; y < cols; y++) {
          Poly[x * cols + y] = (x === y ? a : 0) + b * S[x * cols + y] + c * S2[x * cols + y];
        }
      }
      
      // Gnew = Gc @ Poly
      for (let r = 0; r < rows; r++) {
        for (let y = 0; y < cols; y++) {
          let acc = 0;
          for (let x = 0; x < cols; x++) {
            acc += Gc[r * cols + x] * Poly[x * cols + y];
          }
          Gnew[r * cols + y] = acc;
        }
      }
      
      // Gc = Gnew
      for (let i = 0; i < rows * cols; i++) {
        Gc[i] = Gnew[i];
      }
    }
    
    return Gc;
  }

  /**
   * Perform optimizer step on parameters.
   * 
   * @param {Array<{ weight: Float32Array, grad: Float32Array, mom: Float32Array, rows: number, cols: number }>} params
   */
  async step(params) {
    for (const param of params) {
      const { weight, grad, mom, rows, cols } = param;
      const len = weight.length;
      
      // Average gradients over accumulation steps
      for (let i = 0; i < len; i++) {
        grad[i] /= this.gradAccum;
      }
      
      // Update momentum: mom = momentum * mom + grad
      for (let i = 0; i < len; i++) {
        mom[i] = this.momentum * mom[i] + grad[i];
      }
      
      // Orthogonalize momentum via Newton-Schulz-5
      const orth = await this.newtonSchulz5(mom, rows, cols);
      
      // Update weights: w = w * (1 - lr * wd) - lr * orth
      for (let i = 0; i < len; i++) {
        weight[i] = weight[i] * (1 - this.lr * this.wd) - this.lr * orth[i];
      }
      
      // Yield to event loop
      await new Promise(resolve => setImmediate(resolve));
    }
  }

  /**
   * Zero out gradients for all parameters.
   * 
   * @param {Array<{ grad: Float32Array }>} params
   */
  zeroGrads(params) {
    for (const param of params) {
      param.grad.fill(0);
    }
  }
}
