// model/rwkv7block.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// RWKV-v7 "Goose" block with PoST decay gates

import { BitLinear } from './bitlinear.js';
import { LayerNorm } from './layernorm.js';

/**
 * Initialize PoST decay gate biases.
 * Tapered logarithmic decay rates from fast to slow.
 * 
 * @param {number} dim - Dimension
 * @returns {Float32Array} Log decay biases [dim]
 */
function initPoSTDecayGates(dim) {
  const minLogDecay = Math.log(1e-4);
  const maxLogDecay = Math.log(1 - 1e-4);
  
  const logRates = new Float32Array(dim);
  for (let k = 0; k < dim; k++) {
    const t = k / Math.max(1, dim - 1);
    logRates[k] = minLogDecay + t * (maxLogDecay - minLogDecay);
  }
  
  const tapered = new Float32Array(dim);
  for (let k = 0; k < dim; k++) {
    const alpha = (dim - k) / Math.max(1, dim - 1);
    tapered[k] = logRates[k] * alpha;
  }
  
  return tapered;
}

/**
 * RWKVv7Block: RWKV-v7 "Goose" architecture with PoST decay gates
 * 
 * Time-mixing: Gated WKV with removal mechanism and PoST decay
 * Channel-mixing: Squared ReLU gated FFN
 * 
 * Reference: HTMLNLM.html RWKVv7Block class
 */
export class RWKVv7Block {
  /**
   * @param {number} dim - Model dimension
   * @param {number} dimFFN - FFN hidden dimension
   */
  constructor(dim, dimFFN) {
    this.dim = dim;
    this.dimFFN = dimFFN;
    
    // Layer normalization
    this.ln1 = new LayerNorm(dim);
    this.ln2 = new LayerNorm(dim);
    
    // Time-mixing projections (renamed from W_x_* to att_*)
    this.att_r = new BitLinear(dim, dim);      // receptance
    this.att_k = new BitLinear(dim, dim);      // key
    this.att_v = new BitLinear(dim, dim);      // value
    this.att_o = new BitLinear(dim, dim);      // output
    this.att_g = new BitLinear(dim, dim);      // gate
    
    // PoST components
    this.W_x_kappa = new BitLinear(dim, dim);  // removal direction
    this.W_x_a = new BitLinear(dim, dim);      // removal gate
    this.W_x_w = new BitLinear(dim, dim);      // decay projection
    this.log_w_bias = initPoSTDecayGates(dim); // decay biases
    
    // Channel-mixing projections
    this.W_x_rx2 = new BitLinear(dim, dim);
    this.W_x_kx2 = new BitLinear(dim, dim);
    this.Wr2 = new BitLinear(dim, dim);
    this.Wk2 = new BitLinear(dim, dimFFN);
    this.Wv2 = new BitLinear(dimFFN, dim);
    
    // Token shift interpolation parameters
    this.mu_r = new Float32Array(dim).fill(0.5);
    this.mu_k = new Float32Array(dim).fill(0.5);
    this.mu_v = new Float32Array(dim).fill(0.5);
    this.mu_kappa = new Float32Array(dim).fill(0.5);
    this.mu_a = new Float32Array(dim).fill(0.5);
    this.mu_w = new Float32Array(dim).fill(0.5);
  }

  /**
   * Sigmoid activation with clipping.
   * 
   * @param {Float32Array} x - Input
   * @returns {Float32Array} Sigmoid output
   */
  static sigmoid(x) {
    const o = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      o[i] = 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x[i]))));
    }
    return o;
  }

  /**
   * Token shift: interpolate between current and previous token.
   * 
   * @param {Float32Array} x - Current token
   * @param {Float32Array} prevX - Previous token
   * @param {Float32Array} mu - Interpolation weights
   * @returns {Float32Array} Shifted token
   */
  tokenShift(x, prevX, mu) {
    const out = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      out[i] = x[i] + mu[i] * (prevX[i] - x[i]);
    }
    return out;
  }

  /**
   * Forward pass through RWKV-v7 block.
   * 
   * @param {Float32Array} x - Input [dim]
   * @param {Object} state - Recurrent state { prevX, prevX2, num, den }
   * @returns {{ out: Float32Array, state: Object, cache: Object }}
   */
  forward(x, state) {
    const dim = this.dim;
    
    // ===== TIME-MIXING =====
    const ln1Res = this.ln1.forward(x);
    const x_ln = ln1Res.out;
    const prevX = state.prevX || new Float32Array(dim);
    
    // Token-shifted projections
    const xr = this.tokenShift(x_ln, prevX, this.mu_r);
    const xk = this.tokenShift(x_ln, prevX, this.mu_k);
    const xv = this.tokenShift(x_ln, prevX, this.mu_v);
    const xkappa = this.tokenShift(x_ln, prevX, this.mu_kappa);
    const xa = this.tokenShift(x_ln, prevX, this.mu_a);
    const xw = this.tokenShift(x_ln, prevX, this.mu_w);
    
    // Compute r, k, v, kappa, a, w
    const r = RWKVv7Block.sigmoid(this.att_r.forward(xr));
    const k = this.att_k.forward(xk);
    const v = this.att_v.forward(xv);
    const kappa_raw = this.W_x_kappa.forward(xkappa);
    const a = RWKVv7Block.sigmoid(this.W_x_a.forward(xa));
    const w_proj = this.W_x_w.forward(xw);
    
    // PoST decay: w = exp(-exp(clamp(w_proj + log_w_bias - 0.5)))
    const w = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      w[i] = Math.exp(-Math.exp(Math.max(-8, Math.min(0, w_proj[i] + this.log_w_bias[i] - 0.5))));
    }
    
    // Normalize kappa (removal direction)
    let kappaNorm = 0;
    for (let i = 0; i < dim; i++) {
      kappaNorm += kappa_raw[i] * kappa_raw[i];
    }
    kappaNorm = Math.sqrt(kappaNorm) + 1e-8;
    const kappa = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      kappa[i] = kappa_raw[i] / kappaNorm;
    }
    
    // WKV with removal mechanism
    const num = new Float32Array(state.num || new Float32Array(dim));
    const den = new Float32Array(state.den || new Float32Array(dim).fill(1));
    
    let removalDot = 0;
    for (let i = 0; i < dim; i++) {
      removalDot += a[i] * kappa[i] * num[i];
    }
    
    const wkv = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      num[i] = w[i] * num[i] - kappa[i] * removalDot + k[i] * v[i];
      den[i] = w[i] * den[i] - kappa[i] * removalDot + k[i];
      wkv[i] = num[i] / (Math.abs(den[i]) + 1e-8);
    }
    
    // Gate and output projection
    const g = RWKVv7Block.sigmoid(this.att_g.forward(x_ln));
    const gated = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      gated[i] = r[i] * wkv[i] * g[i];
    }
    const tmix_out = this.att_o.forward(gated);
    
    // Residual connection
    const h = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      h[i] = x[i] + tmix_out[i];
    }
    
    // ===== CHANNEL-MIXING =====
    const ln2Res = this.ln2.forward(h);
    const h_ln = ln2Res.out;
    const prevX2 = state.prevX2 || new Float32Array(dim);
    
    const rx2 = this.tokenShift(h_ln, prevX2, this.mu_r);
    const kx2 = this.tokenShift(h_ln, prevX2, this.mu_k);
    
    const r2 = RWKVv7Block.sigmoid(this.Wr2.forward(rx2));
    const k_cmix = this.Wk2.forward(kx2);
    
    // Squared ReLU
    const k_relu2 = new Float32Array(k_cmix.length);
    for (let i = 0; i < k_cmix.length; i++) {
      k_relu2[i] = Math.max(0, k_cmix[i]) ** 2;
    }
    
    const cmix_out = this.Wv2.forward(k_relu2);
    
    // Gated residual
    const out = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      out[i] = h[i] + r2[i] * cmix_out[i];
    }
    
    // Update state
    const newState = {
      prevX: new Float32Array(x_ln),
      prevX2: new Float32Array(h_ln),
      num: new Float32Array(num),
      den: new Float32Array(den)
    };
    
    // Cache for backward pass
    const cache = {
      x, x_ln, ln1Res, prevX,
      xr, xk, xv, xkappa, xa, xw,
      r, k, v, kappa_raw, kappa, a, w_proj, w,
      num: state.num, den: state.den, removalDot,
      wkv, g, gated, tmix_out,
      h, h_ln, ln2Res, prevX2,
      rx2, kx2, r2, k_cmix, k_relu2, cmix_out
    };
    
    return { out, state: newState, cache };
  }

  /**
   * Get all BitLinear layers in this block.
   * 
   * @returns {BitLinear[]} Array of all linear layers
   */
  getLinearLayers() {
    return [
      this.att_r, this.att_k, this.att_v, this.att_o, this.att_g,
      this.W_x_kappa, this.W_x_a, this.W_x_w,
      this.W_x_rx2, this.W_x_kx2,
      this.Wr2, this.Wk2, this.Wv2
    ];
  }
}
