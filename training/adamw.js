// training/adamw.js — AdamW over model.trainable() (base weights and/or LoRA A/B)
// MIT License — ConsciousNode SoftWorks
export class AdamW {
  constructor(lr = 1e-4, { beta1 = 0.9, beta2 = 0.99, eps = 1e-8, wd = 0.0, clip = 1.0 } = {}) {
    Object.assign(this, { lr, beta1, beta2, eps, wd, clip }); this.t = 0; this.m = new Map(); this.v = new Map();
  }
  /** @param {Array<{weight, grad}>} params  @param {number} [gradScale] e.g. 1/accumulated steps */
  step(params, gradScale = 1) {
    this.t++;
    let norm = 0; for (const p of params) for (let i = 0; i < p.grad.length; i++) norm += (p.grad[i] * gradScale) ** 2;
    norm = Math.sqrt(norm); const k = gradScale * (this.clip && norm > this.clip ? this.clip / norm : 1);
    const c1 = 1 - this.beta1 ** this.t, c2 = 1 - this.beta2 ** this.t;
    for (const p of params) {
      if (!this.m.has(p.weight)) { this.m.set(p.weight, new Float32Array(p.weight.length)); this.v.set(p.weight, new Float32Array(p.weight.length)); }
      const m = this.m.get(p.weight), v = this.v.get(p.weight), w = p.weight, g = p.grad;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * k; m[i] = this.beta1 * m[i] + (1 - this.beta1) * gi; v[i] = this.beta2 * v[i] + (1 - this.beta2) * gi * gi;
        w[i] -= this.lr * (m[i] / c1 / (Math.sqrt(v[i] / c2) + this.eps) + this.wd * w[i]);
      }
    }
    return norm;
  }
}
