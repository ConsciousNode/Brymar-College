// training/oomb.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
//
// OOMB chunk-recurrent BPTT for model/rwkv7.js (rewritten 2026-10-11, see AUDIT.md).
// Pass 1 runs the sequence forward and keeps only the recurrent state at each chunk boundary.
// Pass 2 replays one chunk at a time, last chunk first, and backpropagates through it, carrying the
// state gradient (token shift + per-head matrix memory) across chunk boundaries. The gradient is the
// exact gradient of the whole sequence's loss; only one chunk's activations live in memory at once.
// (tests/oomb.mjs checks chunked == unchunked.)

/** Softmax cross-entropy: returns { loss, grad } with grad = (p − onehot) · scale. */
export function crossEntropy(logits, target, scale = 1) {
  const V = logits.length, g = new Float32Array(V); let mx = -Infinity;
  for (let i = 0; i < V; i++) if (logits[i] > mx) mx = logits[i];
  let s = 0; for (let i = 0; i < V; i++) { g[i] = Math.exp(logits[i] - mx); s += g[i]; }
  const loss = -(logits[target] - mx - Math.log(s));
  for (let i = 0; i < V; i++) g[i] = (g[i] / s - (i === target ? 1 : 0)) * scale;
  return { loss, grad: g };
}

export class OOMBTrainer {
  /**
   * @param {import('../model/rwkv7.js').RWKV7} model
   * @param {object} [opts] { chunkSize = 128 }
   */
  constructor(model, opts = {}) { this.model = model; this.chunkSize = opts.chunkSize ?? 128; this.step = 0; }

  /**
   * Accumulate the gradient of the mean next-token loss over `tokens` (length ≥ 2).
   * @param {number[]} tokens
   * @param {object} [opts] { lossMask: boolean[] (loss only where true, e.g. only on the persona's replies), state }
   * @returns {Promise<number>} mean loss over the counted positions
   */
  async trainSequence(tokens, opts = {}) {
    const m = this.model, T = tokens.length - 1, cs = this.chunkSize;
    if (T < 1) return 0;
    const mask = opts.lossMask, count = mask ? mask.slice(1).filter(Boolean).length || 1 : T;
    // pass 1: forward, keep chunk-boundary states
    const starts = [];
    let st = opts.state ? m.cloneState(opts.state) : m.getEmptyState();
    for (let s = 0; s < T; s += cs) {
      starts.push(m.cloneState(st));
      for (let t = s; t < Math.min(s + cs, T); t++) st = m.forward(tokens[t], st).state;
      await new Promise(r => setImmediate(r));
    }
    // pass 2: replay chunks backwards, BPTT with the state gradient carried across chunks
    let gState = null, total = 0;
    for (let ci = starts.length - 1; ci >= 0; ci--) {
      const s = ci * cs, e = Math.min(s + cs, T), acts = [];
      let cur = starts[ci];
      for (let t = s; t < e; t++) { const r = m.forward(tokens[t], cur); acts.push(r); cur = r.state; }
      for (let j = acts.length - 1; j >= 0; j--) {
        const pos = s + j + 1, counted = !mask || mask[pos];
        let g;
        if (counted) { const r = crossEntropy(acts[j].logits, tokens[pos], 1 / count); total += r.loss; g = r.grad; }
        else g = new Float32Array(m.V);
        gState = m.backward(g, acts[j].cache, gState);
      }
      await new Promise(r => setImmediate(r));
    }
    this.step++;
    return total / count;
  }
}
