// Finite-difference check of Brymar's backward passes (backwardFull and backwardLoRAOnly) on a tiny
// model, over a 3-token sequence. BitLinear runs a float forward here (STE semantics: the weight
// gradient is the gradient of the float model), so any mismatch is the backward's, not quantization's.
// A correct backward gives cos ≈ 1 and |analytic|/|true| ≈ 1 for every parameter.   node tests/gradcheck.mjs
import { RWKVModel } from '../model/model.js';
import { BitLinear } from '../model/bitlinear.js';
import { injectLoRA } from '../model/lora.js';

let seed = 3; Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
BitLinear.prototype.forward = function (x) { const o = new Float32Array(this.outDim); for (let r = 0; r < this.outDim; r++) { let a = 0; for (let i = 0; i < this.inDim; i++) a += this.weight[r * this.inDim + i] * x[i]; o[r] = a; } return o; };

const seq = [3, 8, 14], target = 5, V = 20;
const ce = lg => { let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0; for (const v of lg) s += Math.exp(v - mx); return -(lg[target] - mx - Math.log(s)); };
function analytic(m, mode) {
  const outs = []; let st = m.getEmptyState(); for (const t of seq) { const r = m.forward(t, st); st = r.state; outs.push(r); }
  const p = Float32Array.from(outs.at(-1).logits); let mx = -1e9; for (const v of p) mx = Math.max(mx, v); let s = 0;
  for (let i = 0; i < V; i++) { p[i] = Math.exp(p[i] - mx); s += p[i]; } for (let i = 0; i < V; i++) p[i] /= s; p[target] -= 1;
  const bw = (mode === 'full' ? m.backwardFull : m.backwardLoRAOnly).bind(m);
  let g = bw(p, outs.at(-1).caches, outs.at(-1).loraCaches, null).gradStatesOut;
  for (let t = outs.length - 2; t >= 0; t--) g = bw(new Float32Array(V), outs[t].caches, outs[t].loraCaches, g).gradStatesOut;
}
const loss = m => { let st = m.getEmptyState(), r; for (const t of seq) { r = m.forward(t, st); st = r.state; } return ce(r.logits); };
const eps = 1e-3; let total = 0, bad = 0;
function check(m, label, W, G) {
  if (!W || !G) { console.log(`${label.padEnd(18)} (no gradient buffer)`); return; }
  let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) { const o = W[i]; W[i] = o + eps; const a = loss(m); W[i] = o - eps; const b = loss(m); W[i] = o; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  total++;
  if (nn < 1e-12) { console.log(`${label.padEnd(18)} (does not affect the loss)`); return; }
  const cos = na ? dot / Math.sqrt(na * nn) : 0, ratio = Math.sqrt(na / nn), ok = cos > 0.99 && Math.abs(ratio - 1) < 0.05; if (!ok) bad++;
  console.log(`${label.padEnd(18)} cos=${cos.toFixed(3).padStart(6)}  |analytic|/|true|=${ratio.toFixed(3)}  ${ok ? 'ok' : 'WRONG'}`);
}
const cfg = { vocabSize: V, dim: 8, numLayers: 2, dimFFN: 16 };

console.log('── backwardFull (full-weight mode) ──');
{ const m = new RWKVModel(cfg); m.unfreezeAll(); m.zeroGradAll(); analytic(m, 'full');
  check(m, 'head', m.head.weight, m.head.grad);
  for (const L of [1, 0]) { const b = m.blocks[L];
    for (const n of ['att_r', 'att_k', 'att_v', 'att_o', 'att_g', 'W_x_kappa', 'W_x_a', 'W_x_w', 'Wr2', 'Wk2', 'Wv2']) check(m, `L${L}.${n}`, b[n].weight, b[n].grad);
    check(m, `L${L}.ln1.weight`, b.ln1.weight, b.ln1.grad_w); }
  check(m, 'embedding', m.embedding, m.embeddingGrad); }

console.log('\n── backwardLoRAOnly (LoRA mode: the only trained parameters are the adapters) ──');
{ const m = new RWKVModel(cfg); m.freezeAll(); injectLoRA(m, 2, 4);
  for (const a of m.getLoRALayers()) for (let i = 0; i < a.B.length; i++) a.B[i] = (Math.random() - 0.5) * 0.4;   // B starts at 0; make the path live
  for (const a of m.getLoRALayers()) a.zeroGrad();
  analytic(m, 'lora');
  for (const L of [1, 0]) for (const n of ['att_r', 'att_k', 'att_v', 'att_o']) { const a = m.blocks[L][n].loraAdapter; check(m, `L${L}.${n}.lora.A`, a.A, a.gradA); check(m, `L${L}.${n}.lora.B`, a.B, a.gradB); } }

console.log(`\n${bad} of ${total} parameter groups get a WRONG gradient`);
process.exit(bad ? 1 : 0);
