// Finite-difference check of model/rwkv7.js: every parameter (full mode) and every LoRA matrix (LoRA mode),
// over a 4-token sequence with BPTT through the matrix memory, token shift and the value residual.
// float64, central differences, step 1e-6.   node tests/gradcheck-rwkv7.mjs
import { RWKV7 } from '../model/rwkv7.js';
let seed = 11; Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const gauss = () => { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); };
const cfg = { vocabSize: 11, dim: 8, numLayers: 2, headSize: 4, dimFFN: 16, dDecay: 3, dAAA: 3, dMV: 2, dGate: 3 };
const seq = [3, 7, 1, 9], targets = [7, 1, 9, 4];
function randomize(m) { for (const [n, w] of Object.entries(m.p)) for (let i = 0; i < w.length; i++) w[i] = /x_|k_a/.test(n) ? Math.random() : n.endsWith('w0') ? -1 - Math.random() : gauss() * 0.6; }
const ce = (lg, t) => { let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0; for (const v of lg) s += Math.exp(v - mx); return -(lg[t] - mx - Math.log(s)); };
function loss(m) { let st = m.getEmptyState(), L = 0; seq.forEach((t, i) => { const r = m.forward(t, st); st = r.state; L += ce(r.logits, targets[i]); }); return L; }
function analytic(m) {
  m.zeroGrad(); let st = m.getEmptyState(); const outs = [];
  for (const t of seq) { const r = m.forward(t, st); st = r.state; outs.push(r); }
  let g = null;
  for (let i = outs.length - 1; i >= 0; i--) {
    const lg = outs[i].logits, p = new Float64Array(lg.length); let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0;
    for (let j = 0; j < lg.length; j++) { p[j] = Math.exp(lg[j] - mx); s += p[j]; } for (let j = 0; j < lg.length; j++) p[j] /= s; p[targets[i]] -= 1;
    g = m.backward(p, outs[i].cache, g);
  }
}
let fails = 0, n = 0, worst = 1;
function check(m, label, W, G) {
  const eps = 1e-6; let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) { const o = W[i]; W[i] = o + eps; const a = loss(m); W[i] = o - eps; const b = loss(m); W[i] = o; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  n++; if (nn < 1e-20 && na < 1e-20) { console.log(`${label.padEnd(30)} (no effect, no gradient)  ok`); return; }
  const cos = dot / Math.sqrt(na * nn + 1e-300), ratio = Math.sqrt(na / (nn + 1e-300)), ok = cos > 0.999999 && Math.abs(ratio - 1) < 1e-5;
  if (!ok) fails++; worst = Math.min(worst, cos);
  if (!ok || process.env.VERBOSE) console.log(`${label.padEnd(30)} cos=${cos.toFixed(8)} ratio=${ratio.toFixed(8)} ${ok ? 'ok' : 'FAIL'}`);
}
console.log('── full mode: every parameter ──');
{ const m = new RWKV7(cfg, { Float: Float64Array }); randomize(m); analytic(m); for (const [name, w] of Object.entries(m.p)) check(m, name, w, m.g[name]); }
const full = n;
console.log(`checked ${full} parameter tensors`);
console.log('── LoRA mode: base frozen, adapters on receptance/key/value/output ──');
{ const m = new RWKV7(cfg, { Float: Float64Array }); randomize(m); m.injectLoRA(2, 4);
  for (const a of Object.values(m.lora)) for (let i = 0; i < a.B.length; i++) a.B[i] = gauss() * 0.3;   // B starts at 0; make the path live
  analytic(m);
  for (const [name, a] of Object.entries(m.lora)) { check(m, name.replace('blocks.', 'L').replace('.att', '') + '.A', a.A, a.gA); check(m, name.replace('blocks.', 'L').replace('.att', '') + '.B', a.B, a.gB); }
  const leaked = Object.values(m.g).some(g => g.some(v => v !== 0));
  if (leaked) { fails++; console.log('frozen base weights received gradient  FAIL'); } else console.log('frozen base weights: no gradient  ok'); }
console.log(`checked ${n - full} LoRA tensors`);
console.log(fails ? `\n${fails} FAILED` : `\nALL OK — ${n} tensors exact (worst cos ${worst.toFixed(10)})`);
process.exit(fails ? 1 : 0);
