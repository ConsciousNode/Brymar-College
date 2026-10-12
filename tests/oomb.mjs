// The chunked trainer must give the SAME gradient as one unchunked pass (and as a by-hand BPTT).
import { RWKV7 } from '../model/rwkv7.js';
import { OOMBTrainer } from '../training/oomb.js';
let seed = 5; Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const cfg = { vocabSize: 13, dim: 8, numLayers: 2, headSize: 4, dimFFN: 16, dDecay: 3, dAAA: 3, dMV: 2, dGate: 3 };
const m = new RWKV7(cfg, { Float: Float64Array });
for (const w of Object.values(m.p)) for (let i = 0; i < w.length; i++) w[i] += (Math.random() - 0.5) * 0.5;
const toks = Array.from({ length: 23 }, () => Math.floor(Math.random() * 13));
const grads = async cs => { m.zeroGrad(); const L = await new OOMBTrainer(m, { chunkSize: cs }).trainSequence(toks); return { L, g: Object.values(m.g).map(g => Float64Array.from(g)) }; };
const a = await grads(1000), b = await grads(5), c = await grads(1);
let d = 0; for (const other of [b, c]) other.g.forEach((g, k) => g.forEach((v, i) => { d = Math.max(d, Math.abs(v - a.g[k][i])); }));
const ok = d < 1e-12 && Math.abs(a.L - b.L) < 1e-12;
console.log(`loss ${a.L.toFixed(6)} · max |Δgrad| chunk 5 / chunk 1 vs unchunked: ${d.toExponential(2)}  ${ok ? 'ok' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
