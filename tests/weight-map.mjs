// What happens when Brymar loads a real RWKV-7 checkpoint?
//
// Builds a small checkpoint with the exact tensor NAMES and SHAPES of an official RWKV-7 (x070) model
// (taken from BlinkDL's own loader in the `rwkv` pip package, rwkv/model.py), runs it through Brymar's
// real model/loader.js, and reports, per tensor: loaded, silently skipped (size mismatch), or ignored
// (no mapping). Then reports which of Brymar's own parameters are left at random initialisation.
// Finally measures what ternary rounding does to the matrices that DO load.
//
// usage: node tests/weight-map.mjs [path/to/real.safetensors]
//   With no argument it builds a miniature checkpoint (same names and shapes, small dims).
//   Pass a real RWKV-7 .safetensors to run the identical report on real weights.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadModel } from '../model/loader.js';
import { TMAC } from '../model/tmac.js';

// ── a miniature RWKV-7 checkpoint with official names/shapes ───────────────
function gauss() { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }
function buildMini(file) {
  const C = 128, H = 2, N = C / H, L = 2, V = 1000, F = 4 * C, DW = 32, DA = 32, DV = 16, DG = 64;
  const T = {};
  const t = (name, shape, scale = 0.02) => { const n = shape.reduce((a, b) => a * b, 1); const d = new Float32Array(n); for (let i = 0; i < n; i++) d[i] = gauss() * scale; T[name] = { shape, d }; };
  t('emb.weight', [V, C]);
  for (let i = 0; i < L; i++) {
    const b = `blocks.${i}.`, a = b + 'att.', f = b + 'ffn.';
    if (i === 0) { t(b + 'ln0.weight', [C], 1); t(b + 'ln0.bias', [C]); }
    t(b + 'ln1.weight', [C], 1); t(b + 'ln1.bias', [C]); t(b + 'ln2.weight', [C], 1); t(b + 'ln2.bias', [C]);
    for (const x of ['x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g']) t(a + x, [1, 1, C], 0.3);
    t(a + 'w0', [1, 1, C], 0.5); t(a + 'w1', [C, DW]); t(a + 'w2', [DW, C]);
    t(a + 'a0', [1, 1, C], 0.5); t(a + 'a1', [C, DA]); t(a + 'a2', [DA, C]);
    if (i > 0) { t(a + 'v0', [1, 1, C], 0.5); t(a + 'v1', [C, DV]); t(a + 'v2', [DV, C]); }
    t(a + 'g1', [C, DG]); t(a + 'g2', [DG, C]);
    t(a + 'k_k', [1, 1, C], 0.5); t(a + 'k_a', [1, 1, C], 0.5); t(a + 'r_k', [H, N], 0.5);
    for (const m of ['receptance', 'key', 'value', 'output']) t(a + m + '.weight', [C, C]);
    t(a + 'ln_x.weight', [C], 1); t(a + 'ln_x.bias', [C]);
    t(f + 'x_k', [1, 1, C], 0.3); t(f + 'key.weight', [F, C]); t(f + 'value.weight', [C, F]);
  }
  t('ln_out.weight', [C], 1); t('ln_out.bias', [C]); t('head.weight', [V, C]);
  // write safetensors (F32)
  let off = 0; const header = {};
  for (const [k, v] of Object.entries(T)) { header[k] = { dtype: 'F32', shape: v.shape, data_offsets: [off, off + v.d.length * 4] }; off += v.d.length * 4; }
  let hs = JSON.stringify(header); while ((8 + hs.length) % 8) hs += ' ';
  const buf = Buffer.alloc(8 + hs.length + off); buf.writeBigUInt64LE(BigInt(hs.length), 0); buf.write(hs, 8, 'utf8');
  let p = 8 + hs.length; for (const v of Object.values(T)) { Buffer.from(v.d.buffer).copy(buf, p); p += v.d.length * 4; }
  fs.writeFileSync(file, buf);
}

const file = process.argv[2] || path.join(os.tmpdir(), 'brymar-rwkv7-mini.safetensors');
if (!process.argv[2]) buildMini(file);

// ── read the checkpoint's tensor list ourselves ───────────────────────────
const raw = fs.readFileSync(file), hlen = Number(raw.readBigUInt64LE(0)), header = JSON.parse(raw.toString('utf8', 8, 8 + hlen)), base = 8 + hlen;
const tensors = Object.entries(header).filter(([k]) => k !== '__metadata__');
const readF32 = (meta) => { const [s, e] = meta.data_offsets; if (meta.dtype !== 'F32') return null; return new Float32Array(raw.buffer.slice(raw.byteOffset + base + s, raw.byteOffset + base + e)); };

// ── run Brymar's real loader (quiet terminal) ─────────────────────────────
const logs = [], terminal = { log: (m) => logs.push(m) };
const model = await loadModel({ modelName: file, loraRank: 4, loraAlpha: 8, trainingMode: 'full' }, terminal);
console.log('Brymar says:', logs.find(l => l.startsWith('Applied')) || '(no "Applied" line)');

// which checkpoint tensors ended up inside the model? (exact array match)
const modelArrays = model.getAllWeights();
const same = (a, b) => { if (!a || !b || a.length !== b.length) return false; for (let i = 0; i < a.length; i += Math.max(1, Math.floor(a.length / 64))) if (a[i] !== b[i]) return false; return true; };
const landed = [], skipped = [];
for (const [k, meta] of tensors) {
  const d = readF32(meta); const n = meta.shape.reduce((a, b) => a * b, 1);
  const where = modelArrays.find(w => same(w.data, d));
  if (where) landed.push(`${k} → ${where.name}`); else skipped.push(`${k} [${meta.shape.join('×')}]`);
}
const randomLeft = modelArrays.filter(w => !tensors.some(([, meta]) => same(w.data, readF32(meta)))).map(w => `${w.name} [${w.rows}×${w.cols}]`);
const blockOf = s => (s.match(/blocks?\.(\d+)\./) || [])[1];
console.log(`\ncheckpoint tensors: ${tensors.length}   landed in the model: ${landed.length}   NOT used: ${skipped.length}`);
console.log('\nNOT used (block 1 shown; block 0 is the same plus ln0):');
for (const s of skipped.filter(s => blockOf(s) === '1' || blockOf(s) === undefined)) console.log('  ' + s);
console.log(`\nBrymar parameters left at RANDOM initialisation (${randomLeft.length} of ${modelArrays.length}; block 1 shown):`);
for (const s of randomLeft.filter(s => blockOf(s) === '1' || blockOf(s) === undefined)) console.log('  ' + s);

// ── what ternary rounding does to the matrices that DID load ─────────────
let worst = 0, sum = 0, cnt = 0;
for (const w of modelArrays) {
  if (w.rows < 2 || !landed.some(l => l.endsWith('→ ' + w.name))) continue;
  const packed = new Uint8Array(Math.ceil(w.data.length / 4)), g = TMAC.quantizeWeights(w.data, packed);
  let e = 0, n = 0;
  for (let i = 0; i < w.data.length; i++) { const q = Math.max(-1, Math.min(1, Math.round(w.data[i] / g))) * g; e += (w.data[i] - q) ** 2; n += w.data[i] ** 2; }
  const rel = Math.sqrt(e / n); worst = Math.max(worst, rel); sum += rel; cnt++;
}
console.log(`\nternary rounding of the loaded matrices: relative error ${(100 * sum / cnt).toFixed(0)}% on average (worst ${(100 * worst).toFixed(0)}%)` +
  (process.argv[2] ? '' : '\n  (miniature checkpoint: Gaussian weights; pass a real .safetensors to measure real ones)'));
if (!process.argv[2]) fs.unlinkSync(file);
