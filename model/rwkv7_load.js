// model/rwkv7_load.js — read official RWKV-7 checkpoints into model/rwkv7.js (names map 1:1)
// MIT License — ConsciousNode SoftWorks
import fs from 'node:fs';

/** Parse a .safetensors file: { name: { shape, data: Float32Array } } (F32 / F16 / BF16). */
export function readSafetensors(file) {
  const buf = fs.readFileSync(file), hlen = Number(buf.readBigUInt64LE(0)), header = JSON.parse(buf.toString('utf8', 8, 8 + hlen)), base = 8 + hlen, out = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = meta.data_offsets, bytes = buf.subarray(base + s, base + e);
    let data;
    if (meta.dtype === 'F32') data = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
    else if (meta.dtype === 'BF16' || meta.dtype === 'F16') {
      const u16 = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length)); data = new Float32Array(u16.length);
      if (meta.dtype === 'BF16') { const u32 = new Uint32Array(data.buffer); for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16; }
      else for (let i = 0; i < u16.length; i++) { const h = u16[i], sg = h & 0x8000 ? -1 : 1, ex = (h >> 10) & 31, f = h & 1023;
        data[i] = ex === 0 ? sg * 2 ** -14 * (f / 1024) : ex === 31 ? (f ? NaN : sg * Infinity) : sg * 2 ** (ex - 15) * (1 + f / 1024); }
    } else throw new Error(`unsupported dtype ${meta.dtype} for ${name}`);
    out[name] = { shape: meta.shape, data };
  }
  return out;
}

/** Infer RWKV7 constructor config from official tensor shapes. */
export function configFromTensors(t) {
  const need = n => { if (!t[n]) throw new Error(`not an RWKV-7 (x070) checkpoint: missing ${n}`); return t[n].shape; };
  const [V, C] = need('emb.weight'), [H, N] = need('blocks.0.att.r_k');
  let L = 0; for (const k of Object.keys(t)) { const m = k.match(/^blocks\.(\d+)\./); if (m) L = Math.max(L, +m[1] + 1); }
  const sz = s => s[s.length - 1];
  return { vocabSize: V, dim: C, numLayers: L, headSize: N, dimFFN: need('blocks.0.ffn.key.weight')[0],
    dDecay: sz(need('blocks.0.att.w1')), dAAA: sz(need('blocks.0.att.a1')), dMV: L > 1 ? sz(need('blocks.1.att.v1')) : 32, dGate: sz(need('blocks.0.att.g1')) };
}

/** Copy tensors into the model. Returns a full report; nothing is skipped silently. */
export function loadIntoRWKV7(model, t) {
  const unused = [], wrongSize = [];
  let loaded = 0;
  for (const [name, { data }] of Object.entries(t)) {
    const dst = model.p[name];
    if (!dst) { unused.push(name); continue; }
    if (dst.length !== data.length) { wrongSize.push(`${name}: file ${data.length} vs model ${dst.length}`); continue; }
    dst.set(data); loaded++;
  }
  const missing = Object.keys(model.p).filter(n => !t[n]);
  if (wrongSize.length) throw new Error('size mismatch:\n  ' + wrongSize.join('\n  '));
  return { loaded, total: Object.keys(t).length, unused, missing };
}

/** Write named Float32 tensors as .safetensors (F32). tensors: [{ name, data, shape }] */
export function writeSafetensors(file, tensors, metadata = {}) {
  const header = { __metadata__: Object.fromEntries(Object.entries(metadata).map(([k, v]) => [k, String(v)])) };
  let off = 0;
  for (const t of tensors) { const n = t.data.length * 4; header[t.name] = { dtype: 'F32', shape: t.shape, data_offsets: [off, off + n] }; off += n; }
  let hs = Buffer.from(JSON.stringify(header), 'utf8'); const pad = (8 - (hs.length % 8)) % 8; if (pad) hs = Buffer.concat([hs, Buffer.alloc(pad, 0x20)]);
  const fd = fs.openSync(file, 'w'), len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hs.length)); fs.writeSync(fd, len); fs.writeSync(fd, hs);
  for (const t of tensors) { const f32 = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data); fs.writeSync(fd, Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength)); }
  fs.closeSync(fd);
}

/** All model parameters with official names/shapes (for full checkpoints and .rofl export). */
export function modelTensors(model) { return Object.entries(model.p).map(([name, data]) => ({ name, data, shape: model.shape[name] })); }

/**
 * Resolve a model source and build an RWKV7 from it.
 *   - local .safetensors: read directly
 *   - local .pth / .bin: converted to .safetensors with scripts/convert_pth.py (Python 3, no PyTorch), then read
 *   - anything else: treated as a Hugging Face repo id with a model.safetensors
 * Logs exactly what loaded, what was unused and what is missing. Nothing is skipped silently.
 */
export async function loadRWKV7From(source, log = console.log, { RWKV7 }) {
  let file = source;
  if (fs.existsSync(source) && /\.(pth|bin)$/i.test(source)) {
    const { execFileSync } = await import('node:child_process'), os = await import('node:os'), path = await import('node:path');
    file = path.join(os.tmpdir(), `brymar_${Date.now()}.safetensors`);
    log(`Converting ${source} → safetensors (scripts/convert_pth.py)…`, 'info');
    execFileSync('python3', [new URL('../scripts/convert_pth.py', import.meta.url).pathname, source, file], { stdio: 'pipe' });
  } else if (!fs.existsSync(source)) {
    log(`"${source}" is not a local file; trying Hugging Face (${source}/model.safetensors)…`, 'info');
    const { downloadFile } = await import('@huggingface/hub');
    const blob = await downloadFile({ repo: source, path: 'model.safetensors' });
    if (!blob) throw new Error(`could not download ${source}/model.safetensors. Official RWKV-7 weights are .pth files: download one from huggingface.co/BlinkDL/rwkv-7-world and pass --model path/to/file.pth`);
    file = (await import('node:path')).join((await import('node:os')).tmpdir(), `brymar_dl_${Date.now()}.safetensors`);
    fs.writeFileSync(file, Buffer.from(await blob.arrayBuffer()));
  }
  const t = readSafetensors(file), cfg = configFromTensors(t), model = new RWKV7(cfg), rep = loadIntoRWKV7(model, t);
  if (file !== source && /brymar_/.test(file)) { try { fs.unlinkSync(file); } catch {} }
  const nParams = Object.values(model.p).reduce((a, w) => a + w.length, 0);
  log(`RWKV-7 loaded: ${(nParams / 1e6).toFixed(1)}M params · dim ${cfg.dim} · ${cfg.numLayers} layers · ${cfg.dim / cfg.headSize} heads × ${cfg.headSize} · vocab ${cfg.vocabSize}`, 'ok');
  log(`  tensors: ${rep.loaded}/${rep.total} loaded` + (rep.unused.length ? ` · ${rep.unused.length} UNUSED: ${rep.unused.slice(0, 6).join(', ')}${rep.unused.length > 6 ? '…' : ''}` : '') + (rep.missing.length ? ` · ${rep.missing.length} MISSING (left at init): ${rep.missing.slice(0, 6).join(', ')}` : ''), rep.unused.length || rep.missing.length ? 'warn' : 'ok');
  return { model, cfg, report: rep };
}
