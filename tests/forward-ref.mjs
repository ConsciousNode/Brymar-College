// Does model/rwkv7.js compute exactly what the official RWKV-7 computes?
// python3 tests/ref/rwkv7_ref.py <dir> writes a random checkpoint in official format + the reference logits
// (numpy transcription of BlinkDL's rwkv/model.py). This loads the same checkpoint into RWKV7 and compares.
// usage: node tests/forward-ref.mjs <dir>     (float64; expect max |Δlogit| ~1e-12)
import fs from 'node:fs';
import path from 'node:path';
import { RWKV7 } from '../model/rwkv7.js';
import { readSafetensors, configFromTensors, loadIntoRWKV7 } from '../model/rwkv7_load.js';

const dir = process.argv[2];
if (!dir) { console.error('usage: node tests/forward-ref.mjs <dir written by tests/ref/rwkv7_ref.py>'); process.exit(2); }
const exp = JSON.parse(fs.readFileSync(path.join(dir, 'expected.json'), 'utf8'));
const tensors = readSafetensors(path.join(dir, 'mini.safetensors'));
const cfg = configFromTensors(tensors);
const m = new RWKV7(cfg, { Float: Float64Array });
const rep = loadIntoRWKV7(m, tensors);
console.log(`config: dim ${cfg.dim}, heads ${cfg.dim / cfg.headSize}×${cfg.headSize}, layers ${cfg.numLayers}, vocab ${cfg.vocabSize} · loaded ${rep.loaded}/${rep.total} tensors, ${rep.unused.length} unused, ${rep.missing.length} model params missing from file`);
let st = m.getEmptyState(), maxd = 0;
exp.tokens.forEach((t, i) => { const r = m.forward(t, st); st = r.state; for (let j = 0; j < r.logits.length; j++) maxd = Math.max(maxd, Math.abs(r.logits[j] - exp.logits[i][j])); });
const ok = maxd < 1e-9 && rep.unused.length === 0 && rep.missing.length === 0;
console.log(`max |logit − official reference| over ${exp.tokens.length} tokens: ${maxd.toExponential(2)}  ${ok ? 'MATCH' : 'MISMATCH'}`);
process.exit(ok ? 0 : 1);
