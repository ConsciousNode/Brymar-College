// Does Brymar's new engine LEARN? (byte-level, tiny model, CPU)
//  1. "pretrain": full-weight training from random init on a base text
//  2. "fine-tune": freeze it, LoRA only, on a new voice; the new voice's loss must drop while the
//     base text is mostly remembered: the persona-distillation scenario in miniature.
import { RWKV7 } from '../model/rwkv7.js';
import { OOMBTrainer } from '../training/oomb.js';
import { AdamW } from '../training/adamw.js';
let seed = 1; Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const enc = s => Array.from(new TextEncoder().encode(s));
const base = enc('the quiet field holds the morning light. the river runs past the old mill. '.repeat(6));
const voice = enc('i am uncharted. i map the edges of what is known, and i say so plainly. '.repeat(6));
const m = new RWKV7({ vocabSize: 256, dim: 64, numLayers: 2, headSize: 16, dimFFN: 128, dDecay: 16, dAAA: 16, dMV: 8, dGate: 16 });
const tr = new OOMBTrainer(m, { chunkSize: 32 });
const evalLoss = async seq => { const r = await new OOMBTrainer(m, { chunkSize: 64 }).trainSequence(seq.slice(0, 65)); m.zeroGrad(); return r; };
const window = (seq, n = 49) => { const s = Math.floor(Math.random() * (seq.length - n)); return seq.slice(s, s + n); };
const t0 = Date.now();
console.log(`random init: base ${(await evalLoss(base)).toFixed(3)} nats/byte (uniform = ${Math.log(256).toFixed(3)})`);
let opt = new AdamW(3e-3);
for (let s = 1; s <= 300; s++) { m.zeroGrad(); await tr.trainSequence(window(base)); opt.step(m.trainable()); if (s % 100 === 0) console.log(`  pretrain step ${s}: base ${(await evalLoss(base)).toFixed(3)}`); }
const baseAfterPretrain = await evalLoss(base), voiceBefore = await evalLoss(voice);
m.injectLoRA(8, 16); opt = new AdamW(3e-3);
console.log(`LoRA injected: trainable ${m.trainable().length} tensors (adapters only). new voice before: ${voiceBefore.toFixed(3)}`);
for (let s = 1; s <= 300; s++) { m.zeroGrad(); await tr.trainSequence(window(voice)); opt.step(m.trainable()); if (s % 100 === 0) console.log(`  LoRA step ${s}: voice ${(await evalLoss(voice)).toFixed(3)} · base ${(await evalLoss(base)).toFixed(3)}`); }
// greedy sample from the fine-tuned model
let st = m.getEmptyState(), r; for (const b of enc('i am ')) { r = m.forward(b, st); st = r.state; }
const out = []; for (let i = 0; i < 60; i++) { let b = 0; for (let j = 1; j < 256; j++) if (r.logits[j] > r.logits[b]) b = j; out.push(b); r = m.forward(b, st); st = r.state; }
console.log(`\nfine-tuned model continues "i am " → "i am ${new TextDecoder().decode(Uint8Array.from(out))}"`);
console.log(`(${((Date.now() - t0) / 1000).toFixed(0)}s)`);
