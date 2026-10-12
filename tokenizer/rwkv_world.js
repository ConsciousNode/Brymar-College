// tokenizer/rwkv_world.js — the official RWKV "World" tokenizer (rwkv_vocab_v20230424), in JS
// MIT License — ConsciousNode SoftWorks (vocabulary: BlinkDL/RWKV-LM, Apache-2.0)
//
// A pretrained RWKV World model only understands ITS OWN token ids. Fine-tuning one on text tokenized
// by any other tokenizer feeds it gibberish. Greedy longest-match over UTF-8 bytes, exactly like
// BlinkDL's TRIE_TOKENIZER (tests/tokenizer-ref.py checks it against the Python original).
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export class RWKVWorldTokenizer {
  constructor(vocabFile = join(dirname(fileURLToPath(import.meta.url)), 'rwkv_world_vocab.json')) {
    const { tokens } = JSON.parse(readFileSync(vocabFile, 'utf8'));
    this.size = tokens.length;                      // 65536
    this.idx2bytes = tokens.map(h => h ? Uint8Array.from(h.match(/../g).map(b => parseInt(b, 16))) : null);
    this.root = { to: new Map(), id: -1 };
    this.idx2bytes.forEach((bytes, id) => {
      if (!bytes) return;
      let node = this.root;
      for (const b of bytes) { let nx = node.to.get(b); if (!nx) { nx = { to: new Map(), id: -1 }; node.to.set(b, nx); } node = nx; }
      node.id = id;
    });
  }
  encodeBytes(src) {
    const out = []; let i = 0;
    while (i < src.length) {
      let node = this.root, j = i, bestId = -1, bestEnd = i;
      while (j < src.length) { const nx = node.to.get(src[j]); if (!nx) break; node = nx; j++; if (node.id >= 0) { bestId = node.id; bestEnd = j; } }
      if (bestId < 0) throw new Error(`byte ${src[i]} at ${i} has no token`);
      out.push(bestId); i = bestEnd;
    }
    return out;
  }
  encode(text) { return this.encodeBytes(new TextEncoder().encode(text)); }
  decodeBytes(ids) { const parts = ids.map(id => this.idx2bytes[id] || new Uint8Array(0)), n = parts.reduce((a, p) => a + p.length, 0), out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out; }
  decode(ids) { return new TextDecoder('utf-8', { fatal: false }).decode(this.decodeBytes(ids)); }
}
