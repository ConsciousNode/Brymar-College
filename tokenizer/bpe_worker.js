// tokenizer/bpe_worker.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// BPE training Worker Thread script — Phase 4
//
// Runs inside a Node.js worker_threads Worker.
// Receives: { text: string, targetVocab: number }
// Posts back:
//   { type: 'progress', pct: number, warn?: string }
//   { type: 'done', vocab: Object, merges: Array, actualVocab: number }
//   { type: 'error', msg: string }
//
// Ported from HTMLNLM.html bpeScript (browser inline Worker),
// adapted for Node.js worker_threads parentPort API.
//
// Reference: HTMLNLM.html — const bpeScript (lines 1053-1108)

import { parentPort } from 'node:worker_threads';

const CAP = 8_000_000; // 8 MB corpus cap

parentPort.on('message', async (data) => {
  try {
    const { text, targetVocab } = data;

    // Corpus size guard
    const truncated = text.length > CAP;
    const src = truncated ? text.substring(0, CAP) : text;
    if (truncated) {
      parentPort.postMessage({ type: 'progress', pct: 0, warn: 'Corpus truncated to 8MB.' });
    }

    // Byte-encode source text
    const bytes = new TextEncoder().encode(src);
    let tokens = new Uint32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) tokens[i] = bytes[i];

    // Seed vocabulary: 256 byte literals
    const vocab = {};
    for (let i = 0; i < 256; i++) vocab[i] = String.fromCharCode(i);
    const merges = [];
    let nextId = 256;

    // BPE merge loop
    const numMerges = targetVocab - 256;
    for (let m = 0; m < numMerges; m++) {
      if (tokens.length < 2) break;

      // Count bigram frequencies
      const freq = new Map();
      for (let i = 0; i < tokens.length - 1; i++) {
        const key = tokens[i] * 65536 + tokens[i + 1];
        freq.set(key, (freq.get(key) || 0) + 1);
      }

      // Find most frequent pair
      let bestKey = -1, maxCount = 0;
      for (const [k, c] of freq) {
        if (c > maxCount) { maxCount = c; bestKey = k; }
      }
      if (maxCount < 2) break; // Nothing left to merge

      const left  = Math.floor(bestKey / 65536);
      const right = bestKey % 65536;
      const newId = nextId++;
      vocab[newId] = (vocab[left] || '') + (vocab[right] || '');
      merges.push([bestKey, newId]);

      // Apply merge across token sequence
      const nt = new Uint32Array(tokens.length);
      let j = 0, i = 0;
      while (i < tokens.length) {
        if (i < tokens.length - 1 && tokens[i] === left && tokens[i + 1] === right) {
          nt[j++] = newId;
          i += 2;
        } else {
          nt[j++] = tokens[i++];
        }
      }
      tokens = nt.slice(0, j);

      // Report progress every 50 merges
      if (m % 50 === 0) {
        parentPort.postMessage({ type: 'progress', pct: m / numMerges });
        // Yield to event loop
        await new Promise(r => setImmediate(r));
      }
    }

    parentPort.postMessage({ type: 'done', vocab, merges, actualVocab: nextId });
  } catch (err) {
    parentPort.postMessage({ type: 'error', msg: err.message });
  }
});
