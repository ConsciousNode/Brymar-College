// tokenizer/bpe.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// NanoTokenizer — BPE tokenizer using Worker Thread — Phase 4
//
// Train: spawns tokenizer/bpe_worker.js in a Worker Thread for non-blocking BPE.
// Encode/Decode: greedy merge pass matching HTMLNLM.html NanoTokenizer.encode().
//
// Reference: HTMLNLM.html — class NanoTokenizer (lines 1110-1166)

import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKER_PATH = join(__dirname, 'bpe_worker.js');

/**
 * NanoTokenizer: BPE tokenizer for Brymar College.
 *
 * API mirrors the HTMLNLM.html browser version exactly so that checkpoints
 * produced here can be exchanged with the browser app.
 */
export class NanoTokenizer {
  constructor() {
    /** @type {Object<number, string>} id → string */
    this.vocab = {};
    /** @type {Array<[number, number]>} [bigram_key, new_id] */
    this.merges = [];
    /** @type {number} Actual vocabulary size after training */
    this.size = 0;
    /** @type {boolean} */
    this.trained = false;
  }

  // ── Training ──────────────────────────────────────────────────────────────

  /**
   * Train BPE on text corpus using a Worker Thread.
   *
   * @param {string} text        - Raw UTF-8 corpus text
   * @param {number} targetVocab - Target vocabulary size (≥ 256)
   * @param {Function} [onProgress] - Optional callback({ pct, warn? })
   * @returns {Promise<void>}
   */
  async train(text, targetVocab, onProgress) {
    return new Promise((resolve, reject) => {
      const worker = new Worker(WORKER_PATH, {
        // Allow the worker script to use ESM imports
        execArgv: []
      });

      worker.on('message', (msg) => {
        if (msg.type === 'progress') {
          if (onProgress) onProgress(msg);
        } else if (msg.type === 'error') {
          worker.terminate();
          reject(new Error(msg.msg));
        } else if (msg.type === 'done') {
          this.vocab   = msg.vocab;
          this.merges  = msg.merges;
          this.size    = msg.actualVocab;
          this.trained = true;
          worker.terminate();
          resolve();
        }
      });

      worker.on('error', reject);
      worker.postMessage({ text, targetVocab });
    });
  }

  // ── Encode ────────────────────────────────────────────────────────────────

  /**
   * Encode a string to token IDs using trained BPE merges.
   *
   * @param {string} text
   * @returns {number[]} Token ID sequence
   */
  encode(text) {
    if (!this.trained) return [];

    // Byte-encode text → initial token sequence
    let tokens = Array.from(new TextEncoder().encode(text));

    // Build merge lookup map
    const mergeMap = new Map(this.merges);

    // Greedy left-to-right merge pass
    let changed = true;
    while (changed) {
      changed = false;
      const res = [];
      for (let i = 0; i < tokens.length; i++) {
        if (i < tokens.length - 1) {
          const key = tokens[i] * 65536 + tokens[i + 1];
          if (mergeMap.has(key)) {
            res.push(mergeMap.get(key));
            i++;
            changed = true;
            continue;
          }
        }
        res.push(tokens[i]);
      }
      tokens = res;
    }

    return tokens;
  }

  // ── Decode ────────────────────────────────────────────────────────────────

  /**
   * Decode token IDs back to string.
   *
   * @param {number[]} ids
   * @returns {string}
   */
  decode(ids) {
    const vocabMap = new Map(
      Object.entries(this.vocab).map(([k, v]) => [parseInt(k), v])
    );
    return ids.map(id => vocabMap.get(id) ?? '\uFFFD').join('');
  }

  // ── Serialization ─────────────────────────────────────────────────────────

  /**
   * Serialize tokenizer state to a plain JSON-safe object.
   *
   * @returns {Object}
   */
  toJSON() {
    return {
      version: 'nanotokenizer-v1',
      size:    this.size,
      vocab:   this.vocab,
      merges:  this.merges
    };
  }

  /**
   * Restore tokenizer from a serialized object.
   *
   * @param {Object} obj
   */
  fromJSON(obj) {
    if (obj.version !== 'nanotokenizer-v1') {
      throw new Error(`Unknown tokenizer version: ${obj.version}`);
    }
    this.vocab   = obj.vocab;
    this.merges  = obj.merges;
    this.size    = obj.size;
    this.trained = true;
  }
}
