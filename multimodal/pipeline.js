// multimodal/pipeline.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// MultimodalPipeline — Phase 4
//
// Orchestrates text + visual + audio modalities into a unified token sequence
// for the RWKV-v7 model. Each modality is encoded and injected as additional
// tokens prepended to the text token stream.
//
// Architecture:
//   Text   → NanoTokenizer BPE IDs
//   Visual → ElasticTok → patch features → flattened to pseudo-token IDs
//   Audio  → SpikeVox → rate-coded features → pseudo-token IDs
//
// Pseudo-token IDs for visual/audio occupy the range [vocabSize, vocabSize + k).
// The model's embedding lookup is extended lazily to cover these IDs when
// the pipeline runs in multimodal mode.
//
// Modality separator tokens:
//   <|img|>   = vocabSize + 0
//   <|audio|> = vocabSize + 1
//   <|end|>   = vocabSize + 2

import { NanoTokenizer } from '../tokenizer/bpe.js';
import { ElasticTok }    from './elastictok.js';
import { SpikeVox }      from './spikevox.js';

export const MODALITY_TOKENS = {
  IMG:   0,   // Added to vocabSize
  AUDIO: 1,
  END:   2
};

/**
 * MultimodalPipeline: text + visual + audio → unified token stream.
 */
export class MultimodalPipeline {
  /**
   * @param {NanoTokenizer} tokenizer  - Trained BPE tokenizer
   * @param {ElasticTok}    elastictok - Visual patch tokenizer
   * @param {SpikeVox}      spikevox   - LIF audio encoder
   * @param {number}        [baseVocabSize=256]
   */
  constructor(tokenizer, elastictok, spikevox, baseVocabSize = 256) {
    this.tokenizer     = tokenizer;
    this.elastictok    = elastictok;
    this.spikevox      = spikevox;
    this.baseVocabSize = baseVocabSize;

    // Spike rate coding accumulators (per-neuron EMA)
    this._spikeAccum = null;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Build a unified token sequence from text + optional visual/audio inputs.
   *
   * @param {string}   text           - Text segment to encode
   * @param {Array<{pixels: Uint8Array, width: number, height: number}>} [images]
   *   Each entry is an RGBA pixel buffer with dimensions.
   * @param {Array<Float32Array>} [audioBuffers]
   *   Each entry is a PCM buffer ∈ [-1, 1].
   * @returns {Promise<number[]>} Unified token ID sequence
   */
  async buildSequence(text, images = [], audioBuffers = []) {
    const tokens = [];
    const V = this.baseVocabSize;

    // ── Visual tokens ──────────────────────────────────────────────────────
    for (const img of images) {
      const result = this.elastictok.processFrame(
        img.pixels, img.width, img.height
      );
      if (result) {
        tokens.push(V + MODALITY_TOKENS.IMG);
        const patchTokens = this._quantizePatchFeatures(result.patchFeatures, result.nPatches, V + 3);
        tokens.push(...patchTokens);
        tokens.push(V + MODALITY_TOKENS.END);
      }
    }

    // ── Audio tokens ───────────────────────────────────────────────────────
    for (const pcm of audioBuffers) {
      const { spikes, dormant } = this.spikevox.processSamples(pcm);
      if (!dormant) {
        tokens.push(V + MODALITY_TOKENS.AUDIO);
        if (!this._spikeAccum || this._spikeAccum.length !== this.spikevox.dim) {
          this._spikeAccum = new Float32Array(this.spikevox.dim);
        }
        const dense = this.spikevox.spikesToFeatures(spikes, this._spikeAccum);
        const audioTokens = this._quantizeDenseFeatures(dense, V + 64);
        tokens.push(...audioTokens);
        tokens.push(V + MODALITY_TOKENS.END);
      }
    }

    // ── Text tokens ────────────────────────────────────────────────────────
    if (this.tokenizer.trained) {
      tokens.push(...this.tokenizer.encode(text));
    } else {
      // Byte-level fallback
      tokens.push(...Array.from(new TextEncoder().encode(text)));
    }

    return tokens;
  }

  /**
   * Run a validation pass and return a summary object.
   *
   * Creates a synthetic image and audio buffer to verify the pipeline
   * produces non-empty token sequences.
   *
   * @param {string} [testText='Hello, ConsciousNode.']
   * @returns {Promise<{ totalTokens: number, text: number, visual: number, audio: number }>}
   */
  async validate(testText = 'Hello, ConsciousNode.') {
    const V = this.baseVocabSize;

    // Synthetic 64×64 RGBA image (random noise)
    const W = 64, H = 64;
    const pixels = new Uint8Array(W * H * 4);
    for (let i = 0; i < pixels.length; i++) pixels[i] = Math.floor(Math.random() * 256);

    // Synthetic audio PCM (440 Hz sine wave at 8 kHz, 1 second)
    const sampleRate = 8000;
    const pcm = new Float32Array(sampleRate);
    for (let i = 0; i < sampleRate; i++) {
      pcm[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate);
    }

    const tokens = await this.buildSequence(
      testText,
      [{ pixels, width: W, height: H }],
      [pcm]
    );

    const textCount   = tokens.filter(t => t < V).length;
    const visualCount = tokens.filter(t => t >= V + 3 && t < V + 64).length;
    const audioCount  = tokens.filter(t => t >= V + 64).length;

    return { totalTokens: tokens.length, text: textCount, visual: visualCount, audio: audioCount };
  }

  // ── Private Helpers ───────────────────────────────────────────────────────

  /**
   * Quantize patch RGB features into pseudo-token IDs.
   *
   * Bins each float ∈ [0, 1] into N discrete levels and maps to
   * IDs ∈ [tokenBase, tokenBase + N*3).
   *
   * @param {Float32Array} patchFeatures - [nPatches * 3]
   * @param {number}       nPatches
   * @param {number}       tokenBase     - First pseudo-token ID to use
   * @param {number}       [bins=32]     - Discretization levels
   * @returns {number[]}
   */
  _quantizePatchFeatures(patchFeatures, nPatches, tokenBase, bins = 32) {
    const tokens = [];
    for (let i = 0; i < nPatches; i++) {
      for (let c = 0; c < 3; c++) {
        const val = patchFeatures[i * 3 + c];
        const bin = Math.min(bins - 1, Math.floor(val * bins));
        tokens.push(tokenBase + c * bins + bin);
      }
    }
    return tokens;
  }

  /**
   * Quantize a dense feature vector ∈ [0, 1] into pseudo-token IDs.
   *
   * @param {Float32Array} features
   * @param {number}       tokenBase
   * @param {number}       [bins=32]
   * @returns {number[]}
   */
  _quantizeDenseFeatures(features, tokenBase, bins = 32) {
    const tokens = [];
    for (let i = 0; i < features.length; i++) {
      const bin = Math.min(bins - 1, Math.floor(features[i] * bins));
      tokens.push(tokenBase + i * bins + bin);
    }
    return tokens;
  }
}
