// tokenizer/corpus.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// CorpusLoader — Phase 4 + Phase 5 Multimodal
//
// Loads text, image, and audio files from a corpus directory.
// Text files are concatenated and tokenized via BPE.
// Images are decoded via sharp and tokenized via ElasticTok.
// Audio is decoded via native WAV parser and tokenized via SpikeVox.
//
// Supported formats:
//   Text:  .txt .md .rst .csv .tsv .json .jsonl .xml .html .htm
//          .yaml .yml .toml .log .rtf .tex .org .adoc
//   Image: .png .jpg .jpeg .webp .tiff .tif .gif .avif .svg .bmp
//   Audio: .wav (native), others require FFmpeg

import { readdir, readFile, unlink } from 'node:fs/promises';
import { join, extname } from 'node:path';
import { tmpdir } from 'node:os';
import { NanoTokenizer } from './bpe.js';
import { CORPUS_EXTENSIONS } from '../src/config.js';

// Lazy-loaded dependencies
let sharp = null;
let parseWAV = null;
let ElasticTok = null;
let SpikeVox = null;

/**
 * CorpusLoader: Multimodal disk-backed corpus for Brymar College training.
 *
 * Usage:
 *   const loader = new CorpusLoader('./corpus/', terminal);
 *   const tokens = await loader.load({ vocabSize: 2048 });
 *   while (true) {
 *     const seq = loader.nextSequence(512);
 *     // ...train on seq...
 *   }
 */
export class CorpusLoader {
  /**
   * @param {string}   corpusDir          - Directory containing corpus files
   * @param {import('../src/ui/terminal.js').Terminal} [terminal]
   */
  constructor(corpusDir, terminal = null, opts = {}) {
    this.dir      = corpusDir;
    this.t        = terminal;
    this.opts     = opts;
    /** @type {number[] | null} Flat token ID sequence after load() */
    this.tokens   = null;
    /** @type {number} Read cursor */
    this.cursor   = 0;
    /** @type {number} Total tokens loaded */
    this.totalTokens = 0;
    /** @type {NanoTokenizer} */
    this.tokenizer = new NanoTokenizer();

    // File counters for reporting
    this._stats = { text: 0, image: 0, audio: 0, skipped: 0 };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Load corpus from disk, train BPE if needed, and return token array.
   *
   * Scans the corpus directory for all supported file types and processes
   * them by modality:
   *   - Text → BPE tokenization
   *   - Image → ElasticTok patch features → pseudo-token IDs
   *   - Audio → SpikeVox LIF encoding → pseudo-token IDs
   *
   * @param {Object} [opts]
   * @param {number} [opts.vocabSize=2048]    - BPE vocabulary size target
   * @param {boolean} [opts.byteLevel=false]  - Skip BPE, use raw bytes
   * @returns {Promise<number[]>} Flat token ID array
   */
  async load(opts = {}) {
    const vocabSize = opts.vocabSize ?? 2048;
    const byteLevel = opts.byteLevel ?? false;

    const log = (msg, level = 'info') =>
      this.t ? this.t.log(msg, level) : console.log(`[${level.toUpperCase()}] ${msg}`);

    // ── Scan directory ──────────────────────────────────────────────────────
    const { textFiles, imageFiles, audioFiles, videoFiles, skippedFiles } = await this._scanDir(this.dir);

    this._stats = {
      text: textFiles.length,
      image: imageFiles.length,
      audio: audioFiles.length,
      video: videoFiles.length,
      skipped: skippedFiles.length,
    };

    log(`Corpus scan: ${textFiles.length} text, ${imageFiles.length} image, ${audioFiles.length} audio, ${videoFiles.length} video files`, 'info');
    if (skippedFiles.length > 0) {
      log(`  ${skippedFiles.length} files skipped (unsupported format)`, 'warn');
    }

    // ── Process text files ──────────────────────────────────────────────────
    const rawText = await this._readTextFiles(textFiles);
    if (!rawText && imageFiles.length === 0 && audioFiles.length === 0) {
      throw new Error(`Corpus directory not found or empty: ${this.dir}`);
    }

    let tokenIds = [];

    if (rawText) {
      log(`Text corpus: ${rawText.length.toLocaleString()} chars from ${textFiles.length} files`, 'info');

      if (byteLevel) {
        tokenIds = Array.from(new TextEncoder().encode(rawText));
        this.tokenizer.trained = false;
        log(`Byte-level encoding: ${tokenIds.length.toLocaleString()} tokens`, 'ok');
      } else {
        log(`Training BPE (vocab=${vocabSize})…`, 'info');
        let lastPct = -1;
        await this.tokenizer.train(rawText, vocabSize, (msg) => {
          if (msg.warn) log(msg.warn, 'warn');
          if (msg.pct !== undefined) {
            const currentPct = Math.round(msg.pct * 100);
            if (currentPct !== lastPct && currentPct % 10 === 0) {
              log(`  BPE progress: ${currentPct}%`, 'info');
              lastPct = currentPct;
            }
          }
        });
        log(`BPE trained. Actual vocab: ${this.tokenizer.size}`, 'ok');
        tokenIds = this.tokenizer.encode(rawText);
        log(`Encoded text corpus: ${tokenIds.length.toLocaleString()} tokens`, 'ok');
      }
    }

    // ── Process image files ─────────────────────────────────────────────────
    if (imageFiles.length > 0) {
      const imageTokens = await this._processImages(imageFiles, log);
      tokenIds = tokenIds.concat(imageTokens);
    }

    // ── Process video frames ────────────────────────────────────────────────
    if (videoFiles.length > 0) {
      const videoTokens = await this._processVideoFrames(videoFiles, log);
      tokenIds = tokenIds.concat(videoTokens);
    }

    // ── Process audio files ─────────────────────────────────────────────────
    if (audioFiles.length > 0) {
      const audioTokens = await this._processAudio(audioFiles, log);
      tokenIds = tokenIds.concat(audioTokens);
    }

    log(`Total corpus: ${tokenIds.length.toLocaleString()} tokens (text + multimodal)`, 'ok');

    this.tokens      = tokenIds;
    this.totalTokens = tokenIds.length;
    this.cursor      = 0;
    return tokenIds;
  }

  /**
   * Get the next fixed-length token sequence from the corpus.
   *
   * Wraps around at end of corpus (ring buffer behavior).
   *
   * @param {number} maxLen - Sequence length (including next-token target)
   * @returns {number[]} Token ID slice of length ≤ maxLen
   */
  nextSequence(maxLen) {
    if (!this.tokens || this.tokens.length < 2) return [];

    const start = this.cursor;
    const end   = Math.min(start + maxLen, this.tokens.length);
    const seq   = this.tokens.slice(start, end);

    // Advance cursor; wrap around
    this.cursor = end >= this.tokens.length ? 0 : end;

    return seq;
  }

  /**
   * Sample a random window of length `maxLen` from the corpus.
   *
   * Used by TrainingLoop for stochastic sampling.
   *
   * @param {number} maxLen
   * @returns {number[]}
   */
  sampleRandom(maxLen) {
    if (!this.tokens || this.tokens.length < 2) return [];
    const maxStart = Math.max(0, this.tokens.length - maxLen - 1);
    const start    = Math.floor(Math.random() * maxStart);
    return this.tokens.slice(start, start + maxLen);
  }

  /** @returns {number} Actual vocabulary size (256 if byte-level) */
  get vocabSize() {
    return this.tokenizer.trained ? this.tokenizer.size : 256;
  }

  /** @returns {Object} File type counts from last scan */
  get stats() {
    return { ...this._stats };
  }

  // ── Private: Directory scanning ────────────────────────────────────────────

  /**
   * Scan a directory and classify files by modality.
   *
   * @param {string} dir
   * @returns {Promise<{ textFiles: string[], imageFiles: string[], audioFiles: string[], skippedFiles: string[] }>}
   */
  async _scanDir(dir) {
    let entries;
    try {
      entries = await readdir(dir);
    } catch {
      return { textFiles: [], imageFiles: [], audioFiles: [], videoFiles: [], skippedFiles: [] };
    }

    const textFiles = [];
    const imageFiles = [];
    const audioFiles = [];
    const videoFiles = [];
    const skippedFiles = [];
    const log = (msg, level = 'info') =>
      this.t ? this.t.log(msg, level) : console.log(`[${level.toUpperCase()}] ${msg}`);

    for (const entry of entries.sort()) {
      const ext = extname(entry).toLowerCase();
      const fullPath = join(dir, entry);

      if (CORPUS_EXTENSIONS.text.has(ext)) {
        textFiles.push(fullPath);
      } else if (CORPUS_EXTENSIONS.image.has(ext)) {
        imageFiles.push(fullPath);
      } else if (CORPUS_EXTENSIONS.audio.has(ext)) {
        audioFiles.push(fullPath);
      } else if (CORPUS_EXTENSIONS.audioRequiresFFmpeg.has(ext)) {
        audioFiles.push(fullPath);
      } else if (CORPUS_EXTENSIONS.videoRequiresFFmpeg && CORPUS_EXTENSIONS.videoRequiresFFmpeg.has(ext)) {
        audioFiles.push(fullPath);
        videoFiles.push(fullPath);
      } else if (ext && !ext.startsWith('.git')) {
        skippedFiles.push(fullPath);
      }
    }

    return { textFiles, imageFiles, audioFiles, videoFiles, skippedFiles };
  }

  // ── Private: Text processing ───────────────────────────────────────────────

  /**
   * Read all text files and concatenate.
   *
   * @param {string[]} files
   * @returns {Promise<string | null>}
   */
  async _readTextFiles(files) {
    if (files.length === 0) return null;

    const parts = [];
    for (const file of files) {
      try {
        const content = await readFile(file, 'utf8');
        const ext = extname(file).toLowerCase();
        
        if (ext === '.json' || ext === '.jsonl') {
          // Native JSON parsing for Claude/ChatGPT exports
          const uName = this.opts.userName || 'USER';
          const aName = this.opts.aiName || 'CLAUDE';
          
          let data;
          try { data = JSON.parse(content); } catch { data = null; }
          
          if (data) {
            let rawConvs = [];
            if (Array.isArray(data)) rawConvs = data;
            else if (data.conversations && Array.isArray(data.conversations)) rawConvs = data.conversations;
            else rawConvs = [data];

            let jsonText = '';
            for (let i = 0; i < rawConvs.length; i++) {
              const conv = rawConvs[i];
              let messages = [];
              if (conv.chat_messages && Array.isArray(conv.chat_messages)) messages = conv.chat_messages;
              else if (conv.messages && Array.isArray(conv.messages)) messages = conv.messages;
              else if (conv.mapping && typeof conv.mapping === 'object') {
                const msgMap = conv.mapping;
                const sortedKeys = Object.keys(msgMap).sort((a, b) => {
                  const timeA = msgMap[a]?.message?.create_time || 0;
                  const timeB = msgMap[b]?.message?.create_time || 0;
                  return timeA - timeB;
                });
                messages = sortedKeys.map(k => msgMap[k].message).filter(m => m && m.content);
              }
              
              if (messages.length === 0) continue;

              messages.forEach(msg => {
                let role = msg.sender || msg.role || msg.author?.role || 'unknown';
                let isUser = role.toLowerCase().includes('user') || role.toLowerCase().includes('human');
                
                let contentField = msg.text || msg.content;
                let textContent = "";
                if (typeof contentField === 'string') textContent = contentField;
                else if (Array.isArray(contentField)) {
                  textContent = contentField.map(item => {
                    if (typeof item === 'string') return item;
                    if (item.text) return item.text;
                    if (item.content && typeof item.content === 'string') return item.content;
                    return "";
                  }).filter(t => t).join("");
                }
                else if (typeof contentField === 'object' && contentField.text) textContent = contentField.text;
                
                if (!textContent.trim()) return;
                jsonText += `[${isUser ? uName : aName}]:\n${textContent}\n\n---\n\n`;
              });
            }
            if (jsonText) parts.push(jsonText);
            else parts.push(content); // Fallback to raw text if no dialogue structure found
          } else {
            parts.push(content);
          }
        } else {
          parts.push(content);
        }
      } catch {
        // Skip unreadable files
      }
    }

    if (parts.length === 0) return null;
    return parts.join('\n\n');
  }

  // ── Private: Image processing ──────────────────────────────────────────────

  /**
   * Process image files via sharp → ElasticTok → pseudo-token IDs.
   *
   * @param {string[]} files
   * @param {Function} log
   * @returns {Promise<number[]>}
   */
  async _processImages(files, log) {
    // Lazy-load sharp
    if (!sharp) {
      try {
        sharp = (await import('sharp')).default;
      } catch (e) {
        log(`sharp not available — skipping ${files.length} image files: ${e.message}`, 'warn');
        return [];
      }
    }

    // Lazy-load ElasticTok
    if (!ElasticTok) {
      try {
        const mod = await import('../multimodal/elastictok.js');
        ElasticTok = mod.ElasticTok;
      } catch (e) {
        log(`ElasticTok not available — skipping images: ${e.message}`, 'warn');
        return [];
      }
    }

    const tok = new ElasticTok(16, 64);
    const tokens = [];
    const V = 256; // base vocab size for pseudo-token offset
    let processed = 0;

    for (const file of files) {
      try {
        const { data, info } = await sharp(file)
          .removeAlpha()
          .raw()
          .toBuffer({ resolveWithObject: true });

        const result = tok.processRGB(data, info.width, info.height);
        if (result) {
          // Add image separator and patch tokens
          tokens.push(V + 0); // <|img|>
          for (let i = 0; i < result.nPatches; i++) {
            for (let c = 0; c < 3; c++) {
              const val = result.patchFeatures[i * 3 + c];
              const bin = Math.min(31, Math.floor(val * 32));
              tokens.push(V + 3 + c * 32 + bin);
            }
          }
          tokens.push(V + 2); // <|end|>
          processed++;
        }
      } catch (e) {
        log(`  Image load failed: ${file} — ${e.message}`, 'warn');
      }
    }

    if (processed > 0) {
      log(`Processed ${processed}/${files.length} images → ${tokens.length} tokens`, 'ok');
    }

    return tokens;
  }

  // ── Private: Video visual processing ───────────────────────────────────────

  /**
   * Extract frames from video files via FFmpeg and process via sharp -> ElasticTok.
   *
   * @param {string[]} files
   * @param {Function} log
   * @returns {Promise<number[]>}
   */
  async _processVideoFrames(files, log) {
    let ffmpeg = null;
    try {
      ffmpeg = (await import('fluent-ffmpeg')).default;
      const ffmpegPath = (await import('@ffmpeg-installer/ffmpeg')).default.path;
      ffmpeg.setFfmpegPath(ffmpegPath);
    } catch (e) {
      log(`FFmpeg not available — skipping video visuals: ${e.message}`, 'warn');
      return [];
    }

    const { mkdir, readdir, rm } = await import('node:fs/promises');
    let allTokens = [];

    for (const file of files) {
      const frameDir = join(tmpdir(), `brymar_frames_${Date.now()}_${Math.random().toString(36).slice(2)}`);
      try {
        await mkdir(frameDir, { recursive: true });
        
        await new Promise((resolve, reject) => {
          ffmpeg(file)
            .outputOptions(['-vf fps=24', '-q:v 2']) // 24 fps as requested
            .output(join(frameDir, 'frame_%04d.jpg'))
            .on('end', resolve)
            .on('error', reject)
            .run();
        });

        // Collect generated frames
        const frames = (await readdir(frameDir))
          .filter(f => f.endsWith('.jpg'))
          .sort()
          .map(f => join(frameDir, f));

        if (frames.length > 0) {
          const tokens = await this._processImages(frames, log);
          allTokens = allTokens.concat(tokens);
        }
      } catch (e) {
        log(`  Video visual extraction failed: ${file} — ${e.message}`, 'warn');
      } finally {
        await rm(frameDir, { recursive: true, force: true }).catch(() => {});
      }
    }

    return allTokens;
  }

  // ── Private: Audio processing ──────────────────────────────────────────────

  /**
   * Process audio files via WAV parser → SpikeVox → pseudo-token IDs.
   *
   * @param {string[]} files
   * @param {Function} log
   * @returns {Promise<number[]>}
   */
  async _processAudio(files, log) {
    // Lazy-load WAV parser
    if (!parseWAV) {
      try {
        const mod = await import('./wav.js');
        parseWAV = mod.parseWAV;
      } catch (e) {
        log(`WAV parser not available — skipping audio: ${e.message}`, 'warn');
        return [];
      }
    }

    // Lazy-load FFmpeg
    let ffmpeg = null;
    try {
      ffmpeg = (await import('fluent-ffmpeg')).default;
      const ffmpegPath = (await import('@ffmpeg-installer/ffmpeg')).default.path;
      ffmpeg.setFfmpegPath(ffmpegPath);
    } catch (e) {
      log(`FFmpeg not available — non-WAV media will be skipped: ${e.message}`, 'warn');
    }

    // Lazy-load SpikeVox
    if (!SpikeVox) {
      try {
        const mod = await import('../multimodal/spikevox.js');
        SpikeVox = mod.SpikeVox;
      } catch (e) {
        log(`SpikeVox not available — skipping audio: ${e.message}`, 'warn');
        return [];
      }
    }

    const sv = new SpikeVox();
    const tokens = [];
    const V = 256;
    let processed = 0;
    const spikeAccum = new Float32Array(64);

    for (const file of files) {
      let wavFile = file;
      let isTemp = false;
      try {
        const ext = extname(file).toLowerCase();
        if (ext !== '.wav') {
          if (!ffmpeg) {
            log(`  Skipping ${file}: requires FFmpeg`, 'warn');
            continue;
          }
          wavFile = join(tmpdir(), `brymar_audio_${Date.now()}_${Math.random().toString(36).slice(2)}.wav`);
          isTemp = true;
          
          await new Promise((resolve, reject) => {
            ffmpeg(file)
              .noVideo()
              .audioCodec('pcm_s16le')
              .audioChannels(1)
              .audioFrequency(16000)
              .format('wav')
              .on('end', resolve)
              .on('error', reject)
              .save(wavFile);
          });
        }

        const { samples } = await parseWAV(wavFile);
        
        // Process in chunks (~100ms windows)
        const chunkSize = 4000;
        tokens.push(V + 1); // <|audio|>

        for (let i = 0; i < samples.length; i += chunkSize) {
          const chunk = samples.slice(i, i + chunkSize);
          const { spikes, dormant } = sv.processSamples(chunk);

          if (!dormant) {
            const dense = sv.spikesToFeatures(spikes, spikeAccum);
            for (let j = 0; j < dense.length; j++) {
              const bin = Math.min(31, Math.floor(dense[j] * 32));
              tokens.push(V + 64 + j * 32 + bin);
            }
          }
        }

        tokens.push(V + 2); // <|end|>
        processed++;
        sv.reset();
      } catch (e) {
        log(`  Audio load failed: ${file} — ${e.message}`, 'warn');
      } finally {
        if (isTemp) await unlink(wavFile).catch(() => {});
      }
    }

    if (processed > 0) {
      log(`Processed ${processed}/${files.length} audio files → ${tokens.length} tokens`, 'ok');
    }

    return tokens;
  }
}
