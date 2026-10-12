// tokenizer/wav.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Zero-dependency WAV file parser — Phase 5
//
// Reads standard RIFF/WAVE files and outputs normalized Float32Array ∈ [-1, 1].
// Supports PCM int8/int16/int24/int32 and IEEE float32/float64.
// Multi-channel files are downmixed to mono.

import { readFile } from 'node:fs/promises';

/**
 * Parse a WAV file from disk and return normalized mono PCM data.
 *
 * @param {string} filePath - Absolute or relative path to .wav file
 * @returns {Promise<{ samples: Float32Array, sampleRate: number, channels: number, bitsPerSample: number }>}
 */
export async function parseWAV(filePath) {
  const buffer = await readFile(filePath);
  return decodeWAV(buffer);
}

/**
 * Decode a WAV buffer (Buffer or ArrayBuffer) to normalized Float32Array.
 *
 * @param {Buffer | ArrayBuffer} input
 * @returns {{ samples: Float32Array, sampleRate: number, channels: number, bitsPerSample: number }}
 */
export function decodeWAV(input) {
  const buf = Buffer.isBuffer(input) ? input : Buffer.from(input);

  // ── RIFF header ───────────────────────────────────────────────────────
  const riff = buf.toString('ascii', 0, 4);
  if (riff !== 'RIFF') {
    throw new Error(`Not a RIFF file (got "${riff}")`);
  }

  const wave = buf.toString('ascii', 8, 12);
  if (wave !== 'WAVE') {
    throw new Error(`Not a WAVE file (got "${wave}")`);
  }

  // ── Find chunks ───────────────────────────────────────────────────────
  let fmtChunk = null;
  let dataChunk = null;
  let offset = 12;

  while (offset < buf.length - 8) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    const chunkData = offset + 8;

    if (chunkId === 'fmt ') {
      fmtChunk = { offset: chunkData, size: chunkSize };
    } else if (chunkId === 'data') {
      dataChunk = { offset: chunkData, size: chunkSize };
    }

    // Move to next chunk (pad to even boundary)
    offset = chunkData + chunkSize + (chunkSize % 2);

    if (fmtChunk && dataChunk) break;
  }

  if (!fmtChunk) throw new Error('WAV: missing fmt chunk');
  if (!dataChunk) throw new Error('WAV: missing data chunk');

  // ── Parse fmt chunk ───────────────────────────────────────────────────
  const audioFormat   = buf.readUInt16LE(fmtChunk.offset);
  const channels      = buf.readUInt16LE(fmtChunk.offset + 2);
  const sampleRate    = buf.readUInt32LE(fmtChunk.offset + 4);
  // byteRate        = buf.readUInt32LE(fmtChunk.offset + 8);  // unused
  // blockAlign      = buf.readUInt16LE(fmtChunk.offset + 12); // unused
  const bitsPerSample = buf.readUInt16LE(fmtChunk.offset + 14);

  const FORMAT_PCM   = 1;
  const FORMAT_FLOAT = 3;

  if (audioFormat !== FORMAT_PCM && audioFormat !== FORMAT_FLOAT) {
    throw new Error(`Unsupported WAV format code ${audioFormat} (only PCM=1 and IEEE float=3 supported)`);
  }

  // ── Decode samples ────────────────────────────────────────────────────
  const bytesPerSample = bitsPerSample / 8;
  const totalSamples   = Math.floor(dataChunk.size / (bytesPerSample * channels));
  const raw            = new Float32Array(totalSamples * channels);
  let readOffset       = dataChunk.offset;

  for (let i = 0; i < totalSamples * channels; i++) {
    if (audioFormat === FORMAT_FLOAT) {
      if (bitsPerSample === 32) {
        raw[i] = buf.readFloatLE(readOffset);
      } else if (bitsPerSample === 64) {
        raw[i] = buf.readDoubleLE(readOffset);
      }
    } else {
      // PCM integer formats → normalize to [-1, 1]
      switch (bitsPerSample) {
        case 8:
          // uint8 → [-1, 1] (WAV 8-bit is unsigned, centered at 128)
          raw[i] = (buf.readUInt8(readOffset) - 128) / 128;
          break;
        case 16:
          raw[i] = buf.readInt16LE(readOffset) / 32768;
          break;
        case 24: {
          // 24-bit little-endian signed
          const lo = buf.readUInt16LE(readOffset);
          const hi = buf.readInt8(readOffset + 2);
          raw[i] = ((hi << 16) | lo) / 8388608;
          break;
        }
        case 32:
          raw[i] = buf.readInt32LE(readOffset) / 2147483648;
          break;
        default:
          throw new Error(`Unsupported PCM bit depth: ${bitsPerSample}`);
      }
    }
    readOffset += bytesPerSample;
  }

  // ── Mono downmix ──────────────────────────────────────────────────────
  let samples;
  if (channels === 1) {
    samples = raw;
  } else {
    samples = new Float32Array(totalSamples);
    for (let i = 0; i < totalSamples; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) {
        sum += raw[i * channels + c];
      }
      samples[i] = sum / channels;
    }
  }

  // Clamp to [-1, 1]
  for (let i = 0; i < samples.length; i++) {
    samples[i] = Math.max(-1, Math.min(1, samples[i]));
  }

  return { samples, sampleRate, channels, bitsPerSample };
}
