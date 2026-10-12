// persistence/quantize.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Quantization strategies for .rofl / .lmao / .fuctrump export formats
//
// Three tiers:
//   ROFL     — FP32 pass-through (no quantization)
//   LMAO     — Q4 symmetric per-group quantization (group_size=32)
//   FUCTRUMP — Q2 ternary {-1, 0, +1} quantization (extends BitNet b1.58)

/**
 * Q4 symmetric quantization — 4-bit per group.
 *
 * For each group of `groupSize` weights:
 *   1. Compute absmax scale = max(|w_i|) / 7
 *   2. Quantize: q_i = round(clamp(w_i / scale, -8, 7))
 *   3. Pack two 4-bit values into one byte (low nibble first)
 *
 * Storage: scale (Float32) per group + packed nibbles.
 *
 * @param {Float32Array} weights - Source weight array
 * @param {number} [groupSize=32] - Quantization group size
 * @returns {{ packed: Uint8Array, scales: Float32Array, groupSize: number, origLength: number }}
 */
export function quantizeQ4(weights, groupSize = 32) {
  const n = weights.length;
  const nGroups = Math.ceil(n / groupSize);
  const scales = new Float32Array(nGroups);
  // Two values per byte → ceil(groupSize / 2) bytes per group
  const packedBytesPerGroup = Math.ceil(groupSize / 2);
  const packed = new Uint8Array(nGroups * packedBytesPerGroup);

  for (let g = 0; g < nGroups; g++) {
    const gStart = g * groupSize;
    const gEnd = Math.min(gStart + groupSize, n);

    // Find absmax for this group
    let absmax = 0;
    for (let i = gStart; i < gEnd; i++) {
      const a = Math.abs(weights[i]);
      if (a > absmax) absmax = a;
    }

    // Scale: map [-absmax, absmax] → [-8, 7]
    const scale = absmax / 7 || 1e-12;
    scales[g] = scale;

    // Quantize and pack
    const packBase = g * packedBytesPerGroup;
    for (let i = 0; i < gEnd - gStart; i += 2) {
      const idx0 = gStart + i;
      const idx1 = gStart + i + 1;

      const q0 = Math.max(-8, Math.min(7, Math.round(weights[idx0] / scale)));
      const q1 = idx1 < gEnd
        ? Math.max(-8, Math.min(7, Math.round(weights[idx1] / scale)))
        : 0;

      // Pack: low nibble = q0 + 8, high nibble = q1 + 8
      packed[packBase + (i >> 1)] = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
    }
  }

  return { packed, scales, groupSize, origLength: n };
}

/**
 * Dequantize Q4 data back to Float32.
 *
 * @param {{ packed: Uint8Array, scales: Float32Array, groupSize: number, origLength: number }} q4
 * @returns {Float32Array}
 */
export function dequantizeQ4(q4) {
  const { packed, scales, groupSize, origLength } = q4;
  const out = new Float32Array(origLength);
  const nGroups = scales.length;
  const packedBytesPerGroup = Math.ceil(groupSize / 2);

  for (let g = 0; g < nGroups; g++) {
    const gStart = g * groupSize;
    const gEnd = Math.min(gStart + groupSize, origLength);
    const scale = scales[g];
    const packBase = g * packedBytesPerGroup;

    for (let i = 0; i < gEnd - gStart; i += 2) {
      const byte = packed[packBase + (i >> 1)];
      const q0 = (byte & 0x0F) - 8;
      const q1 = ((byte >> 4) & 0x0F) - 8;

      out[gStart + i] = q0 * scale;
      if (gStart + i + 1 < gEnd) {
        out[gStart + i + 1] = q1 * scale;
      }
    }
  }

  return out;
}

/**
 * Q2 ternary quantization — 2-bit {-1, 0, +1}.
 *
 * Same principle as BitNet b1.58 / TMAC, but applied at export time
 * to full model weights rather than at inference time.
 *
 * For the entire weight array:
 *   1. Compute gamma = mean(|w_i|)
 *   2. Quantize: t_i = round(clamp(w_i / gamma, -1, 1))
 *   3. Pack 4 ternary values into one byte using base-81 encoding
 *
 * @param {Float32Array} weights
 * @returns {{ packed: Uint8Array, gamma: number, origLength: number }}
 */
export function quantizeQ2(weights) {
  const n = weights.length;
  const blocks = Math.ceil(n / 4);
  const packed = new Uint8Array(blocks);

  // Compute gamma (mean absolute value)
  let sumAbs = 0;
  for (let i = 0; i < n; i++) sumAbs += Math.abs(weights[i]);
  const gamma = (sumAbs / n) || 1e-8;
  const invGamma = 1.0 / gamma;

  for (let b = 0; b < blocks; b++) {
    const base = b * 4;
    const t0 = base     < n ? Math.max(-1, Math.min(1, Math.round(weights[base]     * invGamma))) : 0;
    const t1 = base + 1 < n ? Math.max(-1, Math.min(1, Math.round(weights[base + 1] * invGamma))) : 0;
    const t2 = base + 2 < n ? Math.max(-1, Math.min(1, Math.round(weights[base + 2] * invGamma))) : 0;
    const t3 = base + 3 < n ? Math.max(-1, Math.min(1, Math.round(weights[base + 3] * invGamma))) : 0;
    packed[b] = (t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1) + 27 * (t3 + 1);
  }

  return { packed, gamma, origLength: n };
}

/**
 * Dequantize Q2 ternary data back to Float32.
 *
 * @param {{ packed: Uint8Array, gamma: number, origLength: number }} q2
 * @returns {Float32Array}
 */
export function dequantizeQ2(q2) {
  const { packed, gamma, origLength } = q2;
  const out = new Float32Array(origLength);

  for (let b = 0; b < packed.length; b++) {
    let rem = packed[b];
    const base = b * 4;
    for (let j = 0; j < 4 && base + j < origLength; j++) {
      const t = (rem % 3) - 1;
      rem = Math.floor(rem / 3);
      out[base + j] = t * gamma;
    }
  }

  return out;
}

/**
 * Serialize quantized data to a binary buffer for file export.
 *
 * Binary layout:
 *   [4 bytes] magic: 'ROFL', 'LMAO', or 'FCTR'
 *   [4 bytes] format version (uint32 LE)
 *   [4 bytes] quantBits (uint32 LE)
 *   [4 bytes] origLength (uint32 LE)
 *   [... ]    format-specific payload
 *
 * ROFL payload: raw Float32 array
 * LMAO payload: [4] groupSize, [4*nGroups] scales, [packed bytes]
 * FCTR payload: [8] gamma (float64), [packed bytes]
 *
 * @param {string} format - 'rofl' | 'lmao' | 'fuctrump'
 * @param {Float32Array} weights - Source weights (used for rofl, or pre-quantized data)
 * @param {Object} [quantData] - Output from quantizeQ4 or quantizeQ2
 * @returns {Buffer}
 */
export function serializeWeights(format, weights, quantData = null) {
  const headerSize = 16; // magic(4) + version(4) + quantBits(4) + origLength(4)

  if (format === 'rofl') {
    // FP32 — no quantization
    const buf = Buffer.alloc(headerSize + weights.length * 4);
    buf.write('ROFL', 0, 4, 'ascii');
    buf.writeUInt32LE(1, 4);    // version
    buf.writeUInt32LE(32, 8);   // quantBits
    buf.writeUInt32LE(weights.length, 12);
    Buffer.from(weights.buffer, weights.byteOffset, weights.byteLength).copy(buf, headerSize);
    return buf;
  }

  if (format === 'lmao') {
    const q = quantData || quantizeQ4(weights);
    const scalesBytes = q.scales.length * 4;
    const buf = Buffer.alloc(headerSize + 4 + scalesBytes + q.packed.length);
    buf.write('LMAO', 0, 4, 'ascii');
    buf.writeUInt32LE(1, 4);
    buf.writeUInt32LE(4, 8);
    buf.writeUInt32LE(q.origLength, 12);
    buf.writeUInt32LE(q.groupSize, headerSize);
    Buffer.from(q.scales.buffer, q.scales.byteOffset, q.scales.byteLength).copy(buf, headerSize + 4);
    Buffer.from(q.packed.buffer, q.packed.byteOffset, q.packed.byteLength).copy(buf, headerSize + 4 + scalesBytes);
    return buf;
  }

  if (format === 'fuctrump') {
    const q = quantData || quantizeQ2(weights);
    const buf = Buffer.alloc(headerSize + 8 + q.packed.length);
    buf.write('FCTR', 0, 4, 'ascii');
    buf.writeUInt32LE(1, 4);
    buf.writeUInt32LE(2, 8);
    buf.writeUInt32LE(q.origLength, 12);
    buf.writeDoubleBE(q.gamma, headerSize);
    Buffer.from(q.packed.buffer, q.packed.byteOffset, q.packed.byteLength).copy(buf, headerSize + 8);
    return buf;
  }

  throw new Error(`Unknown quantization format: ${format}`);
}
