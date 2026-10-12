// model/tmac.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// T-MAC ternary microkernel — direct port from HTMLNLM.html

/**
 * TMAC: Ternary Matrix Acceleration via Lookup Table
 * 
 * Quantizes Float32 weights to {-1, 0, 1} and performs GEMM via precomputed
 * lookup tables. Packs 4 ternary values into base-81 indices for cache efficiency.
 * 
 * Reference: HTMLNLM.html TMAC class
 */
export class TMAC {
  static UNPACK_TABLE = new Int8Array(81 * 4);

  /**
   * Initialize the unpacking lookup table.
   * Maps base-81 indices to 4 ternary digits in {-1, 0, 1}.
   */
  static initTable() {
    for (let i = 0; i < 81; i++) {
      let rem = i;
      TMAC.UNPACK_TABLE[i * 4]     = (rem % 3) - 1; rem = Math.floor(rem / 3);
      TMAC.UNPACK_TABLE[i * 4 + 1] = (rem % 3) - 1; rem = Math.floor(rem / 3);
      TMAC.UNPACK_TABLE[i * 4 + 2] = (rem % 3) - 1; rem = Math.floor(rem / 3);
      TMAC.UNPACK_TABLE[i * 4 + 3] = (rem % 3) - 1;
    }
  }

  /**
   * Pack 4 ternary values {-1, 0, 1} into a single base-81 index.
   */
  static packTernary(t0, t1, t2, t3) {
    return (t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1) + 27 * (t3 + 1);
  }

  /**
   * Quantize Float32 weights to ternary {-1, 0, 1} and pack into Uint8Array.
   * 
   * @param {Float32Array} masterWeights - Source weights
   * @param {Uint8Array} packedOut - Output packed ternary indices
   * @returns {number} gamma - Scaling factor for dequantization
   */
  static quantizeWeights(masterWeights, packedOut) {
    const n = masterWeights.length;
    const blocks = Math.ceil(n / 4);
    
    let sumAbs = 0;
    for (let i = 0; i < n; i++) sumAbs += Math.abs(masterWeights[i]);
    const gamma = (sumAbs / n) || 1e-8;
    const invGamma = 1.0 / gamma;
    
    for (let b = 0; b < blocks; b++) {
      const base = b * 4;
      const t0 = base     < n ? Math.max(-1, Math.min(1, Math.round(masterWeights[base]     * invGamma))) : 0;
      const t1 = base + 1 < n ? Math.max(-1, Math.min(1, Math.round(masterWeights[base + 1] * invGamma))) : 0;
      const t2 = base + 2 < n ? Math.max(-1, Math.min(1, Math.round(masterWeights[base + 2] * invGamma))) : 0;
      const t3 = base + 3 < n ? Math.max(-1, Math.min(1, Math.round(masterWeights[base + 3] * invGamma))) : 0;
      packedOut[b] = TMAC.packTernary(t0, t1, t2, t3);
    }
    
    return gamma;
  }

  /**
   * Execute ternary GEMM: output = packedWeights @ input * gamma
   * 
   * @param {Uint8Array} packedWeights - Packed ternary weight matrix [outDim × ceil(inDim/4)]
   * @param {Float32Array} input - Input vector [inDim]
   * @param {number} outDim - Output dimension
   * @param {number} inDim - Input dimension
   * @param {number} gamma - Scaling factor from quantization
   * @param {Float32Array} outputArr - Output buffer [outDim]
   * @param {Float32Array} scratchLut - Scratch buffer [ceil(inDim/4) × 81]
   */
  static execute(packedWeights, input, outDim, inDim, gamma, outputArr, scratchLut) {
    const blocksPerRow = Math.ceil(inDim / 4);
    
    // Build lookup tables for each input block
    for (let b = 0; b < blocksPerRow; b++) {
      const base = b * 4;
      const v0 = base     < inDim ? input[base]     : 0;
      const v1 = base + 1 < inDim ? input[base + 1] : 0;
      const v2 = base + 2 < inDim ? input[base + 2] : 0;
      const v3 = base + 3 < inDim ? input[base + 3] : 0;
      
      const lutOffset = b * 81;
      for (let i = 0; i < 81; i++) {
        const uBase = i * 4;
        scratchLut[lutOffset + i] =
          (TMAC.UNPACK_TABLE[uBase]     * v0) + (TMAC.UNPACK_TABLE[uBase + 1] * v1) +
          (TMAC.UNPACK_TABLE[uBase + 2] * v2) + (TMAC.UNPACK_TABLE[uBase + 3] * v3);
      }
    }
    
    // Compute output via LUT lookups
    for (let o = 0; o < outDim; o++) {
      let acc = 0;
      const rowBase = o * blocksPerRow;
      for (let b = 0; b < blocksPerRow; b++) {
        acc += scratchLut[b * 81 + packedWeights[rowBase + b]];
      }
      outputArr[o] = acc * gamma;
    }
  }
}

// Initialize lookup table at module load time
TMAC.initTable();
