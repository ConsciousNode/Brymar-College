// multimodal/elastictok.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// ElasticTok v2 — Adaptive Visual Tokenization — Phase 4
//
// Variable-density patch tokenizer: static patches skipped,
// high-motion patches emit full tokens.
//
// Node.js adaptation: operates on raw RGBA Uint8Array pixel buffers
// (e.g. from the 'sharp' library or any other image decoder) rather
// than browser ImageData. API remains identical to the reference.
//
// Reference: HTMLNLM_Evangelion_Phase_6_new_patch.html — class ElasticTok
// (Kehai Interim / Phase 3)

/**
 * ElasticTok v2 — Adaptive Visual Tokenization
 *
 * Input:  RGBA pixel buffer (Uint8Array or Uint8ClampedArray), width, height
 * Output: { patchFeatures, saliency, coords, gridW, gridH, nPatches }
 *         or null if scene is globally static.
 *
 * Patch features: RGB means per patch (3 floats each).
 * Saliency: temporal delta magnitude per patch (1 float each).
 * Coords: normalized [cx, cy] ∈ (0,1)² per patch (2 floats each).
 *
 * For the first frame (no lastFrame reference), all patches are
 * considered fully salient.
 */
export class ElasticTok {
  /**
   * @param {number} [patchSize=16]  - Pixel side length of each patch
   * @param {number} [maxPatches=64] - Hard cap on output token count
   */
  constructor(patchSize = 16, maxPatches = 64) {
    this.patchSize  = patchSize;
    this.maxPatches = maxPatches;
    /** @type {Float32Array | null} Previous frame patch RGB means */
    this.lastFrame  = null;
    this.tokenCount = 0;
    this.totalTokensGenerated = 0;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Process one video frame.
   *
   * @param {Uint8Array | Uint8ClampedArray} pixels  - RGBA pixel buffer
   * @param {number} width
   * @param {number} height
   * @param {number} [deltaThreshold=0.02]           - Motion threshold (0–1)
   * @returns {{ patchFeatures: Float32Array, saliency: Float32Array,
   *             coords: Float32Array, gridW: number, gridH: number,
   *             nPatches: number } | null}
   */
  processFrame(pixels, width, height, deltaThreshold = 0.02) {
    const ps       = this.patchSize;
    const gridW    = Math.floor(width  / ps);
    const gridH    = Math.floor(height / ps);
    const nTotal   = gridW * gridH;
    const patchDim = 3; // R, G, B channel means

    if (nTotal === 0) return null;

    // ── Extract patch RGB means (normalized 0–1) ──────────────────────────
    const current = new Float32Array(nTotal * patchDim);

    for (let pr = 0; pr < gridH; pr++) {
      for (let pc = 0; pc < gridW; pc++) {
        const idx = pr * gridW + pc;
        let rSum = 0, gSum = 0, bSum = 0, count = 0;

        for (let y = pr * ps; y < Math.min((pr + 1) * ps, height); y++) {
          for (let x = pc * ps; x < Math.min((pc + 1) * ps, width); x++) {
            const p = (y * width + x) * 4; // RGBA stride
            rSum += pixels[p];
            gSum += pixels[p + 1];
            bSum += pixels[p + 2];
            count++;
          }
        }

        current[idx * patchDim]     = rSum / (count * 255);
        current[idx * patchDim + 1] = gSum / (count * 255);
        current[idx * patchDim + 2] = bSum / (count * 255);
      }
    }

    // ── Per-patch saliency (temporal delta from last frame) ───────────────
    const saliency = new Float32Array(nTotal);
    if (this.lastFrame) {
      for (let i = 0; i < nTotal; i++) {
        let delta = 0;
        for (let c = 0; c < patchDim; c++) {
          const d = current[i * patchDim + c] - this.lastFrame[i * patchDim + c];
          delta += d * d;
        }
        saliency[i] = Math.sqrt(delta / patchDim);
      }
    } else {
      saliency.fill(1.0); // First frame: all patches salient
    }

    this.lastFrame = current;

    // ── Global delta gate ─────────────────────────────────────────────────
    let maxSaliency = 0;
    for (let i = 0; i < nTotal; i++) {
      if (saliency[i] > maxSaliency) maxSaliency = saliency[i];
    }
    // Skip if nothing moved (and we have a reference frame)
    if (maxSaliency < deltaThreshold && this.lastFrame !== null) return null;

    // ── Select top-K salient patches ──────────────────────────────────────
    const ranked = Array.from({ length: nTotal }, (_, i) => i)
      .sort((a, b) => saliency[b] - saliency[a]);

    // Dynamic token count: proportional to max saliency, capped at maxPatches
    const nPatches = Math.min(
      this.maxPatches,
      Math.max(4, Math.ceil(nTotal * maxSaliency))
    );

    // Restore spatial order for consistent Re-WKV processing
    const selected = ranked.slice(0, nPatches).sort((a, b) => a - b);

    // ── Pack output ───────────────────────────────────────────────────────
    const patchFeatures = new Float32Array(nPatches * patchDim);
    const patchSaliency = new Float32Array(nPatches);
    const coords        = new Float32Array(nPatches * 2);

    for (let i = 0; i < nPatches; i++) {
      const idx = selected[i];
      const pr  = Math.floor(idx / gridW);
      const pc  = idx % gridW;

      for (let c = 0; c < patchDim; c++) {
        patchFeatures[i * patchDim + c] = current[idx * patchDim + c];
      }
      patchSaliency[i]  = saliency[idx];
      coords[i * 2]     = (pc + 0.5) / gridW;  // normalized x ∈ (0,1)
      coords[i * 2 + 1] = (pr + 0.5) / gridH;  // normalized y ∈ (0,1)
    }

    this.tokenCount = nPatches;
    this.totalTokensGenerated += nPatches;

    return { patchFeatures, saliency: patchSaliency, coords, gridW, gridH, nPatches };
  }

  /**
   * Reset temporal reference (use at scene cuts or stream restarts).
   */
  reset() {
    this.lastFrame = null;
    this.tokenCount = 0;
  }

  /**
   * Process a grayscale or RGB raw pixel buffer.
   *
   * Convenience wrapper that constructs a synthetic RGBA buffer from
   * flat RGB data (e.g. from PNG decoded without alpha channel).
   *
   * @param {Uint8Array} rgbBuffer - Flat RGB pixels (3 bytes/pixel)
   * @param {number} width
   * @param {number} height
   * @param {number} [deltaThreshold=0.02]
   */
  processRGB(rgbBuffer, width, height, deltaThreshold = 0.02) {
    // Convert RGB → RGBA
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4]     = rgbBuffer[i * 3];
      rgba[i * 4 + 1] = rgbBuffer[i * 3 + 1];
      rgba[i * 4 + 2] = rgbBuffer[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
    return this.processFrame(rgba, width, height, deltaThreshold);
  }
}
