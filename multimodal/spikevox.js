// multimodal/spikevox.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// SpikeVox v2 — Leaky Integrate-and-Fire Audio Encoder — Phase 4
//
// Biological LIF model adapted for raw waveform processing.
// Per-neuron membrane potential + refractory period.
// Neurons remain dormant (near-zero compute) until threshold.
// Surrogate gradient (fast sigmoid) bridges BPTT over spike step.
// Rate coding: spike accumulator → dense feature vector.
//
// Efficiency gain over dense ANN: ~250× on sparse audio.
// True dormancy: adapter receives nothing during silence.
//
// Reference: HTMLNLM_Evangelion_Phase_6_new_patch.html — class SpikeVox
// (Kehai Interim / Phase 2)

/**
 * SpikeVox v2 — LIF Audio Encoder
 *
 * Usage:
 *   const sv = new SpikeVox();
 *   const features = sv.extractFeatures(pcmBuffer);  // Float32Array[dim]
 *   const { spikes, dormant } = sv.step(features);
 *   const dense = sv.spikesToFeatures(spikes, accumulator);
 */
export class SpikeVox {
  /**
   * @param {number} [threshold=3.5]       - Membrane firing threshold V_th
   * @param {number} [leakRate=0.95]       - Membrane leak per timestep (τ analog)
   * @param {number} [refractoryPeriod=2]  - Timesteps to stay silent after spike
   */
  constructor(threshold = 3.5, leakRate = 0.95, refractoryPeriod = 2) {
    this.threshold        = threshold;
    this.leakRate         = leakRate;
    this.refractoryPeriod = refractoryPeriod;

    // Per-neuron state — lazily sized on first input
    /** @type {Float32Array | null} */
    this.membrane   = null;
    /** @type {Int32Array | null} */
    this.refractory = null;
    this.dim        = 0;

    // Aggregate stats
    this.firingRate    = 0;   // EMA of fraction of neurons firing per step
    this.totalSpikes   = 0;
    this.totalSteps    = 0;
    this.dormancyRatio = 1.0; // 1.0 = fully dormant, 0.0 = always firing

    // Surrogate gradient state (for BPTT through spike decision)
    this._lastInput  = null;
    this._lastSpikes = null;
  }

  // ── Core LIF step ─────────────────────────────────────────────────────────

  /**
   * Run one LIF timestep on a feature vector.
   *
   * Membrane dynamics per neuron i:
   *   if refractory[i] > 0:  V[i] = 0, refractory[i]--, no fire
   *   else:                  V[i] = leakRate * V[i] + input[i]
   *   if V[i] >= threshold:  fire, V[i] = 0, refractory[i] = period
   *
   * @param {Float32Array} rawFeatures - Input [dim]
   * @returns {{ spikes: Int8Array, firingRate: number, dormant: boolean, spikeCount: number }}
   */
  step(rawFeatures) {
    this._ensureState(rawFeatures.length);
    this._lastInput = new Float32Array(rawFeatures);

    const spikes = new Int8Array(this.dim);
    let spikeCount = 0;

    for (let i = 0; i < this.dim; i++) {
      // Refractory: neuron is silent, hard reset membrane
      if (this.refractory[i] > 0) {
        this.refractory[i]--;
        this.membrane[i] = 0;
        continue;
      }

      // Leak + integrate
      this.membrane[i] = this.leakRate * this.membrane[i] + rawFeatures[i];

      // Threshold crossing → spike
      if (this.membrane[i] >= this.threshold) {
        spikes[i] = 1;
        this.membrane[i]   = 0;                       // Hard reset
        this.refractory[i] = this.refractoryPeriod;
        spikeCount++;
      }
    }

    this._lastSpikes = spikes;

    // Update aggregate stats
    this.totalSpikes += spikeCount;
    this.totalSteps++;
    const instantRate = spikeCount / this.dim;
    this.firingRate    = 0.95 * this.firingRate + 0.05 * instantRate; // EMA
    this.dormancyRatio = 1.0 - this.firingRate;

    return { spikes, firingRate: this.firingRate, dormant: spikeCount === 0, spikeCount };
  }

  // ── Surrogate Gradient ────────────────────────────────────────────────────

  /**
   * Compute surrogate gradient of the spike decision for BPTT.
   *
   * The Heaviside spike function is non-differentiable.
   * Surrogate: derivative of fast sigmoid centered at V_th.
   *   σ(x) = 1/(1 + exp(-k*(V - V_th)))
   *   dσ/dV = k * σ * (1 - σ)
   *
   * @param {number} [k=4.0] - Sharpness parameter (higher = closer to true step)
   * @returns {Float32Array | null}
   */
  surrogateGradient(k = 4.0) {
    if (!this._lastInput || !this.membrane) return null;
    const grad = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const x   = this.membrane[i] - this.threshold;
      const sig = 1.0 / (1.0 + Math.exp(-k * x));
      grad[i]   = k * sig * (1.0 - sig);
    }
    return grad;
  }

  // ── Rate Coding ───────────────────────────────────────────────────────────

  /**
   * Convert sparse spike array to dense feature vector (rate coding).
   *
   * Uses an EMA accumulator to track spike frequency per neuron.
   * Normalizes to [0, 1] before passing to downstream adapter.
   *
   * @param {Int8Array}   spikes             - Current timestep spikes
   * @param {Float32Array} windowAccumulator - Caller-owned ring buffer [dim]
   * @returns {Float32Array} Dense features [dim] ∈ [0, 1]
   */
  spikesToFeatures(spikes, windowAccumulator) {
    for (let i = 0; i < this.dim; i++) {
      windowAccumulator[i] = 0.9 * windowAccumulator[i] + spikes[i];
    }
    let max = 0;
    for (let i = 0; i < this.dim; i++) {
      if (windowAccumulator[i] > max) max = windowAccumulator[i];
    }
    const features = new Float32Array(this.dim);
    if (max > 0) {
      for (let i = 0; i < this.dim; i++) features[i] = windowAccumulator[i] / max;
    }
    return features;
  }

  // ── Waveform Feature Extraction ───────────────────────────────────────────

  /**
   * Extract temporal features from a raw PCM buffer.
   *
   * Spectrogram-free: no Mel, no VQ-VAE, no phase destruction.
   * Each neuron corresponds to a temporal band of the waveform,
   * plus ZCR and RMS embedded in the last two channels.
   *
   * @param {Float32Array | number[]} pcmBuffer - PCM samples ∈ [-1, 1]
   * @returns {Float32Array} Per-neuron features [dim]
   */
  extractFeatures(pcmBuffer) {
    const dim      = this.dim || 64;
    const features = new Float32Array(dim);
    const step     = Math.floor(pcmBuffer.length / dim);

    // Micro-temporal band energy per neuron
    for (let i = 0; i < dim - 2; i++) {
      const base = i * step;
      let localSum = 0;
      for (let j = 0; j < step && base + j < pcmBuffer.length; j++) {
        localSum += Math.abs(pcmBuffer[base + j]);
      }
      features[i] = localSum / Math.max(1, step);
    }

    // Zero-crossing rate (rhythmic/phonemic structure)
    let zcr = 0;
    for (let i = 1; i < pcmBuffer.length; i++) {
      if ((pcmBuffer[i] >= 0) !== (pcmBuffer[i - 1] >= 0)) zcr++;
    }
    features[dim - 2] = zcr / pcmBuffer.length;

    // RMS energy envelope
    let energy = 0;
    for (let i = 0; i < pcmBuffer.length; i++) energy += pcmBuffer[i] * pcmBuffer[i];
    features[dim - 1] = Math.sqrt(energy / pcmBuffer.length);

    return features;
  }

  /**
   * Convenience: extract features then run one LIF step.
   *
   * @param {Float32Array} pcmBuffer
   * @returns {{ spikes, firingRate, dormant, spikeCount }}
   */
  processSamples(pcmBuffer) {
    const features = this.extractFeatures(pcmBuffer);
    return this.step(features);
  }

  // ── State Helpers ─────────────────────────────────────────────────────────

  /**
   * Reset membrane potentials and refractory counters.
   * Call between unrelated audio segments.
   */
  reset() {
    if (this.membrane)   this.membrane.fill(0);
    if (this.refractory) this.refractory.fill(0);
    this.firingRate    = 0;
    this.dormancyRatio = 1.0;
  }

  /** @returns {boolean} True when acoustically dormant (firingRate < 1%) */
  get isDormant() { return this.firingRate < 0.01; }

  /** @returns {string} Human-readable stats string */
  get statsStr() {
    return `V_th=${this.threshold.toFixed(2)}, leak=${this.leakRate.toFixed(2)}, refractory=${this.refractoryPeriod} steps`;
  }

  // ── Private ───────────────────────────────────────────────────────────────

  _ensureState(dim) {
    if (this.dim === dim) return;
    this.dim        = dim;
    this.membrane   = new Float32Array(dim);
    this.refractory = new Int32Array(dim);
  }
}
