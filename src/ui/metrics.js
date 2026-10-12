// src/ui/metrics.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Training metrics aggregation — Phase 3

/**
 * Metrics: lightweight ring-buffer for training statistics.
 *
 * Collects (step, loss, tokPerSec) tuples and provides windowed averages.
 */
export class Metrics {
  constructor() {
    /** @type {Array<{ step: number, loss: number, tokPerSec: number, timestamp: number }>} */
    this.history = [];
  }

  /**
   * Record a new data point.
   *
   * @param {number} step
   * @param {number} loss
   * @param {number} tokPerSec
   */
  record(step, loss, tokPerSec) {
    this.history.push({ step, loss, tokPerSec, timestamp: Date.now() });
  }

  /**
   * Compute the average loss over the last `window` recorded steps.
   *
   * @param {number} [window=10] - Number of most-recent entries to average
   * @returns {number} Average loss (0.0 if no history)
   */
  getAverage(window = 10) {
    if (this.history.length === 0) return 0.0;
    const slice = this.history.slice(-window);
    const sum   = slice.reduce((acc, e) => acc + e.loss, 0);
    return sum / slice.length;
  }

  /**
   * Average throughput (tok/s) over the last `window` steps.
   *
   * @param {number} [window=10]
   * @returns {number}
   */
  getAverageThroughput(window = 10) {
    if (this.history.length === 0) return 0.0;
    const slice = this.history.slice(-window);
    const sum   = slice.reduce((acc, e) => acc + e.tokPerSec, 0);
    return sum / slice.length;
  }

  /**
   * Return the minimum loss seen so far.
   *
   * @returns {number}
   */
  getBestLoss() {
    if (this.history.length === 0) return Infinity;
    return Math.min(...this.history.map(e => e.loss));
  }

  /**
   * Return a summary object suitable for logging.
   *
   * @returns {Object}
   */
  summary() {
    return {
      totalSteps:   this.history.length,
      avgLoss:      this.getAverage(10),
      avgThroughput: this.getAverageThroughput(10),
      bestLoss:     this.getBestLoss()
    };
  }
}
