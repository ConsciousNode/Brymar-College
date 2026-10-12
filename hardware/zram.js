// hardware/zram.js
// Brymar College — zRAM Memory Management for Linux
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
//
// Probes the Linux kernel for zRAM compressed swap devices, reads /proc/meminfo
// for physical RAM stats, and provides periodic memory pressure snapshots during
// training. Degrades gracefully on non-Linux or when zRAM is not loaded.
//
// Target: Intel Core i7 Sandy Bridge, 12GB RAM + 9GB zRAM

import { readFile, readdir } from 'node:fs/promises';
import { platform } from 'node:os';

/**
 * ZRamMonitor — Linux zRAM memory management and pressure monitoring.
 *
 * Usage:
 *   const zram = new ZRamMonitor();
 *   const info = await zram.probe(terminal);  // logs hardware status at startup
 *   const snap = await zram.snapshot();        // periodic pressure check
 */
export class ZRamMonitor {
  constructor() {
    /** @type {boolean} Whether we're running on Linux */
    this.isLinux = platform() === 'linux';
    /** @type {string[]} Detected zRAM device paths under /sys/block/ */
    this.devices = [];
    /** @type {boolean} Whether probe() has been called */
    this.probed = false;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Probe the system for zRAM devices and physical RAM.
   * Logs results to the terminal in CRT format.
   *
   * @param {import('../src/ui/terminal.js').Terminal} [terminal]
   * @returns {Promise<{ physTotalMB: number, physAvailMB: number, zramTotalMB: number, zramUsedMB: number, zramComprRatio: number, effectiveMB: number }>}
   */
  async probe(terminal) {
    const log = (msg, level = 'info') =>
      terminal ? terminal.log(msg, level) : console.log(`[${level.toUpperCase()}] ${msg}`);

    if (!this.isLinux) {
      log('zRAM: Not Linux — memory monitoring limited to process RSS', 'warn');
      const rss = process.memoryUsage().rss;
      this.probed = true;
      return {
        physTotalMB: 0, physAvailMB: 0,
        zramTotalMB: 0, zramUsedMB: 0, zramComprRatio: 0,
        effectiveMB: Math.round(rss / 1048576),
      };
    }

    // ── Read /proc/meminfo ────────────────────────────────────────────────
    const meminfo = await this._readMeminfo();
    const physTotalMB  = meminfo.MemTotal;
    const physAvailMB  = meminfo.MemAvailable;
    const swapTotalMB  = meminfo.SwapTotal;
    const swapFreeMB   = meminfo.SwapFree;

    log(`Physical RAM: ${physTotalMB.toFixed(0)} MB total, ${physAvailMB.toFixed(0)} MB available`, 'mem');

    // ── Detect zRAM devices ──────────────────────────────────────────────
    this.devices = await this._findZramDevices();
    let zramTotalMB = 0;
    let zramUsedMB = 0;
    let zramOrigMB = 0;
    let zramComprMB = 0;

    if (this.devices.length > 0) {
      for (const dev of this.devices) {
        const stats = await this._readZramStats(dev);
        zramTotalMB += stats.diskSizeMB;
        zramUsedMB  += stats.dataUsedMB;
        zramOrigMB  += stats.origDataMB;
        zramComprMB += stats.comprDataMB;
      }

      const ratio = zramComprMB > 0 ? (zramOrigMB / zramComprMB) : 0;
      log(`zRAM: ${this.devices.length} device(s), ${zramTotalMB.toFixed(0)} MB capacity, ${zramUsedMB.toFixed(0)} MB used`, 'mem');
      if (ratio > 0) {
        log(`zRAM compression ratio: ${ratio.toFixed(2)}:1 (${zramOrigMB.toFixed(0)} MB → ${zramComprMB.toFixed(0)} MB)`, 'mem');
      }
    } else {
      log('zRAM: kernel module not loaded — swap-only or no compressed memory', 'warn');
    }

    const effectiveMB = physTotalMB + zramTotalMB;
    log(`Effective memory headroom: ${(effectiveMB / 1024).toFixed(1)} GB`, 'ok');

    this.probed = true;
    return {
      physTotalMB, physAvailMB,
      zramTotalMB, zramUsedMB,
      zramComprRatio: zramComprMB > 0 ? zramOrigMB / zramComprMB : 0,
      effectiveMB,
    };
  }

  /**
   * Take a memory pressure snapshot. Lightweight, suitable for periodic
   * polling during training (e.g. every 50 steps).
   *
   * @returns {Promise<{ physAvailMB: number, physUsedPct: number, zramUsedMB: number, rssNodeMB: number, pressureLevel: 'low'|'medium'|'high'|'critical' }>}
   */
  async snapshot() {
    const rss = process.memoryUsage().rss;
    const rssNodeMB = rss / 1048576;

    if (!this.isLinux) {
      return {
        physAvailMB: 0,
        physUsedPct: 0,
        zramUsedMB: 0,
        rssNodeMB,
        pressureLevel: 'low',
      };
    }

    const meminfo = await this._readMeminfo();
    const physUsedPct = 100 * (1 - meminfo.MemAvailable / Math.max(1, meminfo.MemTotal));

    let zramUsedMB = 0;
    for (const dev of this.devices) {
      const stats = await this._readZramStats(dev);
      zramUsedMB += stats.dataUsedMB;
    }

    // Pressure levels:
    //   low      < 60% physical used
    //   medium   60-80% physical used
    //   high     80-95% physical used (zRAM kicking in)
    //   critical > 95% physical used (OOM risk)
    let pressureLevel = 'low';
    if (physUsedPct > 95)      pressureLevel = 'critical';
    else if (physUsedPct > 80)  pressureLevel = 'high';
    else if (physUsedPct > 60)  pressureLevel = 'medium';

    return {
      physAvailMB: meminfo.MemAvailable,
      physUsedPct,
      zramUsedMB,
      rssNodeMB,
      pressureLevel,
    };
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /**
   * Parse /proc/meminfo into an object of MB values.
   * @returns {Promise<{ MemTotal: number, MemAvailable: number, SwapTotal: number, SwapFree: number }>}
   */
  async _readMeminfo() {
    const defaults = { MemTotal: 0, MemAvailable: 0, SwapTotal: 0, SwapFree: 0 };
    try {
      const raw = await readFile('/proc/meminfo', 'utf8');
      for (const line of raw.split('\n')) {
        const match = line.match(/^(\w+):\s+(\d+)\s+kB/);
        if (match && match[1] in defaults) {
          defaults[match[1]] = parseInt(match[2]) / 1024; // kB → MB
        }
      }
    } catch {
      // Not Linux or /proc not mounted
    }
    return defaults;
  }

  /**
   * Find zRAM block devices under /sys/block/.
   * @returns {Promise<string[]>} e.g. ['/sys/block/zram0']
   */
  async _findZramDevices() {
    try {
      const entries = await readdir('/sys/block');
      return entries
        .filter(e => e.startsWith('zram'))
        .map(e => `/sys/block/${e}`);
    } catch {
      return [];
    }
  }

  /**
   * Read stats for a single zRAM device from sysfs.
   *
   * Tries mm_stat first (modern kernels), falls back to individual sysfs files.
   *
   * mm_stat columns (space-separated):
   *   0: orig_data_size    1: compr_data_size   2: mem_used_total
   *   3: mem_limit         4: mem_used_max      5: same_pages
   *   6: pages_compacted   7: huge_pages (optional)
   *
   * @param {string} devPath - e.g. '/sys/block/zram0'
   * @returns {Promise<{ diskSizeMB: number, dataUsedMB: number, origDataMB: number, comprDataMB: number }>}
   */
  async _readZramStats(devPath) {
    const result = { diskSizeMB: 0, dataUsedMB: 0, origDataMB: 0, comprDataMB: 0 };

    try {
      // disksize is always available
      const disksize = await readFile(`${devPath}/disksize`, 'utf8');
      result.diskSizeMB = parseInt(disksize.trim()) / 1048576;
    } catch { /* no disksize — device may not be initialized */ }

    try {
      // Modern kernels: mm_stat
      const mmstat = await readFile(`${devPath}/mm_stat`, 'utf8');
      const fields = mmstat.trim().split(/\s+/).map(Number);
      if (fields.length >= 3) {
        result.origDataMB  = fields[0] / 1048576;
        result.comprDataMB = fields[1] / 1048576;
        result.dataUsedMB  = fields[2] / 1048576;
      }
    } catch {
      // Fallback: older kernels with individual files
      try {
        const orig = await readFile(`${devPath}/orig_data_size`, 'utf8');
        result.origDataMB = parseInt(orig.trim()) / 1048576;
      } catch { /* not available */ }
      try {
        const compr = await readFile(`${devPath}/compr_data_size`, 'utf8');
        result.comprDataMB = parseInt(compr.trim()) / 1048576;
      } catch { /* not available */ }
      try {
        const used = await readFile(`${devPath}/mem_used_total`, 'utf8');
        result.dataUsedMB = parseInt(used.trim()) / 1048576;
      } catch { /* not available */ }
    }

    return result;
  }
}
