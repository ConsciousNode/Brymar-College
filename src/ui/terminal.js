// src/ui/terminal.js
// Brymar College — Terminal UI Module
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Licensed under the MIT License

import chalk from 'chalk';
import { SingleBar, Presets } from 'cli-progress';

const SEVERITY = {
  info: chalk.cyan('[INFO]'),
  warn: chalk.yellow('[WARN]'),
  error: chalk.red('[ERR ]'),
  train: chalk.green('[STEP]'),
  ok: chalk.greenBright('[OK ]'),
  mem: chalk.magenta('[MEM ]'),
};

export class Terminal {
  constructor() {
    this.startTime = Date.now();
    this._bar = null;
  }

  _ts() {
    const s = ((Date.now() - this.startTime) / 1000).toFixed(1);
    return chalk.dim(`+${s}s`);
  }

  log(msg, level = 'info') {
    const tag = SEVERITY[level] || SEVERITY.info;
    console.log(`${this._ts()} ${tag} ${msg}`);
  }

  banner() {
    console.log(chalk.cyan('\n ══════════════════════════════════════════════'));
    console.log(chalk.cyan(' ║') + chalk.bold(' BRYMAR COLLEGE ') + chalk.dim('ConsciousNode SoftWorks') + chalk.cyan(' ║'));
    console.log(chalk.cyan(' ║') + chalk.dim(' LoRA fine-tuning studio for RWKV-v7 ') + chalk.cyan('║'));
    console.log(chalk.cyan(' ══════════════════════════════════════════════\n'));
  }

  hardwareReport(hw) {
    this.log(`CPU: ${hw.cpu_threads} threads, L3 ${hw.l3_cache_bytes/1024/1024}MB`, 'mem');
    this.log(`RAM: ${hw.physical_ram_gb}GB physical + ${hw.zram_headroom_gb}GB zRAM = ${hw.effective_ram_gb}GB effective`, 'mem');
    this.log(`GPU: GCN 1.0, ${hw.vram_gb}GB VRAM, OpenCL ${hw.opencl_enabled ? 'ENABLED' : 'DISABLED'}`, 'mem');
  }

  startProgress(total, label = 'Training') {
    if (process.env.NL_PORT || (process.stdout && !process.stdout.isTTY)) {
      this._bar = null;
      return;
    }
    this._bar = new SingleBar({
      format: ` ${chalk.cyan('{bar}')} {percentage}% | {value}/{total} steps | loss {loss} | {speed} tok/s`,
      barCompleteChar: '█',
      barIncompleteChar: '░',
      hideCursor: true,
    }, Presets.shades_classic);
    this._bar.start(total, 0, { loss: '?.????', speed: '0' });
  }

  updateProgress(step, loss, tokPerSec, total = '?') {
    if (this._bar) {
      this._bar.update(step, {
        loss: loss.toFixed(4),
        speed: Math.round(tokPerSec).toString(),
      });
    } else {
      this.log(`step=${step}/${total} loss=${loss.toFixed(4)} ${Math.round(tokPerSec)} tok/s`, 'train');
    }
  }

  stopProgress() {
    if (this._bar) { 
      this._bar.stop(); 
      this._bar = null; 
    }
  }

  memStats(physMB, zramMB) {
    this.log(`Phys: ${physMB.toFixed(0)}MB | zRAM: ${zramMB.toFixed(0)}MB`, 'mem');
  }
}
