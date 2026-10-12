// index.js
// Brymar College — CLI Entry Point
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Licensed under the MIT License
// Named for Deltron 3030's Upgrade. Because it trains in multimodality.

import { program } from 'commander';
import { Terminal } from './src/ui/terminal.js';
import { HARDWARE, MODEL, TRAINING, PATHS } from './src/config.js';
import { ZRamMonitor } from './hardware/zram.js';
import { OpenCLAccelerator } from './hardware/opencl.js';

const t = new Terminal();

program
  .name('brymar')
  .description('ConsciousNode LoRA & full-weight fine-tuning studio for real RWKV-7 (x070) models')
  .version('4.0.0');

program
  .command('train')
  .description('Fine-tune RWKV-v7 with LoRA or full-weight training on soul corpus')
  .option('-m, --model <name>', 'Base model name', MODEL.base)
  .option('-r, --rank <n>', 'LoRA rank', String(MODEL.lora_rank))
  .option('-s, --steps <n>', 'Training steps', String(TRAINING.total_steps))
  .option('--corpus <path>', 'Corpus directory', PATHS.corpus)
  .option('--resume <checkpoint>', 'Resume from checkpoint')
  .option('--mode <mode>', 'Training mode: lora | full', TRAINING.mode)
  .option('--schedule <type>', 'LR schedule: manual | cosine | step | exp', 'cosine')
  .option('--user-name <name>', 'User identity for JSON parsing', 'USER')
  .option('--ai-name <name>', 'AI identity for JSON parsing', 'CLAUDE')
  .option('--friston-alpha <val>', 'Friston Active Inference Entropy Penalty', parseFloat, 0.0)
  .option('--backend <type>', 'Compute backend: auto | cpu | opencl', 'auto')
  .action(async (opts) => {
    t.banner();
    t.hardwareReport(HARDWARE);

    // ── Hardware probes ─────────────────────────────────────────────────
    const zram = new ZRamMonitor();
    await zram.probe(t);

    const opencl = new OpenCLAccelerator();
    await opencl.probe(t, opts.backend);

    t.log('', 'info'); // visual separator

    const { TrainingLoop } = await import('./training/loop.js');
    const loop = new TrainingLoop({ ...opts, terminal: t, zram, opencl });

    process.on('SIGINT', () => loop.stop());
    process.on('SIGTERM', () => loop.stop());

    await loop.run();
  });

program
  .command('eval')
  .description('Run evaluation / inference on a trained checkpoint')
  .option('-c, --checkpoint <path>', 'Checkpoint to evaluate')
  .option('--prompt <text>', 'Evaluation prompt', 'Hello')
  .option('--max-tokens <n>', 'Maximum tokens to generate', '128')
  .option('--temperature <t>', 'Sampling temperature (0 = greedy)', '0')
  .option('-m, --model <path>', 'Base model weights (defaults to the one recorded in the checkpoint)')
  .action(async (opts) => {
    t.banner();
    t.log('Evaluation mode', 'info');

    if (!opts.checkpoint) {
      t.log('No checkpoint specified. Use --checkpoint <path>', 'error');
      return;
    }

    const maxTokens   = parseInt(opts.maxTokens) || 128;
    const temperature = parseFloat(opts.temperature) || 0;
    const { RWKV7 } = await import('./model/rwkv7.js');
    const { loadRWKV7From } = await import('./model/rwkv7_load.js');
    const { RWKVWorldTokenizer } = await import('./tokenizer/rwkv_world.js');
    const { readCheckpointMeta, loadCheckpoint } = await import('./persistence/checkpoint.js');

    let meta;
    try { meta = await readCheckpointMeta(opts.checkpoint); } catch (e) { t.log(`Cannot use checkpoint: ${e.message}`, 'error'); return; }
    let model;
    try {
      if (meta.baseModel && !meta.weightsFile) ({ model } = await loadRWKV7From(opts.model || meta.baseModel, (m, l) => t.log(m, l), { RWKV7 }));
      else model = new RWKV7(meta.modelCfg);
      await loadCheckpoint(opts.checkpoint, model);
    } catch (e) { t.log(`Could not build the model: ${e.message} (pass --model <base weights> if the base moved)`, 'error'); return; }
    t.log(`Checkpoint restored: step ${meta.step}, ${meta.mode} mode`, 'ok');

    const tok = model.V >= 65530 ? new RWKVWorldTokenizer() : null;
    const encode = s => tok ? tok.encode(s) : Array.from(new TextEncoder().encode(s));
    const promptIds = encode(opts.prompt);
    t.log(`Prompt: "${opts.prompt}" (${promptIds.length} tokens) · generating ${maxTokens} (temp ${temperature})`, 'info');
    t.log('---', 'info');

    // feed the prompt once; the last forward's logits predict the first new token
    let state = model.getEmptyState(), logits = null;
    for (const id of promptIds) { const r = model.forward(id, state); state = r.state; logits = r.logits; }
    const out = [];
    let printed = 0;
    for (let i = 0; i < maxTokens; i++) {
      let next = 0;
      if (temperature <= 0) { for (let j = 1; j < logits.length; j++) if (logits[j] > logits[next]) next = j; }
      else { let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v / temperature); let s = 0; const p = new Float64Array(logits.length);
        for (let j = 0; j < logits.length; j++) { p[j] = Math.exp(logits[j] / temperature - mx); s += p[j]; }
        let r = Math.random() * s; for (next = 0; next < p.length - 1; next++) { r -= p[next]; if (r <= 0) break; } }
      if (tok && next === 0) break;              // end-of-text
      out.push(next);
      const text = tok ? tok.decode(out) : new TextDecoder().decode(Uint8Array.from(out));
      const safe = text.replace(/�+$/, '');  // hold back a half-finished UTF-8 character
      process.stdout.write(safe.slice(printed)); printed = safe.length;
      const r = model.forward(next, state); state = r.state; logits = r.logits;
    }
    console.log();
    t.log('---', 'info');
    t.log(`Generated ${out.length} tokens`, 'ok');
  });

program
  .command('export')
  .description('Export trained model weights')
  .option('-c, --checkpoint <path>', 'Checkpoint to export')
  .option('-o, --output <path>', 'Output path', PATHS.exports)
  .option('-f, --format <format>', 'Export format: rofl | lmao | fuctrump | json', 'rofl')
  .action(async (opts) => {
    t.banner();
    const { exportAdapter } = await import('./persistence/export.js');
    await exportAdapter(opts, t);
  });

program.parse();
