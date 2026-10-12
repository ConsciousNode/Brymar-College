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
  .description('ConsciousNode LoRA & full-weight fine-tuning studio for RWKV-v7')
  .version('3.3.4');

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
  .action(async (opts) => {
    t.banner();
    t.log('Evaluation mode', 'info');

    if (!opts.checkpoint) {
      t.log('No checkpoint specified. Use --checkpoint <path>', 'error');
      return;
    }

    const maxTokens   = parseInt(opts.maxTokens) || 128;
    const temperature = parseFloat(opts.temperature) || 0;

    // Dynamic imports
    const { RWKVModel }   = await import('./model/model.js');
    const { loadModel }   = await import('./model/loader.js');
    const { loadCheckpoint } = await import('./persistence/checkpoint.js');
    const { readFile }    = await import('node:fs/promises');

    // Read checkpoint to get modelCfg before loading
    t.log(`Loading checkpoint: ${opts.checkpoint}`, 'info');
    let payload;
    try {
      const raw = await readFile(opts.checkpoint, 'utf8');
      payload = JSON.parse(raw);
    } catch (e) {
      t.log(`Failed to read checkpoint: ${e.message}`, 'error');
      return;
    }

    if (!payload.modelCfg) {
      t.log('Checkpoint missing modelCfg — cannot reconstruct model', 'error');
      return;
    }

    // Build model from saved config
    const model = new RWKVModel(payload.modelCfg);
    t.log(`Model: ${payload.modelCfg.numLayers}L × dim${payload.modelCfg.dim} (vocab ${payload.modelCfg.vocabSize})`, 'info');

    // Restore weights
    try {
      await loadCheckpoint(opts.checkpoint, model);
      t.log('Checkpoint restored', 'ok');
    } catch (e) {
      t.log(`Checkpoint restore failed: ${e.message}`, 'error');
      return;
    }

    // Encode prompt as bytes (byte-level until BPE tokenizer state is checkpointed)
    const promptBytes = Array.from(new TextEncoder().encode(opts.prompt));
    t.log(`Prompt: "${opts.prompt}" (${promptBytes.length} tokens)`, 'info');
    t.log(`Generating ${maxTokens} tokens (temp=${temperature})...`, 'info');
    t.log('---', 'info');

    // Autoregressive inference
    let state = model.getEmptyState();
    let generated = '';

    // Feed prompt tokens
    for (const tok of promptBytes) {
      const result = model.forward(tok, state);
      state = result.state;
    }

    // Generate
    let lastLogits = null;
    {
      const result = model.forward(promptBytes[promptBytes.length - 1], state);
      state = result.state;
      lastLogits = result.logits;
    }

    for (let i = 0; i < maxTokens; i++) {
      let nextToken;

      if (temperature <= 0) {
        // Greedy: argmax
        let maxVal = -Infinity;
        nextToken = 0;
        for (let j = 0; j < lastLogits.length; j++) {
          if (lastLogits[j] > maxVal) {
            maxVal = lastLogits[j];
            nextToken = j;
          }
        }
      } else {
        // Temperature sampling
        const scaled = new Float32Array(lastLogits.length);
        let maxLogit = -Infinity;
        for (let j = 0; j < lastLogits.length; j++) {
          scaled[j] = lastLogits[j] / temperature;
          if (scaled[j] > maxLogit) maxLogit = scaled[j];
        }
        // Softmax
        let sumExp = 0;
        for (let j = 0; j < scaled.length; j++) {
          scaled[j] = Math.exp(scaled[j] - maxLogit);
          sumExp += scaled[j];
        }
        // Sample from distribution
        let r = Math.random() * sumExp;
        nextToken = 0;
        for (let j = 0; j < scaled.length; j++) {
          r -= scaled[j];
          if (r <= 0) { nextToken = j; break; }
        }
      }

      // Decode byte to character
      if (nextToken < 256) {
        try {
          const char = new TextDecoder().decode(new Uint8Array([nextToken]));
          generated += char;
          process.stdout.write(char);
        } catch {
          generated += '?';
          process.stdout.write('?');
        }
      }

      // Next step
      const result = model.forward(nextToken, state);
      state = result.state;
      lastLogits = result.logits;
    }

    console.log(); // newline after generation
    t.log('---', 'info');
    t.log(`Generated ${generated.length} characters`, 'ok');
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
