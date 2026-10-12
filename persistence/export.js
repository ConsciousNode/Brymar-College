// persistence/export.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Model export — Phase 3 + Phase 5
//
// Export formats:
//   .rofl     — Recurrent Omnimodal Friston Language model (FP32, full precision)
//   .lmao     — Language Model, Active-inference, Omnimodal (Q4, 4-bit symmetric)
//   .fuctrump — Fristonian Underpinning Capabilities Transformer-agnostic
//               Recurrent Unconcerned with Model Parameters (Q2, 2-bit ternary)
//   .json     — Legacy JSON format (backward compatible)

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, basename } from 'node:path';
import { EXPORT_FORMATS } from '../src/config.js';
import { quantizeQ4, quantizeQ2, serializeWeights } from './quantize.js';

/**
 * Export a trained model from a checkpoint file.
 *
 * @param {Object} opts
 * @param {string} opts.checkpoint - Path to a checkpoint-XXXXXX.json file
 * @param {string} [opts.output]   - Output directory (default './exports/')
 * @param {string} [opts.format]   - 'rofl' | 'lmao' | 'fuctrump' | 'json' (default 'rofl')
 * @param {import('../src/ui/terminal.js').Terminal} [terminal]
 */
export async function exportAdapter(opts, terminal) {
  const log = (msg, level = 'info') =>
    terminal ? terminal.log(msg, level) : console.log(`[${level.toUpperCase()}] ${msg}`);

  if (!opts.checkpoint) {
    log('No checkpoint specified. Use --checkpoint <path>', 'error');
    return;
  }

  const format = opts.format ?? 'rofl';

  log(`Loading checkpoint: ${opts.checkpoint}`, 'info');

  let payload;
  try {
    const raw = await readFile(opts.checkpoint, 'utf8');
    payload = JSON.parse(raw);
  } catch (e) {
    log(`Failed to read checkpoint: ${e.message}`, 'error');
    return;
  }

  if (payload.version !== 'brymar-v1') {
    log(`Unknown checkpoint version "${payload.version}"`, 'error');
    return;
  }

  const outDir  = opts.output ?? './exports/';
  const stem    = basename(opts.checkpoint, '.json');

  await mkdir(outDir, { recursive: true });

  // ── Legacy JSON export ─────────────────────────────────────────────────────
  if (format === 'json') {
    const outFile = join(outDir, `${stem}.json`);
    const exportPayload = {
      ...payload,
      exportedAt: new Date().toISOString(),
      exportFormat: 'json'
    };
    await writeFile(outFile, JSON.stringify(exportPayload, null, 2), 'utf8');
    log(`Exported → ${outFile}`, 'ok');
    log(`  Adapters: ${payload.adapters?.length ?? 0} | Step: ${payload.step} | Loss: ${payload.loss?.toFixed(4)}`, 'info');
    return;
  }

  // ── ConsciousNode format export (.rofl / .lmao / .fuctrump) ────────────────
  const formatDef = EXPORT_FORMATS[format];
  if (!formatDef) {
    log(`Unknown export format: "${format}". Valid: rofl, lmao, fuctrump, json`, 'error');
    return;
  }

  const ext     = formatDef.extension;
  const outFile = join(outDir, `${stem}${ext}`);

  log(`Format: ${format.toUpperCase()} — ${formatDef.fullName}`, 'info');
  log(`  ${formatDef.description}`, 'info');

  // Collect all weight data from the checkpoint
  const allWeights = [];
  const weightMeta = [];

  // Full weights (from full-weight checkpoint)
  if (payload.weights && payload.weights.length > 0) {
    for (const w of payload.weights) {
      allWeights.push(new Float32Array(w.data));
      weightMeta.push({ name: w.name, rows: w.rows, cols: w.cols, length: w.data.length });
    }
  }

  // LoRA adapters
  if (payload.adapters && payload.adapters.length > 0) {
    for (let i = 0; i < payload.adapters.length; i++) {
      const a = payload.adapters[i];
      allWeights.push(new Float32Array(a.A));
      weightMeta.push({ name: `lora.${i}.A`, rows: a.rank, cols: a.inDim, length: a.A.length });
      allWeights.push(new Float32Array(a.B));
      weightMeta.push({ name: `lora.${i}.B`, rows: a.outDim, cols: a.rank, length: a.B.length });
    }
  }

  if (allWeights.length === 0) {
    log('No weights found in checkpoint — nothing to export', 'warn');
    return;
  }

  // Concatenate all weights into a single buffer
  const totalLen = allWeights.reduce((sum, w) => sum + w.length, 0);
  const flatWeights = new Float32Array(totalLen);
  let offset = 0;
  for (const w of allWeights) {
    flatWeights.set(w, offset);
    offset += w.length;
  }

  log(`Total weights: ${totalLen.toLocaleString()} parameters across ${allWeights.length} tensors`, 'info');

  // Build the binary payload
  let quantData = null;
  if (format === 'lmao') {
    log('Quantizing to Q4 (4-bit symmetric, group size 32)…', 'info');
    quantData = quantizeQ4(flatWeights, formatDef.groupSize);
    const ratio = (flatWeights.length * 4) / (quantData.packed.length + quantData.scales.length * 4);
    log(`  Compression ratio: ${ratio.toFixed(1)}×`, 'ok');
  } else if (format === 'fuctrump') {
    log('Quantizing to Q2 (2-bit ternary)…', 'info');
    quantData = quantizeQ2(flatWeights);
    const ratio = (flatWeights.length * 4) / quantData.packed.length;
    log(`  Compression ratio: ${ratio.toFixed(1)}×`, 'ok');
  }

  const weightBuf = serializeWeights(format, flatWeights, quantData);

  // Build metadata header (JSON, length-prefixed before binary payload)
  const metadata = {
    format:       format,
    formatName:   formatDef.fullName,
    version:      'brymar-v1',
    quantBits:    formatDef.quantBits,
    step:         payload.step,
    loss:         payload.loss,
    mode:         payload.mode ?? 'lora',
    modelCfg:     payload.modelCfg,
    tensors:      weightMeta,
    exportedAt:   new Date().toISOString(),
    totalParams:  totalLen,
  };
  const metaJSON = JSON.stringify(metadata);
  const metaBuf  = Buffer.from(metaJSON, 'utf8');

  // Final file: [4 bytes metaLen LE] [meta JSON] [weight binary]
  const finalBuf = Buffer.alloc(4 + metaBuf.length + weightBuf.length);
  finalBuf.writeUInt32LE(metaBuf.length, 0);
  metaBuf.copy(finalBuf, 4);
  weightBuf.copy(finalBuf, 4 + metaBuf.length);

  await writeFile(outFile, finalBuf);

  const sizeMB = (finalBuf.length / (1024 * 1024)).toFixed(2);
  log(`Exported → ${outFile} (${sizeMB} MB)`, 'ok');
  log(`  Format: ${formatDef.fullName}`, 'info');
  log(`  Step: ${payload.step} | Loss: ${payload.loss?.toFixed(4)} | Tensors: ${weightMeta.length}`, 'info');
}
