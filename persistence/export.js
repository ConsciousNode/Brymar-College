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

  // brymar-v2: rebuild the model (base + adapters, or full weights) and merge LoRA into it
  const { RWKV7 } = await import('../model/rwkv7.js');
  const { loadRWKV7From, modelTensors, writeSafetensors } = await import('../model/rwkv7_load.js');
  const { readCheckpointMeta, loadCheckpoint } = await import('./checkpoint.js');
  let payload, model;
  try {
    payload = await readCheckpointMeta(opts.checkpoint);
    if (payload.baseModel && !payload.weightsFile) ({ model } = await loadRWKV7From(opts.model || payload.baseModel, log, { RWKV7 }));
    else model = new RWKV7(payload.modelCfg);
    await loadCheckpoint(opts.checkpoint, model);
  } catch (e) { log(`Cannot export: ${e.message}`, 'error'); return; }
  if (Object.keys(model.lora).length) { model.mergeLoRA(); log('LoRA adapters merged into the base weights', 'ok'); }

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

  if (format === 'rofl') {
    // .rofl = the merged model in full precision, as a standard .safetensors with official RWKV-7 names
    writeSafetensors(outFile, modelTensors(model), { format: 'rofl', fullName: formatDef.fullName, step: payload.step, loss: payload.loss, tokenizer: payload.tokenizer ?? '' });
    const { statSync } = await import('node:fs');
    log(`Exported → ${outFile} (${(statSync(outFile).size / 1048576).toFixed(1)} MB, safetensors, loadable with --model)`, 'ok');
    return;
  }

  log(`Format: ${format.toUpperCase()} — ${formatDef.fullName}`, 'info');
  log(`  ${formatDef.description}`, 'info');

  // Collect the merged model's weights (official names)
  const allWeights = [], weightMeta = [];
  for (const t of modelTensors(model)) { allWeights.push(t.data instanceof Float32Array ? t.data : Float32Array.from(t.data)); weightMeta.push({ name: t.name, shape: t.shape, length: t.data.length }); }

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
    version:      'brymar-v2',
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
