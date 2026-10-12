// persistence/checkpoint.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
//
// Checkpoints for model/rwkv7.js (format brymar-v2, 2026-10-11).
//   LoRA mode: one small JSON: config + where the base model came from + every adapter by name.
//   Full mode: JSON metadata + a sidecar <name>.safetensors holding every parameter under its official
//              RWKV-7 name (a 0.1B model is ~760 MB as F32; JSON arrays cannot hold that).
// brymar-v1 checkpoints came from the pre-audit engine (see AUDIT.md) and are not loadable.

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, dirname, basename } from 'node:path';
import { writeSafetensors, readSafetensors, modelTensors, loadIntoRWKV7 } from '../model/rwkv7_load.js';

export async function saveCheckpoint(model, step, loss, dir, mode = 'lora', extra = {}) {
  await mkdir(dir, { recursive: true });
  const stem = `checkpoint-${String(step).padStart(6, '0')}`, outPath = join(dir, `${stem}.json`);
  const payload = { version: 'brymar-v2', arch: 'rwkv7', mode, step, loss, timestamp: new Date().toISOString(), modelCfg: model.cfg, baseModel: extra.baseModel ?? null, tokenizer: extra.tokenizer ?? null };
  if (mode === 'full') {
    payload.weightsFile = `${stem}.safetensors`;
    writeSafetensors(join(dir, payload.weightsFile), modelTensors(model), { format: 'brymar-v2', step });
  }
  payload.adapters = Object.entries(model.lora).map(([name, a]) => ({ name, rank: a.r, scale: a.scale, inDim: a.inD, outDim: a.outD, A: Array.from(a.A), B: Array.from(a.B) }));
  await writeFile(outPath, JSON.stringify(payload), 'utf8');
  return outPath;
}

/** Read a checkpoint's metadata (no weights). */
export async function readCheckpointMeta(path) {
  const payload = JSON.parse(await readFile(path, 'utf8'));
  if (payload.version === 'brymar-v1') throw new Error('this checkpoint is from Brymar ≤ v3.3.4, whose model was not RWKV-7 and whose gradients were wrong (see AUDIT.md); it cannot be loaded into the new engine');
  if (payload.version !== 'brymar-v2') throw new Error(`unknown checkpoint version: ${payload.version}`);
  return payload;
}

/** Restore weights (full mode) and/or LoRA adapters into an already-built RWKV7. Returns the step. */
export async function loadCheckpoint(path, model) {
  const payload = await readCheckpointMeta(path);
  if (payload.weightsFile) loadIntoRWKV7(model, readSafetensors(join(dirname(path), payload.weightsFile)));
  if (payload.adapters?.length) {
    const rank = payload.adapters[0].rank, scale = payload.adapters[0].scale;
    if (!Object.keys(model.lora).length) model.injectLoRA(rank, scale * rank, [...new Set(payload.adapters.map(a => a.name.split('.')[3]))]);
    for (const a of payload.adapters) {
      const dst = model.lora[a.name];
      if (!dst || dst.A.length !== a.A.length || dst.B.length !== a.B.length) throw new Error(`adapter ${a.name} does not fit this model`);
      dst.A.set(a.A); dst.B.set(a.B); dst.scale = a.scale;
    }
  }
  return payload.step;
}

export const checkpointStem = p => basename(p, '.json');
