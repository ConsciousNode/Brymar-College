// persistence/checkpoint.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// LoRA + Full-weight checkpoint save/load — Phase 3 + Phase 5
//
// LoRA mode: saves only LoRA adapter weights (A, B matrices) and training metadata.
// Full mode: saves entire model state (embeddings, LN, all BitLinear, head).
//
// Momentum buffers are deliberately excluded to halve storage footprint
// (mirrors the HTMLNLM.html persistence design note).
//
// Format: JSON with Float32Array values encoded as regular JS arrays.
// Files are written to <dir>/checkpoint-<step>.json

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

/**
 * Serialize a Float32Array to a plain JS Array for JSON storage.
 *
 * @param {Float32Array} arr
 * @returns {number[]}
 */
function f32ToArray(arr) {
  return Array.from(arr);
}

/**
 * Deserialize a plain JS Array back into a Float32Array.
 *
 * @param {number[]} arr
 * @returns {Float32Array}
 */
function arrayToF32(arr) {
  return new Float32Array(arr);
}

/**
 * Collect all LoRA adapter weight arrays from the model.
 *
 * Returns a list of { A, B, inDim, outDim, rank } objects —
 * one per adapter layer, in stable block × target-name order.
 *
 * @param {import('../model/model.js').RWKVModel} model
 * @returns {Array<Object>}
 */
function extractAdapters(model) {
  const adapters = model.getLoRALayers();
  return adapters.map(adapter => ({
    A:      f32ToArray(adapter.A),
    B:      f32ToArray(adapter.B),
    inDim:  adapter.inDim,
    outDim: adapter.outDim,
    rank:   adapter.rank,
    scale:  adapter.scale ?? 1.0
  }));
}

/**
 * Restore LoRA adapter weights into the model from a serialized list.
 *
 * @param {import('../model/model.js').RWKVModel} model
 * @param {Array<Object>} adapterData
 */
function restoreAdapters(model, adapterData) {
  const adapters = model.getLoRALayers();
  if (adapters.length !== adapterData.length) {
    throw new Error(
      `Checkpoint has ${adapterData.length} adapters but model has ${adapters.length}`
    );
  }
  for (let i = 0; i < adapters.length; i++) {
    adapters[i].A     = arrayToF32(adapterData[i].A);
    adapters[i].B     = arrayToF32(adapterData[i].B);
    adapters[i].scale = adapterData[i].scale ?? 1.0;
  }
}

/**
 * Collect all model weights for full-weight checkpointing.
 *
 * @param {import('../model/model.js').RWKVModel} model
 * @returns {Array<Object>}
 */
function extractFullWeights(model) {
  return model.getAllWeights().map(w => ({
    name:   w.name,
    data:   f32ToArray(w.data),
    rows:   w.rows,
    cols:   w.cols,
  }));
}

/**
 * Restore all model weights from a full checkpoint.
 *
 * @param {import('../model/model.js').RWKVModel} model
 * @param {Array<Object>} weightData
 */
function restoreFullWeights(model, weightData) {
  const modelWeights = model.getAllWeights();
  if (modelWeights.length !== weightData.length) {
    throw new Error(
      `Checkpoint has ${weightData.length} weight tensors but model has ${modelWeights.length}`
    );
  }
  for (let i = 0; i < modelWeights.length; i++) {
    const restored = arrayToF32(weightData[i].data);
    // Copy into existing buffer (preserves references held by layers)
    modelWeights[i].data.set(restored);
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Save a checkpoint to disk.
 *
 * @param {import('../model/model.js').RWKVModel} model    - Model with injected LoRA (or full weights)
 * @param {number}                                step     - Current global training step
 * @param {number}                                loss     - Current average loss
 * @param {string}                                dir      - Output directory (will be created)
 * @param {string}                                [mode='lora'] - 'lora' | 'full'
 * @returns {Promise<string>} Absolute path to the saved checkpoint file
 */
export async function saveCheckpoint(model, step, loss, dir, mode = 'lora') {
  // Ensure checkpoint directory exists
  await mkdir(dir, { recursive: true });

  const filename = `checkpoint-${String(step).padStart(6, '0')}.json`;
  const outPath  = join(dir, filename);

  const payload = {
    version:   'brymar-v1',
    mode,
    step,
    loss,
    timestamp: new Date().toISOString(),
    modelCfg: {
      vocabSize:  model.vocabSize,
      dim:        model.dim,
      numLayers:  model.numLayers,
      dimFFN:     model.dimFFN
    },
  };

  if (mode === 'full') {
    payload.weights = extractFullWeights(model);
    // Also save LoRA adapters if present (hybrid mode)
    const adapters = extractAdapters(model);
    if (adapters.length > 0) {
      payload.adapters = adapters;
    }
  } else {
    payload.adapters = extractAdapters(model);
  }

  await writeFile(outPath, JSON.stringify(payload), 'utf8');
  return outPath;
}

/**
 * Load a checkpoint from disk and restore model weights.
 *
 * @param {string}                                path  - Path to checkpoint .json file
 * @param {import('../model/model.js').RWKVModel} model - Model to restore into
 * @returns {Promise<number>} The step number stored in the checkpoint
 */
export async function loadCheckpoint(path, model) {
  const raw     = await readFile(path, 'utf8');
  const payload = JSON.parse(raw);

  if (payload.version !== 'brymar-v1') {
    throw new Error(`Unknown checkpoint version: ${payload.version}`);
  }

  // Restore full weights if present
  if (payload.weights) {
    restoreFullWeights(model, payload.weights);
  }

  // Restore LoRA adapters if present
  if (payload.adapters && payload.adapters.length > 0) {
    try {
      restoreAdapters(model, payload.adapters);
    } catch (e) {
      // Adapter count mismatch is non-fatal for full-weight checkpoints
      if (payload.mode !== 'full') throw e;
    }
  }

  return payload.step;
}
