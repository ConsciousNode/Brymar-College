// model/loader.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// RWKV-v7 model weight loader from HuggingFace

import { downloadFile } from '@huggingface/hub';
import { readFile } from 'fs/promises';
import { RWKVModel } from './model.js';
import { injectLoRA } from './lora.js';
import { gguf } from '@huggingface/gguf';
import unzipper from 'unzipper';
import fs from 'node:fs';

/**
 * Load RWKV-v7 model weights from HuggingFace or Local File.
 * 
 * Downloads weights if not cached, loads into RWKVModel, and configures
 * for either LoRA or full-weight training.
 * 
 * @param {Object} cfg - Model configuration
 * @param {string} cfg.modelName - HuggingFace model name or local path
 * @param {number} cfg.loraRank - LoRA rank
 * @param {number} cfg.loraAlpha - LoRA alpha
 * @param {string} [cfg.trainingMode='lora'] - 'lora' | 'full'
 * @param {Object} terminal - Terminal instance for logging
 * @returns {Promise<RWKVModel>} Loaded model
 */
export async function loadModel(cfg, terminal) {
  terminal.log(`Loading model: ${cfg.modelName}`, 'info');
  
  const trainingMode = cfg.trainingMode ?? 'lora';
  
  let weightPath = null;
  let isSafetensors = false;
  let isGGUF = false;
  let isPth = false;
  
  if (fs.existsSync(cfg.modelName) && fs.statSync(cfg.modelName).isFile()) {
    weightPath = cfg.modelName;
    if (weightPath.endsWith('.safetensors')) isSafetensors = true;
    else if (weightPath.endsWith('.gguf')) isGGUF = true;
    else if (weightPath.endsWith('.pth') || weightPath.endsWith('.bin')) isPth = true;
    terminal.log(`Using local file: ${weightPath}`, 'info');
  } else {
    try {
      terminal.log('Attempting to download .safetensors from HuggingFace...', 'info');
      weightPath = await downloadFile({
        repo: cfg.modelName,
        path: 'model.safetensors'
      });
      isSafetensors = true;
    } catch (error) {
      terminal.log('.safetensors not found or download failed, trying .gguf...', 'warn');
      try {
        weightPath = await downloadFile({
          repo: cfg.modelName,
          path: 'model.gguf'
        });
        isGGUF = true;
      } catch (e2) {
        terminal.log('.gguf not found, falling back to .pth...', 'warn');
        try {
          weightPath = await downloadFile({
            repo: cfg.modelName,
            path: 'pytorch_model.bin'
          });
          isPth = true;
        } catch (e3) {
          terminal.log(`Failed to download weights: ${e3.message}`, 'warn');
        }
      }
    }
  }
  
  let tensors = {};
  let autoCfg = { vocabSize: 65536, dim: 2048, numLayers: 24, dimFFN: 5461 }; // Fallback
  
  if (weightPath) {
    terminal.log(`Weights resolved at: ${weightPath}`, 'info');
    
    try {
      if (isSafetensors) {
        terminal.log(`Detected format: SAFETENSORS`, 'info');
        tensors = await loadSafetensors(weightPath);
      } else if (isGGUF) {
        terminal.log(`Detected format: GGUF`, 'info');
        tensors = await loadGGUF(weightPath);
      } else if (isPth) {
        terminal.log(`Detected format: PYTORCH (.pth / .bin)`, 'info');
        tensors = await loadWeightsPth(weightPath, terminal);
      } else {
        terminal.log(`Unknown format, trying SAFETENSORS...`, 'warn');
        tensors = await loadSafetensors(weightPath);
      }
      
      if (Object.keys(tensors).length > 0) {
        autoCfg = autoDetectConfig(tensors);
        const sizeB = ((autoCfg.dim * autoCfg.dim * 4 * autoCfg.numLayers) / 1e9).toFixed(1);
        terminal.log(`Auto-detected architecture: ~${sizeB}B (vocab: ${autoCfg.vocabSize}, dim: ${autoCfg.dim}, layers: ${autoCfg.numLayers}, dimFFN: ${autoCfg.dimFFN})`, 'info');
      }
    } catch (e) {
      terminal.log(`Failed to parse weights: ${e.message}`, 'error');
    }
  } else {
    terminal.log('Using randomly initialized weights', 'warn');
  }
  
  // Merge auto-detected config with passed-in config
  const finalCfg = { ...autoCfg, ...cfg };
  
  // Create model structure
  const model = new RWKVModel({
    vocabSize: finalCfg.vocabSize,
    dim: finalCfg.dim,
    numLayers: finalCfg.numLayers,
    dimFFN: finalCfg.dimFFN
  });
  
  terminal.log('Initializing base weights...', 'info');
  
  if (Object.keys(tensors).length > 0) {
    applyWeightsToModel(model, tensors, terminal);
  }
  
  // Configure for training mode
  configureTrainingMode(model, finalCfg, trainingMode, terminal);
  
  terminal.log('Model loaded successfully', 'ok');
  
  return model;
}

/**
 * Configure model for the specified training mode.
 * 
 * @param {RWKVModel} model
 * @param {Object} cfg
 * @param {string} mode - 'lora' | 'full'
 * @param {Object} terminal
 */
function configureTrainingMode(model, cfg, mode, terminal) {
  if (mode === 'full') {
    // Full-weight: unfreeze everything, skip LoRA
    unfreezeBaseModel(model, terminal);
    terminal.log('Mode: Full-weight training — all parameters trainable', 'info');
  } else {
    // LoRA: freeze base, inject adapters
    freezeBaseModel(model, terminal);
    terminal.log(`Injecting LoRA adapters (rank=${cfg.loraRank}, alpha=${cfg.loraAlpha})`, 'info');
    injectLoRA(model, cfg.loraRank, cfg.loraAlpha);
  }
}

/**
 * Freeze all base model layers.
 * 
 * @param {RWKVModel} model - Model instance
 * @param {Object} terminal - Terminal for logging
 */
function freezeBaseModel(model, terminal) {
  let frozenCount = 0;
  
  // Freeze all BitLinear layers in blocks
  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i];
    const layers = block.getLinearLayers();
    
    for (const layer of layers) {
      layer.frozen = true;
      frozenCount++;
    }
  }
  
  // Freeze head
  model.head.frozen = true;
  frozenCount++;
  
  terminal.log(`Froze ${frozenCount} base layers`, 'info');
}

/**
 * Unfreeze all base model layers for full-weight training.
 * 
 * @param {RWKVModel} model - Model instance
 * @param {Object} terminal - Terminal for logging
 */
function unfreezeBaseModel(model, terminal) {
  let unfrozenCount = 0;
  
  // Unfreeze all BitLinear layers in blocks
  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i];
    const layers = block.getLinearLayers();
    
    for (const layer of layers) {
      layer.frozen = false;
      unfrozenCount++;
    }
  }
  
  // Unfreeze head
  model.head.frozen = false;
  unfrozenCount++;
  
  terminal.log(`Unfroze ${unfrozenCount} base layers for full-weight training`, 'info');
}

/**
 * Map HuggingFace weight keys to model attributes.
 * 
 * RWKV-v7 World format:
 * - blocks.{i}.ln1.weight → blocks[i].ln1.weight
 * - blocks.{i}.ln1.bias → blocks[i].ln1.bias
 * - blocks.{i}.att.receptance.weight → blocks[i].att_r.weight
 * - etc.
 * 
 * @param {string} key - HuggingFace weight key
 * @returns {string|null} Model attribute path or null if not mapped
 */
function mapWeightKey(key) {
  if (key === 'emb.weight') return 'embedding';
  if (key === 'blocks.0.ln0.weight') return 'ln_in.weight';
  if (key === 'blocks.0.ln0.bias') return 'ln_in.bias';
  
  const blockMatch = key.match(/^blocks\.(\d+)\.(.+)$/);
  if (blockMatch) {
    const b = blockMatch[1];
    const rest = blockMatch[2];
    
    // LayerNorms
    if (rest === 'ln1.weight') return `blocks[${b}].ln1.weight`;
    if (rest === 'ln1.bias') return `blocks[${b}].ln1.bias`;
    if (rest === 'ln2.weight') return `blocks[${b}].ln2.weight`;
    if (rest === 'ln2.bias') return `blocks[${b}].ln2.bias`;
    
    // Time Mix Projections
    if (rest === 'att.receptance.weight') return `blocks[${b}].att_r.weight`;
    if (rest === 'att.key.weight') return `blocks[${b}].att_k.weight`;
    if (rest === 'att.value.weight') return `blocks[${b}].att_v.weight`;
    if (rest === 'att.output.weight') return `blocks[${b}].att_o.weight`;
    if (rest === 'att.gate.weight') return `blocks[${b}].att_g.weight`;
    
    // Time Mix Vectors (PoST)
    if (rest === 'att.time_mix_w') return `blocks[${b}].W_x_w.weight`;
    if (rest === 'att.time_mix_k') return `blocks[${b}].W_x_kappa.weight`;
    if (rest === 'att.time_mix_a') return `blocks[${b}].W_x_a.weight`;
    
    // Channel Mix Projections
    if (rest === 'ffn.receptance.weight') return `blocks[${b}].Wr2.weight`;
    if (rest === 'ffn.key.weight') return `blocks[${b}].Wk2.weight`;
    if (rest === 'ffn.value.weight') return `blocks[${b}].Wv2.weight`;
    
    // Channel Mix Vectors
    if (rest === 'ffn.time_mix_k') return `blocks[${b}].W_x_kx2.weight`;
    if (rest === 'ffn.time_mix_r') return `blocks[${b}].W_x_rx2.weight`;
  }
  
  // Head
  if (key === 'ln_out.weight') return 'ln_out.weight';
  if (key === 'ln_out.bias') return 'ln_out.bias';
  if (key === 'head.weight') return 'head.weight';
  
  return null;
}

/**
 * Load weights from a .safetensors file natively without dependencies.
 * 
 * Parses the 8-byte JSON header length, decodes the metadata, and maps
 * the binary offsets directly to Float32Array views (converting BF16/FP16 if necessary).
 * 
 * @param {string} path - Path to .safetensors file
 * @returns {Promise<Object>} Dictionary mapping string keys to tensor objects
 */
async function loadSafetensors(path) {
  const buffer = await readFile(path);
  const headerLength = Number(buffer.readBigUInt64LE(0));
  const headerStr = buffer.toString('utf8', 8, 8 + headerLength);
  let header;
  try {
    header = JSON.parse(headerStr);
  } catch (e) {
    throw new Error('Invalid Safetensors JSON header');
  }
  
  const tensorOffset = 8 + headerLength;
  const tensors = {};
  
  for (const [key, meta] of Object.entries(header)) {
    if (key === '__metadata__') continue;
    
    const start = tensorOffset + meta.data_offsets[0];
    const end = tensorOffset + meta.data_offsets[1];
    const byteLength = end - start;
    
    let arr;
    if (meta.dtype === 'F32') {
      arr = new Float32Array(buffer.buffer, buffer.byteOffset + start, byteLength / 4);
    } else if (meta.dtype === 'BF16') {
      const f16view = new Uint16Array(buffer.buffer, buffer.byteOffset + start, byteLength / 2);
      arr = new Float32Array(f16view.length);
      const u32view = new Uint32Array(arr.buffer);
      for (let i = 0; i < f16view.length; i++) {
        u32view[i] = f16view[i] << 16;
      }
    } else if (meta.dtype === 'F16') {
      const f16view = new Uint16Array(buffer.buffer, buffer.byteOffset + start, byteLength / 2);
      arr = new Float32Array(f16view.length);
      for (let i = 0; i < f16view.length; i++) {
        const h = f16view[i];
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;
        if (e === 0) {
          arr[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        } else if (e === 0x1F) {
          arr[i] = f ? NaN : (s ? -Infinity : Infinity);
        } else {
          arr[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
        }
      }
    } else {
      throw new Error(`Unsupported tensor dtype: ${meta.dtype} for ${key}`);
    }
    
    tensors[key] = { shape: meta.shape, data: arr };
  }
  
  return tensors;
}

/**
 * Load weights from a .gguf file.
 * 
 * @param {string} path - Path to .gguf file
 * @returns {Promise<Object>} Dictionary mapping string keys to tensor objects
 */
async function loadGGUF(path) {
  const parsed = await gguf(path, { allowLocalFile: true });
  const fd = fs.openSync(path, 'r');
  const tensors = {};
  const dataOffset = Number(parsed.tensorDataOffset);
  
  for (const info of parsed.tensorInfos) {
    const offset = dataOffset + Number(info.offset);
    const shape = [...info.shape].reverse(); // Reverse GGUF shape to match PyTorch/RWKV layout
    
    // Map GGUF names to HuggingFace names
    let hfName = info.name;
    if (hfName === 'token_embd.weight') hfName = 'emb.weight';
    else if (hfName === 'output_norm.weight') hfName = 'ln_out.weight';
    else if (hfName === 'output_norm.bias') hfName = 'ln_out.bias';
    else if (hfName === 'output.weight') hfName = 'head.weight';
    else {
      const blockMatch = hfName.match(/^blk\.(\d+)\.(.+)$/);
      if (blockMatch) {
        const b = blockMatch[1];
        const rest = blockMatch[2];
        if (rest === 'attn_norm.weight') hfName = `blocks.${b}.ln1.weight`;
        else if (rest === 'attn_norm.bias') hfName = `blocks.${b}.ln1.bias`;
        else if (rest === 'ffn_norm.weight') hfName = `blocks.${b}.ln2.weight`;
        else if (rest === 'ffn_norm.bias') hfName = `blocks.${b}.ln2.bias`;
        else if (rest === 'attn_receptance.weight') hfName = `blocks.${b}.att.receptance.weight`;
        else if (rest === 'attn_key.weight') hfName = `blocks.${b}.att.key.weight`;
        else if (rest === 'attn_value.weight') hfName = `blocks.${b}.att.value.weight`;
        else if (rest === 'attn_output.weight') hfName = `blocks.${b}.att.output.weight`;
        else if (rest === 'attn_gate.weight') hfName = `blocks.${b}.att.gate.weight`;
        else if (rest === 'attn_time_mix_w') hfName = `blocks.${b}.att.time_mix_w`;
        else if (rest === 'attn_time_mix_k') hfName = `blocks.${b}.att.time_mix_k`;
        else if (rest === 'attn_time_mix_a') hfName = `blocks.${b}.att.time_mix_a`;
        else if (rest === 'ffn_receptance.weight') hfName = `blocks.${b}.ffn.receptance.weight`;
        else if (rest === 'ffn_key.weight') hfName = `blocks.${b}.ffn.key.weight`;
        else if (rest === 'ffn_value.weight') hfName = `blocks.${b}.ffn.value.weight`;
        else if (rest === 'ffn_time_mix_k') hfName = `blocks.${b}.ffn.time_mix_k`;
        else if (rest === 'ffn_time_mix_r') hfName = `blocks.${b}.ffn.time_mix_r`;
      }
    }
    
    if (info.dtype === 0) { // GGML_TYPE_F32
      const byteLength = shape.reduce((a, b) => a * b, 1) * 4;
      const buf = Buffer.alloc(byteLength);
      fs.readSync(fd, buf, 0, byteLength, offset);
      tensors[hfName] = { shape, data: new Float32Array(buf.buffer, buf.byteOffset, byteLength / 4) };
    } else if (info.dtype === 1) { // GGML_TYPE_F16
      const byteLength = shape.reduce((a, b) => a * b, 1) * 2;
      const buf = Buffer.alloc(byteLength);
      fs.readSync(fd, buf, 0, byteLength, offset);
      const f16view = new Uint16Array(buf.buffer, buf.byteOffset, byteLength / 2);
      const f32arr = new Float32Array(f16view.length);
      for (let i = 0; i < f16view.length; i++) {
        const h = f16view[i];
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;
        if (e === 0) {
          f32arr[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        } else if (e === 0x1F) {
          f32arr[i] = f ? NaN : (s ? -Infinity : Infinity);
        } else {
          f32arr[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
        }
      }
      tensors[hfName] = { shape, data: f32arr };
    }
  }
  fs.closeSync(fd);
  return tensors;
}

/**
 * Load weights from a PyTorch .pth / .bin file natively in JS (best effort).
 * 
 * Scans the data.pkl for tensor names and assumes the sequential data/* files 
 * in the ZIP map directly to these tensor names. 
 * 
 * @param {string} path - Path to weight file
 * @param {Object} terminal - Terminal instance
 * @returns {Promise<Object>} Weight dictionary
 */
async function loadWeightsPth(path, terminal) {
  terminal.log(`Detected chunked PyTorch storage. Attempting to convert to .safetensors via Python...`, 'info');
  
  const { execSync } = await import('node:child_process');
  const { join } = await import('node:path');
  const { tmpdir } = await import('node:os');
  
  const tempSafeTensors = join(tmpdir(), `brymar_converted_${Date.now()}.safetensors`);
  
  try {
    // Run the conversion script
    execSync(`python3 scripts/convert_pth.py "${path}" "${tempSafeTensors}"`, {
      encoding: 'utf8',
      stdio: 'pipe' // Capture output so it doesn't pollute the main process stdio
    });
    
    terminal.log(`Successfully converted .pth to .safetensors format natively.`, 'ok');
    
    // Now just load the safetensors file
    const tensors = await loadSafetensors(tempSafeTensors);
    
    // Cleanup the temp file
    try {
      const fs = await import('node:fs');
      fs.unlinkSync(tempSafeTensors);
    } catch(e) {}
    
    return tensors;
    
  } catch (e) {
    throw new Error(`Failed to convert .pth file: ${e.message}. Is Python 3 installed?`);
  }
}

/**
 * Automatically determine RWKV architecture dimensions from loaded tensors.
 * 
 * @param {Object} tensors - Loaded tensor dictionary
 * @returns {Object} { vocabSize, dim, numLayers, dimFFN }
 */
function autoDetectConfig(tensors) {
  const cfg = {};
  
  if (tensors['emb.weight']) {
    cfg.vocabSize = tensors['emb.weight'].shape[0];
    cfg.dim = tensors['emb.weight'].shape[1];
  } else {
    throw new Error('Failed to auto-detect: emb.weight not found');
  }
  
  let maxLayer = -1;
  for (const key of Object.keys(tensors)) {
    const match = key.match(/^blocks\.(\d+)\./);
    if (match) {
      const l = parseInt(match[1], 10);
      if (l > maxLayer) maxLayer = l;
    }
  }
  if (maxLayer >= 0) {
    cfg.numLayers = maxLayer + 1;
  } else {
    throw new Error('Failed to auto-detect: no blocks found');
  }
  
  const ffnKey = 'blocks.0.ffn.value.weight';
  if (tensors[ffnKey]) {
    // FFN Value projection is [dim, dimFFN]
    // Or depending on orientation, but usually dimFFN is the larger dimension
    cfg.dimFFN = Math.max(tensors[ffnKey].shape[0], tensors[ffnKey].shape[1]);
  } else {
    // Default RWKV formula if not found
    cfg.dimFFN = Math.floor((cfg.dim * 3.5) / 32) * 32;
  }
  
  return cfg;
}

/**
 * Apply loaded tensor data into the instantiated model object.
 * 
 * @param {RWKVModel} model - Instantiated model
 * @param {Object} tensors - Loaded tensors dictionary
 * @param {Object} terminal - Terminal for logging
 */
function applyWeightsToModel(model, tensors, terminal) {
  let mapped = 0;
  let missed = 0;
  
  for (const [key, tensor] of Object.entries(tensors)) {
    const modelPath = mapWeightKey(key);
    if (!modelPath) {
      missed++;
      continue;
    }
    
    try {
      // Evaluate the path to get the target float32 array
      // e.g. "blocks[0].att_r.weight"
      const pathParts = modelPath.split('.');
      let target = model;
      for (let i = 0; i < pathParts.length - 1; i++) {
        let part = pathParts[i];
        if (part.includes('[')) {
          const prop = part.split('[')[0];
          const idx = parseInt(part.match(/\[(\d+)\]/)[1]);
          target = target[prop][idx];
        } else {
          target = target[part];
        }
      }
      
      const leafProp = pathParts[pathParts.length - 1];
      
      if (target[leafProp] && target[leafProp].length === tensor.data.length) {
        target[leafProp].set(tensor.data);
        mapped++;
      }
    } catch (e) {
      terminal.log(`Failed to apply weight ${key} -> ${modelPath}: ${e.message}`, 'warn');
    }
  }
  
  terminal.log(`Applied ${mapped} weight tensors (ignored ${missed})`, 'ok');
}
