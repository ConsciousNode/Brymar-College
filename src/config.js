// src/config.js
// Brymar College — Configuration Module
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Licensed under the MIT License

/**
 * Single source of truth for all configuration constants.
 * 
 * HARDWARE: Target hardware specifications (Intel Core i7 Sandy Bridge, AMD Radeon HD 7570)
 * MODEL: LoRA adapter configuration and target layers
 * TRAINING: Optimizer hyperparameters and training schedule
 * EXPORT_FORMATS: Quantized model export format definitions
 * CORPUS_EXTENSIONS: Supported file types for multimodal corpus ingestion
 * PATHS: Filesystem path defaults
 */

export const HARDWARE = {
  cpu_threads: 8,
  l3_cache_bytes: 8 * 1024 * 1024,
  physical_ram_gb: 12,
  zram_headroom_gb: 9,
  effective_ram_gb: 21,
  vram_gb: 1,
  opencl_enabled: false, // Phase 6
};

export const MODEL = {
  base: 'RWKV-v7-World-1.5B',
  lora_rank: 4,
  lora_alpha: 8,
  lora_targets: ['receptance', 'key', 'value', 'output'],
  lora_exclude: ['time_mix', 'time_decay', 'time_faaaa'],
};

export const TRAINING = {
  chunk_size: 512,
  grad_accum_steps: 4,
  lr: 0.005,
  muon_momentum: 0.95,
  muon_wd: 0.01,
  warmup_steps: 100,
  total_steps: 2000,
  min_lr: 0.0001,
  checkpoint_every: 200,
  modality_ratio: { text: 1.0, visual: 0.0, audio: 0.0 },
  mode: 'lora',  // 'lora' | 'full'
};

export const EXPORT_FORMATS = {
  rofl: {
    extension: '.rofl',
    fullName: 'Recurrent Omnimodal Friston Language model',
    quantBits: 32,
    description: 'Full-precision FP32 weights — no quantization',
  },
  lmao: {
    extension: '.lmao',
    fullName: 'Language Model, Active-inference, Omnimodal',
    quantBits: 4,
    groupSize: 32,
    description: 'Q4 symmetric quantization — balanced quality/size',
  },
  fuctrump: {
    extension: '.fuctrump',
    fullName: 'Fristonian Underpinning Capabilities Transformer-agnostic Recurrent Unconcerned with Model Parameters',
    quantBits: 2,
    description: 'Q2 ternary quantization — aggressive compression',
  },
};

export const CORPUS_EXTENSIONS = {
  text: new Set([
    '.txt', '.md', '.rst', '.csv', '.tsv', '.json', '.jsonl',
    '.xml', '.html', '.htm', '.yaml', '.yml', '.toml', '.log',
    '.rtf', '.tex', '.org', '.adoc',
  ]),
  image: new Set([
    '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif',
    '.gif', '.avif', '.svg', '.bmp',
  ]),
  audio: new Set([
    '.wav',
  ]),
  // Recognized and decoded via FFmpeg to temporary PCM buffers
  audioRequiresFFmpeg: new Set([
    '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.opus', '.wma',
  ]),
  videoRequiresFFmpeg: new Set([
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'
  ]),
};

export const PATHS = {
  corpus: './corpus/',
  models: './models/',
  checkpoints: './checkpoints/',
  exports: './exports/',
  logs: './logs/',
};
