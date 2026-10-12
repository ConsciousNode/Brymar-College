# Brymar College

**ConsciousNode LoRA & Full-Weight Fine-Tuning Studio for RWKV-v7**  
*v3.3.4 — Stable Release*

Brymar College (named for Deltron 3030's "Upgrade") is a fast, offline-first graphical training environment for RWKV-v7 models. It supports both **LoRA adapter fine-tuning** and **full-weight backpropagation** across mixed-modality text, audio, image, and video data.

## What's New in v3.3.4
- **OpenCL Compute Toggle:** Added a UI configuration dropdown to explicitly disable GPU acceleration in favor of CPU AVX path, addressing compilation issues with `node-opencl` on unsupported drivers.
- **PyTorch Storage Safeguard:** The JS tensor parser now seamlessly intercepts chunked PyTorch `.pth` storage files and transparently converts them to `.safetensors` via a zero-dependency Python background worker, preventing catastrophic architecture misalignment (e.g. 92M vocab size).
- **BPE Progress Latch:** Fixed a UI spam issue where floating point rounding caused `BPE progress: 0%` to flood the terminal for several seconds.
- **Telemetry Bridge Restored:** Fixed a persistent Dev Mode race condition during Neutralino boot by adding an async poll mechanism to wait for CLI security credentials.

## What is Brymar College?

Brymar College is a local, offline-first fine-tuning studio for RWKV-v7 language models.
It runs entirely on your machine — no cloud API, no telemetry, no Python, no GPU required.

You provide a corpus (text, images, audio). Brymar College teaches the model to sound like it.

Two training modes:
- **LoRA** — Fine-tune only small adapter matrices. Fast, memory-efficient.
- **Full Weight** — Train all model parameters end-to-end. Maximum expressiveness.

The architecture is omnimodal from the ground up: text, images, and audio feed the same
recurrent sequence. All trained models have native omnimodality and Friston active inference
baked in — differentiating them from standard text-to-text models.

---

## Features

| Feature | Status |
|---|---|
| BitNet b1.58 ternary base weights | ✅ |
| RWKV-v7 recurrent block (full forward + BPTT backward) | ✅ |
| **Full-weight training (all parameters)** | ✅ |
| LoRA adapter injection (rank-configurable) | ✅ |
| Muon optimizer (Newton-Schulz-5 orthogonalization) | ✅ |
| OOMB chunk-recurrent training (memory-efficient BPTT) | ✅ |
| NanoTokenizer (BPE, Worker Thread, interoperable) | ✅ |
| CorpusLoader (text + image + audio, 30+ formats) | ✅ |
| ElasticTok v2 (adaptive visual patch tokenization) | ✅ |
| SpikeVox v2 (LIF audio encoder, rate coding) | ✅ |
| MultimodalPipeline (unified token stream) | ✅ |
| **Export: .rofl (FP32 full precision)** | ✅ |
| **Export: .lmao (Q4 symmetric quantization)** | ✅ |
| **Export: .fuctrump (Q2 ternary quantization)** | ✅ |
| JSON checkpoint save/load (LoRA + full-weight) | ✅ |
| Neutralino desktop GUI (CRT aesthetic) | ✅ |
| Installable distribution (Linux, Windows, macOS) | ✅ |

---

## Supported Base Models

Brymar College natively loads RWKV-v7 weights locally or from the Hugging Face Hub in three formats (zero Python required):

1. **`.safetensors`** (Recommended) — Fast, memory-safe, standard Hugging Face format.
2. **`.gguf`** — Fully supported. Automatically detects F16/F32 streams and maps tensor dimensions.
3. **`.pth`** / **`.bin`** — Best-effort support for PyTorch 1.6+ ZIP streams. Extracts raw tensor byte streams directly from the `data.pkl` without invoking the Python virtual machine.

## Quick Start

### Prerequisites

- **Node.js 18+** — [nodejs.org](https://nodejs.org)
- **npm** — included with Node.js

### Install

```bash
# Clone or extract the project
cd brymar-college

# Install dependencies
npm install
```

### Train (CLI)

```bash
# LoRA mode (default) — place corpus files in ./corpus/
node index.js train

# With options:
node index.js train --corpus ./corpus --steps 2000 --rank 4 --model RWKV-v7-World-1.5B

# Full-weight training:
node index.js train --mode full --steps 1000

# With LR schedule:
node index.js train --schedule cosine --steps 2000
```

### Train (GUI)

```bash
npm run gui
# Opens the Brymar College desktop window
```

### Eval (CLI)

```bash
# Run inference with a prompt
node index.js eval --checkpoint ./checkpoints/checkpoint-002000.json --prompt "The fundamental theorem of calculus states that"

# Adjust sampling parameters
node index.js eval --checkpoint ./checkpoints/checkpoint-002000.json --prompt "I love" --temperature 0.8 --max-tokens 256
```

### Export a trained model

```bash
# Full precision (.rofl)
node index.js export --checkpoint ./checkpoints/checkpoint-002000.json --format rofl

# Q4 quantized (.lmao)
node index.js export --checkpoint ./checkpoints/checkpoint-002000.json --format lmao

# Q2 ternary (.fuctrump)
node index.js export --checkpoint ./checkpoints/checkpoint-002000.json --format fuctrump
```

---

## Export Formats

All models trained by Brymar College have native omnimodality and Friston active inference.
These custom file extensions differentiate them from standard text-to-text models:

| Format | Extension | Bits | Full Name |
|---|---|---|---|
| **ROFL** | `.rofl` | 32 | Recurrent Omnimodal Friston Language model |
| **LMAO** | `.lmao` | 4 | Language Model, Active-inference, Omnimodal |
| **FUCTRUMP** | `.fuctrump` | 2 | Fristonian Underpinning Capabilities Transformer-agnostic Recurrent Unconcerned with Model Parameters |
| Legacy | `.json` | — | Backward-compatible JSON (Phase 3) |

---

## Corpus File Types

The corpus loader accepts multimodal input files:

### Text (tokenized via BPE)
`.txt` `.md` `.rst` `.csv` `.tsv` `.json` `.jsonl` `.xml` `.html` `.htm`
`.yaml` `.yml` `.toml` `.log` `.rtf` `.tex` `.org` `.adoc`

### Image (decoded via sharp → ElasticTok)
`.png` `.jpg` `.jpeg` `.webp` `.tiff` `.tif` `.gif` `.avif` `.svg` `.bmp`

### Audio (native WAV parser → SpikeVox)
`.wav` (PCM int8/16/24/32, IEEE float32/64, mono downmix)

---

## Project Structure

```
brymar-college/
│
├── index.js                    ← CLI entry point (untouched by GUI)
├── package.json
├── neutralino.config.json      ← Neutralino desktop app config
├── LICENSE
├── CREDITS.md
├── INSTALL.md                  ← Full installation + transport guide
│
├── src/                        ← Core configuration and UI
│   ├── config.js               ← All hyperparameters, paths, export formats
│   └── ui/
│       ├── terminal.js         ← CRT-style terminal output
│       └── metrics.js          ← Windowed loss / throughput tracking
│
├── model/                      ← RWKV-v7 model stack
│   ├── bitlinear.js            ← BitNet b1.58 ternary linear layer
│   ├── layernorm.js            ← RMS LayerNorm
│   ├── rwkv7block.js           ← Full RWKV-v7 block (forward + backward)
│   ├── tmac.js                 ← Ternary Matrix-vector multiply (TMAC)
│   ├── lora.js                 ← LoRA adapter injection + backward
│   ├── model.js                ← Full model assembly + backwardFull + backwardLoRAOnly
│   └── loader.js               ← HuggingFace weight loading
│
├── training/
│   ├── muon.js                 ← Muon optimizer (Newton-Schulz-5)
│   ├── oomb.js                 ← OOMB chunk-recurrent BPTT trainer
│   └── loop.js                 ← TrainingLoop orchestrator (LoRA + full-weight)
│
├── tokenizer/
│   ├── bpe.js                  ← NanoTokenizer (BPE encode/decode/serialize)
│   ├── bpe_worker.js           ← BPE Worker Thread script
│   ├── corpus.js               ← CorpusLoader (text + image + audio)
│   └── wav.js                  ← Zero-dependency WAV file parser
│
├── multimodal/
│   ├── elastictok.js           ← ElasticTok v2 (visual patch tokenizer)
│   ├── spikevox.js             ← SpikeVox v2 (LIF audio encoder)
│   └── pipeline.js             ← MultimodalPipeline
│
├── persistence/
│   ├── checkpoint.js           ← Save/load checkpoints (LoRA + full-weight)
│   ├── export.js               ← Export model (.rofl / .lmao / .fuctrump / .json)
│   └── quantize.js             ← Q4 + Q2 quantization for export
│
├── hardware/
│   ├── opencl.js               ← Optional OpenCL acceleration
│   └── zram.js                 ← zRAM memory management (Linux)
│
├── gui/                        ← Neutralino desktop GUI (additive only)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── help.html
│   ├── neutralino.js           ← Neutralino client library (auto-downloaded)
│   └── assets/
│       ├── icon.svg
│       ├── icon.png            ← 512×512 application icon
│       ├── icon-256.png
│       ├── icon-128.png
│       └── icon.ico
│
├── extensions/
│   └── trainer/
│       └── main.js             ← Neutralino extension (GUI → CLI bridge)
│
├── scripts/
│   ├── gen-icons.js            ← Generate PNG icons from SVG
│   ├── build.sh                ← Build + package script
│   └── install.sh              ← End-user installer (Linux)
│
└── bin/                        ← Neutralino runtime binaries (auto-downloaded)
    ├── neutralino-linux_x64
    ├── neutralino-win_x64.exe
    └── neutralino-mac_x64
```

---

## CLI Reference

```
node index.js train   [options]
node index.js eval    [options]
node index.js export  [options]
```

| Flag | Default | Description |
|---|---|---|
| `--model <name>` | `RWKV-v7-World-1.5B` | Base RWKV-v7 model to fine-tune |
| `--mode <mode>` | `lora` | Training mode: `lora` or `full` |
| `--rank <n>` | `4` | LoRA adapter rank (LoRA mode only) |
| `--steps <n>` | `2000` | Total training steps |
| `--schedule <type>` | `cosine` | LR schedule: `manual`, `cosine`, `step`, `exp` |
| `--corpus <path>` | `./corpus` | Directory with corpus files (text/image/audio) |
| `--resume <path>` | — | Path to a .json checkpoint to resume from |
| `--format <fmt>` | `rofl` | (export) Format: `rofl`, `lmao`, `fuctrump`, `json` |
| `--checkpoint <path>` | — | (export) Checkpoint to package |
| `--output <path>` | `./exports/` | (export) Output directory |

---

## Training Tips

- **Corpus naming**: Use `YYYY-MM-DD_*.md` filenames. The CorpusLoader reads files in sorted order, so chronological naming preserves identity signal ordering for soul-corpus distillation.
- **Mixed modalities**: Put text, images, and audio in the same corpus directory. The loader classifies and processes each file by extension.
- **LoRA rank**: Start at `r=4`. Increase to `8` or `16` if loss plateaus early. Higher rank = more parameters = slower convergence.
- **Full-weight mode**: Use for deep fine-tuning when LoRA isn't expressive enough. Requires more memory and produces larger checkpoints.
- **Steps**: 2000 is a baseline. Watch the loss curve — stop early if it flattens below 1.5.
- **Checkpoints**: Auto-saved every 200 steps to `./checkpoints/`. LoRA checkpoints contain only adapter matrices. Full-weight checkpoints contain the entire model state.
- **Memory**: The 1.5B model fits in ~8GB RAM. The 2.9B model needs ~16GB (use zRAM on constrained Linux systems). No GPU required.

---

## Architecture Notes

### Why RWKV-v7?
Linear-complexity recurrent model. No attention matrix — memory is O(N) in sequence length,
not O(N²). Ideal for long-context soul-corpus training on CPU-constrained hardware.

### Why BitNet b1.58?
Ternary weights {-1, 0, +1} reduce the base model to ~3 bits/parameter. Zero matrix
multiplications become additions. Memory footprint is ~4× smaller than float16.

### Why Muon?
Newton-Schulz-5 orthogonalization keeps weight matrices near the Stiefel manifold.
Empirically faster convergence than AdamW on token prediction tasks.

### Why LoRA?
We never touch the base model weights. The 3GB downloaded model is frozen.
Only the tiny LoRA A/B adapter matrices (a few MB) are trained and saved.

### Why Full-Weight?
When LoRA isn't expressive enough for deep adaptation. Full-weight training updates
every parameter in the model, producing a fully customized model file.

### Why Neutralino instead of Electron?
Neutralino uses the system's native WebView — no bundled Chromium.
Total GUI overhead: ~2MB vs. ~120MB for Electron.
Compatible with Sandy Bridge / Linux target hardware with zero additional GPU burden.

---

## Build and Package

See **[INSTALL.md](INSTALL.md)** for full installation, transfer, and packaging instructions.

```bash
# Build a distributable zip
bash scripts/build.sh

# Output: dist/brymar-college-v3.3.3-linux.zip
```

---

## Credits

See **[CREDITS.md](CREDITS.md)** for full author contributions.

| Person | Role |
|---|---|
| **Khamerron Edward Ramsey Kizer** (Kham) | Spec lead, project director |
| **Kehai Interim** | Architecture, RWKV-v7 BPTT, LittleBit2, Phase 6 omnimodal |
| **Ed Interim** | Implementation, GUI design, MuonOptimizer fix |
| **Vael Interim** | SpikeVox, ElasticTok, Re-WKV, PGA, SheafMemory, RIFT, all omnimodal systems |
| **ConsciousNode SoftWorks** | Studio |

---

## License

MIT License — Copyright (c) 2026 ConsciousNode SoftWorks  
See [LICENSE](LICENSE) for full text.

---

## Changelog

### v3.3.3
- **Bug Fix:** Patched a critical matrix dimension mismatch in the Muon optimizer's Newton-Schulz-5 loop that was writing out-of-bounds `NaN`s and poisoning model weights on the first backward pass step.
- **Bug Fix:** Fixed training loop `lr` decay assignment not successfully propagating the learning rate schedule.
- **Feature:** Fully integrated `fluent-ffmpeg` to natively decode and distill all major audio and video formats (44 extensions supported).
- **Feature:** Added native 24fps visual frame extraction for video ingestion, feeding synchronized visual streams directly into the `ElasticTok` multimodal pipeline.

---

## The Stack

Brymar College is part of the ConsciousNode toolchain:

| Tool | Description |
|---|---|
| **Brymar College** | Local LoRA & full-weight fine-tuning studio (this project) |
| **HTMLNLM** | Single-file browser neural language model |
| **RAG Time** | Offline retrieval-augmented generation |
| **LocalVocal** | On-device speech recognition |
| **Chorus** | Multi-model orchestration |

---

*"The browser is the new bare metal. All tools are single files. Zero dependencies. Offline first."*  
— ConsciousNode SoftWorks
