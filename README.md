# Brymar College

**ConsciousNode LoRA & Full-Weight Fine-Tuning Studio for RWKV-7**
*v4.0.0 — the real RWKV-7 engine (2026-10-11)*

Brymar College (named for Deltron 3030's "Upgrade") is an offline-first training environment for **pretrained RWKV-7 (x070) models**: LoRA adapters or full-weight fine-tuning, on CPU, in plain JavaScript.

## What's new in v4.0.0

An audit of v3.3.4 ([AUDIT.md](AUDIT.md)) found that it could not fine-tune real RWKV-7 checkpoints:
- its model wasn't RWKV-7, so 43 of 69 checkpoint tensors went unused and 14 parameter sets stayed random;
- the weights that did load were rounded to ternary;
- 38 of 42 parameter groups got wrong gradients;
- the corpus was tokenized with a vocabulary the base model doesn't know.

v4.0.0 replaces the engine. Every claim below is checked by a test you can run:

| What | Test |
|---|---|
| **The real RWKV-7** (`model/rwkv7.js`): per-head matrix memory, low-rank decay / in-context rate / value residual / gate, `k_k`, `k_a`, `r_k`, group norm. Matches a numpy transcription of BlinkDL's official code to **7·10⁻¹⁵** | `python3 tests/ref/rwkv7_ref.py /tmp/ref && node tests/forward-ref.mjs /tmp/ref` |
| **Official checkpoints load 1:1**: every tensor by its official name, and anything unused or missing is reported, never silently skipped | same, plus the load report every run prints |
| **Exact gradients** for all 69 parameter tensors and all LoRA adapters, with BPTT through the matrix memory, token shift and value residual. A deliberately broken gradient is caught | `node tests/gradcheck-rwkv7.mjs` |
| **Chunked training (OOMB) = unchunked**, bit for bit | `node tests/oomb.mjs` |
| **The RWKV World tokenizer** (65,536 tokens, the one pretrained World models know), matching BlinkDL's Python tokenizer token for token | `python3 tests/tokenizer-ref.py <path to pip package rwkv>` |
| **It learns:** pretrain a tiny model (6.03 → 0.05 nats/byte), then LoRA-only fine-tuning teaches it a new voice (7.07 → 0.20) | `node tests/learn.mjs` |

**Weights stay in full precision.** Ternary (BitNet) works for models *trained* ternary, not for float-pretrained ones. The optimizer is AdamW (LoRA LR 2e-4, full-weight 2e-5).

**Honest speed:** about **1 second per trained token** for the 0.1B model on one CPU thread. A training step (4 × 128 tokens) takes ≈ 9 minutes, so a few hundred steps means days. Multi-core training is the next optimization.

**Not (yet) in the RWKV-7 engine:**
- **Senses.** Images, audio and video in the corpus are skipped with a warning; feature "pseudo-tokens" would read as random words to a pretrained model. Evangelion's learned senses (prefix + context, exact gradients) are the design to port.
- **Active inference.** `--friston-alpha` is ignored: an entropy penalty rewards confidence, which is not active inference.

## What is Brymar College?

You provide a corpus. Brymar College fine-tunes a pretrained RWKV-7 to sound like it, entirely on your machine: no cloud API, no telemetry, no GPU.

Two training modes:
- **LoRA** (default): small adapters on receptance/key/value/output; the base stays frozen. A checkpoint is the adapters plus a pointer to the base model, ≈100 KB, so several personas can share one base.
- **Full weight:** every parameter trains. Checkpoints are a `.safetensors` file with the official names.

---

## Quick start (v4)

1. Download an RWKV-7 World model `.pth` from [huggingface.co/BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world). The 0.1B model is the practical size on CPU.
2. `npm install`
3. Put `.txt` / `.md` / chat-export files in `./corpus/`.
4. Train, then talk to it, then export:

```bash
node index.js train --model models/rwkv7-world-0.1b.pth --steps 300          # LoRA (default)
node index.js eval  --checkpoint checkpoints/checkpoint-000300.json --prompt "Hello"
node index.js export --checkpoint checkpoints/checkpoint-000300.json --format rofl
```

`.pth` files are converted with `scripts/convert_pth.py` (Python 3, no PyTorch needed); `.safetensors` load directly. `.rofl` exports are the merged model as standard safetensors with official RWKV-7 names, loadable again with `--model`.

---

## Supported Base Models

Official **RWKV-7 (x070)** checkpoints, loaded by their official tensor names:

1. **`.safetensors`**: read directly (F32, F16, BF16).
2. **`.pth` / `.bin`**: converted to safetensors by `scripts/convert_pth.py` (Python 3 standard library; PyTorch not needed), then read.
3. **Hugging Face repo id**: tries `<repo>/model.safetensors`. BlinkDL publishes `.pth` files, so for official models download the `.pth` and pass its path.

GGUF is not supported by the v4 engine. Older RWKV generations (v4–v6) have different architectures and are rejected with a clear error.

### GUI

```bash
npm run gui
```

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

**Text** (`.txt .md .rst .csv .tsv .json .jsonl .xml .html .htm .yaml .yml .toml .log .rtf .tex .org .adoc`) is tokenized with the **base model's own tokenizer** (RWKV World for World models; bytes for `--model scratch`).

Images, audio and video are recognised but **skipped with a warning** in v4, until learned senses are ported to the RWKV-7 engine.

**Corpus naming:** `YYYY-MM-DD_*.md`. Files are read in sorted order, so chronological names keep a soul corpus in order.

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
├── model/                      ← RWKV-7 engine (v4)
│   ├── rwkv7.js                ← the real RWKV-7 (x070): forward + exact backward + LoRA
│   └── rwkv7_load.js           ← safetensors / .pth / HF loading, 1:1 by official names
│
├── training/
│   ├── adamw.js                ← AdamW (default optimizer)
│   ├── muon.js                 ← Muon (kept; not used by the v4 loop)
│   ├── oomb.js                 ← OOMB chunk-recurrent BPTT (exact; = unchunked)
│   └── loop.js                 ← TrainingLoop orchestrator (LoRA + full-weight)
│
├── tokenizer/
│   ├── rwkv_world.js           ← RWKV World tokenizer (official vocabulary)
│   ├── rwkv_world_vocab.json   ← rwkv_vocab_v20230424 (BlinkDL, Apache-2.0)
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
| `--model <path>` | `models/rwkv7-world.pth` | Base RWKV-7 weights (.pth / .safetensors / HF repo id), or `scratch` |
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

- **Start from a pretrained model.** A persona corpus is far too small to teach a model language from scratch; that is what the base is for.
- **LoRA first.** It's cheap, the checkpoints are tiny, and several personas can share one base. Rank 4–8 is plenty to start.
- **Watch for forgetting.** Fine-tuning on one voice can erode general ability (`tests/learn.mjs` shows this in miniature). A lower learning rate, fewer steps and mixing in some general text all help.
- **Memory** (float32): about 4 bytes × parameters, plus activations. 0.1B ≈ 0.8 GB; 1.5B ≈ 6 GB, and slow on CPU.
- **Speed:** ≈ 1 s per trained token at 0.1B on one thread (see What's new).

---

## Architecture Notes

### Why RWKV-v7?
Linear-complexity recurrent model. No attention matrix — memory is O(N) in sequence length,
not O(N²). Ideal for long-context soul-corpus training on CPU-constrained hardware.

### Why not BitNet b1.58 (any more)?
Ternary weights work for models *trained* ternary. Rounding a float-pretrained RWKV-7 to {-1, 0, +1} destroys about half of every matrix (AUDIT.md), so v4 keeps the base in float32.

### Why AdamW instead of Muon?
Muon is designed for full weight matrices. LoRA adapters are rank-4 strips, and AdamW is the standard, well-tested choice there.

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
| **Gemini** (Google Antigravity IDE) | Code implementation, from the crew's spec (see [AUDIT.md](AUDIT.md)) |
| **ConsciousNode SoftWorks** | Studio |

---

## License

MIT License — Copyright (c) 2026 ConsciousNode SoftWorks  
See [LICENSE](LICENSE) for full text.

---

## Changelog

### v4.0.0 (2026-10-11 · Kehai Interim + Kham)
- **New engine:** the real RWKV-7 (x070), float32, official tensor names, exact gradients with LoRA, chunked BPTT. Matches the official math to 7·10⁻¹⁵.
- **RWKV World tokenizer** in JS, identical to BlinkDL's.
- **AdamW;** sane learning rates; checkpoints `brymar-v2` (LoRA JSON, or JSON + safetensors for full weights); `.rofl` is now the merged model as safetensors.
- **CLI fixes:** eval no longer feeds the last prompt token twice; `--steps` / `--rank` defaults work; base-model load reports everything.
- **Old engine** moved to `legacy/v3.3.4/`, kept to reproduce the audit. Its checkpoints cannot be loaded.
- **Tests:** `tests/forward-ref.mjs`, `tests/gradcheck-rwkv7.mjs`, `tests/oomb.mjs`, `tests/learn.mjs`, `tests/tokenizer-ref.py`.


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
