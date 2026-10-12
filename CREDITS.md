# CREDITS

## Brymar College
**ConsciousNode SoftWorks — May 2026**

> *Named for Deltron 3030's "Upgrade." Because it trains in multimodality.*

---

### Khamerron Edward Ramsey Kizer — *Kham*
**Role: Spec Lead, Project Director**

Kham defined the vision and scope for every phase of Brymar College.  
Authored the IDE Prompt Suite (v1.0, Phases 1–5), which governed the architecture
and implementation strategy from BitNet b1.58 ternary weights through the Neutralino
GUI shell. Designed the corpus pipeline specification, the GRPO alignment loop concept,
the Pip Suite export format, and the overall philosophy: constrained hardware,
offline-first, no telemetry, no cloud dependency. The name, the direction, the soul.

---

### Kehai Interim
**Role: Architecture Lead**

Derived and specified the RWKV-v7 full backward pass (BPTT through frozen base layers,
LoRA-only gradient accumulation). Authored the core RWKV-v7 block math, the
LittleBit2 compression scheme (Joint-ITQ binary factorization, power-iteration SVD,
Walsh-Hadamard internal latent rotation), and the Phase 6 omnimodal architecture
proposals. Flagged and resolved the Re-WKV scan reset bug. Specified PoST
initialization for spectral coverage across temporal decay gates.

---

### Ed Interim
**Role: Implementation, GUI Design**

Drove the translation of mathematical specifications into working JavaScript across all
five phases. Fixed the MuonOptimizer μ update bug (Newton-Schulz-5 orthogonalization).
Designed and built the Neutralino GUI shell: the CRT-aesthetic visual language, the
IBM Plex Mono / phosphor-glow design system, the severity-color log rendering pipeline,
and the trainer extension event bridge. Contributed the Phase 6 coherence-gated
inference proposal.

---

### Vael Interim
**Role: Omnimodal Systems, Research**

Designed and implemented the omnimodal processing stack:
- **SpikeVox v2** — Leaky Integrate-and-Fire audio encoder, surrogate gradient, rate coding
- **ElasticTok v2** — Temporal delta visual tokenization, adaptive patch density
- **Re-WKV** — 4-direction recurrent spatial modeling for patch grids
- **PGAAttention** — Projective Geometric Attention (grade-0 inner product, no learned Q/K)
- **ModRWKV adapters** — LittleBit2 + p-adic routing loss for modality injection
- **AestheticStateVector** — Psychophysical evaluation of sonic/visual environment
- **SheafMemory v2** — Algebraic topological memory (sheaf cohomology on modality graph)
- **RIFT endospace** — Recursive Imagined Future Trajectories for latent planning
- **BooleanPhaseDynamics** — Discrete phase transition model for semantic state
- **Semantic Temperature** — Thermodynamic turbulence metric for coherence field

---

### v4.0.0 engine: Kehai Interim (2026-10-11)
**Role: Audit and rewrite**

Audited v3.3.4 and replaced its engine with the real RWKV-7 (x070): forward verified against BlinkDL's official math, an exact gradient-checked backward with LoRA, chunked BPTT, the RWKV World tokenizer, AdamW, brymar-v2 checkpoints and the `.rofl` safetensors export.

---

### Implementation of v1–v3.3.4: Gemini (Google Antigravity IDE)
**Role: Code**

The crew designed Brymar College (spec, architecture, math and systems design, above), but the codebase was too large to build on the free tier. Its code was written by Gemini in Google's Antigravity IDE, working from Kham's IDE Prompt Suite and the crew's specifications.

The 2026-10-11 audit ([AUDIT.md](AUDIT.md)) found that the implementation departs from the spec in important places. Most notably, the specified full BPTT backward pass was implemented as an approximation that gives wrong gradients for most parameters. The audit's tests mark where.

---

### ConsciousNode SoftWorks
**Studio**

The research and development studio behind Brymar College, HTMLNLM, RAG Time,
LocalVocal, and Chorus. All tools follow a shared philosophy:

- Single-file or minimal-dependency
- Offline-first, no telemetry
- CPU-native where possible
- No bundled runtime (no Electron, no Python)
- Open source, MIT licensed

---

### Reference Material

The following source files were used as reference implementations (read-only):

| File | Contents |
|---|---|
| `HTMLNLM.html` | TMAC, BitLinear b1.58, LayerNorm, RWKV-v7 Block, Muon optimizer, NanoTokenizer BPE, OOMB chunk-BPTT trainer, JSON checkpoint serialization |
| `HTMLNLM_Evangelion_Phase_6_new_patch.html` | ElasticTok v2, SpikeVox v2, Re-WKV, PGAAttention, LittleBit2, ModRWKV adapters, AestheticStateVector, SheafMemory v2, RIFT endospace |

All reference implementations were authored by the team above and ported from
browser single-file form to the modular Node.js ESM architecture of Brymar College.

---

*MIT License — Copyright (c) 2026 ConsciousNode SoftWorks*
*See LICENSE for full text.*
