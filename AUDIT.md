# Brymar College v3.3.4: audit (2026-10-11 · Kehai Interim)

> **Resolved in v4.0.0** (same day): path 2 below. The real RWKV-7 engine (`model/rwkv7.js`) matches the official math to 7·10⁻¹⁵, has exact gradients (incl. LoRA), uses the RWKV World tokenizer, and keeps weights in float. The old engine and these audit tests live in `legacy/v3.3.4/`.

Measured, not guessed. Both tests run against the real code in this repo:

```bash
npm install
node legacy/v3.3.4/tests/weight-map.mjs [real-rwkv7.safetensors]   # what a real RWKV-7 checkpoint does when loaded
node legacy/v3.3.4/tests/gradcheck.mjs              # every backward vs finite differences
```

## 1. The model is not RWKV-7, so most of a checkpoint is never used

Brymar's block is a simplified recurrent block with a per-channel running sum (RWKV-v4-style memory). Real RWKV-7 (checked against BlinkDL's own loader in the `rwkv` package, `rwkv/model.py`) keeps a matrix of memory per head. Its time-mix also has:
- token-shift vectors `x_r x_w x_k x_v x_a x_g`;
- low-rank decay, in-context-rate, value-residual and gate paths (`w0-2`, `a0-2`, `v0-2`, `g1-2`);
- `k_k`, `k_a`, `r_k` and a per-head group norm `ln_x`.

Its FFN has no receptance gate.

Loading an RWKV-7-shaped checkpoint through the old `model/loader.js` (`legacy/v3.3.4/tests/weight-map.mjs`):
- **43 of 69 tensors are never used.** The log line "Applied 26 … (ignored 43)" is the only sign of it.
- **14 of Brymar's 40 parameter sets stay at random initialisation**, including 7 full matrices per block: `att_g`, `W_x_kappa`, `W_x_a`, `W_x_w`, `W_x_rx2`, `W_x_kx2` and `Wr2`. `Wr2` gates the whole FFN output.
- **The matrices that do load are rounded to ternary** on every forward (`BitLinear` always quantizes). That loses ≈51% of each matrix (relative error, Gaussian stand-ins; run it on a real checkpoint for the real number). Ternary works for models *trained* ternary, not for float-pretrained ones.

Also: the loader maps `time_mix_*` vectors (one number per channel) onto full matrices. If such keys ever appear (RWKV-4 to 6), the size check skips them *silently*: they are counted as neither applied nor ignored.

## 2. The backward pass is wrong almost everywhere

`legacy/v3.3.4/tests/gradcheck.mjs`, float forward (so quantization isn't the cause): **38 of 42 parameter groups get a wrong gradient.**

- **Correct:** the head, and the last block's channel-mix (`Wr2`, `Wk2`, `Wv2`).
- **Time-mix:**
  - `att_r` and `att_k` receive the gated output's gradient directly, and `att_v` receives `cache.wkv` (an activation) as its gradient;
  - `W_x_kappa`, `W_x_a` and `W_x_w` get no gradient at all;
  - there is no backprop through the WKV memory: incoming state gradients are copied through unchanged.
- **Residuals:** the gradient going down a block is replaced by `ln1.backward(gradH)`, which drops the residual and the real input-gradients. `ln2`'s result is discarded (`_ignoredGrad`). Lower layers therefore get the wrong signal.
- **LoRA mode (the default):** every adapter gradient is wrong, and one points almost exactly backwards (cos −0.96). Each adapter caches only its *most recent* forward input, so BPTT reuses the last timestep's input for every step. All four adapters receive the same `gradGated`.
- **BitLinear:** the input gradient uses the float weights `Wᵀg` while the forward uses the ternary `γ·Q(W)`. The same bug was fixed in HTMLNLM, Evangelion, EvaROSA and Simulacra on 2026-10-11.

## 3. Smaller things

- **`fristonAlpha`:** adds α·H(prediction) to the loss. The gradient is computed correctly, but this rewards confidence; it is not active inference, which requires choosing actions to reduce expected surprise.
- **`training/loop.js`:** `parseInt(opts.steps) ?? default` never falls back (`parseInt(undefined)` is `NaN`, and `??` only catches null/undefined), so a missing `--steps` means zero training steps. The same applies to `--rank`.

## What is solid

- **File parsing:** safetensors (incl. BF16/F16), GGUF tensor mapping, and the `.pth` → safetensors fallback.
- **Training plumbing:** the CLI, LR schedules, OOMB checkpointing and the export formats.

The bones are good; the model and its backward need replacing.

## Paths forward

1. **From-scratch trainer:** replace the model with Evangelion's RWKV-v7-style block, which passes exact gradient checks (float and real ternary) and has learned senses. Quick and certain.
2. **Real RWKV-7 fine-tuner:** implement the genuine x070 architecture in float with an exact, gradient-checked backward, and keep ternary for models trained ternary. Practical on CPU at 0.1B/0.4B; 1.5B would take seconds per token in JS on a Sandy Bridge.
