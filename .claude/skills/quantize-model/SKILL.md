---
name: quantize-model
description: Quantize a dense checkpoint to 4-bit with bitsandbytes, save it for reuse, and measure what quantization cost in fidelity. Use when the user wants to run a model that does not fit in VRAM, mentions bitsandbytes/nf4/4-bit/quantization, or says /quantize-model.
---

# Quantize a model for interpretability work

## Step 1 — ask, and check the cheaper path first

Before quantizing anything, **check whether a pre-quantized repo already
exists** (e.g. `unsloth/<model>-bnb-4bit`). Quantize-on-load downloads the full
dense checkpoint first — ~65GB for a 32B model versus ~19GB pre-quantized — so
this question can save an hour.

Then use `AskUserQuestion` for:

1. **Which model, and what is the VRAM budget?** Check with `nvidia-smi`. A 4-bit
   model needs roughly `params × 0.55` GB plus activations.
2. **Quant type.** `nf4` (default, better for normally-distributed weights) or
   `fp4`. Double quantization on unless there is a reason.
3. **Fidelity check.** Strongly recommend yes. It needs one dense load, which is
   the expensive part — but running interpretability experiments on quantized
   activations without knowing the drift is how a finding turns out to be a
   quantization artifact.

## Step 2 — run it

```bash
python quantize_model.py --model <dense-repo> --out ./q/<name>-nf4
```

This loads dense, captures a reference, **frees it**, quantizes, saves, reloads
from disk, and writes `fidelity_report.json`. The two-phase capture matters: at
32B you cannot hold dense and quantized in memory at once, so never restructure
this to load both.

Afterwards the model loads in seconds:

```python
model, tok = loading.load_hf("./q/<name>-nf4", "prequantized")
```

## Step 3 — read the fidelity report before trusting any result

- **top-1 agreement < 0.95** → treat it as a different model. Reproduce any
  headline result on dense weights.
- **mean KL** in nats over next-token distributions. Small top-1 disagreement
  with large KL means the ranking survived but the distribution moved — probes
  reading probabilities are affected even when argmax looks fine.
- **residual-stream cosine by layer** — drift compounds with depth, so late
  layers are always worst. A probe or patch sited late is the most exposed.

Reference point from Qwen2.5-1.5B nf4 on this stack: top-1 agreement 1.000,
mean KL 0.119 nats, final-layer cosine 0.9885. So even a "clean" quantization
moves the residual stream by ~1-2% — enough to shift a marginal probe result,
not enough to move a sharp causal crossover.

## Step 4 — verify the stack still works

```bash
python smoke_test.py --model ./q/<name>-nf4 --quant prequantized
```

All checks should pass. Watch the logit-lens line: under nf4 the penultimate
slot can differ from dense, which is drift showing up exactly where the report
predicts it.
