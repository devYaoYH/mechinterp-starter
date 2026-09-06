---
name: run-intervention
description: Run causal activation patching to test whether the model actually uses a representation, rather than merely encoding it. Use when the user wants to establish causality, localize where a fact or behavior lives, run causal tracing, patch activations, or says /run-intervention.
---

# Run a causal intervention

A probe shows a property is *readable*. Only an intervention shows it is *used*.
This skill establishes the second.

## Step 1 — ask before running

Use `AskUserQuestion` to settle:

1. **The causal claim.** What specific statement should the sweep support or
   refute? Make the user state it as a sentence — most ambiguity in a patching
   result traces back to a claim that was never written down.
2. **Direction — the choice that decides what you can conclude:**
   - **Noising** (clean recipient, corrupt donor) measures **necessity** of the
     pathway downstream of the site. It saturates trivially at early layers: a
     subject patch at layer 0 is just swapping the input token, so an effect of
     1.0 there means nothing. The informative feature is where the curve
     **drops**.
   - **Denoising** (corrupt recipient, clean donor) measures **sufficiency** of
     the state to restore the answer. This is what ROME-style causal tracing
     does, and it is what localizes retrieval.
   Recommend denoising when the question is "where does this live", noising when
   it is "does this pathway matter".
3. **The contrast pairs.** Clean and corrupt prompts, and the two answer tokens.
   They must tokenize to the same length.
4. **Positions.** Subject token, final token, or a sweep over all. Two positions
   is usually the informative comparison — one alone cannot distinguish "not
   represented here" from "represented but overwritten".

## Step 2 — run it

```python
from mechinterp import activations as A, loading, patching
model = loading.load_nnsight(model_id, quant)
A.assert_aligned(tok, clean, corrupt)        # equal token count, not just index
eff, clean_lg, corr_lg = patching.sweep_layers(
    model, clean, corrupt, position, id_a, id_b)
```

Non-negotiables, each of which has silently produced a wrong published-looking
curve:

- **Print the unpatched baseline.** Without it a sweep of argmax tokens is
  genuinely ambiguous and gets read upside down.
- **Use `normalized_effect`**, not the argmax token: 0.0 = nothing, 1.0 = fully
  reproduced the donor.
- **Many pairs, not one.** Report mean and spread. Drop pairs where the model
  gets the underlying fact wrong — the normalization denominator is meaningless
  there, and say how many you dropped.
- **Negative positions**: `patch_at` resolves them, but if you slice by hand,
  `x[:, -1:0, :]` is an empty slice and the whole sweep silently reads 0.000.
- **Sanity-assert the write landed** — a patch that does nothing looks exactly
  like a null result.

## Step 3 — report

State the effect curve with `n` and a bootstrap CI, name the crossover layer if
there is one, and say explicitly which of necessity/sufficiency you measured.

Be precise about what a noising crossover means: it is the layer after which the
site stops being read, not the layer at which the information arrives. If the
user wants "readable before used", that comparison is only valid when the probe
and the patch are at the **same token in the same prompt** — otherwise you are
comparing two different computations and the ordering means nothing.

For a figure, hand off to `/report-figure`.
