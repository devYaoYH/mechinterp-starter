# mechinterp-starter

A verified substrate for mechanistic-interpretability work on a single GPU:
model loading, activation extraction, causal patching, logit lens, and a probing
harness with the controls attached. No research claims — the point is that the
plumbing is known-good so you can start on the actual question.

Everything here has been run end-to-end on an RTX 5090 (compute capability 12.0)
against `Qwen/Qwen2.5-1.5B-Instruct` with `torch 2.11.0+cu128`,
`transformers 5.16.1`, `nnsight 0.7.0`.

## Quickstart

```bash
git clone <this repo> && cd mechinterp-starter
./setup.sh
source .venv/bin/activate
python smoke_test.py          # ~1 min: verifies GPU, API shapes, and the harness
python example_causal_trace.py --n-pairs 12
```

If `smoke_test.py` is green, the stack works and any wrong number after that is
your experiment, not your environment. Run it again after any dependency bump.

## What's here

```
smoke_test.py             10 checks over the whole stack -- run this first
example_causal_trace.py   worked example: layer sweep, two positions, 30 pairs
mechinterp/loading.py     model loading; the bf16 / nf4 / prequantized branch
mechinterp/activations.py extraction, token lookup, prompt-alignment assert
mechinterp/patching.py    cross-run patching + the normalized effect metric
mechinterp/readout.py     logit lens
mechinterp/probing.py     linear probe with shuffled null, leak test, split
setup.sh                  GPU-arch-aware bootstrap
```

## The four traps this encodes

Each of these produced a *plausible wrong number* rather than an error, and each
has a smoke-test check.

**1. GPU arch decides the CUDA wheel, not the driver.** Blackwell (cc ≥ 10.0:
RTX 50-series, B200) needs CUDA ≥ 12.8 wheels. A `cu124` build installs cleanly,
reports the device correctly, and dies at the first real kernel with `no kernel
image is available for execution on the device`. `setup.sh` reads
`nvidia-smi --query-gpu=compute_cap` and picks the index URL; the smoke test runs
an actual matmul rather than trusting `torch.cuda.is_available()`.

**2. `hidden_states[-1]` is already normed.** In HF decoder models the final
entry of `output_hidden_states` has had `model.model.norm` applied; every other
entry has not. Applying the norm again for a logit lens does not raise — it
returns confident nonsense (`' the'`, `' a'`, `' in'` instead of the answer).
Use `readout.lens_at`, which handles the bookkeeping.

Indexing convention throughout: `hidden_states[L + 1]` is the output of decoder
layer `L`, matching `model.model.layers[L].output` under nnsight. Keeping those
aligned is what lets an observation sweep and an intervention sweep be overlaid.

**3. A negative position silently patches nothing.** `x[:, -1:-1+1, :]` is
`x[:, -1:0, :]` — an empty slice. A sweep written that way returns exactly
`0.000` at every layer, which reads as a finding. `patching.resolve_position`
converts negative indices first.

**4. An intervention that does nothing looks like a result.** The smoke test
asserts a layer-0 subject patch *fully flips* the answer and a last-layer subject
patch *does not*, so both a dead write and an overly broad one are caught.

## Reading a patching sweep

`patching.normalized_effect` returns 0.0 for "did nothing" and 1.0 for "fully
reproduced the donor run". Two habits stop a sweep from being read upside down:

- **Print the unpatched baseline.** Without it you cannot tell which end of the
  scale "no effect" is at, and a sweep of argmax tokens is genuinely ambiguous.
- **Know which direction you ran.** *Noising* (clean recipient, corrupt donor)
  measures necessity of the downstream pathway and saturates trivially at early
  layers — a subject patch at layer 0 is just swapping the input token, so an
  effect of 1.0 there means nothing. *Denoising* (corrupt recipient, clean donor)
  measures sufficiency and is what localizes retrieval, ROME-style. The
  informative feature of a noising sweep is where the effect *drops*.

## Probing: controls are part of the measurement

`probing.probe()` returns the score together with its controls, because an
uncontrolled probe number is not interpretable:

- `shuffled_score` — same pipeline, permuted labels. If this isn't near chance,
  the evaluation is broken.
- `leak_score` — the probe applied to inputs where a shortcut feature is still
  present but the label should *not* follow. High means the probe took the
  shortcut. A probe for "which capital is coming" scores well by decoding "which
  country was mentioned"; when the label is a deterministic function of something
  trivially in the input, the two are the same partition and the probe proves
  nothing.
- `underdetermined` — flags `n_features >= n_train`. Residual streams are 1.5k–8k
  dimensional, so a few hundred examples are *always* linearly separable and a
  high training score is guaranteed.
- `grouped_split` — split by item id, never by row, so the same sentence's tokens
  can't land on both sides.

Also worth doing and not automated here: check that a probe generalizes across
categories (train on one fact type, test on another), and compare against the
model's own output probability for the target — often a trained probe is just
reading token surprisal.

## Notes on this environment

- **Prefer pre-quantized checkpoints.** `BitsAndBytesConfig(load_in_4bit=True)`
  against a full-precision repo downloads the bf16 weights first (~65GB for a 32B
  model) and then quantizes. An `unsloth/*-bnb-4bit` repo is ~19GB and runs the
  same. Use `--quant prequantized` for those, `nf4` only when no such repo exists.
  *(The `nf4` and `prequantized` paths are not covered by the verified run above —
  smoke-test them on your box before relying on them.)*
- **Extraction uses plain transformers, intervention uses nnsight.**
  `output_hidden_states=True` returns every layer in one pass, which is simpler
  and much faster than looping a tracer; nnsight earns its place only for the
  donor/recipient patching pattern.
- **Vast.ai `/workspace` is not automatically persistent.** Check
  `vast-capabilities | jq '.instance.workspace_is_volume'`. Even when it is a
  volume, a shared HF cache there can hand you stale NFS handles on blobs —
  symptom is `OSError: [Errno 116] Stale file handle` mid-download. Point
  `HF_HOME` at local disk to get moving.
- **Model-agnostic within reason.** Everything assumes `model.model.layers[i]`,
  `model.model.norm`, and `model.lm_head` — true of Llama/Qwen/Mistral-family
  causal LMs. Run the smoke test against a new architecture before trusting it.
