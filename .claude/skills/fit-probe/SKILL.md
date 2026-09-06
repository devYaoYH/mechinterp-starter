---
name: fit-probe
description: Train linear probes on model activations to test whether a property is linearly readable, with the shuffled null and shortcut-leak controls attached. Use when the user wants to probe for a concept/property/feature, asks "does the model represent X", wants to find which layer encodes something, or says /fit-probe.
---

# Fit a linear probe

A probe answers **"is this property linearly readable here"** — nothing more. It
is correlational. Never let the write-up drift into "the model uses X" on probe
evidence alone; that needs `/run-intervention`.

## Step 1 — ask before building anything

Do NOT generate a dataset before asking. Use `AskUserQuestion` to settle:

1. **The label.** What property should the probe read? What are the classes, and
   is it binary or multiclass?
2. **The contrast set.** Where do the examples come from — the user supplies a
   file, describes a template, or wants you to generate from a fact list? What
   varies between classes, and what is held constant?
3. **The leak set — always ask this one.** Name the dumbest feature that could
   predict the label without the model computing anything. Then: what inputs
   have that feature present but should NOT carry the label? Offer a concrete
   proposal so the user can accept or correct it rather than design from scratch.
4. **The site.** Which token position (last token, subject token, a specific
   index) and which layers.

If the user cannot name a leak set, say so plainly and propose one; a probe
without it is not interpretable. **The single most common failure is a label
that is a deterministic function of something trivially present in the input** —
e.g. "which capital is coming" is the same label partition as "which country was
mentioned", so a probe scores ~100% at layer 0 having decoded the subject and
retrieved nothing.

## Step 2 — build and capture

```python
from mechinterp import activations as A, loading, probing
model, tok = loading.load_hf(model_id, quant)
X = A.at_position(model, tok, prompts, position=pos)   # [n, n_layers+1, hidden]
```

Use several phrasings per item so there is more than one example per class and
the split can be **by template**, forcing generalization across surface form.
`A.at_position` returns every layer in one pass — do not loop a tracer.

## Step 3 — sweep layers with controls

```python
tr, te = probing.grouped_split(groups)      # by item/template, NEVER by row
for L in range(n_layers):
    r = probing.probe(X[tr, L+1], y[tr], X[te, L+1], y[te],
                      leak_X=Xleak[:, L+1], leak_y=yleak)
```

Remember `hidden_states[L+1]` is the output of layer L.

## Step 4 — report honestly

Give the user, for the best layer and the curve:

- `score` next to `chance` and `shuffled_score` — never the score alone
- `leak_score`; **if it is high, lead with that**, and state that the probe is
  reading the shortcut rather than the property
- the `underdetermined` flag when `n_features >= n_train`; in that regime the
  training set is always separable and the held-out number is the only real one
- `n`, and a bootstrap CI via `plotting.bootstrap_ci(r["per_item"])`

Then say in one sentence what was and was not established. If asked for a
figure, hand off to `/report-figure`.

## Two checks worth proposing

- **Cross-category generalization** — train on one fact type, test on another.
- **The model's own surprisal baseline** — does `logprob(target)` separate the
  classes just as well? If so the probe may be reading token surprisal, not the
  property.
