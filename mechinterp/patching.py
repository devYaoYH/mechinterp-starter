"""Cross-run activation patching, via nnsight.

Copy the residual stream at (layer, position) from a donor run into a recipient
run and read the resulting logits. Sweeping over layers localizes computation.

DIRECTION decides what you may conclude:
  NOISING    recipient=clean, donor=corrupt. Measures NECESSITY of the pathway
             downstream. Saturates trivially early -- patching the subject at
             layer 0 is just swapping the input token, so an effect of 1.0 there
             means nothing. The informative feature is where the curve DROPS.
  DENOISING  recipient=corrupt, donor=clean. Measures SUFFICIENCY to restore the
             answer; this is what localizes retrieval (ROME-style).
`sweep_layers` does noising; swap the prompts for denoising.
"""
import torch


def resolve_position(model, prompt, position):
    """Negative index -> absolute. Non-negotiable: `x[:, -1:-1+1, :]` is
    `x[:, -1:0, :]`, an EMPTY slice, so a position=-1 patch silently writes
    nothing and the whole sweep reads 0.000."""
    if position >= 0:
        return position
    return len(model.tokenizer(prompt)["input_ids"]) + position


def baseline_logits(model, prompt):
    """Final-position logits of an unpatched run. Always report these: without
    them you cannot tell which end of the effect scale 'no effect' sits at."""
    with torch.no_grad(), model.trace(prompt):
        logits = model.lm_head.output[:, -1, :].save()
    return logits[0]


def patch_at(model, donor_prompt, recipient_prompt, layer, position):
    """-> final-position logits of the patched recipient run."""
    p = resolve_position(model, recipient_prompt, position)
    with torch.no_grad():
        with model.trace(donor_prompt):
            rep = model.model.layers[layer].output[:, p:p + 1, :].save()
        with model.trace(recipient_prompt):
            model.model.layers[layer].output[:, p:p + 1, :] = rep
            logits = model.lm_head.output[:, -1, :].save()
    return logits[0]


def normalized_effect(clean, corrupt, patched, id_a, id_b):
    """Fraction of the clean->corrupt logit difference the patch moved.
    0.0 = did nothing, 1.0 = fully reproduced the donor. Report this, never the
    argmax token: argmax hides the size of the effect."""
    def ld(lg):
        return (lg[id_a] - lg[id_b]).item()
    denom = ld(clean) - ld(corrupt)
    if abs(denom) < 1e-6:
        raise ValueError("clean and corrupt runs are not separated on these tokens")
    return (ld(clean) - ld(patched)) / denom


def sweep_layers(model, clean_prompt, corrupt_prompt, position, id_a, id_b, layers=None):
    """Noising sweep -> ({layer: effect}, clean_logits, corrupt_logits)."""
    layers = range(len(model.model.layers)) if layers is None else layers
    clean = baseline_logits(model, clean_prompt)
    corrupt = baseline_logits(model, corrupt_prompt)
    eff = {l: normalized_effect(clean, corrupt,
                                patch_at(model, corrupt_prompt, clean_prompt, l, position),
                                id_a, id_b) for l in layers}
    return eff, clean, corrupt
