"""Cross-run activation patching, via nnsight.

Copy the residual stream at (layer, position) from a donor run into a recipient
run, and read the resulting logits. Sweeping this over layers is the standard
causal-localization move.

Direction matters, and the two directions answer different questions:
  NOISING    recipient=clean, donor=corrupt. Measures NECESSITY of the pathway
             downstream of that site. Saturates trivially at early layers --
             patching layer 0 at the subject position is just swapping the
             input token, so a full flip there is not evidence of anything.
  DENOISING  recipient=corrupt, donor=clean. Measures SUFFICIENCY of the state
             to restore the answer. This is what ROME-style causal tracing does,
             and it is the one that localizes where a fact is retrieved.
`sweep_layers` does noising by default; pass the prompts the other way round
for denoising.
"""
import torch


def resolve_position(model, prompt, position):
    """Turn a negative index into an absolute one.

    Non-negotiable: `x[:, -1:-1+1, :]` is `x[:, -1:0, :]`, an EMPTY slice, so a
    patch written with position=-1 silently does nothing and the whole sweep
    reads 0.000. Always resolve before slicing.
    """
    if position >= 0:
        return position
    return len(model.tokenizer(prompt)["input_ids"]) + position


def patch_at(model, donor_prompt, recipient_prompt, layer, position):
    """Patch one (layer, position) from donor into recipient. -> final-position
    logits [vocab] of the patched recipient run. Negative positions are fine."""
    position = resolve_position(model, recipient_prompt, position)
    with torch.no_grad():
        with model.trace(donor_prompt):
            rep = model.model.layers[layer].output[:, position:position + 1, :].save()
        with model.trace(recipient_prompt):
            model.model.layers[layer].output[:, position:position + 1, :] = rep
            logits = model.lm_head.output[:, -1, :].save()
    return logits[0]


def baseline_logits(model, prompt):
    with torch.no_grad(), model.trace(prompt):
        logits = model.lm_head.output[:, -1, :].save()
    return logits[0]


def normalized_effect(clean_logits, corrupt_logits, patched_logits, id_a, id_b):
    """Fraction of the clean->corrupt logit difference that the patch moved.

    0.0 = patch did nothing;  1.0 = patch fully reproduced the corrupt run.
    Always report this rather than the argmax token: argmax hides the size of
    the effect, and the unpatched baseline is what tells you which end of the
    scale "no effect" sits at. Reporting a sweep without the baseline is how a
    curve gets read upside down.
    """
    def ld(lg):
        return (lg[id_a] - lg[id_b]).item()
    denom = ld(clean_logits) - ld(corrupt_logits)
    if abs(denom) < 1e-6:
        raise ValueError("clean and corrupt runs are not separated on these tokens")
    return (ld(clean_logits) - ld(patched_logits)) / denom


def sweep_layers(model, clean_prompt, corrupt_prompt, position, id_a, id_b, layers=None):
    """Noising sweep. -> {layer: normalized effect}, plus the two baselines.

    id_a / id_b: token ids of the clean and corrupt answers.
    """
    n = len(model.model.layers)
    layers = range(n) if layers is None else layers
    clean = baseline_logits(model, clean_prompt)
    corrupt = baseline_logits(model, corrupt_prompt)
    out = {}
    for l in layers:
        patched = patch_at(model, corrupt_prompt, clean_prompt, l, position)
        out[l] = normalized_effect(clean, corrupt, patched, id_a, id_b)
    return out, clean, corrupt
