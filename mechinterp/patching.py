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
`sweep_layers` does noising; swap the prompts for denoising. `sweep_pair`
sweeps several positions off one donor trace.
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


def donor_states(model, donor_prompt, layers, positions):
    """Save the donor residual stream at every (layer, position) in ONE trace.

    A sweep patches many layers from the SAME donor prompt, so reading them one
    at a time re-runs the donor once per layer -- about half the forward passes
    in a sweep, spent on nothing.

    `positions` must be ABSOLUTE, resolved against the RECIPIENT prompt
    (`assert_aligned` is what lets the two agree).

    TRAP: the body of a trace must be plain statements. nnsight traces this
    frame, and a comprehension runs in a frame of its own -- writing the loop
    below as `{(l, p): ....save() for l in layers}` leaves `saved` unbound, or
    deadlocks. Same reason the return is outside the block.

    -> {(layer, position): state}
    """
    saved = {}
    with torch.no_grad(), model.trace(donor_prompt):
        for l in layers:
            for p in positions:
                saved[(l, p)] = model.model.layers[l].output[:, p:p + 1, :].save()
    return saved


def patch_with(model, recipient_prompt, layer, position, state):
    """Patch a pre-saved donor state in -> final-position logits of the
    recipient run. `position` must be absolute (see `resolve_position`)."""
    with torch.no_grad(), model.trace(recipient_prompt):
        model.model.layers[layer].output[:, position:position + 1, :] = state
        logits = model.lm_head.output[:, -1, :].save()
    return logits[0]


def patch_at(model, donor_prompt, recipient_prompt, layer, position):
    """One-shot patch -> final-position logits of the patched recipient run.
    For more than one site use `sweep_pair`, which reads the donor once."""
    p = resolve_position(model, recipient_prompt, position)
    state = donor_states(model, donor_prompt, [layer], [p])[(layer, p)]
    return patch_with(model, recipient_prompt, layer, p, state)


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
    eff, clean, corrupt = sweep_pair(model, clean_prompt, corrupt_prompt,
                                     {"site": position}, id_a, id_b, layers)
    return eff["site"], clean, corrupt


def sweep_pair(model, clean_prompt, corrupt_prompt, positions, id_a, id_b,
               layers=None, baselines=None):
    """Sweep layers at several positions off ONE donor trace.

    Two positions is usually the informative comparison -- one alone cannot
    distinguish "not represented here" from "represented but overwritten".

    positions: {name: index}, negative allowed.
    baselines: (clean, corrupt) logits if you already have them -- a caller that
    screens pairs on them first would otherwise pay twice.
    -> ({name: {layer: effect}}, clean_logits, corrupt_logits)
    """
    layers = list(range(len(model.model.layers))) if layers is None else list(layers)
    clean, corrupt = baselines if baselines is not None else (
        baseline_logits(model, clean_prompt), baseline_logits(model, corrupt_prompt))
    at = {name: resolve_position(model, clean_prompt, i) for name, i in positions.items()}
    donor = donor_states(model, corrupt_prompt, layers, sorted(set(at.values())))

    eff = {}
    for name, p in at.items():
        eff[name] = {}
        for l in layers:
            patched = patch_with(model, clean_prompt, l, p, donor[(l, p)])
            eff[name][l] = normalized_effect(clean, corrupt, patched, id_a, id_b)
    return eff, clean, corrupt
