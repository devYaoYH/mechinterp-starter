"""Activation extraction.

Plain transformers `output_hidden_states=True` returns every layer in one pass --
simpler and faster than looping a tracer. Use nnsight (patching.py) only to
*intervene*.

INDEXING: hidden_states[L + 1] is the output of decoder layer L; [0] is the
embedding. This matches model.model.layers[L].output under nnsight, which is
what lets an observation sweep and an intervention sweep be overlaid.

TRAP: hidden_states[-1] has already had the final norm applied (see readout.py).
"""
import numpy as np
import torch


def all_layers(model, tok, prompt, device=None):
    """-> [n_layers + 1, seq, hidden] for one prompt."""
    ids = tok(prompt, return_tensors="pt").to(device or model.device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return torch.stack([h[0] for h in hs])


def pool_tokens(hs, mode="last"):
    """[n_slots, seq, hidden] -> [n_slots, hidden].

    mode: "last" | "mean" | "suffix:k" (mean of final k) | "ema:a" (decay a in
    (0,1]; larger a weights recent tokens more; a=1.0 == "last").

    Worth trying more than one: a probe result that survives only one pooling
    choice is a fact about the pooling.
    """
    seq = hs.shape[1]
    if mode == "last":
        return hs[:, -1, :]
    if mode == "mean":
        return hs.mean(dim=1)
    if mode.startswith("suffix:"):
        return hs[:, -max(1, min(seq, int(mode[7:]))):, :].mean(dim=1)
    if mode.startswith("ema:"):
        a = float(mode[4:])
        if not 0 < a <= 1:
            raise ValueError("ema decay must be in (0, 1]")
        w = a * (1 - a) ** torch.arange(seq - 1, -1, -1, device=hs.device, dtype=hs.dtype)
        return (hs * (w / w.sum())[None, :, None]).sum(dim=1)
    raise ValueError(f"unknown pooling mode {mode!r}")


def pooled(model, tok, prompts, mode="last", device=None):
    """-> [n_prompts, n_layers + 1, hidden] float32 on CPU."""
    return np.stack([pool_tokens(all_layers(model, tok, p, device), mode)
                     .float().cpu().numpy() for p in prompts])


def at_position(model, tok, prompts, position=-1, device=None):
    """-> [n_prompts, n_layers + 1, hidden] at one token position."""
    return np.stack([all_layers(model, tok, p, device)[:, position, :]
                     .float().cpu().numpy() for p in prompts])


def token_index(tok, prompt, word):
    """Index of `word` in prompt's tokens. Raises if absent or ambiguous --
    guessing here yields a sweep that looks fine and means nothing."""
    toks = [tok.decode([t]).strip() for t in tok(prompt)["input_ids"]]
    hits = [i for i, t in enumerate(toks) if t == word]
    if len(hits) != 1:
        raise ValueError(f"{word!r} appears {len(hits)}x in {toks}")
    return hits[0]


def assert_aligned(tok, a, b):
    """Position-wise patching needs equal token counts. Checking only the
    subject index misses a multi-token entity shifting everything after it."""
    ta, tb = tok(a)["input_ids"], tok(b)["input_ids"]
    if len(ta) != len(tb):
        raise ValueError(f"token-count mismatch ({len(ta)} vs {len(tb)}):\n"
                         f"  {[tok.decode([t]) for t in ta]}\n"
                         f"  {[tok.decode([t]) for t in tb]}")
    return len(ta)


def capture_hooked(model, tok, prompts, position=-1, layers=None, device=None):
    """Lower-memory alternative to `at_position` for long sequences.

    `output_hidden_states=True` retains all n_layers+1 tensors of shape
    [batch, seq, hidden]. This hooks the layers you ask for and slices to one
    position inside the hook, so the full-sequence tensors are never retained.

    Measured on Qwen2.5-1.5B, batch 1 (peak forward memory above weights):
        seq  512 : 216 MB retained-all -> 172 MB hooked  (21% lower)
        seq 4096 : 1728 MB            -> 1376 MB         (20% lower)
    The rest of the peak is transient attention/MLP intermediates, which no
    capture strategy avoids. For short prompts the difference is negligible --
    use `at_position`, it is simpler.

    TRAP: the slice MUST be cloned. `out[:, -1, :]` is a VIEW that keeps the
    whole [batch, seq, hidden] base tensor alive, so a hook that forgets the
    clone saves exactly nothing (measured: identical peak to retaining all).

    Equals `at_position(...)[:, L + 1]` for every layer EXCEPT the last: a hook
    on layers[NL-1] sees the raw layer output, while hidden_states[NL] has had
    the model's final norm applied. Same trap as readout.lens_at, from the other
    side. If you want the normed final state, take it from hidden_states.

    -> [n_prompts, len(layers), hidden]; `layers` defaults to all, and indexes
    decoder layers directly (layer L, not slot L+1).
    """
    layers = list(range(len(model.model.layers))) if layers is None else list(layers)
    out = []
    for p in prompts:
        got = {}

        def make(i):
            def hook(mod, inp, o):
                got[i] = o[:, position, :].detach().clone()   # clone, not a view
            return hook

        handles = [model.model.layers[i].register_forward_hook(make(i)) for i in layers]
        try:
            with torch.no_grad():
                model(**tok(p, return_tensors="pt").to(device or model.device))
        finally:
            for h in handles:
                h.remove()
        out.append(np.stack([got[i][0].float().cpu().numpy() for i in layers]))
    return np.stack(out)
