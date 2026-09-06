"""Activation extraction.

Uses plain transformers `output_hidden_states=True` rather than a tracing
framework: one forward pass returns every layer at once, which is both simpler
and substantially faster than looping a tracer over layers. Reach for nnsight
(see patching.py) only when you need to *intervene*, not merely observe.

INDEXING CONVENTION -- the one thing to get right:
    hidden_states[0]     = embedding output (before any decoder layer)
    hidden_states[L + 1] = output of decoder layer L
So `layer L` in this codebase always means hidden_states[L + 1], which matches
what `model.model.layers[L].output` gives you under nnsight. Keeping these two
aligned is what lets an observation sweep and a patching sweep be overlaid.
"""
import numpy as np
import torch


def all_layers(model, tok, prompt, device=None):
    """-> tensor [n_layers + 1, seq, hidden] for one prompt (batch size 1)."""
    ids = tok(prompt, return_tensors="pt").to(device or model.device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return torch.stack([h[0] for h in hs])


def at_position(model, tok, prompts, position=-1, device=None):
    """Activations at one token position, every layer, for many prompts.

    position: int index into the sequence; -1 = last token.
    -> np.ndarray [n_prompts, n_layers + 1, hidden] (float32, on CPU)
    """
    out = []
    for p in prompts:
        hs = all_layers(model, tok, p, device=device)
        out.append(hs[:, position, :].float().cpu().numpy())
    return np.stack(out)


def token_index(tok, prompt, word):
    """Index of `word` in `prompt`'s token sequence. Raises if absent or
    ambiguous -- silently guessing here is a classic source of a sweep that
    looks fine and means nothing."""
    ids = tok(prompt)["input_ids"]
    toks = [tok.decode([t]).strip() for t in ids]
    hits = [i for i, t in enumerate(toks) if t == word]
    if not hits:
        raise ValueError(f"{word!r} not found in tokens {toks}")
    if len(hits) > 1:
        raise ValueError(f"{word!r} is ambiguous in {toks} (indices {hits})")
    return hits[0]


def assert_aligned(tok, prompt_a, prompt_b):
    """Two prompts must tokenize to the same length for position-wise patching
    to be meaningful. Checking only the subject index is not enough -- a
    multi-token entity shifts every position after it."""
    a, b = tok(prompt_a)["input_ids"], tok(prompt_b)["input_ids"]
    if len(a) != len(b):
        raise ValueError(
            f"token-count mismatch ({len(a)} vs {len(b)}):\n"
            f"  {[tok.decode([t]) for t in a]}\n  {[tok.decode([t]) for t in b]}")
    return len(a)
