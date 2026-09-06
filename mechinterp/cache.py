"""On-disk cache for extracted activations.

Buys back the RE-RUN, not the first run: changing a probe's C, the split, or the
plot re-runs the script but not the forward passes.

Cache at the POOLED boundary, never at `all_layers`: pooled is
[n_layers + 1, hidden] -- 0.18 MB/prompt on a 1.5B, 1.3 MB on a 32B -- while a
full [n_layers + 1, seq, hidden] capture is seq-times larger.

No invalidation logic: the key covers everything that changes the numbers, so a
different setup is a different file. The trap it must cover is quantization --
nf4 and bf16 give DIFFERENT activations for the same prompt, so a cache keyed on
the model NAME alone serves dense numbers into a quantized run and never raises.
"""
import hashlib
import os

import numpy as np


def model_fingerprint(model):
    """Identity of the weights as loaded: repo name, quantization, dtype."""
    cfg = model.config
    q = getattr(cfg, "quantization_config", None)
    quant = f"{q.quant_method}:{getattr(q, 'bnb_4bit_quant_type', '')}" if q else "dense"
    return f"{cfg._name_or_path}|{quant}|{model.dtype}"


def key(fingerprint, spec, prompt):
    """`spec` names the extraction ("pool:last", "pos:-1") so two poolings of one
    prompt are two entries."""
    return hashlib.sha1("\x00".join([fingerprint, spec, prompt]).encode()).hexdigest()[:16]


def load(cache_dir, k):
    path = os.path.join(cache_dir, k + ".npy")
    return np.load(path) if os.path.exists(path) else None


def store(cache_dir, k, arr):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, k + ".npy"), arr)
    return arr
