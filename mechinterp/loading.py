"""Model loading. One place for the quantization branch.

Assumes model.model.layers[i], model.model.norm, model.lm_head -- true of
Llama/Qwen/Mistral-family causal LMs. Run smoke_test.py against a new
architecture before trusting it.
"""
import argparse

import torch

QUANT_MODES = ("none", "nf4", "prequantized")


def load_kwargs(quant="none"):
    """none = bf16. nf4 = quantize on load (downloads the FULL-precision
    checkpoint first). prequantized = weights already 4-bit on disk; prefer it."""
    if quant == "none":
        return dict(device_map="auto", dtype=torch.bfloat16)
    if quant == "prequantized":
        return dict(device_map="auto")
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        return dict(device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16))
    raise ValueError(f"quant must be one of {QUANT_MODES}, got {quant!r}")


def load_hf(model_id, quant="none"):
    """Plain transformers. Use for extraction and forward-hook interventions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return (AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs(quant)).eval(),
            AutoTokenizer.from_pretrained(model_id))


def load_nnsight(model_id, quant="none"):
    """nnsight. Use for cross-run patching, where donor/recipient tracing is far
    more ergonomic than raw hooks."""
    from nnsight import LanguageModel
    return LanguageModel(model_id, **load_kwargs(quant))


def n_layers(model):
    return len(model.model.layers)


def cli(**extra):
    """Shared argparse for the scripts: --model and --quant, plus `extra` as
    {flag: dict(argparse kwargs)}."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--quant", default="none", choices=QUANT_MODES)
    for flag, kw in extra.items():
        ap.add_argument(f"--{flag.replace('_', '-')}", **kw)
    return ap.parse_args()
