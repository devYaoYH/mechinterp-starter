"""Model loading. One place for the quantization branch, so experiment scripts
don't each carry a copy of it."""
import torch

QUANT_MODES = ("none", "nf4", "prequantized")


def load_kwargs(quant="none"):
    """kwargs for from_pretrained / nnsight.LanguageModel.

    none          bf16 full precision.
    nf4           quantize on load via bitsandbytes. Downloads the FULL-precision
                  checkpoint first, then quantizes -- only use when no
                  pre-quantized repo exists (~65GB vs ~19GB for a 32B model).
    prequantized  checkpoint already stores NF4 weights + its own
                  quantization_config (e.g. unsloth/*-bnb-4bit). Prefer this.
    """
    if quant == "none":
        return dict(device_map="auto", dtype=torch.bfloat16)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        return dict(device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16))
    if quant == "prequantized":
        return dict(device_map="auto")
    raise ValueError(f"quant must be one of {QUANT_MODES}, got {quant!r}")


def load_hf(model_id, quant="none"):
    """Plain transformers model + tokenizer. Use for extraction (fast path:
    output_hidden_states) and for forward-hook interventions during generate()."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs(quant)).eval()
    return model, tok


def load_nnsight(model_id, quant="none"):
    """nnsight LanguageModel. Use for cross-run activation patching, where the
    donor/recipient trace pattern is much more ergonomic than raw hooks."""
    from nnsight import LanguageModel
    return LanguageModel(model_id, **load_kwargs(quant))


def n_layers(model):
    """Works for both wrappers, and for Llama/Qwen/Mistral-family causal LMs."""
    return len(model.model.layers)
