"""Quantize a dense checkpoint with bitsandbytes, and check what it cost you.

Two things this exists to make routine:

  1. **Quantize once, reuse.** `BitsAndBytesConfig(load_in_4bit=True)` against a
     dense repo re-downloads and re-quantizes every run. `quantize_and_save()`
     writes the 4-bit weights to disk so subsequent loads are `quant=
     "prequantized"` and cost seconds.
  2. **Know whether quantization changed your finding.** This is the part that
     is specific to interpretability and usually skipped. NF4 is lossy; your
     probe accuracies and patching curves are computed on perturbed
     activations. `capture_reference()` / `compare_against()` quantify the
     damage before you trust a result.

The fidelity API is deliberately two-phase so you never hold both models in
memory at once -- at 32B you cannot. Capture the dense reference, free it,
load the quantized model, compare:

    ref = capture_reference(model, tok, prompts)      # dense
    del model; gc.collect(); torch.cuda.empty_cache()
    qmodel, tok = loading.load_hf(path, "prequantized")
    report = compare_against(qmodel, tok, prompts, ref)
"""
import gc

import numpy as np
import torch


def quantize_and_save(model_id, out_dir, quant_type="nf4", compute_dtype=torch.bfloat16,
                      double_quant=True):
    """Load a dense checkpoint in 4-bit and persist the quantized weights.

    Note the download is still of the FULL-precision checkpoint (~65GB for a
    32B model) -- quantization happens locally, after. If a pre-quantized repo
    exists (e.g. unsloth/*-bnb-4bit, ~19GB) prefer that and skip this entirely.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=double_quant)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                 quantization_config=cfg)
    tok = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir, model.get_memory_footprint()


@torch.no_grad()
def capture_reference(model, tok, prompts, layers=None):
    """Record what the dense model does, so it can be compared after it's freed.

    -> {"logits": [n, vocab], "acts": [n, n_slots, hidden], "top1": [n]}
    """
    logits, acts = [], []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        out = model(**ids, output_hidden_states=True)
        logits.append(out.logits[0, -1].float().cpu().numpy())
        hs = torch.stack([h[0, -1] for h in out.hidden_states])
        acts.append(hs.float().cpu().numpy())
    L = np.stack(logits)
    return {"logits": L, "acts": np.stack(acts), "top1": L.argmax(-1), "prompts": list(prompts)}


def _kl(p_logits, q_logits):
    """KL(P || Q) over next-token distributions, in nats."""
    p = torch.log_softmax(torch.from_numpy(p_logits).double(), -1)
    q = torch.log_softmax(torch.from_numpy(q_logits).double(), -1)
    return float((p.exp() * (p - q)).sum())


@torch.no_grad()
def compare_against(model, tok, prompts, ref):
    """Compare a (quantized) model against a captured dense reference.

    -> dict with top1_agreement, mean_kl, logit_r, and per-layer activation
       cosine similarity. Read `layer_cos` as "how far has the residual stream
       drifted by this depth" -- drift compounds, so late layers are always
       worse, and a probe or patch sited late is the most exposed.
    """
    cur = capture_reference(model, tok, prompts)
    top1 = float((cur["top1"] == ref["top1"]).mean())
    kls = [_kl(ref["logits"][i], cur["logits"][i]) for i in range(len(prompts))]
    r = float(np.corrcoef(ref["logits"].ravel(), cur["logits"].ravel())[0, 1])

    A, B = ref["acts"], cur["acts"]          # [n, slots, hidden]
    num = (A * B).sum(-1)
    cos = num / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-9)
    return {"n_prompts": len(prompts), "top1_agreement": top1,
            "mean_kl": float(np.mean(kls)), "max_kl": float(np.max(kls)),
            "logit_r": r, "layer_cos": cos.mean(0)}


def format_report(rep, name="quantized vs dense"):
    lc = rep["layer_cos"]
    marks = [0, len(lc) // 4, len(lc) // 2, 3 * len(lc) // 4, len(lc) - 1]
    lines = [
        f"{name}  (n={rep['n_prompts']} prompts)",
        f"  top-1 next-token agreement : {rep['top1_agreement']:.3f}",
        f"  mean KL(dense||quant)      : {rep['mean_kl']:.4f} nats  (max {rep['max_kl']:.4f})",
        f"  logit correlation          : {rep['logit_r']:.4f}",
        "  residual-stream cosine vs dense, by slot:",
        "    " + "  ".join(f"{m}:{lc[m]:.4f}" for m in marks),
    ]
    if rep["top1_agreement"] < 0.95:
        lines.append("  WARNING: top-1 agreement below 0.95 -- the quantized model is a "
                     "different model for your purposes; re-check any conclusion on dense.")
    if lc[-1] < 0.99:
        lines.append(f"  WARNING: final-layer cosine {lc[-1]:.4f} -- late-sited probes and "
                     "patches are the most exposed to this drift.")
    return "\n".join(lines)


def free(model):
    """Drop a model and reclaim VRAM, so the next one can load."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
