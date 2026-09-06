"""Quantize with bitsandbytes, and measure what it cost.

Quantize once and persist: BitsAndBytesConfig against a dense repo re-downloads
and re-quantizes every run, so save the 4-bit weights and reload them as
"prequantized".

NF4 is lossy, and your probes and patching curves are computed on the perturbed
activations. The fidelity API is two-phase so both models are never resident at
once -- at 32B you cannot hold them:

    ref = capture_reference(dense, tok, prompts); free(dense)
    q, tok = loading.load_hf(path, "prequantized")
    print(format_report(compare_against(q, tok, prompts, ref)))
"""
import gc

import numpy as np
import torch


def quantize_and_save(model_id, out_dir, quant_type="nf4",
                      compute_dtype=torch.bfloat16, double_quant=True):
    """Load dense in 4-bit and persist. The DOWNLOAD is still full-precision
    (~65GB for 32B) -- prefer an existing unsloth/*-bnb-4bit repo (~19GB)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant_type,
                             bnb_4bit_compute_dtype=compute_dtype,
                             bnb_4bit_use_double_quant=double_quant)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                 quantization_config=cfg)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir, model.get_memory_footprint()


@torch.no_grad()
def capture_reference(model, tok, prompts):
    """Record final-position logits and per-layer activations, so the dense
    model can be freed before the quantized one loads."""
    logits, acts = [], []
    for p in prompts:
        out = model(**tok(p, return_tensors="pt").to(model.device), output_hidden_states=True)
        logits.append(out.logits[0, -1].float().cpu().numpy())
        acts.append(torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
    L = np.stack(logits)
    return {"logits": L, "acts": np.stack(acts), "top1": L.argmax(-1)}


def _kl(p_logits, q_logits):
    """KL(P || Q) over next-token distributions, in nats."""
    p = torch.log_softmax(torch.from_numpy(p_logits).double(), -1)
    q = torch.log_softmax(torch.from_numpy(q_logits).double(), -1)
    return float((p.exp() * (p - q)).sum())


def compare_against(model, tok, prompts, ref):
    """-> dict(top1_agreement, mean_kl, max_kl, logit_r, layer_cos).

    layer_cos is "how far has the residual stream drifted by this depth". Drift
    compounds, so late layers are worst and a late-sited probe or patch is the
    most exposed.
    """
    cur = capture_reference(model, tok, prompts)
    kls = [_kl(ref["logits"][i], cur["logits"][i]) for i in range(len(prompts))]
    A, B = ref["acts"], cur["acts"]
    cos = (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-9)
    return {"n_prompts": len(prompts),
            "top1_agreement": float((cur["top1"] == ref["top1"]).mean()),
            "mean_kl": float(np.mean(kls)), "max_kl": float(np.max(kls)),
            "logit_r": float(np.corrcoef(ref["logits"].ravel(), cur["logits"].ravel())[0, 1]),
            "layer_cos": cos.mean(0)}


def format_report(rep, name="quantized vs dense"):
    lc = rep["layer_cos"]
    marks = [0, len(lc) // 4, len(lc) // 2, 3 * len(lc) // 4, len(lc) - 1]
    out = [f"{name}  (n={rep['n_prompts']} prompts)",
           f"  top-1 next-token agreement : {rep['top1_agreement']:.3f}",
           f"  mean KL(dense||quant)      : {rep['mean_kl']:.4f} nats (max {rep['max_kl']:.4f})",
           f"  logit correlation          : {rep['logit_r']:.4f}",
           "  residual cosine vs dense   : "
           + "  ".join(f"{m}:{lc[m]:.4f}" for m in marks)]
    if rep["top1_agreement"] < 0.95:
        out.append("  WARNING: top-1 below 0.95 -- this is a different model for your "
                   "purposes; re-check conclusions on dense weights.")
    if lc[-1] < 0.99:
        out.append(f"  WARNING: final-layer cosine {lc[-1]:.4f} -- late-sited probes and "
                   "patches are the most exposed to this drift.")
    return "\n".join(out)


def free(model):
    """Drop a model and reclaim VRAM so the next one can load."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
