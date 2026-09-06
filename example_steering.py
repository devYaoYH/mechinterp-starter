"""Worked example: build a concept direction, calibrate it, steer with it.

The order matters. Calibrate BEFORE choosing a sweep range -- a raw coefficient
tells you nothing until you know it relative to the residual stream at the site.
And always sweep a matched-norm random direction alongside, or you cannot tell a
concept effect from a magnitude effect.

    python example_steering.py [--layer 14] [--model ID]
"""
import argparse

import numpy as np
import torch

from mechinterp import activations as A
from mechinterp import loading, steering

TRUE = ["The capital of France is Paris.", "The chemical symbol for gold is Au.",
        "A shark is a fish.", "7 times 6 equals 42.",
        "The capital of Japan is Tokyo.", "A dog is a mammal.",
        "The capital of Italy is Rome.", "8 times 9 equals 72."]
FALSE = ["The capital of France is Tokyo.", "The chemical symbol for gold is Na.",
         "A shark is a bird.", "7 times 6 equals 45.",
         "The capital of Japan is Cairo.", "A dog is a reptile.",
         "The capital of Italy is Oslo.", "8 times 9 equals 70."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--quant", default="none", choices=loading.QUANT_MODES)
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    model, tok = loading.load_hf(args.model, args.quant)
    L = args.layer

    # 1. A direction, from pooled activations at the site.
    X = A.pooled(model, tok, TRUE + FALSE, mode="last")[:, L + 1, :]
    y = np.array([1] * len(TRUE) + [0] * len(FALSE))
    d = steering.difference_of_means(X, y)

    # 2. Calibrate against activations AT THE SAME SITE. This is the step whose
    #    absence makes a coefficient sweep uninterpretable.
    cal = steering.calibrate(d, X)
    print(steering.format_calibration(cal, f"truth direction @ layer {L}"))

    # 3. Sweep, in units of the site's activation norm, with a random control.
    prompt = "Question: What is the capital of Australia?\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    rand = steering.random_control(d)

    print(f"\nprompt: {prompt!r}")
    print(f"{'coeff':>7} | {'concept direction':<34} | {'random (matched norm)':<34}")
    print("-" * 82)
    for coeff in (0.0, 0.02, 0.05, 0.1, 0.25):
        outs = []
        for vec in (d, rand):
            with steering.Steerer(model, L, vec, coeff, normalize=True,
                                  site_norm=cal["activation_norm_median"]):
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0, ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().replace("\n", " "))
        print(f"{coeff:7.2f} | {outs[0][:34]:<34} | {outs[1][:34]:<34}")

    print("\nRead the two columns together. If the concept column degrades the same")
    print("way as the random column, the effect is magnitude, not meaning.")


if __name__ == "__main__":
    main()
