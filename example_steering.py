"""Worked example: build a concept direction, calibrate it, steer with it.

Order matters. Calibrate BEFORE choosing a sweep range -- a raw coefficient
means nothing until you know it relative to the residual stream at the site.
And always sweep a matched-norm random direction alongside, or you cannot tell
a concept effect from a magnitude effect.

    python example_steering.py [--layer 14]
"""
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
PROMPT = "Question: What is the capital of Australia?\nAnswer:"


def main():
    args = loading.cli(layer=dict(type=int, default=14))
    model, tok = loading.load_hf(args.model, args.quant)
    L = args.layer

    # 1. A direction from pooled activations at the site.
    X = A.pooled(model, tok, TRUE + FALSE)[:, L + 1, :]
    d = steering.difference_of_means(X, np.array([1] * len(TRUE) + [0] * len(FALSE)))

    # 2. Calibrate against activations AT THE SAME SITE.
    cal = steering.calibrate(d, X)
    print(steering.format_calibration(cal, f"truth direction @ layer {L}"))

    # 3. Sweep in units of the site's activation norm, with a random control.
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    rand = steering.random_control(d)
    print(f"\nprompt: {PROMPT!r}")
    print(f"{'coeff':>7} | {'concept direction':<34} | {'random (matched norm)':<34}")
    print("-" * 82)
    for coeff in (0.0, 0.02, 0.05, 0.1, 0.25):
        outs = []
        for vec in (d, rand):
            with steering.Steerer(model, L, vec, coeff,
                                  site_norm=cal["activation_norm_median"]):
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0, ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().replace("\n", " "))
        print(f"{coeff:7.2f} | {outs[0][:34]:<34} | {outs[1][:34]:<34}")

    print("\nRead the columns together: if the concept column degrades like the")
    print("random column, the effect is magnitude, not meaning.")


if __name__ == "__main__":
    main()
