"""Worked example: where does a factual association live?

Patches the residual stream at two positions (subject, and the final token)
across every layer, over many prompt pairs, and reports the normalized effect
with a spread. Use it as the template for a new task -- swap `PAIRS` and the
prompt template for your own contrast.

Three things here are deliberate, and are what separate a readable sweep from
an unreadable one:
  * the UNPATCHED baseline is printed first, so you know which end of the scale
    "no effect" is at;
  * the metric is normalized (0 = nothing, 1 = full flip), not an argmax token;
  * more than one pair, so the numbers carry a spread.

    python example_causal_trace.py [--model ID] [--n-pairs 30]
"""
import argparse
import itertools
import statistics as st

import torch

from mechinterp import activations as A
from mechinterp import loading, patching

FACTS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
         ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
         ("Russia", "Moscow"), ("Spain", "Madrid"), ("Greece", "Athens"),
         ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
         ("Poland", "Warsaw"), ("Turkey", "Ankara"), ("Kenya", "Nairobi"),
         ("Peru", "Lima"), ("Chile", "Santiago"), ("Thailand", "Bangkok")]

TEMPLATE = "The capital of {} is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--quant", default="none", choices=loading.QUANT_MODES)
    ap.add_argument("--n-pairs", type=int, default=30)
    args = ap.parse_args()

    model = loading.load_nnsight(args.model, args.quant)
    tok, NL = model.tokenizer, loading.n_layers(model)

    def single_token(w):
        return len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1

    facts = [(c, a) for c, a in FACTS
             if single_token(a) and len(tok(TEMPLATE.format(c))["input_ids"])
             == len(tok(TEMPLATE.format(FACTS[0][0]))["input_ids"])]
    pairs = list(itertools.permutations(facts, 2))[:args.n_pairs]
    print(f"{args.model}: {NL} layers, {len(facts)} usable facts, {len(pairs)} pairs\n")

    subj_idx = A.token_index(tok, TEMPLATE.format(facts[0][0]), facts[0][0])
    eff = {"subject": {l: [] for l in range(NL)}, "final": {l: [] for l in range(NL)}}
    skipped = 0

    for (c_clean, a_clean), (c_corr, a_corr) in pairs:
        p_clean, p_corr = TEMPLATE.format(c_clean), TEMPLATE.format(c_corr)
        A.assert_aligned(tok, p_clean, p_corr)
        id_a = tok(" " + a_clean, add_special_tokens=False)["input_ids"][0]
        id_b = tok(" " + a_corr, add_special_tokens=False)["input_ids"][0]

        clean = patching.baseline_logits(model, p_clean)
        corr = patching.baseline_logits(model, p_corr)
        # Only keep pairs the model actually gets right, or the normalization
        # denominator is meaningless.
        if (clean[id_a] - clean[id_b]).item() < 2 or (corr[id_a] - corr[id_b]).item() > -2:
            skipped += 1
            continue

        for name, idx in [("subject", subj_idx), ("final", -1)]:
            for l in range(NL):
                patched = patching.patch_at(model, p_corr, p_clean, l, idx)
                eff[name][l].append(
                    patching.normalized_effect(clean, corr, patched, id_a, id_b))

    n = len(eff["subject"][0])
    print(f"used {n} pairs ({skipped} skipped: model got a fact wrong)\n")
    print(f"{'layer':>5} {'depth%':>7} | {'subject':>8} {'sd':>6} | {'final':>8} {'sd':>6}")
    print("-" * 50)
    for l in range(NL):
        s, f = eff["subject"][l], eff["final"][l]
        print(f"{l:5d} {100 * l / NL:7.1f} | {st.mean(s):8.3f} {st.pstdev(s):6.3f} "
              f"| {st.mean(f):8.3f} {st.pstdev(f):6.3f}")

    print("\nRead it as: 1.0 = patch fully flipped the answer, 0.0 = patch did nothing.")
    print("A subject-position effect of ~1.0 at layer 0 is expected and trivial --")
    print("it is just swapping the input token. The informative layer is where the")
    print("two columns cross: that is where the fact stops being read from the")
    print("subject and starts being read from the final position.")


if __name__ == "__main__":
    main()
