"""Template: a fresh question, end to end. Copy this and replace the task.

  1. DEFINE    a contrast set, and a LEAK set sharing the shortcut but not the
               label. Without the leak set, step 2 is not interpretable.
  2. PROBE     every layer, with controls -> where is it READABLE (correlational).
  3. INTERVENE at two positions -> where is it USED (causal).
  4. PLOT      with a reference line, a CI band, and n stated.

    python template_new_task.py [--n-pairs 12] [--out figures]
"""
import itertools
import os

import numpy as np

from mechinterp import activations as A
from mechinterp import loading, patching, plotting, probing
from mechinterp.quantization import free

FACTS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
         ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
         ("Russia", "Moscow"), ("Spain", "Madrid"), ("Greece", "Athens"),
         ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
         ("Poland", "Warsaw"), ("Turkey", "Ankara"), ("Kenya", "Nairobi"),
         ("Peru", "Lima"), ("Chile", "Santiago"), ("Thailand", "Bangkok")]

# Several phrasings per item, so there is >1 example per class and the split can
# be BY TEMPLATE -- forcing generalization across surface form.
TEMPLATES = ["The capital of {c} is", "{c}'s capital is", "The capital city of {c} is",
             "In {c}, the capital city is", "Question: What is the capital of {c}? Answer:",
             "The seat of government of {c} is", "{c} has its capital at"]
HELD_OUT = {1, 4}

# Same subjects, answer is NOT the capital. A probe still predicting the capital
# here is decoding the SUBJECT -- the confound that invalidates most "the model
# knows X" probe results.
LEAK = ["The currency of {c} is", "The official language of {c} is"]


def collect(model, tok, templates, facts):
    """-> X [n, slots, hidden], y (class), groups (template id)."""
    prompts = [t.format(c=c) for t in templates for c, _ in facts]
    y = np.array([i for _ in templates for i, _ in enumerate(facts)])
    g = np.array([ti for ti, _ in enumerate(templates) for _ in facts])
    return A.pooled(model, tok, prompts, mode="last"), y, g


def main():
    args = loading.cli(n_pairs=dict(type=int, default=12), out=dict(default="figures"))
    os.makedirs(args.out, exist_ok=True)

    model, tok = loading.load_hf(args.model, args.quant)
    NL = loading.n_layers(model)
    facts = [(c, a) for c, a in FACTS
             if len(tok(" " + a, add_special_tokens=False)["input_ids"]) == 1]
    depths = [100 * L / NL for L in range(NL)]
    chance = 1 / len(facts)
    print(f"{args.model}: {NL} layers, {len(facts)} facts, chance {chance:.3f}")

    # ---- 2. PROBE: where is it readable? ----
    X, y, g = collect(model, tok, TEMPLATES, facts)
    Xl, yl, _ = collect(model, tok, LEAK, facts)
    tr, te = ~np.isin(g, list(HELD_OUT)), np.isin(g, list(HELD_OUT))
    results = [probing.probe(X[tr, L + 1], y[tr], X[te, L + 1], y[te],
                             leak_X=Xl[:, L + 1], leak_y=yl) for L in range(NL)]
    best = max(range(NL), key=lambda L: results[L]["score"])
    print(probing.format_result(results[best], f"best layer {best}"))

    plotting.save(plotting.probe_panel(
        depths, [r["per_item"] for r in results], [r["shuffled_score"] for r in results],
        chance, "Where is the answer linearly readable?", n=int(te.sum()),
        ylabel="Held-out accuracy (split by template)",
        leak=[r["leak_score"] for r in results]).figure, f"{args.out}/probe_by_layer.png")

    # ---- 3. INTERVENE: where is it used? ----
    free(model)          # release VRAM before nnsight loads the model again
    nn = loading.load_nnsight(args.model, args.quant)

    tmpl = "The capital of {} is"
    n_tok = len(tok(tmpl.format(facts[0][0]))["input_ids"])
    usable = [(c, a) for c, a in facts if len(tok(tmpl.format(c))["input_ids"]) == n_tok]
    subj = A.token_index(tok, tmpl.format(usable[0][0]), usable[0][0])
    eff = {"subject token": [[] for _ in range(NL)], "final token": [[] for _ in range(NL)]}
    used = 0

    for (cc, ac), (dc, ad) in list(itertools.permutations(usable, 2))[:args.n_pairs]:
        pc, pd = tmpl.format(cc), tmpl.format(dc)
        A.assert_aligned(tok, pc, pd)
        ia = tok(" " + ac, add_special_tokens=False)["input_ids"][0]
        ib = tok(" " + ad, add_special_tokens=False)["input_ids"][0]
        clean, corr = patching.baseline_logits(nn, pc), patching.baseline_logits(nn, pd)
        # Skip pairs the model gets wrong -- the normalization denominator is
        # meaningless there.
        if (clean[ia] - clean[ib]).item() < 2 or (corr[ia] - corr[ib]).item() > -2:
            continue
        used += 1
        for name, idx in (("subject token", subj), ("final token", -1)):
            for L in range(NL):
                eff[name][L].append(patching.normalized_effect(
                    clean, corr, patching.patch_at(nn, pd, pc, L, idx), ia, ib))

    print(f"causal sweep: {used} pairs used")

    # ---- 4. PLOT ----
    plotting.save(plotting.layer_sweep(
        eff, depths, "Where is the answer causally used?",
        ylabel="Normalized causal effect", n=used).figure,
        f"{args.out}/causal_by_layer.png")
    grid = np.stack([[np.mean(eff[k][L]) for L in range(NL)] for k in eff])
    plotting.save(plotting.effect_grid(
        grid, list(eff), depths, "Causal effect by position and layer", n=used)[0],
        f"{args.out}/effect_grid.png")
    print(f"wrote 3 figures to {args.out}/")


if __name__ == "__main__":
    main()
