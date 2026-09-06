"""Template: take a fresh question from contrast set -> probe -> intervention -> figure.

Copy this file, replace `build_task()`, keep the rest. The worked task here is
"where does the model represent, and where does it use, the answer to a factual
lookup" -- but the arc is the point, not the task:

  1. DEFINE   a contrast set, and a LEAK set that shares the shortcut but not
              the label. Without the leak set step 2 is not interpretable.
  2. PROBE    every layer, with the shuffled null and leak control alongside.
              This tells you where the property is READABLE. It is correlational.
  3. INTERVENE at the layers the probe likes, and only then claim causality.
              A probe that reads a property does not show the model uses it.
  4. PLOT     with a reference line, a CI band, and n stated.

    python template_new_task.py [--n-pairs 12] [--out figures/]
"""
import argparse
import itertools
import os

import numpy as np

from mechinterp import activations as A
from mechinterp import loading, patching, plotting, probing

FACTS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
         ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
         ("Russia", "Moscow"), ("Spain", "Madrid"), ("Greece", "Athens"),
         ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
         ("Poland", "Warsaw"), ("Turkey", "Ankara"), ("Kenya", "Nairobi"),
         ("Peru", "Lima"), ("Chile", "Santiago"), ("Thailand", "Bangkok")]

# Several phrasings per item, so the probe has >1 example per class and the
# train/test split can be BY TEMPLATE -- forcing generalization across surface
# form rather than memorization of one string.
TEMPLATES = [("The capital of {c}", " is"), ("{c}'s capital", " is"),
             ("The capital city of {c}", " is"), ("In {c}, the capital city", " is"),
             ("Question: What is the capital of {c}?", " Answer:"),
             ("The seat of government of {c}", " is"), ("{c} has its capital", " at")]
HELD_OUT_TEMPLATES = {1, 4}

# Same subjects, but the answer is NOT the capital. A probe that still predicts
# the capital here is decoding the SUBJECT, not the retrieved answer -- the
# confound that invalidates most "the model knows X" probe results.
LEAK_TEMPLATES = [("The currency of {c}", " is"), ("The official language of {c}", " is")]


def collect(model, tok, templates, facts, n_layers):
    """-> X [n, n_layers+1, hidden], y (class per item), groups (template id)."""
    X, y, g = [], [], []
    for ti, (pre, suf) in enumerate(templates):
        for country, cap in facts:
            ids_pre = tok(pre.format(c=country))["input_ids"]
            prompt = pre.format(c=country) + suf
            hs = A.all_layers(model, tok, prompt)
            X.append(hs[:, -1, :].float().cpu().numpy())
            y.append(facts.index((country, cap)))
            g.append(ti)
    return np.stack(X), np.array(y), np.array(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--quant", default="none", choices=loading.QUANT_MODES)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = loading.load_hf(args.model, args.quant)
    NL = loading.n_layers(model)
    facts = [(c, a) for c, a in FACTS
             if len(tok(" " + a, add_special_tokens=False)["input_ids"]) == 1]
    depths = [100 * L / NL for L in range(NL)]
    print(f"{args.model}: {NL} layers, {len(facts)} facts")

    # ---- 2. PROBE: where is the answer READABLE? (correlational) ----
    X, y, g = collect(model, tok, TEMPLATES, facts, NL)
    Xl, yl, _ = collect(model, tok, LEAK_TEMPLATES, facts, NL)
    tr = ~np.isin(g, list(HELD_OUT_TEMPLATES))
    te = ~tr
    print(f"probe: train={tr.sum()} test={te.sum()} (split by template), "
          f"leak set={len(yl)}, chance={1/len(facts):.3f}")

    scores, shuffled, leak, per_item = [], [], [], []
    for L in range(NL):
        r = probing.probe(X[tr, L + 1], y[tr], X[te, L + 1], y[te],
                          leak_X=Xl[:, L + 1], leak_y=yl)
        scores.append(r["score"]); shuffled.append(r["shuffled_score"])
        leak.append(r["leak_score"]); per_item.append(r["per_item"])
    print(probing.format_result(
        dict(score=scores[-1], shuffled_score=shuffled[-1], leak_score=leak[-1],
             chance=1 / len(facts), underdetermined=False), "final layer"))

    fig_p = plotting.probe_panel(
        depths, per_item, shuffled, 1 / len(facts),
        "Where is the answer linearly readable?", n=int(te.sum()),
        ylabel="Held-out accuracy (split by template)", leak=leak)
    plotting.save(fig_p.figure, f"{args.out}/probe_by_layer.png")

    # ---- 3. INTERVENE: where is it USED? (causal) ----
    del model
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    nn = loading.load_nnsight(args.model, args.quant)
    tmpl = "The capital of {} is"
    usable = [(c, a) for c, a in facts
              if len(tok(tmpl.format(c))["input_ids"]) == len(tok(tmpl.format(facts[0][0]))["input_ids"])]
    pairs = list(itertools.permutations(usable, 2))[:args.n_pairs]
    subj = A.token_index(tok, tmpl.format(usable[0][0]), usable[0][0])

    eff = {"subject token": [[] for _ in range(NL)], "final token": [[] for _ in range(NL)]}
    used = 0
    for (cc, ac), (dc, ad) in pairs:
        pc, pd = tmpl.format(cc), tmpl.format(dc)
        A.assert_aligned(tok, pc, pd)
        ia = tok(" " + ac, add_special_tokens=False)["input_ids"][0]
        ib = tok(" " + ad, add_special_tokens=False)["input_ids"][0]
        clean, corr = patching.baseline_logits(nn, pc), patching.baseline_logits(nn, pd)
        if (clean[ia] - clean[ib]).item() < 2 or (corr[ia] - corr[ib]).item() > -2:
            continue
        used += 1
        for name, idx in [("subject token", subj), ("final token", -1)]:
            for L in range(NL):
                p = patching.patch_at(nn, pd, pc, L, idx)
                eff[name][L].append(patching.normalized_effect(clean, corr, p, ia, ib))

    # ---- 4. PLOT ----
    ax = plotting.layer_sweep(
        eff, depths, "Where is the answer causally used?",
        ylabel="Normalized causal effect", reference=0.0, reference_label="no effect",
        n=used)
    plotting.save(ax.figure, f"{args.out}/causal_by_layer.png")

    grid = np.stack([[np.mean(eff[k][L]) for L in range(NL)] for k in eff])
    fig_g, _ = plotting.effect_grid(grid, list(eff), depths,
                                    "Causal effect by position and layer", n=used)
    plotting.save(fig_g, f"{args.out}/effect_grid.png")
    print(f"\nwrote 3 figures to {args.out}/")


if __name__ == "__main__":
    main()
