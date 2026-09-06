"""Verify the whole stack before writing research code.

Every check corresponds to a failure that actually happens on a fresh box or a
library bump, and most of them fail SILENTLY -- producing plausible numbers
rather than an exception. Run first on any new instance, and after any upgrade.

    python smoke_test.py [--model ID] [--quant none|nf4|prequantized]
"""
import sys
import warnings

import numpy as np
import torch

from mechinterp import activations as A
from mechinterp import loading, patching, plotting, probing, readout, rollout, steering

CHECKS = []


def check(section):
    def register(fn):
        CHECKS.append((section, fn))
        return fn
    return register


# --------------------------------------------------------------- environment
@check("environment")
def gpu_kernels_run(c):
    """Blackwell trap: an old CUDA wheel imports fine and reports the device
    correctly, then dies on the first real kernel."""
    assert torch.cuda.is_available(), "no CUDA device visible"
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    cc = torch.cuda.get_device_capability()
    return f"{torch.cuda.get_device_name()}, cc {cc[0]}.{cc[1]}, torch {torch.__version__}"


# ---------------------------------------------------------------- extraction
@check("extraction")
def hidden_state_indexing(c):
    hs = A.all_layers(c["model"], c["tok"], c["prompt"])
    assert hs.shape[0] == c["NL"] + 1, f"expected {c['NL']+1} slots, got {hs.shape[0]}"
    X = A.at_position(c["model"], c["tok"], [c["prompt"], c["corrupt"]])
    assert X.shape[:2] == (2, c["NL"] + 1) and np.isfinite(X).all()
    return f"hs = [{c['NL']}+1, seq={hs.shape[1]}, hid={hs.shape[2]}], finite"


@check("extraction")
def pooling_modes(c):
    hs = A.all_layers(c["model"], c["tok"], c["prompt"])
    shapes = {m: A.pool_tokens(hs, m).shape for m in ("last", "mean", "suffix:3", "ema:0.5")}
    assert len(set(shapes.values())) == 1, shapes
    assert torch.allclose(A.pool_tokens(hs, "ema:1.0"), A.pool_tokens(hs, "last"), atol=1e-2)
    assert not torch.allclose(A.pool_tokens(hs, "mean"), A.pool_tokens(hs, "last"))
    return f"4 modes -> {tuple(shapes['last'])}, ema:1.0 == last"


@check("extraction")
def alignment_and_lookup(c):
    n = A.assert_aligned(c["tok"], c["prompt"], c["corrupt"])
    return f"{n} tokens, subject at index {A.token_index(c['tok'], c['prompt'], 'France')}"


@check("extraction")
def logit_lens_matches_model(c):
    """hs[-1] has ALREADY had the final norm applied; norming twice returns
    confident nonsense rather than raising."""
    hs = A.all_layers(c["model"], c["tok"], c["prompt"])
    top = readout.top_k(c["tok"], readout.lens_at(c["model"], hs, c["NL"]), 1)[0][0]
    assert top.strip() == "Paris", f"final-slot lens gave {top!r}, expected ' Paris'"
    mid = readout.top_k(c["tok"], readout.lens_at(c["model"], hs, c["NL"] - 1), 1)[0][0]
    return f"final slot -> {top!r}; penultimate -> {mid!r}"


# -------------------------------------------------------------- intervention
@check("intervention")
def nnsight_api_shapes(c):
    """nnsight>=0.7 .save() yields a realized tensor; transformers>=5 layer
    .output is the hidden_states tensor, not a (tensor,) tuple."""
    with torch.no_grad(), c["nn"].trace(c["prompt"]):
        out = c["nn"].model.layers[0].output.save()
    assert isinstance(out, torch.Tensor) and out.ndim == 3, type(out)
    return f"layer.output is a Tensor {tuple(out.shape)}"


@check("intervention")
def patch_lands_and_is_localized(c):
    """A patch that silently does nothing returns plausible numbers -- an
    all-zero curve reads as a finding. Assert it moves the logits, and that
    position=-1 does not become the empty slice x[:, -1:0, :]."""
    nn, tok, NL = c["nn"], c["tok"], c["NL"]
    ida = tok(" Paris", add_special_tokens=False)["input_ids"][0]
    idb = tok(" Berlin", add_special_tokens=False)["input_ids"][0]
    idx = A.token_index(tok, c["prompt"], "France")
    clean = patching.baseline_logits(nn, c["prompt"])
    corr = patching.baseline_logits(nn, c["corrupt"])
    assert tok.decode([int(clean.argmax())]).strip() == "Paris"
    assert tok.decode([int(corr.argmax())]).strip() == "Berlin"

    def eff(layer, pos):
        return patching.normalized_effect(
            clean, corr, patching.patch_at(nn, c["corrupt"], c["prompt"], layer, pos), ida, idb)

    e0, eL, eN = eff(0, idx), eff(NL - 1, idx), eff(NL - 1, -1)
    assert e0 > 0.9, f"layer-0 subject patch {e0:.3f}, expected ~1.0 (write not landing?)"
    assert eL < 0.2, f"last-layer subject patch {eL:.3f}, expected ~0 (patching too much?)"
    assert eN > 0.9, f"position=-1 patch {eN:.3f}, expected ~1.0 (empty slice?)"
    return f"subject L0={e0:.3f}, subject L{NL-1}={eL:.3f}, final L{NL-1}={eN:.3f}"


# ------------------------------------------------------------------ steering
@check("steering")
def calibration_math(c):
    rng = np.random.RandomState(0)
    acts = rng.randn(64, 128) * 3.0
    d = steering.difference_of_means(acts, (rng.rand(64) > 0.5).astype(int))
    cal = steering.calibrate(d, acts)
    ratio = cal["direction_norm"] / cal["activation_norm_median"]
    assert abs(cal["perturbation_at_coeff_1"] - ratio) < 1e-6
    for f, coeff in cal["coeff_for"].items():
        assert abs(coeff * ratio - f) < 1e-6, f"coeff_for[{f}] wrong"
    assert steering.calibrate(d * 50, acts)["warning"], "no warning on oversized direction"
    r = steering.random_control(d)
    assert abs(np.linalg.norm(r) - np.linalg.norm(d)) < 1e-6, "control norm mismatch"
    return f"coeff=1 -> {ratio:.1%} of ||h||; warning fires; control norm-matched"


@check("steering")
def hook_lands_and_is_removed(c):
    model, ids = c["model"], c["tok"](c["prompt"], return_tensors="pt").to(c["model"].device)
    with torch.no_grad():
        base = model(**ids).logits[0, -1].clone()
    d = np.ones(model.config.hidden_size)
    with steering.Steerer(model, 5, d, 0.0):
        with torch.no_grad():
            assert torch.allclose(base, model(**ids).logits[0, -1]), "coeff=0 changed output"
    with steering.Steerer(model, 5, d, 0.5, site_norm=50.0):
        with torch.no_grad():
            assert not torch.allclose(base, model(**ids).logits[0, -1]), "steering did nothing"
    with torch.no_grad():
        assert torch.allclose(base, model(**ids).logits[0, -1]), "hook leaked past with-block"
    return "coeff=0 no-op, coeff>0 moves logits, hook removed on exit"


@check("steering")
def rollouts_align_to_tokens(c):
    rs = rollout.sample_rollouts(c["model"], c["tok"], [c["prompt"]], max_new_tokens=6,
                                 n_samples=2, temperature=0.8,
                                 verifier=rollout.contains("Paris"))
    for r in rs:
        assert r.acts.shape[0] == len(r.token_ids), f"{r.acts.shape} vs {len(r.token_ids)}"
        assert r.acts.shape[1] == c["NL"] + 1 and r.correct is not None
    X, y, pos, grp = rollout.stack_steps(rs, layer=5, relative_to="end")
    assert len(X) == len(y) == len(pos) == len(grp) and pos.max() < 0
    return f"{len(rs)} rollouts, acts aligned, {len(X)} probe rows"


# ------------------------------------------------------------------- probing
@check("probing")
def probe_and_null(c):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 64)
    y = (X[:, 0] > 0).astype(int)
    tr, te = probing.grouped_split(np.arange(200) // 2)
    r = probing.probe(X[tr], y[tr], X[te], y[te])
    assert r["score"] > 0.8, f"probe failed on separable data: {r['score']:.3f}"
    assert r["shuffled_score"] < 0.7, f"shuffled null too high: {r['shuffled_score']:.3f}"
    return probing.format_result(r).strip()


@check("probing")
def leak_and_transfer_controls(c):
    rng = np.random.RandomState(1)
    X = rng.randn(200, 32)
    y = (X[:, 0] > 0).astype(int)
    Xb = rng.randn(80, 32)
    yb = (Xb[:, 1] > 0).astype(int)          # a different feature entirely
    r = probing.probe(X[:140], y[:140], X[140:], y[140:],
                      leak_X=X[140:], leak_y=1 - y[140:], transfer_X=Xb, transfer_y=yb)
    assert r["leak_score"] < 0.3, f"leak control did not register: {r['leak_score']}"
    assert r["score"] > 0.8 > r["transfer_score"], "transfer control did not separate"
    return f"leak={r['leak_score']:.3f}, in-domain {r['score']:.3f} vs transfer {r['transfer_score']:.3f}"


@check("probing")
def underdetermined_is_flagged(c):
    rng = np.random.RandomState(2)
    X, y = rng.randn(40, 512), rng.randint(0, 2, 40)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = probing.probe(X[:28], y[:28], X[28:], y[28:])
    assert r["underdetermined"] and any("n_features" in str(x.message) for x in w)
    return "n_features >= n_train is flagged"


# ------------------------------------------------------------------- figures
@check("figures")
def figures_render_headless(c):
    import os
    import tempfile
    rng = np.random.RandomState(0)
    depths = [100 * i / 12 for i in range(12)]
    per_layer = [list(rng.rand(8)) for _ in range(12)]
    with tempfile.TemporaryDirectory() as td:
        plotting.save(plotting.layer_sweep({"a": per_layer, "b": per_layer},
                                           depths, "t", n=8).figure, f"{td}/a.png")
        plotting.save(plotting.probe_panel(depths, per_layer, [0.1] * 12, 0.1, "t",
                                           n=8, leak=[0.2] * 12).figure, f"{td}/b.png")
        plotting.save(plotting.effect_grid(rng.rand(2, 12), ["x", "y"], depths, "t", n=8)[0],
                      f"{td}/c.png")
        sizes = [os.path.getsize(f"{td}/{f}.png") for f in "abc"]
    assert all(s > 5000 for s in sizes), f"suspiciously small renders: {sizes}"
    try:
        plotting.layer_sweep({str(i): per_layer for i in range(4)}, depths, "t")
    except ValueError:
        pass
    else:
        raise AssertionError("4-series cap not enforced")
    return f"3 forms render ({', '.join(f'{s//1024}kB' for s in sizes)}), cap enforced"


def main():
    args = loading.cli()
    ctx, failed, section = {}, [], None
    for sec, fn in CHECKS:
        if sec != section:
            section = sec
            print(f"\n=== {sec} ===")
        if sec == "extraction" and "model" not in ctx:
            ctx["model"], ctx["tok"] = loading.load_hf(args.model, args.quant)
            ctx["NL"] = loading.n_layers(ctx["model"])
            ctx.update(prompt="The capital of France is", corrupt="The capital of Germany is")
            print(f"loaded {args.model} (quant={args.quant}), {ctx['NL']} layers")
        if sec == "intervention" and "nn" not in ctx:
            ctx["nn"] = loading.load_nnsight(args.model, args.quant)
        try:
            print(f"[  ok  ] {fn.__name__} -- {fn(ctx)}")
        except Exception as e:
            print(f"[ FAIL ] {fn.__name__} -- {type(e).__name__}: {e}")
            failed.append(fn.__name__)
            if fn.__name__ == "gpu_kernels_run":
                print("\nStop here: fix the torch/CUDA build before anything else.")
                sys.exit(1)
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(CHECKS)} checks passed. The stack is good -- go write the experiment.")


if __name__ == "__main__":
    main()
