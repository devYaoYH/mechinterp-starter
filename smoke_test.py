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
from mechinterp import cache, loading, patching, plotting, probing, readout, rollout, steering

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
def hooked_capture_matches(c):
    """Lower-memory path must agree with the simple one -- except at the last
    layer, where hidden_states has had the final norm applied but a raw hook has
    not. The slice must also be CLONED: a view keeps the whole [seq, hidden]
    base alive and saves nothing."""
    NL = c["NL"]
    layers = [0, NL // 2, NL - 1]
    hooked = A.capture_hooked(c["model"], c["tok"], [c["prompt"]], layers=layers)
    full = A.at_position(c["model"], c["tok"], [c["prompt"]])
    for j, L in enumerate(layers[:-1]):
        assert np.allclose(hooked[0, j], full[0, L + 1], atol=1e-3), f"layer {L} mismatch"
    assert not np.allclose(hooked[0, -1], full[0, NL], atol=1e-3), \
        "last layer matched hidden_states -- the final norm should make them differ"
    return f"layers {layers[:-1]} match slot L+1; last layer differs (final norm)"


@check("extraction")
def cache_round_trips_and_keys_on_weights(c):
    """A cache keyed on the model NAME alone serves dense activations into a
    quantized run and never raises. The key must cover the weights as loaded."""
    import os
    import tempfile
    model, tok, prompts = c["model"], c["tok"], [c["prompt"], c["corrupt"]]
    with tempfile.TemporaryDirectory() as td:
        cold = A.pooled(model, tok, prompts, cache=td)
        assert np.array_equal(cold, A.pooled(model, tok, prompts, cache=td))
        A.pooled(model, tok, prompts, mode="mean", cache=td)
        assert len(os.listdir(td)) == 4, "pooling mode is not part of the key"

    fp = cache.model_fingerprint(model)
    assert cache.key(fp, "pool:last", c["prompt"]) != \
        cache.key(fp.replace("dense", "bitsandbytes:nf4"), "pool:last", c["prompt"])
    return f"round-trip ok, key = {fp}"


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


@check("intervention")
def hoisted_sweep_matches_per_layer(c):
    """sweep_pair reads the donor once for every (layer, position) instead of
    re-tracing per layer. Only a speedup if it is also identical -- a state
    saved for the wrong layer still produces a smooth, plausible curve."""
    nn, tok, NL = c["nn"], c["tok"], c["NL"]
    ida = tok(" Paris", add_special_tokens=False)["input_ids"][0]
    idb = tok(" Berlin", add_special_tokens=False)["input_ids"][0]
    idx = A.token_index(tok, c["prompt"], "France")
    layers = [0, NL // 2, NL - 1]
    base = (patching.baseline_logits(nn, c["prompt"]),
            patching.baseline_logits(nn, c["corrupt"]))

    got, _, _ = patching.sweep_pair(nn, c["prompt"], c["corrupt"],
                                    {"subj": idx, "final": -1}, ida, idb,
                                    layers=layers, baselines=base)
    for name, pos in (("subj", idx), ("final", -1)):
        for L in layers:
            want = patching.normalized_effect(
                *base, patching.patch_at(nn, c["corrupt"], c["prompt"], L, pos), ida, idb)
            assert abs(got[name][L] - want) < 1e-4, \
                f"{name} L{L}: hoisted {got[name][L]:.5f} vs per-layer {want:.5f}"
    return f"identical to patch_at over {len(layers)} layers x 2 positions"


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
def ablation_removes_the_component(c):
    """Two failures at once. An ablation hook that misses looks exactly like
    "the model did not need that direction" -- the null it exists to rule out.
    And `output_hidden_states` is BLIND to forward-hook substitution, so the
    obvious way to check the first one reports a miss that did not happen."""
    model, tok, L = c["model"], c["tok"], c["NL"] // 2
    X = A.pooled(model, tok, [c["prompt"], c["corrupt"]])[:, L + 1, :]
    d = steering.difference_of_means(X, np.array([1, 0]))
    dh = d / np.linalg.norm(d)

    def component(**kw):
        with steering.Ablator(model, L, d, **kw):
            got = A.capture_hooked(model, tok, [c["prompt"]], position=-1, layers=[L])
        return float(got[0, 0] @ dh)

    base, mc = float(X[0] @ dh), steering.mean_component(d, X)
    full, half, kept = (component(coeff=1.0), component(coeff=0.5),
                        component(coeff=1.0, mean_component=mc))
    assert abs(full) < 0.05 * abs(base), f"component {full:.4f} survived from {base:.4f}"
    assert abs(half - 0.5 * base) < 0.1 * abs(base), f"coeff=0.5 gave {half:.4f}"
    assert abs(kept - mc) < 0.05 * abs(mc), f"mean-ablated to {kept:.4f}, wanted {mc:.4f}"

    with steering.Ablator(model, L, d, coeff=1.0) as ab:
        blind = A.all_layers(model, tok, c["prompt"])[L + 1][-1].float().cpu().numpy() @ dh
        assert ab.n_fired > 0, "hook never fired"
    assert abs(float(blind) - base) < 1e-3, (
        "output_hidden_states now REFLECTS forward hooks -- transformers changed; "
        "drop the warning in activations.all_layers and steering.Steerer")
    after = float(A.capture_hooked(model, tok, [c["prompt"]], position=-1, layers=[L])[0, 0] @ dh)
    assert abs(after - base) < 1e-3, f"hook leaked past the with-block ({after:.4f})"
    return f"{base:.3f} -> zero {full:.3f}, half {half:.3f}, mean {kept:.3f}"


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
def recall_at_fpr_math(c):
    """A too-small control set does not raise -- it returns an unstable number
    read off the largest two or three control scores."""
    rng = np.random.RandomState(3)
    ctl = rng.randn(2000)
    sep = probing.recall_at_fpr(ctl + 10, ctl)
    null = probing.recall_at_fpr(rng.randn(500), ctl)
    assert sep["recall"] > 0.99 and sep["warning"] is None, sep
    assert null["recall"] < 0.05, f"separated the unseparable: {null['recall']:.3f}"
    assert null["ci"][0] <= null["recall"] <= null["ci"][1], null["ci"]
    assert probing.recall_at_fpr(ctl + 10, ctl[:30])["warning"], "small control not flagged"

    X = rng.randn(300, 16)
    y = (X[:, 0] > 0).astype(int)
    r = probing.probe(X[:200], y[:200], X[200:], y[200:],
                      control_X=X[200:][y[200:] == 0], fpr=0.1)
    assert r["recall_at_fpr"]["recall"] > 0.5, r["recall_at_fpr"]
    return f"separated={sep['recall']:.2f}, null={null['recall']:.2f}, wired into probe()"


@check("probing")
def leak_metric_matches_headline(c):
    """An AUROC score printed next to an accuracy leak_score reads as
    comparable. It is not, and nothing raises."""
    rng = np.random.RandomState(4)
    X = rng.randn(200, 16)
    y = (X[:, 0] > 0).astype(int)
    r = probing.probe(X[:140], y[:140], X[140:], y[140:],
                      leak_X=X[140:], leak_y=y[140:], metric="auroc")
    one = probing.probe(X[:140], y[:140], X[140:], y[140:],
                        leak_X=X[140:], leak_y=np.ones(60, int), metric="auroc")
    assert 0.9 < r["leak_score"] <= 1.0, r["leak_score"]
    assert one["leak_score"] is None, "single-class AUROC should be None, not a number"
    return f"auroc leak={r['leak_score']:.3f}, single-class -> None"


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
