"""Verify the whole stack before writing any research code.

Every check here corresponds to a failure that actually happens on a fresh box
or a library bump, and several of them fail SILENTLY -- producing plausible
numbers rather than an exception. Run this first on any new instance, and again
after any dependency upgrade.

    python smoke_test.py [--model ID] [--quant none|nf4|prequantized]
"""
import argparse
import sys

import numpy as np
import torch

from mechinterp import activations as A
from mechinterp import loading, patching, probing, readout

OK, BAD = "  ok  ", " FAIL "
failures = []


def check(name, fn):
    try:
        detail = fn()
        print(f"[{OK}] {name}" + (f" -- {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"[{BAD}] {name} -- {type(e).__name__}: {e}")
        failures.append(name)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--quant", default="none", choices=loading.QUANT_MODES)
    args = ap.parse_args()

    print(f"\n=== environment ===")

    def gpu():
        assert torch.cuda.is_available(), "no CUDA device visible"
        cc = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        # The Blackwell trap: an old CUDA wheel imports fine and reports the
        # device correctly, then dies on the first real kernel.
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        _ = (x @ x).sum().item()
        return f"{name}, cc {cc[0]}.{cc[1]}, torch {torch.__version__}, matmul ok"

    if not check("GPU present and kernels actually run", gpu):
        print("\nStop here: fix the torch/CUDA build before anything else.")
        sys.exit(1)

    print(f"\n=== model: {args.model} (quant={args.quant}) ===")
    model, tok = loading.load_hf(args.model, args.quant)
    NL = loading.n_layers(model)
    print(f"loaded, {NL} decoder layers")

    prompt = "The capital of France is"
    corrupt = "The capital of Germany is"

    def hidden_shape():
        hs = A.all_layers(model, tok, prompt)
        assert hs.shape[0] == NL + 1, f"expected {NL+1} hidden states, got {hs.shape[0]}"
        return f"hidden_states = [{NL}+1, seq={hs.shape[1]}, hid={hs.shape[2]}]"

    check("output_hidden_states indexing (hs[L+1] == layer L)", hidden_shape)

    def extract():
        X = A.at_position(model, tok, [prompt, corrupt], position=-1)
        assert X.shape[0] == 2 and X.shape[1] == NL + 1
        assert np.isfinite(X).all(), "non-finite activations"
        return f"[n=2, layers={X.shape[1]}, hid={X.shape[2]}], finite"

    check("activation extraction at a position", extract)

    def align():
        n = A.assert_aligned(tok, prompt, corrupt)
        i = A.token_index(tok, prompt, "France")
        return f"{n} tokens, subject at index {i}"

    check("token alignment + subject lookup", align)

    def lens():
        hs = A.all_layers(model, tok, prompt)
        lg = readout.lens_at(model, hs, NL)   # slot NL == final, already normed
        top = readout.top_k(tok, lg, 1)[0][0]
        assert top.strip() == "Paris", f"final-layer logit lens gave {top!r}, expected ' Paris'"
        mid = readout.top_k(tok, readout.lens_at(model, hs, NL - 1), 1)[0][0]
        return f"final slot -> {top!r}; penultimate slot -> {mid!r}"

    check("logit lens agrees with the model's own output", lens)

    # ---- nnsight: API drift + the silent-no-op patch ----
    print(f"\n=== interventions ===")
    nn = loading.load_nnsight(args.model, args.quant)

    def nnsight_api():
        # nnsight >= 0.7: .save() yields the realized tensor outside trace()
        # (older tutorials use .value). transformers >= 5: a decoder layer's
        # .output is the hidden_states tensor, NOT a (tensor,) tuple.
        with torch.no_grad(), nn.trace(prompt):
            out = nn.model.layers[0].output.save()
        assert isinstance(out, torch.Tensor), f"layer .output is {type(out)}, not a Tensor"
        assert out.ndim == 3, f"expected [batch, seq, hid], got {tuple(out.shape)}"
        return f"layer.output is a Tensor {tuple(out.shape)}"

    check("nnsight/transformers API shapes", nnsight_api)

    def patch_does_something():
        """The important one. A patch that silently does nothing returns
        perfectly plausible numbers -- an all-zero effect curve looks like a
        finding. Assert the intervention actually moves the logits."""
        id_p = tok(" Paris", add_special_tokens=False)["input_ids"][0]
        id_b = tok(" Berlin", add_special_tokens=False)["input_ids"][0]
        idx = A.token_index(tok, prompt, "France")
        clean = patching.baseline_logits(nn, prompt)
        corr = patching.baseline_logits(nn, corrupt)
        assert tok.decode([int(clean.argmax())]).strip() == "Paris"
        assert tok.decode([int(corr.argmax())]).strip() == "Berlin"
        # Patching the subject at layer 0 == swapping the input token, so this
        # must fully flip. If it does not, the write is not landing.
        p0 = patching.patch_at(nn, corrupt, prompt, 0, idx)
        e0 = patching.normalized_effect(clean, corr, p0, id_p, id_b)
        assert e0 > 0.9, f"layer-0 subject patch had effect {e0:.3f}, expected ~1.0 (write not landing?)"
        # And the last layer must NOT flip -- if everything flips, you are
        # probably patching the whole run rather than one site.
        pl = patching.patch_at(nn, corrupt, prompt, NL - 1, idx)
        el = patching.normalized_effect(clean, corr, pl, id_p, id_b)
        assert el < 0.2, f"last-layer subject patch had effect {el:.3f}, expected ~0 (patching too much?)"
        # Negative positions must resolve to absolute indices: x[:, -1:0, :] is
        # an EMPTY slice, so a position=-1 patch silently writes nothing and the
        # whole sweep reads exactly 0.000. Patching the final token at the last
        # layer must flip the answer.
        pn = patching.patch_at(nn, corrupt, prompt, NL - 1, -1)
        en = patching.normalized_effect(clean, corr, pn, id_p, id_b)
        assert en > 0.9, f"position=-1 patch had effect {en:.3f}, expected ~1.0 (empty slice?)"
        return (f"subject L0={e0:.3f} (flips), subject L{NL-1}={el:.3f} (no-op), "
                f"final L{NL-1}={en:.3f} via position=-1")

    check("patching actually changes the output (and not everywhere)", patch_does_something)

    # ---- probing harness + its controls ----
    print(f"\n=== probing harness ===")

    def probe_controls():
        rng = np.random.RandomState(0)
        # Separable synthetic data: probe should win, shuffled control should not.
        X = rng.randn(200, 64)
        y = (X[:, 0] > 0).astype(int)          # fully determined by one feature
        tr, te = probing.grouped_split(np.arange(200) // 2, test_frac=0.3)
        r = probing.probe(X[tr], y[tr], X[te], y[te])
        assert r["score"] > 0.85, f"probe failed on separable data: {r['score']:.3f}"
        assert r["shuffled_score"] < 0.7, f"shuffled control too high: {r['shuffled_score']:.3f}"
        return probing.format_result(r).strip()

    check("probe wins on signal, shuffled null does not", probe_controls)

    def probe_detects_leak():
        rng = np.random.RandomState(1)
        X = rng.randn(200, 64)
        y = (X[:, 0] > 0).astype(int)
        # Leak set: same shortcut feature, labels deliberately inverted.
        r = probing.probe(X[:140], y[:140], X[140:], y[140:],
                          leak_X=X[140:], leak_y=1 - y[140:])
        assert r["leak_score"] is not None and r["leak_score"] < 0.3, \
            f"leak control did not register: {r['leak_score']}"
        return f"leak control fires correctly (leak={r['leak_score']:.3f} on inverted labels)"

    check("leak control is wired up", probe_detects_leak)

    def underdetermined_warning():
        rng = np.random.RandomState(2)
        X, y = rng.randn(40, 512), rng.randint(0, 2, 40)
        import warnings as W
        with W.catch_warnings(record=True) as w:
            W.simplefilter("always")
            r = probing.probe(X[:28], y[:28], X[28:], y[28:])
        assert any("n_features" in str(x.message) for x in w), "no warning raised"
        assert r["underdetermined"]
        return "n_features >= n_train is flagged"

    check("underdetermined regime is flagged", underdetermined_warning)

    print(f"\n=== figures ===")

    def figures_render():
        import tempfile

        from mechinterp import plotting
        d = [100 * i / 12 for i in range(12)]
        rng = np.random.RandomState(0)
        per_layer = [list(rng.rand(8)) for _ in range(12)]
        with tempfile.TemporaryDirectory() as td:
            ax = plotting.layer_sweep({"a": per_layer, "b": per_layer}, d, "t", n=8)
            plotting.save(ax.figure, f"{td}/sweep.png")
            ax2 = plotting.probe_panel(d, per_layer, [0.1] * 12, 0.1, "t", n=8,
                                       leak=[0.2] * 12)
            plotting.save(ax2.figure, f"{td}/probe.png")
            fig, _ = plotting.effect_grid(rng.rand(2, 12), ["x", "y"], d, "t", n=8)
            plotting.save(fig, f"{td}/grid.png")
            import os as _os
            sizes = [_os.path.getsize(f"{td}/{f}") for f in
                     ("sweep.png", "probe.png", "grid.png")]
        assert all(s > 5000 for s in sizes), f"suspiciously small renders: {sizes}"
        try:
            plotting.layer_sweep({str(i): per_layer for i in range(4)}, d, "t")
        except ValueError:
            pass
        else:
            raise AssertionError("4-series cap not enforced")
        return f"3 forms render ({', '.join(f'{s//1024}kB' for s in sizes)}), series cap enforced"

    check("figures render headless", figures_render)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The stack is good -- go write the experiment.")


if __name__ == "__main__":
    main()
