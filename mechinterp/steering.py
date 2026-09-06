"""Concept directions, and adding them back to the residual stream.

The load-bearing function here is `calibrate`. A steering coefficient is
meaningless on its own: `coeff=1.0` is a gentle nudge or a wrecking ball
depending entirely on the direction's norm relative to the residual stream at
the site you inject it. In one real case a difference-of-means direction had
norm 29.1 at a site whose typical activation norm was ~99 -- so the *smallest*
coefficient tried was already a 29% perturbation applied to every token, and
the whole sweep was measuring "how broken is the model" rather than anything
about the concept. That was only caught by measuring it explicitly.

So: call `calibrate()` and read `perturbation_at_coeff_1` BEFORE picking a
sweep range, and prefer `Steerer(..., normalize=True)`, which puts the
coefficient in units of the site's typical activation norm -- `coeff=0.05` then
means "perturb by 5%" on any model, layer, or direction.

Always run `random_control()` alongside a steering sweep. If a random direction
of matched norm degrades the model the same way, the effect is about magnitude,
not about your concept.
"""
import numpy as np
import torch


def difference_of_means(X, y, positive=1):
    """Concept direction as mean(positive) - mean(negative). X: [n, hidden].

    Preferred over a logistic-regression weight vector for *steering*: the LR
    weights are shaped by whatever separates the classes given the regularizer,
    including low-variance directions that are discriminative but tiny, and
    pushing along those is unpredictable. Diff-of-means points along the actual
    displacement between the two class clusters.
    """
    X, y = np.asarray(X), np.asarray(y)
    pos, neg = X[y == positive], X[y != positive]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need examples of both classes")
    return pos.mean(axis=0) - neg.mean(axis=0)


def calibrate(direction, site_activations, coeffs=(0.5, 1.0, 2.0, 4.0)):
    """How big is this direction relative to the residual stream where it lands?

    site_activations: [n, hidden] activations sampled at the SAME layer and
    position family the direction will be added to. Using a different site makes
    the ratio meaningless.

    -> dict(direction_norm, activation_norm_median/p10/p90,
            perturbation_at_coeff_1, coeff_for, table, warning)

    `coeff_for[f]` is the coefficient that perturbs by fraction f of the median
    activation norm -- the number you actually want when choosing a sweep.
    """
    d = np.asarray(direction, dtype=np.float64)
    A = np.asarray(site_activations, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != d.shape[0]:
        raise ValueError(f"site_activations {A.shape} incompatible with direction {d.shape}")
    dn = float(np.linalg.norm(d))
    norms = np.linalg.norm(A, axis=1)
    med = float(np.median(norms))
    ratio = dn / (med + 1e-9)

    out = {
        "direction_norm": dn,
        "activation_norm_median": med,
        "activation_norm_p10": float(np.percentile(norms, 10)),
        "activation_norm_p90": float(np.percentile(norms, 90)),
        "perturbation_at_coeff_1": ratio,
        "coeff_for": {f: f / (ratio + 1e-9) for f in (0.01, 0.02, 0.05, 0.1, 0.25)},
        "table": [(c, c * ratio) for c in coeffs],
        "warning": None,
    }
    if ratio > 0.1:
        out["warning"] = (
            f"coeff=1.0 perturbs by {ratio:.1%} of the median activation norm. "
            f"A raw-coefficient sweep starting at 1.0 begins in the destructive "
            f"regime; use normalize=True, or sweep "
            f"{out['coeff_for'][0.01]:.3g}-{out['coeff_for'][0.1]:.3g}.")
    return out


def format_calibration(cal, label="direction"):
    lines = [
        f"{label}: ||d|| = {cal['direction_norm']:.3f}",
        f"  site ||h||: median {cal['activation_norm_median']:.3f} "
        f"(p10 {cal['activation_norm_p10']:.3f}, p90 {cal['activation_norm_p90']:.3f})",
        f"  coeff=1.0 perturbs by {cal['perturbation_at_coeff_1']:.1%} of median ||h||",
        "  raw coeff -> perturbation:  "
        + "  ".join(f"{c:g}:{p:.1%}" for c, p in cal["table"]),
        "  for a target perturbation, use coeff:  "
        + "  ".join(f"{int(f*100)}%:{c:.3g}" for f, c in cal["coeff_for"].items()),
    ]
    if cal["warning"]:
        lines.append("  WARNING: " + cal["warning"])
    return "\n".join(lines)


def random_control(direction, seed=0):
    """A random direction with the same norm. Run your sweep with this too --
    without it you cannot tell a concept effect from a magnitude effect."""
    d = np.asarray(direction, dtype=np.float64)
    rng = np.random.RandomState(seed)
    r = rng.randn(*d.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(d)


class Steerer:
    """Add `coeff * direction` to the residual stream at one layer, via a
    forward hook, for the duration of a `with` block.

    normalize=True (default) rescales the direction to unit norm and multiplies
    by the site's median activation norm, so `coeff` reads as a *fraction* of
    typical activation magnitude. Pass `site_norm` from `calibrate()`; without
    it, normalization falls back to per-call norms of the tensor being modified.

    positions: "last" perturbs only the final position of each forward call
    (i.e. every decoding step), "all" perturbs every position.

    To gate the intervention (fire only on some steps), subclass and override
    `_should_fire`.

    Caveat worth knowing: the hook fires on the layer's *output*, after that
    layer has written its keys/values into the KV cache. So the perturbation
    influences later layers for the current token but does NOT enter the cache
    that future tokens attend to. That is standard for steering, but it means
    the effect does not accumulate through context the way a prompt change does.
    """

    def __init__(self, model, layer, direction, coeff, normalize=True,
                 site_norm=None, positions="last"):
        self.model, self.layer, self.coeff = model, layer, coeff
        self.normalize, self.site_norm = normalize, site_norm
        self.positions = positions
        self.n_calls = self.n_fired = 0
        d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
        if normalize:
            d = d / (d.norm() + 1e-9)
        self.direction = d
        self.handle = None

    def _should_fire(self, h):
        """Override to gate the intervention on the current state."""
        return True

    def _hook(self, module, inputs, output):
        h = output
        if self.coeff == 0:
            return output
        self.n_calls += 1
        if not self._should_fire(h):
            return output
        self.n_fired += 1
        d = self.direction.to(dtype=h.dtype, device=h.device)
        scale = self.coeff
        if self.normalize:
            scale = scale * (self.site_norm if self.site_norm is not None
                             else float(h[:, -1, :].norm(dim=-1).median()))
        h = h.clone()
        if self.positions == "all":
            h = h + scale * d
        else:
            h[:, -1:, :] = h[:, -1:, :] + scale * d
        return h

    def __enter__(self):
        self.handle = self.model.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    @property
    def fire_rate(self):
        return self.n_fired / max(1, self.n_calls)
