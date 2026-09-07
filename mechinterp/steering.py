"""Concept directions and residual-stream steering.

`calibrate` is the load-bearing function. A coefficient is meaningless until you
know it relative to the residual stream where it lands: a direction of norm 29 at
a site whose typical norm is 99 makes even coeff=0.5 a 15% perturbation of every
token, so the sweep measures "how broken is the model", not the concept.

Always sweep `random_control()` alongside. If a matched-norm random direction
degrades output the same way, the effect is magnitude, not meaning.

To CHECK either one landed, read the site with `activations.capture_hooked`,
not `all_layers`/`pooled` -- `output_hidden_states` does not see forward-hook
substitutions, so an intervention that worked reads as one that did nothing.

`Steerer` adds a direction; `Ablator` projects one out. Adding tests whether the
direction can drive behaviour, ablating tests whether the model needs it -- the
second is what a probe result actually asks.
"""
import numpy as np
import torch


def difference_of_means(X, y, positive=1):
    """mean(positive) - mean(negative). X: [n, hidden].

    Preferred over LR weights for *steering*: LR weights are shaped by the
    regularizer and can point along low-variance discriminative directions that
    are unpredictable to push on. This points along the actual class displacement.
    """
    X, y = np.asarray(X), np.asarray(y)
    pos, neg = X[y == positive], X[y != positive]
    if not len(pos) or not len(neg):
        raise ValueError("need examples of both classes")
    return pos.mean(0) - neg.mean(0)


def calibrate(direction, site_activations, coeffs=(0.5, 1.0, 2.0, 4.0)):
    """Direction norm as a fraction of the residual norm where it lands.

    site_activations [n, hidden] must come from the SAME layer and position
    family you will inject at, or the ratio is meaningless.

    -> dict; `coeff_for[f]` is the coefficient giving a perturbation of
    fraction f -- the number you actually want when picking a sweep range.
    """
    d, A = np.asarray(direction, float), np.asarray(site_activations, float)
    if A.ndim != 2 or A.shape[1] != d.shape[0]:
        raise ValueError(f"site_activations {A.shape} vs direction {d.shape}")
    dn, norms = float(np.linalg.norm(d)), np.linalg.norm(A, axis=1)
    med = float(np.median(norms))
    ratio = dn / (med + 1e-9)
    cal = {"direction_norm": dn, "activation_norm_median": med,
           "activation_norm_p10": float(np.percentile(norms, 10)),
           "activation_norm_p90": float(np.percentile(norms, 90)),
           "perturbation_at_coeff_1": ratio,
           "coeff_for": {f: f / (ratio + 1e-9) for f in (0.01, 0.02, 0.05, 0.1, 0.25)},
           "table": [(c, c * ratio) for c in coeffs], "warning": None}
    if ratio > 0.1:
        cal["warning"] = (f"coeff=1.0 perturbs by {ratio:.1%} of median ||h||; a raw "
                          f"sweep from 1.0 starts in the destructive regime. Use "
                          f"normalize=True or sweep {cal['coeff_for'][0.01]:.3g}"
                          f"-{cal['coeff_for'][0.1]:.3g}.")
    return cal


def format_calibration(cal, label="direction"):
    out = [f"{label}: ||d|| = {cal['direction_norm']:.3f}",
           f"  site ||h||: median {cal['activation_norm_median']:.3f} "
           f"(p10 {cal['activation_norm_p10']:.3f}, p90 {cal['activation_norm_p90']:.3f})",
           f"  coeff=1.0 perturbs by {cal['perturbation_at_coeff_1']:.1%} of median ||h||",
           "  raw coeff -> perturbation:  "
           + "  ".join(f"{c:g}:{p:.1%}" for c, p in cal["table"]),
           "  for a target perturbation, use coeff:  "
           + "  ".join(f"{int(f*100)}%:{c:.3g}" for f, c in cal["coeff_for"].items())]
    return "\n".join(out + ([f"  WARNING: {cal['warning']}"] if cal["warning"] else []))


def random_control(direction, seed=0):
    """Random direction of matched norm -- the control that separates a concept
    effect from a magnitude effect."""
    d = np.asarray(direction, float)
    r = np.random.RandomState(seed).randn(*d.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(d)


class Steerer:
    """Add `coeff * direction` at one layer for the duration of a `with` block.

    normalize=True puts coeff in units of the site's activation norm, so
    coeff=0.05 means "perturb by 5%" on any model or layer. Pass `site_norm`
    from calibrate(); otherwise it is taken per-call from the tensor itself.

    positions: "last" (every decoding step) or "all". Override `_should_fire`
    to gate the intervention.

    CAVEAT: the hook fires on the layer output, after that layer wrote its KV
    for this token. The perturbation reaches later layers for the current token
    but never enters the cache future tokens attend to -- so it does not
    accumulate through context the way a prompt change does.
    """

    def __init__(self, model, layer, direction, coeff, normalize=True,
                 site_norm=None, positions="last"):
        self.model, self.layer, self.coeff = model, layer, coeff
        self.normalize, self.site_norm, self.positions = normalize, site_norm, positions
        self.n_calls = self.n_fired = 0
        self.handle = None
        d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
        self.direction = d / (d.norm() + 1e-9) if normalize else d

    def _should_fire(self, h):
        return True

    def _hook(self, module, inputs, h):
        if self.coeff == 0:
            return h
        self.n_calls += 1
        if not self._should_fire(h):
            return h
        self.n_fired += 1
        scale = self.coeff
        if self.normalize:
            scale *= (self.site_norm if self.site_norm is not None
                      else float(h[:, -1, :].norm(dim=-1).median()))
        delta = scale * self.direction.to(dtype=h.dtype, device=h.device)
        h = h.clone()
        if self.positions == "all":
            return h + delta
        h[:, -1:, :] += delta
        return h

    def __enter__(self):
        self.handle = self.model.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    @property
    def fire_rate(self):
        return self.n_fired / max(1, self.n_calls)


def mean_component(direction, site_activations):
    """Mean projection of the site's activations onto the direction.

    Pass to `Ablator(mean_component=...)` to mean-ablate rather than zero-ablate.
    Zeroing a component the model never sets to zero moves the stream off
    distribution, so a behaviour change can mean "I broke it" rather than "it
    needed that direction". The mean removes the variation, not the stream.
    """
    d = np.asarray(direction, float)
    A = np.asarray(site_activations, float)
    return float((A @ (d / (np.linalg.norm(d) + 1e-9))).mean())


class Ablator(Steerer):
    """Project a direction OUT of the residual stream at one layer:

        h <- h - coeff * ((h.d_hat) - mean_component) * d_hat

    The causal test that pairs with a PROBE. `Steerer` asks "does ADDING this
    direction change behaviour", which a direction the model ignores can still
    answer yes to if you push hard enough. This asks "does the model still work
    WITHOUT it" -- the question a probe result actually raises.

    coeff is a FRACTION of the component: 1.0 removes it, 0.5 halves it. So
    there is no destructive regime to calibrate away from, unlike steering.

    Still sweep `random_control(direction)` alongside: ablating ANY direction
    costs the model something, and the claim needs this one to cost more.

    The KV-cache caveat on `Steerer` applies unchanged.
    """

    def __init__(self, model, layer, direction, coeff=1.0, positions="last",
                 mean_component=0.0):
        super().__init__(model, layer, direction, coeff, normalize=True,
                         positions=positions)
        self.mean_component = mean_component

    def _hook(self, module, inputs, h):
        if self.coeff == 0:
            return h
        self.n_calls += 1
        if not self._should_fire(h):
            return h
        self.n_fired += 1
        d = self.direction.to(dtype=h.dtype, device=h.device)
        sl = slice(None) if self.positions == "all" else slice(-1, None)
        h = h.clone()
        comp = (h[:, sl, :] * d).sum(-1, keepdim=True)          # [b, k, 1]
        h[:, sl, :] -= self.coeff * (comp - self.mean_component) * d
        return h
