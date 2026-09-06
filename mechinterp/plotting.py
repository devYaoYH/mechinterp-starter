"""Presentation-ready figures for layer sweeps and probe results.

Three forms cover almost every mech-interp figure, and each encodes the metric
reporting you should not have to remember:

    layer_sweep()   several series over network depth      -> multi-line + CI band
    probe_panel()   one probe score against its controls   -> emphasis (accent + gray)
    effect_grid()   position x layer effects               -> sequential heatmap

Every one of them draws its reference line (chance, or zero effect), shows a
confidence band, and prints `n` and the CI method in the footer -- because a
mech-interp curve without a baseline is genuinely ambiguous to read, and one
without a band invites reading noise as structure.

Colors are the first three slots of a CVD-validated categorical palette
(all-pairs deltaE 9.2 deutan / 24.0 normal, light surface). Do not add a fourth
line: fold it into "other" or facet into small multiples. Aqua sits below 3:1
contrast on the light surface, so series are direct-labeled as well as
legended -- identity is never carried by color alone.
"""
import os

import numpy as np

# Series slots 1-3 of the validated categorical palette. Order is the
# CVD-safety mechanism, not cosmetic -- assign in order, never cycle.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, CONTEXT, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e2e1db"
SEQ_HUE = "#2a78d6"


def _plt():
    import matplotlib
    if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def style(ax, xlabel, ylabel, title=None):
    """Recessive axes and grid; text in ink tokens, never in a series color."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    return ax


def footer(fig, n, extra=None, ci="95% bootstrap CI, 2000 resamples", ci_prefix="band"):
    """State n and the interval method on the figure. Not optional: a band with
    an unstated method is not a reportable number."""
    bits = [f"n = {n}", f"{ci_prefix} = {ci}"]
    if extra:
        bits.append(extra)
    fig.text(0.01, 0.01, "  ·  ".join(bits), color=MUTED, fontsize=8, ha="left")


def bootstrap_ci(samples, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. -> (lo, mean, hi)."""
    a = np.asarray(samples, dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = a[rng.randint(0, a.size, (n_boot, a.size))].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(a.mean()),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def layer_sweep(series, depths, title, ylabel="Causal effect", reference=0.0,
                reference_label="no effect", ax=None, n=None, ci_label=None):
    """Several series over depth, with bootstrap CI bands and direct labels.

    series: {label: [[v, ...] per layer]} -- per-layer lists of per-item values,
            so the band is computed here; or {label: [scalar per layer]} for a
            line with no band.
    depths: x values (layer index or % depth), one per layer.
    """
    plt = _plt()
    if len(series) > 3:
        raise ValueError(f"{len(series)} series: cap is 3. Facet into small "
                         "multiples or fold the tail into 'other'.")
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))

    if reference is not None:
        ax.axhline(reference, color=CONTEXT, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(reference_label, (depths[0], reference), xytext=(0, 5),
                    textcoords="offset points", color=CONTEXT, fontsize=8, va="bottom")

    for i, (label, vals) in enumerate(series.items()):
        color = SERIES[i]
        has_band = np.ndim(vals[0]) > 0
        if has_band:
            stats = [bootstrap_ci(v) for v in vals]
            lo, mid, hi = (np.array([s[j] for s in stats]) for j in range(3))
            ax.fill_between(depths, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        else:
            mid = np.asarray(vals, dtype=float)
        ax.plot(depths, mid, color=color, linewidth=2, marker="o", markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=3)
        # Direct label: required here, since one slot is below 3:1 contrast.
        # Nudge clear of the reference line rather than printing on top of it.
        dy = 0
        if reference is not None and abs(mid[-1] - reference) < 0.04 * (
                np.nanmax(mid) - np.nanmin(mid) + 1e-9):
            dy = 10
        ax.annotate(label, (depths[-1], mid[-1]), xytext=(6, dy),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="bold")

    style(ax, "Layer depth (% of network)", ylabel, title)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="best")
    ax.set_xlim(min(depths) - 2, max(depths) + 14)
    if fig is not None:
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        if n is not None:
            footer(fig, n, ci=ci_label or "95% bootstrap CI, 2000 resamples")
    return ax


def probe_panel(depths, scores, shuffled, chance, title, n=None,
                ylabel="Held-out accuracy", leak=None, ax=None):
    """Emphasis form: the probe is the point, its controls are context.

    The controls are deliberately gray, not extra categorical colors -- they are
    not peer series, and coloring them as such invites reading the null as a
    finding. A probe curve shown without them is not interpretable.
    """
    plt = _plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))

    ax.axhline(chance, color=CONTEXT, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"chance ({chance:.2f})", (depths[0], chance), xytext=(0, 4),
                textcoords="offset points", color=CONTEXT, fontsize=8, va="bottom")
    ax.plot(depths, shuffled, color=CONTEXT, linewidth=1.5, linestyle=":",
            label="shuffled-label null", zorder=2)
    if leak is not None:
        ax.plot(depths, leak, color=CONTEXT, linewidth=1.5, linestyle="-.",
                alpha=0.75, label="shortcut-leak control", zorder=2)

    has_band = np.ndim(scores[0]) > 0
    if has_band:
        stats = [bootstrap_ci(v) for v in scores]
        lo, mid, hi = (np.array([s[j] for s in stats]) for j in range(3))
        ax.fill_between(depths, lo, hi, color=SERIES[0], alpha=0.18, linewidth=0, zorder=3)
    else:
        mid = np.asarray(scores, dtype=float)
    ax.plot(depths, mid, color=SERIES[0], linewidth=2, marker="o", markersize=4,
            markeredgecolor="white", markeredgewidth=0.5, label="probe", zorder=4)
    ax.annotate("probe", (depths[-1], mid[-1]), xytext=(6, 0), textcoords="offset points",
                color=SERIES[0], fontsize=9, va="center", fontweight="bold")

    style(ax, "Layer depth (% of network)", ylabel, title)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="best")
    ax.set_xlim(min(depths) - 2, max(depths) + 12)
    ax.set_ylim(-0.03, 1.03)
    if fig is not None:
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        if n is not None:
            footer(fig, n)
    return ax


def effect_grid(matrix, row_labels, depths, title, cbar_label="Causal effect",
                diverging=False, n=None):
    """position x layer heatmap. Sequential (one hue, light->dark) by default;
    diverging (warm/cool poles, neutral gray midpoint) when the value is signed
    and zero is meaningful. Never a rainbow."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    M = np.asarray(matrix, dtype=float)

    if diverging:
        cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#eeeeea", SEQ_HUE])
        norm = TwoSlopeNorm(vmin=min(M.min(), -1e-6), vcenter=0, vmax=max(M.max(), 1e-6))
    else:
        cmap = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", SEQ_HUE, "#14335c"])
        norm = None

    fig, ax = plt.subplots(figsize=(8.5, 1.1 + 0.55 * len(row_labels)))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   extent=(depths[0], depths[-1], len(row_labels) - 0.5, -0.5))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, color=INK)
    style(ax, "Layer depth (% of network)", "", title)
    ax.grid(False)          # after style(), which turns it on
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label, color=MUTED, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if n is not None:
        footer(fig, n, ci="mean over items", ci_prefix="color")
    return fig, ax


def save(fig, path, dpi=200):
    """Presentation default: 200 dpi on an off-white surface."""
    fig.savefig(path, dpi=dpi, facecolor="#fcfcfb", bbox_inches="tight")
    return path
