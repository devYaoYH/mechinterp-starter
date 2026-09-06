"""Presentation-ready figures.

    layer_sweep()   several series over depth        -> multi-line + CI band
    probe_panel()   one result against its controls  -> emphasis (accent + gray)
    effect_grid()   position x layer                 -> sequential heatmap

Each draws its reference line (chance, or zero effect), a bootstrap CI band, and
a footer stating n and the interval method -- a sweep without a baseline is
ambiguous to read, and one without a band invites reading noise as structure.

Colors are the first three slots of a CVD-validated palette (all-pairs dE 9.2
deutan / 24.0 normal, light surface), assigned in order, never cycled. Series
are direct-labeled as well as legended because one slot is below 3:1 contrast.
No fourth series: facet, or fold the tail into "other".
"""
import os

import numpy as np

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, CONTEXT, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e2e1db"
DASH = (0, (4, 3))


def _plt():
    import matplotlib
    if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def bootstrap_ci(samples, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean -> (lo, mean, hi)."""
    a = np.asarray(samples, float)
    if not a.size:
        return (np.nan,) * 3
    m = a[np.random.RandomState(seed).randint(0, a.size, (n_boot, a.size))].mean(1)
    return (float(np.percentile(m, 100 * alpha / 2)), float(a.mean()),
            float(np.percentile(m, 100 * (1 - alpha / 2))))


def style(ax, xlabel, ylabel, title=None):
    """Recessive axes and grid; text in ink tokens, never a series color."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    return ax


def footer(fig, n, ci="95% bootstrap CI, 2000 resamples", prefix="band"):
    """n and the interval method. A band with an unstated method is not a
    reportable number."""
    fig.text(0.01, 0.01, f"n = {n}  ·  {prefix} = {ci}", color=MUTED, fontsize=8)


def _series(ax, x, vals, color, label, ms=3.5):
    """Draw one series. `vals` is per-x lists (band drawn) or per-x scalars."""
    if np.ndim(vals[0]) > 0:
        lo, mid, hi = (np.array(v) for v in zip(*[bootstrap_ci(v) for v in vals]))
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
    else:
        mid = np.asarray(vals, float)
    ax.plot(x, mid, color=color, linewidth=2, marker="o", markersize=ms,
            markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=3)
    return mid


def _reference(ax, x0, value, label):
    ax.axhline(value, color=CONTEXT, linewidth=1, linestyle=DASH, zorder=1)
    ax.annotate(label, (x0, value), xytext=(0, 5), textcoords="offset points",
                color=CONTEXT, fontsize=8, va="bottom")


def _direct_label(ax, x, y, text, color, avoid=None, span=1.0):
    """Label at the line end, nudged clear of the reference line."""
    dy = 10 if (avoid is not None and abs(y - avoid) < 0.04 * span) else 0
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                color=color, fontsize=9, va="center", fontweight="bold")


def _finish(fig, ax, depths, xpad, n, ci=None):
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="best")
    ax.set_xlim(min(depths) - 2, max(depths) + xpad)
    if fig is not None:
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        if n is not None:
            footer(fig, n, **({"ci": ci} if ci else {}))
    return ax


def layer_sweep(series, depths, title, ylabel="Causal effect", reference=0.0,
                reference_label="no effect", ax=None, n=None, ci_label=None):
    """series: {label: per-layer lists (banded) or per-layer scalars}."""
    if len(series) > 3:
        raise ValueError(f"{len(series)} series: cap is 3. Facet, or fold into 'other'.")
    fig, ax = (None, ax) if ax is not None else _plt().subplots(figsize=(7.5, 4.6))
    if reference is not None:
        _reference(ax, depths[0], reference, reference_label)
    for i, (label, vals) in enumerate(series.items()):
        mid = _series(ax, depths, vals, SERIES[i], label)
        _direct_label(ax, depths[-1], mid[-1], label, SERIES[i], reference,
                      np.nanmax(mid) - np.nanmin(mid) + 1e-9)
    style(ax, "Layer depth (% of network)", ylabel, title)
    return _finish(fig, ax, depths, 14, n, ci_label)


def probe_panel(depths, scores, shuffled, chance, title, n=None,
                ylabel="Held-out accuracy", leak=None, transfer=None, ax=None):
    """Emphasis form: the probe is the point, its controls are context.

    Controls are gray, not categorical colors -- they are not peer series, and
    coloring them as peers invites reading the null as a finding.
    """
    fig, ax = (None, ax) if ax is not None else _plt().subplots(figsize=(7.5, 4.6))
    _reference(ax, depths[0], chance, f"chance ({chance:.2f})")
    for vals, ls, lab in ((shuffled, ":", "shuffled-label null"),
                          (leak, "-.", "shortcut-leak control"),
                          (transfer, (0, (1, 1)), "cross-domain transfer")):
        if vals is not None:
            ax.plot(depths, vals, color=CONTEXT, linewidth=1.5, linestyle=ls,
                    label=lab, zorder=2)
    mid = _series(ax, depths, scores, SERIES[0], "probe", ms=4)
    _direct_label(ax, depths[-1], mid[-1], "probe", SERIES[0])
    style(ax, "Layer depth (% of network)", ylabel, title)
    ax.set_ylim(-0.03, 1.03)
    return _finish(fig, ax, depths, 12, n)


def effect_grid(matrix, row_labels, depths, title, cbar_label="Causal effect",
                diverging=False, n=None):
    """Sequential (one hue, light->dark) by default; diverging (warm/cool poles,
    neutral midpoint) when the value is signed and zero means something."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    M, plt = np.asarray(matrix, float), _plt()
    if diverging:
        cmap = LinearSegmentedColormap.from_list("d", [SERIES[1], "#eeeeea", SERIES[0]])
        norm = TwoSlopeNorm(min(M.min(), -1e-6), 0, max(M.max(), 1e-6))
    else:
        cmap = LinearSegmentedColormap.from_list("s", ["#f4f7fc", SERIES[0], "#14335c"])
        norm = None
    fig, ax = plt.subplots(figsize=(8.5, 1.1 + 0.55 * len(row_labels)))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   extent=(depths[0], depths[-1], len(row_labels) - 0.5, -0.5))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, color=INK)
    style(ax, "Layer depth (% of network)", "", title)
    ax.grid(False)                      # after style(), which turns it on
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label, color=MUTED, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if n is not None:
        footer(fig, n, ci="mean over items", prefix="color")
    return fig, ax


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, facecolor="#fcfcfb", bbox_inches="tight")
    return path
