---
name: report-figure
description: Build a presentation-ready scientific figure from experiment results, choosing the form from what the figure needs to communicate. Use when the user wants to plot results, make a chart or figure for a talk or paper, visualize a layer sweep or probe curve, or says /report-figure.
---

# Build a report figure

The most common way a figure fails is that it was drawn before anyone decided
what it should say. Ask first.

## Step 1 — ask what it must communicate

Use `AskUserQuestion` before writing any plotting code:

1. **The takeaway, as one sentence.** "What should someone conclude from this
   figure?" If the answer is "here is the data", push back once — a figure with
   no claim becomes a table.
2. **The data's job**, which picks the form:
   - distinct series compared over depth → `layer_sweep` (multi-line)
   - one result against its controls → `probe_panel` (emphasis: accent + gray)
   - a position × layer grid → `effect_grid` (sequential heatmap)
   - one number → a stat line in the text, not a chart
3. **Audience and medium** — slide, paper, notebook. Slides want fewer series
   and a larger base font; papers want the CI method spelled out.
4. **Where the numbers are** — a CSV, a JSON, or in-memory results, and whether
   per-item values exist (needed for a CI band).

## Step 2 — plot

```python
from mechinterp import plotting
ax = plotting.layer_sweep(series, depths, title, n=n)     # per-layer lists -> CI bands
ax = plotting.probe_panel(depths, per_item, shuffled, chance, title, n=n, leak=leak)
fig, _ = plotting.effect_grid(matrix, row_labels, depths, title, n=n)
plotting.save(ax.figure, "figures/name.png")
```

The module already enforces the reporting rules, so do not hand-roll around it:
a reference line (chance, or zero effect), a bootstrap CI band, and a footer
stating `n` and the interval method.

Rules it will not let you break, and should not be worked around:

- **Three series maximum.** A fourth is not a new hue — facet into small
  multiples or fold the tail into "other". `layer_sweep` raises on a fourth.
- **One y-axis, ever.** Two measures of different scale → two charts or index
  both to a common base. A dual-axis chart invents a correlation that is not in
  the data.
- **Controls are gray, not categorical colors.** The shuffled null and leak
  control are context, not peer series; coloring them as peers invites reading
  the null as a finding.
- Colors are the first three slots of a CVD-validated palette, assigned in
  order, never cycled. Series are direct-labeled as well as legended, because
  one slot sits below 3:1 contrast on the light surface.

## Step 3 — render it and look at it

Save the PNG and **read it back** with the Read tool. The palette validator
checks color, not layout. Look for: labels colliding with each other or with the
reference line, legend sitting on top of the data, axis overflow, and a title
that states the takeaway rather than naming the variables.

Fix what you see, re-render, and only then show the user.

## Step 4 — caption

Offer a one-line caption carrying the takeaway, `n`, and the interval method —
the three things a reader needs and a figure usually omits.
