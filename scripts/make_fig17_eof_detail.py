#!/usr/bin/env python3
"""
================================================================================
 FIGURE 17 — ONE FIGURE PER EOF: WHERE THE MODE ACTS, AND WHEN
================================================================================

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig17_eof_detail.py

WHAT THIS ADDS OVER FIGURE 16
-----------------------------
Figure 14 puts every mode on one page and shows only the MAPS. A map answers
"where" and nothing else — it cannot say when the mode acts, or how much of the
emission change in a given decade it accounts for. That is the SCORE, the PC
time series, and it is not stored in the checkpoint: it is recomputed by
projecting the conditioning data onto the persisted basis
(scripts/dump_cond_scores.py, which verifies itself against sklearn's
`explained_variance_` and refuses to write a projection that disagrees).

So each mode gets its own page: the map on the left, the score and its decadal
means on the right, for all three species at once. Read across a row and you
get "this emission, this pattern, acting this hard in these decades".

READ THE DECADE BARS AS THE ANSWER
----------------------------------
The bars are the decade-mean score. What matters for "which emission change
drives the variability" is how far the bar MOVES between decades, not its
height: a mode with a large but constant score is a constant offset and forces
nothing. The per-decade change is printed in the panel.

PROVENANCE, AND IT MATTERS
--------------------------
The scores are projected from `emissions_ssp370_only_timefixed_bc.nc` — the
PRE-co2fix file. That is not a mistake: the checkpoint's persisted basis matches
that file's CO2 to 2.4% and the _co2fix file only to 33%, so the basis was
fitted before the CO2 correction. Projecting the fixed file through this basis
would be the inconsistent choice. It does mean the CO2 trajectory here is the
doubled one.
"""

# =============================================================================
#  SETTINGS
# =============================================================================

EOF_FILE = "plots/cond_eofs_ep0863.npz"
SCORE_FILE = "plots/cond_scores_ssp370_ep0863.npz"

OUT = "plots/fig17/fig17_eof{mode}.png"     # the .pdf sibling is written alongside
TABLE = "plots/fig17/fig17_decades.tex"

BASIS = "ssp370"
BASIS_YEARS = "2015-2100"

# One figure per mode in this list.
MODES = [1, 2, 3]

# Species order down the rows, with the colour used for the score line.
SPECIES = {"CO2": "#0072B2", "SUL": "#E69F00", "BC": "#009E73"}

# Decade edges for the bars. The last is 2091-2100.
DECADE_START = 2020
DECADE_END = 2100

# =============================================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
except ImportError:
    sys.exit("[error] cartopy missing — use the plotting env")

for path in (EOF_FILE, SCORE_FILE):
    if not os.path.exists(path):
        sys.exit(f"[error] {path} not found — build it on LUMI first "
                 f"(run_dump_eofs.sh / run_dump_scores.sh) and scp it into plots/")

eofs = np.load(EOF_FILE)
scores_store = np.load(SCORE_FILE)
years = scores_store["years"].astype(int)

# =============================================================================
#  STEP 1 — orient each mode, and carry the SAME flip into its score
# =============================================================================
# A principal component is fixed only up to sign. Figure 14 flips each map so
# its area-weighted mean is positive; the same flip must be applied to the
# SCORE, or the pair (map, score) describes the opposite of what it plots.
# Flipping both leaves their product — the actual field contribution —
# unchanged, which is the point.

maps, scores, ratio = {}, {}, {}
for species in SPECIES:
    component = eofs[f"components_{BASIS}_{species}"]
    latitude = np.linspace(-90, 90, component.shape[1])
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None],
                              component.shape[1:])
    raw_scores = scores_store[f"scores_{species}"]
    ratio[species] = scores_store[f"ratio_{species}"]

    oriented_maps, oriented_scores = [], []
    for k in range(max(MODES)):
        sign = -1.0 if np.average(component[k], weights=weights) < 0 else 1.0
        oriented_maps.append(sign * component[k])
        oriented_scores.append(sign * raw_scores[:, k])
    maps[species] = np.array(oriented_maps)
    scores[species] = np.array(oriented_scores)
    print(f"[step 1] {species}: maps {maps[species].shape}, "
          f"scores {scores[species].shape}, "
          f"variance {np.round(100 * ratio[species][:3], 1)}")

# =============================================================================
#  STEP 2 — decade means, and the change between decades
# =============================================================================
# The bar heights are decade means; the ANSWER to "which decade did this mode
# change the emissions" is the difference between consecutive bars.

edges = list(range(DECADE_START, DECADE_END + 1, 10))
decade_labels = [f"{e}s" for e in edges[:-1]]
decade_mean = {}
for species in SPECIES:
    table = []
    for k in range(max(MODES)):
        row = []
        for start, stop in zip(edges[:-1], edges[1:]):
            mask = (years >= start) & (years < stop)
            row.append(float(scores[species][k][mask].mean()) if mask.any()
                       else np.nan)
        table.append(row)
    decade_mean[species] = np.array(table)

# =============================================================================
#  STEP 3 — one figure per mode
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

os.makedirs("plots/fig17", exist_ok=True)
summary_rows = []

for mode in MODES:
    fig = plt.figure(figsize=(11.4, 2.5 * len(SPECIES) + 1.0))
    # wspace has to clear TWO labels facing each other across the gutter: the
    # map's colour-bar label on the right of column 0, and the time series'
    # y-label on the left of column 1.
    grid = fig.add_gridspec(len(SPECIES), 2, width_ratios=[1.15, 1.0],
                            hspace=0.42, wspace=0.34)

    for row, (species, colour) in enumerate(SPECIES.items()):
        k = mode - 1

        # ---- left: the pattern -------------------------------------------
        ax_map = fig.add_subplot(grid[row, 0],
                                 projection=ccrs.Robinson(central_longitude=0))
        field = maps[species][k]
        scale = float(np.nanpercentile(np.abs(field), 99.5))
        image = ax_map.pcolormesh(np.linspace(0, 360, field.shape[1]),
                                  np.linspace(-90, 90, field.shape[0]),
                                  field, cmap="RdBu_r", vmin=-scale, vmax=scale,
                                  shading="auto", transform=ccrs.PlateCarree())
        ax_map.coastlines(linewidth=0.35, color="0.25")
        ax_map.set_global()
        ax_map.set_title(f"{species} EOF{mode} — "
                         f"{100 * ratio[species][k]:.1f}% of variance",
                         fontsize=9.5, loc="left", pad=4)
        bar = fig.colorbar(image, ax=ax_map, orientation="vertical",
                           fraction=0.026, pad=0.02, extend="both")
        bar.ax.tick_params(labelsize=8)

        # ---- right: when it acts -----------------------------------------
        ax_time = fig.add_subplot(grid[row, 1])
        centres = [e + 5 for e in edges[:-1]]
        bars = decade_mean[species][k]
        ax_time.bar(centres, bars, width=8.5, color=colour, alpha=0.35,
                    edgecolor=colour, linewidth=0.6, zorder=2)
        ax_time.plot(years, scores[species][k], color=colour, lw=1.8, zorder=3)
        ax_time.axhline(0, lw=0.8, color="0.3", zorder=1)
        ax_time.set_xlim(years.min() - 1, years.max() + 1)
        ax_time.set_ylabel("PC score")
        if row == len(SPECIES) - 1:
            ax_time.set_xlabel("Year")
        else:
            ax_time.tick_params(labelbottom=False)

        # The decade-to-decade change is what forces anything; the level does
        # not. Report the largest one on the panel.
        steps = np.diff(bars)
        if np.isfinite(steps).any():
            j = int(np.nanargmax(np.abs(steps)))
            ax_time.set_title(
                f"total swing {np.nanmax(bars) - np.nanmin(bars):+.1f}; "
                f"biggest decade step {steps[j]:+.1f} "
                f"({decade_labels[j]}→{decade_labels[j+1]})",
                fontsize=8.8, loc="left", pad=4)
            summary_rows.append(
                (species, mode, 100 * ratio[species][k],
                 float(np.nanmax(bars) - np.nanmin(bars)),
                 float(steps[j]), f"{decade_labels[j]}-{decade_labels[j+1]}"))

    fig.suptitle(f"EOF{mode}: where the mode acts, and when — "
                 f"'{BASIS}' basis, {BASIS_YEARS}", fontsize=11, y=0.98)
    out_path = OUT.format(mode=mode)
    for path in (out_path, os.path.splitext(out_path)[0] + ".pdf"):
        fig.savefig(path, bbox_inches="tight")
        print(f"[step 3] wrote {path}")
    plt.close(fig)

# =============================================================================
#  STEP 4 — the decade table
# =============================================================================

rows_tex = [f"{sp} EOF{m} & {var:.1f} & {swing:+.1f} & {step:+.1f} & {when} \\\\"
            for sp, m, var, swing, step, when in summary_rows]
caption = (
    f"How each conditioning mode changes over the {BASIS_YEARS} record of the "
    f"``{BASIS}'' basis. ``Variance'' is the share of that channel's temporal "
    f"variance the mode carries. ``Swing'' is the range of its decade-mean "
    f"score and ``biggest step'' the largest change between consecutive "
    f"decades, with the decade pair in which it happens. IT IS THE STEP, NOT "
    f"THE LEVEL, THAT FORCES ANYTHING: a mode with a large but constant score "
    f"is a fixed offset in the emissions and drives no change. Scores are "
    f"projections of the conditioning data onto the checkpoint's persisted "
    f"basis, verified against the stored explained variances to within 4\\%.")
table_tex = "\n".join([
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\caption{" + caption + "}", r"\label{tab:fig17}",
    r"\begin{tabular}{|l|r|r|r|l|}", r"\hline",
    r"\textbf{Mode} & \textbf{Variance (\%)} & \textbf{Swing} & "
    r"\textbf{Biggest step} & \textbf{When} \\", r"\hline",
    *rows_tex, r"\hline", r"\end{tabular}", r"\end{table}"])
with open(TABLE, "w") as handle:
    handle.write(f"% Conditioning modes over decades, '{BASIS}' basis.\n"
                 f"% Built by scripts/make_fig17_eof_detail.py — do not edit.\n")
    handle.write(table_tex + "\n")
print(f"[step 4] wrote {TABLE}")
for r in summary_rows:
    print(f"[step 4] {r[0]:4s} EOF{r[1]}  var {r[2]:5.1f}%  swing {r[3]:+8.1f}  "
          f"biggest step {r[4]:+8.1f} at {r[5]}")
