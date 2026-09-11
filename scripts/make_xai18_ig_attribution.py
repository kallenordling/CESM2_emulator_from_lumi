#!/usr/bin/env python3
"""
================================================================================
 FIGURE 20 — WHICH CONDITIONING MODES THE ATTRIBUTION ASSIGNS N TO
================================================================================

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig18_ig_attribution.py

Renders the output of scripts/ig_nonlinear.py: Integrated Gradients on
N = dALL - dGHG - dAAER, decomposed exactly onto the persisted conditioning
modes, for each output region.

READ THE CAVEAT BEFORE THE FIGURE
---------------------------------
The attributed quantity is NOT the emulator's physical nonlinearity. It is a
single denoising step at fixed noise, reduced to a regional mean in NORMALISED
model space, for one year and one seed. The physical N is a decade mean of the
full 50-step sampler in degC.

The two disagree in sign: this proxy gives a global N of -0.032, while the
physical N at 2041-2050 is +0.637 K. So these bars describe what the DENOISER
is sensitive to, and whether that tracks the emulator's actual nonlinearity is
an open question that a correlation test against the per-year N fields will
settle. Until then the figure is titled and captioned as a proxy, and the
disagreement is printed on the figure itself rather than buried in a caption.

scripts/ig_verify.py established that the IG implementation is correct FOR THIS
TARGET (completeness 0.01% at 32 steps). It did not establish that the target
is the right one.
"""

# =============================================================================
#  SETTINGS
# =============================================================================

RESULT = "plots/ig_nonlinear_2040.npz"
OUT = "plots/xai18/xai18.png"          # the .pdf sibling is written alongside

# Modes to display. CO2 carries 30 components and the aerosols 5, but the tail
# is negligible everywhere; showing the first five of each keeps the columns
# readable and the omitted ones are reported as a residual so nothing is hidden.
N_MODES_SHOWN = 5
SPECIES = ["CO2", "SUL", "BC"]

# The physical value the proxy is being checked against — from figs 09/10,
# CESM2-side emulator N at 2041-2050. Printed on the figure as the discrepancy.
PHYSICAL_N = 0.637
PHYSICAL_LABEL = "emulator N at 2041–2050 = +0.637 K (figs 09/11)"

# =============================================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

if not os.path.exists(RESULT):
    sys.exit(f"[error] {RESULT} not found — run scripts/ig_nonlinear.py on LUMI "
             f"and scp the .npz into plots/")

store = np.load(RESULT, allow_pickle=True)
regions = [str(r) for r in store["regions"]]
mode_names = [str(m) for m in store["mode_names"]]
modes = store["modes"]                      # (region, mode)
n_values = store["N"]
errors = store["error"]
year = int(store["year"])

# =============================================================================
#  STEP 1 — select the modes to show, and keep the rest as an honest residual
# =============================================================================

shown, labels = [], []
for species in SPECIES:
    for k in range(1, N_MODES_SHOWN + 1):
        name = f"{species}_EOF{k}"
        if name in mode_names:
            shown.append(mode_names.index(name))
            labels.append(f"{species}\nEOF{k}")
shown = np.array(shown)
residual = modes.sum(axis=1) - modes[:, shown].sum(axis=1)
print(f"[step 1] showing {len(shown)} of {len(mode_names)} modes; "
      f"omitted tail carries {np.abs(residual).max():.4f} at most")

matrix = modes[:, shown]

# =============================================================================
#  STEP 2 — the figure
# =============================================================================
# Two panels: the full region x mode matrix, and the per-region totals with the
# completeness error attached, because a region whose N is near zero has a
# meaningless relative error and must be visibly flagged.

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(13.0, 6.4))
grid = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.30)

# ---- panel (a): the attribution matrix -------------------------------------
ax = fig.add_subplot(grid[0, 0])
limit = float(np.abs(matrix).max())
image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                  norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_yticks(range(len(regions)))
ax.set_yticklabels(regions, fontsize=9.5)
# Separators between species blocks — the grouping is real structure.
for boundary in range(N_MODES_SHOWN, len(labels), N_MODES_SHOWN):
    ax.axvline(boundary - 0.5, color="0.25", lw=1.2)
for i in range(len(regions)):
    for j in range(len(labels)):
        value = matrix[i, j]
        if abs(value) > 0.4 * limit:
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.7 * limit else "0.15")
ax.set_title("(a) Attribution of N to conditioning modes", fontsize=11,
             loc="left", pad=8)
# HORIZONTAL, under panel (a). A vertical bar sits in the gutter between the
# panels and its label lands on top of panel (b)'s region names.
fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13,
             aspect=45, label="contribution to N (normalised model units)")

# ---- panel (b): totals, with the completeness check visible ----------------
ax2 = fig.add_subplot(grid[0, 1])
order = np.arange(len(regions))
colours = ["#A94436" if v < 0 else "#2F5D7C" for v in n_values]
ax2.barh(order, n_values, color=colours, alpha=0.85)
ax2.set_yticks(order)
ax2.set_yticklabels(regions, fontsize=9.5)
ax2.invert_yaxis()
ax2.axvline(0, color="0.3", lw=0.9)
ax2.set_xlabel("N (normalised model units)")
ax2.set_title("(b) N per region", fontsize=11, loc="left", pad=8)
for i, (value, error) in enumerate(zip(n_values, errors)):
    # Flag regions where the decomposition does not close. These are always
    # regions whose N is near zero, so it is the SMALL DENOMINATOR, not a broken
    # split — but either way the row must not be read.
    if error > 2.0:
        ax2.text(value, i, f"  completeness {error:.0f}% — do not read",
                 va="center", fontsize=7.5, color="#A94436",
                 ha="left" if value >= 0 else "right")

fig.suptitle(
    f"Integrated Gradients on N = ALL $-$ GHG $-$ AAER, year {year}, "
    f"temperature channel\n"
    f"PROXY TARGET: one denoising step in normalised units — global N here is "
    f"{n_values[0]:+.3f}, against {PHYSICAL_LABEL}. Not yet validated as the "
    f"physical nonlinearity.",
    fontsize=10.5, y=1.005)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
for path in (OUT, os.path.splitext(OUT)[0] + ".pdf"):
    fig.savefig(path, bbox_inches="tight")
    print(f"[step 2] wrote {path}")
plt.close(fig)

# =============================================================================
#  STEP 3 — the same numbers as text, ranked
# =============================================================================

print("\n[step 3] leading modes per region (proxy units):")
for i, region in enumerate(regions):
    top = np.argsort(-np.abs(matrix[i]))[:4]
    flag = "  [completeness poor]" if errors[i] > 2.0 else ""
    print(f"  {region:12s} N {n_values[i]:+.5f}{flag}")
    print("               " + ",  ".join(
        f"{labels[j].replace(chr(10), ' ')} {matrix[i, j]:+.4f}" for j in top))
