#!/usr/bin/env python3
"""
================================================================================
 FIGURE 22 — WHICH EMISSIONS, FROM WHERE, DRIVE N IN ONE OUTPUT REGION
================================================================================

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig20_source_attribution.py \
        --output-region "N. Atlantic"

THE QUESTION THIS ANSWERS
-------------------------
"For the North Atlantic, what emissions and from where contribute most to the
nonlinear term?" — as a ranked, quantified decomposition rather than a map to
eyeball.

The IG attribution for one output region is a per-cell field over the whole
input grid that sums EXACTLY to N for that region. So partitioning the input
grid into source regions and summing gives a complete accounting:

    N(N. Atlantic) = sum over (species x source region) of their contributions

Nothing is left over: the source boxes are supplemented by an explicit
"elsewhere" term so the parts add back to the whole, and the reconstruction is
checked and printed.

WHY THIS IS LEGITIMATE WHERE A REGIONAL PERTURBATION IS NOT
-----------------------------------------------------------
A regional emission PERTURBATION is not representable by this model: the
conditioning bottleneck destroys ~91-95% of it and scatters the remainder
outside the region perturbed. That is a statement about counterfactuals — about
inputs the pipeline cannot deliver.

This is not a counterfactual. It partitions a change that ACTUALLY HAPPENED —
the 1850 to 2040 conditioning displacement, already projected through the PCA
exactly as the model received it — into the contributions of its parts. Slicing
an existing displacement by geography is always well defined; manufacturing a
new one confined to a box is not. Keep the distinction in the wording: this says
"the sulfate change that occurred over Europe accounted for X of N", never "if
Europe's sulfate changed, N would move by X".
"""

# =============================================================================
#  SETTINGS
# =============================================================================

RESULT = "plots/ig_nonlinear_2040_maps.npz"
OUT = "plots/xai20/xai20_{region}.png"

# Source regions to attribute to. They need not tile the globe — whatever falls
# outside them is reported as "Elsewhere" so the decomposition stays complete.
# Longitudes in the files' own 0-360 convention.
SOURCE_REGIONS = [
    ("East Asia",    20.0,  50.0, 100.0, 145.0),
    ("South Asia",    5.0,  30.0,  65.0, 100.0),
    ("Europe",       35.0,  65.0,   0.0,  40.0),
    ("N. America",   30.0,  60.0, 230.0, 300.0),
    ("Africa",      -35.0,  35.0,   0.0,  50.0),
    ("S. America",  -55.0,  12.0, 280.0, 325.0),
    ("Russia/N Asia",50.0,  75.0,  40.0, 180.0),
    ("Middle East",  12.0,  40.0,  35.0,  65.0),
]

SPECIES_COLOUR = {"CO2": "#0072B2", "SUL": "#E69F00", "BC": "#009E73"}

# =============================================================================

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import cartopy.crs as ccrs
except ImportError:
    sys.exit("[error] cartopy missing — use the plotting env")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-region", default="N. Atlantic")
parser.add_argument("--result", default=RESULT)
parser.add_argument("--top", type=int, default=14,
                    help="how many (species x source) terms to show")
args = parser.parse_args()

if not os.path.exists(args.result):
    sys.exit(f"[error] {args.result} not found — run scripts/ig_nonlinear.py on "
             f"LUMI with the map-saving version and scp it into plots/")

store = np.load(args.result, allow_pickle=True)
regions = [str(r) for r in store["regions"]]
cond_vars = [str(v) for v in store["cond_vars"]]
lat, lon = store["lat"], store["lon"]
year = int(store["year"])
if "attr_maps" not in store.files:
    sys.exit(f"[error] {args.result} has no attr_maps — it was written by the "
             f"older version of ig_nonlinear.py. Rerun it.")
if args.output_region not in regions:
    sys.exit(f"[error] '{args.output_region}' not in {regions}")

index = regions.index(args.output_region)
attribution = store["attr_maps"][index]            # (channel, lat, lon)
n_value = float(store["N"][index])
completeness = float(store["error"][index])

# =============================================================================
#  STEP 1 — partition the input grid, keeping the remainder explicit
# =============================================================================

lon_grid, lat_grid = np.meshgrid(lon, lat)
assigned = np.zeros(lat_grid.shape, dtype=bool)
terms = []
for name, lat_min, lat_max, lon_min, lon_max in SOURCE_REGIONS:
    box = ((lat_grid >= lat_min) & (lat_grid <= lat_max)
           & (lon_grid >= lon_min) & (lon_grid <= lon_max))
    # Boxes may overlap (Middle East against Africa/Asia); first claim wins so
    # every cell is counted exactly once and the total is conserved.
    box = box & ~assigned
    assigned |= box
    for c, species in enumerate(cond_vars):
        terms.append((species, name, float(attribution[c][box].sum())))
for c, species in enumerate(cond_vars):
    terms.append((species, "Elsewhere", float(attribution[c][~assigned].sum())))

total = sum(v for _, _, v in terms)
gap = 100 * abs(total - n_value) / max(abs(n_value), 1e-12)
print(f"[step 1] {args.output_region}: N = {n_value:+.5f}, "
      f"sum of parts = {total:+.5f}  (reconstruction gap {gap:.3f}%)")
if gap > 1.0:
    print("[step 1] WARNING: the parts do not add back to N — do not interpret")

terms.sort(key=lambda t: -abs(t[2]))
print(f"\n[step 1] ranked contributions to N over {args.output_region}, "
      f"year {year}:")
for species, source, value in terms[:args.top]:
    print(f"   {species:4s} from {source:14s} {value:+.5f}   "
          f"({100 * value / n_value:+6.1f}% of N)")

# =============================================================================
#  STEP 2 — the figure
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(12.8, 7.4))
grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

# ---- (a) ranked bars -------------------------------------------------------
ax = fig.add_subplot(grid[0, 0])
shown = terms[:args.top][::-1]
positions = np.arange(len(shown))
values = [v for _, _, v in shown]
colours = [SPECIES_COLOUR.get(s, "0.5") for s, _, _ in shown]
ax.barh(positions, values, color=colours, alpha=0.9)
ax.set_yticks(positions)
ax.set_yticklabels([f"{s} · {r}" for s, r, _ in shown], fontsize=9)
ax.axvline(0, color="0.3", lw=0.9)
ax.set_xlabel("contribution to N (normalised model units)")
ax.set_title(f"(a) What drives N over the {args.output_region}", fontsize=11,
             loc="left", pad=8)
ax.legend(handles=[Patch(facecolor=c, label=s)
                   for s, c in SPECIES_COLOUR.items() if s in cond_vars],
          frameon=False, ncols=3, loc="lower left",
          bbox_to_anchor=(0.0, 1.005), fontsize=9)
ax.annotate(f"N = {n_value:+.4f}   ·   parts sum to {total:+.4f} "
            f"(gap {gap:.2f}%)",
            xy=(0.98, 0.02), xycoords="axes fraction", fontsize=8.5,
            color="0.35", ha="right")

# ---- (b) where the attribution sits, PER SPECIES ---------------------------
# NOT summed over species. Summing |attribution| across CO2, SUL and BC collapses
# exactly the distinction the question asks about — which emission, from where —
# and it hides sign cancellation between species at the same location. One map
# each, on a SHARED signed scale so the three are directly comparable in
# magnitude as well as in pattern.
species_grid = grid[0, 1].subgridspec(len(cond_vars), 1, hspace=0.16)
scale = float(np.nanpercentile(np.abs(attribution), 99.5))
species_axes = []
for c, species in enumerate(cond_vars):
    ax2 = fig.add_subplot(species_grid[c, 0],
                          projection=ccrs.Robinson(central_longitude=0))
    image = ax2.pcolormesh(lon, lat, attribution[c], cmap="RdBu_r",
                           vmin=-scale, vmax=scale, shading="auto",
                           transform=ccrs.PlateCarree())
    ax2.coastlines(linewidth=0.3, color="0.3")
    ax2.set_global()
    for name, lat_min, lat_max, lon_min, lon_max in SOURCE_REGIONS:
        ax2.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                 [lat_min, lat_min, lat_max, lat_max, lat_min],
                 color="#2F5D7C", lw=0.7, transform=ccrs.PlateCarree())
    share = 100 * attribution[c].sum() / n_value
    ax2.set_title(f"{species} — {attribution[c].sum():+.4f}  ({share:+.0f}% of N)",
                  fontsize=9.5, loc="left", pad=3)
    species_axes.append(ax2)

fig.colorbar(image, ax=species_axes, orientation="horizontal", fraction=0.04,
             pad=0.03, aspect=34, extend="both",
             label="contribution to N per cell (signed)")

fig.suptitle(
    f"Emission sources of the nonlinear term over the {args.output_region}, "
    f"year {year}\n"
    f"a partition of the 1850→{year} conditioning change that occurred — "
    f"NOT a counterfactual perturbation",
    fontsize=10.5, y=1.0)

path = OUT.format(region=args.output_region.replace(". ", "").replace(" ", "_"))
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
for p in (path, os.path.splitext(path)[0] + ".pdf"):
    fig.savefig(p, bbox_inches="tight")
    print(f"\n[step 2] wrote {p}")
