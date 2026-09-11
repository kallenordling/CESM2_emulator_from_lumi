#!/usr/bin/env python3
"""
================================================================================
 FIGURE 16 — THE EMISSION MODES THE EMULATOR CAN ACTUALLY SEE
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig16_eofs.py

WHAT THIS IS
------------
The aerosol conditioning channels are PCA-denoised to FIVE components each
(`n_components_cond: [30, 5, 5]` for CO2, SUL, BC) and reconstructed back to map
space before the network sees them. So the model's entire vocabulary of aerosol
patterns is five spatial modes per species: any question of the form "which
emissions, and where, drive the response" has to be answered in this basis,
because a per-grid-cell perturbation is an input the emulator was never trained
on.

This plots those modes. They come from `scripts/dump_cond_eofs.py`, which pulls
them out of a checkpoint's persisted PCA — the basis eval actually uses, not a
refit.

READ THE VARIANCE NUMBERS FIRST
-------------------------------
EOF1 carries 82-97% of the variance in every basis. It is the global emission
pattern getting stronger or weaker in time, and it is very nearly the whole
story: modes 4 and 5 are under 1% and are noise for this purpose. So "five
modes" is really ONE amplitude mode plus TWO OR THREE regional redistribution
modes, and those regional modes are the only place a "where" answer can live.

THE SIGN OF AN EOF IS ARBITRARY
-------------------------------
PCA fixes each component only up to a sign. Each mode here is flipped so its
area-weighted mean is positive, which makes EOF1 read as "more emissions"
rather than "less". The pairing of a mode with its time series carries the
physical sign, and that is not in this figure.
"""

# =============================================================================
#  SETTINGS
# =============================================================================

# Written by scripts/dump_cond_eofs.py, fetched from LUMI.
EOF_FILE = "plots/cond_eofs_ep0863.npz"

OUT = "plots/xai14/xai14.png"            # the .pdf sibling is written alongside
TABLE = "plots/xai14/xai14_stats.tex"

# Which basis to draw. The per-scenario bases are what eval uses; ssp370 is the
# one that governs 2031-2050, the window figures 11-13 cover.
BASIS = "ssp370"

# The years each basis was FITTED on — time is PCA's sample axis, so this is
# what the modes are modes OF. Taken from the cond files named in
# configs/config_data_ybias_BCprect.yaml (emissions_<scenario>_only_timefixed_bc_co2fix.nc);
# they are not carried in the .npz, so they are recorded here.
#
# NOTE ssp370 IS 21st CENTURY ONLY. Its EOF1 is the ssp370 DECLINE of sulfate,
# not the historical rise-and-fall — the hist basis, fitted on 1850-2014, gives
# a different leading mode. ssp370 is the right basis for the 2031-2050 window
# of figures 11-13, and the wrong one for a statement about the whole record.
BASIS_YEARS = {"hist": "1850-2014 (n=165)", "ssp370": "2015-2100 (n=86)",
               "aaer": "1850-2050 (n=201)", "ghg": "1850-2050 (n=201)"}

# Species to draw, and how many modes each. Beyond mode 3 the variance is under
# 1% everywhere and the maps are noise.
#
# CO2 IS INCLUDED even though it is the smooth channel — it gets 30 components
# rather than 5 and no spatial smoothing (`cond_smooth_sigma: [0, 2, 2]`),
# because cumulative emissions are already smooth. Its modes are still spatial
# and a regional CO2 question has to be answered in them.
SPECIES = ["CO2", "SUL", "BC"]
N_MODES = 3

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
    sys.exit("[error] cartopy is missing — run with "
             "/home/nordling/miniconda3/envs/plotting/bin/python")

if not os.path.exists(EOF_FILE):
    sys.exit(f"[error] {EOF_FILE} not found. Build it on LUMI:\n"
             f"        bash run_dump_eofs.sh runs/run_mseyb_BCprect_863.pt "
             f"cond_eofs_ep0863.npz\n"
             f"        then scp it into plots/")

store = np.load(EOF_FILE)

# =============================================================================
#  STEP 1 — pull the modes, and orient them
# =============================================================================
# A DEGENERATE BASIS IS A REAL RESULT, NOT A FILE ERROR. The single-forcing runs
# hold the other species at 1850 levels, so their conditioning channel has ZERO
# variance in time and fit_pca_denoise falls back to a 1-component dummy whose
# explained variance is NaN. That is why ghg has no SUL or BC basis and aaer has
# no CO2 basis — and it is the reason the interaction N = ALL - GHG - AAER can
# only be differentiated with respect to both species in the hist and ssp370
# bases, where both actually vary.

modes, variance = {}, {}
for species in SPECIES:
    key = f"components_{BASIS}_{species}"
    if key not in store:
        sys.exit(f"[error] {EOF_FILE} has no {key}")
    components = store[key]
    ratios = store[f"variance_{BASIS}_{species}"]
    if not np.isfinite(ratios).all():
        sys.exit(f"[error] the '{BASIS}' basis for {species} is DEGENERATE "
                 f"(NaN variance): that channel is constant in this scenario, "
                 f"so it has no modes to plot. Pick a basis where it varies.")

    # Latitude weights: the grid is regular in degrees, so a mode that looks
    # large near the pole covers very little area.
    height = components.shape[1]
    latitude = np.linspace(-90, 90, height)
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None],
                              components.shape[1:])

    oriented = []
    for index in range(N_MODES):
        mode = components[index]
        if np.average(mode, weights=weights) < 0:
            mode = -mode
        oriented.append(mode)
    modes[species] = np.array(oriented)
    variance[species] = ratios
    print(f"[step 1] {species}: {components.shape[0]} modes, "
          f"variance {np.round(100 * ratios, 1)}")

# =============================================================================
#  STEP 2 — the figure
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

fig = plt.figure(figsize=(4.5 * N_MODES, 2.6 * len(SPECIES) + 1.0))
grid = fig.add_gridspec(len(SPECIES), N_MODES, hspace=0.32, wspace=0.06)
projection = ccrs.Robinson(central_longitude=0)

# Hold the map axes explicitly: `fig.axes` grows as colour bars are added, so
# indexing it by row/column attaches the second row's bar to the first row's
# panels. Same trap as make_fig1314_maps.py.
species_axes = {name: [] for name in SPECIES}
for row, species in enumerate(SPECIES):
    # ONE SCALE PER SPECIES, across its modes: the higher modes are genuinely
    # smaller than EOF1, and rescaling each panel to fill its own colour range
    # would hide exactly that.
    scale = float(np.nanpercentile(np.abs(modes[species]), 99.5))
    for column in range(N_MODES):
        ax = fig.add_subplot(grid[row, column], projection=projection)
        image = ax.pcolormesh(np.linspace(0, 360, modes[species].shape[2]),
                              np.linspace(-90, 90, modes[species].shape[1]),
                              modes[species][column], cmap="RdBu_r",
                              vmin=-scale, vmax=scale, shading="auto",
                              transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.35, color="0.25")
        ax.set_global()
        ax.set_title(f"{species} EOF{column + 1} of "
                     f"{len(variance[species])} — "
                     f"{100 * variance[species][column]:.1f}% of variance",
                     fontsize=9.5, pad=4)
        species_axes[species].append(ax)

    fig.colorbar(image, ax=species_axes[species], orientation="vertical",
                 fraction=0.018, pad=0.015, extend="both",
                 label=f"{species} loading")

fig.suptitle(
    f"The emission modes the emulator can see — '{BASIS}' basis, "
    f"fitted on {BASIS_YEARS.get(BASIS, 'unknown years')}",
    fontsize=11, y=0.99)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
for path in (OUT, os.path.splitext(OUT)[0] + ".pdf"):
    fig.savefig(path, bbox_inches="tight")
    print(f"[step 2] wrote {path}")
plt.close(fig)

# =============================================================================
#  STEP 3 — where each mode puts its weight
# =============================================================================
# The point of the table: name the modes. A mode whose loading concentrates over
# East Asia is an "East Asian emissions" mode, and that is the language a
# sensitivity result has to be reported in.

REGIONS = [
    ("East Asia",      20.0,  50.0, 100.0, 145.0),
    ("South Asia",      5.0,  30.0,  65.0, 100.0),
    ("Europe",         35.0,  65.0,   0.0,  40.0),
    ("N. America",     30.0,  60.0, 230.0, 300.0),
    ("Africa",        -35.0,  35.0,   0.0,  50.0),
    ("S. America",    -55.0,  12.0, 280.0, 325.0),
]

rows_tex = []
for species in SPECIES:
    height, width = modes[species].shape[1:]
    latitude = np.linspace(-90, 90, height)
    longitude = np.linspace(0, 360, width)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    weights = np.cos(np.deg2rad(latitude))[:, None]

    for index in range(N_MODES):
        mode = modes[species][index]
        # Share of the mode's total absolute loading that falls in each region:
        # what fraction of "this pattern" is that part of the world.
        total = float(np.average(np.abs(mode),
                                 weights=np.broadcast_to(weights, mode.shape)))
        shares = []
        for _, lat_min, lat_max, lon_min, lon_max in REGIONS:
            inside = ((latitude_grid >= lat_min) & (latitude_grid <= lat_max)
                      & (longitude_grid >= lon_min) & (longitude_grid <= lon_max))
            area = np.where(inside, np.broadcast_to(weights, inside.shape), 0.0)
            local = float(np.average(np.abs(mode), weights=area))
            shares.append(100 * local * float(area.sum())
                          / (total * float(np.broadcast_to(
                              weights, mode.shape).sum())))
        rows_tex.append(
            f"{species} EOF{index + 1} & "
            f"{100 * variance[species][index]:.1f} & "
            + " & ".join(f"{s:.1f}" for s in shares) + r" \\")
        print(f"[step 3] {species} EOF{index + 1}: "
              + ", ".join(f"{name} {s:.1f}%"
                          for (name, *_), s in zip(REGIONS, shares)))

caption = (
    f"The conditioning modes, from the persisted ``{BASIS}'' PCA basis of the "
    f"checkpoint behind figures 9--13, fitted on "
    f"{BASIS_YEARS.get(BASIS, 'unknown years')} --- time is the sample axis, so "
    f"these are the modes of how the emissions VARY over that period, and the "
    f"ssp370 basis therefore describes the 21st century alone. The CO$_2$ "
    f"channel is denoised to 30 components and the "
    f"sulfate and black-carbon channels to five "
    f"each before the network sees them, so these patterns are the emulator's "
    f"complete vocabulary for ``where the emissions are'': a perturbation of a "
    f"single grid cell is an input it was never trained on. ``Variance'' is "
    f"the share of the channel's temporal variance the mode carries, and the "
    f"regional columns give the share of the mode's total absolute loading "
    f"falling in each region, so they say what the mode is a pattern OF. Note "
    f"that EOF1 alone carries the great majority of the variance --- it is the "
    f"global pattern scaling up and down in time --- so the regional "
    f"redistribution that a ``where'' question asks about lives in EOF2 and "
    f"EOF3. The sign of a principal component is arbitrary; each mode here is "
    f"flipped so its area-weighted mean is positive.")

table_tex = "\n".join([
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{" + caption + "}",
    r"\label{tab:xai14}",
    r"\begin{tabular}{|l|r|" + "r|" * len(REGIONS) + "}",
    r"\hline",
    r"\textbf{Mode} & \textbf{Variance} & "
    + " & ".join(r"\textbf{" + name + "}" for name, *_ in REGIONS) + r" \\",
    r" & (\%) & \multicolumn{" + str(len(REGIONS))
    + r"}{c|}{share of the mode's loading, \%} \\",
    r"\hline", *rows_tex, r"\hline",
    r"\end{tabular}", r"\end{table}",
])

os.makedirs(os.path.dirname(TABLE) or ".", exist_ok=True)
with open(TABLE, "w") as handle:
    handle.write(f"% The conditioning EOFs, '{BASIS}' basis.\n"
                 f"% Built from {EOF_FILE} by scripts/make_fig16_eofs.py\n"
                 "% — do not edit by hand.\n")
    handle.write(table_tex + "\n")
print(f"[step 3] wrote {TABLE}")
