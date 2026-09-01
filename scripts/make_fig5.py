#!/usr/bin/env python3
"""
================================================================================
 FIGURES 5 AND 6 — the climate offset, over each experiment's last 30 years
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig5.py

Everything configurable is in the SETTINGS block below. There are no
command-line options and no helper functions: the script runs top to bottom in
six numbered steps. Same shape as make_fig1.py, make_fig2.py and make_fig3.py.

WHAT THESE FIGURES ARE FOR
--------------------------
The statistical tests say every experiment's mean differs: the emulator runs
0.05-0.16 degC cool for temperature, and cool for precipitation too. That is a
number in a table. These figures show it as a picture, and show at the same time
WHY it is easy to miss.

Two things are drawn per panel, and the contrast between them is the point:

  * THICK LINES — each side's ensemble mean, year by year. The vertical gap
    between them is the offset. It persists: it is not a few bad years, it is
    every year, in the same direction.

  * SHADED BAND — the range of individual CESM2 members. It is several times
    wider than the gap, which is exactly why a single realization cannot reveal
    the offset and why the timeseries figures 1 and 2 show the bias comfortably
    inside their spread band.

The horizontal dashed lines are each side's mean over the whole window. Their
separation is the offset with the year-to-year wiggle removed.

ABSOLUTE VALUES, NOT ANOMALIES. An anomaly plot subtracts each side's own
baseline and so removes the very thing being shown.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# The emulator's evaluation output: one NetCDF per scenario, holding
# {var}_model with dims (member, year, lat, lon).
EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"

# The CESM2 reference, built by scripts/make_cesm2_reference.py. Held-out
# members only, already in the emulator's units.
REFERENCE_DIR = "/home/nordling/mnt/lumi_sc/emulator_data/cesm2_reference"

# One figure per variable, each a 2x2 grid of experiments.
FIGURE_NAME = {"TREFHT": "fig05", "PRECT": "fig06"}

OUT = "plots/{name}.png"               # the .pdf sibling is written alongside

# Cap the emulator at the CESM2 member count, per experiment. The eval has 25
# members and CESM2 has 6-11, so leaving them unequal means the two ensemble
# means are converged to different degrees and the tests compare samples of very
# different size. Matching makes every comparison n against n. Selection is the
# first N members — deterministic, never random.
MATCH_MEMBER_COUNTS = True

# How many years at the END of each experiment to show.
N_YEARS = 30

# Variables, with the label and axis unit.
VARIABLES = {"TREFHT": ("Temperature", "$^{\\circ}$C"),
             "PRECT":  ("Precipitation", "mm day$^{-1}$")}

# The experiments, in plotting order: key -> label.
SCENARIOS = {
    "hist":   "Historical",
    "ssp370": "SSP3-7.0",
    "aaer":   "Aerosol-only (AAER)",
    "ghg":    "Greenhouse-gas-only (GHG)",
}

EMULATOR_COLOUR = "#D55E00"
CESM_COLOUR = "#0072B2"

# =============================================================================

import os

import matplotlib
matplotlib.use("Agg")                  # write files; no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

# =============================================================================
#  STEP 1 — the emulator's global means over the last N years
# =============================================================================
# The eval file holds maps, so the global mean is computed here, cos(lat)-
# weighted: grid cells shrink towards the poles, and an unweighted mean over
# this grid is wrong by degrees, not decimals.

emulator = {}          # (variable, scenario) -> (years, values shaped (member, year))
for variable in VARIABLES:
    for scenario in SCENARIOS:
        dataset = xr.open_dataset(f"{EVAL_DIR}/{variable}_{scenario}.nc")
        field = dataset[f"{variable}_model"].isel(year=slice(-N_YEARS, None))
        weights = np.cos(np.deg2rad(field["lat"]))
        global_mean = field.weighted(weights).mean(("lat", "lon")).compute()
        emulator[(variable, scenario)] = (global_mean["year"].values,
                                          global_mean.values)
        dataset.close()
        print(f"[step 1] {variable:6s} {scenario:7s} emulator "
              f"{global_mean.sizes['member']:2d} members, "
              f"{global_mean['year'].values[0]}-{global_mean['year'].values[-1]}")

# =============================================================================
#  STEP 2 — the same for CESM2
# =============================================================================
# Held-out members only; that filtering happened when the reference was built,
# so nothing here can accidentally score the emulator against its own training
# data.

cesm = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        dataset = xr.open_dataset(f"{REFERENCE_DIR}/{variable}_{scenario}.nc")
        field = dataset[f"{variable}_cesm"].isel(year=slice(-N_YEARS, None))
        weights = np.cos(np.deg2rad(field["lat"]))
        global_mean = field.weighted(weights).mean(("lat", "lon")).compute()
        cesm[(variable, scenario)] = (global_mean["year"].values,
                                      global_mean.values)
        dataset.close()
        print(f"[step 2] {variable:6s} {scenario:7s} CESM2    "
              f"{global_mean.sizes['member']:2d} members, "
              f"{global_mean['year'].values[0]}-{global_mean['year'].values[-1]}")

# =============================================================================
#  STEP 2b — same n on both sides
# =============================================================================
# The cap for each experiment is CESM2's own member count, so this waits until
# both sides are read. Truncating rather than subsampling keeps it reproducible.

if MATCH_MEMBER_COUNTS:
    for variable in VARIABLES:
        for scenario in SCENARIOS:
            years, values = emulator[(variable, scenario)]
            n_cesm = cesm[(variable, scenario)][1].shape[0]
            if values.shape[0] > n_cesm:
                emulator[(variable, scenario)] = (years, values[:n_cesm])
                print(f"[step 2b] {variable:6s} {scenario:7s} emulator "
                      f"{values.shape[0]} -> {n_cesm} members")

# =============================================================================
#  STEP 3 — the offset, and the spread it hides inside
# =============================================================================
# Two numbers per panel. The offset is what the figure demonstrates; the ratio
# of the CESM2 member range to that offset is why it takes an ensemble to see.

offsets = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        _, emulator_values = emulator[(variable, scenario)]
        _, cesm_values = cesm[(variable, scenario)]
        offset = float(emulator_values.mean() - cesm_values.mean())
        # Typical width of the CESM2 member envelope in one year.
        band_width = float(np.mean(cesm_values.max(axis=0) - cesm_values.min(axis=0)))
        offsets[(variable, scenario)] = dict(offset=offset, band=band_width)
        print(f"[step 3] {variable:6s} {scenario:7s} offset {offset:+.3f}, "
              f"CESM2 member range {band_width:.3f} "
              f"({band_width / abs(offset):.1f}x the offset)")

# =============================================================================
#  STEP 4 — draw one 2x2 figure per variable
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

for variable in VARIABLES:
    variable_label, unit_axis = VARIABLES[variable]
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))

    for panel_index, scenario in enumerate(SCENARIOS):
        axis = axes[panel_index // 2][panel_index % 2]
        emulator_years, emulator_values = emulator[(variable, scenario)]
        cesm_years, cesm_values = cesm[(variable, scenario)]

        # CESM2's member envelope first, so the lines sit on top of it. This is
        # the noise the offset hides inside.
        axis.fill_between(cesm_years, cesm_values.min(axis=0), cesm_values.max(axis=0),
                          color=CESM_COLOUR, alpha=0.18, lw=0, zorder=1)

        # The two ensemble means. The gap between these IS the offset.
        axis.plot(cesm_years, cesm_values.mean(axis=0), color=CESM_COLOUR,
                  lw=2.2, zorder=4, label="CESM2")
        axis.plot(emulator_years, emulator_values.mean(axis=0), color=EMULATOR_COLOUR,
                  lw=2.2, zorder=4, label="Emulator")

        # Window means: the offset with the year-to-year wiggle removed.
        axis.axhline(cesm_values.mean(), color=CESM_COLOUR, lw=1.1, ls="--", zorder=3)
        axis.axhline(emulator_values.mean(), color=EMULATOR_COLOUR, lw=1.1, ls="--",
                     zorder=3)

        row = offsets[(variable, scenario)]
        axis.set_title(f"{SCENARIOS[scenario]}\n"
                       f"offset {row['offset']:+.3f}, "
                       f"CESM2 member range {row['band'] / abs(row['offset']):.0f}$\\times$ wider",
                       fontsize=9.5, loc="left")
        axis.text(0.03, 0.95, f"({'abcd'[panel_index]})", transform=axis.transAxes,
                  fontweight="bold", va="top", fontsize=9)
        if panel_index // 2 == 1:
            axis.set_xlabel("Year")
        if panel_index % 2 == 0:
            axis.set_ylabel(f"{variable_label} ({unit_axis})")

    legend_entries = [
        Line2D([], [], color=CESM_COLOUR, lw=2.2, label="CESM2 ensemble mean"),
        Line2D([], [], color=EMULATOR_COLOUR, lw=2.2, label="Emulator ensemble mean"),
        Patch(facecolor=CESM_COLOUR, alpha=0.18, label="CESM2 member range"),
        Line2D([], [], color="0.35", lw=1.1, ls="--", label=f"{N_YEARS}-year mean"),
    ]
    figure.tight_layout()
    legend = figure.legend(handles=legend_entries, frameon=False, ncols=4,
                           loc="lower center", bbox_to_anchor=(0.5, 1.005))

    out_png = OUT.format(name=FIGURE_NAME[variable])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    for path in (out_png, os.path.splitext(out_png)[0] + ".pdf"):
        # The legend sits OUTSIDE the axes, so it has to be named here; a tight
        # bbox crops whatever it is not told about.
        figure.savefig(path, bbox_inches="tight", bbox_extra_artists=[legend])
        print(f"[step 4] wrote {path}")
    plt.close(figure)

# =============================================================================
#  STEP 5 — say what the figure shows
# =============================================================================

print(f"\nOffset over the last {N_YEARS} years (emulator minus CESM2)")
for variable in VARIABLES:
    for scenario in SCENARIOS:
        row = offsets[(variable, scenario)]
        print(f"  {VARIABLES[variable][0]:13s} {SCENARIOS[scenario]:26s} "
              f"{row['offset']:+.3f}   hidden inside a member range "
              f"{row['band'] / abs(row['offset']):.0f}x wider")

# =============================================================================
#  STEP 6 — the sign is the story
# =============================================================================
# Every offset having the same sign is a stronger statement than any one of
# them: a cold bias in one experiment could be chance, a cold bias in all eight
# cannot.

signs = [np.sign(offsets[(v, s)]["offset"]) for v in VARIABLES for s in SCENARIOS]
if len(set(signs)) == 1:
    direction = "LOW" if signs[0] < 0 else "HIGH"
    print(f"\nAll {len(signs)} offsets are the same sign: the emulator runs "
          f"{direction} in every experiment and both variables.")
else:
    print(f"\nOffsets are not all the same sign: {signs}")
