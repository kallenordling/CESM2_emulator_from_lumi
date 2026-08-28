#!/usr/bin/env python3
"""
================================================================================
 FIGURE 2 — emulated vs CESM2 global-mean precipitation
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig2.py

Everything you might want to change is in the SETTINGS block below. There are no
command-line options and no helper functions: the script runs top to bottom in
eleven numbered steps, so you can read it as a description of how the figure is
made.

(scripts/paper_fig_timeseries.py is the flexible version of this — it takes
arguments, handles precipitation and the unseen scenarios, and is what fig02,
fig06 and fig07 use. This file exists to be READ.)

WHAT THE FIGURE SHOWS
---------------------
Panel (a): global-mean precipitation change vs 1850-1900, four experiments,
emulator (solid) against held-out CESM2 (dashed with circles), with CESM2's
member range shaded.

PRECIPITATION IS SHOWN AS A PERCENTAGE, not an absolute rate. A few hundredths
of a mm/day means nothing without the ~2.9 mm/day it is relative to, and the two
sides are compared as fractional change so a small offset in the mean state does
not masquerade as a difference in response. Panels (b)-(d) are therefore in
PERCENTAGE POINTS.

Panels (b-d): the bias — emulator ensemble mean minus CESM2 ensemble mean — over
a grey band of +/-2 sigma of CESM2's own members about their mean. A bias line
inside that band is indistinguishable from internal variability.

THE ONE IDEA THAT MATTERS
-------------------------
Both sides are ENSEMBLES, and neither is truth. CESM2's members differ from each
other by internal variability alone, so the question is never "is the emulator
exactly right" but "is it as close as two CESM2 members are to each other".
Everything below exists to make that comparison fair: held-out members only,
equal ensemble sizes, and each side referenced to its own pre-industrial.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# Where the emulator's evaluation output lives (one NetCDF per scenario).
EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25"

# Where the CESM2 training trees live: <TREE_ROOT>/<scenario>/<member>/chunk_*.nc
TREE_ROOT = "/home/nordling/mnt/lumi_sc/emulator_data/training_data/PRECT"

# Which CESM2 members were TRAINED on. Everything else on disk is held out and
# usable as a reference. Read from the training config so the two cannot drift.
DATA_CONFIG = "configs/config_data_ybias_BCprect.yaml"

# Reading ~37 member directories over a network mount takes several minutes, so
# the per-member global means are cached here. Delete this file to force a
# re-read (necessary after repairing the underlying data).
CACHE = "plots/fig2_cesm2_members.csv"

OUT = "plots/fig2.png"          # the .pdf sibling is written alongside

# The four experiments, in plotting order:
#   key -> (legend label, tree subdirectory, colour)
# Colours are Okabe-Ito, which stay distinguishable in greyscale and to
# colour-blind readers.
SCENARIOS = {
    "hist":   ("Historical",                "hist",   "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "ssp370", "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "AAER",   "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "GHG",    "#009E73"),
}

BASELINE = (1850, 1900)   # the anomaly reference window
BASELINE_SLICE = slice(*BASELINE)   # used as da.sel(year=BASELINE_SLICE)
YEAR_MAX = 2100

# Cap the emulator at the CESM2 member count. A 25-member mean is better
# converged than a 10-member one, and comparing them would flatter the emulator.
MATCH_MEMBER_COUNTS = True

# The LENS2 trees store precipitation as m/s; the emulator writes mm/day.
# 1 m/s = 1000 mm/s = 1000 x 86400 mm/day.
TREE_TO_MM_PER_DAY = 1000.0 * 86400.0

# =============================================================================

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")                      # no display needed; write files only
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr
import yaml

print(__doc__.split("WHAT THE FIGURE")[0])

# =============================================================================
#  STEP 1 — work out which CESM2 members are held out
# =============================================================================
# The emulator was trained on some members of each experiment. Scoring it
# against those would be marking its own homework, so the reference is built
# from the members on disk that do NOT appear in the training config.

# There are two sources of truth to reconcile.
#
#   1. The TRAINING CONFIG says which realizations the emulator was fitted on.
#      Its experiment_configs section looks like:
#          - scenario_name: hist
#            realizations: [LE2-1001.001, LE2-1011.001, ...]
#   2. The DISK holds every realization that was staged, trained on or not.
#
# Held out = on disk, minus trained on. Reading the list from the config rather
# than hardcoding it means the two cannot drift apart when training changes.

config = yaml.safe_load(open(DATA_CONFIG))
trained_members = {
    experiment["scenario_name"]: set(experiment.get("realizations", []))
    for experiment in config["experiment_configs"]
}

heldout = {}
for key, (_, subdir, _) in SCENARIOS.items():
    experiment_dir = f"{TREE_ROOT}/{subdir}"

    # One subdirectory per realization. "diagnostics" is a folder of staging
    # plots that sits alongside them and is not a member.
    members_on_disk = {
        name for name in os.listdir(experiment_dir)
        if name != "diagnostics" and os.path.isdir(f"{experiment_dir}/{name}")
    }
    members_trained_on = trained_members.get(key, set())
    heldout[key] = sorted(members_on_disk - members_trained_on)

    print(f"[step 1] {key:7s} {len(members_on_disk):2d} on disk"
          f" - {len(members_trained_on):2d} trained"
          f" = {len(heldout[key]):2d} held out")


# =============================================================================
#  STEP 2 — read the held-out CESM2 members (or reuse the cache)
# =============================================================================
# For every member we want ONE number per year: the global mean temperature.
# The average must be cos(lat)-weighted — grid cells shrink towards the poles,
# and an unweighted mean over this grid is wrong by degrees, not decimals.
#
# The result per experiment is a table of year x member.

# Everything below is an xarray.DataArray with dims (member, year), so a
# baseline is `da.sel(year=slice(1850, 1900))` and an anomaly is a subtraction —
# the same notation the NetCDF files themselves invite.

cesm = {}

if os.path.exists(CACHE):
    print(f"[step 2] reusing {CACHE}")
    cached = pd.read_csv(CACHE)          # long form: one row per member-year

    for scenario, rows in cached.groupby("scenario"):
        # long form -> a table with one column per member, one row per year
        table = rows.pivot(index="year", columns="member", values="gmean")
        table = table.sort_index()
        # table -> a DataArray with dims (member, year)
        cesm[scenario] = table.to_xarray().to_array("member")
else:
    print(f"[step 2] reading members from {TREE_ROOT} (minutes, not seconds)")
    for key, (_, subdir, _) in SCENARIOS.items():
        columns = {}
        for i, member in enumerate(heldout[key], 1):
            files = sorted(glob.glob(f"{TREE_ROOT}/{subdir}/{member}/*.nc"))
            print(f"          [{i}/{len(heldout[key])}] {subdir}/{member}", flush=True)
            ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
            weights = np.cos(np.deg2rad(ds["PRECT"]["lat"]))
            gmean = ds["PRECT"].weighted(weights).mean(("lat", "lon")).compute()
            gmean = gmean * TREE_TO_MM_PER_DAY          # m/s -> mm/day
            years = np.asarray(ds["time" if "time" in gmean.dims else "year"]
                               .values).astype(int)
            columns[member] = pd.Series(np.asarray(gmean.values, float),
                                        index=years).sort_index()
            ds.close()
        # one Series per member -> a year x member table -> (member, year)
        table = pd.DataFrame(columns).sort_index().rename_axis("year")
        cesm[key] = table.to_xarray().to_array("member")
    # Flatten back to long form for the cache: one row per member-year, which
    # survives a CSV round-trip without needing the column names to be parsed.
    rows = []
    for scenario, da in cesm.items():
        for member in da["member"].values:
            series = da.sel(member=member)
            for year, value in zip(series["year"].values, series.values):
                rows.append(dict(scenario=scenario, member=str(member),
                                 year=int(year), gmean=float(value)))

    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"[step 2] cached {len(rows)} member-years to {CACHE}")

# NOTE ON BAD DATA: a member with a corrupt year shows up here as a NaN or an
# obvious outlier, and would drag the reference mean and inflate its spread.
# LE2-1231.012 had 1930-1939 broken in TEMPERATURE, and scripts/refetch_member.py
# repaired it; its precipitation was clean throughout, which is what showed the
# damage was staging rather than the source. Every statistic below skips NaN, so
# a member with a missing year still contributes its good ones.

# =============================================================================
#  STEP 3 — read the emulator's output
# =============================================================================
# eval_aero.py writes one NetCDF per scenario holding each ensemble member's
# global-mean series as PRECT_model_gmean_m1, _m2, ... already in mm/day.

emulator = {}
for key in SCENARIOS:
    ds = xr.open_dataset(f"{EVAL_DIR}/PRECT_{key}.nc")

    # The file has one variable per member, named with the member number:
    #     PRECT_model_gmean_m1, PRECT_model_gmean_m2, ... _m25
    # Collect them as {member number: variable name}. Taking the number from
    # the name — rather than sorting the names — matters: sorted() on strings
    # gives m1, m10, m11, ..., m2, so member 10 would end up second and every
    # member-by-member comparison after this would be against the wrong one.
    variable_of_member = {}
    for variable in ds.data_vars:
        match = re.fullmatch(r"PRECT_model_gmean_m(\d+)", variable)
        if match:
            variable_of_member[int(match.group(1))] = variable

    member_numbers = sorted(variable_of_member)          # 1, 2, 3, ... 25
    series_per_member = [ds[variable_of_member[n]].values for n in member_numbers]

    emulator[key] = xr.DataArray(
        np.stack(series_per_member),
        dims=("member", "year"),
        coords={"member": member_numbers,
                "year": ds["year"].values.astype(int)})
    ds.close()
    print(f"[step 3] {key:7s} {len(member_numbers):2d} emulator members")

# =============================================================================
#  STEP 4 — put both ensembles on the same footing
# =============================================================================
# The eval has 25 members; CESM2 has 6-11 depending on the experiment. A mean of
# 25 is closer to the true forced response than a mean of 6, so comparing them
# would credit the emulator for an advantage in sampling rather than in physics.
# Keep the first N members — deterministic, never random.

if MATCH_MEMBER_COUNTS:
    for key in SCENARIOS:
        n = cesm[key].sizes["member"]
        have = emulator[key].sizes["member"]
        emulator[key] = emulator[key].isel(member=slice(None, n))
        print(f"[step 4] {key:7s} emulator capped {have} -> {n}")

# =============================================================================
#  STEP 5 — anomalies: each side referenced to ITS OWN pre-industrial
# =============================================================================
# The emulator's absolute climate is close to CESM2's but not identical, and a
# constant offset between them is not what this figure is about. Referencing
# each side to its own 1850-1900 mean removes that offset and leaves the
# WARMING, which is the quantity being compared.
#
# Experiments that start in 2015 (ssp370) have no pre-industrial of their own,
# so they inherit the historical baseline — the same convention on both sides.

# hist, aaer and ghg all start in 1850, so each has its own baseline:
#     da_anom = da - da.sel(year=BASELINE_SLICE).mean()
cesm_base, emu_base = {}, {}
for key in SCENARIOS:
    if key == "ssp370":
        continue
    cesm_base[key] = float(cesm[key].sel(year=BASELINE_SLICE).mean())
    emu_base[key] = float(emulator[key].sel(year=BASELINE_SLICE).mean())

# ssp370 is the exception: it begins in 2015 and has no pre-industrial of its
# own, so it inherits the historical baseline — on BOTH sides, which is what
# keeps the two comparable.
cesm_base["ssp370"] = cesm_base["hist"]
emu_base["ssp370"] = emu_base["hist"]

# Both sides are now mm/day, and each baseline is that side's own pre-industrial
# precipitation rate — roughly 2.9 mm/day. It is the DENOMINATOR of the percent
# change below, not just an offset, which is why the conversion in step 2 has to
# be right: a wrong factor would rescale every anomaly in the figure.

# =============================================================================
#  STEP 6 — lay out the figure
# =============================================================================
# One wide overview panel on top, and beneath it one bias panel per experiment.
# hist and ssp370 share a bias panel because they are one continuous
# trajectory: hist ends in 2014, ssp370 begins in 2015.

bias_panels = [(("hist", "ssp370"), "Historical + SSP3-7.0"),
               (("aaer",), SCENARIOS["aaer"][0]),
               (("ghg",),  SCENARIOS["ghg"][0])]

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
fig = plt.figure(figsize=(9.5, 7.6))
grid = fig.add_gridspec(2, len(bias_panels), height_ratios=[2.3, 1.0],
                        hspace=0.30, wspace=0.12)
ax_main = fig.add_subplot(grid[0, :])
ax_bias = [fig.add_subplot(grid[1, i], sharey=None if i == 0 else None)
           for i in range(len(bias_panels))]

# =============================================================================
#  STEP 7 — draw panel (a), one experiment at a time
# =============================================================================
# Per experiment: CESM2's member range as shading, CESM2's mean as a dashed line
# with open circles, and the emulator's mean as a thick solid line.
#
# The circles matter. Solid-vs-dashed in the same colour is unreadable where the
# two curves coincide, which is most of the record — a marker shape survives
# overlap, greyscale and print size.

stats, bias_series = {}, {}
for key, (label, _, colour) in SCENARIOS.items():
    # Anomaly = change from the side's own baseline, as a PERCENTAGE of it.
    ref_anom = (100 * (cesm[key] - cesm_base[key]) / cesm_base[key]
                ).sel(year=slice(None, YEAR_MAX))
    emu_anom = (100 * (emulator[key] - emu_base[key]) / emu_base[key]
                ).sel(year=slice(None, YEAR_MAX))

    ref_mean = ref_anom.mean("member")
    lo, hi = ref_anom.min("member"), ref_anom.max("member")
    years_ref = ref_anom["year"].values

    ax_main.fill_between(years_ref, lo, hi, color=colour, alpha=0.26,
                         lw=0, zorder=1)
    for edge in (lo, hi):                           # a thin edge pins the band
        ax_main.plot(years_ref, edge, color=colour, lw=0.7, alpha=0.55, zorder=1)
    ax_main.plot(years_ref, ref_mean, color=colour, lw=1.2, ls="--",
                 marker="o", markersize=3.4, markevery=8, markerfacecolor="white",
                 markeredgecolor=colour, zorder=5,
                 path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    ax_main.plot(emu_anom["year"].values, emu_anom.mean("member"),
                 color=colour, lw=2.6, zorder=4, label=label)

    # ── the numbers, on the years both sides cover ───────────────────────────
    # The ranges differ — CESM2's aaer/ghg trees stop in 2050 while the emulator
    # runs to 2100 — so compare only where both exist.
    common = np.intersect1d(emu_anom["year"].values, years_ref)
    emulator_mean = emu_anom.mean("member").sel(year=common)
    cesm_mean = ref_mean.sel(year=common)

    # This is the line drawn in panels (b)-(d).
    difference = emulator_mean - cesm_mean
    # ... and this is the grey band it is drawn over: how far a single CESM2
    # realization strays from the forced response by chance. A difference
    # inside +/-2 sigma is no larger than the disagreement between two CESM2
    # runs, which is the standard the emulator is being held to.
    sigma = ref_anom.sel(year=common).std("member", ddof=1)
    bias_series[key] = (common, difference, sigma)
    stats[key] = dict(
        n_emu=emu_anom.sizes["member"], n_cesm=ref_anom.sizes["member"],
        bias=float(difference.mean()),
        rmse=float(np.sqrt((difference ** 2).mean())),
        inside=float((abs(difference) <= 2 * sigma).mean()) * 100)
    print(f"[step 7] {key:7s} bias {stats[key]['bias']:+.3f} %-points, "
          f"rmse {stats[key]['rmse']:.3f}, "
          f"{stats[key]['inside']:.0f}% of years within CESM2's own spread")

# =============================================================================
#  STEP 8 — draw the bias panels
# =============================================================================
# The grey band is +/-2 sigma of CESM2's members about their own mean, computed
# per year. Where the coloured bias line sits inside it, the emulator differs
# from CESM2 by no more than one CESM2 member differs from another.

for i, (group, title) in enumerate(bias_panels):
    ax = ax_bias[i]
    ax.axhline(0, lw=0.8, color="0.3", zorder=1)
    for key in group:
        years, difference, sigma = bias_series[key]
        ax.fill_between(years, -2 * sigma, 2 * sigma,
                        color="0.45", alpha=0.22, lw=0, zorder=0)
        ax.plot(years, difference, color=SCENARIOS[key][2], lw=1.4, zorder=3)
    ax.set_title(f"{title}\nn = "
                 f"{'/'.join(str(stats[k]['n_emu']) for k in group)} emulator, "
                 f"{'/'.join(str(stats[k]['n_cesm']) for k in group)} CESM2",
                 fontsize=9, loc="left", pad=4)
    ax.text(0.02, 0.94, f"({'bcd'[i]})", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=9)
    ax.set_xlabel("Year")
    if i == 0:
        ax.set_ylabel("Bias (percentage points)\nensemble means")
    else:
        # All three panels share one y-scale (set below), so repeating the tick
        # labels adds nothing and collides with the neighbouring panel's text.
        ax.tick_params(labelleft=False)

# Every bias panel gets ONE y-scale, so a bias in one is the same size on the
# page as a bias in another, and ONE x-range — the same 1850-2100 as panel (a),
# so the columns line up under it. aaer and ghg end in 2050 and simply stop
# there, which is honest: the run is short, not the bias small.
limit = 1.15 * max(max(abs(float(d.min())), abs(float(d.max())),
                       2 * float(s.max()))
                   for _, d, s in bias_series.values())
for ax in ax_bias:
    ax.set_ylim(-limit, limit)
    ax.set_xlim(BASELINE[0], YEAR_MAX)
    # Prune the ticks at both ends: with three panels side by side spanning the
    # same years, a label on the right edge of one lands on the label at the
    # left edge of the next.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both",
                                           steps=[1, 5, 10]))

# =============================================================================
#  STEP 9 — legends
# =============================================================================
# Two of them, stacked above the panel so they cover no data: the scenario
# colours, and what the line styles and shadings mean.

style = [
    Line2D([], [], color="0.35", lw=2.6, label="EMULATOR — ensemble mean"),
    Line2D([], [], color="0.35", lw=1.2, ls="--", marker="o", markersize=3.4,
           markerfacecolor="white", markeredgecolor="0.35",
           label="CESM2 — held-out ensemble mean"),
    Patch(facecolor="0.35", alpha=0.26, label="(a) CESM2 member range (min–max)"),
    Patch(facecolor="0.55", alpha=0.20,
          label="(b–d) CESM2 spread about its mean (±2σ)"),
]
legend_scenarios = ax_main.legend(frameon=False, ncols=4, loc="lower left",
                                  bbox_to_anchor=(0.0, 1.14), handlelength=2.2)
ax_main.add_artist(legend_scenarios)
legend_style = ax_main.legend(handles=style, frameon=False, ncols=2, fontsize=8.2,
                              loc="lower left", bbox_to_anchor=(0.0, 1.005),
                              handlelength=2.6)

# =============================================================================
#  STEP 10 — finish panel (a)
# =============================================================================

ax_main.axhline(0, ls=":", lw=0.8, color="0.3")
ax_main.axvspan(*BASELINE, color="0.9", alpha=0.6, lw=0, zorder=0)  # baseline window
ax_main.set_ylabel("Precipitation change (%, vs 1850–1900)")
ax_main.set_xlabel("Year")
ax_main.set_xlim(BASELINE[0], YEAR_MAX)
ax_main.text(0.005, 0.97, "(a)", transform=ax_main.transAxes, fontweight="bold",
             va="top")

# =============================================================================
#  STEP 11 — save, and print the table
# =============================================================================
# bbox_extra_artists keeps the legends inside the tight bounding box; without it
# the top row gets cropped, because they sit outside the axes.

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
for path in (OUT, os.path.splitext(OUT)[0] + ".pdf"):
    fig.savefig(path, bbox_inches="tight",
                bbox_extra_artists=[legend_scenarios, legend_style])
    print(f"[step 11] wrote {path}")

print("\nEmulator vs held-out CESM2 (percentage points, years both sides cover)")
print(pd.DataFrame(stats).T.round(3).to_string())
