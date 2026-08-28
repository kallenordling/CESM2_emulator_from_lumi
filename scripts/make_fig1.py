#!/usr/bin/env python3
"""
================================================================================
 FIGURE 1 — emulated vs CESM2 global-mean temperature
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig1.py

Everything you might want to change is in the SETTINGS block below. There are no
command-line options and no helper functions: the script runs top to bottom in
eleven numbered steps, so you can read it as a description of how the figure is
made.

(scripts/paper_fig_timeseries.py is the flexible version of this — it takes
arguments, handles precipitation and the unseen scenarios, and is what fig02,
fig06 and fig07 use. This file exists to be READ.)

WHAT THE FIGURE SHOWS
---------------------
Panel (a): global-mean temperature anomaly vs 1850-1900, four experiments,
emulator (solid) against held-out CESM2 (dashed with circles), with CESM2's
member range shaded.

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
TREE_ROOT = "/home/nordling/mnt/lumi_sc/emulator_data/training_data/TREFHT"

# Which CESM2 members were TRAINED on. Everything else on disk is held out and
# usable as a reference. Read from the training config so the two cannot drift.
DATA_CONFIG = "configs/config_data_ybias_BCprect.yaml"

# Reading ~37 member directories over a network mount takes several minutes, so
# the per-member global means are cached here. Delete this file to force a
# re-read (necessary after repairing the underlying data).
CACHE = "plots/fig1_cesm2_members.csv"

OUT = "plots/fig1.png"          # the .pdf sibling is written alongside

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
YEAR_MAX = 2100

# Cap the emulator at the CESM2 member count. A 25-member mean is better
# converged than a 10-member one, and comparing them would flatter the emulator.
MATCH_MEMBER_COUNTS = True

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

cfg = yaml.safe_load(open(DATA_CONFIG))
trained = {e["scenario_name"]: set(e.get("realizations", []))
           for e in cfg["experiment_configs"]}

heldout = {}
for key, (_, subdir, _) in SCENARIOS.items():
    on_disk = {d for d in os.listdir(f"{TREE_ROOT}/{subdir}")
               if os.path.isdir(f"{TREE_ROOT}/{subdir}/{d}") and d != "diagnostics"}
    heldout[key] = sorted(on_disk - trained.get(key, set()))
    print(f"[step 1] {key:7s} {len(on_disk):2d} on disk, "
          f"{len(trained.get(key, set())):2d} trained, "
          f"{len(heldout[key]):2d} held out")

# =============================================================================
#  STEP 2 — read the held-out CESM2 members (or reuse the cache)
# =============================================================================
# For every member we want ONE number per year: the global mean temperature.
# The average must be cos(lat)-weighted — grid cells shrink towards the poles,
# and an unweighted mean over this grid is wrong by degrees, not decimals.
#
# The result per experiment is a table of year x member.

if os.path.exists(CACHE):
    print(f"[step 2] reusing {CACHE}")
    _c = pd.read_csv(CACHE)
    cesm = {k: g.pivot(index="year", columns="member", values="gmean").sort_index()
            for k, g in _c.groupby("scenario")}
else:
    print(f"[step 2] reading members from {TREE_ROOT} (minutes, not seconds)")
    cesm = {}
    for key, (_, subdir, _) in SCENARIOS.items():
        columns = {}
        for i, member in enumerate(heldout[key], 1):
            files = sorted(glob.glob(f"{TREE_ROOT}/{subdir}/{member}/*.nc"))
            print(f"          [{i}/{len(heldout[key])}] {subdir}/{member}", flush=True)
            ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
            weights = np.cos(np.deg2rad(ds["TREFHT"]["lat"]))
            gmean = ds["TREFHT"].weighted(weights).mean(("lat", "lon")).compute()
            years = np.asarray(ds["time" if "time" in gmean.dims else "year"]
                               .values).astype(int)
            series = pd.Series(np.asarray(gmean.values, float), index=years)
            columns[member] = series[~series.index.duplicated()].sort_index()
            ds.close()
        cesm[key] = pd.DataFrame(columns).sort_index()
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    pd.DataFrame([dict(scenario=k, member=m, year=int(y), gmean=float(v))
                  for k, df in cesm.items() for m in df.columns
                  for y, v in df[m].dropna().items()]).to_csv(CACHE, index=False)
    print(f"[step 2] cached to {CACHE}")

# NOTE ON BAD DATA: a member with a corrupt year shows up here as a NaN or an
# obvious outlier, and would drag the reference mean and inflate its spread.
# LE2-1231.012 had 1930-1939 broken until scripts/refetch_member.py re-fetched
# them from AWS. Every statistic below skips NaN, so a member with a missing
# year still contributes its good ones.

# =============================================================================
#  STEP 3 — read the emulator's output
# =============================================================================
# eval_aero.py writes one NetCDF per scenario holding each ensemble member's
# global-mean series as TREFHT_model_gmean_m1, _m2, ... already in degC.

emulator = {}
for key in SCENARIOS:
    path = f"{EVAL_DIR}/TREFHT_{key}.nc"
    ds = xr.open_dataset(path)
    names = sorted([v for v in ds.data_vars
                    if re.fullmatch(r"TREFHT_model_gmean_m\d+", v)],
                   key=lambda s: int(s.rsplit("_m", 1)[1]))
    emulator[key] = (ds["year"].values.astype(int),
                     np.stack([ds[n].values for n in names]))   # (member, year)
    ds.close()
    print(f"[step 3] {key:7s} {len(names):2d} emulator members")

# =============================================================================
#  STEP 4 — put both ensembles on the same footing
# =============================================================================
# The eval has 25 members; CESM2 has 6-11 depending on the experiment. A mean of
# 25 is closer to the true forced response than a mean of 6, so comparing them
# would credit the emulator for an advantage in sampling rather than in physics.
# Keep the first N members — deterministic, never random.

if MATCH_MEMBER_COUNTS:
    for key in SCENARIOS:
        n = cesm[key].shape[1]
        years, members = emulator[key]
        if members.shape[0] > n:
            emulator[key] = (years, members[:n])
            print(f"[step 4] {key:7s} emulator capped {members.shape[0]} -> {n}")

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

cesm_base, emu_base = {}, {}
for key in SCENARIOS:
    # CESM2 side
    window = cesm[key].loc[BASELINE[0]:BASELINE[1]]
    cesm_base[key] = float(window.mean(axis=1).mean()) if len(window) else np.nan
    # emulator side
    years, members = emulator[key]
    in_base = (years >= BASELINE[0]) & (years <= BASELINE[1])
    emu_base[key] = float(members[:, in_base].mean()) if in_base.any() else np.nan

# ssp370 begins in 2015 and has no pre-industrial of its own, so it inherits
# the historical baseline — the SAME convention applied to both sides.
for key in SCENARIOS:
    if not np.isfinite(cesm_base[key]):
        cesm_base[key] = cesm_base["hist"]
    if not np.isfinite(emu_base[key]):
        emu_base[key] = emu_base["hist"]
    print(f"[step 5] {key:7s} baseline: CESM2 {cesm_base[key]:.3f}, "
          f"emulator {emu_base[key]:.3f}"
          + ("   (inherited from hist)" if key == "ssp370" else ""))
# The two numbers differ by ~273 because the tree stores kelvin and the eval
# writes degrees Celsius. That never matters: each side is only ever compared
# with its own baseline subtracted, and a difference of anomalies is unitless
# in that respect.

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
    ref = cesm[key]
    ref = ref[ref.index <= YEAR_MAX]
    ref_anom = ref - cesm_base[key]                 # year x member
    ref_mean = ref_anom.mean(axis=1)

    lo, hi = ref_anom.min(axis=1), ref_anom.max(axis=1)
    ax_main.fill_between(ref_anom.index, lo, hi, color=colour, alpha=0.26,
                         lw=0, zorder=1)
    for edge in (lo, hi):                           # a thin edge pins the band
        ax_main.plot(ref_anom.index, edge, color=colour, lw=0.7, alpha=0.55,
                     zorder=1)
    ax_main.plot(ref_anom.index, ref_mean.values, color=colour, lw=1.2, ls="--",
                 marker="o", markersize=3.4, markevery=8, markerfacecolor="white",
                 markeredgecolor=colour, zorder=5,
                 path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

    years, members = emulator[key]
    keep = years <= YEAR_MAX
    emu_anom = members[:, keep] - emu_base[key]
    ax_main.plot(years[keep], emu_anom.mean(axis=0), color=colour, lw=2.6,
                 zorder=4, label=label)

    # ── the numbers, on the years both sides cover ───────────────────────────
    common = np.intersect1d(years[keep], ref_anom.index.values)
    e = pd.Series(emu_anom.mean(axis=0), index=years[keep]).loc[common]
    c = ref_mean.loc[common]
    difference = e - c
    # sigma of CESM2's members about their own mean, per year: the internal
    # variability a single realization shows by chance.
    sigma = ref_anom.loc[common].sub(c, axis=0).std(axis=1)
    bias_series[key] = (common, difference, sigma)
    stats[key] = dict(
        n_emu=emu_anom.shape[0], n_cesm=ref_anom.shape[1],
        bias=float(difference.mean()),
        rmse=float(np.sqrt((difference ** 2).mean())),
        inside=float((difference.abs() <= 2 * sigma).mean()) * 100)
    print(f"[step 7] {key:7s} bias {stats[key]['bias']:+.3f} degC, "
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
        ax.fill_between(years, -2 * sigma.values, 2 * sigma.values,
                        color="0.45", alpha=0.22, lw=0, zorder=0)
        ax.plot(years, difference.values, color=SCENARIOS[key][2], lw=1.4, zorder=3)
    ax.set_title(f"{title}\nn = "
                 f"{'/'.join(str(stats[k]['n_emu']) for k in group)} emulator, "
                 f"{'/'.join(str(stats[k]['n_cesm']) for k in group)} CESM2",
                 fontsize=9, loc="left", pad=4)
    ax.text(0.02, 0.94, f"({'bcd'[i]})", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=9)
    ax.text(0.02, 0.04, "\n".join(
        f"{SCENARIOS[k][0].split(' (')[0]}: {stats[k]['bias']:+.2f} "
        f"+/- {stats[k]['rmse']:.2f} degC, {stats[k]['inside']:.0f}% in band"
        for k in group), transform=ax.transAxes, fontsize=7.2, va="bottom",
        color="0.25")
    ax.set_xlabel("Year")
    if i == 0:
        ax.set_ylabel("Bias (degC)\nensemble means")
    else:
        # All three panels share one y-scale (set below), so repeating the tick
        # labels adds nothing and collides with the neighbouring panel's text.
        ax.tick_params(labelleft=False)

# every bias panel on one y-scale, so their sizes are comparable by eye
limit = 1.15 * max(max(abs(d.min()), abs(d.max()), 2 * s.max())
                   for _, d, s in bias_series.values())
for ax in ax_bias:
    ax.set_ylim(-limit, limit)

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
ax_main.set_ylabel("GMST anomaly (°C, vs 1850–1900)")
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

print("\nEmulator vs held-out CESM2 (degC, years both sides cover)")
print(pd.DataFrame(stats).T.round(3).to_string())
