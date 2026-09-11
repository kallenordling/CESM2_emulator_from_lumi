#!/usr/bin/env python3
"""
================================================================================
 FIGURES 11 AND 12 — THE NONLINEARITY N AS A MAP, LAST 20 YEARS
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig1314_maps.py

Everything configurable is in the SETTINGS block below. No command-line options
and no helper functions except the two statistical ones, which are shared with
scripts/paper_fig_maps.py and repeated here rather than imported so this script
stays readable top to bottom.

WHAT THIS IS
------------
The map twin of scripts/make_fig1112_from_csv.py. That figure shows

    N(t) = Delta(ALL)(t) - Delta(GHG)(t) - Delta(AAER)(t)

as a global-mean time series; this one shows the same quantity as a map,
averaged over the LAST 20 YEARS of the single-forcing runs.

IT CANNOT USE THE CSVs. Those hold global means only — one number per year per
member — so this script goes back to the gridded NetCDFs: the evaluation output
for the emulator and the cesm2_reference files for CESM2, exactly the two
directories scripts/make_fig12_csv.py read before reducing them.

THE WINDOW
----------
2031-2050: the last 20 years the single-forcing runs cover, and entirely
post-2015, so no prescribed volcanic eruption falls inside it. That matters
because N IS NOT A PURE INTERACTION TERM — the single-forcing runs hold ozone,
land use, biomass burning, solar and volcanic forcing at 1850 levels, so

    N = (forcings absent from both single-forcing runs) + (nonlinearity)

Before 2015 the eruptions dominate it outright. Even in a clean window the
Amazon and Congo hotspots are land-use and biomass-burning signal, not
interaction. Note the window differs from the 2041-2050 of figures 11 and 12,
which took the last clean DECADE; this takes the last 20 years as asked, and
the two windows give the same picture with slightly different amplitudes.

WHY mm/day AND NOT PERCENT
--------------------------
Precipitation maps stay in mm/day even though figures 2, 4 and 10 use percent.
A percentage residual divides by the 1850-1900 climatology, which over the
subtropical deserts is near zero, and the quotient there swamps every real
feature on the map. The same decision, for the same reason, as
scripts/paper_fig_attribution.py.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# The same two directories scripts/make_fig12_csv.py reads.
EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"
REFERENCE_DIR = "/home/nordling/mnt/lumi_sc/emulator_data/cesm2_reference"

FIGURE_NAME = {"TREFHT": "fig13", "PRECT": "fig14"}

# A polar figure, for TEMPERATURE ONLY. The Arctic carries the largest
# nonlinearity anywhere on the map (+1.5 K against a global +0.47) and Robinson
# squeezes it into a thin strip at the top edge, so the one region the reader
# most needs to see is the one that projection shows worst. Precipitation gets
# no polar figure: its residual is a tropical signal, concentrated in the ITCZ
# and the monsoon regions, and the poles have essentially none of it.
POLAR_VARIABLE = "TREFHT"
POLAR_FIGURE_NAME = "fig15"
# Equatorward edge of each polar panel, in degrees latitude.
POLAR_EDGE = 50.0
OUT = "plots/{name}/{name}.png"          # the .pdf sibling is written alongside
TABLE = "plots/{name}/{name}_stats.tex"

BASELINE = (1850, 1900)

# The last 20 years of the single-forcing runs, which end in 2050.
WINDOW = (2031, 2050)

# Reading eight files of (member, year, 192, 288) over the sshfs mount takes
# about five minutes; the maps they reduce to are a few MB. They are cached
# here so that iterating on the figure costs seconds. DELETE THIS FILE after
# re-running an evaluation, or after changing WINDOW or BASELINE — it is keyed
# on neither.
CACHE = "plots/fig1314_maps_cache.npz"

# Cap the emulator at the CESM2 member count per experiment, as the CSV export
# and every figure built on it do. Set False to use all 25 emulator members —
# which TIGHTENS the emulator's side of every interval and so makes MORE of the
# difference map significant. It does not make the emulator better.
MATCH_MEMBER_COUNTS = True

# Point-wise significance on the difference panel, then Benjamini-Hochberg at
# q = 2*ALPHA across the map, matching scripts/paper_fig_maps.py. Hatching marks
# where the emulator's N differs from CESM2's beyond noise — MORE hatching is a
# stronger detection of disagreement, not a better score.
ALPHA = 0.05

# Variables:
#   label     -> the noun used in captions
#   unit      -> matplotlib text for the colour-bar label
#   unit_tex  -> the same unit, typeset, for the LaTeX table
#   cmap      -> diverging, centred on zero
VARIABLES = {
    "TREFHT": ("Temperature", "$^{\\circ}$C", "$^{\\circ}$C", "RdBu_r"),
    "PRECT":  ("Precipitation", "mm day$^{-1}$", "mm\\,day$^{-1}$", "BrBG"),
}

# The three ingredients of N. "all" is built in step 2 from hist and ssp370.
INGREDIENTS = ("all", "ghg", "aaer")

# Named regions for the table: (label, lat_min, lat_max, lon_min, lon_max),
# longitudes in the files' own 0-360 convention. The same regions
# paper_fig_attribution.py reports, so the two are comparable.
REGIONS = [
    ("Arctic",          66.5,  90.0,    0.0, 360.0),
    ("N. Atlantic",     45.0,  65.0,  300.0, 350.0),
    ("N. America",      30.0,  60.0,  230.0, 300.0),
    ("Europe",          35.0,  65.0,    0.0,  40.0),
    ("East Asia",       20.0,  50.0,  100.0, 145.0),
    ("South Asia",       5.0,  30.0,   65.0, 100.0),
    ("Sahel",           10.0,  20.0,    0.0,  40.0),
    ("Tropics",        -20.0,  20.0,    0.0, 360.0),
    ("Antarctic",      -90.0, -66.5,    0.0, 360.0),
]

# =============================================================================

import os
import sys

import numpy as np
import xarray as xr
import scipy.stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cartopy is NOT optional here. Without it the panels lose the Robinson
# projection AND the coastlines, and the figure that gets written looks
# plausible enough to end up in the paper. paper_fig_maps.py learned this the
# hard way — it once overwrote a good figure with flat lat/lon panels.
try:
    import cartopy.crs as ccrs
    import matplotlib.path as mpath
except ImportError:
    sys.exit("[error] cartopy is not available in this interpreter — the maps "
             "would be written without a projection or coastlines.\n"
             "        Run with /home/nordling/miniconda3/envs/plotting/bin/python")

# =============================================================================
#  STEP 1 — read the gridded fields
# =============================================================================
# Both sides carry (member, year, lat, lon) in the emulator's units, degrees
# Celsius and mm/day — cesm2_reference was built that way so nothing downstream
# has to remember which side is which.
#
# Reduced immediately to TWO maps per member: the window mean and the 1850-1900
# baseline. Everything after this works on (member, lat, lon), which is small.

window_mean, baseline_mean, coords = {}, {}, {}

if os.path.exists(CACHE):
    # The keys are flattened to strings for npz, and unflattened here.
    stored = np.load(CACHE, allow_pickle=False)
    for key in stored.files:
        kind, variable, scenario, side = key.split("|")
        if kind in ("lat", "lon"):
            continue
        target = window_mean if kind == "window" else baseline_mean
        value = stored[key]
        # A zero-length array is how "this scenario does not cover that period"
        # survives the round trip: hist has no 2031-2050 and ssp370 no
        # 1850-1900.
        target[(variable, scenario, side)] = None if value.size == 0 else value
    for variable in VARIABLES:
        coords[variable] = (stored[f"lat|{variable}||"], stored[f"lon|{variable}||"])
    print(f"[step 1] read {CACHE} — delete it to re-read the NetCDFs")
else:
    for variable in VARIABLES:
        for scenario in ("hist", "ssp370", "ghg", "aaer"):
            for side, directory, field_name in (
                    ("emulator", EVAL_DIR, f"{variable}_model"),
                    ("cesm2", REFERENCE_DIR, f"{variable}_cesm")):
                path = f"{directory}/{variable}_{scenario}.nc"
                dataset = xr.open_dataset(path)
                field = dataset[field_name]

                # A scenario need not cover both periods: hist ends in 2014 and
                # has no window, ssp370 starts in 2015 and has no baseline.
                # Both are stored as None and resolved in step 2.
                in_window = field.sel(year=slice(*WINDOW))
                in_baseline = field.sel(year=slice(*BASELINE))
                window_mean[(variable, scenario, side)] = (
                    in_window.mean("year").values
                    if in_window.sizes["year"] else None)
                baseline_mean[(variable, scenario, side)] = (
                    in_baseline.mean("year").values
                    if in_baseline.sizes["year"] else None)
                coords[variable] = (field["lat"].values, field["lon"].values)
                print(f"[step 1] {variable:6s} {scenario:7s} {side:8s} "
                      f"{field.sizes['member']} members, "
                      f"window {in_window.sizes['year']} yr, "
                      f"baseline {in_baseline.sizes['year']} yr", flush=True)
                dataset.close()

    payload = {}
    for (variable, scenario, side), value in window_mean.items():
        payload[f"window|{variable}|{scenario}|{side}"] = (
            np.empty(0) if value is None else value)
    for (variable, scenario, side), value in baseline_mean.items():
        payload[f"baseline|{variable}|{scenario}|{side}"] = (
            np.empty(0) if value is None else value)
    for variable, (lat, lon) in coords.items():
        payload[f"lat|{variable}||"], payload[f"lon|{variable}||"] = lat, lon
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print(f"[step 1] wrote {CACHE}")

# The emulator is capped to the CESM2 member count, per experiment, exactly as
# the CSV export does — so figures 13 and 14 rest on the same ensembles as 9
# and 10 and the two can be quoted side by side. The count is read from
# whichever period that scenario actually covers.
if MATCH_MEMBER_COUNTS:
    for variable in VARIABLES:
        for scenario in ("hist", "ssp370", "ghg", "aaer"):
            reference = (window_mean[(variable, scenario, "cesm2")]
                         if window_mean[(variable, scenario, "cesm2")] is not None
                         else baseline_mean[(variable, scenario, "cesm2")])
            n_cesm = len(reference)
            for store in (window_mean, baseline_mean):
                block = store[(variable, scenario, "emulator")]
                if block is not None and len(block) > n_cesm:
                    store[(variable, scenario, "emulator")] = block[:n_cesm]
            print(f"[step 1] {variable:6s} {scenario:7s} emulator capped to "
                  f"{n_cesm} members")

# =============================================================================
#  STEP 2 — the all-forcing ingredient
# =============================================================================
# ALL is the historical run continued by ssp370. The window is 2031-2050, so it
# lies ENTIRELY inside ssp370 and no splicing of the window itself is needed —
# but ssp370 begins in 2015 and has no pre-industrial of its own, so its
# BASELINE has to come from hist. That is the whole splice, and it is why the
# two are read at all.
#
# hist and ssp370 carry the same members in the same order on both sides (the
# CSV export checks the names; here the arrays are positional), so member i of
# the baseline is member i of the window.

for variable in VARIABLES:
    for side in ("emulator", "cesm2"):
        window_mean[(variable, "all", side)] = window_mean[(variable, "ssp370", side)]
        baseline_mean[(variable, "all", side)] = baseline_mean[(variable, "hist", side)]
        n_window = len(window_mean[(variable, "all", side)])
        n_baseline = len(baseline_mean[(variable, "all", side)])
        if n_window != n_baseline:
            sys.exit(f"[error] {variable}/{side}: ssp370 has {n_window} members "
                     f"but hist has {n_baseline} — the per-member baseline "
                     f"would pair unrelated runs")
        print(f"[step 2] {variable:6s} {side:8s} ALL = ssp370 {WINDOW[0]}-{WINDOW[1]} "
              f"on a hist {BASELINE[0]}-{BASELINE[1]} baseline, {n_window} members")

# =============================================================================
#  STEP 3 — anomalies, N, and its uncertainty
# =============================================================================
# Per member: its window map minus its OWN 1850-1900 map. Each side is
# referenced to its own pre-industrial, so the constant offset between the
# emulator's mean state and CESM2's is gone before any subtraction and N is a
# difference of RESPONSES.
#
# N combines three ensemble means, so at every grid point three sampling
# variances add. The degrees of freedom come from Satterthwaite — Welch's
# approximation for two groups, written out for three.

n_map, variance_map, group_variance, group_n = {}, {}, {}, {}
for variable in VARIABLES:
    for side in ("emulator", "cesm2"):
        means, variances, n = {}, {}, {}
        for key in INGREDIENTS:
            anomaly = (window_mean[(variable, key, side)]
                       - baseline_mean[(variable, key, side)])   # (member, lat, lon)
            means[key] = anomaly.mean(axis=0)
            variances[key] = anomaly.var(axis=0, ddof=1) / len(anomaly)
            n[key] = len(anomaly)

        n_map[(variable, side)] = means["all"] - means["ghg"] - means["aaer"]
        variance_map[(variable, side)] = sum(variances[k] for k in INGREDIENTS)
        group_variance[(variable, side)] = variances
        group_n[(variable, side)] = n
        print(f"[step 3] {variable:6s} {side:8s} N map built, n = "
              + "/".join(f"{k} {n[k]}" for k in INGREDIENTS))

# =============================================================================
#  STEP 4 — where the two Ns differ, with the false-discovery rate controlled
# =============================================================================
# A point-wise Welch test on the DIFFERENCE of the two residuals, whose variance
# is the sum of all six groups'. Then Benjamini-Hochberg at q = 2*ALPHA over the
# whole map, because 55296 grid points tested at p < 0.05 would produce ~2700
# false rejections on their own.
#
# READ THE HATCHING IN THE RIGHT DIRECTION: it marks where the emulator's
# nonlinearity differs from CESM2's beyond noise. MORE hatching is a stronger
# detection of disagreement, not a better score.

difference_map, significant, threshold, pattern_r = {}, {}, {}, {}
for variable in VARIABLES:
    difference = n_map[(variable, "emulator")] - n_map[(variable, "cesm2")]
    variance = variance_map[(variable, "emulator")] + variance_map[(variable, "cesm2")]

    degrees_freedom = variance ** 2 / sum(
        group_variance[(variable, side)][key] ** 2
        / (group_n[(variable, side)][key] - 1)
        for side in ("emulator", "cesm2") for key in INGREDIENTS)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_statistic = difference / np.sqrt(variance)
        p_value = 2.0 * sstats.t.sf(np.abs(t_statistic), degrees_freedom)

    # Benjamini-Hochberg, inline: sort the p-values, find the largest one that
    # falls under its own rank's threshold, reject everything at or below it.
    finite = p_value[np.isfinite(p_value)]
    ordered = np.sort(finite)
    below = ordered <= (np.arange(1, ordered.size + 1) / ordered.size) * (2.0 * ALPHA)
    if below.any():
        p_threshold = float(ordered[np.nonzero(below)[0].max()])
        mask = np.isfinite(p_value) & (p_value <= p_threshold)
    else:
        p_threshold, mask = 0.0, np.zeros(p_value.shape, dtype=bool)

    # Area-weighted pattern correlation of the two N maps. This is the number
    # that separates "learned the pattern, overstated the amplitude" from "did
    # not learn the pattern" — the distinction the RAMIP comparison turns on.
    lat, lon = coords[variable]
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None],
                              n_map[(variable, "cesm2")].shape).ravel()
    a = n_map[(variable, "cesm2")].ravel()
    b = n_map[(variable, "emulator")].ravel()
    a_centred = a - np.average(a, weights=weights)
    b_centred = b - np.average(b, weights=weights)
    pattern_r[variable] = float(
        np.average(a_centred * b_centred, weights=weights)
        / np.sqrt(np.average(a_centred ** 2, weights=weights)
                  * np.average(b_centred ** 2, weights=weights)))

    difference_map[variable] = difference
    significant[variable] = mask
    threshold[variable] = p_threshold
    area_significant = 100 * float(np.average(mask.astype(float),
                                              weights=weights.reshape(mask.shape)))
    print(f"[step 4] {variable:6s} pattern r = {pattern_r[variable]:.3f}, "
          f"{area_significant:.1f}% of area significant "
          f"(BH q = {2 * ALPHA:g}, p <= {p_threshold:.2e})")

# =============================================================================
#  STEP 5 — the figure: three panels
# =============================================================================
# CESM2's N, the emulator's N, and the difference. The first two SHARE a colour
# scale, because they are the same quantity and the reader is meant to compare
# them directly; the difference gets its own, since it is several times smaller
# and would be invisible on theirs.

# hatch.linewidth is a global rcParam, not a contourf argument: the default
# 1.0 prints as a grey smear at this panel size and 0.45 stays legible.
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "hatch.linewidth": 0.45})

table_rows = {}
for variable, (label, unit, unit_tex, cmap) in VARIABLES.items():
    figure_name = FIGURE_NAME[variable]
    lat, lon = coords[variable]
    weights = np.cos(np.deg2rad(lat))[:, None]

    # The panels have a FIXED aspect ratio (Robinson is a projection, not a
    # stretchable box), so a figure taller than three panels need cannot be
    # filled — the axes simply centre themselves and leave sky above. 13.2/3 is
    # a 4.4-inch panel, ~2.2 inches tall, plus a colour bar and a title.
    fig = plt.figure(figsize=(13.2, 3.1))
    projection = ccrs.Robinson(central_longitude=0)
    axes = [fig.add_subplot(1, 3, i + 1, projection=projection) for i in range(3)]

    # 99th percentile, not the maximum: a handful of polar grid points would
    # otherwise set a scale on which the rest of the map is blank. extend="both"
    # carries the tail honestly.
    shared_max = float(np.nanpercentile(
        np.abs(np.concatenate([n_map[(variable, "cesm2")].ravel(),
                               n_map[(variable, "emulator")].ravel()])), 99))
    difference_max = float(np.nanpercentile(np.abs(difference_map[variable]), 99))

    panels = [
        ("(a) CESM2", n_map[(variable, "cesm2")], shared_max, None),
        ("(b) Emulator", n_map[(variable, "emulator")], shared_max, None),
        ("(c) Emulator $-$ CESM2", difference_map[variable], difference_max,
         significant[variable]),
    ]

    images = []
    for ax, (title, field, vmax, mask) in zip(axes, panels):
        image = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=-vmax, vmax=vmax,
                              shading="auto", transform=ccrs.PlateCarree())
        if mask is not None:
            # SHIFTED BY ONE, and it matters: `hatches` maps to the INTERVALS
            # between `levels`, so a raw 0/1 mask puts every point in the first
            # interval and nothing is ever hatched. False -> 1.0 lands in
            # [0.5, 1.5) and gets "", True -> 2.0 lands in [1.5, 2.5) and gets
            # the dots. Same convention as paper_fig_maps.py.
            ax.contourf(lon, lat, mask.astype(float) + 1.0,
                        levels=[0.5, 1.5, 2.5],
                        colors="none", hatches=["", "...."],
                        transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.35, color="0.25")
        ax.set_global()
        ax.set_title(title, fontsize=10, loc="left", pad=5)
        images.append(image)

    # Global means, printed under each panel: the map says where, this says how
    # much, and the two together are what the caption quotes.
    for ax, (_, field, _, _) in zip(axes, panels):
        ax.text(0.5, -0.10, f"global mean {np.average(field, weights=np.broadcast_to(weights, field.shape)):+.3f} {unit}",
                transform=ax.transAxes, ha="center", fontsize=8.6)

    fig.colorbar(images[0], ax=axes[:2], orientation="horizontal",
                 fraction=0.055, pad=0.09, aspect=45, extend="both",
                 label=f"N = ALL $-$ GHG $-$ AAER ({unit})")
    fig.colorbar(images[2], ax=[axes[2]], orientation="horizontal",
                 fraction=0.055, pad=0.09, aspect=22, extend="both",
                 label=f"Difference in N ({unit})")

    fig.suptitle(
        f"{label}: the nonlinearity N = ALL $-$ GHG $-$ AAER, "
        f"{WINDOW[0]}–{WINDOW[1]}     "
        f"pattern correlation r = {pattern_r[variable]:.3f}",
        fontsize=11, y=0.99)

    out_path = OUT.format(name=figure_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for path in (out_path, os.path.splitext(out_path)[0] + ".pdf"):
        fig.savefig(path, bbox_inches="tight")
        print(f"[step 5] wrote {path}")
    plt.close(fig)

    # =========================================================================
    #  STEP 6 — the regional numbers
    # =========================================================================
    # Area-weighted means of both Ns over named regions, so the paper can say
    # WHERE the residual lives rather than only how big it is globally.

    rows = []
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    for name, lat_min, lat_max, lon_min, lon_max in REGIONS:
        inside = ((lat_grid >= lat_min) & (lat_grid <= lat_max)
                  & (lon_grid >= lon_min) & (lon_grid <= lon_max))
        region_weights = np.where(inside, np.broadcast_to(weights, inside.shape), 0.0)
        cesm_value = float(np.average(n_map[(variable, "cesm2")], weights=region_weights))
        emulator_value = float(np.average(n_map[(variable, "emulator")],
                                          weights=region_weights))
        fraction = 100 * float(np.average(significant[variable].astype(float),
                                          weights=region_weights))
        rows.append((name, cesm_value, emulator_value, fraction))
        print(f"[step 6] {variable:6s} {name:14s} CESM2 {cesm_value:+.3f}, "
              f"emulator {emulator_value:+.3f}, {fraction:.0f}% significant")
    table_rows[variable] = rows

# =============================================================================
#  STEP 7 — the same numbers as LaTeX tables
# =============================================================================
# A COMPLETE `table` float — caption, label and tabular — so \input drops it
# straight into the paper, matching fig01-fig12. Plain LaTeX, no booktabs.

for variable, (label, _, unit_tex, _) in VARIABLES.items():
    figure_name = FIGURE_NAME[variable]
    lat, lon = coords[variable]
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None],
                              n_map[(variable, "cesm2")].shape)
    global_cesm = float(np.average(n_map[(variable, "cesm2")], weights=weights))
    global_emulator = float(np.average(n_map[(variable, "emulator")], weights=weights))
    area_significant = 100 * float(np.average(
        significant[variable].astype(float), weights=weights))

    rows_tex = [f"Global & {global_cesm:+.3f} & {global_emulator:+.3f} & "
                f"{global_emulator - global_cesm:+.3f} & "
                f"{area_significant:.0f} \\\\", r"\hline"]
    for name, cesm_value, emulator_value, fraction in table_rows[variable]:
        rows_tex.append(
            f"{name} & {cesm_value:+.3f} & {emulator_value:+.3f} & "
            f"{emulator_value - cesm_value:+.3f} & {fraction:.0f} \\\\")

    caption = (
        f"The nonlinearity $N = \\Delta(\\mathrm{{ALL}}) - "
        f"\\Delta(\\mathrm{{GHG}}) - \\Delta(\\mathrm{{AAER}})$ in "
        f"{label.lower()}, averaged over {WINDOW[0]}--{WINDOW[1]} --- the last "
        f"20 years the single-forcing runs cover --- as area-weighted means "
        f"over named regions. $\\Delta$ is the anomaly relative to each side's "
        f"own 1850--1900 mean, and $N$ is computed separately on each side, so "
        f"neither is a reference for the other. Units are "
        f"{unit_tex}; precipitation is left in "
        f"mm\\,day$^{{-1}}$ rather than converted to a percentage because a "
        f"percentage residual divides by a near-zero climatology over the "
        f"subtropical deserts. The area-weighted pattern correlation between "
        f"the two maps is $r = {pattern_r[variable]:.3f}$: the emulator "
        f"reproduces where the nonlinearity is, and the ``difference'' column "
        f"says by how much it overstates it. ``Significant'' is the percentage "
        f"of the region's area where the two differ at $p < {ALPHA:g}$, "
        f"point-wise, with the false-discovery rate controlled at "
        f"$q = {2 * ALPHA:g}$ across the whole map --- so MORE of it means a "
        f"stronger detection of disagreement, not a better emulator. "
        f"NOTE THAT $N$ IS NOT A PURE INTERACTION TERM: the single-forcing "
        f"runs hold ozone, land use, biomass burning, solar and volcanic "
        f"forcing at 1850 levels, so $N$ = (those forcings) + "
        f"(nonlinearity), and the Amazon and Congo signals in particular are "
        f"land use and biomass burning.")

    table_tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}",
        r"\label{tab:" + figure_name + "}",
        r"\begin{tabular}{|l|r|r|r|r|}",
        r"\hline",
        r"\textbf{Region} & \textbf{CESM2} & \textbf{Emulator} & "
        r"\textbf{Difference} & \textbf{Significant} \\",
        r" & \multicolumn{3}{c|}{(" + unit_tex + r")} & (\%) \\",
        r"\hline",
        *rows_tex,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_path = TABLE.format(name=figure_name)
    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
    with open(table_path, "w") as handle:
        handle.write(
            f"% {label}: the nonlinearity N = ALL - GHG - AAER as a map, "
            f"{WINDOW[0]}-{WINDOW[1]}.\n"
            f"% Built from the gridded NetCDFs by scripts/make_fig1314_maps.py\n"
            "% — do not edit by hand. The caption below says what the numbers\n"
            "% are; \\input this file directly.\n")
        handle.write(table_tex + "\n")
    print(f"[step 7] wrote {table_path}")

# =============================================================================
#  STEP 8 — the polar figure (temperature only)
# =============================================================================
# The same three panels as figure 11, twice: the Arctic on top and the
# Antarctic below, in polar stereographic projections that give each pole its
# real area instead of the smear Robinson makes of it.
#
# THE COLOUR SCALE IS RECOMPUTED OVER THE POLAR DOMAIN, not inherited from
# figure 11. Sharing it would make these panels comparable to that one at the
# cost of making them unreadable in themselves — the global 99th percentile is
# set by the tropics, where N is small, so on that scale both poles saturate.
# The global means printed under each panel are the domain's own, so the two
# figures can still be reconciled by number where they cannot by colour.

variable = POLAR_VARIABLE
label, unit, unit_tex, cmap = VARIABLES[variable]
lat, lon = coords[variable]
weights = np.cos(np.deg2rad(lat))[:, None]

hemispheres = [
    ("Arctic", ccrs.NorthPolarStereo(), (-180, 180, POLAR_EDGE, 90),
     lat >= POLAR_EDGE),
    ("Antarctic", ccrs.SouthPolarStereo(), (-180, 180, -90, -POLAR_EDGE),
     lat <= -POLAR_EDGE),
]

fig = plt.figure(figsize=(9.6, 8.2))
# An explicit gridspec, not plt.subplot's default spacing: each row carries a
# colour bar UNDER it and a two-line title ABOVE it, and at the default hspace
# the first row's colour-bar label lands on the second row's titles.
polar_grid = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.10)

# The panels this figure needs, in the order they are drawn.
columns = [
    ("CESM2", n_map[(variable, "cesm2")], None),
    ("Emulator", n_map[(variable, "emulator")], None),
    ("Emulator $-$ CESM2", difference_map[variable], significant[variable]),
]

# HOLD THE MAP AXES EXPLICITLY. `fig.axes` grows every time a colour bar is
# added, so indexing into it puts the second row's bars underneath the first
# row's panels — the indices no longer mean what they meant when the loop
# started.
map_axes = {}
images = {}
for row, (hemisphere, projection, extent, band) in enumerate(hemispheres):
    # Scales are per HEMISPHERE, and the two N panels share theirs, so the
    # emulator's map can be read against CESM2's directly. The Arctic residual
    # is an order of magnitude larger than the Antarctic one, so a single scale
    # for the figure would leave the bottom row blank.
    shared_max = float(np.nanpercentile(
        np.abs(np.concatenate([n_map[(variable, "cesm2")][band].ravel(),
                               n_map[(variable, "emulator")][band].ravel()])), 99))
    difference_max = float(np.nanpercentile(
        np.abs(difference_map[variable][band]), 99))

    band_mask = np.broadcast_to(band[:, None], n_map[(variable, "cesm2")].shape)
    band_weights = np.where(band_mask,
                            np.broadcast_to(weights, band_mask.shape), 0.0)

    for column, (title, field, mask) in enumerate(columns):
        ax = fig.add_subplot(polar_grid[row, column], projection=projection)
        ax.set_extent(extent, ccrs.PlateCarree())

        # Without this the panel is a SQUARE with the pole in the middle and
        # the corners filled by whatever lies beyond the extent. The circular
        # boundary is what makes a polar panel read as a polar panel.
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.set_boundary(mpath.Path(np.column_stack(
            [0.5 + 0.5 * np.sin(theta), 0.5 + 0.5 * np.cos(theta)])),
            transform=ax.transAxes)

        vmax = difference_max if mask is not None else shared_max
        image = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=-vmax, vmax=vmax,
                              shading="auto", transform=ccrs.PlateCarree())
        if mask is not None:
            ax.contourf(lon, lat, mask.astype(float) + 1.0,
                        levels=[0.5, 1.5, 2.5], colors="none",
                        hatches=["", "...."], transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.35, color="0.25")
        ax.gridlines(linewidth=0.3, color="0.6", alpha=0.5)

        # The domain mean goes in the TITLE, not under the panel: a circular
        # boundary leaves no room beneath it before the colour bar begins.
        domain_mean = float(np.average(field, weights=band_weights))
        ax.set_title(f"({'abcdef'[row * 3 + column]}) {title}\n"
                     f"{hemisphere} mean {domain_mean:+.3f} {unit}",
                     fontsize=9.5, pad=6)

        map_axes[(row, column)] = ax
        images[(row, column)] = image

    # One pair of colour bars per ROW, attached to THIS row's axes by name.
    fig.colorbar(images[(row, 0)],
                 ax=[map_axes[(row, 0)], map_axes[(row, 1)]],
                 orientation="horizontal", fraction=0.048, pad=0.04, aspect=34,
                 extend="both", label=f"N ({unit})")
    fig.colorbar(images[(row, 2)], ax=[map_axes[(row, 2)]],
                 orientation="horizontal", fraction=0.048, pad=0.04, aspect=17,
                 extend="both", label=f"Difference ({unit})")

    map_axes[(row, 0)].text(
        -0.10, 0.5, f"{hemisphere}\npoleward of {POLAR_EDGE:.0f}$^\\circ$",
        transform=map_axes[(row, 0)].transAxes, rotation=90, va="center",
        ha="center", fontsize=10)

fig.suptitle(
    f"{label}: the nonlinearity N = ALL $-$ GHG $-$ AAER at the poles, "
    f"{WINDOW[0]}–{WINDOW[1]}", fontsize=11, y=0.98)

out_path = OUT.format(name=POLAR_FIGURE_NAME)
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
for path in (out_path, os.path.splitext(out_path)[0] + ".pdf"):
    fig.savefig(path, bbox_inches="tight")
    print(f"[step 8] wrote {path}")
plt.close(fig)

for hemisphere, band in (("Arctic", lat >= POLAR_EDGE),
                         ("Antarctic", lat <= -POLAR_EDGE)):
    band_weights = np.where(
        np.broadcast_to(band[:, None], n_map[(variable, "cesm2")].shape),
        np.broadcast_to(weights, n_map[(variable, "cesm2")].shape), 0.0)
    cesm_value = float(np.average(n_map[(variable, "cesm2")], weights=band_weights))
    emulator_value = float(np.average(n_map[(variable, "emulator")],
                                      weights=band_weights))
    fraction = 100 * float(np.average(significant[variable].astype(float),
                                      weights=band_weights))
    print(f"[step 8] {hemisphere:10s} poleward of {POLAR_EDGE:.0f}: "
          f"CESM2 {cesm_value:+.3f}, emulator {emulator_value:+.3f}, "
          f"{fraction:.0f}% significant")
