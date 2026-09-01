#!/usr/bin/env python3
"""
================================================================================
 FIGURES 7 AND 8 — are the RUNNING MEANS the same?
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig7.py

Everything configurable is in the SETTINGS block below. No command-line options
and no helper functions: seven numbered steps, top to bottom. Same shape as
make_fig1.py, make_fig3.py and make_fig5.py.

THE QUESTION
------------
A running mean averages weather away and leaves the forced response, so
comparing running means asks whether the emulator follows the same TRAJECTORY as
CESM2 — not whether it matches year by year, which no ensemble member does
either.

HOW IT IS TESTED, AND WHY THIS WAY
----------------------------------
At each year, Welch's t-test across MEMBERS: the running means of the emulator's
members against the running means of CESM2's. Members are independent
realizations, so at any single year those are independent samples and the test
is valid.

WHAT THAT TEST IS NOT. The years are emphatically NOT independent of each other
— a W-year running mean makes neighbouring years share W-1 of their W values by
construction, on top of the autocorrelation the climate already has. So the
per-year p-values are a profile showing WHERE the trajectories separate, not a
set of independent tests to be counted or corrected. The single summary number
this script reports is the fraction of years that reject, and it should be read
as "how much of the record differs", never as a multiple-comparison result.

A running mean also cannot be computed for the first and last W//2 years, so the
curves are shorter than the record at both ends.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"
REFERENCE_DIR = "/home/nordling/mnt/lumi_sc/emulator_data/cesm2_reference"

FIGURE_NAME = {"TREFHT": "fig07", "PRECT": "fig08"}
OUT = "plots/{name}.png"               # the .pdf sibling is written alongside
TABLE = "plots/{name}_running.tex"

# Width of the running mean, in years. Odd, so the window is centred.
WINDOW = 21

# Cap the emulator at the CESM2 member count, per experiment, so every
# comparison is n against n.
MATCH_MEMBER_COUNTS = True

# Significance level for the per-year test.
ALPHA = 0.05

VARIABLES = {"TREFHT": ("Temperature", "$^{\\circ}$C", "degC"),
             "PRECT":  ("Precipitation", "mm day$^{-1}$", "mm/day")}

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import xarray as xr
from scipy import stats

# =============================================================================
#  STEP 1 — global-mean series for both sides, every year
# =============================================================================
# cos(lat)-weighted, computed here from the maps: grid cells shrink towards the
# poles and an unweighted mean over this grid is wrong by degrees.

series = {}            # (side, variable, scenario) -> (years, (member, year))
for variable in VARIABLES:
    for scenario in SCENARIOS:
        for side, directory, field_name in (
                ("emulator", EVAL_DIR, f"{variable}_model"),
                ("cesm", REFERENCE_DIR, f"{variable}_cesm")):
            dataset = xr.open_dataset(f"{directory}/{variable}_{scenario}.nc")
            field = dataset[field_name]
            weights = np.cos(np.deg2rad(field["lat"]))
            global_mean = field.weighted(weights).mean(("lat", "lon")).compute()
            series[(side, variable, scenario)] = (global_mean["year"].values,
                                                  global_mean.values)
            dataset.close()
        print(f"[step 1] {variable:6s} {scenario:7s} "
              f"emulator {series[('emulator', variable, scenario)][1].shape}, "
              f"CESM2 {series[('cesm', variable, scenario)][1].shape}")

# =============================================================================
#  STEP 2 — same n on both sides
# =============================================================================

if MATCH_MEMBER_COUNTS:
    for variable in VARIABLES:
        for scenario in SCENARIOS:
            years, emulator_values = series[("emulator", variable, scenario)]
            n_cesm = series[("cesm", variable, scenario)][1].shape[0]
            if emulator_values.shape[0] > n_cesm:
                series[("emulator", variable, scenario)] = (years,
                                                            emulator_values[:n_cesm])
                print(f"[step 2] {variable:6s} {scenario:7s} emulator "
                      f"{emulator_values.shape[0]} -> {n_cesm} members")

# =============================================================================
#  STEP 3 — running mean, per member
# =============================================================================
# Per MEMBER, not on the ensemble mean: the test in step 4 needs the spread
# across members at each year, which averaging first would destroy.
#
# mode="valid" drops the ends rather than padding them; a padded running mean
# is a different quantity at the edges and would be compared against nothing.

running = {}           # (side, variable, scenario) -> (years, (member, year))
kernel = np.ones(WINDOW) / WINDOW
for variable in VARIABLES:
    for scenario in SCENARIOS:
        for side in ("emulator", "cesm"):
            years, values = series[(side, variable, scenario)]
            smoothed = np.stack([np.convolve(member, kernel, mode="valid")
                                 for member in values])
            centre_years = years[WINDOW // 2: WINDOW // 2 + smoothed.shape[1]]
            running[(side, variable, scenario)] = (centre_years, smoothed)
        print(f"[step 3] {variable:6s} {scenario:7s} {WINDOW}-year running mean: "
              f"{running[('emulator', variable, scenario)][1].shape[1]} years "
              f"({running[('emulator', variable, scenario)][0][0]}-"
              f"{running[('emulator', variable, scenario)][0][-1]})")

# =============================================================================
#  STEP 4 — per-year test across members
# =============================================================================
# Welch at each year, across members. Valid AT A GIVEN YEAR because members are
# independent realizations; NOT valid as a family, because the running mean
# makes neighbouring years share W-1 of their W values. Read the fraction as
# "how much of the record differs", not as a corrected multiple test.

results = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        emulator_years, emulator_smoothed = running[("emulator", variable, scenario)]
        cesm_years, cesm_smoothed = running[("cesm", variable, scenario)]
        common = np.intersect1d(emulator_years, cesm_years)
        emulator_common = emulator_smoothed[:, np.searchsorted(emulator_years, common)]
        cesm_common = cesm_smoothed[:, np.searchsorted(cesm_years, common)]

        p_values = np.array([
            stats.ttest_ind(emulator_common[:, i], cesm_common[:, i],
                            equal_var=False).pvalue
            for i in range(len(common))])
        difference = emulator_common.mean(axis=0) - cesm_common.mean(axis=0)

        results[(variable, scenario)] = dict(
            years=common, p=p_values, difference=difference,
            emulator=emulator_common, cesm=cesm_common,
            fraction_differ=100.0 * np.mean(p_values < ALPHA),
            max_difference=float(np.abs(difference).max()),
            mean_difference=float(difference.mean()),
            n_members=emulator_common.shape[0])
        row = results[(variable, scenario)]
        print(f"[step 4] {variable:6s} {scenario:7s} "
              f"{row['fraction_differ']:5.1f}% of years differ, "
              f"mean gap {row['mean_difference']:+.3f}, "
              f"largest {row['max_difference']:.3f} "
              f"(n={row['n_members']} per side)")

# =============================================================================
#  STEP 5 — draw one 2x2 figure per variable
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

for variable in VARIABLES:
    variable_label, unit_axis, _ = VARIABLES[variable]
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))

    for panel_index, scenario in enumerate(SCENARIOS):
        axis = axes[panel_index // 2][panel_index % 2]
        row = results[(variable, scenario)]
        years = row["years"]

        # Years where the running means differ, marked along the bottom. Drawn
        # first so the curves sit on top.
        differ = row["p"] < ALPHA
        if differ.any():
            axis.fill_between(years, 0, 1, where=differ, transform=axis.get_xaxis_transform(),
                              color="0.6", alpha=0.22, lw=0, zorder=0)

        # Each side's member spread of running means, then its ensemble mean.
        axis.fill_between(years, row["cesm"].min(axis=0), row["cesm"].max(axis=0),
                          color=CESM_COLOUR, alpha=0.20, lw=0, zorder=1)
        for edge in (row["emulator"].min(axis=0), row["emulator"].max(axis=0)):
            axis.plot(years, edge, color=EMULATOR_COLOUR, lw=0.9, ls=":",
                      alpha=0.75, zorder=2)
        axis.plot(years, row["cesm"].mean(axis=0), color=CESM_COLOUR, lw=2.2,
                  zorder=4)
        axis.plot(years, row["emulator"].mean(axis=0), color=EMULATOR_COLOUR,
                  lw=2.2, zorder=4)

        axis.set_title(f"{SCENARIOS[scenario]}\n"
                       f"{row['fraction_differ']:.0f}% of years differ "
                       f"($p<{ALPHA}$), mean gap {row['mean_difference']:+.3f}",
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
        Patch(facecolor=CESM_COLOUR, alpha=0.20, label="CESM2 member range"),
        Line2D([], [], color=EMULATOR_COLOUR, lw=0.9, ls=":",
               label="Emulator member range"),
        Patch(facecolor="0.6", alpha=0.22, label=f"years differing at $p<{ALPHA}$"),
    ]
    figure.tight_layout()
    legend = figure.legend(handles=legend_entries, frameon=False, ncols=3,
                           loc="lower center", bbox_to_anchor=(0.5, 1.005))

    out_png = OUT.format(name=FIGURE_NAME[variable])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    for path in (out_png, os.path.splitext(out_png)[0] + ".pdf"):
        figure.savefig(path, bbox_inches="tight", bbox_extra_artists=[legend])
        print(f"[step 5] wrote {path}")
    plt.close(figure)

# =============================================================================
#  STEP 6 — the same numbers as a LaTeX table
# =============================================================================
# Plain LaTeX with borders: \toprule and \cmidrule need booktabs, and without it
# \cmidrule silently typesets "(lr)2-3" into the table.

for variable in VARIABLES:
    variable_label, _, unit_table = VARIABLES[variable]
    table_rows = []
    for scenario in SCENARIOS:
        row = results[(variable, scenario)]
        table_rows.append(
            f"{SCENARIOS[scenario]} & {row['n_members']} & "
            f"{row['mean_difference']:+.3f} & {row['max_difference']:.3f} & "
            f"{row['fraction_differ']:.0f} \\\\")

    table_tex = "\n".join([
        r"\begin{tabular}{|l|r|r|r|r|}",
        r"\hline",
        r"\textbf{Experiment} & \textbf{Members} & "
        r"\multicolumn{2}{c|}{\textbf{Gap (%s)}} & \textbf{Years differing} \\" % unit_table,
        r"\cline{3-4}",
        r" & (per side) & Mean & Largest & (\%) \\",
        r"\hline",
        *table_rows,
        r"\hline",
        r"\end{tabular}",
    ])

    table_path = TABLE.format(name=FIGURE_NAME[variable])
    with open(table_path, "w") as handle:
        handle.write(
            f"% {variable_label}: are the {WINDOW}-year running means of the\n"
            f"% emulator and CESM2 the same? Gaps are emulator minus CESM2 in\n"
            f"% {unit_table}.\n"
            f"%\n"
            f"% 'Years differing' is the percentage of years at which Welch's\n"
            f"%   t-test across MEMBERS rejects at p<{ALPHA}. That test is valid at\n"
            f"%   any single year, because members are independent realizations.\n"
            f"%   The years are NOT independent of one another — a {WINDOW}-year\n"
            f"%   running mean makes neighbours share {WINDOW - 1} of their {WINDOW}\n"
            f"%   values — so this is a profile of how much of the record differs,\n"
            f"%   NOT a family of tests to be counted or corrected.\n"
            f"%\n"
            f"% Generated by scripts/make_fig7.py — do not edit by hand.\n")
        handle.write(table_tex + "\n")
    print(f"[step 6] wrote {table_path}")

# =============================================================================
#  STEP 7 — the summary
# =============================================================================

print(f"\nAre the {WINDOW}-year running means the same?")
for variable in VARIABLES:
    for scenario in SCENARIOS:
        row = results[(variable, scenario)]
        verdict = ("differ over most of the record" if row["fraction_differ"] > 66 else
                   "differ over part of the record" if row["fraction_differ"] > 10 else
                   "indistinguishable almost everywhere")
        print(f"  {VARIABLES[variable][0]:13s} {SCENARIOS[scenario]:26s} "
              f"{row['fraction_differ']:5.1f}%  — {verdict}")
