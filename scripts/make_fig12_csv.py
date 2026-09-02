#!/usr/bin/env python3
"""
================================================================================
 CSV EXPORT of the data behind figures 1 and 2
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig12_csv.py

Everything configurable is in the SETTINGS block below. Four numbered steps, no
helper functions, same shape as the make_fig*.py scripts.

WHAT IT WRITES
--------------
One CSV per variable, experiment and side — 2 x 4 x 2 = 16 files:

    <OUT_DIR>/<variable>_<scenario>_<side>.csv

Each file is a plain table:

    * ROWS are years, indexed by the `year` column
    * COLUMNS are ensemble members, one per member

so it loads with `pandas.read_csv(path, index_col="year")` and needs no
reshaping. CESM2 columns carry the realization NAMES (LE2-1231.001), because
those identify specific runs; emulator columns are m1, m2, ... because its
members are interchangeable random draws.

WHICH NUMBERS
-------------
Global means, cos(lat)-weighted, as ABSOLUTE values — degrees Celsius and
mm/day, what the model and CESM2 actually produce.

NOTE THAT FIGURES 1 AND 2 PLOT ANOMALIES, so a column here will not reproduce a
plotted line directly. <OUT_DIR>/baselines.csv carries each side's own 1850-1900
mean; subtract the matching row to get the figure's quantity. Absolute is the
better default for a data file — it is the raw number, and the anomaly is one
subtraction away, whereas going the other direction needs a baseline the file
would not otherwise carry. ssp370 begins in 2015 and has no baseline period of
its own, so its row reports the historical one, the convention the figures use.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"
REFERENCE_DIR = "/home/nordling/mnt/lumi_sc/emulator_data/cesm2_reference"

OUT_DIR = "plots/fig12_data"

# Cap the emulator at the CESM2 member count per experiment, as the figures do.
# Set False to export all 25 emulator members.
MATCH_MEMBER_COUNTS = True

# ABSOLUTE values: what the model and CESM2 actually produce, in degC and
# mm/day. Figures 1 and 2 plot anomalies, so a column here will not equal the
# plotted line — subtract the matching row of baselines.csv to get that.
# Absolute is the better default for a data file: it is the raw quantity, and
# the anomaly is one subtraction away, whereas the reverse needs a baseline
# the file would not carry.
ANOMALY = False
BASELINE = (1850, 1900)   # still used to REPORT baselines alongside

VARIABLES = ["TREFHT", "PRECT"]
SCENARIOS = ["hist", "ssp370", "aaer", "ghg"]

# =============================================================================

import os

import numpy as np
import pandas as pd
import xarray as xr

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
#  STEP 1 — read both sides, as global means
# =============================================================================
# The files hold maps, so the global mean is computed here: cos(lat)-weighted,
# because grid cells shrink towards the poles and an unweighted mean over this
# grid is wrong by degrees, not decimals.

table = {}        # (variable, scenario, side) -> DataFrame(year x member)
for variable in VARIABLES:
    for scenario in SCENARIOS:
        for side, directory, field_name in (
                ("emulator", EVAL_DIR, f"{variable}_model"),
                ("cesm2", REFERENCE_DIR, f"{variable}_cesm")):
            dataset = xr.open_dataset(f"{directory}/{variable}_{scenario}.nc")
            field = dataset[field_name]
            weights = np.cos(np.deg2rad(field["lat"]))
            global_mean = field.weighted(weights).mean(("lat", "lon")).compute()

            # Column labels: real names for CESM2, m1..mN for the emulator.
            if side == "cesm2":
                columns = [str(m) for m in global_mean["member"].values]
            else:
                columns = [f"m{i}" for i in range(1, global_mean.sizes["member"] + 1)]

            table[(variable, scenario, side)] = pd.DataFrame(
                global_mean.values.T,                      # (year, member)
                index=pd.Index(global_mean["year"].values.astype(int), name="year"),
                columns=columns)
            dataset.close()
        print(f"[step 1] {variable:6s} {scenario:7s} "
              f"emulator {table[(variable, scenario, 'emulator')].shape}, "
              f"CESM2 {table[(variable, scenario, 'cesm2')].shape}  (year x member)")

# =============================================================================
#  STEP 2 — same number of members on both sides
# =============================================================================

if MATCH_MEMBER_COUNTS:
    for variable in VARIABLES:
        for scenario in SCENARIOS:
            n_cesm = table[(variable, scenario, "cesm2")].shape[1]
            emulator_table = table[(variable, scenario, "emulator")]
            if emulator_table.shape[1] > n_cesm:
                table[(variable, scenario, "emulator")] = emulator_table.iloc[:, :n_cesm]
                print(f"[step 2] {variable:6s} {scenario:7s} emulator "
                      f"{emulator_table.shape[1]} -> {n_cesm} members")

# =============================================================================
#  STEP 3 — each side's 1850-1900 baseline
# =============================================================================
# Always computed and always written to baselines.csv; ANOMALY decides only
# whether it is SUBTRACTED from the exported values. With ANOMALY = False the
# CSVs stay absolute and the baseline file is what turns them into the anomalies
# figures 1 and 2 plot.
#
# ssp370 starts in 2015 and has no baseline period of its own, so it inherits
# the historical one — the same convention applied to both sides, which is what
# keeps them comparable.

baseline_rows = []
if True:                       # always compute them; ANOMALY decides whether to SUBTRACT
    baseline_value = {}
    for variable in VARIABLES:
        for side in ("emulator", "cesm2"):
            for scenario in SCENARIOS:
                frame = table[(variable, scenario, side)]
                window = frame.loc[BASELINE[0]:BASELINE[1]]
                baseline_value[(variable, scenario, side)] = (
                    float(window.values.mean()) if len(window) else np.nan)
            if not np.isfinite(baseline_value[(variable, "ssp370", side)]):
                baseline_value[(variable, "ssp370", side)] = \
                    baseline_value[(variable, "hist", side)]

    for variable in VARIABLES:
        for scenario in SCENARIOS:
            for side in ("emulator", "cesm2"):
                base = baseline_value[(variable, scenario, side)]
                if ANOMALY:
                    table[(variable, scenario, side)] = (
                        table[(variable, scenario, side)] - base)
                baseline_rows.append(dict(variable=variable, scenario=scenario,
                                          side=side, baseline=round(base, 6)))
            print(f"[step 3] {variable:6s} {scenario:7s} baselines "
                  f"emulator {baseline_value[(variable, scenario, 'emulator')]:8.3f}, "
                  f"CESM2 {baseline_value[(variable, scenario, 'cesm2')]:8.3f}"
                  + ("   (inherited from hist)" if scenario == "ssp370" else ""))

# =============================================================================
#  STEP 4 — write the files
# =============================================================================

written = 0
for variable in VARIABLES:
    for scenario in SCENARIOS:
        for side in ("emulator", "cesm2"):
            path = f"{OUT_DIR}/{variable}_{scenario}_{side}.csv"
            table[(variable, scenario, side)].to_csv(path, float_format="%.6f")
            frame = table[(variable, scenario, side)]
            print(f"[step 4] {path}  "
                  f"{frame.shape[0]} years x {frame.shape[1]} members")
            written += 1

if baseline_rows:
    baseline_path = f"{OUT_DIR}/baselines.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    print(f"[step 4] {baseline_path}  "
          + ("(add these back to recover absolute values)" if ANOMALY
             else "(subtract these to get the anomalies figures 1 and 2 plot)"))
    written += 1

print(f"\n{written} files in {OUT_DIR}/")
print("Load one with:  pandas.read_csv(path, index_col='year')")
print(f"Units: TREFHT degC, PRECT mm/day — "
      + (f"anomalies vs {BASELINE[0]}-{BASELINE[1]}" if ANOMALY
         else "ABSOLUTE (see baselines.csv for the anomaly conversion)"))
