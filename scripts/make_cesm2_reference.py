#!/usr/bin/env python3
"""
================================================================================
 CESM2 REFERENCE FILES — the model data the emulator is judged against
================================================================================

Run it with no arguments:

    python scripts/make_cesm2_reference.py

Everything configurable is in the SETTINGS block below. The script then runs top
to bottom in five numbered steps, with no helper functions, so it can be read as
a description of what the reference data is and where it comes from.

WHY THIS EXISTS
---------------
eval_aero.py writes only the emulator's own output — `{var}_model` with dims
(member, year, lat, lon) and nothing else. So the CESM2 side has to come from
somewhere, and the raw form is awkward: one directory per member, forty chunk
files inside each, thirty-odd members per experiment. Walking that takes two
minutes on LUMI and ten to fifteen over a network mount, and every analysis
script that needs a reference was repeating it.

This walks it ONCE and writes, per variable and scenario:

    {var}_cesm   (member, year, lat, lon)   absolute, in the emulator's units

which mirrors the eval file exactly, so a plotting script opens one file per
side and compares them directly.

THREE THINGS THE FILE GUARANTEES
--------------------------------
1. HELD-OUT MEMBERS ONLY. The emulator was fitted on some CESM2 members;
   scoring against those would be marking its own homework. Filtering here,
   rather than in each analysis, makes the mistake impossible downstream
   instead of merely discouraged.

2. MEMBER NAMES, NOT NUMBERS. The emulator's members are interchangeable
   random draws, so integers suffice for it. CESM2's identify specific
   realizations — you need the name when one turns out to be corrupt, as
   LE2-1231.012 was.

3. THE EMULATOR'S UNITS. Kelvin becomes degrees Celsius and m/s becomes
   mm/day here, so nothing downstream has to remember which side is which.
"""

# =============================================================================
#  SETTINGS
# =============================================================================

# Where emulator_data lives. The default is LUMI's copy, where the training
# tree is on fast local storage; over the sshfs mount this script still works
# but takes ten to fifteen minutes per variable instead of one.
DATA_ROOT = "/scratch/project_462001112/emulator_data"

OUT_DIR = f"{DATA_ROOT}/cesm2_reference"

# The training config decides which members are OFF LIMITS as a reference.
DATA_CONFIG = "configs/config_data_ybias_BCprect.yaml"

VARIABLES = ["TREFHT", "PRECT"]
SCENARIOS = ["hist", "ssp370", "aaer", "ghg", "ssp126", "ssp245"]

# Set False to include the members the emulator trained on. Only ever useful
# for diagnosing the emulator's fit to its own training data.
HELD_OUT_ONLY = True

# 0 = every member. A small number makes a quick structural test.
MAX_MEMBERS = 0

# ── Where each scenario's CESM2 data comes from ──────────────────────────────
# hist, ssp370, aaer and ghg have a LENS2 training tree, one directory per
# member. Note the capitalisation: the tree uses AAER and GHG where the rest of
# the codebase says aaer and ghg.
TREE_SUBDIR = {"hist": "hist", "ssp370": "ssp370", "aaer": "AAER", "ghg": "GHG"}

# ssp126 and ssp245 were never trained on and have no tree. Their reference is
# a pre-aggregated 3-member CMIP6 ensemble in a single file.
CMIP6_FILE = {
    ("ssp126", "TREFHT"): ("cmip6/ssp126.nc", "tas"),
    ("ssp126", "PRECT"):  ("cmip6/ssp126_pr.nc", "pr"),
    ("ssp245", "TREFHT"): ("cmip6/ssp245.nc", "tas"),
    ("ssp245", "PRECT"):  ("cmip6/ssp245_pr.nc", "pr"),
}

# ── Unit conversions into what the emulator writes ───────────────────────────
# Keyed by the `units` attribute found in the source file. Anything not listed
# stops the script rather than being guessed at: a wrong factor here would
# silently rescale every comparison made with these files.
TO_EMULATOR_UNITS = {
    "TREFHT": {
        "K":    lambda x: x - 273.15,
        "degC": lambda x: x,
    },
    "PRECT": {
        "m/s":        lambda x: x * 1000.0 * 86400.0,   # m->mm, s->day
        "kg m-2 s-1": lambda x: x * 86400.0,            # CMIP6 `pr`, 1 kg/m2 = 1 mm
        "mm/day":     lambda x: x,
    },
}
EMULATOR_UNITS = {"TREFHT": "degC", "PRECT": "mm/day"}

# =============================================================================

import glob
import os
import sys

import numpy as np
import xarray as xr
import yaml

os.makedirs(OUT_DIR, exist_ok=True)
inventory = []          # one row per file written, summarised at the end

# =============================================================================
#  STEP 1 — find out which members the emulator was trained on
# =============================================================================
# The training config lists them per experiment:
#     experiment_configs:
#       - scenario_name: hist
#         realizations: [LE2-1001.001, LE2-1011.001, ...]
# Everything else on disk is fair game as a reference.

trained_members = {}
if HELD_OUT_ONLY:
    config = yaml.safe_load(open(DATA_CONFIG))
    for experiment in config["experiment_configs"]:
        trained_members[experiment["scenario_name"]] = set(
            experiment.get("realizations", []))
    print(f"[step 1] {DATA_CONFIG}: "
          + ", ".join(f"{k} {len(v)} trained" for k, v in trained_members.items()))
else:
    print("[step 1] HELD_OUT_ONLY is off — including trained members")

# The remaining steps run once per (variable, scenario).
for variable in VARIABLES:
    for scenario in SCENARIOS:

        # =====================================================================
        #  STEP 2 — decide where this scenario's data comes from
        # =====================================================================
        from_tree = scenario in TREE_SUBDIR

        if from_tree:
            source_dir = f"{DATA_ROOT}/training_data/{variable}/{TREE_SUBDIR[scenario]}"
            if not os.path.isdir(source_dir):
                print(f"[{variable}/{scenario}] no tree at {source_dir} — skipped")
                continue
        else:
            spec = CMIP6_FILE.get((scenario, variable))
            if spec is None:
                print(f"[{variable}/{scenario}] no CMIP6 file listed — skipped")
                continue
            source_file = f"{DATA_ROOT}/{spec[0]}"
            cmip6_variable = spec[1]
            if not os.path.exists(source_file):
                print(f"[{variable}/{scenario}] {spec[0]} does not exist — skipped")
                continue

        # =====================================================================
        #  STEP 3 — read the members
        # =====================================================================
        # Either way this produces: `data` (member, year, lat, lon),
        # `member_names`, `years`, `lat`, `lon` — and the units it arrived in.

        if from_tree:
            # One directory per member, forty chunk files inside each.
            all_members = sorted(
                name for name in os.listdir(source_dir)
                if name != "diagnostics"                 # a folder of staging plots
                and os.path.isdir(f"{source_dir}/{name}"))

            # Split them explicitly, so what went in and what stayed out are
            # both named rather than one being the leftover of the other.
            member_names, excluded_as_trained = [], []
            for name in all_members:
                if HELD_OUT_ONLY and name in trained_members.get(scenario, set()):
                    excluded_as_trained.append(name)
                else:
                    member_names.append(name)

            dropped_by_limit = []
            if MAX_MEMBERS:
                dropped_by_limit = member_names[MAX_MEMBERS:]
                member_names = member_names[:MAX_MEMBERS]

            if not member_names:
                print(f"[{variable}/{scenario}] every member on disk was trained "
                      f"on — nothing to use as a reference")
                continue

            per_member = []
            years = lat = lon = source_units = None
            for i, name in enumerate(member_names, 1):
                print(f"[{variable}/{scenario}] [{i}/{len(member_names)}] {name}",
                      flush=True)
                # Sort the chunks NUMERICALLY: lexically they run 0, 1, 10, 11,
                # 2, ... and the time axis would come out scrambled.
                chunks = sorted(
                    glob.glob(f"{source_dir}/{name}/chunk_*.nc"),
                    key=lambda path: int(os.path.basename(path)[len("chunk_"):-3]))
                member = xr.open_mfdataset(chunks, combine="by_coords",
                                           decode_times=False)
                field = member[variable]
                per_member.append(field.values)
                if years is None:
                    time_dim = field.dims[0]          # the tree stores plain years
                    years = np.asarray(member[time_dim].values).astype(int)
                    lat = member["lat"].values
                    lon = member["lon"].values
                    source_units = field.attrs.get("units")
                member.close()

            data = np.stack(per_member)
            source_description = f"training_data/{variable}/{TREE_SUBDIR[scenario]}"

        else:
            # One file, members already stacked on a `member` dimension.
            dataset = xr.open_dataset(source_file)
            field = dataset[cmip6_variable].transpose("member", "year", "lat", "lon")
            data = field.values
            member_names = [str(m) for m in field["member"].values]
            years = field["year"].values.astype(int)
            lat = field["lat"].values
            lon = field["lon"].values
            source_units = field.attrs.get("units")
            dataset.close()
            source_description = spec[0]
            # These scenarios were never trained on, so every member is usable.
            excluded_as_trained, dropped_by_limit = [], []

        # =====================================================================
        #  STEP 4 — convert to the emulator's units
        # =====================================================================
        convert = TO_EMULATOR_UNITS[variable].get(source_units)
        if convert is None:
            sys.exit(f"[{variable}/{scenario}] source units {source_units!r} have "
                     f"no conversion to {EMULATOR_UNITS[variable]}. Known: "
                     f"{sorted(TO_EMULATOR_UNITS[variable])}")
        data = convert(data)

        # =====================================================================
        #  STEP 5 — write one file, shaped like the emulator's
        # =====================================================================
        reference = xr.Dataset(
            {f"{variable}_cesm": xr.DataArray(
                data,
                dims=["member", "year", "lat", "lon"],
                coords={"member": member_names, "year": years,
                        "lat": lat, "lon": lon},
                attrs={"units": EMULATOR_UNITS[variable],
                       "long_name": f"CESM2 {variable}, absolute, all members"})},
            attrs={
                "experiment": scenario,
                "description": "CESM2 reference for emulator evaluation",
                # ── what is in this file ─────────────────────────────────────
                "source": source_description,
                "source_units": str(source_units),
                "units": EMULATOR_UNITS[variable],
                "n_members": len(member_names),
                "member_names": ", ".join(member_names),
                "years": f"{years[0]}-{years[-1]}",
                # ── and what was deliberately left out ───────────────────────
                "member_selection": ("held-out only" if HELD_OUT_ONLY
                                     else "all members, trained ones included"),
                "excluded_trained_members": (", ".join(excluded_as_trained)
                                             or "none"),
                "excluded_by_max_members": (", ".join(dropped_by_limit) or "none"),
                "training_config": DATA_CONFIG if HELD_OUT_ONLY else "n/a",
            },
        )
        out_path = f"{OUT_DIR}/{variable}_{scenario}.nc"
        reference.to_netcdf(out_path)
        print(f"[{variable}/{scenario}] wrote {out_path}  "
              f"({os.path.getsize(out_path) / 1e6:.0f} MB)")
        print(f"    source   : {source_description}  [{source_units} -> "
              f"{EMULATOR_UNITS[variable]}]")
        print(f"    years    : {years[0]}-{years[-1]}  ({len(years)})")
        print(f"    IN  ({len(member_names):2d}): {', '.join(member_names)}")
        if excluded_as_trained:
            print(f"    OUT ({len(excluded_as_trained):2d}): trained on — "
                  f"{', '.join(excluded_as_trained)}")
        if dropped_by_limit:
            print(f"    OUT ({len(dropped_by_limit):2d}): MAX_MEMBERS — "
                  f"{', '.join(dropped_by_limit)}")
        inventory.append(dict(variable=variable, scenario=scenario,
                              members=len(member_names),
                              excluded=len(excluded_as_trained),
                              years=f"{years[0]}-{years[-1]}",
                              source=source_description))


# =============================================================================
#  Summary — every file written, and what went into it
# =============================================================================
print(f"\n{len(inventory)} files in {OUT_DIR}")
print(f"{'variable':9} {'scenario':9} {'members':>7} {'excluded':>8} "
      f"{'years':>10}  source")
for row in inventory:
    print(f"{row['variable']:9} {row['scenario']:9} {row['members']:7d} "
          f"{row['excluded']:8d} {row['years']:>10}  {row['source']}")
