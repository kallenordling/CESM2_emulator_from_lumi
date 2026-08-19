#!/usr/bin/env python3
"""
Emit FaIR-drivable emissions (CO2, SO2, BC) from the emissions the EMULATOR uses.

WHY
---
FaIR's shipped CMIP7 pathways are the ILLUSTRATIVE ones from Figure 1 of the
ScenarioMIP protocol paper, not the final IAM quantification that was gridded
into the input4MIPs `IIASA-IAMC-*` files the emulator is conditioned on. For
High they differ by ~34% at 2100 (71.3 vs 53.2 Gt CO2/yr). Comparing the
emulator against FaIR-as-shipped therefore compares two different scenarios and
attributes the gap to the emulator. This script closes that: it writes the
emulator's own CO2 so FaIR can be driven with identical forcing.

    python run_fair_co2only.py --co2-csv <this output> \\
           --co2-map h=high-extension vl=verylow

UNITS — the reason this reads native-grid data, not the cond files
------------------------------------------------------------------
The conditioning files live in "emulator input space": the extensive regrid onto
the 192x288 CESM2 grid does not conserve the global sum and deflates it ~4.7x,
and that factor is emission-pattern dependent (4.53 for ssp370 against 4.77
measured on the historical), so dividing it back out would inject a ~5% error
into a number FaIR takes literally. The native-grid annual rates carry real
Gt CO2/yr and need no such correction, so the series is built from those via
rebuild_cmip6_co2_cond.build_series() — the same splice, decadal->annual
interpolation and cumsum-free annual path the cond files are built from.

The output is ANNUAL emissions, which is what FaIR wants; the emulator's CO2
channel is the cumulative integral of exactly this series, and its SUL/BC
channels are the annual series unchanged.

SPECIES, AND WHAT IS AND IS NOT IN THEM
---------------------------------------
    CO2  Gt CO2/yr   surface anthro + aircraft   -> FaIR "CO2 FFI"
    SO2  Mt SO2/yr   surface anthro ONLY         -> FaIR "Sulfur"
    BC   Mt BC/yr    surface anthro ONLY         -> FaIR "BC"

The aerosol channels are SURFACE ANTHRO ONLY, because that is what the emulator
is conditioned on (make_aerosol_files.py omits AIR-anthro for both species, and
CEDS anthro excludes open biomass burning, which lives in BB4CMIP). Driving FaIR
with these therefore reproduces the emulator's forcing FAITHFULLY but is NOT the
complete anthropogenic aerosol source — expect weaker aerosol cooling than a
standard FaIR run. That is the intended behaviour here: the point is to match
the emulator, not to be a best-estimate climate projection. CO2 does include
aircraft, matching the cond channel.

Note FaIR takes CO2 as FFI + AFOLU separately; this writes the emulator's total
into CO2 FFI, as the emulator's CO2 has no separate land-use term.

Usage:
    # CMIP6 scenarios, from the local intermediates
    python scripts/make_fair_emissions.py --data-dir ~/data_staging/bc_rebuild \\
        --scenarios ssp370 --out plots/emulator_co2_for_fair.csv

    # CMIP7 h/vl need the raw input4MIPs (slow over sshfs)
    python scripts/make_fair_emissions.py --data-dir ~/mnt/lumi_sc2/emulator_data \\
        --from-raw --scenarios h vl --out plots/emulator_co2_for_fair.csv
"""
import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ESGF/emulator scenario -> the FaIR scenario it should drive. Confirmed against
# the protocol's official codes (H/HL/M/ML/L/VL/LN); see fair_cmip7_scenariomip.
FAIR_NAME = {"h": "high-extension", "vl": "verylow",
             "ssp370": "high-extension", "ssp126": "verylow",
             "ssp245": "medium-extension"}


# What each channel is called and what FaIR wants it in. The aerosols are stored
# in Gt/yr per gridpoint like CO2, but FaIR takes them in Mt/yr, hence the 1e3.
SPECIES = {
    "CO2": dict(fair="CO2 FFI", unit="Gt CO2/yr",  scale=1.0),
    "SO2": dict(fair="Sulfur",  unit="Mt SO2/yr",  scale=1e3),
    "BC":  dict(fair="BC",      unit="Mt BC/yr",   scale=1e3),
}


def load_rebuild():
    """Import data/rebuild_cmip6_co2_cond.py as a module.

    Reusing it rather than reimplementing keeps this script and the cond-file
    build on ONE definition of the CO2 series; a second implementation would be
    free to drift from the thing it is supposed to mirror.
    """
    p = os.path.join(REPO, "data", "rebuild_cmip6_co2_cond.py")
    spec = importlib.util.spec_from_file_location("rebuild_co2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["ssp370"])
    ap.add_argument("--from-raw", dest="input_dir", nargs="?", const="",
                    help="build from raw input4MIPs instead of the "
                         "CO2_cumulative_Gt_per_gridpoint_*.nc intermediates")
    ap.add_argument("--year-start", type=int, default=1750)
    ap.add_argument("--year-end", type=int, default=2100)
    ap.add_argument("--fair-names", action="store_true", default=True,
                    help="write FaIR scenario names (default); --raw-names to keep h/vl")
    ap.add_argument("--raw-names", dest="fair_names", action="store_false")
    ap.add_argument("--species", nargs="+", default=["CO2"],
                    choices=list(SPECIES), metavar="S",
                    help="CO2 (default), SO2, BC — or several. With CO2 alone "
                         "the legacy 3-column format is written so "
                         "run_fair_co2only.py --co2-csv keeps working; with "
                         "more than one, a long format with species/units "
                         "columns is written instead.")
    ap.add_argument("--out", default="plots/emulator_co2_for_fair.csv")
    args = ap.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    if args.input_dir == "":
        args.input_dir = os.path.join(args.data_dir, "emission_data", "inputs4mips")
    elif args.input_dir:
        args.input_dir = os.path.expanduser(args.input_dir)

    m = load_rebuild()
    print(f"[fair-emis] data-dir {args.data_dir}")
    print(f"[fair-emis] source   {args.input_dir or 'intermediates'}")

    rows, summary = [], []
    for sc in args.scenarios:
        print(f"\n=== {sc} ===")
        name = FAIR_NAME.get(sc, sc) if args.fair_names else sc
        if args.fair_names and sc not in FAIR_NAME:
            print(f"  [warn] no FaIR name for {sc!r}; writing it unmapped",
                  file=sys.stderr)
        for sp in args.species:
            meta = SPECIES[sp]
            if sp == "CO2":
                # build_series returns (annual, cumulative); FaIR wants annual.
                series, _ = m.build_series(args.data_dir, sc, args.input_dir)
            else:
                # No cumsum for the aerosols — they are rates in every cond file,
                # and integrating them is the mistake that turned ssp370's SO2
                # negative in an earlier script.
                series = m.build_aero(args.input_dir, sc, sp)
            yrs = np.asarray(series["year"].values, dtype=int)
            # Global sum of a per-gridpoint EXTENSIVE field is the global total.
            glob = series.sum(dim=("lat", "lon")).values.astype(float) * meta["scale"]
            keep = (yrs >= args.year_start) & (yrs <= args.year_end)
            yrs, glob = yrs[keep], glob[keep]
            rows += [(name, int(y), sp, meta["fair"], float(v), meta["unit"])
                     for y, v in zip(yrs, glob)]
            i2100 = np.where(yrs == 2100)[0]
            summary.append((sc, name, sp, meta["unit"], yrs[0], yrs[-1],
                            float(glob[i2100[0]]) if i2100.size else float("nan")))
            print(f"  {sp:3s} {yrs[0]}-{yrs[-1]}, {len(yrs)} yrs "
                  f"-> FaIR {meta['fair']!r} in {meta['unit']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    legacy = args.species == ["CO2"]
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        if legacy:
            # Exactly the old 3-column shape, so run_fair_co2only.py --co2-csv
            # keeps reading it without knowing this script grew other species.
            w.writerow(["scenario", "year", "GtCO2_per_yr"])
            w.writerows([(n, y, v) for n, y, _sp, _f, v, _u in rows])
        else:
            w.writerow(["scenario", "year", "species", "fair_species",
                        "value", "units"])
            w.writerows(rows)
    print(f"\n[fair-emis] wrote {args.out} ({len(rows)} rows)")

    print(f"\n  {'scenario':9s} {'-> FaIR':18s} {'sp':4s} {'years':12s} "
          f"{'value @2100':>13s}  units")
    for sc, name, sp, unit, y0, y1, v in summary:
        print(f"  {sc:9s} {name:18s} {sp:4s} {y0}-{y1}  {v:13.3f}  {unit}")
    print("\n  Drive FaIR with:")
    print(f"    python run_fair_co2only.py --co2-csv {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
