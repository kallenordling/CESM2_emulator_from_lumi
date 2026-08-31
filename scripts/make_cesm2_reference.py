#!/usr/bin/env python3
"""Build CESM2 reference files in the same shape as the emulator's eval output.

WHY
---
eval_aero.py now writes one thing per scenario — the emulator's absolute
fields, `{var}_model` with dims (member, year, lat, lon). The CESM2 side was
deliberately dropped from it: CESM2 is not model output and does not change
between evaluations.

But every analysis then has to rebuild the reference by walking
training_data/<VAR>/<scenario>/<member>/chunk_*.nc, forty files per member and
thirty-odd members, which takes ten to fifteen minutes over a network mount and
is repeated in every script that needs it. This does that walk ONCE and writes:

    {var}_cesm   (member, year, lat, lon)   absolute, in the emulator's units

so a plotting script reads one file per (variable, scenario) and is done.

WHAT IS IN IT
-------------
HELD-OUT MEMBERS ONLY, by default: those present on disk but absent from the
training config's experiment_configs. Scoring an emulator against members it
was fitted on is marking its own homework, and having the reference file
contain only held-out members makes that mistake impossible downstream rather
than merely discouraged. --all-members overrides it.

The member coordinate carries the member NAMES (LE2-1231.012), not indices:
unlike the emulator's interchangeable draws, these identify specific
realizations, and a name is what you need when one turns out to be corrupt.

UNITS are converted to what the emulator writes, so the two files can be
compared without a second thought: TREFHT K -> degC, PRECT m/s -> mm/day.

ssp126 and ssp245 have no LENS2 tree; their reference is the pre-aggregated
CMIP6 ensemble under cmip6/, which this reads instead.

Usage
-----
    python scripts/make_cesm2_reference.py                     # all defaults
    python scripts/make_cesm2_reference.py --variables TREFHT \\
        --scenarios hist ssp370 --out-dir /scratch/.../cesm2_reference
"""

import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr
import yaml

# scenario -> training-tree subdirectory (None = no tree, use the CMIP6 file)
TREE_SUBDIR = {"hist": "hist", "ssp370": "ssp370", "aaer": "AAER", "ghg": "GHG",
               "ssp126": None, "ssp245": None}

# scenario -> (file under the data root, variable inside it) for the CMIP6 route
CMIP6_SOURCE = {
    "ssp126": {"TREFHT": ("cmip6/ssp126.nc", "tas"),
               "PRECT":  ("cmip6/ssp126_pr.nc", "pr")},
    "ssp245": {"TREFHT": ("cmip6/ssp245.nc", "tas"),
               "PRECT":  ("cmip6/ssp245_pr.nc", "pr")},
}

# raw units -> the emulator's units
CONVERT = {
    "TREFHT": {"K": lambda x: x - 273.15, "degC": lambda x: x},
    "PRECT":  {"m/s": lambda x: x * 8.64e7,          # x1000 mm/m, x86400 s/day
               "kg m-2 s-1": lambda x: x * 86400.0,  # CMIP6 `pr`
               "mm/day": lambda x: x},
}
UNITS = {"TREFHT": "degC", "PRECT": "mm/day"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/scratch/project_462001112/emulator_data")
    ap.add_argument("--data-config", default="configs/config_data_ybias_BCprect.yaml",
                    help="its experiment_configs define the TRAINED members")
    ap.add_argument("--out-dir", default=None,
                    help="default <data-root>/cesm2_reference")
    ap.add_argument("--variables", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--scenarios", nargs="+",
                    default=["hist", "ssp370", "aaer", "ghg"])
    ap.add_argument("--all-members", action="store_true",
                    help="include members used in training as well")
    ap.add_argument("--max-members", type=int, default=0,
                    help="keep only the first N (0 = all); for quick tests")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_root, "cesm2_reference")
    os.makedirs(out_dir, exist_ok=True)

    trained = {}
    if not args.all_members:
        cfg = yaml.safe_load(open(args.data_config))
        trained = {e["scenario_name"]: set(e.get("realizations", []))
                   for e in cfg["experiment_configs"]}

    for var in args.variables:
        for scen in args.scenarios:
            out_path = os.path.join(out_dir, f"{var}_{scen}.nc")
            subdir = TREE_SUBDIR.get(scen)

            # ── the CMIP6 route, for scenarios with no LENS2 tree ────────────
            if subdir is None:
                spec = CMIP6_SOURCE.get(scen, {}).get(var)
                src = os.path.join(args.data_root, spec[0]) if spec else None
                if not src or not os.path.exists(src):
                    print(f"[{var}/{scen}] no CMIP6 reference "
                          f"({spec[0] if spec else 'unknown'}) — skipped")
                    continue
                ds = xr.open_dataset(src)
                da = ds[spec[1]]
                raw_units = da.attrs.get("units")
                convert = CONVERT[var].get(raw_units)
                if convert is None:
                    print(f"[{var}/{scen}] units {raw_units!r} unknown — skipped")
                    continue
                da = convert(da).transpose("member", "year", "lat", "lon")
                members = [str(m) for m in da["member"].values]
                data = da.values
                years = da["year"].values.astype(int)
                lat, lon = da["lat"].values, da["lon"].values
                ds.close()
                source = spec[0]

            # ── the training-tree route ─────────────────────────────────────
            else:
                root = os.path.join(args.data_root, "training_data", var, subdir)
                if not os.path.isdir(root):
                    print(f"[{var}/{scen}] {root} missing — skipped")
                    continue
                on_disk = sorted(m for m in os.listdir(root)
                                 if m != "diagnostics"
                                 and os.path.isdir(os.path.join(root, m)))
                members = [m for m in on_disk if m not in trained.get(scen, set())]
                if args.max_members:
                    members = members[:args.max_members]
                if not members:
                    print(f"[{var}/{scen}] no held-out members — skipped")
                    continue

                stack, years, lat, lon = [], None, None, None
                for i, mem in enumerate(members, 1):
                    files = sorted(glob.glob(os.path.join(root, mem, "chunk_*.nc")),
                                   key=lambda f: int(os.path.basename(f)[6:-3]))
                    print(f"[{var}/{scen}] [{i}/{len(members)}] {mem}", flush=True)
                    ds = xr.open_mfdataset(files, combine="by_coords",
                                           decode_times=False)
                    da = ds[var]
                    raw_units = da.attrs.get("units")
                    convert = CONVERT[var].get(raw_units)
                    if convert is None:
                        sys.exit(f"{mem}: units {raw_units!r} have no conversion "
                                 f"to {UNITS[var]}")
                    tdim = da.dims[0]
                    stack.append(convert(da).values)
                    if years is None:
                        years = np.asarray(ds[tdim].values).astype(int)
                        lat, lon = ds["lat"].values, ds["lon"].values
                    ds.close()
                data = np.stack(stack)
                source = f"training_data/{var}/{subdir}"

            out = xr.Dataset(
                {f"{var}_cesm": xr.DataArray(
                    data, dims=["member", "year", "lat", "lon"],
                    coords={"member": members, "year": years,
                            "lat": lat, "lon": lon},
                    attrs={"units": UNITS[var],
                           "long_name": f"CESM2 {var}, absolute, all members"})},
                attrs={"experiment": scen, "source": source,
                       "members": "held-out only" if not args.all_members else "all",
                       "description": "CESM2 reference for emulator evaluation"},
            )
            out.to_netcdf(out_path)
            print(f"[{var}/{scen}] wrote {out_path}  "
                  f"({len(members)} members, {len(years)} years, "
                  f"{os.path.getsize(out_path)/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
