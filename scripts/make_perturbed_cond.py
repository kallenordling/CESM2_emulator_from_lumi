#!/usr/bin/env python3
"""
================================================================================
 PERTURBED CONDITIONING FILES — one emission source region at a time
================================================================================

    python scripts/make_perturbed_cond.py --species SUL --region "East Asia" \
        --scale 0.5 --out-dir plots/perturbed

WHAT IT IS FOR
--------------
To ask "which emissions, from where, drive the nonlinearity N at which
location", you scale ONE species over ONE source region, re-run the emulator,
and difference the resulting N maps. This writes the perturbed cond files; the
forward runs are a separate GPU job.

N = ALL - GHG - AAER, so each perturbation needs the cond files of all three
legs. It only touches the legs where that species actually varies:

    perturb CO2  -> ssp370/hist (ALL) and ghg      (aaer holds CO2 at 1850)
    perturb SUL  -> ssp370/hist (ALL) and aaer     (ghg holds SUL at 1850)
    perturb BC   -> ssp370/hist (ALL) and aaer     (ghg holds BC  at 1850)

That is not a workaround for the degenerate bases — it is what the
single-forcing design means. A species held at 1850 in a run cannot respond to
being scaled there.

THE PERTURBATION IS NOT WHAT THE MODEL SEES
-------------------------------------------
The cond pipeline projects every channel onto its persisted PCA basis (30 EOFs
for CO2, 5 for the aerosols) before the network sees it. A sharp regional box is
NOT in the span of five aerosol EOFs, so the model receives the projection of
your perturbation, not your perturbation. That is unavoidable and it is also
fine — but it must be MEASURED and reported, or the result gets attributed to a
region the model never actually saw a change in.

`--report-effective` does that: it projects both the original and the perturbed
field through the checkpoint's basis and writes the difference, which is the
perturbation the model truly receives. Always look at it before believing a
regional attribution.
"""

import argparse
import os

import numpy as np
import xarray as xr

# Source regions: (label, lat_min, lat_max, lon_min, lon_max), longitudes in the
# files' own 0-360 convention. Same boxes as scripts/make_fig16_eofs.py so the
# perturbations line up with the modes they will inevitably be projected onto.
REGIONS = {
    "East Asia":  (20.0,  50.0, 100.0, 145.0),
    "South Asia": (5.0,   30.0,  65.0, 100.0),
    "Europe":     (35.0,  65.0,   0.0,  40.0),
    "N. America": (30.0,  60.0, 230.0, 300.0),
    "Africa":     (-35.0, 35.0,   0.0,  50.0),
    "S. America": (-55.0, 12.0, 280.0, 325.0),
    "Global":     (-90.0, 90.0,   0.0, 360.0),
}

# Which cond file each leg uses. The ALL leg is hist spliced to ssp370, so both
# are listed; the window of interest (2031-2050) lies in ssp370 alone.
COND_DIR = "/home/nordling/mnt/lumi_sc2/emulator_data"
LEG_FILES = {
    "hist":   f"{COND_DIR}/emissions_hist_only_timefixed_bc_co2fix.nc",
    "ssp370": f"{COND_DIR}/emissions_ssp370_only_timefixed_bc_co2fix.nc",
    "aaer":   f"{COND_DIR}/emissions_aaer_only_timefixed_bc_co2fix.nc",
    "ghg":    f"{COND_DIR}/emissions_ghg_only_timefixed_bc_co2fix.nc",
}
# Which legs each species actually varies in (see the docstring).
SPECIES_LEGS = {"CO2": ["hist", "ssp370", "ghg"],
                "SUL": ["hist", "ssp370", "aaer"],
                "BC":  ["hist", "ssp370", "aaer"]}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--species", required=True, choices=["CO2", "SUL", "BC"])
parser.add_argument("--region", required=True, choices=sorted(REGIONS))
parser.add_argument("--scale", type=float, required=True,
                    help="multiplier inside the region (0.5 = halve, 1.5 = +50%%)")
parser.add_argument("--out-dir", default="plots/perturbed")
parser.add_argument("--from-year", type=int, default=2015,
                    help="apply the perturbation from this year on; earlier "
                         "years are left alone so the 1850-1900 baseline and "
                         "the historical record are untouched")
parser.add_argument("--taper-years", type=int, default=10,
                    help="ramp the perturbation in linearly over this many "
                         "years, so the cond does not step discontinuously")
parser.add_argument("--report-effective", metavar="EOF_NPZ",
                    help="project through this basis dump (from "
                         "scripts/dump_cond_eofs.py) and report what the model "
                         "would actually receive")
args = parser.parse_args()

lat_min, lat_max, lon_min, lon_max = REGIONS[args.region]
os.makedirs(args.out_dir, exist_ok=True)
tag = (f"{args.species}_{args.region.replace('. ', '').replace(' ', '')}"
       f"_x{args.scale:g}")

for leg in SPECIES_LEGS[args.species]:
    path = LEG_FILES[leg]
    if not os.path.exists(path):
        print(f"[skip] {leg}: {path} not found")
        continue
    dataset = xr.open_dataset(path)
    time_name = "year" if "year" in dataset.dims else "time"
    years = dataset[time_name].values.astype(int)

    field = dataset[args.species]
    latitude = dataset["lat"].values
    longitude = dataset["lon"].values
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    inside = ((lat_grid >= lat_min) & (lat_grid <= lat_max)
              & (lon_grid >= lon_min) & (lon_grid <= lon_max))

    # A linear ramp, not a step: a discontinuity at one year is a signal the
    # model has never seen and would show up as a transient of its own.
    ramp = np.clip((years - args.from_year) / max(args.taper_years, 1), 0.0, 1.0)
    factor = 1.0 + (args.scale - 1.0) * ramp                    # (T,)

    weight = np.where(inside, 1.0, 0.0)[None, :, :]             # (1, H, W)
    multiplier = 1.0 + (factor[:, None, None] - 1.0) * weight   # (T, H, W)

    perturbed = dataset.copy(deep=True)
    perturbed[args.species] = (field.dims, field.values * multiplier)

    out_path = f"{args.out_dir}/{os.path.basename(path).replace('.nc', '')}__{tag}.nc"
    perturbed.to_netcdf(out_path)

    changed = 100 * (np.abs(multiplier - 1.0) > 1e-12).mean()
    print(f"[write] {leg:7s} {out_path}")
    print(f"         {changed:.1f}% of (time x space) cells scaled; "
          f"final-year factor inside region {factor[-1]:.3f}")

    if args.report_effective and leg == "ssp370":
        store = np.load(args.report_effective)
        key = f"components_{leg}_{args.species}"
        if key not in store:
            print(f"         [effective] no {key} in the basis dump — skipped")
        else:
            components = store[key].reshape(store[key].shape[0], -1)
            mean = store[f"mean_{leg}_{args.species}"].ravel()
            # Compare the LAST year, where the perturbation is fully ramped.
            raw_before = field.values[-1].ravel().astype(np.float64)
            raw_after = (field.values[-1] * multiplier[-1]).ravel().astype(np.float64)

            # The basis lives in NORMALISED units and this file is raw, so the
            # projection here is indicative of SHAPE retention, not amplitude.
            def project(x):
                centred = x - x.mean()
                scores = components @ centred
                return components.T @ scores

            delta_raw = raw_after - raw_before
            delta_projected = project(raw_after) - project(raw_before)
            retained = (100 * delta_projected.var() / delta_raw.var()
                        if delta_raw.var() > 0 else np.nan)
            correlation = (np.corrcoef(delta_raw, delta_projected)[0, 1]
                           if delta_raw.var() > 0 else np.nan)
            leak = 100 * np.abs(delta_projected[~inside.ravel()]).sum() / \
                   max(np.abs(delta_projected).sum(), 1e-30)
            print(f"         [effective] variance retained after the "
                  f"{components.shape[0]}-EOF projection: {retained:.1f}%")
            print(f"         [effective] shape correlation raw vs projected: "
                  f"{correlation:.3f}")
            print(f"         [effective] {leak:.1f}% of the projected change "
                  f"lands OUTSIDE the region you perturbed")
    dataset.close()
