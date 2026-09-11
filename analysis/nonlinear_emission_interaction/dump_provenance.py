#!/usr/bin/env python3
"""Do the cond files the emulator reads agree with raw input4MIPs?

Three levels of the same quantity, so a discrepancy can be located rather than
just noticed:

  RAW    input4MIPs, kg m-2 s-1, monthly, per sector, on the native 0.5 deg
         grid. Summed over sectors, annual-mean, then area-integrated to a
         global total in Tg/yr.
  COND   the *_bc_co2fix.nc files the training config actually points at, on
         the 192x288 model grid, area-integrated the same way.
  SEEN   the same cond after normalize(), i.e. the model's input. Included
         because that is where the clip lives.

The recorded expectation is that RAW and COND differ by a roughly constant
factor of ~4.7: the regrid to 192x288 treats an extensive field as intensive
and does not conserve the integral. That is a known, deliberate property -- the
emulator is self-consistent in its own units -- so this script exists to CHECK
the factor is constant, not to "fix" it. A constant offset is harmless
bookkeeping; a drifting or scenario-dependent one would mean the cond files
misrepresent the forcing trajectory, which is a different and much worse thing.

    python3 analysis/nonlinear_emission_interaction/dump_provenance.py CKPT
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

# eval_aero imports torch, so defer it: without a checkpoint this script is
# pure xarray and runs anywhere the files are visible.
def build_cond_tensor(*a, **k):
    from eval_aero import build_cond_tensor as _f
    return _f(*a, **k)

COND = "/scratch/project_462001328/emulator_data"
RAW = f"{COND}/emission_data/inputs4mips"

# raw species -> the cond channel it feeds. SO2 becomes the SUL channel.
#
# THE VINTAGE MUST BE PINNED PER SPECIES, not globbed. A bare CMIP_CEDS* match
# picks up BOTH CEDS-2017-05-18 and CEDS-CMIP-2025-04-18 and concatenates them,
# which double-counts 1850-2014 (first run of this script reported 439 years of
# SO2 for a 274-year record). The two vintages disagree by ~37% over the
# post-1950 period, so this is not a cosmetic duplicate.
#
# Which vintage is CORRECT differs by species, and matches how the cond files
# were built: historical BC is CEDS-2025, SO2 is CEDS-2017. That asymmetry is
# the recorded BC vintage mismatch, not a mistake here.
PAIRS = [("BC", "BC"), ("SO2", "SUL"), ("CO2", "CO2")]

# CO2 needs three things the aerosols do not:
#   * it is surface anthro PLUS aircraft (make_co2_files_ssp.py:20-21),
#   * the cond channel is CUMULATIVE, so the raw annual rate must be cumsummed
#     before it can be compared at all,
#   * the vintage the builder asks for, CEDS-2017-05-18, DOES NOT EXIST
#     anywhere -- not on either mount, not in local staging. The closest
#     available surface vintage is CEDS-CMIP-2024-11-25, so CO2 is compared
#     against a DIFFERENT vintage than was used to build the cond, and its
#     ratio must be read with that caveat.
CO2_AIR_VINTAGE = "CMIP_CEDS-CMIP-2024-10-21"
CUMULATIVE = {"CO2"}
RAW_VINTAGE = {
    ("hist", "BC"):  "CMIP_CEDS-CMIP-2025-04-18",
    ("hist", "SO2"): "CMIP_CEDS-2017-05-18",
    ("hist", "CO2"): "CMIP_CEDS-CMIP-2024-11-25",   # 2017 does not exist
}
SCEN = {
    "hist":   dict(cond=f"{COND}/emissions_hist_only_timefixed_bc_co2fix.nc",
                   raw="CMIP_CEDS*"),
    # ScenarioMIP anthro files are stored as DECADAL means, so ~10 timesteps
    # over 2015-2100 is the file's own resolution, not a sampling error.
    "ssp370": dict(cond=f"{COND}/emissions_ssp370_only_timefixed_bc_co2fix.nc",
                   raw="ScenarioMIP_IAMC-AIM-ssp370*"),
}

ap = argparse.ArgumentParser(description=__doc__)
# The RAW and COND levels -- and so the RATIO row, which is the whole test --
# need only xarray, so this runs locally against the sshfs mount instead of
# queueing on LUMI. The SEEN level needs torch for the checkpoint's COND_NORM
# and is skipped when it is unavailable; the figure does not use it.
ap.add_argument("checkpoint", nargs="?", default=None)
ap.add_argument("--raw-root", default=None,
                help="where the input4MIPs files live, if not under "
                     "<data-root>/emission_data/inputs4mips. The matched "
                     "CEDS-2017 pair for BOTH species is in "
                     "/home/nordling/data_staging/inputs4mips; the LUMI copy "
                     "the cond files were built from has BC 2025 against SO2 "
                     "2017, which confounds the BC comparison.")
ap.add_argument("--bc-vintage", default="CMIP_CEDS-CMIP-2025-04-18",
                help="historical BC vintage to compare against. The builder "
                     "(make_aerosol_files.py:69) asks for CMIP_CEDS-2017-05-18, "
                     "which is absent on LUMI -- hence the default reflecting "
                     "what is actually there, and this switch for when the "
                     "matched pair is available.")
ap.add_argument("--data-root", default=None,
                help="override the emulator_data root, e.g. the sshfs mount "
                     "/home/nordling/mnt/lumi_sc2/emulator_data")
ap.add_argument("--data-config", default="configs/config_data_ybias_BCprect.yaml")
ap.add_argument("--out", default=os.path.join(_HERE, "results", "provenance.npz"))
args = ap.parse_args()

if args.data_root:
    COND = args.data_root
    RAW = f"{COND}/emission_data/inputs4mips"
    for k in SCEN:
        SCEN[k]["cond"] = os.path.join(
            COND, os.path.basename(SCEN[k]["cond"]))
    print(f"[prov] data root overridden -> {COND}")
if args.raw_root:
    RAW = args.raw_root
    print(f"[prov] raw root overridden  -> {RAW}")
RAW_VINTAGE[("hist", "BC")] = args.bc_vintage
print(f"[prov] hist BC vintage: {args.bc_vintage}   "
      f"hist SO2 vintage: {RAW_VINTAGE[('hist', 'SO2')]}"
      + ("   MATCHED" if "2017" in args.bc_vintage else
         "   MISMATCHED -- BC comparison is confounded"))

WANT_SEEN = args.checkpoint is not None
n_comp = None
if WANT_SEEN:
    import torch  # noqa: E402
    from omegaconf import OmegaConf  # noqa: E402
    from data.climate_dataset import set_minmax_override  # noqa: E402

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cn = ck.get("COND_NORM")
    if not cn:
        sys.exit("[error] checkpoint has no COND_NORM")
    set_minmax_override(cn)
    del ck
    n_comp = OmegaConf.load(args.data_config).get("n_components_cond", None)
else:
    print("[prov] no checkpoint given — RAW and COND only, skipping SEEN "
          "(the figure does not plot it)")
CVARS = ["CO2", "SUL", "BC"]
SIG = [0.0, 2.0, 2.0]

R_EARTH = 6.371e6
SEC_PER_YR = 365.25 * 24 * 3600


def cell_area(lat, lon):
    """Spherical cell areas [m2] from cell centres, assuming a regular grid."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlon = np.deg2rad(abs(float(np.median(np.diff(lon)))))
    dlat = np.deg2rad(abs(float(np.median(np.diff(lat)))))
    return (R_EARTH ** 2 * dlon *
            (np.sin(np.deg2rad(lat) + dlat / 2) -
             np.sin(np.deg2rad(lat) - dlat / 2)))[:, None] * np.ones(len(lon))


payload = {}
for scen, cfg in SCEN.items():
    # ── RAW ────────────────────────────────────────────────────────────────
    for sp, chan in PAIRS:
        tag = RAW_VINTAGE.get((scen, sp), cfg["raw"])
        pat = f"{RAW}/{sp}-em-anthro_input4MIPs_emissions_{tag}_gn_*.nc"
        files = sorted(glob.glob(pat))
        vints = sorted({re.search(r"emissions_(.+?)_gn_", os.path.basename(f)).group(1)
                        for f in files})
        if len(vints) > 1:
            sys.exit(f"[error] {scen} {sp}: glob spans {len(vints)} vintages "
                     f"{vints} — pin one in RAW_VINTAGE or the record is "
                     f"double-counted")
        if not files:
            print(f"[prov] {scen} {sp}: no raw files matching {os.path.basename(pat)}")
            continue
        if sp == "CO2":
            air_tag = (CO2_AIR_VINTAGE if scen == "hist" else cfg["raw"])
            files = files + sorted(glob.glob(
                f"{RAW}/CO2-em-AIR-anthro_input4MIPs_emissions_{air_tag}_gn_*.nc"))
            print(f"[prov] {scen} CO2: {len(files)} files incl. aircraft")
        acc = {}
        for f in files:
            d = xr.open_dataset(f, decode_times=True)
            v = [k for k in d.data_vars
                 if k.endswith("_em_anthro") or k.endswith("_em_AIR_anthro")]
            if not v:
                d.close(); continue
            a = d[v[0]]
            if "sector" in a.dims:
                a = a.sum("sector")               # all sectors, kg m-2 s-1
            for extra_dim in ("level", "lev", "plev"):
                if extra_dim in a.dims:           # aircraft is 3-D
                    a = a.sum(extra_dim)
            area = cell_area(d["lat"].values, d["lon"].values)
            # kg m-2 s-1 -> Tg/yr : * area * seconds, /1e9
            g = (a * xr.DataArray(area, dims=("lat", "lon"))).sum(("lat", "lon"))
            g = g * SEC_PER_YR / 1e9
            y = a["time"].dt.year.values
            for yy in np.unique(y):
                val = float(g.values[y == yy].mean())
                if int(yy) in acc:
                    acc[int(yy)] += val           # surface + aircraft
                else:
                    acc[int(yy)] = val
            d.close()
        yrs = sorted(acc); tot = [acc[k] for k in yrs]
        o = np.argsort(yrs)
        yy_, tt_ = np.array(yrs)[o], np.array(tot)[o]
        if chan in CUMULATIVE:
            # The cond CO2 channel is CUMULATIVE from 1850, so the raw ANNUAL
            # rate has to be integrated before the two are the same quantity.
            tt_ = np.cumsum(tt_)
        payload[f"raw_years_{scen}_{chan}"] = yy_
        payload[f"raw_tg_{scen}_{chan}"] = tt_
        uy = np.unique(np.array(yrs))
        if len(uy) != len(yrs):
            sys.exit(f"[error] {scen} {sp}: {len(yrs)} entries for {len(uy)} "
                     f"unique years — duplicated record")
        print(f"[prov] {scen} {sp}->{chan}: {len(files)} files "
              f"[{vints[0]}], {len(o)} yr {uy.min()}-{uy.max()}, "
              f"{np.array(tot)[o][-1]:.1f} Tg/yr at {np.array(yrs)[o][-1]}")

    # ── COND, un-normalised and normalised ─────────────────────────────────
    raw_ds = xr.open_dataset(cfg["cond"])
    tname = "time" if "time" in raw_ds.variables else "year"
    cy = raw_ds[tname].values
    cy = (np.array([int(str(x)[:4]) for x in cy]) if cy.dtype.kind == "M"
          else cy.astype(int))
    ca = cell_area(raw_ds["lat"].values, raw_ds["lon"].values)
    for ch in CVARS:
        if ch not in raw_ds:
            continue
        arr = raw_ds[ch].values                       # (time, lat, lon)
        payload[f"cond_years_{scen}"] = cy
        payload[f"cond_int_{scen}_{ch}"] = (arr * ca).sum((1, 2))
    raw_ds.close()

    if not WANT_SEEN:
        continue
    t, yy, lat, lon = build_cond_tensor(cfg["cond"], CVARS, "time", None,
                                        n_comp, cond_smooth_sigma=SIG,
                                        cond_smooth_method="gaussian")
    t = np.asarray(t)
    w = np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat)))[:, None], t.shape[-2:])
    for i, ch in enumerate(CVARS):
        payload[f"seen_years_{scen}"] = np.asarray(yy)
        payload[f"seen_gmean_{scen}_{ch}"] = np.array(
            [np.average(t[i, k], weights=w) for k in range(t.shape[1])])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, **payload, allow_pickle=True)
print(f"[prov] wrote {args.out}")
