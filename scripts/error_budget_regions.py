#!/usr/bin/env python3
"""Where does each unseen scenario's error live, and does it clear the noise?

Analysis A of the error-attribution design. Free: reads the existing eval
NetCDFs and the CESM2 reference files, no GPU. Its job is to decide whether
the expensive parts (channel swaps, regional swaps) are worth running -- if a
scenario's error is diffuse or below the noise floor, attributing it to
regions and emissions would be fitting noise.

THE ERROR IS DIFFERENTIAL. The emulator's TOTAL anomalies are accurate to
0-10% in every scenario; what fails is DIFFERENCES between scenarios (the
aerosol-removal signal degrades 0.98 -> 0.36 K by 2070). So two errors are
reported per region:

    raw          model - CESM2 for this scenario
    differential (model - CESM2)_scenario - (model - CESM2)_ssp370

The differential is the one that isolates what is special about this scenario
rather than re-measuring the common bias. For PRECT it matters even more: the
ITCZ dipole appears in every scenario at every sigma, so a raw attribution
would hand every region credit for the same bias.

EVERY NUMBER CARRIES ITS NOISE FLOOR. The CESM2 side is a small ensemble --
ssp126/ssp245/ssp370 have 3 members, RAMIP tas has ONE, RAMIP pr has 10 -- so
the reference mean itself is uncertain by sd/sqrt(n). A region only counts if
its error exceeds that. With one member, "sd across members" does not exist,
so RAMIP tas falls back to a spatial estimate and is flagged: its regional
numbers are expected to be noise-limited, and that is a result, not a failure.

    ~/miniconda3/envs/plotting/bin/python scripts/error_budget_regions.py \
        --eval-dir ~/mnt/lumi_sc/eval_output/manual/mmlin_ep630_ens25
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import BASELINE

REF = os.path.expanduser("~/mnt/lumi_sc2/emulator_data/cmip6")
# key: (reference stem, window, ref var per model var)
SCEN = {
    "ssp370":        ("ssp370",        (2091, 2100)),
    "ssp126":        ("ssp126",        (2091, 2100)),
    "ssp245":        ("ssp245",        (2091, 2100)),
    "ssp370-126aer": ("ssp370-126aer", (2070, 2079)),
}
REFVAR = {"TREFHT": "tas", "PRECT": "pr"}
# RAMIP precipitation has its own 10-member download under a ramip_ prefix;
# temperature still resolves to the 1-member CMIP6 file.
STEM_OVERRIDE = {("ssp370-126aer", "PRECT"): "ramip_ssp370-126aer",
                 # ssp370 precipitation came with the RAMIP download, under the
                 # ramip_ prefix. Without this the ssp370 reference is missing
                 # and EVERY precipitation differential comes out nan -- which
                 # it did on the first run, leaving the raw PRECT numbers
                 # unable to separate the scenario's own error from the common
                 # ITCZ bias.
                 ("ssp370", "PRECT"): "ramip_ssp370"}


def emu(eval_dir, var, key, window):
    """Emulator final-decade anomaly vs its OWN 1850-1900, and member spread."""
    with xr.open_dataset(f"{eval_dir}/{var}_{key}.nc") as ds:
        da = ds[f"{var}_model"]
        yrs = np.asarray(ds["year"].values).astype(int)
        sel = np.where((yrs >= window[0]) & (yrs <= window[1]))[0]
        has_m = "member" in da.dims
        fin_m = np.asarray(da.isel(year=sel).mean("year").values, float)
        if not has_m:
            fin_m = fin_m[None]
        lat, lon = ds["lat"].values, ds["lon"].values
    with xr.open_dataset(f"{eval_dir}/{var}_hist.nc") as ds:
        da = ds[f"{var}_model"]
        yrs = np.asarray(ds["year"].values).astype(int)
        b = np.where((yrs >= BASELINE[0]) & (yrs <= BASELINE[1]))[0]
        dims = ["year"] + (["member"] if "member" in da.dims else [])
        base = np.asarray(da.isel(year=b).mean(dims).values, float)
    return fin_m - base, lat, lon


def ref(var, key, window):
    """CESM2 final-decade anomaly per member (n, H, W), or None."""
    stem = STEM_OVERRIDE.get((key, var), SCEN[key][0])
    suf = "" if var == "TREFHT" else "_pr"
    p = f"{REF}/{stem}{suf}.nc"
    if not os.path.exists(p):
        return None
    with xr.open_dataset(p) as ds:
        da = ds[REFVAR[var]]
        yrs = np.asarray(ds["year"].values).astype(int)
        sel = np.where((yrs >= window[0]) & (yrs <= window[1]))[0]
        if sel.size == 0:
            return None
        fin = np.asarray(da.isel(year=sel).mean("year").values, float)
        if "member" not in da.dims:
            fin = fin[None]
    hp = f"{REF}/historical{suf}.nc"
    if os.path.exists(hp):
        with xr.open_dataset(hp) as ds:
            da = ds[REFVAR[var]]
            yrs = np.asarray(ds["year"].values).astype(int)
            b = np.where((yrs >= BASELINE[0]) & (yrs <= BASELINE[1]))[0]
            dims = ["year"] + (["member"] if "member" in da.dims else [])
            base = np.asarray(da.isel(year=b).mean(dims).values, float)
    else:
        # No CMIP6 historical for this variable; use the LENS2 1850-1900 mean
        # the paper maps already use. Different ensemble, same forcing.
        for c in (f"plots/ensmean_maps/cache_{var}_10y_v2.npz",
                  f"plots/ensmean_maps/cache_{var}_10y.npz"):
            if os.path.exists(c):
                base = np.asarray(np.load(c, allow_pickle=True)["ref"].item()
                                  ["hist"]["base"], float)
                break
        else:
            return None
    if var == "PRECT":                      # detect kg m-2 s-1, do not assume
        if np.nanmax(np.abs(fin)) < 0.01:
            fin = fin * 86400.0
        if np.nanmax(np.abs(base)) < 0.01:
            base = base * 86400.0
    out = fin - base
    if var == "TREFHT" and np.nanmean(np.abs(out)) > 50:   # K vs degC guard
        out = out - 273.15
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir",
                    default=os.path.expanduser("~/mnt/lumi_sc/eval_output/manual/mmlin_ep630_ens25"))
    ap.add_argument("--var", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="plots/error_budget")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import regionmask
    ar6 = regionmask.defined_regions.ar6.land

    rows_all = []
    for var in args.var:
        print(f"\n{'='*78}\n{var}\n{'='*78}")
        # ssp370 is the reference the differential error is taken against.
        # The ssp370 reference window must be one its FILE covers. For
        # precipitation that reference is ramip_ssp370_pr.nc, which stops at
        # 2079, so asking for 2091-2100 selects nothing and the differential
        # silently becomes nan -- which is exactly what happened on the first
        # two runs. Fall back to the latest decade the file supports and SAY
        # which window the common bias was estimated over, since it is then
        # not the same decade as the scenario it is subtracted from.
        w370 = SCEN["ssp370"][1]
        r370 = ref(var, "ssp370", w370)
        if r370 is None:
            for alt in ((2070, 2079), (2060, 2069), (2050, 2059)):
                r370 = ref(var, "ssp370", alt)
                if r370 is not None:
                    w370 = alt
                    print(f"  [ssp370 ref] {SCEN['ssp370'][1]} unavailable for "
                          f"{var}; common bias estimated over {alt[0]}-{alt[1]}")
                    break
        e370, lat, lon = emu(args.eval_dir, var, "ssp370", w370)
        if r370 is None:
            print("  no ssp370 reference at any window — differential unavailable")
        base_err = None if r370 is None else e370.mean(0) - r370.mean(0)

        # regionmask wants a coords object, and a MONOTONIC lon -- the
        # ((lon+180)%360)-180 wrap is not monotonic, which is what made the
        # positional form fail with a broadcast error. The 0-360 convention
        # from the files is already sorted and regionmask handles it.
        grid = xr.Dataset(coords={"lon": ("lon", lon), "lat": ("lat", lat)})
        mask = ar6.mask(grid).values                            # (lat, lon) ids
        w = np.cos(np.deg2rad(lat))[:, None]

        for key in ("ssp126", "ssp245", "ssp370-126aer"):
            em, _, _ = emu(args.eval_dir, var, key, SCEN[key][1])
            rf = ref(var, key, SCEN[key][1])
            if rf is None:
                print(f"\n-- {key}: no CESM2 reference, skipped"); continue
            n = rf.shape[0]
            err = em.mean(0) - rf.mean(0)
            # Noise floor on the CESM2 mean. With ONE member there is no
            # across-member sd, so fall back to a spatial high-pass estimate
            # and say so -- silently reporting a floor of zero would make
            # every region look significant.
            if n > 1:
                floor = rf.std(0, ddof=1) / np.sqrt(n)
                fnote = f"sd/sqrt({n})"
            else:
                from scipy.ndimage import uniform_filter
                sm = uniform_filter(rf[0], size=9, mode="nearest")
                floor = np.full_like(err, float(np.std(rf[0] - sm)))
                fnote = "1 member: spatial high-pass proxy (NOISE-LIMITED)"
            derr = None if base_err is None else err - base_err

            print(f"\n-- {key}  ({n} CESM2 member{'s' if n > 1 else ''}, "
                  f"noise floor = {fnote})")
            recs = []
            for rid in np.unique(mask[~np.isnan(mask)]):
                m = (mask == rid)
                if m.sum() < 5:
                    continue
                ww = np.broadcast_to(w, err.shape)[m]
                e = float(np.average(err[m], weights=ww))
                f = float(np.average(floor[m], weights=ww))
                d = (float(np.average(derr[m], weights=ww))
                     if derr is not None else np.nan)
                # share of the global weighted squared error
                contrib = float((np.broadcast_to(w, err.shape)[m] * err[m]**2).sum())
                recs.append((ar6[int(rid)].name, e, d, f, abs(e)/f if f > 0 else np.inf, contrib))
            tot = sum(r[5] for r in recs) or 1.0
            recs.sort(key=lambda r: -r[5])
            print(f"   {'region':<26} {'raw err':>8} {'diff err':>9} {'floor':>8} "
                  f"{'|err|/floor':>11} {'share%':>7}")
            for nm, e, d, f, s, c in recs[:args.top]:
                print(f"   {nm:<26} {e:+8.3f} {d:+9.3f} {f:8.3f} {s:11.1f} "
                      f"{100*c/tot:7.1f}")
            res = sum(1 for r in recs if r[4] > 2)
            print(f"   regions clearing 2x the noise floor: {res} of {len(recs)}")
            for nm, e, d, f, s, c in recs:
                rows_all.append(dict(var=var, scenario=key, region=nm, raw=e,
                                     diff=d, floor=f, ratio=s, share=100*c/tot,
                                     n_ref=n))

    import csv
    p = os.path.join(args.out, "error_budget_regions.csv")
    with open(p, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows_all[0]))
        wtr.writeheader(); wtr.writerows(rows_all)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
