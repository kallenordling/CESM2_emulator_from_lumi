#!/usr/bin/env python3
"""
Paper figure: spatial bias maps, emulator minus held-out CESM2.

    rows    = variables (temperature, precipitation)
    columns = experiments (historical, SSP3-7.0, AAER, GHG)

Eight panels in one figure. The emulator and CESM2 FIELDS are deliberately not
shown: emulator = CESM2 + difference, so two of the three are complete, and the
timeseries and distribution figures already establish that the response and the
spread are right. What remains to answer is whether the residual is spatially
STRUCTURED or just noise — which is what a bias map shows.

Each panel is annotated with the area-weighted pattern correlation between the
emulated and CESM2 response and the area-weighted RMSE. Those numbers are what
make omitting the field panels safe: r states quantitatively that the pattern is
reproduced, while the map shows where the residual sits.

STIPPLING
---------
Grid points where |bias| is BELOW CESM2's own inter-member spread are hatched:
there the emulator differs from CESM2 by less than CESM2 members differ from
each other, so the difference is not distinguishable from internal variability.
For a well-performing emulator most of the map is hatched, which is the result.

WHAT IS DIFFERENCED
-------------------
Final-decade mean ANOMALY vs each side's own 1850-1900 climatology, so the map
shows RESPONSE error rather than any climatological mean-state offset —
consistent with the other two figures. --absolute differences the raw fields
instead, which includes the climatological bias.

Precipitation maps default to mm/day, NOT percent. Percent change is standard for
global means (and is what paper_fig_timeseries.py uses) but is unusable per grid
point: dividing by a baseline that approaches zero over deserts produces enormous
meaningless values. --percent enables it with a --percent-floor mask.

Usage
-----
    python scripts/paper_fig_maps.py \\
        --eval-dir /path/to/eval_output/run_mseyb_BCprect/best_ep0490
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASELINE = (1850, 1900)

VARS = {
    "TREFHT": dict(row="Temperature", unit="°C", unit_plain="degC",
                   cmap="RdBu_r", vmax=1.0,
                   tree_scale={None: 1.0, "K": 1.0, "degC": 1.0},
                   tree_offset={"K": -273.15, "degC": 0.0, None: 0.0}),
    "PRECT":  dict(row="Precipitation", unit="mm day$^{-1}$", unit_plain="mm/day",
                   cmap="BrBG", vmax=0.5, vmax_pct=20.0,
                   tree_scale={"m/s": 86400.0 * 1000.0, "mm/day": 1.0, None: 1.0},
                   tree_offset={"m/s": 0.0, "mm/day": 0.0, None: 0.0}),
}

SCEN = {
    "hist":   ("Historical",                "hist",   "hist"),
    "ssp370": ("SSP3-7.0",                  "ssp370", "ssp370"),
    "aaer":   ("Aerosol-only (AAER)",       "aaer",   "AAER"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "ghg",    "GHG"),
}


def unseen_members(tree_root: Path, subdir: str, trained: set) -> list:
    d = tree_root / subdir
    have = {p.name for p in d.iterdir() if p.is_dir() and p.name != "diagnostics"}
    return sorted(have - set(trained))


def area_w(lat, lon):
    return np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat)))[:, None],
                           (len(lat), len(lon)))


def emulator_fields(nc_path: Path, var: str, n_years: int):
    """(decadal member-mean map, baseline map, lat, lon, n_members, years)."""
    ds = xr.open_dataset(nc_path)
    years = ds["year"].values.astype(int)
    names = sorted([v for v in ds.data_vars
                    if v.startswith(f"{var}_model_m") and not v.endswith("_anom")
                    and not v.startswith(f"{var}_model_mean")],
                   key=lambda x: int(x.rsplit("_m", 1)[1]))
    if not names:
        raise KeyError(f"{nc_path}: no per-member {var}_model_m* fields")
    keep = years >= years.max() - n_years + 1
    M = np.stack([ds[n].values for n in names])            # (mem, yr, lat, lon)
    bmask = (years >= BASELINE[0]) & (years <= BASELINE[1])
    base = M[:, bmask].mean(axis=(0, 1)) if bmask.any() else None
    dec = M[:, keep].mean(axis=1)                          # (mem, lat, lon)
    lat, lon = ds["lat"].values, ds["lon"].values
    ds.close()
    return dec, base, lat, lon, len(names), (int(years[keep].min()),
                                             int(years[keep].max()))


def cesm_fields(tree_root: Path, subdir: str, members: list, var: str,
                n_years: int, want_baseline: bool):
    """Same, from the held-out training-tree members, converted to model units."""
    meta = VARS[var]
    decs, bases, lat, lon, yr = [], [], None, None, None
    for i, mem in enumerate(members, 1):
        files = sorted((tree_root / subdir / mem).glob("*.nc"))
        if not files:
            print(f"      [{i}/{len(members)}] {mem}: no chunks, skipped", flush=True)
            continue
        ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False,
                               chunks={"time": 4})
        if var not in ds:
            raise KeyError(f"{tree_root/subdir/mem}: no {var!r} "
                           f"(--tree-root must point at the {var} tree)")
        u = ds[var].attrs.get("units")
        sc = meta["tree_scale"].get(u)
        off = meta["tree_offset"].get(u, 0.0)
        if sc is None:
            raise ValueError(f"{mem}: {var} units {u!r} have no conversion to "
                             f"{meta['unit_plain']}")
        tdim = "time" if "time" in ds[var].dims else "year"
        years = np.asarray(ds[tdim].values).astype(int)
        da = ds[var] * sc + off
        keep = np.where(years >= years.max() - n_years + 1)[0]
        decs.append(da.isel({tdim: keep}).mean(tdim).values)
        if want_baseline:
            b = np.where((years >= BASELINE[0]) & (years <= BASELINE[1]))[0]
            if len(b):
                bases.append(da.isel({tdim: b}).mean(tdim).values)
        if lat is None:
            lat, lon = ds["lat"].values, ds["lon"].values
        yr = (int(years[keep].min()), int(years[keep].max()))
        print(f"      [{i}/{len(members)}] {subdir}/{mem} {yr[0]}-{yr[1]}"
              + (f" (+{BASELINE[0]}-{BASELINE[1]})" if want_baseline else ""),
              flush=True)
        ds.close()
    base = np.mean(bases, axis=0) if bases else None
    return np.stack(decs), base, lat, lon, len(decs), yr


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Spatial bias maps: emulator minus held-out CESM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--data-root",
                    default="/home/nordling/mnt/lumi_sc2/emulator_data")
    ap.add_argument("--data-config",
                    default="configs/config_data_ybias_BCprect.yaml")
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"],
                    choices=sorted(VARS))
    ap.add_argument("--scenarios", nargs="+", default=list(SCEN))
    ap.add_argument("--n-years", type=int, default=10)
    ap.add_argument("--n-ref-members", type=int, default=5)
    ap.add_argument("--absolute", action="store_true",
                    help="difference the raw fields instead of anomalies "
                         "(includes the climatological mean-state bias)")
    ap.add_argument("--precip-mm", action="store_true",
                    help="precipitation bias in mm/day instead of the default "
                         "%% of baseline (percent matches the timeseries figure "
                         "but needs the --percent-floor mask over arid regions)")
    ap.add_argument("--percent-floor", type=float, default=0.5,
                    help="mask grid points whose baseline precip (mm/day) is "
                         "below this before dividing; without it, dividing by a "
                         "near-zero desert baseline produces meaningless values")
    ap.add_argument("--no-stipple", action="store_true")
    ap.add_argument("--out", default="plots/paper_fig_maps.png")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.data_config))
    trained = {e["scenario_name"]: set(e.get("realizations", []))
               for e in cfg["experiment_configs"]}
    eval_dir = Path(args.eval_dir)

    # ── gather ──────────────────────────────────────────────────────────────
    F = {}
    for var in args.vars:
        tree_root = Path(args.data_root) / "training_data" / var
        # hist supplies the baseline for scenarios that have no pre-industrial
        hist_base = {"emu": None, "cesm": None}
        for sc in ["hist"] + [s for s in args.scenarios if s != "hist"]:
            if sc not in SCEN:
                continue
            label, ncname, sub = SCEN[sc]
            p = eval_dir / f"{var}_{ncname}.nc"
            if not p.exists():
                print(f"[skip] {var}/{sc}: {p} not found")
                continue
            print(f"\n[{var}/{sc}] emulator …", flush=True)
            edec, ebase, lat, lon, n_emu, eyr = emulator_fields(
                p, var, args.n_years)
            mems = unseen_members(tree_root, sub, trained.get(sc, set()))
            if args.n_ref_members > 0:
                mems = mems[:args.n_ref_members]
            print(f"[{var}/{sc}] CESM2 held-out {mems} …", flush=True)
            need_base = sc == "hist" or hist_base["cesm"] is None
            cdec, cbase, clat, clon, n_c, cyr = cesm_fields(
                tree_root, sub, mems, var, args.n_years,
                want_baseline=(sc != "ssp370"))
            if sc == "hist":
                hist_base = {"emu": ebase, "cesm": cbase}
            if ebase is None:
                ebase = hist_base["emu"]
            if cbase is None:
                cbase = hist_base["cesm"]
            F[(var, sc)] = dict(edec=edec, ebase=ebase, cdec=cdec, cbase=cbase,
                                lat=lat, lon=lon, n_emu=n_emu, n_c=n_c,
                                yr=eyr, label=label)
            print(f"[{var}/{sc}] {eyr[0]}-{eyr[1]}  "
                  f"{n_emu} emulator / {n_c} CESM2 members")

    if not F:
        print("no data", file=sys.stderr)
        return 1

    # ── plot ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAVE_CARTOPY = True
    except ImportError:
        HAVE_CARTOPY = False
        print("\n[warn] cartopy not available — plotting without coastlines. "
              "It lives in the 'plotting' conda env.")

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 9.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    nrow, ncol = len(args.vars), len([s for s in args.scenarios if s in SCEN])
    proj = dict(projection=ccrs.Robinson(central_longitude=0)) if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.15 * nrow + 0.9),
                             subplot_kw=proj, squeeze=False)

    rows = []
    for r, var in enumerate(args.vars):
        meta = VARS[var]
        pct = (var == "PRECT") and not args.precip_mm
        unit = "%" if pct else meta["unit"]
        # common colour scale across the row so panels are comparable
        fields, masked_pct = {}, {}
        for sc in args.scenarios:
            if (var, sc) not in F:
                continue
            d = F[(var, sc)]
            if args.absolute:
                eA = d["edec"].mean(axis=0)
                cM = d["cdec"]
                cA = cM.mean(axis=0)
                spread = cM.std(axis=0, ddof=1)
            else:
                eA = d["edec"].mean(axis=0) - d["ebase"]
                cM = d["cdec"] - d["cbase"]
                cA = cM.mean(axis=0)
                spread = cM.std(axis=0, ddof=1)
            if pct:
                dry = d["cbase"] < args.percent_floor
                base = np.where(dry, np.nan, d["cbase"])
                eA, cA, spread = (100 * eA / base, 100 * cA / base,
                                  100 * spread / base)
                wgt = area_w(d["lat"], d["lon"])
                masked_pct[sc] = 100 * float(np.average(dry.astype(float),
                                                        weights=wgt))
            fields[sc] = (eA, cA, spread, d)
        if not fields:
            continue
        vmax = float(np.nanpercentile(
            np.abs(np.concatenate([(e - c).ravel() for e, c, _, _ in fields.values()])),
            99)) or (meta.get("vmax_pct") if pct else meta["vmax"])

        im = None
        for c, sc in enumerate([s for s in args.scenarios if s in fields]):
            eA, cA, spread, d = fields[sc]
            ax = axes[r][c]
            bias = eA - cA
            w = area_w(d["lat"], d["lon"])
            m = np.isfinite(bias)
            rmse = float(np.sqrt(np.average(bias[m] ** 2, weights=w[m])))
            # area-weighted pattern correlation of the two responses
            ew = np.average(eA[m], weights=w[m]); cw_ = np.average(cA[m], weights=w[m])
            cov = np.average((eA[m] - ew) * (cA[m] - cw_), weights=w[m])
            corr = cov / np.sqrt(np.average((eA[m] - ew) ** 2, weights=w[m])
                                 * np.average((cA[m] - cw_) ** 2, weights=w[m]))
            frac = float(np.average((np.abs(bias[m]) < spread[m]).astype(float),
                                    weights=w[m])) * 100

            kw = dict(cmap=meta["cmap"], vmin=-vmax, vmax=vmax, shading="auto")
            if HAVE_CARTOPY:
                kw["transform"] = ccrs.PlateCarree()
            im = ax.pcolormesh(d["lon"], d["lat"], bias, **kw)
            if not args.no_stipple:
                # hatch where the emulator differs from CESM2 by LESS than
                # CESM2 members differ from each other
                hk = dict(colors="none", hatches=["....", ""], levels=[0.5, 1.5, 2.5])
                if HAVE_CARTOPY:
                    hk["transform"] = ccrs.PlateCarree()
                ax.contourf(d["lon"], d["lat"],
                            (np.abs(bias) < spread).astype(float) + 1.0, **hk)
            if HAVE_CARTOPY:
                ax.coastlines(linewidth=0.35, color="0.25")
                ax.set_global()
            ax.set_title(f"{d['label']}" if r == 0 else "", fontsize=9.5)
            _msk = masked_pct.get(sc)
            ax.text(0.5, -0.10,
                    f"r = {corr:.3f}   RMSE = {rmse:.3f} {unit}\n"
                    f"{frac:.0f}% within CESM2 spread"
                    + (f"   ({_msk:.0f}% arid, masked)" if _msk else ""),
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
                    color="0.25")
            if c == 0:
                ax.text(-0.06, 0.5, meta["row"], transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)
            rows.append(dict(var=var, scenario=sc, years=f"{d['yr'][0]}-{d['yr'][1]}",
                             n_emu=d["n_emu"], n_cesm=d["n_c"],
                             pattern_corr=round(corr, 4), rmse=round(rmse, 4),
                             pct_within_spread=round(frac, 1), unit=unit,
                             pct_area_masked=(round(masked_pct[sc], 1)
                                              if sc in masked_pct else 0.0)))

        if im is None:                     # nothing drawn in this row
            continue
        cb = fig.colorbar(im, ax=list(axes[r]), orientation="vertical",
                          fraction=0.018, pad=0.012, extend="both")
        cb.set_label(f"emulator − CESM2 ({unit})", fontsize=8.5)

    fig.suptitle(
        f"Emulator minus held-out CESM2, {args.n_years}-year mean "
        f"{'field' if args.absolute else 'anomaly vs 1850–1900'}; "
        f"hatching = |bias| below CESM2 inter-member spread"
        + ("" if args.precip_mm else
           f"; precipitation as % of its 1850–1900 baseline"),
        fontsize=10, y=0.99)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    t = pd.DataFrame(rows)
    print("\nSpatial comparison")
    print(t.to_string(index=False))
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
