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
Hatching marks grid points where the emulator differs from CESM2 SIGNIFICANTLY,
i.e. by more than the two ensembles' sampling uncertainty can explain. The test
is a Welch two-sample t-test between the n_emu emulator members and the n_cesm
held-out CESM2 members at each grid point,

    t = (mean_e - mean_c) / sqrt(s_e^2/n_e + s_c^2/n_c)

with Welch-Satterthwaite degrees of freedom. Note the denominator: the relevant
scale is the standard error of the DIFFERENCE OF TWO MEANS, ~sqrt(n) smaller
than the inter-member spread of a single member. For a well-performing emulator
most of the map is UNhatched, which is the result.

Because the test is applied at every one of ~55k grid points, a raw alpha=0.05
would flag ~5% of the map by construction. The field significance is therefore
controlled with a Benjamini-Hochberg false-discovery-rate step at q = 2*alpha
(Wilks 2016, BAMS 97:2263, the standard recommendation for gridded fields).
--no-fdr reports the uncorrected point-wise test instead.

CAVEAT worth stating in the caption: the emulator's "members" are independent
diffusion samples, not independent climate realizations. The test therefore asks
whether the emulator's sampling distribution is offset from CESM2's, which is the
right question here, but the two ensembles' spreads have different origins.

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
    # CMIP6 `pr` is a mass flux in kg m-2 s-1; 1 kg m-2 == 1 mm of water, so
    # x86400 gives mm/day. The LENS2 trees store PRECT as a velocity in m/s,
    # hence the different factor for that key.
    "PRECT":  dict(row="Precipitation", unit="mm day$^{-1}$", unit_plain="mm/day",
                   cmap="BrBG", vmax=0.5, vmax_pct=20.0,
                   tree_scale={"m/s": 86400.0 * 1000.0, "mm/day": 1.0,
                               "kg m-2 s-1": 86400.0, None: 1.0},
                   tree_offset={"m/s": 0.0, "mm/day": 0.0,
                                "kg m-2 s-1": 0.0, None: 0.0}),
}

SCEN = {
    "hist":   ("Historical",                "hist",   "hist"),
    "ssp370": ("SSP3-7.0",                  "ssp370", "ssp370"),
    "aaer":   ("Aerosol-only (AAER)",       "aaer",   "AAER"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "ghg",    "GHG"),
    # OUT-OF-TRAINING scenarios: the emulator never saw these forcing
    # combinations. They have no LENS2 training tree, so their CESM2 reference
    # comes from the CMIP6 archive instead (see CMIP6_REFS).
    "ssp126": ("SSP1-2.6 (unseen)",         "ssp126", None),
    "ssp245": ("SSP2-4.5 (unseen)",         "ssp245", None),
}

# The four scenarios with a held-out LENS2 reference — the default paper figure.
# ssp126/ssp245 are opt-in via --scenarios because their reference is a
# different ensemble with far fewer members (see CMIP6_REFS).
DEFAULT_SCENARIOS = ["hist", "ssp370", "aaer", "ghg"]

# CESM2 reference for the unseen scenarios: pre-aggregated CMIP6 ensembles on
# the native 192x288 model grid, 2015-2100, annual, (year, member, lat, lon).
# `tas` is what eval_aero.py already evaluates against (eval_aero.py:88-119).
#
# n = 3 members (r4/r10/r11) against 10-11 held-out LENS2 members: the Welch SE
# is dominated by this side, so significance is nearly unreachable — read the
# colours, not the hatching.
#
# The `_pr` files do NOT ship with the repo's data. The tas ones came out of the
# Pangeo Google-Cloud archive, which is why precipitation was missing entirely.
# Build them from ESGF, per scenario:
#   python download_cmip6_cesm2.py --experiment ssp126 --variables pr \
#          --members r4i1p1f1 r10i1p1f1 r11i1p1f1
#   python scripts/build_cmip6_annual_ref.py --experiment ssp126 --variable pr
# Until they exist, PRECT for these scenarios is skipped — or plotted as the
# emulator's own anomaly under --emulator-only. Never faked.
CMIP6_REFS = {
    "ssp126": {"TREFHT": ("cmip6/ssp126.nc", "tas"),
               "PRECT":  ("cmip6/ssp126_pr.nc", "pr")},
    "ssp245": {"TREFHT": ("cmip6/ssp245.nc", "tas"),
               "PRECT":  ("cmip6/ssp245_pr.nc", "pr")},
}


def unseen_members(tree_root: Path, subdir: str, trained: set) -> list:
    d = tree_root / subdir
    have = {p.name for p in d.iterdir() if p.is_dir() and p.name != "diagnostics"}
    return sorted(have - set(trained))


def area_w(lat, lon):
    return np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat)))[:, None],
                           (len(lat), len(lon)))


def welch_p(eM, cM):
    """Point-wise Welch two-sample t-test p-values between two member stacks.

    eM, cM are (member, lat, lon). Returns (p, t) with the same map shape.
    Welch rather than Student because the emulator and CESM2 ensembles have no
    reason to share a variance — one is diffusion sampling noise, the other is
    climate internal variability.
    """
    from scipy import stats

    ne, nc = eM.shape[0], cM.shape[0]
    if ne < 2 or nc < 2:
        raise ValueError(f"need >=2 members per side for a t-test, "
                         f"got {ne} emulator / {nc} CESM2")
    ve = eM.var(axis=0, ddof=1) / ne
    vc = cM.var(axis=0, ddof=1) / nc
    se = np.sqrt(ve + vc)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (eM.mean(axis=0) - cM.mean(axis=0)) / se
        # Welch-Satterthwaite
        dof = (ve + vc) ** 2 / (ve ** 2 / (ne - 1) + vc ** 2 / (nc - 1))
    p = np.full(t.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(dof) & (se > 0)
    p[ok] = 2.0 * stats.t.sf(np.abs(t[ok]), dof[ok])
    # se == 0 with a non-zero difference is a degenerate but real difference
    p[(se == 0) & (eM.mean(axis=0) != cM.mean(axis=0))] = 0.0
    p[(se == 0) & (eM.mean(axis=0) == cM.mean(axis=0))] = 1.0
    return p, t


def fdr_mask(p, q):
    """Benjamini-Hochberg: True where the null is rejected at FDR level q.

    Returns (mask, p_threshold). p_threshold is 0 when nothing is rejected.
    """
    flat = p[np.isfinite(p)]
    if flat.size == 0:
        return np.zeros(p.shape, dtype=bool), 0.0
    s = np.sort(flat)
    n = s.size
    below = s <= (np.arange(1, n + 1) / n) * q
    if not below.any():
        return np.zeros(p.shape, dtype=bool), 0.0
    thr = s[np.nonzero(below)[0].max()]
    return np.isfinite(p) & (p <= thr), float(thr)


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


def cesm_fields_cmip6(path: Path, ncvar: str, var: str, n_years: int):
    """Same as cesm_fields, from a pre-aggregated CMIP6 ensemble file.

    Shape (year, member, lat, lon) rather than one directory of chunks per
    member. Returns base=None always: these files start in 2015, so the
    1850-1900 baseline has to come from the historical tree — exactly as it
    already does for ssp370.
    """
    meta = VARS[var]
    ds = xr.open_dataset(path)
    if ncvar not in ds:
        raise KeyError(f"{path}: no {ncvar!r}")
    u = ds[ncvar].attrs.get("units")
    sc = meta["tree_scale"].get(u)
    off = meta["tree_offset"].get(u, 0.0)
    if sc is None:
        raise ValueError(f"{path}: {ncvar} units {u!r} have no conversion to "
                         f"{meta['unit_plain']}")
    years = np.asarray(ds["year"].values).astype(int)
    keep = np.where(years >= years.max() - n_years + 1)[0]
    da = (ds[ncvar] * sc + off).isel(year=keep).mean("year")     # (mem, lat, lon)
    da = da.transpose("member", "lat", "lon")
    dec = da.values
    lat, lon = ds["lat"].values, ds["lon"].values
    mems = [str(m) for m in ds["member"].values]
    yr = (int(years[keep].min()), int(years[keep].max()))
    print(f"      CMIP6 {path.name}: {len(mems)} members {mems} {yr[0]}-{yr[1]}",
          flush=True)
    ds.close()
    return dec, None, lat, lon, len(mems), yr


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
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                    choices=sorted(SCEN),
                    help="ssp126/ssp245 are OUT-OF-TRAINING and referenced "
                         "against 3-member CMIP6 ensembles, temperature only")
    ap.add_argument("--n-years", type=int, default=10)
    ap.add_argument("--match-members", action="store_true",
                    help="cap the emulator ensemble at the CESM2 reference's "
                         "member count per scenario. Welch's SE is dominated "
                         "by the smaller side, so 25-vs-10 buys real power; "
                         "matching answers the different question of how the "
                         "two compare at EQUAL sampling.")
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
    ap.add_argument("--emulator-only", action="store_true",
                    help="plot the emulator's OWN ensemble-mean anomaly instead "
                         "of its difference from CESM2. The only option for "
                         "PRECT under ssp126/ssp245, which have no CESM2 "
                         "precipitation reference at all; no difference means "
                         "no significance test, so nothing is hatched")
    ap.add_argument("--no-stipple", action="store_true",
                    help="omit the significance hatching")
    ap.add_argument("--allow-no-cartopy", action="store_true",
                    help="write flat lat/lon panels if cartopy is missing "
                         "instead of refusing; the default refuses so a run in "
                         "the wrong env cannot overwrite the projected figure")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="two-sided significance level for the point-wise "
                         "Welch t-test between emulator and CESM2 members")
    ap.add_argument("--no-fdr", action="store_true",
                    help="hatch the raw point-wise test instead of controlling "
                         "the false discovery rate at q=2*alpha; with ~55k grid "
                         "points the raw test flags ~alpha of the map by chance")
    ap.add_argument("--out", default="plots/paper_fig_maps.png")
    ap.add_argument("--dump-data", default=None, metavar="DIR",
                    help="write the GRIDDED fields behind each panel (emulator "
                         "anomaly, CESM2 anomaly, their difference, p-value and "
                         "significance mask) as a tidy CSV under DIR")
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
            # Resolve the CMIP6 reference for the unseen scenarios up front: the
            # file for a given variable may simply not have been built yet.
            cmip6_ref = None
            if sc in CMIP6_REFS and not args.emulator_only:
                spec = CMIP6_REFS[sc].get(var)
                if spec is not None and (Path(args.data_root) / spec[0]).exists():
                    cmip6_ref = spec
                else:
                    _rel = spec[0] if spec else f"cmip6/{sc}_<{var}>.nc"
                    print(f"[skip] {var}/{sc}: no CESM2 {var} reference for this "
                          f"unseen scenario ({_rel} not found). Build it with "
                          f"download_cmip6_cesm2.py + "
                          f"scripts/build_cmip6_annual_ref.py, or pass "
                          f"--emulator-only to plot the emulator's own anomaly.")
                    continue
            print(f"\n[{var}/{sc}] emulator …", flush=True)
            edec, ebase, lat, lon, n_emu, eyr = emulator_fields(
                p, var, args.n_years)
            if args.emulator_only:
                # No reference at all: the panel shows the emulator's own
                # anomaly, so nothing on the CESM2 side is loaded or read.
                cdec, cbase, n_c, cyr = None, None, 0, eyr
            elif cmip6_ref is not None:
                rel, ncvar = cmip6_ref
                print(f"[{var}/{sc}] CESM2 CMIP6 ensemble {rel} …", flush=True)
                cdec, cbase, clat, clon, n_c, cyr = cesm_fields_cmip6(
                    Path(args.data_root) / rel, ncvar, var, args.n_years)
            else:
                mems = unseen_members(tree_root, sub, trained.get(sc, set()))
                if args.n_ref_members > 0:
                    mems = mems[:args.n_ref_members]
                print(f"[{var}/{sc}] CESM2 held-out {mems} …", flush=True)
                cdec, cbase, clat, clon, n_c, cyr = cesm_fields(
                    tree_root, sub, mems, var, args.n_years,
                    want_baseline=(sc != "ssp370"))
            if sc == "hist":
                hist_base = {"emu": ebase, "cesm": cbase}
            if ebase is None:
                ebase = hist_base["emu"]
            if cbase is None:
                cbase = hist_base["cesm"]
            # Equal-sampling comparison: drop the emulator's surplus members so
            # neither ensemble mean is better converged than the other. The
            # BASELINE map keeps every member — it is a climatology, not a term
            # in the sampling error, and thinning it would only add noise.
            if args.match_members and n_c and n_emu > n_c:
                edec = edec[:n_c]
                n_emu = edec.shape[0]
                print(f"[{var}/{sc}] match-members: emulator capped to {n_emu}")
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
        # Fail loudly by default: without cartopy the panels silently lose the
        # Robinson projection AND the coastlines, and the result overwrites the
        # good figure with something unusable in the paper.
        if not args.allow_no_cartopy:
            print("\n[error] cartopy not available — the figure would be written "
                  "without a map projection or coastlines, overwriting a good "
                  f"{args.out}.\n        Run it with the 'plotting' env:\n"
                  "          /home/nordling/miniconda3/envs/plotting/bin/python "
                  + " ".join(sys.argv) + "\n"
                  "        or pass --allow-no-cartopy to accept flat lat/lon "
                  "panels.", file=sys.stderr)
            return 2
        print("\n[warn] cartopy not available — plotting without coastlines. "
              "It lives in the 'plotting' conda env.")

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 9.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    # Only rows/columns that actually have data get an axis: asking for PRECT
    # alongside an unseen scenario leaves that cell unreferenced, and an empty
    # axis would render as a blank white panel in the paper figure.
    plot_vars = [v for v in args.vars
                 if any((v, s) in F for s in args.scenarios)]
    plot_scen = [s for s in args.scenarios
                 if any((v, s) in F for v in plot_vars)]
    nrow, ncol = len(plot_vars), len(plot_scen)
    proj = dict(projection=ccrs.Robinson(central_longitude=0)) if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.15 * nrow + 0.9),
                             subplot_kw=proj, squeeze=False)

    rows = []
    _counts = []
    grid_dump = []          # gridded panel fields, for --dump-data
    for r, var in enumerate(plot_vars):
        meta = VARS[var]
        pct = (var == "PRECT") and not args.precip_mm
        unit = "%" if pct else meta["unit"]
        # common colour scale across the row so panels are comparable
        fields, masked_pct = {}, {}
        for sc in plot_scen:
            if (var, sc) not in F:
                continue
            d = F[(var, sc)]
            solo = d["cdec"] is None          # emulator-only: no reference
            if args.absolute:
                eM = d["edec"]
                cM = None if solo else d["cdec"]
            else:
                eM = d["edec"] - d["ebase"]
                cM = None if solo else d["cdec"] - d["cbase"]
            eA = eM.mean(axis=0)
            cA = None if solo else cM.mean(axis=0)
            spread = None if solo else cM.std(axis=0, ddof=1)
            # p-values from the UNSCALED members: the percent conversion below
            # divides both ensembles by the same per-grid-point baseline, which
            # cancels in t, so significance is identical either way.
            pval = None if solo else welch_p(eM, cM)[0]
            if pct:
                # Without a CESM2 baseline the emulator's own is the only
                # denominator available; it is the same field the anomaly is
                # taken against, so the percentage stays self-consistent.
                pbase = d["ebase"] if solo else d["cbase"]
                dry = pbase < args.percent_floor
                base = np.where(dry, np.nan, pbase)
                eA = 100 * eA / base
                if not solo:
                    cA, spread = 100 * cA / base, 100 * spread / base
                if pval is not None:
                    pval = np.where(dry, np.nan, pval)  # don't hatch masked desert
                wgt = area_w(d["lat"], d["lon"])
                masked_pct[sc] = 100 * float(np.average(dry.astype(float),
                                                        weights=wgt))
            fields[sc] = (eA, cA, spread, pval, d)
        if not fields:
            continue
        # The plotted quantity is the difference where there is a reference and
        # the emulator's own anomaly where there is not; the colour scale is set
        # from whichever is actually drawn.
        # 99th percentile suits a DIFFERENCE field, whose tail is thin. A raw
        # anomaly has a fat one — the ITCZ band alone reaches several hundred
        # percent — and letting it set the scale washes out every other feature,
        # so the emulator-only mode clips at the 95th instead and lets
        # extend="both" carry the tail.
        _q = 95 if args.emulator_only else 99
        vmax = float(np.nanpercentile(
            np.abs(np.concatenate([(e if c is None else e - c).ravel()
                                   for e, c, _, _, _ in fields.values()])),
            _q)) or (meta.get("vmax_pct") if pct else meta["vmax"])

        im = None
        # Iterate the FULL column list so column c means the same scenario in
        # every row; a scenario missing from this row hides its axis instead of
        # shifting the ones after it.
        for c, sc in enumerate(plot_scen):
            if sc not in fields:
                axes[r][c].set_visible(False)
                continue
            eA, cA, spread, pval, d = fields[sc]
            ax = axes[r][c]
            solo = cA is None
            bias = eA if solo else eA - cA
            w = area_w(d["lat"], d["lon"])
            m = np.isfinite(bias)
            if solo:
                # Nothing to compare against: every skill statistic below is a
                # comparison, so report the anomaly's own area-weighted mean and
                # leave the rest undefined rather than filling it with zeros.
                gmean = float(np.average(bias[m], weights=w[m]))
                rmse = corr = frac = np.nan
                sig = np.zeros(bias.shape, dtype=bool)
                sig_pct = raw_pct = p_thr = np.nan
            else:
                rmse = float(np.sqrt(np.average(bias[m] ** 2, weights=w[m])))
                # area-weighted pattern correlation of the two responses
                ew = np.average(eA[m], weights=w[m]); cw_ = np.average(cA[m], weights=w[m])
                cov = np.average((eA[m] - ew) * (cA[m] - cw_), weights=w[m])
                corr = cov / np.sqrt(np.average((eA[m] - ew) ** 2, weights=w[m])
                                     * np.average((cA[m] - cw_) ** 2, weights=w[m]))
                frac = float(np.average((np.abs(bias[m]) < spread[m]).astype(float),
                                        weights=w[m])) * 100
                gmean = np.nan
                # significance of the emulator-minus-CESM2 difference
                if args.no_fdr:
                    sig = np.isfinite(pval) & (pval < args.alpha)
                    p_thr = args.alpha
                else:
                    sig, p_thr = fdr_mask(pval, 2.0 * args.alpha)
                raw = np.isfinite(pval) & (pval < args.alpha)
                sig_pct = float(np.average(sig[m].astype(float), weights=w[m])) * 100
                raw_pct = float(np.average(raw[m].astype(float), weights=w[m])) * 100

            if args.dump_data:
                _lat2, _lon2 = np.meshgrid(d["lat"], d["lon"], indexing="ij")
                _n = bias.size
                _blk = dict(var=np.repeat(var, _n), scenario=np.repeat(sc, _n),
                            years=np.repeat(f"{d['yr'][0]}-{d['yr'][1]}", _n),
                            lat=_lat2.ravel(), lon=_lon2.ravel(),
                            emulator=eA.ravel(),
                            cesm2=(np.full(_n, np.nan) if solo else cA.ravel()),
                            difference=bias.ravel(),
                            p_value=(np.full(_n, np.nan) if pval is None
                                     else pval.ravel()),
                            significant=sig.ravel().astype(int),
                            unit=np.repeat(unit if pct else meta["unit_plain"], _n))
                grid_dump.append(pd.DataFrame(_blk))

            kw = dict(cmap=meta["cmap"], vmin=-vmax, vmax=vmax, shading="auto")
            if HAVE_CARTOPY:
                kw["transform"] = ccrs.PlateCarree()
            im = ax.pcolormesh(d["lon"], d["lat"], bias, **kw)
            if not args.no_stipple and not solo:
                # hatch where the emulator differs from CESM2 SIGNIFICANTLY
                hk = dict(colors="none", hatches=["", "...."], levels=[0.5, 1.5, 2.5])
                if HAVE_CARTOPY:
                    hk["transform"] = ccrs.PlateCarree()
                ax.contourf(d["lon"], d["lat"], sig.astype(float) + 1.0, **hk)
            if HAVE_CARTOPY:
                ax.coastlines(linewidth=0.35, color="0.25")
                ax.set_global()
            if r == 0:
                # Year range belongs in the column title: it is a property of
                # the experiment, identical for both variable rows, so putting
                # it here states it once instead of twice per column.
                ax.set_title(f"{d['label']}\n{d['yr'][0]}\u2013{d['yr'][1]}"
                             f"  ({args.n_years} yr)", fontsize=9.5)
            _msk = masked_pct.get(sc)
            # Solo panels carry only the number: the "no reference" caveat is
            # already the loudest line of the suptitle, and repeated under every
            # panel it is wide enough to collide with the neighbouring column.
            _stat = (f"global mean {gmean:+.2f} {unit}" if solo else
                     f"r = {corr:.3f}   RMSE = {rmse:.3f} {unit}\n"
                     f"{sig_pct:.0f}% of area significantly different")
            ax.text(0.5, -0.10,
                    _stat
                    # own line: appended inline it overruns into the neighbour
                    + (f"\n({_msk:.0f}% arid, masked)" if _msk else ""),
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
                    color="0.25")
            if c == 0:
                ax.text(-0.06, 0.5, meta["row"], transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)
            if d["n_emu"] != d["n_c"] and not solo:
                # Unequal sizes are fine for the Welch test (that is the point
                # of Welch), and equalising is usually IMPOSSIBLE anyway: only
                # 10/10/11/6 CESM2 members are held out of hist/ssp370/aaer/ghg.
                # Report it because the two means are converged to different
                # degrees, which matters when reading the colours.
                print(f"  [info] {var}/{sc}: {d['n_emu']} emulator vs "
                      f"{d['n_c']} CESM2 members (Welch handles the imbalance; "
                      f"the smaller side dominates the standard error)")
            _counts.append(dict(n_emu=d["n_emu"], n_cesm=d["n_c"]))
            rows.append(dict(var=var, scenario=sc, years=f"{d['yr'][0]}-{d['yr'][1]}",
                             n_emu=d["n_emu"], n_cesm=d["n_c"],
                             emulator_gmean=round(gmean, 4),
                             pattern_corr=round(corr, 4), rmse=round(rmse, 4),
                             pct_area_significant=round(sig_pct, 1),
                             pct_area_significant_raw=round(raw_pct, 1),
                             p_threshold=round(p_thr, 5),
                             pct_within_spread=round(frac, 1), unit=unit,
                             pct_area_masked=(round(masked_pct[sc], 1)
                                              if sc in masked_pct else 0.0)))

        if im is None:                     # nothing drawn in this row
            continue
        cb = fig.colorbar(im, ax=list(axes[r]), orientation="vertical",
                          fraction=0.018, pad=0.012, extend="both")
        cb.set_label(f"emulator anomaly ({unit})" if args.emulator_only
                     else f"emulator − CESM2 ({unit})", fontsize=8.5)

    _ne = sorted({r_["n_emu"] for r_ in _counts}) or [0]
    _nc = sorted({r_["n_cesm"] for r_ in _counts}) or [0]
    _fmt = lambda v: str(v[0]) if len(v) == 1 else "\u2013".join(
        (str(min(v)), str(max(v))))
    _what = "field" if args.absolute else "anomaly vs 1850\u20131900"
    # "held-out" is only true of the LENS2 reference; the unseen scenarios are
    # referenced against a separate CMIP6 ensemble, not a withheld part of the
    # training ensemble.
    _unseen = [s for s in plot_scen if s in CMIP6_REFS]
    _ref = "CESM2" if _unseen else "held-out CESM2"
    _pct_note = ("" if args.precip_mm or "PRECT" not in plot_vars else
                 "; precipitation as % of its 1850\u20131900 baseline")
    if args.emulator_only:
        # Nothing is differenced and nothing is tested, so a title promising a
        # bias map and a significance test would misdescribe every panel.
        _unseen_lbl = "/".join(SCEN[s][0].split(" (")[0] for s in plot_scen)
        # Kept to short lines: a single long suptitle widens the tight bounding
        # box far past the panels and leaves the maps stranded in the middle.
        fig.suptitle(
            f"EMULATOR ensemble-mean {_what} ({_fmt(_ne)} members), "
            f"{args.n_years}-year mean" + _pct_note + "\n"
            f"{_unseen_lbl}: forcing combination NEVER SEEN in training\n"
            f"NO CESM2 reference exists for this variable \u2014 the emulator's own "
            f"projection, not verified against CESM2",
            fontsize=9.5, y=1.005)
    else:
        if _unseen:
            _un_nc = sorted({r_["n_cesm"] for r_ in rows
                             if r_["scenario"] in _unseen}) or [0]
            _note = (f"{'/'.join(SCEN[s][0].split(' (')[0] for s in _unseen)}: "
                     f"forcing combination NEVER SEEN in training; CESM2 "
                     f"reference is the {_fmt(_un_nc)}-member CMIP6 ensemble\n")
        else:
            _note = ""
        fig.suptitle(
            f"ENSEMBLE-MEAN emulator minus {_ref} "
            f"({_fmt(_ne)} emulator vs {_fmt(_nc)} CESM2 members), "
            f"{args.n_years}-year mean " + _what + "\n" + _note
            + f"hatching = difference significant at p < {args.alpha:g} "
            f"(Welch t-test"
            + ("" if args.no_fdr else f", FDR-controlled at q = {2*args.alpha:g}") + ")"
            + _pct_note,
            fontsize=9.5, y=1.005)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    if args.dump_data and grid_dump:
        os.makedirs(args.dump_data, exist_ok=True)
        _dp = os.path.join(args.dump_data, "maps_gridded.csv")
        _g = pd.concat(grid_dump, ignore_index=True)
        for _c in ("lat", "lon", "emulator", "cesm2", "difference", "p_value"):
            _g[_c] = _g[_c].astype(float).round(5)
        _g.to_csv(_dp, index=False)
        print(f"\n[data] {_dp}  ({len(_g)} grid points across "
              f"{len(grid_dump)} panels)")

    t = pd.DataFrame(rows)
    print("\nSpatial comparison")
    print(t.to_string(index=False))
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
