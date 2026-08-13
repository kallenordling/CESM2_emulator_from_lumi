#!/usr/bin/env python3
"""
Paper figure: emulated vs held-out CESM2 global-mean temperature, four scenarios.

Panels: historical, SSP3-7.0, AAER (aerosol-only), GHG-only.

THE REFERENCE IS EVERY UNSEEN CESM2 MEMBER
------------------------------------------
The held-out set is resolved automatically as (members on disk) MINUS (members
in experiment_configs), so it is not limited to the single member named in
val_experiment_configs. Far more data was never trained on:

    scenario  on disk  trained  UNSEEN
    hist          30       20      10
    ssp370        30       20      10
    aaer          20        9      11
    ghg           15        9       6

Using the whole unseen ensemble matters: comparing a 5-member emulator ensemble
against ONE CESM2 realization pits a mean against a single noisy draw, so the
residual is dominated by CESM2's internal variability rather than model error.
With an unseen ENSEMBLE both sides have a mean and a spread, and the question
becomes whether the emulator reproduces the distribution.

Those held-out members are read STRAIGHT FROM THE TRAINING TREES here, rather
than from the eval NetCDF's CESM arrays, for two reasons:

  * for aaer/ghg the eval stores all ten members (001-010), nine of which ARE
    training data — plotting them would overstate the comparison, and picking
    m10 would rely on the loader's member ordering;
  * for hist/ssp370 the eval's reference is the CMIP6 CESM2 ensemble, a
    different ensemble entirely, not the held-out LENS2 member.

Baselines
---------
Each side is referenced to ITS OWN 1850-1900 mean — emulator to the emulated
historical, CESM2 to the held-out historical member. That isolates the forced
response and removes any absolute offset between them, so the comparison is of
warming, not of mean state. SSP3-7.0 starts in 2015 and has no pre-industrial of
its own, so it inherits the historical baseline from the same source.

Usage
-----
    python scripts/paper_fig_timeseries.py \
        --eval-dir /path/to/eval_output/run_mseyb_BCprect/best_ep0490

    # re-plot instantly from the cached reference series
    python scripts/paper_fig_timeseries.py --eval-dir ... --ref-csv <cached.csv>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASELINE = (1850, 1900)
VAR = "TREFHT"          # overridden by --var in main()

# Per-variable labels and the unit conversion needed on the TRAINING TREES.
# The trees store CESM2 native units (PRECT in m/s) while the emulator writes
# denormalised physical units (mm/day, see DENORM_FN in data/climate_dataset.py),
# so the reference must be converted or it is ~8.6e7 times too small.
VARMETA = {
    "TREFHT": dict(unit="\u00b0C", unit_plain="degC",
                   ylab="GMST anomaly (\u00b0C, vs 1850\u20131900)",
                   blab="Bias (\u00b0C)\nensemble means",
                   title="Emulated vs held-out CESM2 global-mean surface temperature",
                   tree_scale={None: 1.0, "K": 1.0, "degC": 1.0}),
    # CMIP6 `pr` is kg m-2 s-1 (1 kg m-2 == 1 mm of water) -> x86400 for mm/day;
    # the LENS2 trees store PRECT as m/s, hence the different factor there.
    "PRECT":  dict(unit="%", unit_plain="%", percent=True,
                   ylab="Precipitation change (%, vs 1850\u20131900)",
                   blab="Bias (percentage points)\nensemble means",
                   title="Emulated vs held-out CESM2 global-mean precipitation",
                   tree_scale={"m/s": 86400.0 * 1000.0, "mm/day": 1.0,
                               "kg m-2 s-1": 86400.0, None: 1.0}),
}

# scenario -> (panel label, eval NetCDF, (training-tree subdir, held-out member), colour)
# Okabe-Ito colours: distinguishable under deuteranopia/protanopia, unlike
# the red+green pairing this figure used before.
SCEN = {
    "hist":   ("Historical",                "hist",   "hist",   "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "ssp370", "ssp370", "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "aaer",   "AAER",   "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "ghg",    "GHG",    "#009E73"),
    # OUT-OF-TRAINING: forcing combinations the emulator never saw. No LENS2
    # training tree — the reference comes from CMIP6_REFS instead.
    "ssp126": ("SSP1-2.6 (unseen)",         "ssp126", None,     "#CC79A7"),
    "ssp245": ("SSP2-4.5 (unseen)",         "ssp245", None,     "#56B4E9"),
}

# The four scenarios with a held-out LENS2 reference — the default paper figure.
DEFAULT_SCENARIOS = ["hist", "ssp370", "aaer", "ghg"]

# CESM2 reference for the unseen scenarios: the same pre-aggregated CMIP6
# ensembles eval_aero.py evaluates against (eval_aero.py:88-119) — annual,
# native 192x288 grid, 2015-2100, 3 members (r4/r10/r11).
#
# The `_pr` files do NOT ship with the repo's data (the tas ones came from the
# Pangeo Google-Cloud archive). Build them from ESGF, per scenario:
#   python download_cmip6_cesm2.py --experiment ssp126 --variables pr \
#          --members r4i1p1f1 r10i1p1f1 r11i1p1f1
#   python scripts/build_cmip6_annual_ref.py --experiment ssp126 --variable pr
# Until they exist a PRECT run of these scenarios is plotted emulator-only.
CMIP6_REFS = {
    "ssp126": {"TREFHT": ("cmip6/ssp126.nc", "tas"),
               "PRECT":  ("cmip6/ssp126_pr.nc", "pr")},
    "ssp245": {"TREFHT": ("cmip6/ssp245.nc", "tas"),
               "PRECT":  ("cmip6/ssp245_pr.nc", "pr")},
}

# scenario key in the data config -> training-tree subdirectory
CFG_KEY = {"hist": "hist", "ssp370": "ssp370", "aaer": "aaer", "ghg": "ghg",
           "ssp126": "ssp126", "ssp245": "ssp245"}


def area_mean(da: xr.DataArray) -> xr.DataArray:
    """cos(lat)-weighted global mean, matching eval_aero.area_weighted_gmean."""
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(("lat", "lon"))


def unseen_members(tree_root: Path, subdir: str, trained: set) -> list:
    """Members present on disk but absent from experiment_configs."""
    d = tree_root / subdir
    have = {p.name for p in d.iterdir()
            if p.is_dir() and p.name != "diagnostics"}
    return sorted(have - set(trained))


def read_heldout_ensemble(tree_root: Path, subdir: str, members: list) -> pd.DataFrame:
    """Annual global-mean TREFHT per unseen member -> DataFrame(year x member)."""
    cols = {}
    for i, mem in enumerate(members, 1):
        d = tree_root / subdir / mem
        files = sorted(d.glob("*.nc"))
        if not files:
            print(f"      [{i}/{len(members)}] {mem}: NO CHUNKS, skipped", flush=True)
            continue
        print(f"      [{i}/{len(members)}] {subdir}/{mem} ({len(files)} chunks)", flush=True)
        ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
        if VAR not in ds:
            raise KeyError(
                f"{d}: no variable {VAR!r} in the chunk files (found "
                f"{sorted(v for v in ds.data_vars)[:8]}). --tree-root is "
                f"{tree_root}; it must point at the tree for {VAR}, e.g. "
                f"training_data/{VAR}.")
        raw_units = ds[VAR].attrs.get("units")
        scale = VARMETA[VAR]["tree_scale"].get(raw_units)
        if scale is None:
            raise ValueError(
                f"{subdir}/{mem}: {VAR} has units {raw_units!r}, which has no "
                f"conversion to the emulator's {VARMETA[VAR]['unit_plain']}. "
                f"Known: {sorted(k for k in VARMETA[VAR]['tree_scale'] if k)}")
        if i == 1 and scale != 1.0:
            print(f"      [units] tree {VAR} is {raw_units!r} -> x{scale:g} "
                  f"to {VARMETA[VAR]['unit_plain']}", flush=True)
        gm = (area_mean(ds[VAR]) * scale).compute()
        tdim = "time" if "time" in gm.dims else "year"
        years = np.asarray(ds[tdim].values).astype(int)
        sr = pd.Series(np.asarray(gm.values, dtype=float), index=years).sort_index()
        cols[mem] = sr[~sr.index.duplicated(keep="first")]
        ds.close()
    if not cols:
        raise FileNotFoundError(f"no readable members under {tree_root / subdir}")
    return pd.DataFrame(cols).sort_index()


def read_cmip6_ensemble(path: Path, ncvar: str) -> pd.DataFrame:
    """Same shape as read_heldout_ensemble, from a pre-aggregated CMIP6 file.

    Dimensions are (year, member, lat, lon) rather than one directory of chunks
    per member. Only defined for temperature — see CMIP6_REFS.
    """
    ds = xr.open_dataset(path)
    if ncvar not in ds:
        raise KeyError(f"{path}: no {ncvar!r}")
    raw_units = ds[ncvar].attrs.get("units")
    scale = VARMETA[VAR]["tree_scale"].get(raw_units)
    if scale is None:
        raise ValueError(f"{path}: {ncvar} has units {raw_units!r}, which has "
                         f"no conversion to {VARMETA[VAR]['unit_plain']}")
    gm = (area_mean(ds[ncvar]) * scale).compute()      # (year, member)
    years = np.asarray(ds["year"].values).astype(int)
    cols = {str(m): pd.Series(
                np.asarray(gm.sel(member=m).values, dtype=float), index=years)
            for m in ds["member"].values}
    print(f"      CMIP6 {path.name}: {len(cols)} members {sorted(cols)} "
          f"{years.min()}-{years.max()}", flush=True)
    ds.close()
    return pd.DataFrame(cols).sort_index()


def qc_ensemble(df: pd.DataFrame, scenario: str, n_sigma: float = 5.0) -> pd.DataFrame:
    """Mask corrupt points before they contaminate the reference mean/spread.

    Some staged realizations have bad years — LE2-1231.012 reads 286.02 K in
    1930 (1.2 K below the ensemble mean, ~10 sigma) and is NaN for 1931-1935.
    Averaging that in drags the reference mean down and inflates the spread
    exactly where the bias panel is read.

    Points further than n_sigma from the per-year ensemble MEDIAN (robust to the
    outlier itself) are set to NaN; every downstream statistic skips NaN, so a
    member contributes for the years it is good and is simply absent elsewhere.
    """
    med = df.median(axis=1)
    dev = df.sub(med, axis=0)
    sd = float(dev.stack().std())
    bad = dev.abs() > n_sigma * sd
    if bad.any().any():
        for m in df.columns[bad.any()]:
            yrs = df.index[bad[m]].tolist()
            print(f"    [QC] {scenario}: masked {m} at {yrs} "
                  f"(>{n_sigma:g} sigma, sd={sd:.3f} K)")
    nan_before = int(df.isna().sum().sum())
    if nan_before:
        for m in df.columns[df.isna().any()]:
            print(f"    [QC] {scenario}: {m} has {int(df[m].isna().sum())} "
                  f"missing years (excluded per-year)")
    return df.mask(bad)


def read_emulated(nc_path: Path):
    """(ensemble-mean, member matrix, years) absolute global-mean from an eval NetCDF."""
    ds = xr.open_dataset(nc_path)
    years = ds["year"].values.astype(int)
    mean = ds[f"{VAR}_model_gmean_mean"].values
    members = [ds[v].values for v in ds.data_vars
               if v.startswith(f"{VAR}_model_gmean_m")
               and not v.endswith("_anom")
               and not v.startswith(f"{VAR}_model_gmean_mean")]
    ds.close()
    return mean, (np.stack(members) if members else None), years


def anom(values, base):
    """Anomaly vs `base`: an absolute difference, or a PERCENT change when the
    variable is set percent=True (precipitation, where a few tenths of a mm/day
    is meaningless without the ~2.9 mm/day it is relative to)."""
    if VARMETA[VAR].get("percent"):
        return 100.0 * (values - base) / base
    return values - base


def baseline_of(series_years, values) -> float:
    m = (series_years >= BASELINE[0]) & (series_years <= BASELINE[1])
    if not m.any():
        return np.nan
    return float(np.asarray(values)[m].mean())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paper figure: emulated vs held-out CESM2 global-mean timeseries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--var", default="TREFHT", choices=sorted(VARMETA),
                    help="Target variable to plot")
    ap.add_argument("--eval-dir", required=True,
                    help="eval output dir holding TREFHT_<scenario>.nc")
    ap.add_argument("--tree-root", default=None,
                    help="root of the training trees for --var, holding "
                         "hist/, ssp370/, AAER/, GHG/. "
                         "Default: <--data-root>/training_data/<var>")
    ap.add_argument("--data-root",
                    default="/home/nordling/mnt/lumi_sc2/emulator_data",
                    help="emulator_data root; --tree-root is derived from it")
    ap.add_argument("--data-config",
                    default="configs/config_data_ybias_BCprect.yaml",
                    help="data config whose experiment_configs define the TRAINED "
                         "members; everything else on disk counts as unseen")
    ap.add_argument("--n-ref-members", type=int, default=5,
                    help="Use only the first N unseen CESM2 members, so the "
                         "reference ensemble is the same size as the emulator's "
                         "and identical across experiments. 0 = use all.")
    ap.add_argument("--ref-csv", default=None,
                    help="cache of the held-out reference series; read if present, "
                         "written after computing so re-plots are instant")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                    choices=sorted(SCEN),
                    help="ssp126/ssp245 are OUT-OF-TRAINING; their reference is "
                         "a 3-member CMIP6 ensemble and exists for TREFHT only "
                         "(a PRECT run of them is emulator-only)")
    ap.add_argument("--out", default=None,
                    help="default plots/paper_fig_timeseries_<var>.png")
    ap.add_argument("--year-max", type=int, default=2100)
    args = ap.parse_args()

    global VAR
    VAR = args.var
    META = VARMETA[VAR]
    if args.out is None:
        args.out = f"plots/paper_fig_timeseries_{VAR}.png"
    if args.ref_csv is None:
        args.ref_csv = f"plots/heldout_cesm2_ensemble_{VAR}.csv"
    if args.tree_root is None:
        # the trees are per-variable: training_data/TREFHT, training_data/PRECT
        args.tree_root = os.path.join(args.data_root, "training_data", VAR)
    print(f"[var] {VAR} ({META['unit_plain']})")

    eval_dir = Path(args.eval_dir)
    tree_root = Path(args.tree_root)

    # hist is always loaded even when it is not plotted: it supplies the
    # 1850-1900 baseline for every scenario that has no pre-industrial of its own.
    load_scen = ["hist"] + [s for s in args.scenarios if s != "hist"]

    # ── held-out CESM2 reference: EVERY unseen member ───────────────────────
    ref = {}          # scenario -> DataFrame(year x member)
    cached = {}
    if args.ref_csv and os.path.exists(args.ref_csv):
        df = pd.read_csv(args.ref_csv)
        for sc, g in df.groupby("scenario"):
            cached[sc] = qc_ensemble(
                g.pivot(index="year", columns="member", values="gmean_K").sort_index(), sc)
        print(f"[ref] reusing cached ensemble from {args.ref_csv}")
        for sc, d in cached.items():
            print(f"      {sc:7s} {d.shape[1]} unseen members, "
                  f"{int(d.index.min())}-{int(d.index.max())}")
    ref = {sc: cached[sc] for sc in load_scen if sc in cached}

    # Anything requested but not cached is computed now and appended, so a cache
    # written for the four default scenarios does not silently mean "ssp126 has
    # no reference".
    todo = [sc for sc in load_scen if sc not in ref]
    if todo:
        import yaml
        cfg = yaml.safe_load(open(args.data_config))
        trained = {e["scenario_name"]: set(e.get("realizations", []))
                   for e in cfg["experiment_configs"]}
        print(f"[ref] resolving {todo} (on disk MINUS experiment_configs)")
        for sc in todo:
            sub = SCEN[sc][2]
            if sc in CMIP6_REFS:
                spec = CMIP6_REFS[sc].get(VAR)
                if spec is None or not (Path(args.data_root) / spec[0]).exists():
                    # The reference for this variable has not been built yet —
                    # plot the emulator alone rather than fake one.
                    _rel = spec[0] if spec else f"cmip6/{sc}_<{VAR}>.nc"
                    print(f"    {sc:7s} no CESM2 {VAR} reference ({_rel} not "
                          f"found; build with download_cmip6_cesm2.py + "
                          f"scripts/build_cmip6_annual_ref.py) "
                          f"— emulator-only panel")
                    continue
                rel, ncvar = spec
                print(f"    {sc:7s} CMIP6 ensemble {rel}")
                ref[sc] = qc_ensemble(
                    read_cmip6_ensemble(Path(args.data_root) / rel, ncvar), sc)
                continue
            mems = unseen_members(tree_root, sub, trained.get(CFG_KEY[sc], set()))
            print(f"    {sc:7s} {len(mems)} unseen: {mems}")
            ref[sc] = qc_ensemble(
                read_heldout_ensemble(tree_root, sub, mems), sc)
        if args.ref_csv:
            merged = dict(cached)
            merged.update(ref)
            rows = [dict(scenario=sc, member=m, year=int(y), gmean_K=float(v))
                    for sc, d in merged.items()
                    for m in d.columns
                    for y, v in d[m].dropna().items()]
            os.makedirs(os.path.dirname(os.path.abspath(args.ref_csv)) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(args.ref_csv, index=False)
            print(f"[out] {args.ref_csv}")

    # ── equal ensemble size on both sides ───────────────────────────────────
    # The unseen sets differ in size (6-11) and none matches the emulator's 5,
    # so the two means were converged to different degrees and the +/-2 sigma
    # band was estimated from a different N per experiment. Truncating to a
    # common N makes every mean carry the same sampling noise and every spread
    # the same estimator variance, so panels b-d are directly comparable.
    # Selection is the first N sorted member IDs — deterministic, not random.
    if args.n_ref_members and args.n_ref_members > 0:
        for sc in list(ref):
            have = list(ref[sc].columns)
            if len(have) < args.n_ref_members:
                print(f"[ref] WARNING: {sc} has only {len(have)} unseen members, "
                      f"fewer than the requested {args.n_ref_members}")
                continue
            keep = have[:args.n_ref_members]
            ref[sc] = ref[sc][keep]
            print(f"[ref] {sc:7s} using {len(keep)} of {len(have)} unseen "
                  f"members: {keep}")

    # ── emulated ────────────────────────────────────────────────────────────
    emu = {}
    for sc in load_scen:
        nc = SCEN[sc][1]
        p = eval_dir / f"{VAR}_{nc}.nc"
        if not p.exists():
            print(f"[emu] MISSING {p} — panel will show reference only")
            continue
        emu[sc] = read_emulated(p)
        print(f"[emu] {sc:7s} {emu[sc][2][0]}-{emu[sc][2][-1]}  "
              f"{0 if emu[sc][1] is None else emu[sc][1].shape[0]} members")

    # ── baselines: each side referenced to ITS OWN pre-industrial ───────────
    ref_base_hist = baseline_of(ref["hist"].index.values,
                                ref["hist"].mean(axis=1, skipna=True).values)
    emu_base_hist = (baseline_of(emu["hist"][2], emu["hist"][0])
                     if "hist" in emu else np.nan)
    print(f"\n[baseline 1850-1900]  CESM2 held-out hist {ref_base_hist:.4f}   "
          f"emulator hist {emu_base_hist:.4f}   "
          f"(anomalies as {'percent change' if META.get('percent') else 'absolute difference'})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    # ── one panel with every scenario, bias beneath ─────────────────────────
    # (a) combined overview on top; beneath it one bias panel per experiment,
    # with hist+ssp370 sharing a panel since they are one continuous
    # trajectory (hist ends 2014, ssp370 starts 2015).
    # hist+ssp370 share a panel when both are plotted: they are one continuous
    # trajectory (hist ends 2014, ssp370 starts 2015). Everything else gets its
    # own panel, so the layout follows --scenarios instead of being fixed.
    # Scenarios with no CESM2 reference have no bias to draw and are omitted.
    _has_ref = [s for s in args.scenarios if s in ref]
    BIAS_GROUPS = []
    if "hist" in _has_ref and "ssp370" in _has_ref:
        BIAS_GROUPS.append((("hist", "ssp370"), "Historical + SSP3-7.0"))
    for s in _has_ref:
        if s in ("hist", "ssp370") and BIAS_GROUPS and \
                BIAS_GROUPS[0][0] == ("hist", "ssp370"):
            continue
        BIAS_GROUPS.append(((s,), SCEN[s][0]))
    if BIAS_GROUPS:
        fig = plt.figure(figsize=(9.5, 7.6))
        gs = fig.add_gridspec(2, len(BIAS_GROUPS), height_ratios=[2.3, 1.0],
                              hspace=0.30, wspace=0.12)
        ax = fig.add_subplot(gs[0, :])
        axbs = []
        for i in range(len(BIAS_GROUPS)):
            a = fig.add_subplot(gs[1, i], sharey=axbs[0] if axbs else None)
            axbs.append(a)
    else:
        # No CESM2 reference for any plotted scenario (PRECT + unseen
        # scenarios): there is no bias to show, so the figure is the
        # trajectory panel alone rather than a row of empty axes.
        fig = plt.figure(figsize=(9.5, 5.4))
        ax = fig.add_subplot(1, 1, 1)
        axbs = []
    bias_of = {}          # scenario -> (years, bias series)
    rows = []
    sigma_by_scen = {}
    plotted_years = []    # every year actually drawn, for the x-limit

    for sc in args.scenarios:
        label, _, sub, colour = SCEN[sc]
        R = ref.get(sc)                          # DataFrame(year x member), or None
        # scenarios without their own pre-industrial inherit the historical one
        eb = emu_base_hist if sc == "ssp370" else (
            baseline_of(emu[sc][2], emu[sc][0]) if sc in emu else np.nan)
        if not np.isfinite(eb):
            eb = emu_base_hist

        Ra = r_mean = None
        if R is not None:
            r_years = R.index.values
            rb = ref_base_hist if sc == "ssp370" else baseline_of(
                r_years, R.mean(axis=1, skipna=True).values)
            if not np.isfinite(rb):
                rb = ref_base_hist
            keep_r = r_years <= args.year_max
            Ra = anom(R[keep_r], rb)              # anomalies, per member
            r_mean = Ra.mean(axis=1, skipna=True)
            plotted_years += [float(Ra.index.min()), float(Ra.index.max())]

            # CESM2 held-out ensemble: mean dashed + member spread
            ax.fill_between(Ra.index, Ra.min(axis=1, skipna=True),
                            Ra.max(axis=1, skipna=True),
                            color=colour, alpha=0.12, lw=0, zorder=1)
            # CESM2 reference gets OPEN CIRCLE MARKERS, not just a dash pattern.
            # Solid-vs-dashed in the same colour is not readable where the two
            # curves coincide (which is most of the record); a marker shape stays
            # distinguishable regardless of colour, overlap or print size.
            ax.plot(Ra.index, r_mean.values, color=colour, lw=1.2, ls="--",
                    marker="o", markersize=3.4, markevery=8,
                    markerfacecolor="white", markeredgecolor=colour,
                    markeredgewidth=1.0, zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

        if sc not in emu:
            continue
        mean, members, years = emu[sc]
        keep = years <= args.year_max
        if members is not None:
            _em = anom(members[:, keep], eb)
            ax.fill_between(years[keep], _em.min(axis=0), _em.max(axis=0),
                            color=colour, alpha=0.28, lw=0, zorder=2)
        ax.plot(years[keep], anom(mean[keep], eb), color=colour, lw=2.6, zorder=4,
                solid_capstyle="round", label=label)
        plotted_years += [float(years[keep].min()), float(years[keep].max())]

        # ── bias panel ──────────────────────────────────────────────────────
        # Line: difference of ENSEMBLE MEANS. Band: the spread of individual
        # CESM2 members about their own mean — i.e. what a single realization
        # departs from the forced response by chance. A bias line inside that
        # band is indistinguishable from internal variability.
        if Ra is None:                # emulator-only scenario: nothing to bias
            continue
        common = np.intersect1d(years[keep], Ra.index.values)
        if not len(common):
            continue
        e = pd.Series(anom(mean[keep], eb), index=years[keep]).loc[common]
        c = r_mean.loc[common]
        d = e - c
        spread = Ra.loc[common].sub(c, axis=0)
        # Per-scenario sigma of members about their own mean. Recorded for the
        # shared envelope drawn once below; min/max is NOT used because its
        # width depends on member count (6 for ghg vs 11 for aaer) and would
        # not be comparable across scenarios.
        sd_series = spread.std(axis=1, skipna=True)
        sigma_by_scen[sc] = float(sd_series.mean())
        bias_of[sc] = (common, d, sd_series)

        inside = float((d.abs() <= 2 * sd_series).mean()) * 100
        rows.append(dict(scenario=sc,
                         n_emu=(members.shape[0] if members is not None else 0),
                         n_unseen=Ra.shape[1], n_years=len(common),
                         bias=round(float(d.mean()), 3),
                         rmse=round(float(np.sqrt((d ** 2).mean())), 3),
                         corr=round(float(np.corrcoef(e, c)[0, 1]), 3),
                         cesm_sd=round(float(Ra.loc[common].std(axis=1, skipna=True).mean()), 3),
                         pct_within_spread=round(inside, 1)))

    # One grey +/-2 sigma envelope per panel. Sigma is used rather than member
    # min/max because min/max width depends on ensemble size (6 members for ghg
    # vs 11 for aaer); the per-scenario sigmas agree to ~2%, so the same band
    # applies everywhere.
    _sig = float(np.mean(list(sigma_by_scen.values()))) if sigma_by_scen else 0.0
    stats = {r["scenario"]: r for r in rows}

    for i, (group, gtitle) in enumerate(BIAS_GROUPS):
        a = axbs[i]
        a.axhline(0, ls="-", lw=0.8, color="0.3", zorder=1)
        for sc in group:
            if sc not in bias_of:
                continue
            yy, d, sd = bias_of[sc]
            # Grey band = the CESM2 unseen ensemble's own spread, +/-2 sigma
            # computed PER YEAR from its members (not a constant summary), so it
            # shows how much a single CESM2 realization departs from the forced
            # response by chance at that time.
            a.fill_between(yy, -2 * sd.values, 2 * sd.values,
                           color="0.45", alpha=0.22, lw=0, zorder=0)
            a.plot(yy, d.values, color=SCEN[sc][3], lw=1.4, zorder=3)
        _ne = sorted({stats[sc]["n_emu"] for sc in group if sc in stats})
        _nc = sorted({stats[sc]["n_unseen"] for sc in group if sc in stats})
        _fmt = lambda v: str(v[0]) if len(v) == 1 else "\u2013".join(
            (str(min(v)), str(max(v))))
        a.set_title(f"{gtitle}\nn = {_fmt(_ne)} emulator, {_fmt(_nc)} CESM2",
                    fontsize=9.0, loc="left", pad=4, linespacing=1.5)
        a.grid(alpha=0.25)
        a.text(0.02, 0.94, f"({'bcde'[i]})", transform=a.transAxes,
               fontweight="bold", va="top", ha="left", fontsize=9)
        a.text(0.97, 0.95, f"grey: CESM2 spread (\u00b12\u03c3)",
               transform=a.transAxes, fontsize=7.4, va="top", ha="right",
               color="0.30")

        # numbers on the figure rather than only in the console
        txt = "\n".join(
            f"{SCEN[sc][0].split(' (')[0]}: "
            f"{stats[sc]['bias']:+.2f} \u00b1 {stats[sc]['rmse']:.2f} "
            f"{META['unit']}, "
            f"{stats[sc]['pct_within_spread']:.0f}% in band"
            for sc in group if sc in stats)
        if txt:
            a.text(0.02, 0.04, txt, transform=a.transAxes, fontsize=7.2,
                   va="bottom", ha="left", color="0.25")

        a.set_xlabel("Year")
        if i == 0:
            a.set_ylabel(META["blab"])
        else:
            a.tick_params(labelleft=False)

    # legend: scenario colours, plus what solid/dashed mean
    # Member counts for the legend: emulator is the same everywhere; CESM2
    # differs by scenario (6-11), so show the range rather than a single number.
    # Emulator counts come from the eval files, not from `rows`: an
    # emulator-only run produces no bias rows but still draws member curves,
    # and reading "0 members" off the legend of a 25-member ensemble is worse
    # than having no legend at all.
    _ne = sorted({e[1].shape[0] for sc, e in emu.items()
                  if sc in args.scenarios and e[1] is not None}) or [0]
    _n_emu = str(_ne[0]) if len(_ne) == 1 else f"{min(_ne)}\u2013{max(_ne)}"
    _n_c = sorted({r["n_unseen"] for r in rows}) or [0]
    _n_cesm = str(_n_c[0]) if len(_n_c) == 1 else f"{min(_n_c)}\u2013{max(_n_c)}"
    _has_cesm = any(sc in ref for sc in args.scenarios)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.patheffects as pe
    style = [
        Line2D([], [], color="0.35", lw=2.6,
               label=f"EMULATOR \u2014 ensemble mean ({_n_emu} members)"),
        Patch(facecolor="0.35", alpha=0.28, label="EMULATOR member range"),
    ]
    # Only advertise the reference when one was actually drawn.
    if _has_cesm:
        style[1:1] = [
            Line2D([], [], color="0.35", lw=1.2, ls="--", marker="o",
                   markersize=3.4, markerfacecolor="white",
                   markeredgecolor="0.35",
                   label=f"CESM2 \u2014 unseen ensemble mean "
                         f"({_n_cesm} members)"),
        ]
        style.append(Patch(facecolor="0.35", alpha=0.12,
                           label="CESM2 member range"))
    if BIAS_GROUPS:
        style.append(Patch(
            facecolor="0.55", alpha=0.20,
            label=f"CESM2 spread about its mean, "
                  + ("b" if len(BIAS_GROUPS) == 1 else
                     f"b\u2013{'bcde'[len(BIAS_GROUPS) - 1]}") + " "
                  f"(\u00b12\u03c3, mean \u00b1{2*_sig:.2f} {META['unit']})"))
    # Both legends top-left: that corner is empty until ~1950 in every
    # scenario, whereas lower-right sits on top of the AAER curve.
    # Legends ABOVE the axes so they take no data area.
    # The scenario legend sits ABOVE the style legend, so its offset has to
    # clear however many rows the style legend takes. Deriving it from the
    # axes height in inches keeps the two from colliding when the figure is
    # short (emulator-only runs drop the bias row AND two style entries).
    _rows_style = int(np.ceil(len(style) / 2))
    _axh = ax.get_position().height * fig.get_size_inches()[1]
    leg1 = ax.legend(frameon=False, ncols=4, loc="lower left",
                     bbox_to_anchor=(0.0, 1.005 + _rows_style * 0.165 / _axh),
                     handlelength=2.2)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=style, frameon=False, ncols=2, fontsize=8.2,
                     loc="lower left", bbox_to_anchor=(0.0, 1.005),
                     handlelength=2.6)

    ax.axhline(0, ls=":", lw=0.8, color="0.3")
    # The baseline window is only worth shading when it is actually on the axis:
    # scenarios that start in 2015 would otherwise carry a legend-grey block far
    # off to the left of every curve.
    _xmin = min(plotted_years) if plotted_years else BASELINE[0]
    if _xmin <= BASELINE[1]:
        ax.axvspan(*BASELINE, color="0.9", alpha=0.6, lw=0, zorder=0)
    ax.set_ylabel(META["ylab"])
    ax.text(0.005, 0.97, "(a)", transform=ax.transAxes, fontweight="bold",
            va="top", ha="left")


    ax.set_xlabel("Year")
    ax.set_xlim(min(BASELINE[0], _xmin) if _xmin <= BASELINE[1] else _xmin - 2,
                args.year_max)
    # y-limit must clear both the bias lines and the +/-2 sigma band
    _floor = 0.35 if not VARMETA[VAR].get('percent') else 1.0
    _lim = max([_floor]
               + [abs(float(d.min())) for _, d, _sd in bias_of.values()]
               + [abs(float(d.max())) for _, d, _sd in bias_of.values()]
               + [2 * float(_sd.max()) for _, _d, _sd in bias_of.values()]) * 1.15
    if axbs:
        axbs[0].set_ylim(-_lim, _lim)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # Legends live ABOVE the axes; without listing them explicitly the tight
    # bbox can crop their top row.
    _extra = [leg1, leg2]
    fig.savefig(args.out, bbox_inches="tight", bbox_extra_artists=_extra)
    fig.savefig(str(Path(args.out).with_suffix(".pdf")),                       # vector for the journal
                bbox_inches="tight", bbox_extra_artists=_extra)
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    if rows:
        t = pd.DataFrame(rows)
        print(f"\nEmulator vs held-out CESM2 ({META['unit_plain']}, overlapping years)")
        print(t.to_string(index=False))
        if sigma_by_scen:
            _v = list(sigma_by_scen.values())
            print(f"\nCESM2 inter-member sigma by scenario ({META['unit_plain']}): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in sigma_by_scen.items()))
            print(f"  spread across scenarios: {max(_v)-min(_v):.4f} "
                  f"{META['unit_plain']} "
                  f"({100*(max(_v)-min(_v))/np.mean(_v):.1f}% of the mean) "
                  f"-> a single shared envelope is representative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
