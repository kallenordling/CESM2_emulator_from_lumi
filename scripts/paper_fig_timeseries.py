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
VAR = "TREFHT"

# scenario -> (panel label, eval NetCDF, (training-tree subdir, held-out member), colour)
# Okabe-Ito colours: distinguishable under deuteranopia/protanopia, unlike
# the red+green pairing this figure used before.
SCEN = {
    "hist":   ("Historical",                "TREFHT_hist.nc",   "hist",   "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "TREFHT_ssp370.nc", "ssp370", "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "TREFHT_aaer.nc",   "AAER",   "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "TREFHT_ghg.nc",    "GHG",    "#009E73"),
}

# scenario key in the data config -> training-tree subdirectory
CFG_KEY = {"hist": "hist", "ssp370": "ssp370", "aaer": "aaer", "ghg": "ghg"}


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
        gm = area_mean(ds[VAR]).compute()
        tdim = "time" if "time" in gm.dims else "year"
        years = np.asarray(ds[tdim].values).astype(int)
        sr = pd.Series(np.asarray(gm.values, dtype=float), index=years).sort_index()
        cols[mem] = sr[~sr.index.duplicated(keep="first")]
        ds.close()
    if not cols:
        raise FileNotFoundError(f"no readable members under {tree_root / subdir}")
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


def baseline_of(series_years, values) -> float:
    m = (series_years >= BASELINE[0]) & (series_years <= BASELINE[1])
    if not m.any():
        return np.nan
    return float(np.asarray(values)[m].mean())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paper figure: emulated vs held-out CESM2 global-mean timeseries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval-dir", required=True,
                    help="eval output dir holding TREFHT_<scenario>.nc")
    ap.add_argument("--tree-root",
                    default="/home/nordling/mnt/lumi_sc2/emulator_data/training_data/TREFHT",
                    help="root of the TREFHT training trees (holds hist/, ssp370/, AAER/, GHG/)")
    ap.add_argument("--data-config",
                    default="configs/config_data_ybias_BCprect.yaml",
                    help="data config whose experiment_configs define the TRAINED "
                         "members; everything else on disk counts as unseen")
    ap.add_argument("--ref-csv", default=None,
                    help="cache of the held-out reference series; read if present, "
                         "written after computing so re-plots are instant")
    ap.add_argument("--out", default="plots/paper_fig_timeseries.png")
    ap.add_argument("--year-max", type=int, default=2100)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    tree_root = Path(args.tree_root)

    # ── held-out CESM2 reference: EVERY unseen member ───────────────────────
    ref = {}          # scenario -> DataFrame(year x member)
    if args.ref_csv and os.path.exists(args.ref_csv):
        df = pd.read_csv(args.ref_csv)
        for sc, g in df.groupby("scenario"):
            ref[sc] = qc_ensemble(
                g.pivot(index="year", columns="member", values="gmean_K").sort_index(), sc)
        print(f"[ref] reusing cached ensemble from {args.ref_csv}")
        for sc, d in ref.items():
            print(f"      {sc:7s} {d.shape[1]} unseen members, "
                  f"{int(d.index.min())}-{int(d.index.max())}")
    else:
        import yaml
        cfg = yaml.safe_load(open(args.data_config))
        trained = {e["scenario_name"]: set(e.get("realizations", []))
                   for e in cfg["experiment_configs"]}
        print("[ref] resolving unseen members (on disk MINUS experiment_configs)")
        for sc, (_, _, sub, _) in SCEN.items():
            mems = unseen_members(tree_root, sub, trained.get(CFG_KEY[sc], set()))
            print(f"    {sc:7s} {len(mems)} unseen: {mems}")
            ref[sc] = qc_ensemble(
                read_heldout_ensemble(tree_root, sub, mems), sc)
        if args.ref_csv:
            rows = [dict(scenario=sc, member=m, year=int(y), gmean_K=float(v))
                    for sc, d in ref.items()
                    for m in d.columns
                    for y, v in d[m].dropna().items()]
            os.makedirs(os.path.dirname(os.path.abspath(args.ref_csv)) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(args.ref_csv, index=False)
            print(f"[out] {args.ref_csv}")

    # ── emulated ────────────────────────────────────────────────────────────
    emu = {}
    for sc, (_, nc, _, _) in SCEN.items():
        p = eval_dir / nc
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
    print(f"\n[baseline 1850-1900]  CESM2 held-out hist {ref_base_hist:.3f} K   "
          f"emulator hist {emu_base_hist:.3f}")

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
    BIAS_GROUPS = [
        (("hist", "ssp370"), "Historical + SSP3-7.0"),
        (("aaer",),          "Aerosol-only (AAER)"),
        (("ghg",),           "Greenhouse-gas-only (GHG)"),
    ]
    fig = plt.figure(figsize=(9.5, 7.6))
    gs = fig.add_gridspec(2, len(BIAS_GROUPS), height_ratios=[2.3, 1.0],
                          hspace=0.30, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    axbs = []
    for i in range(len(BIAS_GROUPS)):
        a = fig.add_subplot(gs[1, i], sharey=axbs[0] if axbs else None)
        axbs.append(a)
    bias_of = {}          # scenario -> (years, bias series)
    rows = []
    sigma_by_scen = {}

    for sc, (label, _, sub, colour) in SCEN.items():
        R = ref[sc]                              # DataFrame(year x member)
        r_years = R.index.values
        # scenarios without their own pre-industrial inherit the historical one
        rb = ref_base_hist if sc == "ssp370" else baseline_of(
            r_years, R.mean(axis=1, skipna=True).values)
        eb = emu_base_hist if sc == "ssp370" else (
            baseline_of(emu[sc][2], emu[sc][0]) if sc in emu else np.nan)
        if not np.isfinite(rb):
            rb = ref_base_hist
        if not np.isfinite(eb):
            eb = emu_base_hist

        keep_r = r_years <= args.year_max
        Ra = R[keep_r] - rb                       # anomalies, per member
        r_mean = Ra.mean(axis=1, skipna=True)

        # CESM2 held-out ensemble: mean dashed + member spread
        ax.fill_between(Ra.index, Ra.min(axis=1, skipna=True), Ra.max(axis=1, skipna=True),
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
            ax.fill_between(years[keep],
                            (members[:, keep] - eb).min(axis=0),
                            (members[:, keep] - eb).max(axis=0),
                            color=colour, alpha=0.28, lw=0, zorder=2)
        ax.plot(years[keep], mean[keep] - eb, color=colour, lw=2.6, zorder=4,
                solid_capstyle="round", label=label)

        # ── bias panel ──────────────────────────────────────────────────────
        # Line: difference of ENSEMBLE MEANS. Band: the spread of individual
        # CESM2 members about their own mean — i.e. what a single realization
        # departs from the forced response by chance. A bias line inside that
        # band is indistinguishable from internal variability.
        common = np.intersect1d(years[keep], Ra.index.values)
        if not len(common):
            continue
        e = pd.Series(mean[keep] - eb, index=years[keep]).loc[common]
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
        a.text(0.97, 0.95, "grey: CESM2 spread (\u00b12\u03c3)",
               transform=a.transAxes, fontsize=7.4, va="top", ha="right",
               color="0.30")

        # numbers on the figure rather than only in the console
        txt = "\n".join(
            f"{SCEN[sc][0].split(' (')[0]}: "
            f"{stats[sc]['bias']:+.2f} \u00b1 {stats[sc]['rmse']:.2f} \u00b0C, "
            f"{stats[sc]['pct_within_spread']:.0f}% in band"
            for sc in group if sc in stats)
        if txt:
            a.text(0.02, 0.04, txt, transform=a.transAxes, fontsize=7.2,
                   va="bottom", ha="left", color="0.25")

        a.set_xlabel("Year")
        if i == 0:
            a.set_ylabel("Bias (\u00b0C)\nensemble means")
        else:
            a.tick_params(labelleft=False)

    # legend: scenario colours, plus what solid/dashed mean
    # Member counts for the legend: emulator is the same everywhere; CESM2
    # differs by scenario (6-11), so show the range rather than a single number.
    _n_emu = sorted({r["n_emu"] for r in rows}) or [0]
    _n_emu = str(_n_emu[0]) if len(_n_emu) == 1 else f"{min(_n_emu)}\u2013{max(_n_emu)}"
    _n_c = sorted({r["n_unseen"] for r in rows}) or [0]
    _n_cesm = str(_n_c[0]) if len(_n_c) == 1 else f"{min(_n_c)}\u2013{max(_n_c)}"

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.patheffects as pe
    style = [
        Line2D([], [], color="0.35", lw=2.6,
               label=f"EMULATOR \u2014 ensemble mean ({_n_emu} members)"),
        Line2D([], [], color="0.35", lw=1.2, ls="--", marker="o", markersize=3.4,
               markerfacecolor="white", markeredgecolor="0.35",
               label=f"CESM2 \u2014 unseen ensemble mean ({_n_cesm} members)"),
        Patch(facecolor="0.35", alpha=0.28, label="EMULATOR member range"),
        Patch(facecolor="0.35", alpha=0.12, label="CESM2 member range"),
        Patch(facecolor="0.55", alpha=0.20,
              label=f"CESM2 spread about its mean, b\u2013d "
                    f"(\u00b12\u03c3, mean \u00b1{2*_sig:.2f} \u00b0C)"),
    ]
    # Both legends top-left: that corner is empty until ~1950 in every
    # scenario, whereas lower-right sits on top of the AAER curve.
    # Legends ABOVE the axes so they take no data area.
    leg1 = ax.legend(frameon=False, ncols=4, loc="lower left",
                     bbox_to_anchor=(0.0, 1.13), handlelength=2.2)
    ax.add_artist(leg1)
    ax.legend(handles=style, frameon=False, ncols=2, fontsize=8.2,
              loc="lower left", bbox_to_anchor=(0.0, 1.005), handlelength=2.6)

    ax.axhline(0, ls=":", lw=0.8, color="0.3")
    ax.axvspan(*BASELINE, color="0.9", alpha=0.6, lw=0, zorder=0)
    ax.set_ylabel("GMST anomaly (°C, vs 1850–1900)")
    ax.text(0.005, 0.97, "(a)", transform=ax.transAxes, fontweight="bold",
            va="top", ha="left")


    ax.set_xlabel("Year")
    ax.set_xlim(BASELINE[0], args.year_max)
    # y-limit must clear both the bias lines and the +/-2 sigma band
    _lim = max([0.35]
               + [abs(float(d.min())) for _, d, _sd in bias_of.values()]
               + [abs(float(d.max())) for _, d, _sd in bias_of.values()]
               + [2 * float(_sd.max()) for _, _d, _sd in bias_of.values()]) * 1.15
    axbs[0].set_ylim(-_lim, _lim)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")   # vector for the journal
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    if rows:
        t = pd.DataFrame(rows)
        print("\nEmulator vs held-out CESM2 (°C, on overlapping years)")
        print(t.to_string(index=False))
        if sigma_by_scen:
            _v = list(sigma_by_scen.values())
            print("\nCESM2 inter-member sigma by scenario (\u00b0C): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in sigma_by_scen.items()))
            print(f"  spread across scenarios: {max(_v)-min(_v):.4f} \u00b0C "
                  f"({100*(max(_v)-min(_v))/np.mean(_v):.1f}% of the mean) "
                  f"-> a single shared envelope is representative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
