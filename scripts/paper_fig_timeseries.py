#!/usr/bin/env python3
"""
Paper figure: emulated vs held-out CESM2 global-mean temperature, four scenarios.

Panels: historical, SSP3-7.0, AAER (aerosol-only), GHG-only.

THE REFERENCE IS HELD-OUT DATA ONLY
-----------------------------------
Each scenario keeps exactly one CESM2 realization out of training
(configs/config_data_ybias_BCprect.yaml val_experiment_configs):

    hist    trained on 20 LE2 members   -> held out: LE2-1231.001
    ssp370  trained on 20 LE2 members   -> held out: LE2-1231.001
    aaer    trained on 001-009          -> held out: 010
    ghg     trained on 001-009          -> held out: 010

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
SCEN = {
    "hist":   ("Historical",              "TREFHT_hist.nc",   ("hist",   "LE2-1231.001"), "#1f4e79"),
    "ssp370": ("SSP3-7.0",                "TREFHT_ssp370.nc", ("ssp370", "LE2-1231.001"), "#cc2b2b"),
    "aaer":   ("Aerosol-only (AAER)",     "TREFHT_aaer.nc",   ("AAER",   "010"),          "#e08214"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "TREFHT_ghg.nc",  ("GHG",    "010"),          "#2a8a3e"),
}


def area_mean(da: xr.DataArray) -> xr.DataArray:
    """cos(lat)-weighted global mean, matching eval_aero.area_weighted_gmean."""
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(("lat", "lon"))


def read_heldout(tree_root: Path, subdir: str, member: str) -> pd.Series:
    """Annual global-mean TREFHT for one held-out realization, from its chunks."""
    d = tree_root / subdir / member
    files = sorted(d.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"no chunk files in {d}")
    print(f"    {subdir}/{member}: {len(files)} chunks", flush=True)
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
    da = ds[VAR]
    gm = area_mean(da).compute()
    tdim = "time" if "time" in gm.dims else "year"
    years = np.asarray(ds[tdim].values).astype(int)
    s = pd.Series(np.asarray(gm.values, dtype=float), index=years).sort_index()
    ds.close()
    return s[~s.index.duplicated(keep="first")]


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
    ap.add_argument("--ref-csv", default=None,
                    help="cache of the held-out reference series; read if present, "
                         "written after computing so re-plots are instant")
    ap.add_argument("--out", default="plots/paper_fig_timeseries.png")
    ap.add_argument("--year-max", type=int, default=2100)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    tree_root = Path(args.tree_root)

    # ── held-out CESM2 reference ────────────────────────────────────────────
    ref = {}
    if args.ref_csv and os.path.exists(args.ref_csv):
        df = pd.read_csv(args.ref_csv)
        for sc, g in df.groupby("scenario"):
            ref[sc] = pd.Series(g["gmean_K"].values, index=g["year"].values.astype(int))
        print(f"[ref] reusing cached series from {args.ref_csv}")
    else:
        print("[ref] reading held-out realizations from the training trees")
        for sc, (_, _, (sub, mem), _) in SCEN.items():
            ref[sc] = read_heldout(tree_root, sub, mem)
        if args.ref_csv:
            rows = [dict(scenario=sc, year=int(y), gmean_K=float(v))
                    for sc, s in ref.items() for y, v in s.items()]
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
    ref_base_hist = baseline_of(ref["hist"].index.values, ref["hist"].values)
    emu_base_hist = (baseline_of(emu["hist"][2], emu["hist"][0])
                     if "hist" in emu else np.nan)
    print(f"\n[baseline 1850-1900]  CESM2 held-out hist {ref_base_hist:.3f} K   "
          f"emulator hist {emu_base_hist:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharey=True)
    rows = []

    for ax, (sc, (label, _, (sub, mem), colour)) in zip(axes.flat, SCEN.items()):
        # scenarios without their own pre-industrial inherit the historical one
        rb = ref_base_hist if sc == "ssp370" else baseline_of(
            ref[sc].index.values, ref[sc].values)
        eb = emu_base_hist if sc == "ssp370" else (
            baseline_of(emu[sc][2], emu[sc][0]) if sc in emu else np.nan)
        if not np.isfinite(rb):
            rb = ref_base_hist
        if not np.isfinite(eb):
            eb = emu_base_hist

        # held-out CESM2
        r = ref[sc][ref[sc].index <= args.year_max]
        ax.plot(r.index, r.values - rb, color="k", lw=1.6, zorder=4,
                label=f"CESM2 held-out ({mem})")

        # emulator
        if sc in emu:
            mean, members, years = emu[sc]
            keep = years <= args.year_max
            if members is not None:
                lo = (members[:, keep] - eb).min(axis=0)
                hi = (members[:, keep] - eb).max(axis=0)
                ax.fill_between(years[keep], lo, hi, color=colour, alpha=0.25,
                                lw=0, zorder=2, label="Emulator ensemble range")
            ax.plot(years[keep], mean[keep] - eb, color=colour, lw=2.0, zorder=3,
                    label="Emulator ensemble mean")

            # overlap diagnostics for the caption
            common = np.intersect1d(years[keep], r.index.values)
            if len(common):
                e = pd.Series(mean[keep] - eb, index=years[keep]).loc[common]
                c = (r - rb).loc[common]
                rows.append(dict(scenario=sc, n_years=len(common),
                                 bias=round(float((e - c).mean()), 3),
                                 rmse=round(float(np.sqrt(((e - c) ** 2).mean())), 3),
                                 corr=round(float(np.corrcoef(e, c)[0, 1]), 3),
                                 last_emu=round(float(e.iloc[-1]), 3),
                                 last_cesm=round(float(c.iloc[-1]), 3)))

        ax.set_title(label, loc="left")
        ax.axhline(0, ls=":", lw=0.8, color="0.3")

        # Shade the baseline window only where it is actually covered; SSP3-7.0
        # starts in 2015 and inherits the historical baseline, so shading (and
        # spanning the x-axis back to) 1850 there is just empty panel.
        y0 = int(min(r.index.min(), emu[sc][2].min() if sc in emu else r.index.min()))
        y1 = int(max(r.index.max(), emu[sc][2].max() if sc in emu else r.index.max()))
        ax.set_xlim(y0, min(y1, args.year_max))
        if y0 <= BASELINE[1]:
            ax.axvspan(*BASELINE, color="0.9", alpha=0.6, lw=0, zorder=0)

    for ax in axes[-1]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("GMST anomaly (°C, vs 1850–1900)")

    axes.flat[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Emulated vs held-out CESM2 global-mean surface temperature",
                 y=0.99)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(str(Path(args.out).with_suffix(".pdf")))   # vector for the journal
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    if rows:
        t = pd.DataFrame(rows)
        print("\nEmulator vs held-out CESM2 (°C, on overlapping years)")
        print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
