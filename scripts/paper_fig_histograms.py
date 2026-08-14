#!/usr/bin/env python3
"""
Paper figure: emulated vs held-out CESM2 DISTRIBUTIONS over the final decade.

One panel per experiment (historical, SSP3-7.0, AAER, GHG). The quantity is the
GLOBAL-MEAN ANOMALY: each ensemble member contributes one value per year, and
the last N years of every member are pooled into one distribution — n_members x
n_years samples per side (5 x 10 = 50 by default).

This is the distribution of global-mean climate STATES the emulator produces,
i.e. whether its year-to-year and member-to-member spread matches CESM2's
internal variability. The timeseries figure shows the ensemble MEANS agree; this
shows whether the scatter about them does too.

Global means are cos(lat)-weighted, and anomalies are referenced to each side's
OWN 1850-1900 mean, exactly as in paper_fig_timeseries.py. Precipitation is
expressed as percent change for the same reason it is there: a few hundredths of
a mm/day is meaningless without the ~2.9 mm/day it is relative to.

REFERENCE = HELD-OUT MEMBERS ONLY
---------------------------------
CESM2 is read from the training trees, restricted to members absent from the
data config's experiment_configs, and truncated to the same ensemble size as the
emulator (--n-ref-members, default 5). The eval NetCDF's own CESM arrays are NOT
used: for aaer/ghg they contain all ten members, nine of which are training data.

Usage
-----
    python scripts/paper_fig_histograms.py --var TREFHT \\
        --eval-dir /path/to/eval_output/run_mseyb_BCprect/best_ep0490

    python scripts/paper_fig_histograms.py --var PRECT --eval-dir ... --n-years 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VAR = "TREFHT"

VARMETA = {
    "TREFHT": dict(unit="°C", unit_plain="degC",
                   xlab="Global-mean temperature anomaly (°C, vs 1850–1900)",
                   title="temperature",
                   tree_scale={None: 1.0, "K": 1.0, "degC": 1.0},
                   tree_offset={"K": -273.15, "degC": 0.0, None: 0.0}),
    "PRECT":  dict(unit="%", unit_plain="%", percent=True,
                   xlab="Global-mean precipitation change (%, vs 1850–1900)",
                   title="precipitation",
                   tree_scale={"m/s": 86400.0 * 1000.0, "mm/day": 1.0, None: 1.0},
                   tree_offset={"m/s": 0.0, "mm/day": 0.0, None: 0.0}),
}

SCEN = {
    "hist":   ("Historical",                "hist",   "hist",   "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "ssp370", "ssp370", "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "aaer",   "AAER",   "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "ghg",    "GHG",    "#009E73"),
    # OUT-OF-TRAINING: no LENS2 tree. Their CESM2 side comes from the CMIP6
    # ensembles, which paper_fig_timeseries.py writes into the same global-mean
    # cache this script reads — so nothing extra is loaded here.
    "ssp126": ("SSP1-2.6 (unseen)",         "ssp126", None,     "#CC79A7"),
    "ssp245": ("SSP2-4.5 (unseen)",         "ssp245", None,     "#56B4E9"),
}
CFG_KEY = {k: k for k in SCEN}

# The four scenarios with a held-out LENS2 reference — the default paper figure.
DEFAULT_SCENARIOS = ["hist", "ssp370", "aaer", "ghg"]

# Scenarios whose reference is the CMIP6 ensemble rather than withheld LENS2
# members — they are labelled differently, since "held-out" would be wrong.
CMIP6_SCEN = {"ssp126", "ssp245"}


def unseen_members(tree_root: Path, subdir: str, trained: set) -> list:
    d = tree_root / subdir
    have = {p.name for p in d.iterdir() if p.is_dir() and p.name != "diagnostics"}
    return sorted(have - set(trained))


BASELINE = (1850, 1900)


def anom(values, base):
    """Absolute difference, or percent change when the variable sets percent."""
    if VARMETA[VAR].get("percent"):
        return 100.0 * (np.asarray(values) - base) / base
    return np.asarray(values) - base


def baseline_of(years, values):
    m = (np.asarray(years) >= BASELINE[0]) & (np.asarray(years) <= BASELINE[1])
    return float(np.asarray(values)[m].mean()) if m.any() else np.nan


def read_emulator_gmean(nc_path: Path):
    """(years, member matrix) of ABSOLUTE global means from an eval NetCDF."""
    ds = xr.open_dataset(nc_path)
    years = ds["year"].values.astype(int)
    names = [v for v in ds.data_vars
             if v.startswith(f"{VAR}_model_gmean_m") and not v.endswith("_anom")
             and not v.startswith(f"{VAR}_model_gmean_mean")]
    if not names:
        raise KeyError(f"{nc_path}: no per-member {VAR}_model_gmean_m* fields")
    M = np.stack([ds[n].values for n in
                  sorted(names, key=lambda x: int(x.rsplit("_m", 1)[1]))])
    ds.close()
    return years, M                      # (members, years)


def read_cesm_gmean_cache(ref_csv: str):
    """Per-member global means from paper_fig_timeseries.py's cache.

    Reusing that cache is what makes this script fast: the global means have
    already been extracted from the training trees, so nothing here re-reads
    them. Build it by running paper_fig_timeseries.py for the same --var.
    """
    df = pd.read_csv(ref_csv)
    out = {}
    for sc, g in df.groupby("scenario"):
        P = g.pivot(index="year", columns="member", values="gmean_K").sort_index()
        out[str(sc)] = P
    return out


def wq(x, w, q):
    """Weighted quantile."""
    i = np.argsort(x)
    x, w = x[i], w[i]
    c = np.cumsum(w) - 0.5 * w
    return np.interp(np.asarray(q) * w.sum(), c, x)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emulator vs held-out CESM2 distributions over the final decade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--var", default="TREFHT", choices=sorted(VARMETA))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--data-root",
                    default="/home/nordling/mnt/lumi_sc2/emulator_data")
    ap.add_argument("--tree-root", default=None,
                    help="default <data-root>/training_data/<var>")
    ap.add_argument("--data-config",
                    default="configs/config_data_ybias_BCprect.yaml")
    ap.add_argument("--n-years", type=int, default=10,
                    help="pool the last N years of each experiment")
    ap.add_argument("--n-ref-members", type=int, default=5,
                    help="CESM2 members to use; matches the emulator's count")
    ap.add_argument("--ref-csv", default=None,
                    help="paper_fig_timeseries.py's cached CESM2 global means "
                         "for the same --var; default "
                         "plots/heldout_cesm2_ensemble_<var>.csv")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                    choices=sorted(SCEN),
                    help="ssp126/ssp245 are OUT-OF-TRAINING; their reference is "
                         "a 3-member CMIP6 ensemble, so the CESM2 histogram is "
                         "built from 3x n_years samples instead of 10x")
    ap.add_argument("--dump-data", default=None, metavar="DIR",
                    help="write the pooled samples that form each histogram "
                         "as a tidy CSV under DIR")
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    global VAR
    VAR = args.var
    META = VARMETA[VAR]
    if args.out is None:
        args.out = f"plots/paper_fig_hist_{VAR}.png"
    if args.tree_root is None:
        args.tree_root = os.path.join(args.data_root, "training_data", VAR)
    if args.ref_csv is None:
        # paper_fig_timeseries.py WRITES heldout_cesm2_ensemble_<VAR>.csv for
        # every variable, including TREFHT. This script used to read an
        # unsuffixed heldout_cesm2_ensemble.csv for TREFHT only, so the two
        # scripts kept separate temperature caches and this one silently missed
        # whatever the other had just added (e.g. the ssp126/ssp245 CMIP6
        # reference). Prefer the suffixed name, fall back to the legacy one.
        args.ref_csv = f"plots/heldout_cesm2_ensemble_{VAR}.csv"
        _legacy = "plots/heldout_cesm2_ensemble.csv"
        if not os.path.exists(args.ref_csv) and os.path.exists(_legacy):
            print(f"[ref] {args.ref_csv} absent — falling back to {_legacy}")
            args.ref_csv = _legacy
    if not os.path.exists(args.ref_csv):
        print(f"ERROR: CESM2 global-mean cache not found: {args.ref_csv}\n"
              f"Build it first (it extracts the held-out members from the "
              f"training trees):\n"
              f"    python scripts/paper_fig_timeseries.py --var {VAR} "
              f"--eval-dir {args.eval_dir}", file=sys.stderr)
        return 1
    tree_root, eval_dir = Path(args.tree_root), Path(args.eval_dir)
    print(f"[var] {VAR} ({META['unit_plain']})   last {args.n_years} years")

    import yaml
    cfg = yaml.safe_load(open(args.data_config))
    trained = {e["scenario_name"]: set(e.get("realizations", []))
               for e in cfg["experiment_configs"]}

    ref_all = read_cesm_gmean_cache(args.ref_csv)
    print(f"[ref] {args.ref_csv}: "
          + ", ".join(f"{k}={v.shape[1]}m" for k, v in ref_all.items()))

    # baselines: each side vs its OWN 1850-1900 historical mean. ssp370 has no
    # pre-industrial of its own and inherits the historical one, as in the
    # timeseries figure.
    emu_abs, emu_years = {}, {}
    for sc, (_, ncname, _, _) in SCEN.items():
        p = eval_dir / f"{VAR}_{ncname}.nc"
        if p.exists():
            emu_years[sc], emu_abs[sc] = read_emulator_gmean(p)
    ref_base_hist = baseline_of(ref_all["hist"].index.values,
                                ref_all["hist"].mean(axis=1).values)
    emu_base_hist = (baseline_of(emu_years["hist"], emu_abs["hist"].mean(axis=0))
                     if "hist" in emu_abs else np.nan)

    data = {}
    for sc in args.scenarios:
        label, ncname, sub, colour = SCEN[sc]
        if sc not in emu_abs or sc not in ref_all:
            print(f"[skip] {sc}: missing emulator or reference data")
            continue
        R = ref_all[sc]
        if args.n_ref_members > 0:
            R = R[list(R.columns)[:args.n_ref_members]]

        rb = ref_base_hist if sc == "ssp370" else baseline_of(
            R.index.values, R.mean(axis=1).values)
        eb = emu_base_hist if sc == "ssp370" else baseline_of(
            emu_years[sc], emu_abs[sc].mean(axis=0))
        if not np.isfinite(rb):
            rb = ref_base_hist
        if not np.isfinite(eb):
            eb = emu_base_hist

        ey = emu_years[sc]
        ekeep = ey >= ey.max() - args.n_years + 1
        ev = anom(emu_abs[sc][:, ekeep], eb).ravel()

        ry = R.index.values
        rkeep = ry >= ry.max() - args.n_years + 1
        cv = anom(R[rkeep].values, rb).ravel()
        cv = cv[np.isfinite(cv)]

        data[sc] = dict(ev=ev, cv=cv,
                        n_emu=emu_abs[sc].shape[0], n_c=R.shape[1],
                        ey=(int(ey[ekeep].min()), int(ey[ekeep].max())),
                        cy=(int(ry[rkeep].min()), int(ry[rkeep].max())))
        print(f"[{sc}] emulator {data[sc]['ey'][0]}-{data[sc]['ey'][1]} "
              f"({ev.size} values = {data[sc]['n_emu']}m x {ekeep.sum()}y), "
              f"CESM2 {data[sc]['cy'][0]}-{data[sc]['cy'][1]} ({cv.size} values)")

    if args.dump_data:
        os.makedirs(args.dump_data, exist_ok=True)
        _rows = [dict(scenario=sc,
                      source=("cesm2" if nm == "cv" else "emulator"),
                      value=float(v))
                 for sc, d in data.items()
                 for nm in ("ev", "cv") for v in d[nm]]
        _dp = os.path.join(args.dump_data, f"histogram_{VAR}.csv")
        _df = pd.DataFrame(_rows)
        _df["unit"] = META["unit_plain"]
        _df.to_csv(_dp, index=False)
        print(f"[data] {_dp}  ({len(_df)} pooled samples)")

    if not data:
        print("no data", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.transforms as mtransforms
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 10.5, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    # Grid follows the scenario count: the default four keep the 2x2 the paper
    # uses; a two-scenario run gets one row instead of two half-empty ones.
    _n = len(data)
    ncol = 2 if _n > 1 else 1
    nrow = int(np.ceil(_n / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(4.8 * ncol, 3.3 * nrow))
    for _j in range(_n, nrow * ncol):
        axes.flat[_j].set_visible(False)
    rows = []
    for i, (ax, (sc, d)) in enumerate(zip(axes.flat, data.items())):
        label, _, _, colour = SCEN[sc]
        ev, cv = d["ev"], d["cv"]

        # Shared bins spanning both. The global means are already area-weighted,
        # so each sample counts once — no further weighting here.
        lo = min(ev.min(), cv.min())
        hi = max(ev.max(), cv.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        bins = np.linspace(lo - pad, hi + pad, args.bins + 1)

        _reflab = "CESM2 (CMIP6)" if sc in CMIP6_SCEN else "CESM2 (held-out)"
        ax.hist(cv, bins=bins, density=True, histtype="stepfilled",
                color="0.55", alpha=0.45, label=_reflab)
        ax.hist(ev, bins=bins, density=True, histtype="step",
                color=colour, lw=1.8, label="Emulator")
        # No rug marks: below the axis they collided with the tick labels, and
        # at fixed data offsets they collapsed onto y=0 and read as artefacts on
        # the histogram baseline. The sample count is stated in the annotation
        # instead, which is what the rug was there to convey.
        ax.set_ylim(bottom=0)

        st = {}
        for nm, v in (("emu", ev), ("cesm", cv)):
            m = float(np.mean(v)); sd = float(np.std(v, ddof=1))
            q = np.percentile(v, [1, 50, 99])
            st[nm] = (m, sd, q)
        (em, esd, eq), (cm, csd, cq) = st["emu"], st["cesm"]
        rows.append(dict(scenario=sc, years=f"{d['ey'][0]}-{d['ey'][1]}",
                         n_emu=d["n_emu"], n_cesm=d["n_c"],
                         emu_mean=round(em, 3), cesm_mean=round(cm, 3),
                         d_mean=round(em - cm, 3),
                         emu_sd=round(esd, 3), cesm_sd=round(csd, 3),
                         d_sd=round(esd - csd, 3),
                         d_p1=round(eq[0] - cq[0], 3),
                         d_p50=round(eq[1] - cq[1], 3),
                         d_p99=round(eq[2] - cq[2], 3)))

        ax.set_title(f"({'abcd'[i]}) {label}", loc="left")
        ax.set_xlabel(META["xlab"])
        if i % 2 == 0:
            ax.set_ylabel("Density")
        ax.text(0.97, 0.95,
                f"{d['ey'][0]}–{d['ey'][1]}\n"
                f"{d['n_emu']}\u00d7{args.n_years} = {ev.size} emulator, "
                f"{d['n_c']}\u00d7{args.n_years} = {cv.size} CESM2\n"
                f"Δmean {em-cm:+.2f}, Δsd {esd-csd:+.2f} "
                f"{META['unit_plain']}",
                transform=ax.transAxes, fontsize=7.4, va="top", ha="right",
                color="0.25")

    axes.flat[0].legend(frameon=False, loc="upper left")
    _refname = ("CESM2" if any(sc in CMIP6_SCEN for sc in data)
                else "held-out CESM2")
    fig.suptitle(f"Emulated vs {_refname} global-mean {META['title']} "
                 f"anomaly, final {args.n_years} years "
                 f"(all years x all members pooled)",
                 fontsize=10.5, y=0.995)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    t = pd.DataFrame(rows)
    print(f"\nDistribution comparison ({META['unit_plain']}); "
          f"d_* = emulator minus CESM2")
    print(t.to_string(index=False))
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
