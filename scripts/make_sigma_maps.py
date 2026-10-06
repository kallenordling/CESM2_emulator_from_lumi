#!/usr/bin/env python3
"""Final-decade anomaly maps for every sigma arm and every experiment.

Rows are the conditioning smoothing sigmas, columns are experiments: the four
the model trained on (hist, ssp370, aaer, ghg) and the three it did not
(ssp126, ssp245, RAMIP ssp370-126aer). The top row is held-out CESM2 where a
reference exists, so each sigma row reads directly against truth.

Two modes:
  --mode diff   (default) each panel is ARM MINUS CESM2 on one shared scale,
                so the arms are comparable panel to panel. Columns without a
                CESM2 reference are dropped in this mode rather than silently
                showing something else.
  --mode anom   each panel is the arm's own anomaly. Keeps every column, and
                is the honest choice when no reference exists.

The sigma arms are 5-member evals, so small-scale speckle in a panel is
sampling noise, not an arm difference. Pattern r and RMSE under each panel are
cos(lat)-weighted and computed against CESM2's anomaly.

Epochs differ between arms because they chain independently; --epoch picks the
nearest COMPLETE eval to a target for each arm, and every panel is labelled
with the epoch actually used so a mismatched comparison cannot hide.

    ~/miniconda3/envs/plotting/bin/python scripts/make_sigma_maps.py --epoch 330
"""
import argparse
import os
import re
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import (META, BASELINE, weighted_stats, make_axes,
                                     draw_map, to_pm180, panel_label,
                                     apply_baselines, to_celsius)
import make_unseen_maps as MU

EVAL_ROOT = os.path.expanduser("~/mnt/lumi_sc/eval_output")
# (label, eval run dir). sigma 4 is the mmlin arm, trained before the sweep on
# the same pipeline (nopca config, hist+ssp370 min/max anchors, no clip), so it
# belongs in the comparison -- it is simply the ninth point, not a different
# experiment. It is far deeper than the sweep arms, so --epoch still applies
# and its row is labelled with the epoch actually used.
SIGMAS = [("0", "run_sigs0_co2fix"), ("0.5", "run_sigs0p5_co2fix"),
          ("1", "run_sigs1_co2fix"), ("1.5", "run_sigs1p5_co2fix"),
          ("2", "run_sigs2_co2fix"), ("2.5", "run_sigs2p5_co2fix"),
          ("3", "run_sigs3_co2fix"), ("3.5", "run_sigs3p5_co2fix"),
          ("4", "run_mmlin_co2fix")]

# key -> (column label, final-decade window). Trained first, then unseen.
EXPERIMENTS = {
    "hist":          ("Historical",        (2005, 2014)),
    "ssp370":        ("SSP3-7.0",          (2091, 2100)),
    "aaer":          ("AAER",              (2041, 2050)),
    "ghg":           ("GHG",               (2041, 2050)),
    "ssp126":        ("SSP1-2.6 (unseen)", (2091, 2100)),
    "ssp245":        ("SSP2-4.5 (unseen)", (2091, 2100)),
    "ssp370-126aer": ("RAMIP (unseen)",    (2070, 2079)),
}


def complete_evals(run):
    """Epochs whose eval directory actually finished, newest last.

    An eval killed mid-run leaves a directory holding only PRECT files, which
    would otherwise be picked as "newest" and then fail on the TREFHT open.
    """
    out = []
    root = os.path.join(EVAL_ROOT, run)
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        m = re.match(r"^best_ep(\d+)$", d)
        if m and os.path.exists(os.path.join(root, d, "global_mean_anomaly_decadal.csv")):
            out.append(int(m.group(1)))
    return out


def arm_anomaly(var, run, epoch, key, window):
    """One arm's final-decade anomaly for one experiment, vs its OWN 1850-1900."""
    path = os.path.join(EVAL_ROOT, run, f"best_ep{epoch:04d}", f"{var}_{key}.nc")
    if not os.path.exists(path):
        return None, None, None
    with xr.open_dataset(path) as ds:
        da = ds[f"{var}_model"]
        yrs = np.asarray(ds["year"].values).astype(int)
        dims = ["year"] + (["member"] if "member" in da.dims else [])
        sel = (yrs >= window[0]) & (yrs <= window[1])
        if not sel.any():
            return None, None, None
        fin = np.asarray(da.isel(year=np.where(sel)[0]).mean(dims).values, float)
        lat, lon = ds["lat"].values, ds["lon"].values
    # Baseline from this arm's OWN historical run: mixing arms' baselines would
    # fold a mean-state offset into the anomaly.
    hpath = os.path.join(EVAL_ROOT, run, f"best_ep{epoch:04d}", f"{var}_hist.nc")
    with xr.open_dataset(hpath) as ds:
        da = ds[f"{var}_model"]
        yrs = np.asarray(ds["year"].values).astype(int)
        dims = ["year"] + (["member"] if "member" in da.dims else [])
        b = (yrs >= BASELINE[0]) & (yrs <= BASELINE[1])
        base = np.asarray(da.isel(year=np.where(b)[0]).mean(dims).values, float)
    return fin - base, lat, lon


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--var", nargs="+", default=["TREFHT"])
    ap.add_argument("--epoch", type=int, default=330,
                    help="target epoch; each arm uses its nearest COMPLETE eval")
    ap.add_argument("--mode", choices=["diff", "anom"], default="diff")
    ap.add_argument("--outdir", default="plots/sigma_maps")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for var in args.var:
        # CESM2 reference per experiment. The TRAINED four come from the
        # make_ensemble_mean_maps cache (held-out members); the unseen three
        # come from the CMIP6/RAMIP reference files via make_unseen_maps.
        # Consulting only the latter silently dropped every trained column.
        cesm, lat, lon = {}, None, None
        cache = None
        for c_ in (f"plots/ensmean_maps/cache_{var}_10y_v2.npz",
                   f"plots/ensmean_maps/cache_{var}_10y.npz"):
            if os.path.exists(c_):
                cache = c_; break
        trained_ref = {}
        if cache:
            z = np.load(cache, allow_pickle=True)
            ref = apply_baselines(to_celsius(z["ref"].item(), var, "CESM2"))
            for k_, v_ in ref.items():
                trained_ref[k_] = v_["final"] - v_["base"]
                if lat is None:
                    lat, lon = v_["lat"], v_["lon"]
            print(f"[{var}] trained reference from {os.path.basename(cache)}: "
                  f"{sorted(trained_ref)}")
        else:
            print(f"[{var}] WARNING: no ensmean cache — trained columns will "
                  f"have no CESM2 reference")
        for key, (_, win) in EXPERIMENTS.items():
            if key in trained_ref:
                cesm[key] = trained_ref[key]
                continue
            try:
                c = MU.ref_maps(var, key, win) if key in MU.SCEN else None
            except Exception as exc:                              # noqa: BLE001
                print(f"  [ref] {key}: {exc}")
                c = None
            cesm[key] = c

        keys = list(EXPERIMENTS)
        if args.mode == "diff":
            dropped = [k for k in keys if cesm.get(k) is None]
            keys = [k for k in keys if cesm.get(k) is not None]
            if dropped:
                print(f"[{var}] no CESM2 reference, dropped from diff mode: {dropped}")
        if not keys:
            print(f"[{var}] nothing to plot"); continue

        rows = []
        for label, run in SIGMAS:
            eps = complete_evals(run)
            if not eps:
                print(f"  [sigma {label}] no complete eval"); continue
            ep = min(eps, key=lambda e: abs(e - args.epoch))
            fields = {}
            for key in keys:
                f, la, lo = arm_anomaly(var, run, ep, key, EXPERIMENTS[key][1])
                if f is not None and lat is None:
                    lat, lon = la, lo
                fields[key] = f
            rows.append((f"$\\sigma$={label}  ep{ep}", fields))
            print(f"  [sigma {label}] ep{ep}: "
                  f"{sum(v is not None for v in fields.values())}/{len(keys)} experiments")
        if not rows or lat is None:
            print(f"[{var}] no arm data"); continue

        unit = META[var]["unit"]
        if args.mode == "diff":
            pool = [np.ravel(f - cesm[k]) for _, fl in rows for k, f in fl.items()
                    if f is not None and cesm.get(k) is not None]
        else:
            pool = [np.ravel(f) for _, fl in rows for f in fl.values() if f is not None]
        vmax = float(np.nanpercentile(np.abs(np.concatenate(pool)), 99))
        tmax = float(np.nanpercentile(np.abs(np.concatenate(
            [np.ravel(cesm[k]) for k in keys if cesm.get(k) is not None])), 99)) if any(
            cesm.get(k) is not None for k in keys) else vmax

        has_ref_row = any(cesm.get(k) is not None for k in keys)
        nrow = len(rows) + (1 if has_ref_row else 0)
        fig = plt.figure(figsize=(3.5 * len(keys), 2.2 * nrow + 0.9),
                         constrained_layout=True)
        axes = make_axes(fig, nrow, len(keys))
        n = 0
        im_top = im_d = None

        if has_ref_row:
            for j, key in enumerate(keys):
                ax = axes[0][j]
                if cesm.get(key) is not None:
                    im_top = draw_map(ax, to_pm180(cesm[key], lon)[0],
                                      cmap=META[var]["anom_cmap"], vmin=-tmax, vmax=tmax)
                else:
                    ax.set_facecolor("0.95")
                w = EXPERIMENTS[key][1]
                ax.set_title(f"{EXPERIMENTS[key][0]}\n{w[0]}-{w[1]} vs 1850-1900", fontsize=9)
                if j == 0:
                    ax.text(-0.04, 0.5, "CESM2", transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=9)
                panel_label(ax, n); n += 1

        for i, (name, fields) in enumerate(rows, start=1 if has_ref_row else 0):
            for j, key in enumerate(keys):
                ax = axes[i][j]
                f = fields.get(key)
                if f is None:
                    ax.set_facecolor("0.95"); panel_label(ax, n); n += 1; continue
                if args.mode == "diff" and cesm.get(key) is not None:
                    im_d = draw_map(ax, to_pm180(f - cesm[key], lon)[0],
                                    cmap=META[var]["dcmap"], vmin=-vmax, vmax=vmax)
                    st = weighted_stats(f, cesm[key], lat)
                    ax.text(0.5, -0.04, f"r {st['r']:.3f}  RMSE {st['rmse']:.2f}",
                            transform=ax.transAxes, ha="center", va="top", fontsize=7)
                else:
                    im_d = draw_map(ax, to_pm180(f, lon)[0], cmap=META[var]["anom_cmap"],
                                    vmin=-vmax, vmax=vmax)
                    if not has_ref_row and i == 0:
                        w = EXPERIMENTS[key][1]
                        ax.set_title(f"{EXPERIMENTS[key][0]}\n{w[0]}-{w[1]}", fontsize=9)
                if j == 0:
                    ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=8)
                panel_label(ax, n); n += 1

        if im_top is not None:
            fig.colorbar(im_top, ax=list(axes[0].ravel()), shrink=0.8,
                         label=f"CESM2 anomaly ({unit})")
        lab = f"arm minus CESM2 ({unit})" if args.mode == "diff" else f"anomaly ({unit})"
        body = axes[1:] if has_ref_row else axes
        fig.colorbar(im_d, ax=list(body.ravel()), shrink=0.45, label=lab)
        fig.suptitle(f"{var}: final-decade anomaly by conditioning smoothing "
                     f"$\\sigma$ — {'difference from CESM2' if args.mode=='diff' else 'arm anomaly'}"
                     f"  (5-member evals; speckle is sampling noise)", fontsize=11)
        out = os.path.join(args.outdir, f"sigma_maps_{var}_{args.mode}")
        for ext in (".png", ".pdf"):
            fig.savefig(out + ext, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}.png/.pdf")


if __name__ == "__main__":
    main()
